"""Branch 100 (idea 190), evaluator item 1: c1 TTFT regression of EP4 -> EP2xTP2.

The FlashInfer decode chain takes only calls of <= 32 tokens, so the 2048-token c1 prefill (and the
larger c2/c4 prefill batches) run CUTLASS fused_moe (W4A8 MXFP4 x MXFP8, in-kernel MXFP8 quant fold
for M >= 1024, atomic fused finalize) at the new per-rank shape. This benchmarks rank 0 of each layout
on ONE GPU with synthetic weights (no model load):

  ep4    : tp_size=1 ep_size=4 -> 96 local experts x inter 2304  (main)
  ep2tp2 : tp_size=2 ep_size=2 -> 192 local experts x inter 1152 (candidate)

Same weight bytes per rank; with uniform top-6 of 384 routing rank 0 gets ~1.5 rows/token under EP4 and
~3 rows/token under EP2xTP2 (2x permuted rows, half N per row), so permute / FC2 output / finalize
traffic doubles while GEMM FLOPs stay equal. That is the hypothesis this measures.

Per layout and M: CUDA-graph median ms per call, (p90-p10)/median, the autotuner's tactic per
(gemm, M bucket) after tuning (cache keys printed), and the per-prefill-step sum over the MoE layers
(num_hidden_layers - first_k_dense_replace) for comparison with the ~5 ms c1 TTFT delta of pair
1033/1034.

  python experiments/ascent/b100_moe_prefill/bench.py --ckpt {checkpoint} [--ms 2048,4096,8192]
  [--layouts ep4,ep2tp2] [--no-tune]   (--no-tune: default tactic, to see what tuning buys)
"""

import argparse
import json
import os

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402

LAYOUTS = {"ep4": (1, 4), "ep2tp2": (2, 2)}  # (moe_tp, moe_ep)
FOLD_MIN = int(os.environ.get("TRTLLM_MOE_MXFP8_QUANT_FOLD_MIN_TOKENS", "1024"))
ALIGN = 128


def log(*a):
    print(*a, flush=True)


def read_cfg(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if "text_config" in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    n_moe_layers = cfg["num_hidden_layers"] - int(cfg.get("first_k_dense_replace", 0) or 0)
    return (cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"],
            cfg["num_experts_per_tok"], n_moe_layers)


class Rank:
    def __init__(self, hidden, inter, n_exp, topk, tp, ep):
        self.hidden, self.n_exp, self.topk, self.tp, self.ep = hidden, n_exp, topk, tp, ep
        assert inter % tp == 0 and n_exp % ep == 0
        E, H, I = n_exp // ep, hidden, inter // tp
        assert I % ALIGN == 0, f"inter per partition {I} not a multiple of {ALIGN}"
        self.e_local, self.inter_local = E, I
        g = torch.Generator(device="cuda").manual_seed(1234)
        dev = "cuda"
        self.w31 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev,
                                 generator=g).view(torch.int64)
        self.w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev,
                                generator=g).view(torch.int64)
        sf31 = torch.randint(117, 125, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
        sf2 = torch.randint(117, 125, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
        ones = torch.ones(E, dtype=torch.float32, device=dev)
        self.quant_scales = [sf31.view(torch.int32).contiguous(), ones, sf2.view(torch.int32).contiguous(),
                             ones.clone()]

    def inputs(self, m, seed=0):
        """Uniform top-k of all n_exp experts (same routing for both layouts at a given m)."""
        g = torch.Generator(device="cuda").manual_seed(99 + 7 * m + 1000 * seed)
        x = torch.randn(m, self.hidden, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        gc = torch.Generator().manual_seed(5 + 13 * m + 1000 * seed)
        sel = torch.stack([torch.randperm(self.n_exp, generator=gc)[: self.topk] for _ in range(m)]).to(torch.int32)
        scales = torch.softmax(torch.randn(m, self.topk, generator=gc), dim=-1).float()
        return x.contiguous(), sel.cuda(), scales.cuda()

    def run(self, x, sel, scales):
        if x.shape[0] >= FOLD_MIN:  # production: BF16 in, MXFP8 quant folded into expandInputRows
            xin, xsf = x, None
        else:
            xin, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=ALIGN)
        return torch.ops.trtllm.fused_moe(
            xin, sel, scales, self.w31, None, self.w2, None, torch.bfloat16,
            quant_scales=self.quant_scales, input_sf=xsf, swizzled_input_sf=True,
            tp_size=self.tp, tp_rank=0, ep_size=self.ep, ep_rank=0,
            use_mxfp8_act_scaling=True, use_fused_finalize=True,
        )[0]


def graph_time(fn, reps=10, trials=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / reps)
    ts.sort()
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--layouts", default="ep4,ep2tp2")
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    ms = [int(v) for v in args.ms.split(",")]
    hidden, inter, n_exp, topk, n_layers = read_cfg(args.ckpt)
    log("cfg hidden/inter/n_exp/topk/moe_layers:", hidden, inter, n_exp, topk, n_layers,
        "fold_min:", FOLD_MIN, "gpu:", torch.cuda.get_device_name())
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    res, tactics = {}, {}
    for name in args.layouts.split(","):
        tp, ep = LAYOUTS[name]
        MoERunner.runner_dict.clear()
        AutoTuner.get().profiling_cache.cache.clear()
        torch.cuda.empty_cache()
        r = Rank(hidden, inter, n_exp, topk, tp, ep)
        log(f"layout {name}: moe_tp={tp} moe_ep={ep} local experts={r.e_local} inter/partition={r.inter_local}")
        if not args.no_tune:
            x, sel, sc = r.inputs(max(ms))
            with autotune():
                r.run(x, sel, sc)
            torch.cuda.synchronize()
        picks = []
        for k, v in AutoTuner.get().profiling_cache.cache.items():
            if not str(k[0]).startswith("trtllm::fused_moe"):
                continue
            try:
                picks.append((str(k[0]).rsplit(":", 1)[-1], str(k[3][0][0]), str(v[1])))
            except Exception:  # noqa: BLE001  (cache key layout differs: keep it raw)
                picks.append((str(k)[:200], "?", str(v)[:120]))
        picks.sort()
        tactics[name] = picks
        log(f"  tuned tactics (gemm, M bucket, tactic): {picks}")
        for m in ms:
            x, sel, sc = r.inputs(m)
            out = r.run(x, sel, sc)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(out.float()).all())
            rows = int((sel // r.e_local == 0).sum())  # rank-0 (ep_rank 0) rows = permuted rows
            med, noise = graph_time(lambda: r.run(x, sel, sc))
            res[f"{name}_M{m}"] = {"kernel_ms": med, "noise_floor": noise, "local_rows": rows,
                                   "per_prefill_step_ms": med * n_layers, "finite": finite}
            log(f"  {name} M={m} rows={rows} kernel_ms={med:.4f} noise={noise:.4f} "
                f"x{n_layers} layers={med * n_layers:.2f} ms finite={finite}")
        del r
    delta = {}
    for m in ms:
        a, b = res.get(f"ep4_M{m}"), res.get(f"ep2tp2_M{m}")
        if a and b:
            delta[str(m)] = {"per_layer_us": 1e3 * (b["kernel_ms"] - a["kernel_ms"]),
                             "per_prefill_step_ms": b["per_prefill_step_ms"] - a["per_prefill_step_ms"],
                             "rel": b["kernel_ms"] / a["kernel_ms"] - 1}
            log(f"M={m}: ep2tp2 - ep4 = {delta[str(m)]['per_layer_us']:+.1f} us/layer, "
                f"{delta[str(m)]['per_prefill_step_ms']:+.2f} ms/prefill step ({100 * delta[str(m)]['rel']:+.1f}%)")
    print(json.dumps({"batch": res, "delta_ep2tp2_minus_ep4": delta, "tactics": tactics,
                      "output": {"class": "mode_switch"}, "tuned": not args.no_tune}))


if __name__ == "__main__":
    main()
