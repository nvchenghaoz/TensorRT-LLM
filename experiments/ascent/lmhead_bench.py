# lm_head (N=248320, K=5120) microbenchmark + checks for Qwen3.8-27B-NVFP4 on SM120.
# Arm-agnostic: it runs the source tree's own lm_head routing
# (modeling_qwen3_5._lm_head_nvfp4_enabled + _normalize_qwen35_quant_config_dict
# on the checkpoint's {NVFP4, group_size 16} lm_head entry), so on main it
# builds the load-time BF16 dequant LMHead (cuBLAS) and on the branch the
# Marlin W4A16 LMHead. Real lm_head tensors are read from the checkpoint
# (only the lm_head tensors; the model is not loaded).
# usage: python lmhead_bench.py <checkpoint> [bench|check]
import json
import os
import sys
from types import SimpleNamespace as NS

import torch

V, K = 248320, 5120
MS = (1, 2, 4, 8, 16)  # 8/16 also cover MTP verify/draft shapes
WARM, ITERS = 10, 100


def log(*a):
    print(*a, flush=True)


def load_lm_head(ckpt):
    from safetensors import safe_open
    wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in wm if k.startswith("lm_head.")]
    t = {}
    for k in keys:
        with safe_open(os.path.join(ckpt, wm[k]), framework="pt") as f:
            t[k[len("lm_head."):]] = f.get_tensor(k)
    for k, v in t.items():
        log("  ckpt lm_head.%s %s %s" % (k, tuple(v.shape), v.dtype))
    return t


def build_head(t, force_bf16=False):
    from tensorrt_llm._torch.models import modeling_qwen3_5 as mq
    from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import \
        Qwen3_5MoeHfWeightMapper as WM
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    mapping = Mapping()
    pre = NS(tie_word_embeddings=False, vocab_size=V, num_hidden_layers=64,
             torch_dtype=torch.bfloat16)
    mc = NS(quant_config_dict={"lm_head": QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)},
            pretrained_config=pre, mapping=mapping, quant_config=None)
    keep = False if force_bf16 else mq._lm_head_nvfp4_enabled(mc)
    mq._normalize_qwen35_quant_config_dict(mc, keep_lm_head_quant=keep)
    lqc = mc.quant_config_dict.get("lm_head")
    with torch.device("cuda"):
        head = LMHead(V, K, dtype=torch.bfloat16, mapping=mapping,
                      tensor_parallel_mode=TensorParallelMode.COLUMN,
                      gather_output=True, reduce_output=False, quant_config=lqc)
    if lqc is None:
        # main's path: the weight mapper dequantizes at load (row chunks; the
        # function is row-separable so the result is identical).
        fake = NS(config=mc, _E2M1_VALUES=WM._E2M1_VALUES)
        rows, ch = [], 31040
        for r0 in range(0, V, ch):
            sub = {"lm_head.weight": t["weight"][r0:r0 + ch],
                   "lm_head.weight_scale": t["weight_scale"][r0:r0 + ch],
                   "lm_head.weight_scale_2": t["weight_scale_2"],
                   "lm_head.input_scale": t["input_scale"]}
            rows.append(WM._dequantize_lm_head_nvfp4(fake, sub)["lm_head.weight"])
        head.load_weights([{"weight": torch.cat(rows)}])
    else:
        head.load_weights([{k: t[k] for k in ("weight", "weight_scale", "weight_scale_2",
                                              "input_scale") if k in t}])
    head.post_load_weights()
    arm = type(head.quant_method).__name__
    log("  built lm_head: keep_quant=%s quant=%s method=%s weight=%s %s" %
        (keep, None if lqc is None else lqc.quant_algo, arm, tuple(head.weight.shape),
         head.weight.dtype))
    return head, arm


def ref_fp64(t, x):
    """Exact-dequant FP64 reference logits: x @ (lut[nib] * fp8 * gs)^T in FP64."""
    from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import \
        Qwen3_5MoeHfWeightMapper as WM
    lut = torch.tensor(WM._E2M1_VALUES, dtype=torch.float64, device="cuda")
    gs = t["weight_scale_2"].to(torch.float32).double().cuda().reshape(())
    xd = x.double()
    out = torch.empty(x.shape[0], V, dtype=torch.float64, device="cuda")
    ch = 16384
    for r0 in range(0, V, ch):
        w = t["weight"][r0:r0 + ch].cuda()
        s = t["weight_scale"][r0:r0 + ch].cuda().to(torch.float32).double()
        n = w.shape[0]
        vals = torch.empty(n, K, dtype=torch.float64, device="cuda")
        vals[:, 0::2] = lut[(w & 0x0F).long()]
        vals[:, 1::2] = lut[((w >> 4) & 0x0F).long()]
        vals = vals.view(n, K // 16, 16) * (s * gs).unsqueeze(-1)
        out[:, r0:r0 + ch] = xd @ vals.view(n, K).T
        del vals
    return out


def err_stats(out, ref):
    d = (out.double() - ref).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "ref_max_abs": float(ref.abs().max()),
            "argmax_match": float((out.float().argmax(-1) == ref.argmax(-1)).float().mean())}


def make_inputs(seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return {m: (torch.randn(m, K, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
            for m in MS}


def capture(head, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            head(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        out = head(x)
    return g, out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main():
    ckpt = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "bench"
    dev = torch.cuda.get_device_properties(0)
    log("device %s sm%d%d %d SMs L2 %d MB" % (dev.name, dev.major, dev.minor,
                                             dev.multi_processor_count, dev.L2_cache_size >> 20))
    t = load_lm_head(ckpt)
    head, arm = build_head(t)
    cls = "exact" if arm == "UnquantizedLinearMethod" else "precision"
    xs = make_inputs()
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")  # > 4x L2
    res, errs, checks = {}, {}, {}
    ok = True
    for m in MS:
        x = xs[m]
        with torch.inference_mode():
            eager = head(x).clone()
        g, gout = capture(head, x)
        if mode == "bench":
            e0 = [torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)]
            e1 = [torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)]
            for i in range(WARM + ITERS):
                flush.zero_()
                e0[i].record()
                g.replay()
                e1[i].record()
            torch.cuda.synchronize()
            ts = [e0[i].elapsed_time(e1[i]) for i in range(WARM, WARM + ITERS)]
            med = pct(ts, 0.5)
            res[str(m)] = {"kernel_ms": round(med, 5),
                           "noise_floor": round((pct(ts, 0.9) - pct(ts, 0.1)) / med, 5),
                           "p10_ms": round(pct(ts, 0.1), 5), "p90_ms": round(pct(ts, 0.9), 5),
                           "n": ITERS}
            log("  M=%d %s median %.4f ms p10 %.4f p90 %.4f" %
                (m, arm, med, pct(ts, 0.1), pct(ts, 0.9)))
        # untimed correctness sequence: same warmup/iterations/flush points as timing
        ref = ref_fp64(t, x)
        first = None
        n_mis_first = n_mis_eager = 0
        max_err_it = []
        for i in range(WARM + ITERS):
            flush.zero_()
            g.replay()
            torch.cuda.synchronize()
            o = gout.clone()
            if first is None:
                first = o
            n_mis_first += int(not torch.equal(o, first))
            n_mis_eager += int(not torch.equal(o, eager))
            max_err_it.append(float((o.double() - ref).abs().max()))
        e = err_stats(eager, ref)
        e["bf16_output_rounding_floor_max_abs"] = float(
            (ref.to(torch.bfloat16).double() - ref).abs().max())
        errs[str(m)] = e
        checks[str(m)] = {"iters": WARM + ITERS, "mismatch_vs_first": n_mis_first,
                          "mismatch_vs_eager": n_mis_eager,
                          "per_iter_max_err_min": min(max_err_it),
                          "per_iter_max_err_max": max(max_err_it),
                          "finite": bool(torch.isfinite(eager).all())}
        ok = ok and n_mis_first == 0 and n_mis_eager == 0 and checks[str(m)]["finite"]
        log("  M=%d err %s check %s" % (m, e, checks[str(m)]))
        del g, gout, ref
        torch.cuda.empty_cache()
    out = {"arm": arm, "errors_vs_fp64_exact_dequant": errs, "repeat_checks": checks}
    if mode == "bench":
        print(json.dumps({"batch": res, "output": dict(out, **{"class": cls})}))
    else:
        print(json.dumps(dict(out, passed=ok, compared="graph replays vs first replay and vs "
                              "eager (bitwise); eager vs FP64 exact-dequant reference")))


if __name__ == "__main__":
    main()
