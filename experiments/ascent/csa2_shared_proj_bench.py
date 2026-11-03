"""CSA2 shared-input MXFP8 projections: wq_a||wkv (hidden_states) and wq_b||index_wq_b (qr).

Per-rank TP4 shapes of DeepSeek-V4.1-Flash: hidden 5120, q_lora_rank 1280,
wkv 512, wq_b 64*512/4 = 8192, index_wq_b 32*128 = 4096.

Modes (argv[1]):
  bench  - cold-L2 CUDA-graph timing of one attention layer's input projections
           (both a layer with index_wq_b and a SWA layer without). In a tree with
           ``DeepseekV41Attention._project_inputs`` the branch path runs; otherwise
           the base sequence of separate Linears runs.
  check  - bitwise check of the branch path against the separate Linears at
           M=1,2,4,8,2048, for every FlashInfer CUTLASS tactic, under autotuned
           dispatch, plus CUDA-graph capture/replay at batch 1,2,4,8 and a
           repeated-execution sequence matching the bench's warmup/iterations.
"""

import json
import sys
import types

import torch
from torch import nn

HIDDEN, Q_RANK, KV_DIM, QB_N, IDX_N = 5120, 1280, 512, 8192, 4096
EPS = 1e-6
WARMUP, ITERS = 5, 30
SEED = 1234


def log(*args):
    print(*args, flush=True)


def make_linear(k, n, gen):
    import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers trtllm::flashinfer_mm_mxfp8)
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    with torch.device("cuda"):
        lin = Linear(k, n, bias=False, dtype=torch.bfloat16,
                     quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    w = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    raw = torch.randint(118, 128, (n, k // 32), generator=gen, device="cuda", dtype=torch.int32)
    raw = raw.to(torch.uint8)
    sw = torch.ops.trtllm.block_scale_interleave(raw.contiguous()).reshape(-1)
    with torch.no_grad():
        lin.weight.copy_(w)
        assert sw.numel() == lin.weight_scale.numel(), (sw.numel(), lin.weight_scale.numel())
        lin.weight_scale.copy_(sw.view_as(lin.weight_scale))
    return lin


class Probe(nn.Module):
    """The projection-owning fields of DeepseekV41Attention, with its methods when present."""

    def __init__(self, with_index, gen):
        super().__init__()
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        self.projection_quantization = "mxfp8"
        self.dtype = torch.bfloat16
        self._fuse_index_q = False
        self._wq_a_wkv = None
        self._wq_b_index = None
        self.wq_a = make_linear(HIDDEN, Q_RANK, gen)
        self.wq_b = make_linear(Q_RANK, QB_N, gen)
        self.wkv = make_linear(HIDDEN, KV_DIM, gen)
        if with_index:
            self.index_wq_b = make_linear(Q_RANK, IDX_N, gen)
        with torch.device("cuda"):
            self.q_norm = RMSNorm(hidden_size=Q_RANK, eps=EPS, dtype=torch.bfloat16)
        with torch.no_grad():
            self.q_norm.weight.copy_(1 + 0.1 * torch.randn(Q_RANK, generator=gen, device="cuda"))
        self.with_index = with_index

    def separate(self, h):
        """The base tree's sequence (module.py forward + _project_index_q, minus RoPE)."""
        qr = self.q_norm(self.wq_a(h))
        q = self.wq_b(qr)
        swa = self.wkv(h)
        idx = self.index_wq_b(qr) if self.with_index else None
        return qr, q, swa, idx


def attach_branch(probe):
    try:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    except Exception as e:  # pragma: no cover
        log("import failed", e)
        return False
    fn = getattr(DeepseekV41Attention, "_project_inputs", None)
    if fn is None:
        return False
    for name in ("_project_inputs", "_shared_mxfp8_active", "_mxfp8_mm",
                 "fuse_shared_input_projections"):
        setattr(probe, name, types.MethodType(getattr(DeepseekV41Attention, name), probe))
    probe.fuse_shared_input_projections()
    return True


def branch(probe, h):
    qr, q, swa, idx = probe._project_inputs(h)
    if probe.with_index and idx is None:
        idx = probe.index_wq_b(qr)
    return qr, q, swa, idx


def autotune_run(fn, h):
    from flashinfer import autotune

    with autotune(True):
        fn(h)
    torch.cuda.synchronize()


def bench():
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    l2_flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    results = {}
    noises = {}
    mode = None
    probes = {}
    for with_index in (True, False):
        probe = Probe(with_index, gen)
        has_branch = attach_branch(probe)
        mode = "branch" if has_branch else "base"
        probes[with_index] = (probe, (lambda p: (lambda x: branch(p, x)))(probe) if has_branch
                              else probe.separate)
    log("mode", mode)
    for m in (1, 2, 4, 8, 2048):
        for with_index in (True, False):
            probe, fn = probes[with_index]
            h = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            autotune_run(fn, h)
            for _ in range(3):
                fn(h)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn(h)
                with torch.cuda.graph(graph, stream=stream):
                    fn(h)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            times = []
            for i in range(WARMUP + ITERS):
                l2_flush.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                if i >= WARMUP:
                    times.append(start.elapsed_time(end))
            times.sort()
            med = times[len(times) // 2]
            p10 = times[int(0.1 * (len(times) - 1))]
            p90 = times[int(0.9 * (len(times) - 1))]
            key = f"{m}_{'index' if with_index else 'swa'}"
            results[key] = med
            noises[key] = (p90 - p10) / med
            log(f"M={m} {key}: median {med * 1000:.2f} us  p10 {p10 * 1000:.2f}  p90 {p90 * 1000:.2f}")
            del graph
    n_index, n_swa = layer_counts()
    log(f"layers with index_wq_b: {n_index}, without: {n_swa}")
    batch = {}
    for m in (1, 2, 4, 8, 2048):
        ki, ks = f"{m}_index", f"{m}_swa"
        batch[str(m)] = {
            # Per decode step: every attention layer's input projections.
            "kernel_ms": n_index * results[ki] + n_swa * results[ks],
            "noise_floor": max(noises[ki], noises[ks]),
            "index_layer_ms": results[ki],
            "swa_layer_ms": results[ks],
        }
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "mode": mode,
                      "layers": {"index": n_index, "no_index": n_swa}}))


def layer_counts():
    """Attention layers that own index_wq_b (FULL/REINDEX) vs not, from the checkpoint config."""
    path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        cfg = cfg.get("text_config", cfg)
        from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Mode

        layout = CSA2Layout(
            tuple(cfg["compress_ratios"]),
            tuple(cfg["kv_source_layer_ids"]),
            tuple(cfg["index_source_layer_ids"]),
            cfg["candidate_source_layer_id"],
            cfg["candidate_topk_blocks"],
            cfg["candidate_block_size"],
            cfg["index_topk"],
            cfg["sliding_window"],
        )
        n = cfg["num_hidden_layers"]
        idx = sum(layout.layer(i).mode in (CSA2Mode.FULL, CSA2Mode.REINDEX) for i in range(n))
        return idx, n - idx
    except Exception as e:
        log("layer count fallback (config unreadable):", type(e).__name__, e)
        return 20, 20


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and torch.equal(a, b)


def check():
    from flashinfer.gemm.gemm_base import (DEFAULT_WORKSPACE_SIZE, _get_cache_buf,
                                           get_gemm_sm120_module_cutlass_mxfp8)

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    failures = []
    report = {}
    probe = Probe(True, gen)
    probe_swa = Probe(False, gen)
    # Reference outputs from the separate Linears BEFORE the weights are shared.
    hs = {m: torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
          for m in (1, 2, 4, 8, 2048)}
    ref = {m: probe.separate(h) for m, h in hs.items()}
    ref_swa = {m: probe_swa.separate(h) for m, h in hs.items()}
    w_before = [t.clone() for t in (probe.wq_a.weight, probe.wkv.weight, probe.wq_b.weight,
                                    probe.index_wq_b.weight, probe.wq_a.weight_scale,
                                    probe.wkv.weight_scale, probe.wq_b.weight_scale,
                                    probe.index_wq_b.weight_scale)]
    assert attach_branch(probe) and attach_branch(probe_swa), "branch path missing"
    assert probe._wq_a_wkv is not None and probe._wq_b_index is not None
    assert probe_swa._wq_a_wkv is not None and probe_swa._wq_b_index is None
    w_after = [probe.wq_a.weight, probe.wkv.weight, probe.wq_b.weight, probe.index_wq_b.weight,
               probe.wq_a.weight_scale, probe.wkv.weight_scale, probe.wq_b.weight_scale,
               probe.index_wq_b.weight_scale]
    views_ok = all(torch.equal(a, b) for a, b in zip(w_before, w_after))
    views_ok = views_ok and probe.wq_a.weight.data_ptr() == probe._wq_a_wkv[0].data_ptr()
    report["weights_views_unchanged"] = views_ok
    if not views_ok:
        failures.append("weights changed by fusion")
    # Swizzle of concatenated raw scales == concatenation of swizzles (exactness of the layout).
    raw_a = torch.randint(0, 255, (Q_RANK, HIDDEN // 32), generator=gen, device="cuda",
                          dtype=torch.int32).to(torch.uint8)
    raw_b = torch.randint(0, 255, (KV_DIM, HIDDEN // 32), generator=gen, device="cuda",
                          dtype=torch.int32).to(torch.uint8)
    lhs = torch.ops.trtllm.block_scale_interleave(torch.cat((raw_a, raw_b)).contiguous()).reshape(-1)
    rhs = torch.cat((torch.ops.trtllm.block_scale_interleave(raw_a).reshape(-1),
                     torch.ops.trtllm.block_scale_interleave(raw_b).reshape(-1)))
    report["swizzle_concat_exact"] = bool(torch.equal(lhs, rhs))
    if not report["swizzle_concat_exact"]:
        failures.append("swizzle concat")

    names = ("qr", "q", "swa", "index")
    # 1) default dispatch (before tuning), every M.
    for m, h in hs.items():
        out = branch(probe, h)
        out_swa = branch(probe_swa, h)
        ok = all(equal(a, b) for a, b in zip(out, ref[m]))
        ok_swa = all(equal(a, b) for a, b in zip(out_swa, ref_swa[m]))
        strides = [tuple(t.stride()) for t in out]
        report[f"default_M{m}"] = {"bitwise": ok, "swa_layer_bitwise": ok_swa, "strides": strides}
        log(f"default M={m} bitwise={ok} swa_layer={ok_swa} strides={strides}")
        if not (ok and ok_swa):
            failures.append(f"default M={m}")
            for n, a, b in zip(names, out, ref[m]):
                if a is not None and not equal(a, b):
                    log("  mismatch", n, (a.float() - b.float()).abs().max().item())
    # 2) every CUTLASS tactic: fused GEMM vs separate GEMM, all tactics against tactic-0 separate.
    runner = get_gemm_sm120_module_cutlass_mxfp8().cutlass_mxfp8_gemm_runner()
    ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, torch.device("cuda"))

    def run_tactic(act, sf, w, s, t):
        out = torch.empty((act.shape[0], w.shape[0]), dtype=torch.bfloat16, device="cuda")
        runner.forward([act, w.t(), sf, s, torch.bfloat16, out, ws], tactic=t)
        return out

    tactic_report = {}
    for m, h in hs.items():
        act, sf = torch.ops.trtllm.mxfp8_quantize(h, True)
        qr = ref[m][0]
        qact, qsf = torch.ops.trtllm.mxfp8_quantize(qr, True)
        tactics = runner.get_valid_tactics([act, probe.wq_a.weight.t(), sf, probe.wq_a.weight_scale,
                                            torch.bfloat16, None, ws], None)
        bad = []
        base_a = base_kv = base_q = base_i = None
        for t in tactics:
            try:
                sep_a = run_tactic(act, sf, probe.wq_a.weight, probe.wq_a.weight_scale, t)
                sep_kv = run_tactic(act, sf, probe.wkv.weight, probe.wkv.weight_scale, t)
                fused = run_tactic(act, sf, *probe._wq_a_wkv, t)
                sep_q = run_tactic(qact, qsf, probe.wq_b.weight, probe.wq_b.weight_scale, t)
                sep_i = run_tactic(qact, qsf, probe.index_wq_b.weight, probe.index_wq_b.weight_scale, t)
                fused_b = run_tactic(qact, qsf, *probe._wq_b_index, t)
                torch.cuda.synchronize()
            except Exception as e:  # tactic not runnable at this shape
                log(f"  M={m} tactic {t} skipped: {type(e).__name__}")
                continue
            if base_a is None:
                base_a, base_kv, base_q, base_i = sep_a, sep_kv, sep_q, sep_i
            ok = (torch.equal(fused[:, :Q_RANK], sep_a) and torch.equal(fused[:, Q_RANK:], sep_kv)
                  and torch.equal(fused_b[:, :QB_N], sep_q) and torch.equal(fused_b[:, QB_N:], sep_i))
            cross = (torch.equal(sep_a, base_a) and torch.equal(sep_kv, base_kv)
                     and torch.equal(sep_q, base_q) and torch.equal(sep_i, base_i))
            if not ok:
                bad.append(("fused_vs_sep", t))
            if not cross:
                bad.append(("across_tactics", t))
        tactic_report[m] = {"tactics": len(tactics), "bad": bad}
        log(f"tactics M={m}: n={len(tactics)} bad={bad}")
        if any(kind == "fused_vs_sep" for kind, _ in bad):
            failures.append(f"tactic M={m}")
    report["tactics"] = tactic_report
    # 3) autotuned dispatch: tune both paths, then compare.
    for m, h in hs.items():
        autotune_run(probe.separate, h)
        autotune_run(lambda x: branch(probe, x), h)
        autotune_run(probe_swa.separate, h)
        autotune_run(lambda x: branch(probe_swa, x), h)
        out = branch(probe, h)
        sep = probe.separate(h)
        ok = all(equal(a, b) for a, b in zip(out, sep)) and all(
            equal(a, b) for a, b in zip(out, ref[m]))
        ok_swa = all(equal(a, b) for a, b in zip(branch(probe_swa, h), probe_swa.separate(h)))
        report[f"tuned_M{m}"] = {"bitwise": ok, "swa_layer_bitwise": ok_swa}
        log(f"tuned M={m} bitwise={ok} swa_layer={ok_swa}")
        if not (ok and ok_swa):
            failures.append(f"tuned M={m}")
    # 4) CUDA-graph capture/replay at decode batches 1,2,4,8 (+ repeated execution sequence).
    max_err = []
    for m in (1, 2, 4, 8):
        static = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            static.copy_(hs[8][:m])
            branch(probe, static)
            with torch.cuda.graph(graph, stream=stream):
                outs = branch(probe, static)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        per_iter = []
        for i in range(WARMUP + ITERS):
            # reset point: regenerate the input from the seed every iteration.
            g = torch.Generator(device="cuda").manual_seed(SEED + 17 * i + m)
            x = torch.randn((m, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
            static.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            expect = probe.separate(x)
            err = 0.0
            ok = True
            for a, b in zip(outs, expect):
                if a is None:
                    continue
                ok = ok and torch.equal(a, b)
                err = max(err, (a.float() - b.float()).abs().max().item())
            per_iter.append(err)
            if not ok:
                failures.append(f"graph M={m} iter {i}")
        max_err.append({"M": m, "max_abs_err_per_iter_max": max(per_iter), "iters": len(per_iter)})
        log(f"graph M={m}: captured+replayed {len(per_iter)} iters, max abs err {max(per_iter)}")
        del graph
    report["graph_repeat"] = max_err
    passed = not failures
    print(json.dumps({"passed": passed, "failures": failures[:20],
                      "compared": "branch _project_inputs vs separate MXFP8 Linears (base sequence)",
                      "report": report}, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    torch.manual_seed(SEED)
    if mode == "bench":
        bench()
    else:
        check()
