"""Paired microbenchmark: Gemma4 gate_up -> gelu_tanh*up -> NVFP4 quant chain (no model load).

Shapes from <checkpoint>/config.json (gate_up N = 2*intermediate, K = hidden), synthetic NVFP4
operands quantized with trtllm::fp4_quantize. Cold weights: 3 rotating gate_up weight copies
(3 x 123 MB > 2 x 128 MB L2). nvfp4_gemm is autotuned in-process first (like the engine's warmup).
  base (no nvfp4_gelu_gemm_sm120 module, or GELU_EPI_ARM=base): main's _Gemma4GeluQuantMLP chain
       on the original rows: in the chunk band (TRTLLM_GEMMA4_MLP_CHUNK_*, M >= 1664) the M-chunked
       gate_up_gelu_fp4_chunked, else nvfp4_gemm(tuned) -> gelu_tanh_mul_fp4_quant
  candidate: rows permuted once; ge.gate_up_gelu_fp4_permuted, the dispatch
       _Gemma4GeluQuantMLP._gelu_epi_forward runs: (a) pick ('cutlass', 2) -> fused op, (b) chunk
       band -> chunked chain with the permuted act, (c) nvfp4_gemm(tuned) -> permuted act
Per M: CUDA events around one chain call, 5 warmup + 30 timed, median / p10 / p90.
usage: bench.py <checkpoint>
"""
import json
import os
import statistics
import sys
import threading
import time

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(45)
        log("heartbeat")


MS = [1, 16, 64, 300, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
BACKENDS = "cutlass,cublaslt,cutedsl,cuda_core"
# Timed-run protocol, imported by the check scripts (correctness.py, synced_test.py, small_m_gate.py)
# so their repeated sequences mirror it call for call. No L2 flush: cold weights come from 3 rotating
# gate_up weight copies (call i uses copy i % 3).
WARMUP, ITERS = 5, 30
FLUSH = "none (3 rotating weight copies, call i -> copy i % 3)"


def main():
    threading.Thread(target=_hb, daemon=True).start()
    ck = sys.argv[1]
    import torch
    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    hidden, inter = tc["hidden_size"], tc["intermediate_size"]
    N, K = 2 * inter, hidden
    log(f"config hidden={hidden} intermediate={inter} -> gate_up N={N} K={K}")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    from tensorrt_llm._torch.kernels.fused_ops.gate_up_gelu_fp4_chunked import gate_up_gelu_fp4_chunked
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    PATHS = {}

    def chunk_band(m):
        return mg._MLP_CHUNK_ENABLED and mg._MLP_CHUNK_MIN_M <= m <= mg._MLP_CHUNK_MAX_M

    log(f"chunk cfg enabled={mg._MLP_CHUNK_ENABLED} rows={mg._MLP_CHUNK_ROWS} min_m={mg._MLP_CHUNK_MIN_M}")
    try:
        from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    except ImportError:
        ge = None
    arm = os.environ.get("GELU_EPI_ARM", "candidate" if ge is not None else "base")
    if arm == "candidate" and (not ge.gelu_gemm_epi_supported() or ge.get_gelu_gemm_module() is None):
        arm = "base"
    log(f"arm={arm}")
    print(f"[protocol] bench WARMUP={WARMUP} ITERS={ITERS} calls/key={WARMUP + ITERS} flush={FLUSH} "
          f"key order={MS}", flush=True)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    log(f"gpu {props.name} sms={props.multi_processor_count}")

    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    copies = []
    for c in range(3):
        w = torch.randn(N, K, device=dev, generator=g).to(torch.bfloat16)
        gs_w = (448 * 6 / w.float().abs().max()).reshape(1).float()
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs_w, 16, False)
        del w
        if arm == "candidate":
            w_fp4, w_sf = ge.permute_gate_up_rows(w_fp4, w_sf, K)
        copies.append((w_fp4, w_sf, gs_w))
    torch.cuda.empty_cache()
    acts = {}
    for M in MS:
        a = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
        acts[M] = (a_fp4, a_sf, [(1.0 / (gs_a * c[2])).float() for c in copies])
    gs_d = torch.tensor([3.07], device=dev)

    def gemm(M, i):
        a_fp4, a_sf, alphas = acts[M]
        c = copies[i % 3]
        return torch.ops.trtllm.nvfp4_gemm(a_fp4, c[0], a_sf, c[1], alphas[i % 3], torch.bfloat16,
                                           allowed_backends=BACKENDS)

    # engine-like warmup tuning over the bucket map
    # Optional argv[2]: tuner cache file shared by both arms of a pair (the first arm tunes and
    # writes it, the second loads it), so both arms serve the same gate_up picks.
    cache_path = sys.argv[2] if len(sys.argv) > 2 else None
    cache_loaded = bool(cache_path and os.path.exists(cache_path))
    log(f"autotune (cache {cache_path}, loaded={cache_loaded})")
    with autotune(cache_path=cache_path):
        for M in MS:
            gemm(M, 0)
    torch.cuda.synchronize()
    tuner = AutoTuner.get()
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, BACKENDS.split(","))

    def pick(M):
        a_fp4, a_sf, alphas = acts[M]
        c = copies[0]
        hit, _, tac, _ = tuner.profiling_cache.search_cache(
            "trtllm::nvfp4_gemm::gemm", [runner],
            tuple(tuner._get_input_sizes([a_fp4, c[0], a_sf, c[1], alphas[0]])),
            NVFP4GemmUnifiedRunner.tuning_config)
        return hit, (tuple(tac) if isinstance(tac, (list, tuple)) else tac)

    def dense_pick(M):
        # landing 22's dense-bucket entry (trtllm::nvfp4_gemm::gemm_dense), which nvfp4_gemm
        # runs instead of the standard pick for M in the shape's dense range
        try:
            from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
                _nvfp4_dense_range, _nvfp4_dense_tuning_config)
        except ImportError:
            return None
        rng = _nvfp4_dense_range(N, K)
        if rng is None or not (rng[0] <= M <= rng[1]):
            return None
        a_fp4, a_sf, alphas = acts[M]
        c = copies[0]
        dr = NVFP4GemmUnifiedRunner(0, torch.bfloat16, BACKENDS.split(","), dense_min_m=rng[0])
        hit, _, tac, _ = tuner.profiling_cache.search_cache(
            "trtllm::nvfp4_gemm::gemm_dense", [dr],
            tuple(tuner._get_input_sizes([a_fp4, c[0], a_sf, c[1], alphas[0]])),
            _nvfp4_dense_tuning_config(*rng))
        return [hit, list(tac) if isinstance(tac, (list, tuple)) else tac]

    res, picks = {}, {}
    for M in MS:
        hit, tac = pick(M)
        dpk = dense_pick(M)
        band = chunk_band(M)
        a_fp4, a_sf, alphas = acts[M]

        if arm == "base":
            # main's _Gemma4GeluQuantMLP chain: M-chunked in the chunk band, else full M
            def fn(i):
                c = copies[i % 3]
                if band:
                    r = gate_up_gelu_fp4_chunked(a_fp4, a_sf, c[0], c[1], alphas[i % 3], torch.bfloat16,
                                                 BACKENDS.split(","), gs_d, mg._MLP_CHUNK_ROWS)
                    if r is not None:
                        PATHS["main_chunked"] = PATHS.get("main_chunked", 0) + 1
                        return r
                PATHS["main_full"] = PATHS.get("main_full", 0) + 1
                return gelu_tanh_mul_fp4_quant(gemm(M, i), gs_d)
        else:
            # the dispatch _Gemma4GeluQuantMLP._gelu_epi_forward runs on permuted rows
            def fn(i):
                c = copies[i % 3]
                fp4, sf, path = ge.gate_up_gelu_fp4_permuted(
                    a_fp4, a_sf, c[0], c[1], alphas[i % 3], torch.bfloat16, BACKENDS.split(","), gs_d,
                    mg._MLP_CHUNK_ROWS if band else None, lambda: gemm(M, i))
                PATHS[path] = PATHS.get(path, 0) + 1
                return fp4, sf
        PATHS.clear()

        for i in range(WARMUP):
            fn(i)
        torch.cuda.synchronize()
        ts = []
        for i in range(ITERS):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(i)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med = statistics.median(ts)
        p10, p90 = ts[int(0.1 * (ITERS - 1) + 0.5)], ts[int(0.9 * (ITERS - 1) + 0.5)]
        res[str(M)] = {"kernel_ms": round(med, 5), "p10": round(p10, 5), "p90": round(p90, 5),
                       "noise_floor": round((p90 - p10) / med, 4)}
        picks[str(M)] = {"pick": list(tac) if isinstance(tac, tuple) else tac, "hit": hit,
                         "dense_pick": dpk, "chunk_band": band, "paths": dict(PATHS)}
        log(f"M={M} {picks[str(M)]} median {med * 1e3:.1f} us p10 {p10 * 1e3:.1f} p90 {p90 * 1e3:.1f}")
    print(json.dumps({"batch": res, "output": {"class": "exact"}, "arm": arm, "picks": picks,
                      "tuner_cache": {"path": cache_path, "loaded": cache_loaded}}), flush=True)


if __name__ == "__main__":
    main()
