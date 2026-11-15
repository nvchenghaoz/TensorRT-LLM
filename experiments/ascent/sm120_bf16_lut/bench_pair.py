"""Paired microbenchmark (run_gpu_pair): Gemma4 BF16 attention projections as each tree's model runs them.

The op is chosen from the tree itself: if modeling_gemma4.Gemma4Attention passes use_custom_cublas_mm on SM120,
UnquantizedLinearMethod.apply runs torch.ops.trtllm.cublas_mm(x, W.t(), None, out_dtype=None); otherwise F.linear.
Shapes come from <checkpoint>/config.json. Cold L2: 4 rotated weight copies (>= 704 MB per shape).
CUDA events, warmup 5, 20 timed, median, noise_floor=(p90-p10)/median. Two passes (orders A then reversed);
the per-case result is the pass with the lower median (both printed).
Also prints the per-forward attention-GEMM total (50 sliding + 10 global layers) per M.

usage: python bench_pair.py <checkpoint>
"""
import inspect
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt import read_shapes  # noqa: E402

PROTO_M = [300, 512, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
NCOPY = 4
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def tree_gating():
    """(shapes, token_range) the tree's Gemma4Attention routes to cublas_mm on this GPU, or None (F.linear)."""
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    from tensorrt_llm._utils import get_sm_version
    src = inspect.getsource(mg.Gemma4Attention.__init__)
    if get_sm_version() != 120 or "use_custom_cublas_mm = True" not in src:
        return None
    return {k: mg._CublasMmTokenGate(*v) for k, v in mg._SM120_BF16_CUBLAS_MM_ROUTES.items()}


def timeit(fn, warm=5, iters=20):
    for i in range(warm):
        fn(i)
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for i, (s, e) in enumerate(evs):
        s.record()
        fn(warm + i)
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in evs)
    n = len(t)
    med, p10, p90 = t[n // 2], t[int(0.1 * n)], t[min(n - 1, int(0.9 * n))]
    return med, (p90 - p10) / med


def host_us(op, x, W, n=300):
    for _ in range(20):
        op(x, W)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        op(x, W)
    dt = time.perf_counter() - t
    torch.cuda.synchronize()
    return dt / n * 1e6


def main():
    ckpt = sys.argv[1]
    shapes, info = read_shapes(ckpt)
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    gating = tree_gating()
    if gating is not None:
        def op(x, W):
            # same routing as UnquantizedLinearMethod.apply with Gemma4's SM120 token gates
            gate = gating.get((W.shape[1], W.shape[0]))
            if gate is not None and x.shape[0] in gate:
                return torch.ops.trtllm.cublas_mm(x, W.t(), None, None)
            return F.linear(x, W)
        path = "trtllm::cublas_mm where " + "; ".join(
            f"{k}: exact {sorted(g.exact)} or [{g.lo},{g.hi}]" for k, g in gating.items()) + ", else F.linear"
    else:
        def op(x, W):
            return F.linear(x, W)
        path = "F.linear"
    log("path", path, "config", json.dumps(info), "shapes", json.dumps(shapes), torch.cuda.get_device_name())
    layers = {"sl_qkv": info["sliding"], "sl_o": info["sliding"], "gl_qkv": info["layers"] - info["sliding"],
              "gl_o": info["layers"] - info["sliding"]}
    batch = {}
    passes = {}
    host = {}
    for pass_i in range(2):
        order = list(shapes.items()) if pass_i == 0 else list(reversed(list(shapes.items())))
        for sname, (N, K) in order:
            torch.manual_seed(0)
            Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NCOPY)]
            xmax = torch.randn(max(PROTO_M), K, device="cuda", dtype=torch.bfloat16)
            Ms = PROTO_M if pass_i == 0 else list(reversed(PROTO_M))
            for M in Ms:
                x = xmax[:M]
                med, nf = timeit(lambda i: op(x, Ws[i % NCOPY]))
                passes.setdefault(f"{sname}_M{M}", []).append((med, nf))
            if pass_i == 0:
                for M in (128, 300):
                    host[f"{sname}_M{M}"] = round(host_us(op, xmax[:M], Ws[0]), 2)
            del Ws, xmax
            torch.cuda.empty_cache()
        log(f"pass {pass_i} done")
    for key, vals in passes.items():
        med, nf = min(vals)
        batch[key] = {"kernel_ms": round(med, 5), "noise_floor": round(max(nf, 1e-4), 4)}
        log(f"{key}: {med:.4f} ms (passes {[round(v[0], 4) for v in vals]}) noise {nf:.4f}")
    for M in PROTO_M:
        tot = sum(layers[s] * batch[f"{s}_M{M}"]["kernel_ms"] for s in layers)
        nfs = max(batch[f"{s}_M{M}"]["noise_floor"] for s in layers)
        batch[f"fwd_M{M}"] = {"kernel_ms": round(tot, 4), "noise_floor": nfs}
        log(f"per-forward attention GEMMs at M={M}: {tot:.3f} ms")
    log("host enqueue us/call", json.dumps(host))
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}, "path": path, "host_us": host}))


if __name__ == "__main__":
    main()
