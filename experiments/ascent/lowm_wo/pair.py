"""Paired timing of the CSA2 decode output projection (DeepseekV41Attention._project_output), TP4 rank shapes.

Same script in both trees: the base tree runs main's b12x_wo chain (Triton K1 -> b12x wo_a -> b12x wo_b), this
branch runs LowMWOProjection (csa2_lowm_wo_a -> csa2_mxfp8_low_m_gemm). Cold L2: one CUDA graph holds NL calls, each
with its own module copy (weights rotated over ~780 MB > 128 MB L2).
Reset point before every replay: the attention-output buffers are restored from a pristine copy.
Usage: pair.py <ckpt> [--batches 1,2,4,8] [--warmup 5] [--iters 30] [--layers 40]
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b12x_wo"))
import common as C  # noqa: E402
import harness as Hn  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--batches", default="1,2,4,8")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--rounds", type=int, default=3)
args = ap.parse_args()
BATCHES = [int(b) for b in args.batches.split(",")]


def nvml_clock():
    try:
        import pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid).replace("GPU-", "")
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUUID(h)
            u = (u.decode() if isinstance(u, bytes) else u).replace("GPU-", "")
            if u == uuid:
                return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception as e:  # noqa
        return f"nvml error {e!r}"
    return None


dims = C.dims_from_ckpt(args.ckpt)
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(args.layers - 1)]
mods = [Hn.make_module(dims, w, cs) for w in ws]
print(f"tree has b12x_wo={Hn.HAS_B12X_WO} layers={args.layers} dims={dims}", flush=True)

result = {}
diag = {}
for M in BATCHES:
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    pristine = o3.view(M, -1).clone()
    bufs = [pristine.clone() for _ in range(args.layers)]
    t0 = time.time()
    for i in range(args.layers):
        Hn.project(mods[i], bufs[i], pos)
    torch.cuda.synchronize()
    first_s = time.time() - t0
    for b in bufs:
        b.copy_(pristine)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            Hn.project(mods[i], bufs[i], pos)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for b in bufs:
        b.copy_(pristine)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [Hn.project(mods[i], bufs[i], pos) for i in range(args.layers)]
    ts = []
    clocks = []
    for rnd in range(args.rounds):
        for it in range(args.warmup + args.iters):
            for b in bufs:  # reset point
                b.copy_(pristine)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            if it >= args.warmup:
                ts.append(e0.elapsed_time(e1) / args.layers)
        clocks.append(nvml_clock())
    ts.sort()
    med = statistics.median(ts)
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(0.9 * (len(ts) - 1))]
    result[str(M)] = {"kernel_ms": round(med, 6), "noise_floor": round((p90 - p10) / med, 4)}
    diag[str(M)] = {"path": Hn.path_name(mods[0]), "impl": type(mods[0]._b12x_wo).__name__,
                    "first_eager_s": round(first_s, 2), "p10_ms": p10, "p90_ms": p90, "n": len(ts),
                    "sm_clock_mhz": clocks}
    print(f"M={M} {result[str(M)]} {diag[str(M)]}", flush=True)
    del g, outs
    torch.cuda.empty_cache()

print(json.dumps({"batch": result, "output": {"class": "reordered"}, "diag": diag, "warmup": args.warmup,
                  "iters": args.iters, "rounds": args.rounds, "layers": args.layers}))
