"""Paired timing of the CSA2 decode output projection (_project_output), per-rank TP4.

Same script in both trees: the baseline tree runs main's chain, the candidate
tree the b12x path. Cold-L2 CUDA graph: one graph holds ncopy calls, each with
its own module copy (weights rotated over > 300 MB). Before every replay the
attention-output buffers are restored from a pristine copy (main's chain
rotates them in place). Usage: pair.py <ckpt> [--batches 1,2,4,8] [--warmup 5] [--iters 30]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import harness as Hn  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--batches", default="1,2,4,8")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--rot-mb", type=int, default=320)
args = ap.parse_args()
BATCHES = [int(b) for b in args.batches.split(",")]

dims = C.dims_from_ckpt(args.ckpt)
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
wbytes = w0.wa_main.numel() + w0.wb_q.numel() + w0.sa_main.numel() * 4 + w0.wb_sf.numel()
ncopy = max(2, -(-args.rot_mb * 1024 * 1024 // wbytes))
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(ncopy - 1)]
mods = [Hn.make_module(dims, w, cs) for w in ws]
print(f"tree has b12x_wo={Hn.HAS_B12X_WO} ncopy={ncopy} dims={dims}", flush=True)

result = {}
diag = {}
for M in BATCHES:
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    pristine = o3.view(M, -1).clone()
    bufs = [pristine.clone() for _ in range(ncopy)]
    # Eager first call per module copy (candidate: lazy build + prewarm).
    t0 = time.time()
    for i in range(ncopy):
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
        outs = [Hn.project(mods[i], bufs[i], pos) for i in range(ncopy)]
    ts = []
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
            ts.append(e0.elapsed_time(e1) / ncopy)
    ts.sort()
    med = ts[len(ts) // 2]
    noise = (ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med
    result[str(M)] = {"kernel_ms": round(med, 6), "noise_floor": round(noise, 4)}
    diag[str(M)] = {"path": Hn.path_name(mods[0]), "first_eager_s": round(first_s, 2),
                    "p10_ms": ts[int(0.1 * (len(ts) - 1))], "p90_ms": ts[int(0.9 * (len(ts) - 1))]}
    print(f"M={M} {result[str(M)]} {diag[str(M)]}", flush=True)
    del g, outs
    torch.cuda.empty_cache()

clk = None
try:
    import subprocess
    clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except Exception:  # noqa
    pass
print(json.dumps({"batch": result, "output": {"class": "exact"}, "diag": diag, "sm_clock": clk,
                  "warmup": args.warmup, "iters": args.iters, "ncopy": ncopy}))
