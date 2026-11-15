"""Paired microbenchmark (run_gpu_pair): graph-replayed Gemma-4-31B attention-projection GEMM chain per forward
(60 layers in config order, qkv_proj + o_proj, every layer its own cold weights, 17 GB), as this tree dispatches
it (see chain.py), at the BCG buckets M = 300/600/900/1000/1200/1500/1800/2000.

Per M: one eager call, CUDA-graph capture of the whole chain, then per pass 3 warmup replays + 15 timed replays
(CUDA events per replay); 3 passes (M order forward, reversed, forward). kernel_ms = median per-forward ms over
the 45 timed replays, noise_floor = (p90 - p10) / median. 20 s pre-heat; NVML SM clock sampled during the timed
replays. The untimed repeated-execution twin of this sequence is check_enum.py.

usage: python bench_chain.py <checkpoint>
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import Chain  # noqa: E402
from enum_sweep import Clock  # noqa: E402
from ltx import main_dispatch  # noqa: E402

BUCKETS = [300, 600, 900, 1000, 1200, 1500, 1800, 2000]
WARM, ITERS, PASSES = 3, 15, 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def capture(chain, op, M):
    chain.run(op, M)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain.run(op, M)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = chain.run(op, M)
    torch.cuda.synchronize()
    return g, out


def main():
    ckpt = sys.argv[1]
    op, routed, mg = main_dispatch()
    chain = Chain(ckpt, max(BUCKETS))
    log("chain", json.dumps(chain.info), torch.cuda.get_device_name(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    route_desc = {}
    for M in BUCKETS:
        route_desc[M] = {f"{k}_{p}": routed(chain.info[k][p][0], chain.info[k][p][1], M)
                         for k in ("sl", "gl") for p in ("qkv", "o")}
    log("routed to cublas_mm per M:", json.dumps(route_desc))
    graphs = {M: capture(chain, op, M) for M in BUCKETS}
    log("captured", len(graphs), "graphs")
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t_end = time.time() + 20
    while time.time() < t_end:
        for _ in range(20):
            torch.matmul(a, a)
        torch.cuda.synchronize()
    del a
    log("pre-heat done")
    samples = {M: [] for M in BUCKETS}
    clocks = []
    clock = Clock()
    for p in range(PASSES):
        order = BUCKETS if p % 2 == 0 else list(reversed(BUCKETS))
        for M in order:
            g, _ = graphs[M]
            for _ in range(WARM):
                g.replay()
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
            clock.start()
            for s, e in evs:
                s.record()
                g.replay()
                e.record()
            torch.cuda.synchronize()
            ck = clock.stop()
            if ck:
                clocks.append(ck["sm_mhz_med"])
            samples[M] += [s.elapsed_time(e) for s, e in evs]
        log(f"pass {p} done")
    batch = {}
    for M in BUCKETS:
        t = sorted(samples[M])
        n = len(t)
        med, p10, p90 = t[n // 2], t[int(0.1 * n)], t[min(n - 1, int(0.9 * n))]
        batch[str(M)] = {"kernel_ms": round(med, 4), "noise_floor": round(max((p90 - p10) / med, 1e-4), 4),
                         "p10": round(p10, 4), "p90": round(p90, 4), "n": n}
        log(f"M={M}: {med:.3f} ms/forward [p10 {p10:.3f}, p90 {p90:.3f}]")
    cs = sorted(clocks)
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "routed": route_desc,
                      "sm_clock_mhz": {"min": cs[0], "med": cs[len(cs) // 2], "max": cs[-1]} if cs else None}))


if __name__ == "__main__":
    main()
