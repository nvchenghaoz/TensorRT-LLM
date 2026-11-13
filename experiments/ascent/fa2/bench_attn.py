"""Paired microbenchmark: VanillaAttention.forward at the MiniMax-H3 TP2 per-rank shape.

Baseline tree: torch SDPA (FA2 <128,128,64,4>). Candidate tree: fa2_sm120 (bitwise FA2,
256-row/8-warp tile + 64-row tail). Same script in both trees. Prints the evidence JSON last.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_inputs import D, H, make_inputs  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention  # noqa: E402

WARMUP, ITERS = 10, 30


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    p10 = ts[int(0.1 * (n - 1))]
    p90 = ts[int(round(0.9 * (n - 1)))]
    return med, p10, p90


def main():
    sizes = [int(x) for x in os.environ.get("SIZES", "37966,37769").split(",")]
    t0 = time.time()
    attn = VanillaAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)
    print(f"init {time.time() - t0:.1f}s fa2_sm120={getattr(attn, '_fa2_sm120', None)}", flush=True)
    res = {}
    for S in sizes:
        q, k, v, _ = make_inputs(S, seed=S)
        fn = lambda: attn.forward(q, k, v)  # noqa: E731
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ts = []
        for i in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        med, p10, p90 = stats(ts)
        flops = 4.0 * S * S * D * H
        r = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med,
             "tflops": flops / (med * 1e-3) / 1e12, "S": S}
        print("S", S, json.dumps(r), flush=True)
        res[S] = r
        del q, k, v
        torch.cuda.empty_cache()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        q, k, v, _ = make_inputs(1000, seed=1)
        attn.forward(q, k, v)
        torch.cuda.synchronize()
    kn = sorted({e.name[:160] for e in prof.events() if "flash" in e.name})
    print("kernels:", kn, flush=True)
    main_s = sizes[0]
    out = {"batch": {"1": {"kernel_ms": res[main_s]["kernel_ms"], "noise_floor": res[main_s]["noise_floor"]}},
           "output": {"class": "exact"},
           "per_S": {str(s): r for s, r in res.items()},
           "kernels": kn}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
