"""Branch 115 (idea 182) pair bench: branch 100's op.py bench with landing 56's per-step mix.

One sample = CUDA-event-timed replay of ONE CUDA graph holding one denoise step's routed NVFP4
QKV -> E4M3 calls: 28 x M=4176 (single blocks, incl. 24-27 since landing 56) + 4 x M=4096 (image
mode blocks), each flux3_primsts.nvfp4_qkv_rope_fp8_impl (3 planar GEMMs + qknorm_rope_fp8).
Base tree runs landing 54's op (mode 1 + one-program reduce + mode 2); candidate runs the fold
(default env). FMHA excluded (unchanged, identical inputs). 10 s soak, 200 samples, median,
noise (p90-p10)/median. Inputs are allocated once (fixed seeds) and only read.
"""

import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b100_fp8direct"))
import op as b100  # noqa: E402

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)

SOAK_S, SAMPLES = 10.0, 200
MIX = [4176] * 28 + [4096] * 4


def main():
    fold = getattr(qr8, "FOLD", None)
    route = "fold" if fold else "landing54"
    b100.log("route", route, "mix", "28 x 4176 + 4 x 4096")
    sets = {M: b100.mk(M, M) for M in (4176, 4096)}
    pristine = {M: [t.clone() if isinstance(t, torch.Tensor) else t for t in a]
                for M, a in sets.items()}

    def body():
        for M in MIX:
            b100.cand_chain(sets[M])

    body()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.cuda.synchronize()
    for M in sets:  # restore pristine inputs before timing (they are read-only; this is a no-op)
        for t, p in zip(sets[M], pristine[M]):
            if isinstance(t, torch.Tensor):
                t.copy_(p)
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        graph.replay()
        torch.cuda.synchronize()
    ev = []
    for it in range(SAMPLES):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        ev.append((a, b))
        if it % 50 == 0:
            torch.cuda.synchronize()
            b100.log("sample", it)
    torch.cuda.synchronize()
    t = sorted(x.elapsed_time(y) for x, y in ev)
    med = statistics.median(t)
    p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
    mean = statistics.mean(t)
    rel_std = statistics.pstdev(t) / mean
    b100.log(f"{route}: {med * 1000:.1f} us/step noise {(p90 - p10) / med:.4f} rel_std {rel_std:.4f}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": round(med, 5), "noise_floor": round((p90 - p10) / med, 5)}},
        "output": {"class": "exact"},
        "route": route,
        "rel_std": round(rel_std, 5),
        "what": "per denoise step: 28 x M=4176 + 4 x M=4096 routed NVFP4 QKV -> E4M3 q/k/v + "
                "scales (nvfp4_qkv_rope_fp8_impl), one CUDA graph per sample, 200 samples",
        "detail": {"p10": round(p10, 5), "p90": round(p90, 5), "us_per_call": round(med * 1000 / 32, 2)},
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
