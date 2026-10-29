"""Branch 124 (idea 192) pair bench: the changed op alone.

One sample = CUDA-event-timed replay of ONE CUDA graph holding one denoise step's
flux3_sm120_qknorm_rope_fp8.qknorm_rope_fp8 calls (fold mode 1 + mode 2 each): 28 x M=4176 +
4 x M=4096, inputs laid out as the route hands them (planes of one contiguous [3, M, 3072] BF16
GEMM output, BF16 norm weights, FP32 cos_sin [M,128], int64 positions). Default env in both trees:
base = main's landing-58 fold kernel, candidate = the VEC kernel. The QKV GEMMs (unchanged) are
left out, unlike b115_fold/bench.py. 10 s soak, 200 samples, median, noise (p90-p10)/median.
Inputs are allocated once (fixed seeds) and only read; checked unchanged after timing.
"""

import json
import statistics
import time

import torch

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)

SOAK_S, SAMPLES = 10.0, 200
MIX = [4176] * 28 + [4096] * 4
H = 3072
SM_SCALE = 128 ** -0.5


def log(*a):
    print(*a, flush=True)


def main():
    route = "vec" if getattr(qr8, "VEC", False) else ("fold" if qr8.FOLD else "landing54")
    log("route", route, "mix 28 x 4176 + 4 x 4096")
    sets = {}
    for M in (4176, 4096):
        g = torch.Generator(device="cuda").manual_seed(M)
        planes = (torch.randn(3, M, H, device="cuda", generator=g) * 3).bfloat16()
        qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
        cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        sets[M] = (planes[0], planes[1], planes[2], qn, kn, cs, pos)
    pristine = {M: [t.clone() for t in s] for M, s in sets.items()}

    def body():
        for M in MIX:
            qr8.qknorm_rope_fp8(*sets[M], SM_SCALE)

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
            log("sample", it)
    torch.cuda.synchronize()
    unchanged = all(torch.equal(t, p) for M in sets for t, p in zip(sets[M], pristine[M]))
    t = sorted(x.elapsed_time(y) for x, y in ev)
    med = statistics.median(t)
    p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
    rel_std = statistics.pstdev(t) / statistics.mean(t)
    log(f"{route}: {med * 1000:.1f} us/step noise {(p90 - p10) / med:.4f} inputs_unchanged {unchanged}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": round(med, 5), "noise_floor": round((p90 - p10) / med, 5)}},
        "output": {"class": "exact"},
        "route": route,
        "inputs_unchanged": unchanged,
        "rel_std": round(rel_std, 5),
        "what": "per denoise step: 28 x M=4176 + 4 x M=4096 qknorm_rope_fp8 (fold mode 1 + mode 2) "
                "on planar BF16 planes, one CUDA graph per sample, 200 samples",
        "detail": {"p10": round(p10, 5), "p90": round(p90, 5), "us_per_call": round(med * 1000 / 32, 2)},
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
