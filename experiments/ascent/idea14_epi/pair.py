"""Idea 14 paired microbenchmark (run_gpu_pair: the same script in base and candidate trees).

Times the served FP8-direct QKV op, flux3_primsts.nvfp4_qkv_rope_fp8_impl (base: main's QKV3 GEMM
+ qknorm_rope_fp8; candidate: the fused E4M3 epilogue route), on synthetic NVFP4 inputs at the
served shapes. One CUDA graph per case replays the case's per-step call sequence:
  t2i-2048: 4 x M16384 (mode blocks) + 28 x M16464 (single blocks)
  i2i-2048: 4 x M16384 + 28 x M32848
(keep_v=False as served without PAG), plus per-M graphs (R calls each). Each timing = one graph
replay (CUDA events); median of ITERS replays, cases interleaved; noise = (p90 - p10) / median.
Last stdout line: evidence JSON, kernel_ms = ms per denoise step of this op for the case.
"""

import json
import os
import sys
import time

import torch

N = 3072
K = 3072
SM = 128 ** -0.5


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sf_rows(m):
    return (m + 127) // 128 * 128


def make_inputs(m, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + m)
    a = torch.randint(0, 256, (m, K // 2), generator=g, dtype=torch.uint8).to(dev)
    w = torch.randint(0, 256, (3 * N, K // 2), generator=g, dtype=torch.uint8).to(dev)
    sfa = torch.randint(0x30, 0x40, (sf_rows(m) * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    sfb = torch.randint(0x30, 0x40, (3 * N * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    alpha = torch.tensor([0.0075, 0.0125, 0.005], dtype=torch.float32, device=dev)
    qn = (1.0 + 0.3 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev)
    kn = (0.8 + 0.4 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev)
    ang = torch.rand(m, 64, generator=g) * 6.28
    cs = torch.cat([ang.cos(), ang.sin()], -1).float().contiguous().to(dev)
    pos = torch.arange(m, dtype=torch.int64, device=dev)
    return a, w, sfa, sfb, alpha, qn, kn, cs, pos


def stats(t):
    t = sorted(t)
    med = t[len(t) // 2]
    return med, (t[int(0.9 * len(t)) - 1] - t[int(0.1 * len(t))]) / med


def main():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi

        epi.preload(dev)
        tree = f"candidate (fused route enabled={epi.enabled()})"
    except ImportError:
        tree = "base (no fused module)"
    log("tree:", tree)
    ITERS = int(os.environ.get("PAIR_ITERS", "40"))
    R = 4
    inputs = {m: make_inputs(m, dev) for m in (16384, 16464, 32848)}

    def call(m):
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = inputs[m]
        return fp.nvfp4_qkv_rope_fp8_impl(a, w, sfa, sfb, qn, kn, cs, pos, alpha, SM, False)

    seqs = {"t2i-2048": [16384] * 4 + [16464] * 28, "i2i-2048": [16384] * 4 + [32848] * 28}
    for m in (16384, 16464, 32848):
        seqs[f"M{m}"] = [m] * R
    for m in inputs:
        call(m)
    torch.cuda.synchronize()
    graphs = {}
    for name, seq in seqs.items():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for m in seq[:2]:
                call(m)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for m in seq:
                call(m)
        graphs[name] = g
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    times = {n: [] for n in graphs}
    for it in range(ITERS):
        for name, g in graphs.items():
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            times[name].append(e0.elapsed_time(e1))
        if it % 10 == 0:
            log(f"iter {it}")
    batch = {}
    for name in ("t2i-2048", "i2i-2048"):
        med, nf = stats(times[name])
        batch[name] = {"kernel_ms": round(med, 4), "noise_floor": round(nf, 4)}
    per_m = {}
    for m in (16384, 16464, 32848):
        med, nf = stats(times[f"M{m}"])
        per_m[str(m)] = {"ms_per_call": round(med / R, 4), "noise_floor": round(nf, 4)}
    log("cases", json.dumps(batch), "per_call", json.dumps(per_m))
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "per_call": per_m,
                      "tree": tree,
                      "unit": "kernel_ms = ms per denoise step of the FP8-direct QKV op "
                              "(4 x M16384 + 28 x single-block M) in one graph replay"}))


if __name__ == "__main__":
    sys.exit(main())
