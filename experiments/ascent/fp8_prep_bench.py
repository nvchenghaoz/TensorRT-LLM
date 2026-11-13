"""Per-call FP8 attention prep time (per-tensor amax + E4M3 quantize of Q, K, V), one TP2 rank.

Shape: B=1, S=37966, H=28, D=128 BF16; Q, K contiguous, V a strided view of a packed qkv buffer
(row stride 3*H*D = 10752, column offset 2*H*D) exactly as transformer_minimax_h3.py passes it.

Arms (same script in both trees):
  - main (no fp8_attn_prep module): 3 x FlashInferAttention._quantize_e4m3_per_tensor
    (V.contiguous() + C++ quantize_e4m3_per_tensor) + torch.cat of the descales;
  - branch: fp8_attn_prep.quantize_e4m3_qkv_per_tensor (2 Triton launches, V read in place).
Each arm also times main's prep again as an A/A control (same code in both trees).
L2 is flushed (256 MB write) before every timed iteration, outside the timed region; every
iteration ends with the descale readback (.tolist()) as serving does, also outside the timing.
Timing: 10 warmup + 40 timed, CUDA events; median, p10, p90, noise=(p90-p10)/median.
--sweep: branch only, time programs_per_sm x reverse x BLOCK_R x num_warps configs.
Last stdout line: {"batch": {"1": {"kernel_ms", "noise_floor"}}, "output": {"class": "exact"}, ...}
"""

import json
import statistics
import sys

import torch

S, H, D = 37966, 28, 128
WARMUP, ITERS = 10, 40


def log(*a):
    print(*a, flush=True)


def make_inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    W = H * D
    qkv = (torch.randn(1, S, 3 * W, device="cuda", generator=g) * 2.5).bfloat16()
    q = (torch.randn(1, S, H, D, device="cuda", generator=g) * 1.7).bfloat16()
    k = (torch.randn(1, S, H, D, device="cuda", generator=g) * 3.1).bfloat16()
    v = qkv[..., 2 * W:].unflatten(-1, (H, D))
    return q, k, v, qkv


def main_prep(q, k, v):
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention

    qf, qd = FlashInferAttention._quantize_e4m3_per_tensor(q)
    kf, kd = FlashInferAttention._quantize_e4m3_per_tensor(k)
    vf, vd = FlashInferAttention._quantize_e4m3_per_tensor(v)
    return qf, kf, vf, torch.cat([qd, kd, vd])


def time_fn(fn, flush, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        out = fn()
        out[3].tolist()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.fill_(1)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        out[3].tolist()
        e.synchronize()
        times.append(s.elapsed_time(e))
        del out
    times.sort()
    med = statistics.median(times)
    p10 = times[int(0.1 * (iters - 1))]
    p90 = times[int(0.9 * (iters - 1))]
    return med, p10, p90, (p90 - p10) / med


def main():
    import tensorrt_llm  # noqa: F401

    try:
        from tensorrt_llm._torch.visual_gen.attention.backends import fp8_attn_prep as prep
    except ImportError:
        prep = None
    q, k, v, qkv = make_inputs()
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    log(f"shape B=1 S={S} H={H} D={D}; V stride {tuple(v.stride())}; branch module: {prep is not None}")

    if "--sweep" in sys.argv:
        assert prep is not None
        ref = main_prep(q, k, v)
        res = []
        for bR in (8, 16, 32):
            for nw in (4, 8):
                for pps in (2, 3, 4, 6, 8):
                    for rev in (False, True):
                        prep._BLOCK_R, prep._NUM_WARPS = bR, nw

                        def fn():
                            return prep.quantize_e4m3_qkv_per_tensor(q, k, v, programs_per_sm=pps,
                                                                     reverse=rev)
                        o = fn()
                        ok = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                 for a, b in zip(o[:3], ref[:3])) and torch.equal(o[3], ref[3])
                        med, p10, p90, nf = time_fn(fn, flush, iters=20, warmup=3)
                        log(f"BLOCK_R={bR} warps={nw} pps={pps} reverse={rev}: {med:.4f} ms "
                            f"(p10 {p10:.4f} p90 {p90:.4f}) bitwise={ok}")
                        res.append((med, bR, nw, pps, rev, ok))
        res.sort()
        log("best:", res[:5])
        m, *_ = time_fn(lambda: main_prep(q, k, v), flush)
        log(f"main prep: {m:.4f} ms")
        print(json.dumps({"sweep_best": res[:5], "main_ms": m}))
        return

    aa = time_fn(lambda: main_prep(q, k, v), flush)
    log(f"A/A main prep: median {aa[0]:.4f} ms p10 {aa[1]:.4f} p90 {aa[2]:.4f} noise {aa[3]:.4f}")
    if prep is not None:
        arm = "branch: fp8_attn_prep (2 Triton launches)"
        r = time_fn(lambda: prep.quantize_e4m3_qkv_per_tensor(q, k, v), flush)
    else:
        arm = "main: V.contiguous + 3x C++ quantize_e4m3_per_tensor + cat"
        r = time_fn(lambda: main_prep(q, k, v), flush)
    log(f"{arm}: median {r[0]:.4f} ms p10 {r[1]:.4f} p90 {r[2]:.4f} noise {r[3]:.4f}")
    aa2 = time_fn(lambda: main_prep(q, k, v), flush)
    log(f"A/A main prep (after): median {aa2[0]:.4f} ms noise {aa2[3]:.4f}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": round(r[0], 5), "noise_floor": round(r[3], 5)}},
        "output": {"class": "exact"},
        "arm": arm,
        "p10_ms": r[1], "p90_ms": r[2],
        "aa_main_prep_ms": [aa[0], aa2[0]], "aa_noise": [aa[3], aa2[3]],
    }))


if __name__ == "__main__":
    main()
