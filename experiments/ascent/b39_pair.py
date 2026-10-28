"""Branch 39 pair microbenchmark: FLUX.3 SM120 LINEAR op (attn_out + down) per block.

Times torch.ops.trtllm.flux3_primsts_nvfp4_linear (what PreparedNVFP4 dispatches to) at the
FLUX.3 Klein LINEAR shapes, attn_out (M,3072,K=3072) and down (M,3072,K=9216), down fed by the real
CuTe run('packed') producer. M=4176 (24 single blocks, T=80) and M=4096 (4 image blocks).
Main: TRT-LLM FP4GemmRunner tactic 0. Branch: FlashInfer stock SM120 fp4_gemm, pinned tactic.
CUDA events per call, the two M cases interleaved inside each iteration, median of ITERS.
kernel_ms = per-NVFP4-block mean of (attn_out + down) weighted 24:4 over M=4176:4096.
"""

import json
import sys

import torch

sys.path.insert(0, "experiments/ascent")
import tensorrt_llm  # noqa: E402,F401
from b39_linear_sweep import make_case  # noqa: E402

WARMUP = 5
ITERS = 200
TAIL_PAIR = len(sys.argv) > 1 and sys.argv[1] == "tailpair"


def log(*a):
    print(*a, flush=True)


def op(args):
    return torch.ops.trtllm.flux3_primsts_nvfp4_linear(*args)


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def main():
    cases = {M: make_case(M, seed=M) for M in (4176, 4096)}
    tensors = [t for c in cases.values() for args in c.values() for t in args
               if isinstance(t, torch.Tensor)]
    pristine = [t.clone() for t in tensors]
    for _ in range(WARMUP):
        for c in cases.values():
            op(c["attn_out"])
            op(c["down"])
    torch.cuda.synchronize()
    for p, t in zip(pristine, tensors):
        t.copy_(p)
    ev = {(M, s): [] for M in cases for s in ("attn_out", "down")}
    if TAIL_PAIR:
        # Main's real schedule (landing 20): down on the current stream || attn_out on the tail-pair
        # side stream, forked and joined by flux3_primsts._sm120_tail_pair. One event pair spans
        # the whole fork/join; it is booked under "down" and "attn_out" gets 0.
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

        def pair(c):
            return fp._sm120_tail_pair(c["down"][0].device, lambda: op(c["down"]),
                                       lambda: op(c["attn_out"]))

        for _ in range(WARMUP):
            for c in cases.values():
                pair(c)
        torch.cuda.synchronize()
        for p, t in zip(pristine, tensors):
            t.copy_(p)
        ref = {M: tuple(o.clone() for o in pair(c)) for M, c in cases.items()}
        torch.cuda.synchronize()
        for p, t in zip(pristine, tensors):
            t.copy_(p)
    zero = None
    for i in range(ITERS):
        for M, c in cases.items():
            if TAIL_PAIR:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                pair(c)
                b.record()
                ev[(M, "down")].append((a, b))
                if zero is None:
                    zero = torch.cuda.Event(enable_timing=True)
                    zero.record()
                ev[(M, "attn_out")].append((zero, zero))
                continue
            for s in ("attn_out", "down"):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                op(c[s])
                b.record()
                ev[(M, s)].append((a, b))
        if i % 50 == 0:
            log(f"iter {i}")
    torch.cuda.synchronize()
    if TAIL_PAIR:
        # Untimed correctness pass, same calls as the timed loop: repeat outputs vs the first call.
        rep_ok = True
        for _ in range(3):
            for M, c in cases.items():
                out = pair(c)
                rep_ok &= all(torch.equal(x, y) for x, y in zip(out, ref[M]))
        torch.cuda.synchronize()
        log(f"tail-pair repeat bitwise vs first call: {rep_ok}")
    for p, t in zip(pristine, tensors):
        assert torch.equal(p, t), "input mutated"
    ms = {k: [a.elapsed_time(b) for a, b in v] for k, v in ev.items()}
    per = {f"M{M}_{s}": stat(v) for (M, s), v in ms.items()}
    blk = [(24 * (ms[(4176, "attn_out")][i] + ms[(4176, "down")][i])
            + 4 * (ms[(4096, "attn_out")][i] + ms[(4096, "down")][i])) / 28 for i in range(ITERS)]
    med, noise = stat(blk)
    for k, (m, n) in per.items():
        log(f"{k}: {m * 1e3:.2f} us (noise {n:.3f})")
    log(f"per-block attn_out+down: {med * 1e3:.2f} us (noise {noise:.3f}); "
        f"per request x16 steps x28 blocks = {med * 448:.2f} ms")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "per_shape_ms": {k: [m, n] for k, (m, n) in per.items()},
        "per_request_ms": med * 448,
        "tail_pair": TAIL_PAIR,
        "repeat_bitwise": bool(rep_ok) if TAIL_PAIR else None,
        "output": {"class": "reordered"},
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
