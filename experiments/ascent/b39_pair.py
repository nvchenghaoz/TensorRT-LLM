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
    for i in range(ITERS):
        for M, c in cases.items():
            for s in ("attn_out", "down"):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                op(c[s])
                b.record()
                ev[(M, s)].append((a, b))
        if i % 50 == 0:
            log(f"iter {i}")
    torch.cuda.synchronize()
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
        "output": {"class": "reordered"},
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
