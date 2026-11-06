"""run_gpu_pair benchmark for idea 239: CSA2 shared-context prefill attention as each tree runs it.

Times CSA2FlashInfer._forward_selected (the production dispatch) over CSA2's real packed
pools and staging kernels with a minimal attention/metadata stand-in (check.py), eager,
CUDA events. On main this is stage_shared_rows (SharedCompact + 2 SharedDecode) +
stage_shared_indices_and_mask + FA2 paged prefill + sink_epilogue; on the branch it is
the int32 index prep + FlashInfer FullBF16 direct sparse prefill. Same script, same
inputs (fixed seeds) in both arms. Cases: 8 shapes (rows x width) plus the c1 request
(10 x each of 2048x640, 2048x128, 128x640, 128x128) and the c4 follower step
(6144/384 rows) 40-call sequences. Also reports max abs error vs an FP64 sink attention.
Last stdout line: {"batch": {case: {"kernel_ms", "noise_floor"}}, "output": {...}}.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import FakeAttention, forward_args, make_metadata  # noqa: E402
from stage0 import DEV, HEADS, Scenario, errors, gpu_memory, log  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer  # noqa: E402

ITERS, WARMUP = 30, 3


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    med = times[len(times) // 2]
    p10, p90 = times[int(0.1 * (len(times) - 1))], times[int(0.9 * (len(times) - 1))]
    return {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10_ms": p10, "p90_ms": p90}


class Case:
    def __init__(self, s):
        self.s = s
        self.helper = CSA2FlashInfer(attention=FakeAttention(s))
        self.md = make_metadata(s)
        self.out = torch.empty(s.count, HEADS * 512, dtype=torch.bfloat16, device=DEV)
        self.fa = forward_args(s, self.out)
        self.q2d = s.q.view(s.count, HEADS * 512)

    def __call__(self):
        return self.helper._forward_selected(self.q2d, self.md, self.fa)


def main():
    log("gpu memory.used at start:", gpu_memory())
    try:
        import tensorrt_llm._torch.attention.backends.sparse.csa2.fi_sparse_prefill  # noqa: F401

        tree = "branch (FI FullBF16 route present)"
    except ImportError:
        tree = "main (no FI route)"
    log("tree:", tree)
    shapes = [
        (1, 2048, True),
        (1, 2048, False),
        (1, 128, True),
        (1, 128, False),
        (3, 2048, True),
        (3, 2048, False),
        (3, 128, True),
        (3, 128, False),
    ]
    cases = {}
    for i, (r, t, hm) in enumerate(shapes):
        s = Scenario(r, t, hm, seed=1000 + i)
        cases[s.name] = Case(s)
    batch, errs = {}, {}
    for name, c in cases.items():
        c()
        torch.cuda.synchronize()
        ref = c.s.reference()
        errs[name] = errors(c.out.view_as(c.s.q), ref)
        del ref
        torch.cuda.empty_cache()
        batch[name] = timed(c, ITERS, WARMUP)
        log(f"{name}: {batch[name]['kernel_ms']:.4f} ms (noise {batch[name]['noise_floor']:.3f}) err {errs[name]}")
    seqs = {
        "c1_request_40": ["2048x640", "2048x128"] * 10 + ["128x640", "128x128"] * 10,
        "c4_follower_40": ["6144x640", "6144x128"] * 10 + ["384x640", "384x128"] * 10,
    }
    for label, names in seqs.items():
        def run(names=names):
            for nm in names:
                cases[nm]()
        batch[label] = timed(run, 15, 2)
        log(f"{label}: {batch[label]['kernel_ms']:.3f} ms (noise {batch[label]['noise_floor']:.3f})")
    worst = max(e["max_abs"] for e in errs.values())
    print(
        json.dumps(
            {
                "batch": batch,
                "output": {"class": "reordered", "max_abs_vs_fp64": worst, "per_case_vs_fp64": errs},
                "tree": tree,
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
