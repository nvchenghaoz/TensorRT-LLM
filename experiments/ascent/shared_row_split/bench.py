"""Paired timing: DeepseekV4MoE.forward shared chain + add, one TP4 rank (rank 0).

BASE tree = main path (all M rows, *0.5, out-of-place add); CAND tree = row
split (ceil(M/2) rows, no scale, in-place slice add) at M >= T. Warm L2:
weights and inputs stay resident; the routed buffer is restored before each
timed call (reset point, outside the timed region).
Usage: CKPT=<ckpt> python bench.py
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_moe, build_shared_mlp, log, sm_clock_mhz  # noqa: E402

MS = (128, 2048, 6144)
REPEATS, WARMUP, ITERS = 7, 5, 30


def main():
    import tensorrt_llm

    torch.manual_seed(0)
    rank = 0
    mlp, shared_tp, scale, info = build_shared_mlp(rank)
    moe, stub, has_split = build_moe(rank, mlp, shared_tp, scale)
    arm = "CAND" if has_split else "BASE"
    log(arm, "tensorrt_llm", tensorrt_llm.__file__, "info", json.dumps(info),
        "split_parts", getattr(moe, "_shared_row_split_parts", None))
    hidden = info["hidden"]
    gen = torch.Generator(device="cpu").manual_seed(7)
    res = {}
    for M in MS:
        hs = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
        routed0 = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
        buf = torch.empty_like(routed0)
        stub.buf = buf
        rep_meds, all_t = [], []
        for rep in range(REPEATS):
            ts = []
            for it in range(WARMUP + ITERS):
                buf.copy_(routed0)  # reset point
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                out = moe.forward(hs)
                e1.record()
                torch.cuda.synchronize()
                if it >= WARMUP:
                    ts.append(e0.elapsed_time(e1))
            ts.sort()
            med = ts[len(ts) // 2]
            rep_meds.append(med)
            all_t += ts
            log(f"{arm} M={M} rep={rep} median_ms={med:.5f} p10={ts[int(0.1*(len(ts)-1))]:.5f} "
                f"p90={ts[int(0.9*(len(ts)-1))]:.5f}")
        clk = sm_clock_mhz()
        all_t.sort()
        n = len(all_t)
        med_all = all_t[n // 2]
        rep_meds.sort()
        med = rep_meds[len(rep_meds) // 2]
        res[str(M)] = {
            "kernel_ms": med,
            "noise_floor": (all_t[int(0.9 * (n - 1))] - all_t[int(0.1 * (n - 1))]) / med_all,
            "repeat_spread": (rep_meds[-1] - rep_meds[0]) / med,
            "repeat_medians": rep_meds,
            "sm_clock_mhz": clk,
            "out_shape": list(out.shape),
        }
        log(f"{arm} M={M} RESULT kernel_ms={med:.5f} noise={res[str(M)]['noise_floor']:.4f} "
            f"repeat_spread={res[str(M)]['repeat_spread']:.4f} sm_clock={clk}")
        del hs, routed0, buf, out
        torch.cuda.empty_cache()
    print(json.dumps({"arm": arm, "batch": res, "info": info,
                      "output": {"class": "reordered"}}))


if __name__ == "__main__":
    main()
