"""Idea 32 / idea 50 Stage A, GATE 1 (plan 460): at the served i2i single-block M=32848 (80-row
partial tile included), is the 128x128 pair at AlongN faster than main's (63571fc0) route, the
t256 (256x128x128) pair at AlongM? Arms, all on the one shared lpt.workspace:
  n128    : lpt.module().lpt_pair(..., raster=1 AlongN, swizzle=1)    (branch 10's module)
  t256    : flux3_sm120_large_m.pair_module().lpt_pair(...)           (main's route at M>=8192)
  t256_AA : a second capture of the t256 arm (A/A noise)
  m128    : lpt.module() at main's order (raster 0, swizzle 1)       (data: pre-landing-6 route)
Each arm = one CUDA graph of CALLS calls rotating SETS distinct input sets (A operands ~150 MB at
32848, so every call reads cold from DRAM), interleaved replays with rotated order, CUDA-event
medians per call. 16464 / 16384 are recorded as data. Bitwise: every arm vs FI tactic 17.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lpt_raster"))
import bench as lb  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt  # noqa: E402

MS = [32848, 16464, 16384]
SETS = 3
CALLS = 21
SAMPLES = {32848: 30, 16464: 30, 16384: 30}


def call(arm, c, ws):
    args = (c["a_down"], c["w_down"], c["sf_down"], c["sfw_down"], c["al_down"], c["o_down"],
            c["a_attn"], c["w_out"], c["sf_attn"], c["sfw_out"], c["al_attn"], c["o_attn"], ws,
            lpt.sched_mode(), 0)
    if arm == "n128":
        lpt.module().lpt_pair(*args, 1, 1)
    elif arm == "m128":
        lpt.module().lpt_pair(*args, 0, 1)
    else:
        lm.pair_module().lpt_pair(*args)


def main():
    dev = torch.device("cuda", torch.cuda.current_device())
    lb.fp.sm120_linear_preload(dev)
    lpt.module()
    lm.pair_module()
    ws = lpt.workspace(dev)
    lb.log(f"device {torch.cuda.get_device_name()} pair sched {lpt.sched_mode()}")
    out = {}
    passed = True
    for M in MS:
        cs = [lb.make_case(M, 9000 + 10 * M + i) for i in range(SETS)]
        pristine = [[c[k].clone() for k in lb.IN_KEYS] for c in cs]
        bit = {}
        for arm in ("n128", "t256", "m128"):
            ok = True
            for c in cs:
                r17 = (lb.fi_linear(c["a_attn"], c["w_out"], c["sf_attn"], c["sfw_out"],
                                    c["al_attn"], 17),
                       lb.fi_linear(c["a_down"], c["w_down"], c["sf_down"], c["sfw_down"],
                                    c["al_down"], 17))
                for _ in range(2):
                    c["o_attn"].fill_(float("nan"))
                    c["o_down"].fill_(float("nan"))
                    call(arm, c, ws)
                    torch.cuda.synchronize()
                    ok &= torch.equal(c["o_attn"], r17[0]) and torch.equal(c["o_down"], r17[1])
                del r17
            bit[arm] = ok
            passed &= ok
        s = torch.cuda.Stream()
        graphs = {}
        for arm in ("n128", "t256", "t256_AA", "m128"):
            a = "t256" if arm == "t256_AA" else arm
            graphs[arm] = lb.capture([lambda c=cs[i % SETS], a=a: call(a, c, ws)
                                      for i in range(CALLS)], s)
        t = lb.time_graphs(graphs, samples=SAMPLES[M], warm=3, calls=CALLS)
        for p, c in zip(pristine, cs):
            assert all(torch.equal(x, c[k]) for x, k in zip(p, lb.IN_KEYS)), "inputs changed"
        row = {}
        for n, v in t.items():
            med, p10, p90 = lb.stat(v)
            row[n] = {"median_us": round(med, 1), "p10": round(p10, 1), "p90": round(p90, 1),
                      "noise": round((p90 - p10) / med, 4), "n": len(v)}
        aa = abs(row["t256_AA"]["median_us"] - row["t256"]["median_us"]) / row["t256"]["median_us"]
        row["n128_vs_t256"] = round(row["n128"]["median_us"] / row["t256"]["median_us"], 4)
        row["aa_rel_diff"] = round(aa, 4)
        row["bitwise_vs_t17"] = bit
        out[str(M)] = row
        lb.log(f"M={M}: {json.dumps(row)}")
        del graphs, cs, pristine
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(passed), "gate1": out, "sets": SETS, "calls_per_graph": CALLS,
                      "unit": "us per pair call (down K=9216 + attn_out K=3072, N=3072)"}))


if __name__ == "__main__":
    main()
