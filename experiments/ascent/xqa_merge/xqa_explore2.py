#!/usr/bin/env python3
# Exploration only: more variants of the restructured merge (late prologue, fine-grained stats-phase timing).
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xqa_bench as xb  # noqa: E402
import xqa_explore as xe  # noqa: E402
import xqa_newmerge as nm  # noqa: E402
from xqa_bench import _rep, compile_xqa  # noqa: E402

PRO_BEG = "                // Each warp streams its partial result tiles through a ring"
PRO_END = "                // Row max and row sum of the partial results of a chunk"
LOOP_MARK = "                // rescale and accumulate, in the order of the sub-sequences\n"


def late_prologue(src):
    i0 = src.index(PRO_BEG)
    i1 = src.index(PRO_END)
    block = src[i0:i1]
    src = src[:i0] + src[i1:]
    return _rep(src, LOOP_MARK, block + LOOP_MARK)


def fine_timing(src):
    src = xe.patch_timing_common(src)
    src = _rep(src, LOOP_MARK, "                long long const dbgC1 = clock64();\n" + LOOP_MARK)
    src = _rep(src, PRO_END, "                long long const dbgCa = clock64();\n" + PRO_END)
    a = "                // get row max. Max is exact"
    src = _rep(src, a, "                long long dbgCb = 0;\n" + a)
    a = "                    mbbuf.partialRowMax[warpIdx.x].storeFromReg<false>(warp, warpRowMax);\n"
    src = _rep(src, a, "                    dbgCb = clock64();\n" + a)
    a = "                // Scaled row sum of each partial result of a chunk"
    src = _rep(src, a, "                long long const dbgCc = clock64();\n" + a)
    a = "                    dbgMerge[3] = dbgT2;\n"
    src = _rep(src, a, a + "                    dbgMerge[4] = dbgCa - dbgC0;\n                    dbgMerge[5] = dbgCb - dbgCa;\n"
               "                    dbgMerge[6] = dbgCc - dbgCb;\n                    dbgMerge[7] = dbgC1 - dbgCc;\n")
    return src


def fine_timing_late(src):
    # late prologue: the stats phase starts right away; stamp Ca at the same textual place (now before the stats).
    return fine_timing(src)


_prev_variant_patch = xe.variant_patch


def variant_patch(spec):
    parts = spec.split("_")
    head = parts[0]
    if head in ("nml", "nmf", "nmfl"):
        d, c = xe._nm_params(spec)

        def p(s, head=head, d=d, c=c):
            s = nm.patch_newmerge(s, d, c)
            if head in ("nml", "nmfl"):
                s = late_prologue(s)
            if head in ("nmf", "nmfl"):
                s = fine_timing(s)
            return s

        return False, p
    return _prev_variant_patch(spec)


xe.variant_patch = variant_patch


def mode_timeline3(a):
    base = xb.extract_base_tree(a.base_commit)
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    configs = xb.parse_configs(a.configs)
    inp = xb.build_inputs(max(b for b, _ in configs))
    flush = xb.Flusher(a.flush, a.flush_mb)
    off, raw = inp["scratch_off"], inp["scratch_raw"]
    out = {}
    for spec in a.timing_variant.split(","):
        kern = xb.Kernel(xe.make_variant(spec, tree, base))
        for (b, m) in configs:
            r = xb.Runner(kern, inp, b, m)
            nseq = xb.NB_K_HEADS * b
            recs = []
            for _ in range(a.iters):
                raw[off + xb.DBG_MERGE_OFF:off + xb.DBG_MERGE_OFF + 64 * nseq].zero_()
                flush()
                r.graph.replay()
                torch.cuda.synchronize()
                o1, o2 = off + xb.DBG_CTA_OFF, off + xb.DBG_MERGE_OFF
                cta = raw[o1:o1 + 16 * nseq * m].view(torch.int64).view(nseq, m, 2).cpu()
                mg = raw[o2:o2 + 64 * nseq].view(torch.int64).view(nseq, 8).cpu().double()
                t0 = int(cta[:, :, 0].min())
                end = (cta[:, :, 1] - t0).double() / 1e3
                mend = (mg[:, 3] - t0) / 1e3
                rec = dict(last_arrival=float(end.max()), merge_end=float(mend.max()), pre_loop=float(mg[:, 0].mean()),
                           loop=float(mg[:, 1].mean()), tail=float(mg[:, 2].mean()))
                for k, name in ((4, "prologue"), (5, "stats_loads"), (6, "max_combine"), (7, "scales")):
                    rec[name] = float(mg[:, k].mean())
                recs.append(rec)
            med = {k: float(np.median([x[k] for x in recs])) for k in recs[0]}
            out[f"{spec}:{b}x{m}"] = med
            xb.log(f"{spec} B={b} m={m}: " + " ".join(f"{k}={v:.1f}" for k, v in med.items()))
    print(json.dumps({"timeline3": out}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["explore", "timeline3"], default="explore")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush", choices=["read", "write", "none"], default="read")
    p.add_argument("--flush-mb", type=int, default=384)
    p.add_argument("--base-commit", default=xb.BASE_COMMIT)
    p.add_argument("--variants", default="base")
    p.add_argument("--timing-variant", default="nmf_d4_c64")
    p.add_argument("--configs", default="1x48")
    a = p.parse_args()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    xb.ck(xb.cu.cuInit(0), "cuInit")
    if a.mode == "explore":
        xb.mode_explore(a)
    else:
        mode_timeline3(a)


if __name__ == "__main__":
    main()
