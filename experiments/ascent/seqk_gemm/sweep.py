"""Idea 73: M sweep of the production sequential-K kernel (all configs) vs main's dispatch for the
Gemma4 qkv shapes. Cold L2 (4 weight copies), CUDA events, 5 warmup + 25 timed per arm, median
with p10/p90, plus a bitwise (int16) check vs main at every swept M. Synthetic tensors, no model.

usage: sweep.py <ckpt> [shapes=sl_qkv,gl_qkv] [Ms=default]"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_prod_ext, ckpt_dims, gpu_receipt, load_main, preheat, stats,  # noqa: E402
                    time_cold)

CFGS = (0, 1, 2, 3)


def default_ms():
    ms = set(range(288, 1281, 16)) | {300, 600, 900, 1000, 1200}
    return sorted(ms)


def main():
    ckpt = sys.argv[1]
    shape_names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sl_qkv", "gl_qkv"]
    Ms = [int(m) for m in sys.argv[3].split(",")] if len(sys.argv) > 3 else default_ms()
    _, shapes = ckpt_dims(ckpt)
    print("receipt", gpu_receipt(), flush=True)
    main_op, _ = load_main()
    ext = build_prod_ext()
    for c in CFGS:
        print("cfg", c, "tile tokens x weights", ext.tile(c), flush=True)
    preheat()
    out = {}
    for sn in shape_names:
        N, K = shapes[sn]
        torch.manual_seed(0)
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(4)]
        rows = {}
        for M in Ms:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ref = main_op(x, ws[0])
            rec = {"mism": {}}
            for c in CFGS:
                y = ext.seqk(x, ws[0], c)
                rec["mism"][c] = int((y.view(torch.int16) != ref.view(torch.int16)).sum())
            t = {"main": stats(time_cold([lambda w=w: main_op(x, w) for w in ws], warm=5, iters=25))}
            for c in CFGS:
                t[c] = stats(time_cold([lambda w=w, c=c: ext.seqk(x, w, c) for w in ws], warm=5,
                                       iters=25))
            rec["t"] = {str(k): v["median"] for k, v in t.items()}
            rec["noise"] = {str(k): v["noise"] for k, v in t.items()}
            best = min(CFGS, key=lambda c: t[c]["median"])
            rec["best"] = best
            rec["gain"] = 1 - t[best]["median"] / t["main"]["median"]
            rows[M] = rec
            print(f"{sn} M={M}: main {t['main']['median']*1e3:.1f} us | " + " ".join(
                f"c{c} {t[c]['median']*1e3:.1f}" for c in CFGS) +
                f" | best c{best} gain {rec['gain']*100:+.1f}% | mism {rec['mism']}", flush=True)
        out[sn] = rows
        del ws
        torch.cuda.empty_cache()
    print("receipt_end", gpu_receipt(), flush=True)
    ok = all(v == 0 for s in out.values() for r in s.values() for v in r["mism"].values())
    print(json.dumps({"passed": ok, "sweep": out}))


if __name__ == "__main__":
    main()
