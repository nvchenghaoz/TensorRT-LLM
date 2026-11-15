"""Idea 73 stage 0 iteration: bitwise check vs main at a few M, then cold-L2 timing of every
prototype cfg vs main at sl_qkv M=300. Synthetic tensors, no model load."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_ext, ckpt_dims, gpu_receipt, load_main, preheat, stats,  # noqa: E402
                    time_cold)


def main():
    ckpt = sys.argv[1]
    cfgs = [int(c) for c in (sys.argv[2].split(",") if len(sys.argv) > 2 else "0,1,2,3,4".split(","))]
    Ms = [int(m) for m in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["300"])]
    shape_name = sys.argv[4] if len(sys.argv) > 4 else "sl_qkv"
    _, shapes = ckpt_dims(ckpt)
    N, K = shapes[shape_name]
    print("receipt", gpu_receipt(), flush=True)
    main_op, _ = load_main()
    ext = build_ext()
    torch.manual_seed(0)
    ncopy = 4
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    print(f"{shape_name} N={N} K={K}; {ncopy} weight copies = {ncopy * N * K * 2 / 2**20:.0f} MB",
          flush=True)
    res = {}
    # correctness first
    for M in sorted(set(Ms + [288, 300, 301, 319, 320, 321, 1000])):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = main_op(x, ws[0])
        for c in cfgs:
            y = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
            ext.seqk_gemm(x, ws[0], y, c)
            torch.cuda.synchronize()
            mism = int((y.view(torch.int16) != ref.view(torch.int16)).sum())
            res.setdefault("bitwise", {})[f"M{M}_cfg{c}"] = mism
            if mism:
                idx = (y.view(torch.int16) != ref.view(torch.int16)).nonzero()[0].tolist()
                print(f"MISMATCH M={M} cfg={c} n={mism} first={idx} cand={y[idx[0], idx[1]].item()} "
                      f"main={ref[idx[0], idx[1]].item()}", flush=True)
    print("bitwise", res["bitwise"], flush=True)
    preheat()
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ys = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        arms = {"main": [lambda w=w: main_op(x, w) for w in ws]}
        for c in cfgs:
            arms[f"cfg{c}"] = [lambda w=w, c=c: ext.seqk_gemm(x, w, ys, c) for w in ws]
        for rep in range(2):
            order = list(arms) if rep == 0 else list(reversed(list(arms)))
            for name in order:
                ts = time_cold(arms[name], warm=8, iters=40)
                res.setdefault("t", {}).setdefault(f"M{M}", {}).setdefault(name, []).extend(ts)
        for name in arms:
            st = stats(res["t"][f"M{M}"][name])
            res["t"][f"M{M}"][name] = st
            tflops = 2 * M * N * K / (st["median"] * 1e-3) / 1e12
            print(f"M={M} {name}: median {st['median']*1e3:.1f} us p10 {st['p10']*1e3:.1f} "
                  f"p90 {st['p90']*1e3:.1f} -> {tflops:.0f} TFLOPS (useful)", flush=True)
    print("receipt_end", gpu_receipt(), flush=True)
    ok = all(v == 0 for v in res["bitwise"].values())
    print(json.dumps({"passed": ok, "bitwise": res["bitwise"], "timing": res["t"]}))


if __name__ == "__main__":
    main()
