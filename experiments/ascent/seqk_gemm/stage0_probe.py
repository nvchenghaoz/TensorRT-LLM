"""Idea 73 stage 0 diagnostics (not evidence): K-scan intercept and upper-bound probes for the
sequential-K kernel at sl_qkv M=300. cfg 20 = TMA+ldmatrix only (no MMA), cfg 21 = MMA only
(no TMA, no waits). Cold L2 (4 weight copies)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_ext, ckpt_dims, gpu_receipt, load_main, preheat, stats, time_cold  # noqa


def main():
    ckpt = sys.argv[1]
    cfgs = [int(c) for c in sys.argv[2].split(",")]
    _, shapes = ckpt_dims(ckpt)
    N, K0 = shapes["sl_qkv"]
    M = 300
    print("receipt", gpu_receipt(), flush=True)
    main_op, _ = load_main()
    ext = build_ext()
    preheat()
    out = {}
    for K in (K0, K0 // 2, K0 // 4):
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(4)]
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        arms = {"main": [lambda w=w: main_op(x, w) for w in ws]}
        for c in cfgs:
            arms[f"cfg{c}"] = [lambda w=w, c=c: ext.seqk_gemm(x, w, y, c) for w in ws]
        res = {}
        for name, fl in arms.items():
            res[name] = stats(time_cold(fl, warm=8, iters=40))
            print(f"K={K} {name}: {res[name]['median']*1e3:.1f} us", flush=True)
        out[f"K{K}"] = res
        del ws
        torch.cuda.empty_cache()
    print("receipt_end", gpu_receipt(), flush=True)
    print(json.dumps({"passed": True, "probe": out}))


if __name__ == "__main__":
    main()
