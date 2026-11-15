"""Sweep for idea 64 rule design: cold-weight timing of main vs FI single DP
tactics vs wave-aligned splits, 64-row M grid.  Synthetic tensors."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (fi_gemm, fi_module, log, make_act, make_weight,  # noqa
                    read_dims, receipts, time_fn)

import tensorrt_llm  # noqa
from tensorrt_llm._torch.autotuner import autotune  # noqa

N, K, _ = read_dims()
dev = torch.device("cuda")
NW = 5
W = [make_weight(N, K, 100 + i) for i in range(NW)]
fi = fi_module()
ws = torch.empty(fi.fp4_gemm_workspace_size(8192, N, K), dtype=torch.uint8, device=dev)
SFK = ((K // 16) + 3) // 4 * 4
MMIN = int(os.environ.get("MMIN", "4096"))
MMAX = int(os.environ.get("MMAX", "8192"))
STEP = int(os.environ.get("STEP", "64"))
DP = [t for t in range(32) if t % 4 in (0, 1)]

a_fp4, a_sf, ags = make_act(MMAX, K, 1)
a_sf = a_sf.view(-1)
alphas = [(1.0 / (ags * w[2])).float() for w in W]
out = torch.empty(MMAX, N, dtype=torch.bfloat16, device=dev)

with autotune():
    torch.ops.trtllm.nvfp4_gemm(a_fp4, W[0][0], a_sf, W[0][1], alphas[0], torch.bfloat16)
torch.cuda.synchronize()
log("main tuned")
receipts(" start")


def piece(i, r0, r1, t):
    w = W[i % NW]
    fi_gemm(a_fp4[r0:r1], w[0], a_sf[r0 * SFK:], w[1], alphas[i % NW], out[r0:r1], ws, t)


# 1) main-piece time T1(M1, t1), M1 multiple of 128
T1 = {}
for m1 in range(2048, MMAX + 1, 128):
    for t1 in (24, 25, 20, 21, 28, 29):
        T1[(m1, t1)] = time_fn(lambda i: piece(i, 0, m1, t1), iters=10, warmup=3)["med"]
log("T1 done: " + " ".join(f"{m1}:{min(T1[(m1,t)] for t in (24,25)):.0f}"
                           for m1 in range(2048, MMAX + 1, 512)))

# 2) tail time T2(R, t2), measured right after a full main piece on the same
#    weight copy (as in the split, the weights are then L2-warm)
T2 = {}
for r in range(64, 1281, 64):
    for t2 in DP:
        def fn(i, r=r, t2=t2):
            piece(i, MMAX - r, MMAX, t2)

        def pre(i):
            piece(i, 0, 4096, 25)
        try:
            T2[(r, t2)] = time_fn(fn, iters=8, warmup=2, pre=pre)["med"]
        except Exception as e:  # noqa
            pass
best_t2 = {}
for r in range(64, 1281, 64):
    c = sorted((v, t) for (rr, t), v in T2.items() if rr == r)
    best_t2[r] = c[:3]
log("T2 best: " + " ".join(f"{r}:{best_t2[r][0][1]}@{best_t2[r][0][0]:.0f}" for r in best_t2))

# 3) per-M verification
rows = []
for m in range(MMIN, MMAX + 1, STEP):
    cand = []
    for (m1, t1), v in T1.items():
        r = m - m1
        if r == 0:
            cand.append((v, m1, t1, None))
        elif 0 < r <= 1280:
            rr = ((r + 63) // 64) * 64
            for _, t2 in best_t2[rr][:1]:
                cand.append((v + T2[(rr, t2)], m1, t1, t2))
    cand.sort()
    cand = cand[:4]
    res = {}
    res["main"] = time_fn(lambda i: torch.ops.trtllm.nvfp4_gemm(
        a_fp4[:m], W[i % NW][0], a_sf[:((m + 127) // 128) * 128 * SFK], W[i % NW][1], alphas[i % NW], torch.bfloat16), iters=20)
    for t in (24, 25):
        res[f"FI{t}"] = time_fn(lambda i: piece(i, 0, m, t), iters=20)
    for _, m1, t1, t2 in cand:
        def fn(i, m1=m1, t1=t1, t2=t2):
            piece(i, 0, m1, t1)
            if t2 is not None:
                piece(i, m1, m, t2)
        res[f"S{m1}_{t1}_{t2}"] = time_fn(fn, iters=20)
    bk = min(res, key=lambda k: res[k]["med"])
    rows.append(dict(m=m, res=res, best=bk))
    log(f"M={m}: main {res['main']['med']:.0f} FI25 {res['FI25']['med']:.0f} FI24 {res['FI24']['med']:.0f} "
        f"best {bk} {res[bk]['med']:.0f} ({100*(res[bk]['med']/res['main']['med']-1):+.1f}%) "
        f"| cands " + " ".join(f"{k}:{v['med']:.0f}" for k, v in res.items() if k.startswith('S')))
receipts(" end")
art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
json.dump(dict(T1={f"{a}_{b}": v for (a, b), v in T1.items()},
               T2={f"{a}_{b}": v for (a, b), v in T2.items()}, rows=rows),
          open(os.path.join(art, "sweep.json"), "w"), indent=1)
print(json.dumps({"done": len(rows)}))
