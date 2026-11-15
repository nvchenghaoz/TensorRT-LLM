"""STEP 0 for idea 64: are FlashInfer SM120 DP tactics int16-equal to main's
trtllm::nvfp4_gemm on down_proj shapes, and does a wave-aligned M split beat
main cold-L2?  Synthetic tensors, no model load."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (fi_gemm, fi_module, int16_diff, kernel_names, log,  # noqa
                    make_act, make_weight, read_dims, receipts, time_fn)

import tensorrt_llm  # noqa
from tensorrt_llm._torch.autotuner import autotune  # noqa

N, K, _ = read_dims()
log(f"down_proj: N={N} K={K}")
dev = torch.device("cuda")
NW = 5
ws_list = []
for i in range(NW):
    w_fp4, w_sf, wgs = make_weight(N, K, 100 + i)
    ws_list.append((w_fp4, w_sf, wgs))
log(f"weights: fp4 {tuple(ws_list[0][0].shape)} sf {tuple(ws_list[0][1].shape)} x{NW}")
fi = fi_module()

MS_BIT = [int(x) for x in os.environ.get("MS_BIT", "4096,4800,7936,8000,8192").split(",")]
MS_TIME = [int(x) for x in os.environ.get("MS_TIME", "4096,4352,4800,7040,7936,8000,8100,8192").split(",")]
MAXM = max(MS_BIT + MS_TIME)
wsz = max(fi.fp4_gemm_workspace_size(m, N, K) for m in sorted(set(MS_BIT + MS_TIME + [128, 256, 1024])))
log(f"FI workspace bytes max = {wsz}")
for m in [64, 128, 512, 4096, 8192]:
    log(f"  ws({m}) = {fi.fp4_gemm_workspace_size(m, N, K)}")
ws = torch.empty(max(wsz, 1), dtype=torch.uint8, device=dev)


def alpha_for(ags, wgs):
    return (1.0 / (ags * wgs)).to(torch.float32)


def main_gemm(a_fp4, a_sf, w, alpha):
    return torch.ops.trtllm.nvfp4_gemm(a_fp4, w[0], a_sf, w[1], alpha, torch.bfloat16)


# Tune main like the engine warmup does (one call at max M tunes every bucket).
a_fp4, a_sf, ags = make_act(8192, K, 7)
with autotune():
    main_gemm(a_fp4, a_sf, ws_list[0], alpha_for(ags, ws_list[0][2]))
torch.cuda.synchronize()
log("main tuned")

DP = [t for t in range(fi.fp4_gemm_tactic_num()) if t % 4 in (0, 1)]
results = {"bit": {}, "time": {}}


def sf_rows_offset(m1, k):
    return m1 * (((k // 16) + 3) // 4 * 4)


def split_call(a_fp4, a_sf, w, alpha, out, m1, t1, t2):
    m = a_fp4.shape[0]
    off = sf_rows_offset(m1, K)
    a_sf = a_sf.view(-1)
    fi_gemm(a_fp4[:m1], w[0], a_sf[:off], w[1], alpha, out[:m1], ws, t1)
    if m > m1:
        fi_gemm(a_fp4[m1:], w[0], a_sf[off:], w[1], alpha, out[m1:], ws, t2)
    return out


# ---------------- bitwise ----------------
for m in MS_BIT:
    for kind, seed in (("randn", 1), ("heavy", 2)):
        a_fp4, a_sf, ags = make_act(m, K, seed, kind)
        w = ws_list[0]
        alpha = alpha_for(ags, w[2])
        ref = main_gemm(a_fp4, a_sf, w, alpha)
        names = kernel_names(lambda: main_gemm(a_fp4, a_sf, w, alpha))
        row = {"main_kernels": names}
        out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        for t in DP:
            out.fill_(float("nan"))
            try:
                fi_gemm(a_fp4, w[0], a_sf, w[1], alpha, out, ws, t)
                torch.cuda.synchronize()
                nd, first = int16_diff(out, ref)
            except Exception as e:  # noqa
                nd, first = -1, str(e)[:120]
            row[f"FI{t}"] = nd if first is None else [nd, first]
        # one representative split
        m1 = (m // 256) * 256 - 256 if m >= 1024 else m
        out.view(torch.int16).fill_(0x5A5A)
        split_call(a_fp4, a_sf, w, alpha, out, m1, 25, 21)
        torch.cuda.synchronize()
        row[f"split{m1}_25_21"] = int16_diff(out, ref)
        results["bit"][f"{m}_{kind}"] = row
        log(f"BIT M={m} {kind}: main={names[-1:]} " +
            " ".join(f"{k}={v if not isinstance(v, list) else v[0]}"
                     for k, v in row.items() if k != "main_kernels"))
        del a_fp4, a_sf, ref, out
        torch.cuda.empty_cache()

# ---------------- timing (cold weights: rotate 5 copies) ----------------
receipts(" start-timing")
T1S = [int(x) for x in os.environ.get("T1S", "24,25,20,21,28,29").split(",")]
T2S = [int(x) for x in os.environ.get("T2S", "1,5,9,13,17,21,25,0,4,8,12,16,20,24").split(",")]
ITERS = int(os.environ.get("ITERS", "20"))
for m in MS_TIME:
    a_fp4, a_sf, ags = make_act(m, K, 1)
    alphas = [alpha_for(ags, w[2]) for w in ws_list]
    out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
    r = {}
    r["main"] = time_fn(lambda i: main_gemm(a_fp4, a_sf, ws_list[i % NW], alphas[i % NW]), iters=ITERS)
    for t in DP:
        try:
            r[f"FI{t}"] = time_fn(lambda i: fi_gemm(a_fp4, ws_list[i % NW][0], a_sf, ws_list[i % NW][1],
                                                     alphas[i % NW], out, ws, t), iters=ITERS)
        except Exception as e:  # noqa
            r[f"FI{t}"] = str(e)[:80]
    best = None
    m1s = sorted({x for x in range(max(128, m - 1280), m, 128)})
    for m1 in m1s:
        for t1 in T1S:
            # main piece alone first; prune tails on the 3 best main pieces later
            pass
    combos = []
    for m1 in m1s:
        for t1 in T1S:
            st = time_fn(lambda i: fi_gemm(a_fp4[:m1], ws_list[i % NW][0], a_sf.view(-1)[:sf_rows_offset(m1, K)],
                                            ws_list[i % NW][1], alphas[i % NW], out[:m1], ws, t1),
                         iters=10, warmup=3)
            combos.append((st["med"], m1, t1))
    combos.sort()
    top = combos[:6]
    r["main_piece_top"] = [list(c) for c in top]
    for _, m1, t1 in top:
        for t2 in T2S:
            try:
                st = time_fn(lambda i: split_call(a_fp4, a_sf, ws_list[i % NW], alphas[i % NW], out, m1, t1, t2),
                             iters=ITERS)
            except Exception as e:  # noqa
                continue
            key = f"split_{m1}_{t1}_{t2}"
            r[key] = st
            if best is None or st["med"] < best[0]:
                best = (st["med"], key)
    r["best_split"] = best
    fi_best = min(((v["med"], k) for k, v in r.items() if k.startswith("FI") and isinstance(v, dict)))
    r["best_fi_single"] = fi_best
    results["time"][str(m)] = r
    log(f"TIME M={m}: main {r['main']['med']:.1f} (p10 {r['main']['p10']:.1f} p90 {r['main']['p90']:.1f}) | "
        f"best FI single {fi_best[1]} {fi_best[0]:.1f} | best split {best[1]} {best[0]:.1f} | "
        f"main-piece top {top[:3]}")
    del a_fp4, a_sf, out
    torch.cuda.empty_cache()
receipts(" end-timing")

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
with open(os.path.join(art, "step0.json"), "w") as f:
    json.dump(results, f, indent=1, default=str)
summary = {m: dict(main=round(v["main"]["med"], 1), split=v["best_split"], fi=v["best_fi_single"])
           for m, v in results["time"].items()}
print(json.dumps({"summary": summary}))
