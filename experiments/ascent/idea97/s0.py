"""Idea 97 S0: kernel A (QK^T + per-tile row max, Triton) vs main's bmm at T=65536, c=512.

usage: python s0.py [sweep|time]
  sweep: compile+time candidate CFG_A configs (cold L2, AB alternation with bmm), then for the
         best config: S vs bmm (max abs/rel), S vs fp64 on sampled rows within the K bound
         (|S - S64| <= 512 * 2^-24 * sum_k |q_k k_k|, same bound checked for bmm), and the
         row max from the tile maxes bitwise vs S.amax(-1) of kernel A's own S.
"""

import json
import os
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

MODE = sys.argv[1] if len(sys.argv) > 1 else "sweep"
T = 65536
C = 512
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


FLUSH = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        FLUSH.fill_(1.0)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    n = len(ts)
    return {"median_ms": round(ts[n // 2], 3), "min_ms": round(ts[0], 3),
            "spread": round((ts[-1] - ts[0]) / ts[n // 2], 4)}


def main():
    g = torch.Generator(device="cuda").manual_seed(0)
    q = (torch.randn(C, T, device="cuda", generator=g) * 1.5).to(torch.bfloat16)
    k = (torch.randn(C, T, device="cuda", generator=g) * 1.5).to(torch.bfloat16)
    if os.environ.get("LAYOUT") == "cl":  # served channels_last decoder: [C, T] strides (1, C)
        q = q.t().contiguous().t()
        k = k.t().contiguous().t()
    log("strides", q.stride(), k.stride())
    s_out = torch.empty((T, T), device="cuda", dtype=torch.float32)

    def bmm():
        torch.bmm(q.t().unsqueeze(0), k.unsqueeze(0), out_dtype=torch.float32)

    log("bmm warm")
    bmm()
    torch.cuda.synchronize()
    res = {"bmm": timed(bmm, 5)}
    log("bmm", res["bmm"])
    cfgs = [
        dict(BM=128, BN=256, BK=32, GROUP_M=8, num_warps=8, num_stages=3),
        dict(BM=128, BN=256, BK=32, GROUP_M=8, num_warps=8, num_stages=4),
        dict(BM=128, BN=256, BK=32, GROUP_M=4, num_warps=8, num_stages=3),
        dict(BM=128, BN=256, BK=32, GROUP_M=16, num_warps=8, num_stages=3),
        dict(BM=256, BN=128, BK=32, GROUP_M=8, num_warps=8, num_stages=4),
        dict(BM=128, BN=256, BK=64, GROUP_M=8, num_warps=8, num_stages=1),
        dict(BM=256, BN=256, BK=32, GROUP_M=8, num_warps=16, num_stages=2),
        dict(BM=128, BN=128, BK=64, GROUP_M=8, num_warps=8, num_stages=3),
        dict(BM=128, BN=128, BK=64, GROUP_M=8, num_warps=8, num_stages=2),
        dict(BM=128, BN=128, BK=32, GROUP_M=8, num_warps=8, num_stages=4),
        dict(BM=128, BN=128, BK=64, GROUP_M=8, num_warps=4, num_stages=3),
        dict(BM=128, BN=256, BK=64, GROUP_M=8, num_warps=8, num_stages=2),
        dict(BM=128, BN=256, BK=32, GROUP_M=8, num_warps=8, num_stages=3),
        dict(BM=256, BN=128, BK=64, GROUP_M=8, num_warps=8, num_stages=2),
        dict(BM=256, BN=128, BK=32, GROUP_M=8, num_warps=8, num_stages=3),
        dict(BM=128, BN=128, BK=64, GROUP_M=16, num_warps=8, num_stages=3),
        dict(BM=64, BN=256, BK=64, GROUP_M=8, num_warps=4, num_stages=3),
    ]
    if MODE == "time":
        cfgs = [dict(sa.CFG_A)]
    best = None
    sweep = []
    for cfg in cfgs:
        mp = torch.empty((T, T // cfg["BN"]), device="cuda", dtype=torch.float32)
        try:
            sa._launch_a(q, k, s_out, mp, cfg, triton_wrap=False)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            log("cfg", cfg, "FAILED", type(e).__name__, str(e)[:200])
            sweep.append({"cfg": cfg, "error": type(e).__name__})
            continue
        r = timed(lambda: sa._launch_a(q, k, s_out, mp, cfg, triton_wrap=False), 5)
        rb = timed(bmm, 3)
        log("cfg", cfg, r, "bmm", rb)
        sweep.append({"cfg": cfg, **r, "bmm_median_ms": rb["median_ms"]})
        if best is None or r["median_ms"] < best[1]["median_ms"]:
            best = (cfg, r)
        del mp
    res["sweep"] = sweep
    res["best"] = {"cfg": best[0], **best[1]}
    cfg = best[0]
    # numerics of the best config
    mp = torch.empty((T, T // cfg["BN"]), device="cuda", dtype=torch.float32)
    sa._launch_a(q, k, s_out, mp, cfg, triton_wrap=False)
    m_tiles = mp.amax(dim=1)
    m_own = s_out.amax(dim=1)
    res["m_bitwise_vs_own_S"] = bool(torch.equal(m_tiles.view(torch.int32), m_own.view(torch.int32)))
    s_b = torch.bmm(q.t().unsqueeze(0), k.unsqueeze(0), out_dtype=torch.float32)[0]
    d_abs, d_rel, n_eq = 0.0, 0.0, 0
    for i in range(0, T, 4096):
        x, y = s_out[i:i + 4096], s_b[i:i + 4096]
        dd = (x - y).abs()
        d_abs = max(d_abs, dd.max().item())
        n_eq += int((x == y).sum().item())
    res["S_vs_bmm"] = {"max_abs": d_abs, "bitwise_share": n_eq / (T * T)}
    rows = torch.arange(0, T, T // 256, device="cuda")
    qd = q.double()
    kd = k.double()
    s64 = qd[:, rows].t() @ kd  # [256, T]
    bound = 512 * 2.0**-24 * (qd[:, rows].abs().t() @ kd.abs())
    e_c = (s_out[rows].double() - s64).abs()
    e_b = (s_b[rows].double() - s64).abs()
    res["S_vs_fp64"] = {
        "cand_max_abs": e_c.max().item(), "bmm_max_abs": e_b.max().item(),
        "cand_max_ratio_to_bound": (e_c / bound).max().item(),
        "bmm_max_ratio_to_bound": (e_b / bound).max().item(),
        "cand_within_bound": bool((e_c <= bound).all().item()),
    }
    # repeat determinism
    s_first = s_out[rows].clone()
    mp_first = mp.clone()
    rep = True
    for _ in range(5):
        s_out.fill_(float("nan"))
        mp.fill_(float("nan"))
        sa._launch_a(q, k, s_out, mp, cfg, triton_wrap=False)
        rep &= bool(torch.equal(s_out[rows], s_first) and torch.equal(mp, mp_first))
    res["repeat_bitwise"] = rep
    res["passed"] = bool(res["m_bitwise_vs_own_S"] and res["S_vs_fp64"]["cand_within_bound"] and rep)
    log(json.dumps(res))
    print(json.dumps(res))


with torch.inference_mode():
    main()
