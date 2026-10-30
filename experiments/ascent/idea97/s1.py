"""Idea 97 S1: kernel B (PV with in-prologue softmax, e4m3) vs main's row_softmax_e4m3 + _scaled_mm.

usage: python s1.py [sweep|time|check]
S comes from the real producer (kernel A, bitwise equal to main's bmm at this shape, s0 job 757).

DECLARED TOLERANCE (stated before timing, from the brief):
  scale_a: max rel vs main <= 1e-5 (row-sum order only).
  debug pq (kernel B's in-register P, stored when DEBUG): <= 1 e4m3 ulp vs main's pq on >= 99.99%
    (expected bitwise: same m, same expression, same S).
  O^T: max abs error vs an fp64 reference of main's own pq/vq/scales on 256 sampled rows is
    <= 2x main's _scaled_mm error vs the same reference (fp32 accumulation order only).
"""

import json
import random
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import row_softmax_triton as rs
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

MODE = sys.argv[1] if len(sys.argv) > 1 else "sweep"
T = 65536
C = 512
SCALE = C**-0.5
FP8 = torch.float8_e4m3fn
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
    return ts


def st(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 3), "p10_ms": round(p10, 3), "p90_ms": round(p90, 3),
            "noise_floor": round((p90 - p10) / med, 4), "n": n}


def _ord8(t):
    i = t.view(torch.int8).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7F), i)


def main():
    g = torch.Generator(device="cuda").manual_seed(0)
    q = (torch.randn(C, T, device="cuda", generator=g) * 1.5).to(torch.bfloat16)
    k = (torch.randn(C, T, device="cuda", generator=g) * 1.5).to(torch.bfloat16)
    v = torch.randn(C, T, device="cuda", generator=g).to(torch.bfloat16)
    vf = v.float()
    s_v = (vf.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)
    vq = (vf / s_v).clamp(-448.0, 448.0).to(FP8).contiguous()
    del vf
    svf = s_v.reshape(C).contiguous()
    s, mp = sa.qk_tilemax(q, k)
    torch.cuda.synchronize()
    log("S ready")

    def main_b():
        pq, scale_a = rs.row_softmax_e4m3(s, SCALE)
        return torch._scaled_mm(pq, vq.t(), scale_a, s_v.reshape(1, C), out_dtype=torch.bfloat16)

    def main_sm():
        return rs.row_softmax_e4m3(s, SCALE)

    pq_m, sa_m = rs.row_softmax_e4m3(s, SCALE)

    def main_mm():
        return torch._scaled_mm(pq_m, vq.t(), sa_m, s_v.reshape(1, C), out_dtype=torch.bfloat16)

    res = {}
    if MODE in ("sweep", "time"):
        cfgs = [dict(sa.CFG_B)]
        if MODE == "sweep":
            cfgs = [
                dict(BM=64, BN=256, BK=64, num_warps=8, num_stages=2, N_FAST=True),
                dict(BM=64, BN=256, BK=64, num_warps=8, num_stages=3, N_FAST=True),
                dict(BM=64, BN=512, BK=64, num_warps=8, num_stages=2, N_FAST=True),
                dict(BM=64, BN=512, BK=32, num_warps=8, num_stages=3, N_FAST=True),
                dict(BM=128, BN=256, BK=32, num_warps=8, num_stages=3, N_FAST=True),
                dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=2, N_FAST=True),
                dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3, N_FAST=True),
                dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=3, N_FAST=True),
                dict(BM=64, BN=256, BK=32, num_warps=8, num_stages=4, N_FAST=True),
                dict(BM=128, BN=512, BK=32, num_warps=16, num_stages=2, N_FAST=True),
            ]
        main_b()
        torch.cuda.synchronize()
        rt = {"main_softmax": st(timed(main_sm, 5)), "main_scaled_mm": st(timed(main_mm, 5)),
              "main_B": st(timed(main_b, 5))}
        log("main", rt)
        best = None
        sweep = []
        for cfg in cfgs:
            try:
                sa.pv_prologue(s, mp, vq, svf, SCALE, cfg)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                log("cfg", cfg, "FAILED", type(e).__name__, str(e)[:300])
                sweep.append({"cfg": cfg, "error": type(e).__name__})
                continue
            r = st(timed(lambda: sa.pv_prologue(s, mp, vq, svf, SCALE, cfg), 5))
            log("cfg", cfg, r)
            sweep.append({"cfg": cfg, **r})
            if best is None or r["median_ms"] < best[1]["median_ms"]:
                best = (cfg, r)
        res["main"] = rt
        res["sweep"] = sweep
        res["best"] = {"cfg": best[0], **best[1]}
        if MODE == "time":
            # paired A/B alternation: main (softmax+scaled_mm) vs kernel B, seeded order
            rnd = random.Random(0)
            ts = {"main": [], "cand": []}
            fns = {"main": main_b, "cand": lambda: sa.pv_prologue(s, mp, vq, svf, SCALE)}
            for _ in range(20):
                order = ["main", "cand"]
                rnd.shuffle(order)
                for n in order:
                    ts[n] += timed(fns[n], 1)
            res["paired"] = {n: st(v) for n, v in ts.items()}
            log("paired", res["paired"])
    # numerics (best or tree cfg)
    cfg = res.get("best", {}).get("cfg", dict(sa.CFG_B)) if MODE == "sweep" else dict(sa.CFG_B)
    o_m = main_mm()
    o_c, sa_c, pq_c = sa.pv_prologue(s, mp, vq, svf, SCALE, cfg, debug=True)
    torch.cuda.synchronize()
    nd, mu = 0, 0
    for i in range(0, T, 4096):
        u = (_ord8(pq_c[i:i + 4096]) - _ord8(pq_m[i:i + 4096])).abs()
        nd += int((u > 0).sum().item())
        mu = max(mu, int(u.max().item()))
    del pq_c
    res["pq_vs_main"] = {"share_differ": nd / (T * T), "max_ulp": mu}
    rel = ((sa_c - sa_m[:, 0]).abs() / sa_m[:, 0]).max().item()
    res["scale_a_vs_main_max_rel"] = rel
    rows = torch.arange(0, T, T // 256, device="cuda")
    pd = pq_m[rows].double()  # [256, T]
    o64 = (pd @ vq.double().t()) * sa_m[rows].double() * s_v.reshape(1, C).double()
    e_m = (o_m[rows].double() - o64).abs().max().item()
    e_c = (o_c[rows].double() - o64).abs().max().item()
    dd = (o_c.float() - o_m.float()).abs()
    res["o"] = {"main_max_abs_vs_fp64": e_m, "cand_max_abs_vs_fp64": e_c,
                "cand_vs_main_max_abs": dd.max().item(), "cand_vs_main_bitwise_share":
                (o_c == o_m).float().mean().item(), "max_abs_o": o_m.float().abs().max().item()}
    gates = {
        "pq": mu <= 1 and nd / (T * T) <= 1e-4,
        "scale_a": rel <= 1e-5,
        "o_vs_fp64": e_c <= 2 * e_m,
    }
    # repeat determinism (NaN-poisoned outputs each call)
    first = o_c.clone()
    rep = True
    for _ in range(5):
        o2, sa2, _ = sa.pv_prologue(s, mp, vq, svf, SCALE, cfg)
        rep &= bool(torch.equal(o2, first) and torch.equal(sa2, sa_c))
    gates["repeat_bitwise"] = rep
    res["gates"] = gates
    res["passed"] = bool(all(gates.values()))
    log(json.dumps(res))
    print(json.dumps(res))


with torch.inference_mode():
    main()
