"""Idea 196 STAGE 0 (CPU part): expected per-step max over the 4 TP4 ranks of the per-layer
MoE critical path, for the designs MAIN / SHARD / SKEW, from the measured A/B/C tables of
stage0_gpu.py (CPU only, numpy; no GPU needed).

Routing generators (same as analysis 141, fact 1575):
  UNIFORM  each token picks top-6 of 384 experts uniformly without replacement.
  REAL     per step popularity p = 0.5 * uniform + 0.5 * Dirichlet(alpha=2.0) over 384
           experts; each token draws top-6 without replacement from p.  Validated against
           fact 822's c4 per-rank histogram (M=4, EP4: 0 / 1-5 / >=6 local experts =
           1.5% / 82.8% / 15.7%); printed, and flagged if any bin is off by > 2 pp.
Layouts:
  EP4      rank r owns experts [96r, 96r + 96); D_r = distinct local experts hit.
  EP2xTP2  EP group of rank r is {r, r+2} (moe_ep_rank = tp_rank % 2), 192 experts each; the
           two ranks of a group see the same D.
Designs (per layer, per rank cost from the tables at that rank's D; linear interpolation
between measured D, clamped above the top measured D, clamp rate reported):
  MAIN   every rank B(D_r)
  SHARD  every rank C(D_r)
  SKEW   EP4 only: in holder pair {0,2} and {1,3} the rank with fewer local experts takes
         B(D), the heavier takes A(D); ties -> C on both.
E[max_r] x 40 layers in ms/step, sampling SE, and the SE of (design - MAIN) including table
measurement error (parametric bootstrap: each table entry ~ N(us, blk_std / sqrt(NB))).

Usage: stage0_expmax.py <stage0_ep4.json> [<stage0_ep2tp2.json>]   (env S0_STEPS=8192)
"""

import json
import os
import sys

import numpy as np

NE, TOPK, NL = 384, 6, 40
STEPS = int(os.environ.get("S0_STEPS", "8192"))
ITL = {1: 6.86, 2: 7.88, 4: 9.44}  # window 910 ITL ms, c1 / c2 / c4
TARGET_M4 = (1.5, 82.8, 15.7)


REAL_PARAMS = [0.5, 2.0]  # (Dirichlet weight w, alpha); replaced by calibrate()


def draw(gen, m, rng, steps=None, params=None):
    """(steps, m, 6) expert ids."""
    steps = steps or STEPS
    w, alpha = params or REAL_PARAMS
    out = np.empty((steps, m, TOPK), dtype=np.int64)
    for s in range(steps):
        if gen == "UNIFORM":
            for t in range(m):
                out[s, t] = rng.choice(NE, TOPK, replace=False)
        else:
            p = (1.0 - w) / NE + w * rng.dirichlet(np.full(NE, alpha))
            p = p / p.sum()
            for t in range(m):
                out[s, t] = rng.choice(NE, TOPK, replace=False, p=p)
    return out


def m4_hist(params, rng, steps=1500):
    dr = local_counts(draw("REAL", 4, rng, steps=steps, params=params), 4)
    h0 = 100.0 * (dr == 0).mean()
    h6 = 100.0 * (dr >= 6).mean()
    return (h0, 100.0 - h0 - h6, h6)


def calibrate(rng):
    """Fit REAL's (w, alpha) so its M=4 EP4 per-rank histogram matches fact 822."""
    orig = m4_hist((0.5, 2.0), rng)
    best = None
    for w in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
        for alpha in (0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5):
            h = m4_hist((w, alpha), rng, steps=600)
            off = max(abs(a - b) for a, b in zip(h, TARGET_M4))
            if best is None or off < best[0]:
                best = (off, w, alpha, h)
    # refine around the best point with more steps
    _, w0, a0, _ = best
    best = None
    for w in sorted({max(0.3, min(1.0, w0 + dw)) for dw in (-0.05, 0.0, 0.05)}):
        for alpha in sorted({a0 * f for f in (0.7, 0.85, 1.0, 1.2, 1.4)}):
            h = m4_hist((w, alpha), rng, steps=2000)
            off = max(abs(a - b) for a, b in zip(h, TARGET_M4))
            if best is None or off < best[0]:
                best = (off, w, alpha, h)
    print(f"REAL calibration: original (w 0.5, alpha 2.0) M4 hist "
          f"{tuple(round(x, 2) for x in orig)}; calibrated w {best[1]:.3f} alpha {best[2]:.4f} "
          f"hist {tuple(round(x, 2) for x in best[3])} (max off {best[0]:.2f} pp) vs fact 822 "
          f"{TARGET_M4}", flush=True)
    REAL_PARAMS[:] = [best[1], best[2]]
    return {"original_params": [0.5, 2.0], "original_hist": [round(x, 2) for x in orig],
            "params": [best[1], best[2]], "hist": [round(x, 2) for x in best[3]],
            "max_off_pp": round(best[0], 2)}


def local_counts(ids, ngroups):
    """(STEPS, ngroups) distinct experts hit per expert group of NE / ngroups experts."""
    per = NE // ngroups
    flat = ids.reshape(ids.shape[0], -1)
    out = np.zeros((ids.shape[0], ngroups), dtype=np.int64)
    for s in range(flat.shape[0]):
        u = np.unique(flat[s])
        out[s] = np.bincount(u // per, minlength=ngroups)
    return out


def table_fns(table, m, arms, perturb=None):
    """Per arm, an interpolator D -> us at token count m (rows with that M)."""
    rows = sorted((r for r in table if r["M"] == m), key=lambda r: r["nD"])
    ds = np.array([r["nD"] for r in rows], dtype=float)
    fns = {}
    for a in arms:
        us = np.array([r[a]["us"] for r in rows], dtype=float)
        if perturb is not None:
            se = np.array([r[a].get("blk_std", 0.0) for r in rows]) / np.sqrt(perturb[1])
            us = us + perturb[0].normal(0.0, 1.0, size=us.shape) * se
        fns[a] = (lambda d, us=us: np.interp(d, ds, us))
    return fns, ds


def designs(dr, fns, layout):
    A, B, C = fns["A"], fns["B"], fns["C"]
    out = {"MAIN": B(dr).max(1), "SHARD": C(dr).max(1)}
    if layout == "ep4":
        cost = np.empty(dr.shape, dtype=float)
        for lo, hi in ((0, 2), (1, 3)):
            a, b = dr[:, lo], dr[:, hi]
            cost[:, lo] = np.where(a < b, B(a), np.where(a > b, A(a), C(a)))
            cost[:, hi] = np.where(b < a, B(b), np.where(b > a, A(b), C(b)))
        out["SKEW"] = cost.max(1)
    return out


def analyse(res, layout, rng):
    table = res["table"]
    nb = res.get("NB", 6)
    ngroups = 4 if layout == "ep4" else 2
    report = {}
    for gen in ("UNIFORM", "REAL"):
        for m in (1, 2, 4):
            if not any(r["M"] == m for r in table):
                print(f"[{layout}] {gen} M{m}: no table rows at M={m}; skipped", flush=True)
                continue
            ids = draw(gen, m, rng)
            dr = local_counts(ids, ngroups)
            hist = None
            if layout == "ep4":
                h0 = 100.0 * (dr == 0).mean()
                h6 = 100.0 * (dr >= 6).mean()
                hist = (round(h0, 2), round(100.0 - h0 - h6, 2), round(h6, 2))
            fns, ds = table_fns(table, m, ("A", "B", "C"))
            clamp = float((dr > ds.max()).mean())
            below = float((dr < ds.min()).mean())
            d = designs(dr, fns, layout)
            row = {"hist_0_1to5_ge6_pct": hist, "clamp_above_rate": round(clamp, 4),
                   "clamp_below_rate": round(below, 4),
                   "Dmax_mean": round(float(dr.max(1).mean()), 3),
                   "D_mean": round(float(dr.mean()), 3)}
            for k, v in d.items():
                ms = v * NL / 1000.0
                row[k] = {"ms_step": round(float(ms.mean()), 4),
                          "se": round(float(ms.std(ddof=1) / np.sqrt(len(ms))), 4),
                          "pct_itl": round(100.0 * float(ms.mean()) / ITL[m], 2)}
                if k != "MAIN":
                    diff = (v - d["MAIN"]) * NL / 1000.0
                    boots = []
                    for _ in range(200):
                        f2, _ = table_fns(table, m, ("A", "B", "C"), perturb=(rng, nb))
                        d2 = designs(dr, f2, layout)
                        boots.append(float(((d2[k] - d2["MAIN"]) * NL / 1000.0).mean()))
                    se = float(np.sqrt(diff.var(ddof=1) / len(diff) + np.var(boots, ddof=1)))
                    row[k]["minus_main_ms"] = round(float(diff.mean()), 4)
                    row[k]["minus_main_se"] = round(se, 4)
                    row[k]["pays"] = bool(diff.mean() < -se)
            report[f"{gen}_M{m}"] = row
            print(f"[{layout}] {gen} M{m}: {json.dumps(row)}", flush=True)
            if gen == "REAL" and m == 4 and hist is not None:
                off = max(abs(a - b) for a, b in zip(hist, TARGET_M4))
                print(f"[{layout}] REAL M4 histogram {hist} vs fact 822 {TARGET_M4}: max off "
                      f"{off:.2f} pp -> {'OK (within 2 pp)' if off <= 2.0 else 'OFF (> 2 pp)'}",
                      flush=True)
                report["REAL_M4_hist_check"] = {"hist": [float(x) for x in hist],
                                                "target": TARGET_M4, "params": list(REAL_PARAMS),
                                                "max_off_pp": round(float(off), 2),
                                                "ok": bool(off <= 2.0)}
    return report


def verdict(rep):
    pays = {}
    for k in ("SHARD", "SKEW"):
        for m in (1, 4):
            for gen in ("UNIFORM", "REAL"):
                r = rep.get(f"{gen}_M{m}", {}).get(k)
                if r is not None:
                    pays.setdefault(k, []).append((gen, m, r["minus_main_ms"],
                                                   r["minus_main_se"], r["pays"]))
    winners = [k for k, v in pays.items()
               if all(p[4] for p in v if p[0] == "REAL") and any(p[4] for p in v)]
    line = "; ".join(f"{k}: " + ", ".join(f"{g} M{m} {d:+.4f}+-{s:.4f} ms" for g, m, d, s, _ in v)
                     for k, v in pays.items())
    if not winners:
        return f"EARLY EXIT (no design beats MAIN beyond its SE at M=1 and M=4): {line}", None
    best = min(winners, key=lambda k: np.mean([p[2] for p in pays[k]]))
    return f"BUILD {best}: {line}", best


def main():
    rng = np.random.default_rng(141)
    out = {"calibration": calibrate(rng)}
    for path in sys.argv[1:]:
        with open(path) as f:
            res = json.load(f)
        layout = res.get("shape", "ep4")
        out[layout] = analyse(res, layout, rng)
    v, best = verdict(out.get("ep4", next(iter(out.values()))))
    print("VERDICT:", v, flush=True)
    od = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    with open(os.path.join(od, "stage0_expmax.json"), "w") as f:
        json.dump({"report": out, "verdict": v, "build": best, "steps": STEPS}, f, indent=1)
    print(json.dumps({"verdict": v, "build": best}))


if __name__ == "__main__":
    main()
