"""Idea 279 Stage 0 (CPU): E[max_r] x 40 (ms/step) of the decode MoE critical path, SHARD-v1
(main) vs SHARD-v2, from s0_armf.py's table, at the layout the table was measured at.

Routing generators are shared_contention_196/stage0_expmax.py's: UNIFORM, and REAL with the
calibrated parameters of fact 2202 (w 0.85, alpha 0.014) unless EXPMAX_CALIBRATE=1.
Layouts (rank r of TP4):
  ep2tp2  expert group g = r % 2 (192 experts each, slot_start 192 g); shared part r // 2
          (part 0 = blocks [0,5) -> F5, part 1 = [5,9) -> F4).  Ranks 0/2 see group 0's D,
          ranks 1/3 group 1's.
  ep4     group g = r (96 experts); parts: ranks 0,1 -> F5, ranks 2,3 -> F4.
Per layer cost of rank r = arm(D_{g(r)}) by linear interpolation over measured D (clamped
above the top measured D; clamp rate reported).  Designs:
  ROUTED   A on every rank (context: no shared expert)
  SHARD-v1 B5 / B4 by part;  SHARD-v2 V5 / V4 by part.
Reported: ms/step and (SHARD-v2 - SHARD-v1) with SE (sampling + table bootstrap).
Usage: s0_expmax.py <s0_armf_<layout>.json>
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared_contention_196"))
import stage0_expmax as e0  # noqa: E402

PRE45 = {"REAL_M1": -0.201, "REAL_M2": -0.053, "REAL_M4": -0.053}  # fact 2503, EP4, pre-45


def rank_layout(layout):
    if layout == "ep2tp2":
        return 2, [r % 2 for r in range(4)], [r // 2 for r in range(4)]
    return 4, list(range(4)), [0, 0, 1, 1]


def designs(dg, f, layout):
    _ng, grp, part = rank_layout(layout)
    out = {"ROUTED": np.max(np.stack([f["A"](dg[:, g]) for g in grp], 1), 1)}
    for name, pre in (("SHARD-v1", "B"), ("SHARD-v2", "V")):
        cols = [f[f"{pre}{5 if p == 0 else 4}"](dg[:, g]) for g, p in zip(grp, part)]
        out[name] = np.max(np.stack(cols, 1), 1)
    return out


def main():
    rng = np.random.default_rng(279)
    if os.environ.get("EXPMAX_CALIBRATE", "0") == "1":
        cal = e0.calibrate(rng)
    else:
        e0.REAL_PARAMS[:] = [0.85, 0.014]
        cal = {"params": [0.85, 0.014], "source": "fact 2202 calibration"}
    with open(sys.argv[1]) as fh:
        res = json.load(fh)
    layout = res.get("layout", res.get("shape", "ep4"))
    ng = rank_layout(layout)[0]
    table, nb = res["table"], res.get("NB", 4)
    arms = ("A", "B4", "B5", "V4", "V5")
    report = {"calibration": cal, "layout": layout, "steps": e0.STEPS}
    for gen in ("UNIFORM", "REAL"):
        for m in (1, 2, 4):
            if not any(r["M"] == m for r in table):
                continue
            dg = e0.local_counts(e0.draw(gen, m, rng), ng)
            fns, ds = e0.table_fns(table, m, arms)
            d = designs(dg, fns, layout)
            row = {"layout": layout, "clamp_above_rate": round(float((dg > ds.max()).mean()), 4),
                   "clamp_below_rate": round(float((dg < ds.min()).mean()), 4),
                   "D_mean": round(float(dg.mean()), 3),
                   "Dmax_mean": round(float(dg.max(1).mean()), 3)}
            for k, v in d.items():
                ms = v * e0.NL / 1000.0
                row[k] = {"ms_step": round(float(ms.mean()), 4)}
            diff = (d["SHARD-v2"] - d["SHARD-v1"]) * e0.NL / 1000.0
            boots = []
            for _ in range(200):
                f2, _ = e0.table_fns(table, m, arms, perturb=(rng, nb))
                d2 = designs(dg, f2, layout)
                boots.append(float(((d2["SHARD-v2"] - d2["SHARD-v1"]) * e0.NL / 1000.0).mean()))
            se = float(np.sqrt(diff.var(ddof=1) / len(diff) + np.var(boots, ddof=1)))
            row["v2_minus_v1_ms"] = round(float(diff.mean()), 4)
            row["v2_minus_v1_se"] = round(se, 4)
            row["v2_below_v1_beyond_se"] = bool(diff.mean() < -se)
            row["pre45_ep4_fact2503"] = PRE45.get(f"{gen}_M{m}")
            report[f"{gen}_M{m}"] = row
            print(f"layout={layout} {gen} M{m}: {json.dumps(row)}", flush=True)
    passed = any(report.get(f"{g}_M{m}", {}).get("v2_below_v1_beyond_se", False)
                 for g in ("REAL",) for m in (1, 2, 4))
    report["direction_check_real"] = passed
    od = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    with open(os.path.join(od, f"s0_expmax_{layout}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"layout": layout, "direction_check_real": passed,
                      **{k: [v["v2_minus_v1_ms"], v["v2_minus_v1_se"]]
                         for k, v in report.items() if isinstance(v, dict) and
                         "v2_minus_v1_ms" in v}}))


if __name__ == "__main__":
    main()
