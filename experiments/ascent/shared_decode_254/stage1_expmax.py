"""Idea 254: E[max_r] x 40 (ms/step) for MAIN / SHARD (built) / SKEW (built kernel) from the
stage1_gpu.py arm table, beside fact 2202's predictions. CPU only (numpy).

Routing generators, calibration and EP4 local counts are stage0_expmax.py's (fact 2202:
REAL calibrated to fact 822, w 0.85, alpha 0.014).
Per layer, rank cost at its D_r (linear interpolation over measured D):
  MAIN   B(D_r) on every rank
  SHARD  ranks 0,1 (part 0, blocks [0,5)) F5(D_r); ranks 2,3 (part 1, [5,9)) F4(D_r)
  SKEW   pairs {0,2}, {1,3}: the rank with fewer local experts F9(D), the other A(D);
         a tie splits 5/4 (F5 / F4)
Usage: stage1_expmax.py <stage1_ep4.json>
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared_contention_196"))
import stage0_expmax as e0  # noqa: E402

PRED_2202 = {"SHARD": {"REAL_M1": -0.2164, "UNIFORM_M1": -0.2162, "REAL_M4": -0.2112,
                       "UNIFORM_M4": -0.1969},
             "SKEW": {"REAL_M1": -0.4228, "UNIFORM_M1": -0.4232, "REAL_M4": -0.4761,
                      "UNIFORM_M4": -0.4938},
             "MAIN_REAL": {"M1": 2.41, "M2": 3.04, "M4": 3.96}}
ARMS = ("A", "B", "F5", "F4", "F9")


def designs(dr, f):
    out = {"MAIN": f["B"](dr).max(1)}
    sh = np.empty(dr.shape, dtype=float)
    sh[:, 0:2] = f["F5"](dr[:, 0:2])
    sh[:, 2:4] = f["F4"](dr[:, 2:4])
    out["SHARD"] = sh.max(1)
    sk = np.empty(dr.shape, dtype=float)
    for lo, hi in ((0, 2), (1, 3)):
        a, b = dr[:, lo], dr[:, hi]
        sk[:, lo] = np.where(a < b, f["F9"](a), np.where(a > b, f["A"](a), f["F5"](a)))
        sk[:, hi] = np.where(b < a, f["F9"](b), np.where(b > a, f["A"](b), f["F4"](b)))
    out["SKEW"] = sk.max(1)
    return out


def main():
    rng = np.random.default_rng(141)
    cal = e0.calibrate(rng)
    with open(sys.argv[1]) as fh:
        res = json.load(fh)
    table, nb = res["table"], res.get("NB", 4)
    report = {"calibration": cal}
    for gen in ("UNIFORM", "REAL"):
        for m in (1, 2, 4):
            if not any(r["M"] == m for r in table):
                continue
            dr = e0.local_counts(e0.draw(gen, m, rng), 4)
            fns, ds = e0.table_fns(table, m, ARMS)
            d = designs(dr, fns)
            row = {"clamp_above_rate": round(float((dr > ds.max()).mean()), 4)}
            for k, v in d.items():
                ms = v * e0.NL / 1000.0
                row[k] = {"ms_step": round(float(ms.mean()), 4)}
                if k != "MAIN":
                    diff = (v - d["MAIN"]) * e0.NL / 1000.0
                    boots = []
                    for _ in range(200):
                        f2, _ = e0.table_fns(table, m, ARMS, perturb=(rng, nb))
                        d2 = designs(dr, f2)
                        boots.append(float(((d2[k] - d2["MAIN"]) * e0.NL / 1000.0).mean()))
                    se = float(np.sqrt(diff.var(ddof=1) / len(diff) + np.var(boots, ddof=1)))
                    row[k]["minus_main_ms"] = round(float(diff.mean()), 4)
                    row[k]["minus_main_se"] = round(se, 4)
                    row[k]["pred_2202"] = PRED_2202[k].get(f"{gen}_M{m}")
            report[f"{gen}_M{m}"] = row
            print(f"{gen} M{m}: {json.dumps(row)}", flush=True)
    od = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    with open(os.path.join(od, "stage1_expmax.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: {d: v[d].get("minus_main_ms") for d in ("SHARD", "SKEW")}
                      for k, v in report.items() if k != "calibration"}))


if __name__ == "__main__":
    main()
