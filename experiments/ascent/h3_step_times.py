"""Per-forward dense vs cached times from VisualGen "Step k/49 | X s" log lines (no model run).

For each log: splits the lines into requests (a "Step 1/49" starts one), tags each forward as
cached or dense from the static Cache-DiT mask (--cached), and prints per-request mean/median dense
and cached forward time and the denoise total.

Usage: python h3_step_times.py --cached 14,15,17,... <label>=<log path> [...]
"""

import argparse
import json
import re
import statistics

STEP = re.compile(r"Step (\d+)/(\d+) \| ([\d.]+)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", default="")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    cached = {int(x) for x in a.cached.split(",") if x}
    out = {}
    for item in a.logs:
        label, path = item.split("=", 1)
        reqs, cur = [], None
        with open(path, errors="replace") as fh:
            for ln in fh:
                m = STEP.search(ln)
                if not m:
                    continue
                k, n, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if n != 49:
                    continue
                if k == 1:
                    cur = []
                    reqs.append(cur)
                if cur is not None:
                    cur.append((k - 1, t))
        rows = []
        for r in reqs:
            dense = [t for i, t in r if i not in cached]
            cach = [t for i, t in r if i in cached]
            rows.append({"forwards": len(r), "dense_n": len(dense),
                         "dense_mean_s": round(statistics.fmean(dense), 3) if dense else None,
                         "dense_median_s": round(statistics.median(dense), 3) if dense else None,
                         "cached_n": len(cach),
                         "cached_mean_s": round(statistics.fmean(cach), 3) if cach else None,
                         "cached_median_s": round(statistics.median(cach), 3) if cach else None,
                         "denoise_total_s": round(sum(t for _, t in r), 2)})
        out[label] = rows
        print(f"[steps] {label}: {json.dumps(rows)}", flush=True)
    print(json.dumps({"passed": True, "step_times": out}), flush=True)


if __name__ == "__main__":
    main()
