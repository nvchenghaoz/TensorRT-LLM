"""Print per-launch registers, stack and local-memory (spill) metrics from an ncu kernels-raw.csv.

usage: python ncu_spill.py <kernels-raw.csv>
"""
import csv
import json
import sys

rows = list(csv.reader(open(sys.argv[1], newline="")))
hdr = rows[0]
want = [i for i, h in enumerate(hdr)
        if any(s in h for s in ("local", "stack", "registers", "Kernel Name", "Grid Size"))
        or h in ("Kernel Name", "Grid Size", "Block Size")]
print("columns:", [hdr[i] for i in want], flush=True)
out = []
for r in rows[1:]:
    if len(r) != len(hdr) or not r[0] or not r[0][0].isdigit():
        continue
    rec = {hdr[i]: r[i] for i in want}
    out.append(rec)
    print(json.dumps(rec), flush=True)
spill_cols = [h for h in (hdr[i] for i in want) if "spilling" in h or "local" in h]
bad = []
for rec in out:
    for h in spill_cols:
        v = rec.get(h, "").replace(",", "")
        try:
            if float(v) != 0.0:
                bad.append((h, v))
        except ValueError:
            pass
print(json.dumps({"passed": not bad and bool(out), "launches": len(out), "nonzero_local": bad[:20],
                  "local_columns": spill_cols}))
