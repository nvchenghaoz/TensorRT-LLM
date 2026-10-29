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
COUNTS = ("derived__local_spilling_requests", "sass__inst_executed_local_loads",
          "sass__inst_executed_local_stores", "sass__inst_executed_register_spilling_mem_local",
          "smsp__sass_inst_executed_op_local_ld.sum", "smsp__sass_inst_executed_op_local_st.sum",
          "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
          "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")
spill_cols = [h for h in COUNTS if h in hdr]
reg_cols = [h for h in hdr if "registers" in h or "stack" in h]
bad, summary = [], []
for rec in out:
    s = {"grid": rec.get("Grid Size"), **{h: rec.get(h) for h in reg_cols},
         **{h: rec.get(h) for h in spill_cols}}
    summary.append(s)
    for h in spill_cols:
        v = (rec.get(h) or "").replace(",", "")
        try:
            if float(v) != 0.0:
                bad.append((h, v))
        except ValueError:
            pass
print(json.dumps({"passed": not bad and bool(out), "launches": len(out), "nonzero_local": bad[:20],
                  "count_columns": spill_cols, "per_launch": summary[:4]}))
