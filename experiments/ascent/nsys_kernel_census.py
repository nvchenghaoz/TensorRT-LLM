"""Print a kernel-name census from an nsys sqlite export (all kernels, eager vs graph)."""
import json
import sqlite3
import sys

path = sys.argv[1]
con = sqlite3.connect(path)
cur = con.cursor()
cols = [r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
name_col = "demangledName" if "demangledName" in cols else "shortName"
graph = "graphNodeId" in cols
q = (f"SELECT s.value, COUNT(*), SUM(k.end-k.start), "
     f"SUM(CASE WHEN {'k.graphNodeId IS NOT NULL' if graph else '0'} THEN 1 ELSE 0 END) "
     f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.{name_col} "
     f"GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC")
rows = cur.execute(q).fetchall()
out = []
for name, n, tot, ng in rows:
    out.append({"name": name[:160], "calls": n, "graph_calls": ng, "total_us": round(tot / 1e3, 1)})
    print(f"{n:6d} (graph {ng:5d}) {tot/1e3:10.1f}us  {name[:160]}", flush=True)
keys = ("rmsnorm", "RMSNorm", "rope", "Rope", "direct_copy", "Float8", "float8", "copy_kernel",
        "gemma4_qkv", "AppendPagedKV", "elementwise")
summary = {k: sum(r["calls"] for r in out if k in r["name"]) for k in keys}
print(json.dumps({"distinct_kernels": len(out), "match_counts": summary}))
