"""Branch 165: pull L2 -> SM byte metrics per launch out of an ncu report (raw page)."""
import csv
import io
import json
import subprocess
import sys

rep = sys.argv[1]
r = subprocess.run(["ncu", "-i", rep, "--page", "raw", "--csv"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-2000:])
rows = list(csv.reader(io.StringIO(r.stdout)))
hdr, units, data = rows[0], rows[1], rows[2:]
keep = [i for i, h in enumerate(hdr) if any(k in h for k in (
    "Kernel Name", "Grid Size", "Block Size", "gpu__time_duration.sum",
    "lts__t_bytes.sum", "lts__t_sectors_srcunit_tex", "lts__t_sectors_op_read.sum",
    "lts__t_requests_srcunit_tex", "l1tex__m_xbar2l1tex_read_bytes", "lts__t_sectors.sum",
    "dram__bytes_read.sum", "lts__t_sectors_srcunit_l1", "sm__warps_active", "launch__occupancy",
    "launch__registers", "launch__shared_mem", "launch__waves", "lts__t_sector_hit_rate",
    "launch__occupancy_limit"))]
out = []
for d in data:
    out.append({hdr[i] + (f" [{units[i]}]" if units[i] else ""): d[i] for i in keep})
for o in out:
    print(json.dumps(o))
print(json.dumps({"launches": out}))
