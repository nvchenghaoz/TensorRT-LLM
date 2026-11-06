"""One job: hist_bench.py steps (variants / check) and the protected csa2 publish tests."""

import json
import subprocess
import sys

here = "experiments/ascent/csa2_cand_publish/hist_bench.py"
results = {}
for step in sys.argv[1:] or ["variants", "check", "pytest"]:
    if step == "pytest":
        cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-k", "publish or candidate",
               "tests/unittest/_torch/attention/sparse/csa2/"]
    else:
        cmd = [sys.executable, here] + step.split(":")
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.strip():
            last = line.strip()
    p.wait()
    results[step] = {"returncode": p.returncode, "last": last[-4000:]}
ok = all(r["returncode"] == 0 for r in results.values())
summary = {}
for step, r in results.items():
    try:
        summary[step] = json.loads(r["last"])
    except Exception:  # noqa: BLE001
        summary[step] = r["last"][-500:]
if "check" in summary and isinstance(summary["check"], dict):
    ok = ok and bool(summary["check"].get("passed"))
    summary["check"] = {"passed": summary["check"].get("passed")}
print(json.dumps({"passed": bool(ok), "steps": {k: v["returncode"] for k, v in results.items()},
                  "summary": summary}))
