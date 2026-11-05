"""One job: check, bench, then the protected csa2 publish-candidates unit tests."""

import json
import subprocess
import sys

here = "experiments/ascent/csa2_cand_publish/bench.py"
results = {}
for mode in sys.argv[1:] or ["check", "bench", "pytest"]:
    if mode == "pytest":
        cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-k", "publish or candidate",
               "tests/unittest/_torch/attention/sparse/csa2/"]
    else:
        cmd = [sys.executable, here, mode]
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout.strip().splitlines()
    print(p.stdout[-20000:], flush=True)
    print(p.stderr[-6000:], flush=True)
    last = out[-1] if out else ""
    results[mode] = {"returncode": p.returncode, "last": last[-3000:]}
ok = all(r["returncode"] == 0 for r in results.values())
try:
    ok = ok and json.loads(results.get("check", {}).get("last", '{"passed": true}')).get("passed", False)
except Exception:  # noqa: BLE001
    ok = False
print(json.dumps({"passed": bool(ok), "steps": {k: v["returncode"] for k, v in results.items()}}))
