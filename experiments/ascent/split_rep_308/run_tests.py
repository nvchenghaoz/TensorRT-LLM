"""Idea 308: csa2 test_indexer / test_kernel / test_backend in this tree; per-test outcomes.

Runs in either tree (run_gpu_pair) so main's known failures can be subtracted. Prints
JSON with counts and the failed test ids; ``passed`` is true when pytest ran tests.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUITES = [
    "tests/unittest/_torch/attention/sparse/csa2/test_indexer.py",
    "tests/unittest/_torch/attention/sparse/csa2/test_kernel.py",
    "tests/unittest/_torch/attention/sparse/csa2/test_backend.py",
]


def main():
    report = {}
    total_run = 0
    for suite in SUITES:
        if not (ROOT / suite).exists():
            report[suite] = {"missing": True}
            continue
        cmd = [sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", suite]
        print(" ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=dict(os.environ))
        lines = []
        for ln in proc.stdout:
            lines.append(ln.rstrip())
            if len(lines) % 20 == 0 or "passed" in ln or "failed" in ln:
                print(ln.rstrip()[:200], flush=True)
        proc.wait()
        summary = next((ln for ln in reversed(lines)
                        if re.search(r"\d+ (passed|failed|error|skipped)", ln)), "")
        counts = {k: int(v) for v, k in re.findall(
            r"(\d+) (passed|failed|skipped|errors|error|xfailed|deselected)", summary)}
        failed = sorted({ln.split(" ")[1] for ln in lines
                         if ln.startswith(("FAILED ", "ERROR ")) and len(ln.split(" ")) > 1})
        total_run += counts.get("passed", 0) + counts.get("failed", 0)
        report[suite] = {"returncode": proc.returncode, "summary": summary.strip("= "),
                         "counts": counts, "failed": failed[:60]}
        print(json.dumps(report[suite]), flush=True)
    print(json.dumps({"passed": total_run > 0, "root": str(ROOT), "suites": report,
                      "compared": "csa2 unit tests (outcomes; compare failed sets across arms)"}),
          flush=True)


if __name__ == "__main__":
    main()
