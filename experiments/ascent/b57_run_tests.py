"""Branch 57: run the full csa2 test_indexer.py with its two stale tests replaced by the
repaired copies in b57_repaired_tests.py (tests/ is protected). Also reports the
status of the two original tests (expected: the stale-hook test fails on main and
branch alike with calls == []; the old constant-rows test fails on the branch
because it asserted the buggy capture-only caching)."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SUITE = "tests/unittest/_torch/attention/sparse/csa2/test_indexer.py"
STALE = ["test_gathered_decode_graph_grows_past_warmup_prefix",
         "test_constant_rows_retained_only_for_captured_graphs"]


def run(args):
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-s", *args]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout
    lines = out.strip().splitlines()
    summary = ""
    for ln in reversed(lines):
        if re.search(r"\d+ (passed|failed)", ln):
            summary = ln
            break
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)",
                                              summary)}
    failed = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    extra = [ln for ln in lines if "[repaired]" in ln]
    print(out[-5000:], flush=True)
    sys.stderr.write(proc.stderr[-2000:])
    return dict(returncode=proc.returncode, summary=summary.strip("= "), counts=counts,
                failed=failed[:30], notes=extra[:5])


def main():
    suite = run([SUITE, *[a for t in STALE for a in ("--deselect", f"{SUITE}::{t}")]])
    repaired = run([str(HERE / "b57_repaired_tests.py")])
    stale = run([f"{SUITE}::{t}" for t in STALE])
    passed = (suite["returncode"] == 0 and suite["counts"].get("passed", 0) > 100
              and repaired["returncode"] == 0 and repaired["counts"].get("passed", 0) == 5)
    print(json.dumps({
        "passed": bool(passed),
        "suite_minus_two_stale": suite,
        "repaired_tests": repaired,
        "original_stale_tests": stale,
        "compared": "csa2 test_indexer.py (all tests except the two stale ones) + repaired "
                    "gathered-decode graph test (hook counts both gather routes, all replay "
                    "asserts kept) + constant-rows (a)-(d)",
    }))


if __name__ == "__main__":
    main()
