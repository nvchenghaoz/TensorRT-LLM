"""Branch 57: run the FULL csa2 test_indexer.py with its two stale tests repaired in
place by b57_tests_plugin (tests/ is protected for workers), then the two original
stale tests unrepaired for the record (expected: the stale-hook test fails on main
and branch alike with calls == []; the old constant-rows test passes on main and
fails on the branch, because it asserted the buggy capture-only caching)."""

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


def run(args, plugin):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-s"]
    if plugin:
        cmd += ["-p", "b57_tests_plugin"]
    cmd += args
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = proc.stdout
    lines = out.strip().splitlines()
    summary = ""
    for ln in reversed(lines):
        if re.search(r"\d+ (passed|failed|error)", ln):
            summary = ln
            break
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)",
                                              summary)}
    failed = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    stale_status = [ln for ln in lines if ln.startswith(("PASSED", "FAILED", "ERROR"))
                    and any(t in ln for t in STALE)]
    notes = [ln for ln in lines if "[b57-plugin]" in ln]
    print("\n".join(notes), flush=True)
    print(out[-6000:], flush=True)
    sys.stderr.write(proc.stderr[-3000:])
    return dict(returncode=proc.returncode, summary=summary.strip("= "), counts=counts,
                failed=failed[:30], stale_tests=stale_status, plugin_notes=notes[:10])


def main():
    suite = run([SUITE], plugin=True)
    stale = run(["-k", " or ".join(STALE), SUITE], plugin=False)
    repaired_ok = (len([n for n in suite["plugin_notes"] if "passed" in n]) >= 3
                   and all(s.startswith("PASSED") for s in suite["stale_tests"])
                   and len(suite["stale_tests"]) == 2)
    passed = (suite["returncode"] == 0 and suite["counts"].get("passed", 0) > 100
              and not suite["counts"].get("failed") and repaired_ok)
    print(json.dumps({
        "passed": bool(passed),
        "full_suite_with_repaired_tests": suite,
        "original_stale_tests_unrepaired": stale,
        "compared": "full csa2 test_indexer.py; the stale gathered-decode graph test runs with "
                    "its hook counting both gather routes (all replay asserts kept) and the "
                    "old constant-rows test is replaced by (a)-(d)",
    }))


if __name__ == "__main__":
    main()
