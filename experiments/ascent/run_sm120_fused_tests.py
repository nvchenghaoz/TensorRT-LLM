"""Run csa2 test_kernel.py + test_metadata.py with _sm100() admitting SM120 (see sm120_fused_tests_plugin)."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TESTS = [
    "tests/unittest/_torch/attention/sparse/csa2/test_kernel.py",
    "tests/unittest/_torch/attention/sparse/csa2/test_metadata.py",
]


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "pytest", "-p", "sm120_fused_tests_plugin", "-q", "-rA",
           "-p", "no:cacheprovider", *TESTS]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = proc.stdout
    keep = [ln for ln in out.splitlines()
            if re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)", ln) or "sm120-plugin" in ln]
    print("\n".join(keep), flush=True)
    print(out[-4000:], flush=True)
    sys.stderr.write(proc.stderr[-3000:])
    summary = out.strip().splitlines()[-1] if out.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", summary)}
    failed = [ln for ln in keep if ln.startswith(("FAILED", "ERROR"))]
    fused_ran = [ln for ln in keep if ln.startswith("PASSED") and
                 re.search(r"fused_store|fused_gather|native_staging|staging", ln)]
    passed = proc.returncode == 0 and counts.get("passed", 0) > 0 and not failed
    print(json.dumps({
        "passed": passed,
        "returncode": proc.returncode,
        "summary": summary.strip("= "),
        "counts": counts,
        "fused_tests_passed": len(fused_ran),
        "failed": failed[:40],
        "compared": "csa2 test_kernel.py + test_metadata.py on SM120 with _sm100() admitting major 12 "
                    "(fused store exact bytes atol=0, fused gather, native staging); SM100 packed attention stays skipped",
    }))


if __name__ == "__main__":
    main()
