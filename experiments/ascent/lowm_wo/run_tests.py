"""Idea 233: run the csa2 module tests (incl. the inverse-RoPE observer tests) on the branch, with the low-M WO
route on (default) and off (TRTLLM_CSA2_LOWM_WO=0, main's b12x route). tests/ is protected, so it only runs them."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUITE = "tests/unittest/_torch/attention/sparse/csa2/test_module.py"


def run(extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    # Pre-existing failures on unmodified main (top lead rulings 19:5x / 20:10 PST 11-04, knowledge 2200):
    # module.py _shared_mxfp8_active reads wq_a.quant_method on the test's nn.Identity stubs.
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-k",
           "not test_module_sends_whole_context_and_generation_phases", SUITE]
    print(" ".join(cmd), extra_env, flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines()
    summary = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error)", ln)), "")
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors|deselected)", summary)}
    failed = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    rope = [ln for ln in lines if ln.startswith(("PASSED", "FAILED", "ERROR")) and re.search("rope|invers", ln, re.I)]
    print(proc.stdout[-5000:], flush=True)
    sys.stderr.write(proc.stderr[-2000:])
    return {"returncode": proc.returncode, "summary": summary.strip("= "), "counts": counts, "failed": failed[:30],
            "inverse_rope_tests": rope[:30]}


def main():
    if not (ROOT / SUITE).exists():
        print(json.dumps({"passed": False, "error": f"{SUITE} not found under {ROOT}"}))
        return
    on = run({})
    off = run({"TRTLLM_CSA2_LOWM_WO": "0"})
    ok = all(r["returncode"] == 0 and not r["counts"].get("failed") and r["counts"].get("passed", 0) > 0
             for r in (on, off))
    print(json.dumps({"passed": ok, "lowm_on": on, "lowm_off": off}))


if __name__ == "__main__":
    main()
