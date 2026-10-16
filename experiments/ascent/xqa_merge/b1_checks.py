#!/usr/bin/env python3
# Correctness job of idea 93: runs b1_check.py <m> (kernel determinism + FP32-reference rule, acceptance item 2) and
# b1_grid_proof.py <m> <m_b2> (built-library launch grid at B=1/B=2, acceptance item 3) in one job, streaming their
# output, and prints one combined JSON line.
# Usage: b1_checks.py <picked m at B=1> <base m at B=2>
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args):
    cmd = [sys.executable, os.path.join(HERE, script)] + args
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.strip().startswith("{"):
            last = line
    rc = p.wait()
    try:
        r = json.loads(last) if last else {"passed": False, "error": "no JSON"}
    except Exception as e:  # noqa: BLE001
        r = {"passed": False, "error": f"bad JSON: {e}"}
    r["returncode"] = rc
    return r


def main():
    m, m2 = sys.argv[1], sys.argv[2]
    chk = run("b1_check.py", [m])
    grid = run("b1_grid_proof.py", [m, m2])
    print(json.dumps({"passed": bool(chk.get("passed") and grid.get("passed") and chk["returncode"] == 0
                                     and grid["returncode"] == 0),
                      "kernel_check": chk, "grid_proof": grid,
                      "compared": "kernel_check: xqa_evidence --mode check (fp8; 1x48,1x47,1x23; 360 calls each vs "
                                  "call 1 and vs original mha.cu; FP32 reference) + reordered rule vs B1_m48; "
                                  "grid_proof: eager TrtllmAttention decode at num_cached_tokens 36863, kernel_mha grid"}),
          flush=True)


if __name__ == "__main__":
    main()
