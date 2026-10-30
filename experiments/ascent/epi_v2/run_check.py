"""Frozen check.py from a worker job (fact 537 / pitfall 558), waiting for a benchmark copy.

usage: run_check.py <worktree> <checkpoint> [wait_s]

Runner slot copies of experiments/ascent-benchmark exist only while some evaluator job holds them
(job 612 found none; job 637 found slots/3/native/trtllm). This polls the known roots plus a
bounded find every 30 s for up to wait_s (default 1800), then hands off to
experiments/ascent/pair_residual_raster/run_check.py, which requires identical sha256 of the
frozen files across copies, copies one into the worktree and runs check.py unchanged.
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pair_residual_raster"))
import run_check as rc  # noqa: E402

RUN = "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507"


def found():
    hits = []
    for pat in rc.ROOTS:
        hits += [p for p in glob.glob(os.path.join(pat, "experiments", "ascent-benchmark", "check.py"))]
    if not hits:
        r = subprocess.run(["timeout", "60", "find", f"{RUN}/runner/slots", "-maxdepth", "7",
                            "-path", "*experiments/ascent-benchmark/check.py"],
                           capture_output=True, text=True)
        hits = [h for h in r.stdout.split() if h]
        for h in hits:
            root = os.path.dirname(os.path.dirname(os.path.dirname(h)))
            if root not in rc.ROOTS:
                rc.ROOTS.append(root)
    return hits


def main():
    wait_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    t0 = time.time()
    while True:
        hits = found()
        rc.log(f"benchmark copies visible: {hits[:5]}")
        if hits or time.time() - t0 > wait_s:
            break
        time.sleep(30)
    sys.argv = sys.argv[:3]
    rc.main()


if __name__ == "__main__":
    main()
