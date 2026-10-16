#!/usr/bin/env python3
# Candidate arm of idea 93's kernel pair: runs xqa_evidence.py --mode bench (served e4m3 KV shape) with the
# sub-sequence counts this tree's DecoderXQARunner::runImpl launches (given as --configs), and declares the output
# class from the configs instead of the harness's hard-coded "exact" (pitfall 557): any count that differs from the
# base heuristic's 48 moves the split-KV boundaries and the FP32 merge order, so the class is "reordered".
# Usage: b1_pair_arm.py 1x47,2x48,4x48
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_M = 48


def main():
    configs = sys.argv[1]
    env = dict(os.environ, XQA_KV_DTYPE="fp8")
    cmd = [sys.executable, os.path.join(HERE, "xqa_evidence.py"), "--mode", "bench", "--configs", configs]
    cmd += sys.argv[2:]
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.strip():
            last = line
    assert p.wait() == 0, "bench failed"
    r = json.loads(last)
    ms = [int(c.split("x")[1]) for c in configs.split(",") if c]
    r["output"] = {"class": "reordered" if any(m != BASE_M for m in ms) else "exact",
                   "note": "class derived from the launched sub-sequence counts vs the base heuristic's 48; FP32-reference "
                           "errors come from xqa_evidence.py --mode check (b1_check.py)"}
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
