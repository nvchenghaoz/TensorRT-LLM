#!/usr/bin/env python3
"""Idea 38 correctness job on the integrated tree (prefill.cuh carries v2m).

1. Op level (run.py): main's op = 'pin' overlay (reverse patch of the tree) vs the tree, sliding
   d256 BF16 Q + FP8 KV paged, every case incl. M=1/27/1200/4800/8100/8192 routes; per arm the
   untimed repeated-call sequence (warmup+iters calls, each bitwise vs the first; Q and KV pool
   bitwise unchanged), then tree vs pin bitwise and both vs the FP32 reference.
2. ptxas -v spill report for the v1 (2 CTAs/SM) build and the tree build.
3. Full model (model_ab.py): in-process LLM, pin vs tree, first-token ids + top-20 logprobs.
Last line: {"passed": ...}.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ("1x1,1x27,1x300,4x300,16x300,27x300,1x1000,4x1000,8x1000,1x2000,4x2000,16x512,"
         "1x4000,2x4000,1x8000,1x8192,r300+1000+2000+4000,r1+27+300+1200+2500,"
         "r7000+300+300+300")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, tag, keep=None):
    log(f"RUN {tag}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last, kept, prev = None, [], ""
    for line in p.stdout:
        line = line.rstrip("\n")
        if keep is None:
            print(f"[{tag}] {line}", flush=True)
        elif keep(line, prev):
            kept.append(prev.strip()[-200:] + " || " + line.strip()[-200:])
            print(f"[{tag}] {kept[-1]}", flush=True)
        elif "hb " in line or line.startswith("{") or "Building JIT" in line:
            print(f"[{tag}] {line[:300]}", flush=True)
        if line.startswith("{"):
            last = line
        prev = line
    rc = p.wait()
    log(f"{tag} rc={rc}")
    return (json.loads(last) if last else None), kept, rc


def main():
    ckpt = sys.argv[1]
    py = sys.executable
    res = {}
    op, _, _ = run([py, os.path.join(HERE, "run.py"), "--ckpt", ckpt, "--arms", "pin,main",
                    "--ref-arm", "pin", "--rounds", "1", "--cases", CASES], "op")
    op_ok = bool(op and op.get("passed"))
    bitwise = {}
    if op:
        for cs, row in op["numerics"].items():
            m = row.get("main", {})
            bitwise[cs] = m.get("bitwise_vs_main")
        op_ok = op_ok and all(bitwise.values()) and len(bitwise) == len(CASES.split(","))
    res["op"] = {"passed": op_ok, "tree_bitwise_vs_pin": bitwise,
                 "table": op.get("table") if op else None}

    def spill_line(line, prev):
        return ("spill" in line and "BatchPrefillWithPagedKVCache" in prev)
    spills = {}
    for arm in (() if "--no-ptxas" in sys.argv else ("v1", "v1b", "tree")):
        _, kept, rc = run([py, os.path.join(HERE, "bench.py"), "--arm", arm, "--ckpt", ckpt,
                           "--cases", "1x300", "--only-run", "--ptxas-v"], f"ptxas-{arm}",
                          keep=spill_line)
        spills[arm] = {"rc": rc, "paged_kernel_spill_lines": kept[:40]}
    res["ptxas"] = spills

    mab, _, _ = run([py, os.path.join(HERE, "model_ab.py"), "drive", "--arms", "pin,tree,pin",
                     "--ckpt", ckpt], "model")
    res["model_ab"] = mab
    passed = op_ok and bool(mab and mab.get("passed"))
    print(json.dumps({"passed": passed,
                      "compared": "tree (v2m) vs pin overlay (main's op): op bitwise on 16 sliding "
                      "cases + repeated-call sequence + FP32 ref; full-model first token + top-20 "
                      "logprobs bitwise", **res}), flush=True)


if __name__ == "__main__":
    main()
