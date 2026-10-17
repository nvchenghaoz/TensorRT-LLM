#!/usr/bin/env python3
# Correctness check of idea 93 (B=1 XQA multi-block cap): runs xqa_evidence.py --mode check (served e4m3 KV shape)
# with configs 1x48,1x47,1x23 (360 calls each: 3 rounds x (20 warmup + 100 iters), poison + L2 flush before every
# call; every call bitwise vs call 1, semaphores 0, finite, inputs unchanged, interleaved counts), then applies the
# reordered-class rule to the picked count: its FP32-reference max/mean abs error <= 1.25x the B=1 m=48 kernel's
# error + one BF16 ulp of the output magnitude (max |FP32 reference output| over the same inputs).
# Usage: b1_check.py <picked m>
import json
import math
import os
import subprocess
import sys

os.environ["XQA_KV_DTYPE"] = "fp8"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    picked = int(sys.argv[1])
    configs = sys.argv[2] if len(sys.argv) > 2 else "1x48,1x47,1x23"
    cmd = [sys.executable, os.path.join(HERE, "xqa_evidence.py"), "--mode", "check", "--configs", configs]
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.strip():
            last = line
    rc = p.wait()
    rep = json.loads(last)
    # output magnitude for the one-ulp term: the same fixed-seed inputs the check used
    sys.path.insert(0, HERE)
    import torch
    import xqa_evidence as xe
    inp = xe.build_inputs(8)
    ref = xe.reference(inp, 1)
    mag = float(ref.abs().max().item())
    ulp = 2.0**(math.floor(math.log2(mag)) - 7)
    base = rep["reference"]["B1_m48"]
    rule = {}
    for m in sorted({int(c.split("x")[1]) for c in configs.split(",")}):
        e = rep["reference"][f"B1_m{m}"]
        ok_max = e["candidate_max_abs"] <= 1.25 * base["candidate_max_abs"] + ulp
        ok_mean = e["candidate_mean_abs"] <= 1.25 * base["candidate_mean_abs"] + ulp
        rule[f"B1_m{m}"] = dict(max_abs=e["candidate_max_abs"], mean_abs=e["candidate_mean_abs"],
                                max_abs_vs_m48=e["candidate_max_abs"] / base["candidate_max_abs"],
                                mean_abs_vs_m48=e["candidate_mean_abs"] / base["candidate_mean_abs"],
                                within_1p25x_m48_plus_ulp=bool(ok_max and ok_mean))
        print(f"B1 m={m}: max {e['candidate_max_abs']:.3e} ({rule[f'B1_m{m}']['max_abs_vs_m48']:.3f}x m48) mean "
              f"{e['candidate_mean_abs']:.3e} ({rule[f'B1_m{m}']['mean_abs_vs_m48']:.3f}x m48) -> "
              f"{'PASS' if ok_max and ok_mean else 'FAIL'}", flush=True)
    passed = bool(rc == 0 and rep["passed"] and rule[f"B1_m{picked}"]["within_1p25x_m48_plus_ulp"])
    out = {"passed": passed, "picked_m": picked, "reordered_rule": rule, "output_magnitude": mag, "bf16_ulp": ulp,
           "harness_passed": rep["passed"], "repeated": rep["repeated"], "interleaved": rep["interleaved"],
           "reference": rep["reference"], "readonly_inputs_unchanged": rep["readonly_inputs_unchanged"],
           "compared_against": rep["compared_against"] + "; reordered rule: picked m vs B1_m48 FP32-reference error "
                               "<= 1.25x + 1 BF16 ulp", "shape": rep["shape"]}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
