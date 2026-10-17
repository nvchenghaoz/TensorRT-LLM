#!/usr/bin/env python3
# STEP 1 of idea 93 (B=1 XQA multi-block cap): runs xqa_evidence.py --mode bench (served e4m3 KV shape) for the B=1
# sub-sequence counts in two separate processes with opposite config order, and prints a summary JSON with every
# config's trimmed-mean/median/p90 per process plus the pick rule (47 if >=10% below 48 in both; 23 only if >=3% below
# 47 in both). Kernel source is unchanged; only the launch grid's m differs.
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(HERE, "xqa_evidence.py")
KEYS = ("kernel_ms", "median_ms", "p10_ms", "p90_ms", "noise_floor", "output_sha256", "output_finite")


def run(order, extra_args):
    env = dict(os.environ, XQA_KV_DTYPE="fp8")
    # bench keys --configs by batch, so the first count goes in --configs and the rest in --extra-configs (in order)
    cmd = [sys.executable, EVID, "--mode", "bench", "--configs", f"1x{order[0]}", "--extra-configs",
           ",".join(f"1x{m}" for m in order[1:])] + extra_args
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.strip():
            last = line
    assert p.wait() == 0, "bench failed"
    r = json.loads(last)
    res = {f"1x{order[0]}": {k: r["batch"]["1"][k] for k in KEYS}}
    for m in order[1:]:
        res[f"1x{m}"] = {k: r["extra"][f"1x{m}"][k] for k in KEYS}
    return {"order": [f"1x{m}" for m in order], "results": res, "config": r.get("config")}


def main():
    ms = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "48,47,23").split(",")]
    extra = sys.argv[2:]
    out = {"processes": [run(ms, extra), run(list(reversed(ms)), extra)]}
    med = lambda p, m: p["results"][f"1x{m}"]["median_ms"]
    pick = {}
    if 48 in ms and 47 in ms:
        d47 = [1 - med(p, 47) / med(p, 48) for p in out["processes"]]
        pick["m47_below_m48_median"] = d47
        pick["m47_ge_10pct_both"] = all(d >= 0.10 for d in d47)
    if 47 in ms and 23 in ms:
        d23 = [1 - med(p, 23) / med(p, 47) for p in out["processes"]]
        pick["m23_below_m47_median"] = d23
        pick["m23_ge_3pct_both"] = all(d >= 0.03 for d in d23)
    pick["picked_m"] = 23 if pick.get("m23_ge_3pct_both") else 47
    out["pick"] = pick
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
