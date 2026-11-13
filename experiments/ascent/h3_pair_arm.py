"""One arm of a same-slot worker pair for the MiniMax-H3 Cache-DiT branch.

Runs experiments/ascent/h3_cache_check.py (one TP2 VisualGen instance, untimed 2-step
warmup, then the given seeds back to back, nothing reset in between). The cache config is
taken from this tree's experiments/ascent-config/engine.json (cache_config) when present,
so the base tree runs without caching and the branch runs with its engine overlay.

Prints the evidence JSON on the last line: per-request wall time (ms) for batch 1,
noise_floor = (max - min) / median over the timed requests, and whether repeated
same-seed requests are bitwise identical.

Usage: python h3_pair_arm.py <checkpoint> [seed ...]   (default seeds 42 42 42)
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile

import torch


def main() -> None:
    ckpt = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [42, 42, 42]
    here = os.path.dirname(os.path.abspath(__file__))
    engine = os.path.join(here, "..", "ascent-config", "engine.json")
    cache = "none"
    if os.path.isfile(engine):
        cfg = json.load(open(engine)).get("cache_config")
        if cfg:
            cache = json.dumps(cfg)
    tag = "cache" if cache != "none" else "base"
    out = tempfile.mkdtemp(prefix="h3_pair_")
    cmd = [sys.executable, os.path.join(here, "h3_cache_check.py"), "--model", ckpt, "--tag", tag,
           "--cache", cache, "--out", out, "--seeds", *map(str, seeds)]
    print(f"[pair-arm] {tag}: {' '.join(cmd)}", flush=True)
    env = dict(os.environ, TLLM_LOG_LEVEL="INFO")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = None
    decisions = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("{") and '"results"' in line:
            last = json.loads(line)
        if "Cache-DiT decisions" in line:
            decisions.append(line.strip()[-400:])
    rc = proc.wait()
    if rc != 0 or last is None:
        print(json.dumps({"passed": False, "error": f"arm rc={rc}"}))
        sys.exit(1)
    res = last["results"]
    secs = [r["seconds"] for r in res]
    med = statistics.median(secs)
    floor = (max(secs) - min(secs)) / med if len(secs) > 1 else 0.0
    first = torch.load(res[0]["path"])
    repeat = []
    for r in res[1:]:
        if r["seed"] != res[0]["seed"]:
            continue
        o = torch.load(r["path"])
        repeat.append(bool(torch.equal(first["video"], o["video"]))
                      and bool(torch.equal(first["audio"], o["audio"])))
    result = {
        "batch": {"1": {"step_ms": round(1000.0 * med, 1), "noise_floor": round(floor, 4),
                        "request_s": secs}},
        "output": {"class": "mode_switch" if tag == "cache" else "exact",
                   "note": "error vs the uncached base: job 55 (video PSNR 31.2 dB, audio SNR 19.4 dB)"},
        "arm": tag,
        "cache_config": None if cache == "none" else json.loads(cache),
        "repeat_bitwise_equal": repeat,
        "cache_dit_decision_lines": len(decisions),
        "passed": all(repeat),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
