#!/usr/bin/env python3
"""Eager vs breakable-prefill-graph (BCG) check for Gemma-4-31B, in-process LLM API.

Parent mode (no --arm): runs the eager arm and the bcg arm each in a fresh subprocess
(one model load each), then compares their outputs and timings.

Each arm: builds LLM(checkpoint) with max_num_tokens=8192 and block reuse off, then for
each prompt length L in LENGTHS sends a fixed seeded prompt of exactly L token ids,
greedy, max_tokens=4, logprobs=5. Repeated execution: each prompt is re-sent REPEATS times
in the same process (no reset between calls); every repeat is compared with the first.
Timing: wall time of llm.generate for a single prompt (c=1), median/p10/p90 over the
timed repeats after warmup. Also a packed batch of 4x300 prompts.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

LENGTHS = [300, 310, 1000, 1010, 2000]
WARMUP = 3
REPEATS = 12
SEED = 1234


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_prompt(length, salt):
    import random

    rng = random.Random(SEED * 1000003 + length * 7 + salt)
    return [2] + [rng.randrange(1000, 200000) for _ in range(length - 1)]


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def run_arm(arm, checkpoint, buckets, out_path):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    kwargs = dict(
        model=checkpoint,
        max_num_tokens=8192,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
    )
    if arm == "bcg":
        kwargs.update(
            prefill_cuda_graph_backend="breakable",
            prefill_capture_num_tokens=buckets,
            disable_mm_encoder=True,
        )
    elif arm == "mm_off":
        kwargs.update(disable_mm_encoder=True)
    t0 = time.time()
    llm = LLM(**kwargs)
    startup_s = time.time() - t0
    log(f"arm={arm} startup {startup_s:.1f}s")
    sp = SamplingParams(max_tokens=4, temperature=0.0, logprobs=5)

    def one(prompt_ids):
        t = time.perf_counter()
        out = llm.generate([prompt_ids], sp, use_tqdm=False)[0]
        dt = (time.perf_counter() - t) * 1e3
        c = out.outputs[0]
        lps = []
        for d in (c.logprobs or []):
            lps.append({str(k): float(getattr(v, "logprob", v)) for k, v in d.items()})
        return dt, list(c.token_ids), lps

    result = {"arm": arm, "startup_s": startup_s, "cases": {}}
    for L in LENGTHS:
        prompt = make_prompt(L, 0)
        times, outs, all_times = [], [], []
        for i in range(WARMUP + REPEATS):
            dt, toks, lps = one(prompt)
            outs.append((toks, lps))
            all_times.append(dt)
            if i >= WARMUP:
                times.append(dt)
        first = outs[0]
        rep_same = all(o[0] == first[0] for o in outs)
        rep_lp_maxdiff = 0.0
        for o in outs:
            for a, b in zip(o[1], first[1]):
                for k in set(a) & set(b):
                    rep_lp_maxdiff = max(rep_lp_maxdiff, abs(a[k] - b[k]))
        med = statistics.median(times)
        result["cases"][str(L)] = {
            "median_ms": med,
            "p10_ms": pct(times, 0.1),
            "p90_ms": pct(times, 0.9),
            "first_call_ms": all_times[0],
            "tokens": first[0],
            "logprobs": first[1],
            "repeat_tokens_same": rep_same,
            "repeat_logprob_maxdiff": rep_lp_maxdiff,
        }
        log(f"arm={arm} L={L} median {med:.2f} ms p10 {pct(times,0.1):.2f} p90 "
            f"{pct(times,0.9):.2f} toks {first[0]} rep_same {rep_same} rep_lpdiff {rep_lp_maxdiff:.3g}")
    # Packed batch: 4 x 300 in one generate call (one 1200-token context step if packed).
    prompts = [make_prompt(300, s + 1) for s in range(4)]
    btimes, bouts = [], []
    for i in range(WARMUP + REPEATS):
        t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = (time.perf_counter() - t) * 1e3
        bouts.append([list(o.outputs[0].token_ids) for o in outs])
        if i >= WARMUP:
            btimes.append(dt)
    result["cases"]["4x300"] = {
        "median_ms": statistics.median(btimes),
        "p10_ms": pct(btimes, 0.1),
        "p90_ms": pct(btimes, 0.9),
        "tokens": bouts[0],
        "repeat_tokens_same": all(b == bouts[0] for b in bouts),
    }
    log(f"arm={arm} 4x300 median {statistics.median(btimes):.2f} ms toks {bouts[0]}")
    with open(out_path, "w") as f:
        json.dump(result, f)
    llm.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buckets", default="300,320,512,1000,1024,1200,2000")
    ap.add_argument("--arms", default="eager,bcg")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    buckets = [int(x) for x in args.buckets.split(",") if x]
    if args.arm:
        run_arm(args.arm, args.checkpoint, buckets, args.out)
        return
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    res = {}
    step_timing = {}
    for arm in args.arms.split(","):
        out = os.path.join(art, f"arm_{arm}.json")
        log(f"starting arm {arm}")
        env = dict(os.environ)
        if args.diag:
            env["TRTLLM_PREFILL_STEP_TIMING"] = "1"
            env["TLLM_LOG_LEVEL"] = "INFO"
        p = subprocess.Popen([sys.executable, __file__, "--arm", arm, "--checkpoint",
                              args.checkpoint, "--buckets", args.buckets, "--out", out],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=env)
        rows = {}
        import re

        ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        filt = open(os.path.join(art, f"filtered_{arm}.txt"), "w")
        for line in p.stdout:
            line = ansi.sub("", line)
            if any(s in line for s in ("BCG", "PREFILL_STEP", "prefill CUDA graph", "Captured",
                                       "Breakable", "KV cache", "kv cache", "Memory",
                                       "memory", "breakable")):
                filt.write(line)
                filt.flush()
            if "PREFILL_STEP_TIMING" in line:
                kv = dict(t.split("=", 1) for t in line.split("PREFILL_STEP_TIMING", 1)[1].split()
                          if "=" in t)
                key = f"{kv['tokens']}->{kv['run_as']}:{kv['mode']}"
                rows.setdefault(key, {"host": [], "gpu": []})
                rows[key]["host"].append(float(kv["host_enqueue_ms"]))
                rows[key]["gpu"].append(float(kv["gpu_ms"]))
                continue
            print(line, end="", flush=True)
        p.wait()
        step_timing[arm] = {
            k: {"n": len(v["host"]), "host_enqueue_ms_median": statistics.median(v["host"]),
                "gpu_ms_median": statistics.median(v["gpu"]),
                "host_p10_p90": [pct(v["host"], 0.1), pct(v["host"], 0.9)]}
            for k, v in rows.items() if len(v["host"]) >= 3}
        if p.returncode != 0 or not os.path.exists(out):
            print(json.dumps({"passed": False, "error": f"arm {arm} failed rc={p.returncode}"}))
            return
        with open(out) as f:
            res[arm] = json.load(f)
    base = res.get("eager")
    summary = {"passed": True, "compared": "bcg vs eager main path, same process settings",
               "startup_s": {a: r["startup_s"] for a, r in res.items()}, "cases": {},
            "step_timing": step_timing}
    for arm, r in res.items():
        for case, c in r["cases"].items():
            row = summary["cases"].setdefault(case, {})
            row[f"{arm}_median_ms"] = round(c["median_ms"], 3)
            row[f"{arm}_p10_p90"] = [round(c["p10_ms"], 3), round(c["p90_ms"], 3)]
            row[f"{arm}_repeat_same"] = c["repeat_tokens_same"]
            if "first_call_ms" in c:
                row[f"{arm}_first_call_ms"] = round(c["first_call_ms"], 3)
            if not c["repeat_tokens_same"]:
                summary["passed"] = False
            if base is not None and arm != "eager":
                b = base["cases"][case]
                same = c["tokens"] == b["tokens"]
                row[f"{arm}_tokens_match_eager"] = same
                if "logprobs" in c:
                    md = 0.0
                    for x, y in zip(c["logprobs"], b["logprobs"]):
                        for k in set(x) & set(y):
                            md = max(md, abs(x[k] - y[k]))
                    row[f"{arm}_logprob_maxdiff_vs_eager"] = md
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
