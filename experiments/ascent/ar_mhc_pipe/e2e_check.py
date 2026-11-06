"""Idea 96 / branch 199: end-to-end check of the prefill AR pipeline in the real model (LLM API, 4 GPUs, TP4).

usage: python e2e_check.py <checkpoint> [arms=off,on] [reps=8]
Each arm is a fresh process (env TRTLLM_DSV41_PREFILL_AR_PIPELINE: off=0, on=default) with the protocol's server
settings deep-merged with this tree's experiments/ascent-config/engine.json (as lp_ar_trace/trace.py). Per arm: prime
the 32768-token prefix, warm up (c1 + a 4-request batch), then REPS repeated executions, each = a batch of 4 fresh
2048-token suffixes (one prefill step of up to 8192 new tokens, the c4 shape) + one c1 request, greedy with top-2
logprobs, OUT tokens each. The 'on' arm also profiles one 4-request batch (prefill + decode steps) and reduces the
trace per rank: per EXTEND step the LP two-shot / NCCL / lamport counts by stream, the main-stream kernel that
precedes each comm-stream LP preprocess, aux-stream overlap; per DECODE step LP must be 0 and lamport > 0. TAIL_MOE_GRAPH
capture lines (TLLM_LOG_LEVEL=INFO) are counted per arm.
Compare (driver): every request of 'on' vs 'off' token by token with top-2 logprobs: until the first divergence
|dlp| <= lp_tol (reference.json tolerance 1.6835); a divergence must be a near tie in 'off' (top-2 gap <= tie_tol
3.3669). Last stdout line: JSON with "passed".
"""
import glob
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_SETTINGS = {
    "context_parallel_size": 1, "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8], "enable_padding": True},
    "enable_attention_dp": False, "enable_chunked_prefill": True,
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5, "tokens_per_block": 128},
    "max_batch_size": 8, "max_num_tokens": 8192, "max_seq_len": 36864, "moe_expert_parallel_size": 4,
    "pipeline_parallel_size": 1, "tensor_parallel_size": 4, "trust_remote_code": True}
PREFIX, SUFFIX = 32768, 2048
OUT = int(os.environ.get("E2E_OUT_TOKENS", "16"))
LP_TOL, TIE_TOL = 1.6835, 3.3669
CAP_RE = re.compile(r"\[TAIL_MOE_GRAPH\] rank (\d+) layer (\d+): captured")
FAIL_RE = re.compile(r"\[TAIL_MOE_GRAPH\][^\n]*capture failed")
STEP_RE = re.compile(r"^(step\[(EXTEND|DECODE)[^\]]*\])")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ------------------------------------------------------------------------------------------------ trace reduction
def reduce_rank(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    evs = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"]
    ks = sorted([e for e in evs if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    gsteps = sorted([e for e in evs if e.get("cat") == "gpu_user_annotation" and STEP_RE.match(e.get("name", ""))],
                    key=lambda e: e["ts"])
    cpu_steps = sorted([e for e in evs if e.get("cat") == "user_annotation" and STEP_RE.match(e.get("name", ""))],
                       key=lambda e: e["ts"])
    launches = {}
    for e in evs:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launches[c] = e

    def label(k):
        ln = launches.get((k.get("args") or {}).get("correlation"))
        if ln is not None:
            for s in cpu_steps:
                if s.get("tid") == ln.get("tid") and s["ts"] <= ln["ts"] <= s["ts"] + s["dur"]:
                    return (s["ts"], STEP_RE.match(s["name"]).group(1))
        for s in gsteps:
            if s["ts"] <= k["ts"] <= s["ts"] + s["dur"]:
                return (s["ts"], STEP_RE.match(s["name"]).group(1))
        return None

    groups = {}
    for k in ks:
        lab = label(k)
        if lab is not None:
            groups.setdefault(lab, []).append(k)
    steps = []
    for lab in sorted(groups):
        lst = groups[lab]
        stream_of = lambda e: (e.get("args") or {}).get("stream", e.get("tid"))  # noqa: E731
        by_stream = {}
        for e in lst:
            by_stream.setdefault(str(stream_of(e)), []).append(e)
        main_stream = max(by_stream, key=lambda s: len(by_stream[s]))
        lp2 = [e for e in lst if "lowPrecisionTwoShot" in e["name"]]
        lpp = [e for e in lst if "lowPrecisionPreprocess" in e["name"]]
        nccl = [e for e in lst if re.search(r"ncclDevKernel_AllReduce|ncclKernel_AllReduce", e["name"])]
        lam = [e for e in lst if "oneshot_lamport" in e["name"]]
        mains = by_stream[main_stream]
        # main-stream kernel that finished last before each off-main LP preprocess started (its producer)
        pred = {}
        for e in lpp:
            if str(stream_of(e)) == main_stream:
                continue
            before = [m for m in mains if m["ts"] + m.get("dur", 0) <= e["ts"]]
            nm = before[-1]["name"][:60] if before else None
            pred[nm] = pred.get(nm, 0) + 1
        # overlap of each non-main stream's kernels with main-stream kernels (us)
        overlap = {}
        for s, lst_s in by_stream.items():
            if s == main_stream:
                continue
            tot = 0.0
            j = 0
            for e in lst_s:
                a, b = e["ts"], e["ts"] + e.get("dur", 0)
                for m in mains:
                    ma, mb = m["ts"], m["ts"] + m.get("dur", 0)
                    if mb <= a:
                        continue
                    if ma >= b:
                        break
                    tot += min(b, mb) - max(a, ma)
                j += 1
            overlap[s] = {"kernels": len(lst_s), "overlap_with_main_us": round(tot, 1),
                          "top": sorted({e["name"][:50] for e in lst_s})[:6]}
        steps.append({
            "label": lab[1], "kernels": len(lst), "main_stream": main_stream,
            "lp_twoshot": len(lp2), "lp_twoshot_by_stream": {s: sum(1 for e in lp2 if str(stream_of(e)) == s)
                                                             for s in by_stream},
            "lp_twoshot_us_median": sorted(e["dur"] for e in lp2)[len(lp2) // 2] if lp2 else None,
            "nccl_ar": len(nccl), "nccl_ar_by_stream": {s: sum(1 for e in nccl if str(stream_of(e)) == s)
                                                        for s in by_stream},
            "lamport": len(lam), "comm_lp_preprocess_predecessor": pred, "other_streams": overlap})
    ext = [s for s in steps if s["label"].startswith("step[EXTEND")]
    dec = [s for s in steps if s["label"].startswith("step[DECODE")]
    return {"steps": steps, "extend": ext,
            "decode": {"steps": len(dec), "lp": sum(s["lp_twoshot"] for s in dec),
                       "nccl_ar": sum(s["nccl_ar"] for s in dec), "lamport": sum(s["lamport"] for s in dec)}}


# ------------------------------------------------------------------------------------------------ one arm
def run_arm(ckpt, arm, reps, outdir):
    os.makedirs(outdir, exist_ok=True)
    eng = {}
    p = Path.cwd() / "experiments" / "ascent-config" / "engine.json"
    if p.exists():
        eng = json.loads(p.read_text())
    settings = deep_merge(SERVER_SETTINGS, {k: v for k, v in eng.items() if k not in ("env", "serve_args")})
    log(f"arm={arm} TRTLLM_DSV41_PREFILL_AR_PIPELINE={os.environ.get('TRTLLM_DSV41_PREFILL_AR_PIPELINE')} "
        f"MIN_ROWS={os.environ.get('TRTLLM_DSV41_PREFILL_AR_PIPELINE_MIN_ROWS')} settings={settings}")
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    vocab = int(cfg.get("vocab_size", 100000))
    rng = random.Random(1234)
    lo, hi = 1000, min(vocab - 1000, 120000)
    prefix = [rng.randrange(lo, hi) for _ in range(PREFIX)]

    def prompt(i):
        r = random.Random(7700 + i)
        return prefix + [lo + 3 + 37 * i] + [r.randrange(lo, hi) for _ in range(SUFFIX - 1)]

    from tensorrt_llm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=ckpt, **settings)
    log(f"LLM up in {time.time()-t0:.0f}s")
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spn = SamplingParams(max_tokens=OUT, temperature=0.0, ignore_eos=True, logprobs=2)
    res = {"arm": arm, "records": {}, "errors": []}

    def rec(key, o):
        out = o.outputs[0]
        lps = []
        for d in (out.logprobs or []):
            items = sorted(((int(t), float(getattr(v, "logprob", v))) for t, v in d.items()), key=lambda x: -x[1])
            lps.append(items[:2])
        res["records"][key] = {"tokens": list(out.token_ids), "logprobs": lps,
                               "cached": getattr(o, "cached_tokens", None)}

    try:
        llm.generate([prompt(0)], sp1)
        log("prime done")
        llm.generate([prompt(1)], spn)
        llm.generate([prompt(2 + j) for j in range(4)], spn)
        log("warmup done")
        for r in range(reps):
            keys = [f"r{r}-b{j}" for j in range(4)]
            outs = llm.generate([prompt(100 + 8 * r + j) for j in range(4)], spn)
            for k, o in zip(keys, outs):
                rec(k, o)
            o = llm.generate([prompt(100 + 8 * r + 5)], spn)
            rec(f"r{r}-c1", o[0])
            log(f"rep {r} done")
        if arm == "on":
            tdir = Path(outdir) / "trace"
            tdir.mkdir(parents=True, exist_ok=True)
            llm.start_profile(output_dir=str(tdir), num_steps=6, activities=["CPU", "GPU"])
            outs = llm.generate([prompt(900 + j) for j in range(4)], SamplingParams(max_tokens=4, temperature=0.0,
                                                                                    ignore_eos=True))
            try:
                llm.stop_profile()
            except Exception as exc:  # noqa: BLE001
                log(f"stop_profile: {exc!r}")
    except Exception as exc:  # noqa: BLE001
        res["errors"].append(repr(exc)[:800])
        log(traceback.format_exc())
    finally:
        try:
            llm.shutdown()
        except Exception as exc:  # noqa: BLE001
            log(f"shutdown: {exc!r}")
    time.sleep(5)
    if arm == "on":
        per_rank = {}
        for f in sorted(glob.glob(str(Path(outdir) / "trace" / "**" / "*.json*"), recursive=True)):
            m = re.search(r"rank[-_]?(\d+)", os.path.basename(f))
            try:
                per_rank[m.group(1) if m else os.path.basename(f)] = reduce_rank(f)
            except Exception as exc:  # noqa: BLE001
                per_rank[os.path.basename(f)] = {"error": repr(exc)[:300]}
        res["trace"] = per_rank
    res["sha256"] = hashlib.sha256(json.dumps({k: v["tokens"] for k, v in sorted(res["records"].items())})
                                   .encode()).hexdigest()
    (Path(outdir) / f"arm_{arm}.json").write_text(json.dumps(res))
    print(json.dumps({"arm": arm, "n": len(res["records"]), "errors": res["errors"], "sha256": res["sha256"]}))


# ------------------------------------------------------------------------------------------------ driver
def compare(a, b):
    """a = off (reference), b = on. Returns per-record verdicts."""
    rows = {}
    for k, ra in a["records"].items():
        rb = b["records"].get(k)
        if rb is None:
            rows[k] = {"passed": False, "reason": "missing"}
            continue
        max_dlp, div = 0.0, None
        for i, (ta, tb) in enumerate(zip(ra["tokens"], rb["tokens"])):
            la = dict(ra["logprobs"][i]) if i < len(ra["logprobs"]) else {}
            lb = dict(rb["logprobs"][i]) if i < len(rb["logprobs"]) else {}
            if ta == tb:
                if ta in la and ta in lb:
                    max_dlp = max(max_dlp, abs(la[ta] - lb[ta]))
                continue
            gap = None
            if len(ra["logprobs"][i]) >= 2:
                gap = abs(ra["logprobs"][i][0][1] - ra["logprobs"][i][1][1])
            div = {"pos": i, "off": ta, "on": tb, "off_top2_gap": gap}
            break
        ok = max_dlp <= LP_TOL and (div is None or (div["off_top2_gap"] is not None
                                                     and div["off_top2_gap"] <= TIE_TOL))
        rows[k] = {"passed": ok, "max_dlp_before_div": round(max_dlp, 4), "divergence": div,
                   "identical": ra["tokens"] == rb["tokens"]}
    return rows


def main():
    ckpt = sys.argv[1]
    arms = (sys.argv[2] if len(sys.argv) > 2 else "off,on").split(",")
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    os.chdir(root)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts")
    outdir = os.path.join(art, "e2e")
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"head={head} arms={arms} reps={reps}")
    results, caps = {}, {}
    for arm in arms:
        env = dict(os.environ)
        env.setdefault("TLLM_LOG_LEVEL", "INFO")
        for k in list(env):
            if k.startswith(("TRTLLM_DSV41_PREFILL_AR_PIPELINE", "NCCL_", "TRTLLM_DSV4_SHARED_OVERLAP",
                             "TRTLLM_DSV4_TAIL_MOE_GRAPH")):
                env.pop(k)
        if arm == "off":
            env["TRTLLM_DSV41_PREFILL_AR_PIPELINE"] = "0"
        argv = [sys.executable, os.path.abspath(__file__), "--arm", arm, ckpt, str(reps), outdir]
        log(f"start arm {arm}")
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                             env=env, start_new_session=True)
        last = [time.time()]
        captured, failed = set(), []

        def watchdog():
            while p.poll() is None:
                time.sleep(15)
                if time.time() - last[0] > 600 and p.poll() is None:
                    log(f"WATCHDOG arm {arm}: 600 s without a log line; killing")
                    try:
                        os.killpg(p.pid, 9)
                    except Exception:  # noqa: BLE001
                        pass
                    return
        threading.Thread(target=watchdog, daemon=True).start()
        for line in p.stdout:
            if line.strip():
                last[0] = time.time()
            m = CAP_RE.search(line)
            if m:
                captured.add((int(m.group(1)), int(m.group(2))))
            elif FAIL_RE.search(line):
                failed.append(line.strip()[:300])
            sys.stdout.write(line if len(line) < 3000 else line[:1500] + " ...[truncated]\n")
            sys.stdout.flush()
        p.wait()
        caps[arm] = {"captured_pairs": len(captured), "ranks": len({r for r, _ in captured}), "failed": failed[:4],
                     "rc": p.returncode}
        f = Path(outdir) / f"arm_{arm}.json"
        results[arm] = json.loads(f.read_text()) if f.exists() else {"records": {}, "errors": ["no output"]}
        log(f"arm {arm} rc={p.returncode} records={len(results[arm]['records'])} tail_graph={caps[arm]}")
    out = {"head": head, "tail_graph": caps, "sha256": {a: results[a].get("sha256") for a in results},
           "errors": {a: results[a].get("errors") for a in results}}
    passed = all(not results[a].get("errors") and len(results[a]["records"]) == 5 * reps for a in results)
    passed = passed and all(c["captured_pairs"] == 80 and not c["failed"] for c in caps.values())
    if "off" in results and "on" in results:
        rows = compare(results["off"], results["on"])
        out["compare"] = {"n": len(rows), "passed": sum(r["passed"] for r in rows.values()),
                          "identical": sum(r["identical"] for r in rows.values()),
                          "max_dlp_before_div": max([r.get("max_dlp_before_div", 0) for r in rows.values()] or [0]),
                          "divergences": {k: r["divergence"] for k, r in rows.items() if r.get("divergence")}}
        passed = passed and out["compare"]["passed"] == out["compare"]["n"]
        tr = results["on"].get("trace") or {}
        census = {}
        for rk, d in tr.items():
            if "error" in d:
                census[rk] = d
                continue
            census[rk] = {"extend": [{k: s[k] for k in ("label", "lp_twoshot", "lp_twoshot_by_stream", "nccl_ar",
                                                       "lamport", "comm_lp_preprocess_predecessor", "other_streams",
                                                       "lp_twoshot_us_median", "main_stream")}
                                     for s in d["extend"]], "decode": d["decode"]}
        out["trace_on"] = census
        dec_ok = bool(census) and all(c.get("decode", {}).get("lp", 1) == 0 and c.get("decode", {}).get("lamport", 0) > 0
                                      for c in census.values())
        out["decode_lamport_only_0_LP"] = dec_ok
        passed = passed and dec_ok
    out["passed"] = bool(passed)
    Path(outdir, "e2e_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--arm":
        run_arm(sys.argv[3], sys.argv[2], int(sys.argv[4]), sys.argv[5])
    else:
        main()
