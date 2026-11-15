#!/usr/bin/env python3
"""Breakable prefill CUDA graphs (idea 19): in-process A/B timing and bitwise check.

Derived from experiments/ascent/fi_plan_ab.py (same engine settings, prompts and stats).
The engine settings overlay is read from THIS tree's experiments/ascent-config/engine.json
(if present) and passed to LLM(): on main (no engine.json) the run is eager, on the branch it
uses prefill_cuda_graph_backend=breakable + disable_mm_encoder + the branch buckets. So the
same command runs in both arms of run_gpu_pair.

Modes (argv[1]):
  time   : per case (isl@conc) wall time of llm.generate() over `conc` fresh unique prompts of
           `isl` tokens, 1 greedy output token; median of ROUNDS (default 25) timed rounds after 3
           warmup rounds, p10/p90, noise_floor=(p90-p10)/median. Evidence JSON on the last line.
  arm    : argv[2] in {eager, overlay}; runs the bitwise stress set and writes digests to argv[3].
  check  : runs `arm eager` and `arm overlay` in two subprocesses and compares bitwise.
"""
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import numpy as np  # noqa: E402

CKPT = os.environ.get("CKPT") or "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/models/Gemma-4-31B-IT-NVFP4"
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_JSON = os.path.join(HERE, "..", "..", "ascent-config", "engine.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def print_dims():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "global_head_dim", "num_global_key_value_heads", "sliding_window", "intermediate_size"]
    log("config: " + json.dumps({k: tc.get(k) for k in keys}))


MIRROR_JSON = os.path.join(HERE, "engine_overlay.json")
USE_OVERLAY = os.environ.get("BCG_OVERLAY", "1") == "1"


def overlay():
    """engine.json of this tree if the job checkout has it, else the mirrored copy beside this
    script (identical to the branch's experiments/ascent-config/engine.json). BCG_OVERLAY=0
    (baseline arm) runs without it, i.e. main's eager settings."""
    if not USE_OVERLAY:
        return {}
    for path in (ENGINE_JSON, MIRROR_JSON):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


class Prompts:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def make(self, isl):
        return [int(x) for x in self.rng.integers(1000, 200000, size=isl)]


def build_llm(use_overlay=True, **extra):
    ov = dict(overlay()) if use_overlay else {}
    log(f"engine overlay: {json.dumps(ov)}")
    # "env" is the harness's reserved key for server environment variables; set them before
    # tensorrt_llm is imported (single-process worker reads them at import).
    for k, v in (ov.pop("env", None) or {}).items():
        os.environ[k] = str(v)
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    t0 = time.time()
    llm = LLM(model=CKPT,
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, **ov, **extra)
    log(f"startup {time.time() - t0:.1f}s")
    return llm


def sp(logprobs=None):
    from tensorrt_llm import SamplingParams
    kw = dict(max_tokens=1, temperature=0.0, ignore_eos=True)
    if logprobs:
        kw["logprobs"] = logprobs
    return SamplingParams(**kw)


PACK = os.environ.get("BCG_PACK", "0") == "1"


def run_batch(llm, prompts, params):
    if PACK and len(prompts) > 1:
        return run_batch_packed(llm, prompts, params)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return time.perf_counter() - t0, outs


def run_batch_packed(llm, prompts, params):
    """Enqueue all prompts before the in-process executor thread can fetch any of them
    (GIL switch interval raised during submission), so every round of a c>1 case runs the
    same step composition in both arms instead of racing arrival order (fi_plan_ab.py's
    isl300@4 is bimodal for this reason)."""
    old = sys.getswitchinterval()
    t0 = time.perf_counter()
    sys.setswitchinterval(0.5)
    try:
        futs = [llm.generate_async(p, sampling_params=params) for p in prompts]
    finally:
        sys.setswitchinterval(old)
    outs = [f.result() for f in futs]
    return time.perf_counter() - t0, outs


def summarize(o):
    c = o.outputs[0]
    lps = []
    if c.logprobs:
        for d in c.logprobs:
            if isinstance(d, dict):
                lps.append(sorted((int(k), float(getattr(v, "logprob", v))) for k, v in d.items()))
            else:
                lps.append(float(d))
    return {"ids": list(c.token_ids), "lp": lps}


def mode_time():
    print_dims()
    llm = build_llm()
    params = sp()
    pf = Prompts(1234)
    cases = [("isl300@1", 300, 1), ("isl300@4", 300, 4), ("isl1k@1", 1000, 1), ("isl1k@4", 1000, 4),
             ("isl2k@1", 2000, 1), ("isl2k@4", 2000, 4), ("isl4k@1", 4000, 1), ("isl8k@1", 8000, 1),
             ("isl300@16", 300, 16), ("isl4k@16", 4000, 16), ("isl8k@64", 8000, 64)]
    sel = os.environ.get("BCG_CASES")
    if sel:
        extra_cases = {"isl8k@4": ("isl8k@4", 8000, 4), "isl8k@16": ("isl8k@16", 8000, 16)}
        known = {c[0]: c for c in cases}
        known.update(extra_cases)
        cases = [known[s] for s in sel.split(",")]
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    mem = {"after_startup_free_gib": round(free_b / 2**30, 3),
           "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
    log(f"memory after startup: {mem}")
    log("warmup")
    for _, isl, c in cases:
        for _ in range(3 if c * isl < 60000 else 1):
            run_batch(llm, [pf.make(isl) for _ in range(c)], params)
    try:
        from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
        counts = getattr(_g, "_BCG_STEP_COUNTS", None)
    except Exception:  # noqa: BLE001
        counts = None
    res = {}
    steps = {}
    n = int(os.environ.get("ROUNDS", "25"))
    for name, isl, c in cases:
        ts = []
        rounds = n if c * isl < 20000 else max(10, n // 2)
        if name == "isl300@1":
            rounds = 2 * n
        if c * isl >= 200000:
            rounds = 5
        before = dict(counts) if counts is not None else None
        st0 = torch.cuda.memory_stats()
        for i in range(rounds):
            ps = [pf.make(isl) for _ in range(c)]
            dt, _ = run_batch(llm, ps, params)
            ts.append(dt * 1e3)
        st1 = torch.cuda.memory_stats()
        mem[name] = {k: st1.get(k, 0) - st0.get(k, 0)
                     for k in ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free")}
        log(f"{name}: allocator {mem[name]}")
        if counts is not None:
            steps[name] = {"replayed": counts["replayed"] - before["replayed"],
                           "eager": counts["eager"] - before["eager"]}
            log(f"{name}: context steps {steps[name]}")
        ts.sort()
        med = statistics.median(ts)
        p10 = ts[int(0.1 * (len(ts) - 1))]
        p90 = ts[int(round(0.9 * (len(ts) - 1)))]
        res[name] = {"step_ms": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
                     "noise_floor": round((p90 - p10) / med, 4), "n": len(ts)}
        log(f"{name}: median {med:.2f} ms p10 {p10:.2f} p90 {p90:.2f}")
    print(json.dumps({"batch": res, "output": {"class": "exact"}, "overlay": overlay(),
                      "bcg_context_steps": steps, "memory": mem}), flush=True)


def mode_arm(arm, out_path):
    print_dims()
    # Waiting for max_batch_size requests makes the mixed batch composition repeatable.
    llm = build_llm(use_overlay=(arm == "overlay"), batch_wait_timeout_ms=1500.0)
    pf = Prompts(777)
    params = sp(logprobs=20)
    groups = [(300, 64), (1000, 16), (2000, 8), (4000, 4), (8000, 2)]
    lists = [[pf.make(isl) for _ in range(n)] for isl, n in groups]
    mixed = []
    i = 0
    while any(lists):
        lst = lists[i % len(lists)]
        if lst:
            mixed.append(lst.pop())
        i += 1
    # c=1 prompts: 20 per ISL, plus off-bucket lengths that pad (310, 1010, 901, 2047).
    singles = []
    ps = Prompts(4242)
    for isl in (300, 1000, 2000, 4000, 8000):
        singles += [ps.make(isl) for _ in range(20)]
    singles += [ps.make(L) for L in (310, 901, 1010, 2047, 257, 333, 1500)]
    small_batches = [[ps.make(300) for _ in range(4)], [ps.make(300) for _ in range(2)],
                     [ps.make(1000) for _ in range(2)]]
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        run_batch(llm, [warm.make(isl) for _ in range(2)], params)
    out = {}
    for p in range(2):  # repeated execution: two full passes in one process, no reset
        tag = f"pass{p}"
        _, outs = run_batch(llm, mixed, params)
        out[tag + "_mixed"] = [summarize(o) for o in outs]
        s = []
        for pr in singles:
            _, o = run_batch(llm, [pr], params)
            s.append(summarize(o[0]))
        out[tag + "_singles"] = s
        b = []
        for grp in small_batches:
            _, o = run_batch(llm, grp, params)
            b.append([summarize(x) for x in o])
        out[tag + "_batches"] = b
        log(f"{arm} {tag} done; first mixed {out[tag + '_mixed'][0]['ids']}")
    out["single_lens"] = [len(x) for x in singles]
    try:
        from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
        out["bcg_steps"] = dict(getattr(_g, "_BCG_STEP_COUNTS", {}) or {})
    except Exception:  # noqa: BLE001
        out["bcg_steps"] = None
    with open(out_path, "w") as f:
        json.dump(out, f)


def lp_maxdiff(a, b):
    md = 0.0
    for x, y in zip(a, b):
        for la, lb in zip(x["lp"], y["lp"]):
            da, db = dict(la), dict(lb)
            for k in set(da) & set(db):
                md = max(md, abs(da[k] - db[k]))
    return md


def mode_check():
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    res = {}
    for arm in ("eager", "overlay"):
        path = os.path.join(art, f"stress_{arm}.json")
        log(f"arm {arm}")
        p = subprocess.run([sys.executable, __file__, "arm", arm, path])
        if p.returncode != 0:
            print(json.dumps({"passed": False, "error": f"arm {arm} rc={p.returncode}"}))
            return
        res[arm] = json.load(open(path))
    e, o = res["eager"], res["overlay"]
    cmp = {}
    ok = True
    for part in ("mixed", "singles"):
        for p in ("pass0", "pass1"):
            a, b = e[f"pass0_{part}"], o[f"{p}_{part}"]
            diff = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
            cmp[f"overlay_{p}_vs_eager_pass0_{part}"] = {"n": len(a), "n_diff": len(diff),
                                                       "first": diff[:5],
                                                       "lp_maxdiff": lp_maxdiff(a, b)}
            ok &= not diff
        a, b = e[f"pass0_{part}"], e[f"pass1_{part}"]
        diff = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
        cmp[f"eager_pass1_vs_pass0_{part}"] = {"n_diff": len(diff)}
    for p in ("pass0", "pass1"):
        a = [x for g in e["pass0_batches"] for x in g]
        b = [x for g in o[f"{p}_batches"] for x in g]
        diff = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
        cmp[f"overlay_{p}_vs_eager_pass0_batches"] = {"n": len(a), "n_diff": len(diff)}
        ok &= not diff
    has_lp = bool(e["pass0_singles"][0]["lp"])
    print(json.dumps({"passed": bool(ok and has_lp),
                      "compared": "token ids + top-20 logprobs bitwise: BCG overlay (both passes, "
                                  "same process, no reset) vs eager main path; mixed 94-prompt "
                                  "batch, 107 c=1 prompts (20 per ISL 300/1k/2k/4k/8k + 7 padded "
                                  "lengths), 3 small batches",
                      "overlay": overlay(), "comparisons": cmp, "logprobs_present": has_lp,
                      "bcg_context_steps": {"eager_arm": e.get("bcg_steps"),
                                            "overlay_arm": o.get("bcg_steps")}}),
          flush=True)


def mode_inproc():
    """Same-process replay-vs-eager bitwise A/B: ONE LLM with the BCG overlay; replay is
    switched off/on between passes by gating BreakableCUDAGraphRunner.has_graph from this
    script (both the input-prep graph check and the runner dispatch use it), so both modes
    share one AutoTuner cache, one KV pool and one process. Order: A(eager) B(replay)
    A'(eager) B'(replay), then a repeated-replay sequence (5 warmup + 20 calls, no reset)."""
    print_dims()
    llm = build_llm(batch_wait_timeout_ms=1500.0)
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (
        BreakableCUDAGraphRunner,
    )
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
    import tensorrt_llm._torch.attention.backends.flashinfer as _fi

    replay_on = [True]
    orig_has_graph = BreakableCUDAGraphRunner.has_graph
    BreakableCUDAGraphRunner.has_graph = lambda self, n: replay_on[0] and orig_has_graph(self, n)
    counts = getattr(_g, "_BCG_STEP_COUNTS", None)
    host_plan = getattr(_fi, "_HOST_PREFILL_PLAN", None)
    log(f"FlashInfer _HOST_PREFILL_PLAN={host_plan} (main default, same in all passes)")
    params = sp(logprobs=20)
    ps = Prompts(9090)
    # groups submitted together; batch_wait_timeout_ms fixes each group's step composition
    groups = []
    for isl in (300, 1000, 2000, 8000):          # c=1 (8000 > largest bucket: eager in both)
        groups += [("1x%d" % isl, [ps.make(isl)]) for _ in range(5)]
    for n, isl in ((2, 300), (3, 300), (4, 300), (5, 300), (6, 300), (2, 1000), (4, 1000)):
        groups += [("%dx%d" % (n, isl), [ps.make(isl) for _ in range(n)]) for _ in range(2)]
    warm = Prompts(5)
    for isl in (300, 1000, 2000, 8000):
        run_batch(llm, [warm.make(isl)], params)
    passes = {}
    steps = {}
    for tag, on in (("A", False), ("B", True), ("A2", False), ("B2", True)):
        replay_on[0] = on
        before = dict(counts) if counts is not None else None
        res = []
        for _, grp in groups:
            _, outs = run_batch(llm, grp, params)
            res.append([summarize(o) for o in outs])
        passes[tag] = res
        if counts is not None:
            steps[tag] = {k: counts[k] - before[k] for k in counts}
        log(f"pass {tag} replay={on} steps={steps.get(tag)}")
    # repeated replay, no reset: same prompt / same batch, 5 warmup + 20 calls
    replay_on[0] = True
    rep_single = ps.make(300)
    rep_batch = [ps.make(300) for _ in range(4)]
    rep_s, rep_b = [], []
    for i in range(25):
        _, o = run_batch(llm, [rep_single], params)
        rep_s.append(summarize(o[0]))
        _, o = run_batch(llm, rep_batch, params)
        rep_b.append([summarize(x) for x in o])
    replay_on[0] = False
    _, o = run_batch(llm, [rep_single], params)
    eager_s = summarize(o[0])
    _, o = run_batch(llm, rep_batch, params)
    eager_b = [summarize(x) for x in o]
    replay_on[0] = True
    BreakableCUDAGraphRunner.has_graph = orig_has_graph

    def diff(a, b):
        out = {}
        for (label, _), x, y in zip(groups, a, b):
            d = out.setdefault(label, [0, 0])
            d[1] += len(x)
            d[0] += sum(1 for u, v in zip(x, y) if u != v)
        return {k: f"{v[0]}/{v[1]} differ" for k, v in out.items()}, sum(v[0] for v in out.values())

    cmp = {}
    total = 0
    for a, b in (("B", "A"), ("B2", "A"), ("A2", "A"), ("B2", "B")):
        per, n = diff(passes[a], passes[b])
        cmp[f"{a}_vs_{b}"] = {"n_diff": n, "per_group": per}
        total += n
    rep_ok = all(x == rep_s[0] for x in rep_s) and all(x == rep_b[0] for x in rep_b)
    rep_eq_eager = rep_s[0] == eager_s and rep_b[0] == eager_b
    n_out = sum(len(x) for x in passes["A"])
    passed = total == 0 and rep_ok and rep_eq_eager and bool(passes["A"][0][0]["lp"])
    replayed_in_B = (steps.get("B") or {}).get("replayed", 0)
    print(json.dumps({
        "passed": bool(passed and replayed_in_B > 0),
        "compared": "token ids + top-20 logprobs, bitwise, ONE process: BCG replay (B, B2) vs "
                    "eager (A, A2) on the same engine; %d outputs per pass over groups 1x300, "
                    "1x1000, 1x2000, 1x8000 (eager>bucket), 2/3/4/5/6x300, 2x1000, 4x1000; "
                    "repeated replay 25 calls (5 warmup + 20) of 1x300 and 4x300, every call "
                    "== first == eager" % n_out,
        "comparisons": cmp, "repeat_replay_all_equal_first": rep_ok,
        "repeat_replay_equal_eager": rep_eq_eager, "context_steps_per_pass": steps,
        "flashinfer_host_prefill_plan": host_plan, "overlay": overlay()}), flush=True)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "time"
    if m == "time":
        mode_time()
    elif m == "arm":
        mode_arm(sys.argv[2], sys.argv[3])
    elif m == "inproc":
        mode_inproc()
    else:
        mode_check()
