#!/usr/bin/env python3
"""SM120 BF16 cuBLASLt range table (idea 4): in-process end-to-end A/B, adapted from experiments/ascent/fi_plan_ab.py.

Modes (argv[1]):
  time    : step-time pair evidence (run_gpu_pair, base main vs branch). Per case (isl@conc) the wall time of
            llm.generate() over `conc` fresh unique prompts of `isl` tokens, 1 greedy output token. Median of
            ROUNDS (default 25) timed rounds after 3 warmup rounds, p10/p90 spread as the noise floor.
  stress  : bitwise model-level gate. One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), batch_wait_timeout_ms=1500.
            The table is switched off/on/off/on between passes by patching
            tensorrt_llm._torch.modules.linear._custom_cublas_mm_covers (off: every projection takes F.linear,
            i.e. main's path). Prompt sets: isl300@1 (10 prompts, one at a time), isl1k@4 (8 prompts, 2 batches),
            isl8k@1 (4, one at a time), isl8k@4 (4 in one batch), and a packed mix (64 x 300, 16 x 1k, 8 x 2k,
            4 x 4k, 2 x 8k submitted together). First-token ids and top-20 logprobs must be bitwise equal
            on vs off, and off vs off.
Engine: in-process LLM, FP8 KV, block reuse off, max_batch_size 64, max_num_tokens 8192 (as fi_plan_ab.py).
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import numpy as np  # noqa: E402

CKPT = os.environ.get("CKPT") or "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/models/Gemma-4-31B-IT-NVFP4"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Prompts:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def make(self, isl):
        return [int(x) for x in self.rng.integers(1000, 200000, size=isl)]


def build_llm(**extra):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    return LLM(model=CKPT,
               kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False, free_gpu_memory_fraction=0.8),
               max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, **extra)


_ORIG = {}


def set_flag(on):
    import tensorrt_llm._torch.modules.linear as lin
    if "covers" not in _ORIG:
        _ORIG["covers"] = lin._custom_cublas_mm_covers
    lin._custom_cublas_mm_covers = _ORIG["covers"] if on else (lambda module, input: False)


def sp(logprobs=None):
    from tensorrt_llm import SamplingParams
    kw = dict(max_tokens=1, temperature=0.0, ignore_eos=True)
    if logprobs:
        kw["logprobs"] = logprobs
    return SamplingParams(**kw)


def run_batch(llm, prompts, params):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
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
    llm = build_llm()
    params = sp()
    pf = Prompts(1234)
    cases = [("isl300@1", 300, 1), ("isl300@4", 300, 4), ("isl300@16", 300, 16), ("isl1k@1", 1000, 1),
             ("isl1k@4", 1000, 4), ("isl2k@4", 2000, 4), ("isl4k@1", 4000, 1), ("isl8k@1", 8000, 1),
             ("isl8k@4", 8000, 4)]
    log("warmup")
    for _, isl, c in cases:
        for _ in range(3):
            run_batch(llm, [pf.make(isl) for _ in range(c)], params)
    res = {}
    n = int(os.environ.get("ROUNDS", "25"))
    for name, isl, c in cases:
        ts = []
        for _ in range(n):
            dt, _ = run_batch(llm, [pf.make(isl) for _ in range(c)], params)
            ts.append(dt * 1e3)
        ts.sort()
        med = statistics.median(ts)
        p10 = ts[int(0.1 * (len(ts) - 1))]
        p90 = ts[int(round(0.9 * (len(ts) - 1)))]
        res[name] = {"step_ms": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
                     "noise_floor": round((p90 - p10) / med, 4), "n": len(ts)}
        log(f"{name}: median {med:.2f} ms p10 {p10:.2f} p90 {p90:.2f}")
    print(json.dumps({"batch": res, "output": {"class": "exact"}}), flush=True)


def mode_stress():
    llm = build_llm(batch_wait_timeout_ms=1500.0)
    pf = Prompts(777)
    params = sp(logprobs=20)
    # 100 fresh prompts: ISL 300/1k/2k/4k/8k x c1/c4, 10 each (c4: batches of 4, 4, 2), plus a packed mix.
    sets = {}
    for isl in (300, 1000, 2000, 4000, 8000):
        sets[f"isl{isl}@1"] = [[pf.make(isl)] for _ in range(10)]
        ps = [pf.make(isl) for _ in range(10)]
        sets[f"isl{isl}@4"] = [ps[0:4], ps[4:8], ps[8:10]]
    groups = [(300, 64), (1000, 16), (2000, 8), (4000, 4), (8000, 2)]
    lists = [[pf.make(isl) for _ in range(n)] for isl, n in groups]
    mixed = []
    i = 0
    while any(lists):
        lst = lists[i % len(lists)]
        if lst:
            mixed.append(lst.pop())
        i += 1
    sets["mixed"] = [mixed]
    n_prompts = sum(len(b) for bs in sets.values() for b in bs)
    log(f"stress sets: {n_prompts} prompts")
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        run_batch(llm, [warm.make(isl) for _ in range(2)], params)
    passes = {}
    for tag, flag in [("off0", False), ("on0", True), ("off1", False)]:
        set_flag(flag)
        t0 = time.time()
        passes[tag] = {name: [summarize(o) for b in batches for o in run_batch(llm, b, params)[1]]
                       for name, batches in sets.items()}
        log(f"pass {tag} flag={flag} {time.time() - t0:.1f}s")
    set_flag(True)
    ref = passes["off0"]

    def top1(x):
        if x["lp"] and isinstance(x["lp"][0], list) and x["lp"][0]:
            return max(x["lp"][0], key=lambda kv: kv[1])
        return None

    cmp = {}
    tot = {}
    for tag in ("on0", "off1"):
        t = {"n": 0, "bitwise_equal": 0, "token_flips": 0, "dtop1_gt_0125": 0, "max_dtop1": 0.0, "max_dlogprob": 0.0}
        for name in sets:
            a, b = ref[name], passes[tag][name]
            eq = sum(1 for x, y in zip(a, b) if x == y)
            flips = sum(1 for x, y in zip(a, b) if x["ids"] != y["ids"])
            big = 0
            dmax_top1 = 0.0
            dmax = 0.0
            for x, y in zip(a, b):
                tx, ty = top1(x), top1(y)
                if tx and ty:
                    dd = abs(tx[1] - ty[1])
                    dmax_top1 = max(dmax_top1, dd)
                    big += dd > 0.125
                if x["lp"] and y["lp"] and isinstance(x["lp"][0], list):
                    dx, dy = dict(x["lp"][0]), dict(y["lp"][0])
                    for k in set(dx) & set(dy):
                        dmax = max(dmax, abs(dx[k] - dy[k]))
            cmp[f"{tag}_vs_off0/{name}"] = {"n": len(a), "bitwise_equal": eq, "token_flips": flips,
                                            "dtop1_gt_0125": big, "max_dtop1": dmax_top1, "max_dlogprob": dmax}
            t["n"] += len(a)
            t["bitwise_equal"] += eq
            t["token_flips"] += flips
            t["dtop1_gt_0125"] += big
            t["max_dtop1"] = max(t["max_dtop1"], dmax_top1)
            t["max_dlogprob"] = max(t["max_dlogprob"], dmax)
        tot[tag] = t
    log(json.dumps(cmp))
    log("totals " + json.dumps(tot))
    on_eq = tot["on0"]["bitwise_equal"] == tot["on0"]["n"]
    off_eq = tot["off1"]["bitwise_equal"] == tot["off1"]["n"]
    has_lp = top1(ref["isl300@1"][0]) is not None
    print(json.dumps({"passed": bool(on_eq and off_eq and has_lp),
                      "compared": "first-token ids + top-20 logprobs, bitwise, order A(off) X(on) A'(off) in one "
                                  "process; off == main's F.linear path, on == SM120 cuBLASLt range table",
                      "prompts": n_prompts, "X_vs_A": tot["on0"], "Aprime_vs_A": tot["off1"],
                      "on_vs_off_bitwise": on_eq, "off_vs_off_bitwise": off_eq, "logprobs_present": has_lp,
                      "comparisons": cmp}), flush=True)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "time"
    if m == "time":
        mode_time()
    else:
        mode_stress()
