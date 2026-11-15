#!/usr/bin/env python3
"""FlashInfer prefill-plan sync removal (idea 13): in-process A/B, stress and profiling driver.

Modes (argv[1]):
  time    : step-time pair evidence. Per case (isl@conc) the wall time of llm.generate() over `conc`
            fresh unique prompts of `isl` tokens, 1 greedy output token. Median of >=20 timed rounds after
            warmup, p10/p90 spread as noise floor. Prints the evidence JSON as the last line.
  stress  : bitwise check. One process (TLLM_WORKER_USE_SINGLE_PROCESS=1) runs the same mixed prompt
            set (c64 x isl300, c16 x isl1k, 8 x 2k, 4 x 4k, 2 x 8k, interleaved, submitted together so
            the overlap scheduler runs back-to-back context steps with changing composition) four times,
            alternating flashinfer._HOST_PREFILL_PLAN off/on/off/on, plus a c1 repeat subset. Every
            pass's (token, logprob, top-k logprobs) per prompt must be bitwise equal across passes.
  prof    : nsys driver: warmup, then cudaProfilerStart/Stop around isl300@4 and isl1k@4 rounds with
            the flag given by argv[2] (on/off).
No model numerics are changed by the flag; the engine settings below are fixed for both arms.
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


def print_dims():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "global_head_dim", "num_global_key_value_heads", "sliding_window", "intermediate_size"]
    log("config: " + json.dumps({k: tc.get(k) for k in keys}))
    try:
        q = json.load(open(os.path.join(CKPT, "hf_quant_config.json")))
        log("quant: " + json.dumps(q.get("quantization", q))[:400])
    except OSError:
        pass


class Prompts:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def make(self, isl):
        return [int(x) for x in self.rng.integers(1000, 200000, size=isl)]


def build_llm(**extra):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    llm = LLM(model=CKPT,
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, **extra)
    return llm


def set_flag(on):
    try:
        import tensorrt_llm._torch.attention.backends.flashinfer as fi
        fi._HOST_PREFILL_PLAN = bool(on)
    except Exception as e:  # noqa: BLE001
        log(f"flag set failed: {e}")


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
    print_dims()
    llm = build_llm()
    params = sp()
    pf = Prompts(1234)
    cases = [("isl300@1", 300, 1), ("isl300@4", 300, 4), ("isl1k@1", 1000, 1), ("isl1k@4", 1000, 4),
             ("isl2k@4", 2000, 4), ("isl8k@1", 8000, 1), ("isl300@16", 300, 16)]
    log("warmup")
    for _, isl, c in cases:
        for _ in range(3):
            run_batch(llm, [pf.make(isl) for _ in range(c)], params)
    res = {}
    n = int(os.environ.get("ROUNDS", "25"))
    for name, isl, c in cases:
        ts = []
        for i in range(n):
            ps = [pf.make(isl) for _ in range(c)]
            dt, _ = run_batch(llm, ps, params)
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
    print_dims()
    # Arrival timing decides the first step's batch (job 138: off vs off differed on prompt 0 only).
    # Waiting for max_batch_size requests makes the batch composition repeatable across passes.
    llm = build_llm(batch_wait_timeout_ms=1500.0)
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
    rep = mixed[:6]
    log(f"stress set: {len(mixed)} prompts, {sum(len(p) for p in mixed)} tokens")
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        run_batch(llm, [warm.make(isl) for _ in range(2)], params)
    passes = {}
    order = [("off0", False), ("on0", True), ("off1", False), ("on1", True)]
    for tag, flag in order:
        set_flag(flag)
        dt, outs = run_batch(llm, mixed, params)
        passes[tag] = [summarize(o) for o in outs]
        # repeated execution: the first 6 prompts re-sent alone, back to back
        r = []
        for p in rep:
            _, o = run_batch(llm, [p], params)
            r.append(summarize(o[0]))
        passes[tag + "_rep"] = r
        log(f"pass {tag} flag={flag} {dt:.2f}s first={passes[tag][0]['ids']}")
    set_flag(True)
    cmp = {}
    ref = passes["off0"]
    for tag in ("on0", "off1", "on1"):
        diff = [k for k, (a, b) in enumerate(zip(ref, passes[tag])) if a != b]
        cmp[tag + "_vs_off0"] = {"n_diff": len(diff), "first": diff[:5]}
    rref = passes["off0_rep"]
    for tag in ("on0_rep", "off1_rep", "on1_rep"):
        diff = [k for k, (a, b) in enumerate(zip(rref, passes[tag])) if a != b]
        cmp[tag + "_vs_off0_rep"] = {"n_diff": len(diff), "first": diff[:5]}
    on_eq = cmp["on0_vs_off0"]["n_diff"] == 0 and cmp["on1_vs_off0"]["n_diff"] == 0
    rep_eq = cmp["on0_rep_vs_off0_rep"]["n_diff"] == 0 and cmp["on1_rep_vs_off0_rep"]["n_diff"] == 0
    has_lp = bool(ref[0]["lp"])
    log(json.dumps(cmp))
    print(json.dumps({"passed": bool(on_eq and rep_eq and has_lp), "compared": "token ids + top-20 logprobs, "
                      "bitwise, host-plan on vs off (sync device plan, == main path) in one process",
                      "prompts": len(mixed), "repeat_prompts": len(rep), "comparisons": cmp,
                      "off_vs_off_equal": cmp["off1_vs_off0"]["n_diff"] == 0,
                      "logprobs_present": has_lp}), flush=True)


def mode_prof(flag):
    import torch
    llm = build_llm()
    set_flag(flag)
    params = sp()
    pf = Prompts(99)
    for isl, c in ((300, 4), (1000, 4), (8000, 1)):
        for _ in range(3):
            run_batch(llm, [pf.make(isl) for _ in range(c)], params)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    t = {}
    for isl, c in ((300, 4), (1000, 4), (300, 1)):
        ts = []
        for _ in range(4):
            dt, _ = run_batch(llm, [pf.make(isl) for _ in range(c)], params)
            ts.append(dt * 1e3)
        t[f"isl{isl}@{c}"] = ts
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    log(json.dumps(t))
    print(json.dumps({"passed": True, "flag": flag, "latency_ms": t}), flush=True)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "time"
    if m == "time":
        mode_time()
    elif m == "stress":
        mode_stress()
    else:
        mode_prof(len(sys.argv) < 3 or sys.argv[2] != "off")
