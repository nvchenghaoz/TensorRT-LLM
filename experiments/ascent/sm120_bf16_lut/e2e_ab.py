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


def bits_equal(a, b):
    import torch
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def heavy_tailed_check():
    """Every pinned token range of every routed shape, on heavy-tailed data randn * 2^U(-8, 8) (activations and
    weights), 2 seeds: trtllm::cublas_mm (table) must give F.linear's exact bits. Tested M: both ends of every
    routed range plus the served token counts inside them."""
    import torch
    import torch.nn.functional as F
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    served = set()
    for isl in (300, 1000, 2000, 4000, 8000):
        served |= {isl * k for k in range(1, 8192 // isl + 1)}
    res = {"cases": 0, "bitwise": 0, "failures": []}
    for (K, N), ranges in mg._SM120_BF16_CUBLAS_MM_ROUTES.items():
        gate = mg._CublasMmTokenGate(ranges)
        ms = sorted({m for lo, hi in ranges for m in (lo, hi)} | {m for m in served if m in gate})
        for seed in (0, 1):
            g = torch.Generator(device="cuda").manual_seed(1000 + seed)

            def heavy(r, c):
                t = torch.randn(r, c, device="cuda", dtype=torch.bfloat16, generator=g)
                s = torch.empty(r, c, device="cuda", dtype=torch.bfloat16).uniform_(-8, 8, generator=g)
                return (t * torch.exp2(s)).contiguous()

            W = heavy(N, K)
            X = heavy(max(ms), K)
            for m in ms:
                a = F.linear(X[:m], W)
                b = torch.ops.trtllm.cublas_mm(X[:m], W.t(), None, None)
                res["cases"] += 1
                if bits_equal(a, b):
                    res["bitwise"] += 1
                else:
                    res["failures"].append([K, N, m, seed, int((a.view(torch.int16) != b.view(torch.int16)).sum())])
            del W, X
            torch.cuda.empty_cache()
        log(f"heavy-tailed {K}x{N}: {len(ms)} M x 2 seeds done, failures so far {len(res['failures'])}")
    res["ok"] = res["cases"] > 0 and not res["failures"]
    log("heavy-tailed check " + json.dumps(res))
    return res


def real_activation_check(llm, pf, params):
    """Capture the real inputs of qkv_proj / o_proj at layers 0, 30 and 59 while main's path runs (table off) one
    isl8k and one isl1k prompt; then on those activations (full rows, and row prefixes at every routed served
    token count) trtllm::cublas_mm must give F.linear's exact bits."""
    import torch
    import torch.nn.functional as F
    import tensorrt_llm._torch.modules.linear as lin
    cls = lin.UnquantizedLinearMethod
    orig_apply = cls.apply
    state = {"on": False, "calls": 0, "caps": []}
    want = (0, 30, 59)

    def apply(self, module, input, bias, weight=None):
        if state["on"] and getattr(module, "custom_cublas_mm_tokens", None) is not None and input.dim() == 2:
            k = state["calls"]
            state["calls"] += 1
            if k // 2 in want:
                state["caps"].append((state["isl"], k // 2, "qkv" if k % 2 == 0 else "o_proj",
                                      input.detach().to("cpu", copy=True), module))
        return orig_apply(self, module, input, bias, weight)

    cls.apply = apply
    set_flag(False)
    try:
        for isl in (8000, 1000):
            state.update(on=True, calls=0, isl=isl)
            run_batch(llm, [pf.make(isl)], params)
            state["on"] = False
            log(f"real activations: isl{isl} captured, {state['calls']} routed-projection calls in the forward")
    finally:
        cls.apply = orig_apply
        set_flag(True)
    served = set()
    for isl in (300, 1000, 2000, 4000, 8000):
        served |= {isl * k for k in range(1, 8192 // isl + 1)}
    res = {"captures": len(state["caps"]), "cases": 0, "bitwise": 0, "failures": [], "rows": []}
    for isl, layer, which, xin, module in state["caps"]:
        x = xin.to("cuda")
        W = module.weight
        gate = module.custom_cublas_mm_tokens
        ms = sorted({x.shape[0]} | {m for m in served if m <= x.shape[0] and m in gate})
        n_ok = 0
        n = 0
        for m in ms:
            if m not in gate:
                continue
            a = F.linear(x[:m], W)
            b = torch.ops.trtllm.cublas_mm(x[:m], W.t(), None, None)
            n += 1
            if bits_equal(a, b):
                n_ok += 1
            else:
                res["failures"].append([isl, layer, which, m, int((a.view(torch.int16) != b.view(torch.int16)).sum())])
        res["cases"] += n
        res["bitwise"] += n_ok
        res["rows"].append([isl, layer, which, list(W.shape), int(x.shape[0]), n_ok, n])
        del x
    torch.cuda.empty_cache()
    res["ok"] = res["captures"] == 12 and res["cases"] > 0 and not res["failures"]
    log("real-activation check " + json.dumps(res))
    return res


def mode_stress():
    heavy_tailed_first = heavy_tailed_check()
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
    real = real_activation_check(llm, pf, params)
    heavy = heavy_tailed_first
    print(json.dumps({"passed": bool(on_eq and off_eq and has_lp and real["ok"] and heavy["ok"]),
                      "real_activations": real, "heavy_tailed": heavy,
                      "compared": "first-token ids + top-20 logprobs, bitwise, order A(off) X(on) A'(off) in one "
                                  "process; off == main's F.linear path, on == SM120 cuBLASLt range table",
                      "prompts": n_prompts, "X_vs_A": tot["on0"], "Aprime_vs_A": tot["off1"],
                      "on_vs_off_bitwise": on_eq, "off_vs_off_bitwise": off_eq, "logprobs_present": has_lp,
                      "comparisons": cmp}), flush=True)


_TEXT = ("The river bends twice before it reaches the old mill, where the water slows and turns green under the "
         "willows. In spring the farmers bring their grain down the valley road, and the miller weighs each sack "
         "on a scale his grandfather built from oak and iron. Children wait by the wheel to watch it turn, counting "
         "the buckets as they rise and spill. Travellers who stop at the inn hear stories of floods and dry summers, "
         "of a bridge that washed away and was rebuilt stone by stone, and of the night the bells rang without a "
         "hand to pull them. ")


def natural_prompts(tok, isl, n, tag):
    out = []
    for i in range(n):
        ids = tok.encode(f"Story {tag}-{i}: " + _TEXT * (isl // 60 + 4), add_special_tokens=False)
        out.append(ids[:isl])
    return out


def mode_resend():
    """Ruling 1055 (ii): own natural-text prompts, one process, no reset. With the table ON: send the groups
    (c1 and c4 for isl8k and isl1k), then re-send them in the order 8k, 1k, 8k and compare token ids + top-20
    logprobs bitwise with the first send. Then the same with the table OFF, and ON vs OFF on the first sends."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    llm = build_llm(batch_wait_timeout_ms=1500.0)
    params = sp(logprobs=20)
    groups = {
        "isl8k@1": [[p] for p in natural_prompts(tok, 8000, 2, "8k1")],
        "isl8k@4": [natural_prompts(tok, 8000, 4, "8k4")],
        "isl1k@1": [[p] for p in natural_prompts(tok, 1000, 2, "1k1")],
        "isl1k@4": [natural_prompts(tok, 1000, 4, "1k4")],
    }
    order8 = ["isl8k@1", "isl8k@4"]
    order1 = ["isl1k@1", "isl1k@4"]

    def send(names):
        return {n: [summarize(o) for b in groups[n] for o in run_batch(llm, b, params)[1]] for n in names}

    res = {}
    first = {}
    for flag in (True, False):
        set_flag(flag)
        tag = "on" if flag else "off"
        first[tag] = send(order8 + order1)
        diffs = {}
        for k, names in enumerate([order8, order1, order8]):
            again = send(names)
            for n in names:
                d = sum(1 for a, b in zip(first[tag][n], again[n]) if a != b)
                diffs[f"resend{k}/{n}"] = {"n": len(again[n]), "n_diff": d}
        res[tag] = diffs
        log(f"table {tag}: " + json.dumps(diffs))
    set_flag(True)
    onoff = {n: sum(1 for a, b in zip(first["on"][n], first["off"][n]) if a != b) for n in first["on"]}
    ok = all(v["n_diff"] == 0 for d in res.values() for v in d.values()) and all(v == 0 for v in onoff.values())
    print(json.dumps({"passed": bool(ok), "compared": "own natural-text prompts; re-sends 8k,1k,8k in one process "
                      "without reset vs first send (token ids + top-20 logprobs, bitwise), table on and off; "
                      "first sends on vs off", "resend": res, "on_vs_off_first_send_diffs": onoff}), flush=True)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "time"
    if m == "time":
        mode_time()
    elif m == "resend":
        mode_resend()
    else:
        mode_stress()
