#!/usr/bin/env python3
"""Gemma4 fused QKV prep (branch 7): real-activation per-layer dump and in-process on/off A/B.

Modes (argv[1]):
  dump : in-process LLM (TLLM_WORKER_USE_SINGLE_PROCESS=1, CUDA graphs off for this debug pass).
         Gemma4Attention.apply_rope is wrapped: on every call where the fused prep is eligible it runs
         main's unfused chain (split views -> v_norm -> apply_qk_norm -> flashinfer RoPE, then the
         backend's .to(fp8) of K/V) and the fused kernel in both reduction orders ("old" = tl.sum,
         "new" = cute_norm_order) on the SAME qkv and position_ids, byte-compares Q/K/V, and RETURNS
         THE UNFUSED RESULT so the trajectory stays main's. Covers c=1 prompts, packed batches
         (16x300, 4x2000, uneven lengths), and decode steps (max_tokens=8).
  ab   : in-process A/B with main's default engine settings (CUDA graphs on) and batch_wait_timeout_ms.
         modeling_gemma4._FUSED_QKV_PREP flips between passes off, on, off, on; 200 fresh random-token
         prompts (20 per ISL in {300,1000,2000,4000,8000} at c=1, 20 per ISL at c=4) plus chat-template
         prompts and a mixed batch; then an off/on pair with flashinfer._HOST_PREFILL_PLAN=False.
         Compares first-token ids + top-20 logprobs bitwise.
Last stdout line: {"passed": ..., ...}.
"""
import json
import os
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
# Host prefill plan flag: main's default (1) unless the job sets it; read once at flashinfer import.
os.environ.setdefault("TRTLLM_FI_HOST_PREFILL_PLAN", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

CKPT = os.environ.get("CKPT") or (sys.argv[2] if len(sys.argv) > 2 else None)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def print_dims():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "global_head_dim", "num_global_key_value_heads", "sliding_window", "rms_norm_eps",
            "attention_k_eq_v"]
    log("config: " + json.dumps({k: tc.get(k) for k in keys}))
    lt = tc.get("layer_types") or []
    log(f"layer types: {sum(t == 'sliding_attention' for t in lt)} sliding, "
        f"{sum(t == 'full_attention' for t in lt)} full")


class Prompts:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def make(self, isl):
        return [int(x) for x in self.rng.integers(1000, 200000, size=isl)]


_CORPUS = {}


def _corpus_ids(tok):
    """Deterministic natural text: installed packages' METADATA (README) text, sorted by name."""
    if "ids" not in _CORPUS:
        import importlib.metadata as md
        texts = []
        seen = set()
        for d in sorted(md.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
            name = (d.metadata.get("Name") or "").lower()
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                t = d.read_text("METADATA") or ""
            except Exception:  # noqa: BLE001
                t = ""
            body = t.split("\n\n", 1)[1] if "\n\n" in t else ""
            if len(body) > 2000:
                texts.append(body)
        text = "\n\n".join(texts)
        _CORPUS["ids"] = tok(text, add_special_tokens=False)["input_ids"]
        log(f"natural-text corpus: {len(texts)} package READMEs, {len(_CORPUS['ids'])} tokens")
    return _CORPUS["ids"]


class NaturalPrompts:
    """Natural-text prompts in the tokenizer's chat template, cut to an exact ISL (top-lead ruling 914-1)."""

    def __init__(self, seed):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(CKPT)
        self.rng = np.random.default_rng(seed)
        mark = "@@BODY@@"
        s = self.tok.apply_chat_template([{"role": "user", "content": mark}], add_generation_prompt=True,
                                         tokenize=False)
        pre, suf = s.split(mark)
        self.pre = self.tok(pre, add_special_tokens=False)["input_ids"]
        self.suf = self.tok(suf, add_special_tokens=False)["input_ids"]
        self.ids = _corpus_ids(self.tok)

    def make(self, isl):
        n = isl - len(self.pre) - len(self.suf)
        off = int(self.rng.integers(0, len(self.ids) - n))
        return [int(x) for x in self.pre + self.ids[off:off + n] + self.suf]


def build_llm(**extra):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    return LLM(model=CKPT,
               kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False, free_gpu_memory_fraction=0.8),
               max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, **extra)


def sp(logprobs=20, max_tokens=1):
    from tensorrt_llm import SamplingParams
    kw = dict(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    if logprobs:
        kw["logprobs"] = logprobs
    return SamplingParams(**kw)


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


def top1(s):
    """(token, logprob) of the first generated token."""
    lp = s["lp"][0] if s["lp"] else []
    tok = s["ids"][0]
    v = [l for t, l in lp if t == tok] if isinstance(lp, list) else []
    return tok, (v[0] if v else None)


# ----------------------------------------------------------------------------- dump
def _bf16_ulps(a, b):
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai)  # sign-magnitude -> monotone line
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs()


def _fp8_ulps(a, b):
    ai = a.view(torch.uint8).to(torch.int32)
    bi = b.view(torch.uint8).to(torch.int32)
    ai = torch.where(ai >= 128, 128 - ai, ai)
    bi = torch.where(bi >= 128, 128 - bi, bi)
    return (ai - bi).abs()


def mode_dump():
    print_dims()
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant as fused
    orig = mg.Gemma4Attention.apply_rope
    stats = {}
    info = {}
    calls = {"eligible": 0, "ineligible": 0, "capturing": 0, "rows_hist": {}}
    first_bad = []

    def compare(layer, var, nm, c, m, pos, hd):
        e = stats.setdefault(layer, {}).setdefault(var, {}).setdefault(
            nm, {"mismatch": 0, "total": 0, "max_ulp": 0, "nonfinite": 0, "calls": 0})
        assert c.shape == m.shape, (c.shape, m.shape)
        if c.dtype == torch.bfloat16:
            ne = (c.contiguous().view(torch.int16) != m.contiguous().view(torch.int16))
            ul = _bf16_ulps(c.contiguous(), m.contiguous())
        else:
            ne = (c.contiguous().view(torch.uint8) != m.contiguous().view(torch.uint8))
            ul = _fp8_ulps(c.contiguous(), m.contiguous())
        n = int(ne.sum().item())
        e["mismatch"] += n
        e["total"] += ne.numel()
        e["calls"] += 1
        e["max_ulp"] = max(e["max_ulp"], int(ul.max().item()) if ul.numel() else 0)
        e["nonfinite"] += int((~torch.isfinite(c.float())).sum().item())
        if n and len(first_bad) < 40:
            r, col = ne.view(ne.shape[0], -1).nonzero()[0].tolist()
            first_bad.append({"layer": layer, "var": var, "t": nm, "row": r, "head": col // hd, "col": col % hd,
                              "pos": int(pos.view(-1)[r].item()), "rows": int(c.shape[0]),
                              "cand": float(c.reshape(c.shape[0], -1)[r, col].float()),
                              "main": float(m.reshape(m.shape[0], -1)[r, col].float())})

    def hooked(self, q, k, v, position_ids):
        if torch.cuda.is_current_stream_capturing():
            calls["capturing"] += 1
            return orig(self, q, k, v, position_ids)
        eligible = (k is None and v is None and position_ids is not None and q.dtype == torch.bfloat16
                    and not self.is_kv_shared and not self.fuse_qk_norm_rope and not self._fused_prep_blocked
                    and self._fused_qkv_prep_enabled())
        li = int(self.layer_idx)
        if li not in info:
            info[li] = {"sliding": self.attention_window_size is not None, "qk_only_proj": bool(self._qk_only_proj),
                        "fused_eligible": bool(eligible), "head_dim": self.head_dim,
                        "backend": getattr(self.attn, "flashinfer_backend", None),
                        "q_out_fp8": bool(self._fused_qkv_prep_out_fp8),
                        "kv_out_fp8": bool(self._fused_qkv_prep_kv_out_fp8),
                        "cute_norm": bool(getattr(self, "_fused_qkv_prep_cute_norm", False)),
                        "cos_sin": list(self.rotary_emb.rotary_cos_sin.shape) if self.rotary_emb is not None
                        else None, "qkv_width": int(q.shape[-1])}
        if not eligible:
            calls["ineligible"] += 1
            return orig(self, q, k, v, position_ids)
        calls["eligible"] += 1
        rh = calls["rows_hist"]
        rh[str(q.shape[0])] = rh.get(str(q.shape[0]), 0) + 1
        qkv = q
        snap = qkv.clone()
        mg._FUSED_QKV_PREP = False
        try:
            uq, uk, uv = orig(self, qkv, None, None, position_ids)
        finally:
            mg._FUSED_QKV_PREP = True
        assert torch.equal(snap.view(torch.int16), qkv.view(torch.int16)), "unfused chain mutated qkv"
        nkv, hd = self.num_key_value_heads, self.head_dim
        uk8 = uk.reshape(-1, nkv, hd).to(torch.float8_e4m3fn).reshape(uk.shape[0], -1)
        uv8 = uv.reshape(-1, nkv, hd).to(torch.float8_e4m3fn).reshape(uv.shape[0], -1)
        kx = {"k_eq_v": True} if self._qk_only_proj else {}
        for var, cute in (("old", False), ("new", True)):
            fq, fk, fv = fused(qkv, position_ids, self.rotary_emb.rotary_cos_sin, self.q_norm.weight,
                               self.k_norm.weight, self.q_norm.variance_epsilon, self.num_heads, nkv, hd,
                               out_fp8=self._fused_qkv_prep_out_fp8, packed_output=False,
                               kv_out_fp8=self._fused_qkv_prep_kv_out_fp8, cute_norm_order=cute, **kx)
            assert fq.dtype == torch.bfloat16 and fk.dtype == fv.dtype == torch.float8_e4m3fn
            compare(li, var, "q", fq, uq, position_ids, hd)
            compare(li, var, "k", fk, uk8, position_ids, hd)
            compare(li, var, "v", fv, uv8, position_ids, hd)
        assert torch.equal(snap.view(torch.int16), qkv.view(torch.int16)), "fused kernel mutated qkv"
        return uq, uk, uv

    mg.Gemma4Attention.apply_rope = hooked
    llm = build_llm(cuda_graph_config=None, batch_wait_timeout_ms=500.0)
    pf = NaturalPrompts(4242)
    params = sp(logprobs=20)
    cases = [("isl300@1#a", [300]), ("isl300@1#b", [300]), ("isl300@1#c", [300]), ("isl300@1#d", [300]),
             ("isl1k@1", [1000]), ("isl8k@1", [8000]), ("16x300", [300] * 16), ("4x2000", [2000] * 4),
             ("uneven", [333, 1201, 77, 2048, 513, 4097]), ("isl8k@4", [8000] * 4), ("isl1k@4", [1000] * 4),
             ("isl1k@8", [1000] * 8)]
    t0 = time.time()
    for name, isls in cases:
        outs = llm.generate([pf.make(i) for i in isls], sampling_params=params, use_tqdm=False)
        log(f"dump case {name}: {len(outs)} prompts, eligible calls so far {calls['eligible']}")
    # decode steps (eager, CUDA graphs off in this pass)
    outs = llm.generate([pf.make(500), pf.make(900)], sampling_params=sp(logprobs=0, max_tokens=8), use_tqdm=False)
    log(f"decode case: tokens {[len(o.outputs[0].token_ids) for o in outs]}; calls {calls}")
    tot = {var: {t: [0, 0, 0, 0] for t in "qkv"} for var in ("old", "new")}
    for li, d in stats.items():
        for var, dd in d.items():
            for t, e in dd.items():
                tot[var][t][0] += e["mismatch"]
                tot[var][t][1] += e["total"]
                tot[var][t][2] = max(tot[var][t][2], e["max_ulp"])
                tot[var][t][3] += e["nonfinite"]
    first_layer = {}
    for var in ("old", "new"):
        bad = sorted(li for li, d in stats.items() if any(d[var][t]["mismatch"] for t in "qkv"))
        first_layer[var] = bad[0] if bad else None
    per_layer = {str(li): {var: {t: [d[var][t]["mismatch"], d[var][t]["max_ulp"]] for t in "qkv"}
                           for var in d} for li, d in sorted(stats.items())}
    summary = {var: {t: {"mismatch": tot[var][t][0], "total": tot[var][t][1],
                         "frac": tot[var][t][0] / max(1, tot[var][t][1]), "max_ulp": tot[var][t][2],
                         "nonfinite": tot[var][t][3]}
                     for t in "qkv"} for var in tot}
    layers_fused = sorted(li for li, i in info.items() if i["fused_eligible"])
    glob = sorted(li for li, i in info.items() if i["qk_only_proj"])
    by_type = {}
    for kind, lis in (("global_k_eq_v", glob), ("sliding", sorted(set(stats) - set(glob)))):
        by_type[kind] = {"layers": len([li for li in lis if li in stats]),
                         "new_mismatch_bytes": {t: sum(stats[li]["new"][t]["mismatch"] for li in lis if li in stats)
                                                for t in "qkv"},
                         "elements": {t: sum(stats[li]["new"][t]["total"] for li in lis if li in stats)
                                      for t in "qkv"}}
    log("by layer type: " + json.dumps(by_type))
    log("per-layer [mismatch, max_ulp]: " + json.dumps(per_layer))
    log("first mismatches: " + json.dumps(first_bad[:20]))
    log("layer info: " + json.dumps({str(k): v for k, v in sorted(info.items())}))
    new_zero = all(summary["new"][t]["mismatch"] == 0 for t in "qkv")
    print(json.dumps({"passed": bool(new_zero and len(layers_fused) > 0),
                      "compared": "per-layer Q (bf16) / K,V (fp8) bytes of the fused kernel vs main's unfused "
                                  "chain on the same real qkv + position_ids (unfused result returned)",
                      "summary": summary, "by_layer_type": by_type, "first_layer_with_mismatch": first_layer,
                      "fused_layers": layers_fused, "n_fused_layers": len(layers_fused),
                      "unfused_layers": sorted(li for li, i in info.items() if not i["fused_eligible"]),
                      "calls": calls, "first_mismatches": first_bad[:10], "elapsed_s": round(time.time() - t0, 1)}),
          flush=True)


# ----------------------------------------------------------------------------- A/B
def mode_ab():
    print_dims()
    import tensorrt_llm._torch.attention.backends.flashinfer as fi
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    llm = build_llm(batch_wait_timeout_ms=float(os.environ.get("BATCH_WAIT_MS", "500")))
    params = sp(logprobs=20)
    pf = Prompts(20261114)
    isls = [300, 1000, 2000, 4000, 8000]
    n_per = int(os.environ.get("N_PER", "20"))
    groups = []  # (label, [prompts]) submitted together
    for isl in isls:
        for _ in range(n_per):
            groups.append((f"isl{isl}@1", [pf.make(isl)]))
    for isl in isls:
        for _ in range(n_per // 4):
            groups.append((f"isl{isl}@4", [pf.make(isl) for _ in range(4)]))
    groups.append(("mix10x1k+4x8k", [pf.make(1000) for _ in range(10)] + [pf.make(8000) for _ in range(4)]))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CKPT)
        qs = ["What is the capital of France? Answer in one word.", "Explain why the sky is blue.",
              "Write a haiku about autumn leaves.", "List three prime numbers greater than 100.",
              "Summarize the plot of Hamlet in two sentences.", "What does a CPU cache do?",
              "Translate 'good morning' into Spanish.", "Why do cats purr?"]
        for qtext in qs:
            ids = tok.apply_chat_template([{"role": "user", "content": qtext}], add_generation_prompt=True,
                                          tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "keys"):
                ids = ids["input_ids"]
            groups.append(("chat@1", [list(map(int, ids))]))
    except Exception as e:  # noqa: BLE001
        log(f"chat prompts skipped: {e!r}")
    n_prompts = sum(len(g) for _, g in groups)
    log(f"A/B set: {len(groups)} submissions, {n_prompts} prompts")
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        llm.generate([warm.make(isl) for _ in range(2)], sampling_params=params, use_tqdm=False)

    def run_pass(tag, subset=None):
        res = []
        t0 = time.time()
        for label, ps in (groups if subset is None else subset):
            outs = llm.generate(ps, sampling_params=params, use_tqdm=False)
            res.extend((label, summarize(o)) for o in outs)
        log(f"pass {tag}: fused={mg._FUSED_QKV_PREP} host_plan={getattr(fi, '_HOST_PREFILL_PLAN', None)} "
            f"{time.time() - t0:.1f}s")
        return res

    passes = {}
    for tag, flag in (("off0", False), ("on0", True), ("off1", False), ("on1", True)):
        mg._FUSED_QKV_PREP = flag
        passes[tag] = run_pass(tag)
    sub = [g for g in groups if g[0] in ("isl300@1", "isl300@4", "isl1k@4", "isl8k@1")][:24]
    hp_ok = hasattr(fi, "_HOST_PREFILL_PLAN")
    if hp_ok:
        fi._HOST_PREFILL_PLAN = False
        for tag, flag in (("hp0_off", False), ("hp0_on", True)):
            mg._FUSED_QKV_PREP = flag
            passes[tag] = run_pass(tag, sub)
        fi._HOST_PREFILL_PLAN = True
    mg._FUSED_QKV_PREP = True

    def cmp(a, b):
        per = {}
        n_eq = n_flip = n_big = 0
        maxd = 0.0
        for (la, sa), (_, sb) in zip(a, b):
            e = per.setdefault(la, {"n": 0, "bitwise_equal": 0, "token_flips": 0, "max_abs_dlp_top1": 0.0})
            e["n"] += 1
            eq = sa == sb
            e["bitwise_equal"] += int(eq)
            n_eq += int(eq)
            ta, la1 = top1(sa)
            tb, lb1 = top1(sb)
            if ta != tb:
                e["token_flips"] += 1
                n_flip += 1
            elif la1 is not None and lb1 is not None:
                d = abs(la1 - lb1)
                e["max_abs_dlp_top1"] = max(e["max_abs_dlp_top1"], d)
                maxd = max(maxd, d)
                n_big += int(d > 0.125)
        return {"n": len(a), "bitwise_equal": n_eq, "token_flips": n_flip, "dlp_gt_0.125": n_big,
                "max_abs_dlp_top1": maxd, "per_label": per}

    res = {"off1_vs_off0": cmp(passes["off0"], passes["off1"]),
           "on0_vs_off0": cmp(passes["off0"], passes["on0"]),
           "on1_vs_off0": cmp(passes["off0"], passes["on1"]),
           "on1_vs_on0": cmp(passes["on0"], passes["on1"])}
    if hp_ok:
        res["hostplan0_on_vs_off"] = cmp(passes["hp0_off"], passes["hp0_on"])
    for k, v in res.items():
        log(f"{k}: " + json.dumps({kk: vv for kk, vv in v.items() if kk != "per_label"}))
    n = res["on0_vs_off0"]["n"]
    det = res["off1_vs_off0"]["bitwise_equal"] == n
    exact = all(res[k]["bitwise_equal"] == res[k]["n"] for k in res)
    print(json.dumps({"passed": bool(exact and det and passes["off0"][0][1]["lp"]),
                      "compared": "first-token ids + top-20 logprobs, bitwise, fused QKV prep on vs off "
                                  "(modeling_gemma4._FUSED_QKV_PREP) in one process, passes off,on,off,on "
                                  "+ host-plan-off pair",
                      "prompts": n, "off_vs_off_bitwise": det, "comparisons": res}), flush=True)


TOL = {300: 4.8174, 1000: 0.5713, 2000: 5.0, 4000: 0.125, 8000: 0.125}


def _first(s):
    lp = s["lp"][0] if s["lp"] else []
    d = dict(lp) if isinstance(lp, list) else {}
    return s["ids"][0], d


def _check_distance(m, c):
    """Landing-check distance (check.py): same token |dlp|; else near-tie seen from both sides."""
    tm, dm = _first(m)
    tc, dc = _first(c)
    lm, lc = dm.get(tm), dc.get(tc)
    if lm is None or lc is None:
        return float("inf")
    d = abs(lc - lm)
    if tc == tm:
        return d
    a = dm.get(tc)
    b = dc.get(tm)
    a = float("inf") if a is None else max(0.0, lm - a)
    b = float("inf") if b is None else max(0.0, lc - b)
    return max(d, a, b)


def mode_outcome0():
    """Landing-3 diagnosis: OLD (tl.sum order, reordered) fused prep vs main's chain, host plan off/on."""
    print_dims()
    import tensorrt_llm._torch.attention.backends.flashinfer as fi
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    real = mg.gemma4_fused_qkv_norm_rope_quant
    variant = {"cute": True}

    def wrapped(*a, **kw):
        kw["cute_norm_order"] = bool(variant["cute"] and kw.get("cute_norm_order", False))
        return real(*a, **kw)

    mg.gemma4_fused_qkv_norm_rope_quant = wrapped
    llm = build_llm(batch_wait_timeout_ms=float(os.environ.get("BATCH_WAIT_MS", "1500")))
    params = sp(logprobs=20)
    pf = Prompts(774)
    groups = []
    for isl in (300, 1000, 8000):
        for _ in range(10):
            groups.append((isl, f"isl{isl}@1", [pf.make(isl)]))
    groups.append((1000, "isl1k@16", [pf.make(1000) for _ in range(16)]))
    for _ in range(3):
        groups.append((8000, "isl8k@4", [pf.make(8000) for _ in range(4)]))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CKPT)
        base = ("The history of the printing press shows how a single technical change can reshape "
                "how knowledge spreads, who can read, and which institutions hold authority. ")
        for i, n in enumerate((20, 60, 20, 60, 20, 60, 20, 60)):
            ids = tok.apply_chat_template([{"role": "user", "content": base * (n // 20) +
                                            f"Question {i}: summarize the main idea in one sentence."}],
                                          add_generation_prompt=True, tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "keys"):
                ids = ids["input_ids"]
            ids = list(map(int, ids))
            groups.append((300 if len(ids) < 650 else 1000, "chat@1", [ids]))
    except Exception as e:  # noqa: BLE001
        log(f"chat prompts skipped: {e!r}")
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        llm.generate([warm.make(isl) for _ in range(2)], sampling_params=params, use_tqdm=False)

    def run(tag, fused, cute, hp):
        mg._FUSED_QKV_PREP = fused
        variant["cute"] = cute
        fi._HOST_PREFILL_PLAN = hp
        res = []
        t0 = time.time()
        for isl, label, ps in groups:
            outs = llm.generate(ps, sampling_params=params, use_tqdm=False)
            res.extend((isl, label, summarize(o)) for o in outs)
        log(f"arm {tag}: fused={fused} cute={cute} host_plan={hp} {time.time() - t0:.1f}s")
        return res

    arms = [("main@0", False, True, False), ("main@1", False, True, True), ("b7old@0", True, False, False),
            ("b7old@1", True, False, True), ("b7new@1", True, True, True), ("b7new@0", True, True, False),
            ("main@0b", False, True, False)]
    out = {tag: run(tag, f, c, h) for tag, f, c, h in arms}
    mg._FUSED_QKV_PREP, variant["cute"], fi._HOST_PREFILL_PLAN = True, True, True
    ref = out["main@0"]
    rep = {}
    for tag in [a[0] for a in arms[1:]]:
        per = {}
        for (isl, label, m), (_, _, c) in zip(ref, out[tag]):
            e = per.setdefault(label, {"n": 0, "differ": 0, "flips": 0, "flips_margin_gt_1nat": 0,
                                       "max_top20_dlp": 0.0, "check_fail": 0, "flip_margins": []})
            e["n"] += 1
            e["differ"] += int(m != c)
            tm, dm = _first(m)
            tc, dc = _first(c)
            common = set(dm) & set(dc)
            if common:
                e["max_top20_dlp"] = max(e["max_top20_dlp"], max(abs(dm[t] - dc[t]) for t in common))
            if tm != tc:
                e["flips"] += 1
                vals = sorted(dm.values(), reverse=True)
                margin = vals[0] - vals[1] if len(vals) > 1 else float("inf")
                e["flip_margins"].append(round(margin, 4))
                e["flips_margin_gt_1nat"] += int(margin > 1.0)
            e["check_fail"] += int(_check_distance(m, c) > TOL[isl])
        tot = {k: sum(v[k] for v in per.values()) for k in ("n", "differ", "flips", "flips_margin_gt_1nat",
                                                            "check_fail")}
        tot["max_top20_dlp"] = max(v["max_top20_dlp"] for v in per.values())
        rep[tag + "_vs_main@0"] = {"total": tot, "per_label": per}
        log(f"{tag} vs main@0: {json.dumps(tot)}")
    det = rep["main@0b_vs_main@0"]["total"]["differ"] == 0
    print(json.dumps({"passed": bool(det), "compared": "first token + top-20 logprobs vs main@0 (main prep, host "
                      "plan off); arms flip modeling_gemma4._FUSED_QKV_PREP, the fused kernel's cute_norm_order "
                      "(old = landing-3 reordered kernel) and flashinfer._HOST_PREFILL_PLAN in one process",
                      "main0_vs_main0_bitwise": det, "report": rep}), flush=True)


def _cmp_arms(ref, cand):
    """Per ISL x c: prompts differing (ids + top-20 logprobs), first-token flips, max top-20 |dlp|."""
    per = {}
    for (label, i, m), (_, _, c) in zip(ref, cand):
        e = per.setdefault(label, {"n": 0, "differ": 0, "flips": 0, "max_top20_dlp": 0.0, "flip_detail": []})
        e["n"] += 1
        e["differ"] += int(m != c)
        tm, dm = _first(m)
        tc, dc = _first(c)
        common = set(dm) & set(dc)
        if common:
            e["max_top20_dlp"] = max(e["max_top20_dlp"], max(abs(dm[t] - dc[t]) for t in common))
        if tm != tc:
            e["flips"] += 1
            vals = sorted(dm.values(), reverse=True)
            e["flip_detail"].append({"idx": i, "ref_tok": tm, "ref_lp": round(dm.get(tm, float("nan")), 4),
                                     "ref_margin": round(vals[0] - vals[1], 4) if len(vals) > 1 else None,
                                     "to_tok": tc, "to_lp": round(dc.get(tc, float("nan")), 4)})
    tot = {k: sum(v[k] for v in per.values()) for k in ("n", "differ", "flips")}
    tot["max_top20_dlp"] = max([v["max_top20_dlp"] for v in per.values()] or [0.0])
    return {"total": tot, "per_label": per}


def mode_routea():
    """Branch 7 Route A in-process A/B (host prefill plan off in every arm).

    LLM 1 (default engine settings, overlap scheduler on): natural-text prompts at c=1 (isl300/1k/8k),
    4 x 1k and 4 x 8k groups submitted together, a 16 x 1k group and chat prompts. Arms: main (fused prep
    off), b7 (fused prep on: 50 sliding + 10 K=V global layers), repeated. The landing-3 OLD kernel
    (tl.sum order, sliding layers only) runs on the job-246 random-id prompts (Prompts(774), isl300@1 x10,
    isl1k@1 x10) to locate its flips. LLM 2 (mode (ii): disable_overlap_scheduler=True, one request per
    generate call, cuda sync after each): main vs OLD on those same prompts, and main vs b7 on the
    c=1 natural prompts; reports whether each mode-(i) flip persists or vanishes.
    """
    print_dims()
    import gc

    import tensorrt_llm._torch.attention.backends.flashinfer as fi
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    real = mg.gemma4_fused_qkv_norm_rope_quant
    real_enabled = mg.Gemma4Attention._fused_qkv_prep_enabled
    variant = {"cute": True}

    def wrapped(*a, **kw):
        kw["cute_norm_order"] = bool(variant["cute"] and kw.get("cute_norm_order", False))
        return real(*a, **kw)

    def enabled(self):
        ok = real_enabled(self)
        # The OLD (landing-3) kernel never ran on the K=V global layers.
        return ok and (variant["cute"] or not self._qk_only_proj)

    mg.gemma4_fused_qkv_norm_rope_quant = wrapped
    mg.Gemma4Attention._fused_qkv_prep_enabled = enabled
    fi._HOST_PREFILL_PLAN = False
    params = sp(logprobs=20)
    nat = NaturalPrompts(9140)
    groups = []  # (label, [prompts]) submitted together
    for isl, lab, n in ((300, "isl300@1", 8), (1000, "isl1k@1", 8), (8000, "isl8k@1", 3)):
        for _ in range(n):
            groups.append((lab, [nat.make(isl)]))
    for _ in range(2):
        groups.append(("isl1k@4", [nat.make(1000) for _ in range(4)]))
    for _ in range(2):
        groups.append(("isl8k@4", [nat.make(8000) for _ in range(4)]))
    groups.append(("isl1k@16", [nat.make(1000) for _ in range(16)]))
    qs = ["What is the capital of France? Answer in one word.", "Explain why the sky is blue.",
          "Write a haiku about autumn leaves.", "List three prime numbers greater than 100.",
          "Summarize the plot of Hamlet in two sentences.", "What does a CPU cache do?"]
    for qtext in qs:
        ids = nat.tok.apply_chat_template([{"role": "user", "content": qtext}], add_generation_prompt=True,
                                          tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        groups.append(("chat@1", [list(map(int, ids))]))
    c1_nat = [g for g in groups if g[0] in ("isl300@1", "isl1k@1", "isl8k@1", "chat@1")]
    rnd = Prompts(774)
    old_groups = [("rnd_isl300@1", [rnd.make(300)]) for _ in range(10)] + \
                 [("rnd_isl1k@1", [rnd.make(1000)]) for _ in range(10)]
    log(f"Route A set: {len(groups)} submissions, {sum(len(g) for _, g in groups)} natural prompts; "
        f"{len(old_groups)} random-id prompts for the OLD-kernel arm")

    def run(llm, tag, fused, cute, gs, sequential=False):
        mg._FUSED_QKV_PREP = fused
        variant["cute"] = cute
        res = []
        t0 = time.time()
        for label, ps in gs:
            if sequential:
                outs = []
                for p in ps:
                    outs.extend(llm.generate([p], sampling_params=params, use_tqdm=False))
                    torch.cuda.synchronize()
            else:
                outs = llm.generate(ps, sampling_params=params, use_tqdm=False)
            for o in outs:
                res.append((label, len(res), summarize(o)))
        log(f"arm {tag}: fused={fused} cute={cute} host_plan={fi._HOST_PREFILL_PLAN} "
            f"seq={sequential} {time.time() - t0:.1f}s")
        return res

    rep = {}
    # ---- mode (i): default engine (overlap on)
    llm = build_llm(batch_wait_timeout_ms=float(os.environ.get("BATCH_WAIT_MS", "1500")))
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        llm.generate([warm.make(isl) for _ in range(2)], sampling_params=params, use_tqdm=False)
    out = {}
    for tag, f, c in (("main", False, True), ("b7", True, True), ("main_b", False, True), ("b7_b", True, True)):
        out[tag] = run(llm, tag, f, c, groups)
    out["old_main"] = run(llm, "old_main", False, True, old_groups)
    out["old_b7old"] = run(llm, "old_b7old", True, False, old_groups)
    rep["i_b7_vs_main"] = _cmp_arms(out["main"], out["b7"])
    rep["i_main_b_vs_main"] = _cmp_arms(out["main"], out["main_b"])
    rep["i_b7_b_vs_b7"] = _cmp_arms(out["b7"], out["b7_b"])
    rep["i_b7_b_vs_main"] = _cmp_arms(out["main"], out["b7_b"])
    rep["i_old_vs_main_rnd"] = _cmp_arms(out["old_main"], out["old_b7old"])
    for k, v in rep.items():
        log(f"{k}: {json.dumps(v['total'])}")
    llm.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log("LLM 1 freed")
    # ---- mode (ii): overlap scheduler off, one request per generate call, sync per request
    try:
        llm = build_llm(batch_wait_timeout_ms=0.0, disable_overlap_scheduler=True)
        for isl in (300, 1000):
            llm.generate([warm.make(isl)], sampling_params=params, use_tqdm=False)
        o2 = {}
        o2["old_main"] = run(llm, "ii_old_main", False, True, old_groups, sequential=True)
        o2["old_b7old"] = run(llm, "ii_old_b7old", True, False, old_groups, sequential=True)
        o2["main"] = run(llm, "ii_main_c1", False, True, c1_nat, sequential=True)
        o2["b7"] = run(llm, "ii_b7_c1", True, True, c1_nat, sequential=True)
        rep["ii_old_vs_main_rnd"] = _cmp_arms(o2["old_main"], o2["old_b7old"])
        rep["ii_b7_vs_main_c1"] = _cmp_arms(o2["main"], o2["b7"])
        # mode (i) main vs mode (ii) main on the same random prompts
        rep["ii_main_vs_i_main_rnd"] = _cmp_arms(out["old_main"], o2["old_main"])
        fl_i = {d["idx"]: d for v in rep["i_old_vs_main_rnd"]["per_label"].values() for d in v["flip_detail"]}
        fl_ii = {d["idx"] for v in rep["ii_old_vs_main_rnd"]["per_label"].values() for d in v["flip_detail"]}
        rep["old_flip_persistence"] = {str(i): {"label": out["old_main"][i][0], "mode_i": d,
                                                "mode_ii": "persists" if i in fl_ii else "vanishes"}
                                       for i, d in fl_i.items()}
        for k in ("ii_old_vs_main_rnd", "ii_b7_vs_main_c1", "ii_main_vs_i_main_rnd"):
            log(f"{k}: {json.dumps(rep[k]['total'])}")
        log("old flip persistence: " + json.dumps(rep["old_flip_persistence"]))
        llm.shutdown()
    except Exception as e:  # noqa: BLE001
        log(f"mode (ii) failed: {e!r}")
        rep["mode_ii_error"] = repr(e)
    mg._FUSED_QKV_PREP, variant["cute"] = True, True
    b7 = rep["i_b7_vs_main"]["total"]
    det = (rep["i_main_b_vs_main"]["total"]["differ"] == 0 and rep["i_b7_b_vs_b7"]["total"]["differ"] == 0)
    ok_ii = rep.get("ii_b7_vs_main_c1", {"total": {"differ": 1}})["total"]["differ"] == 0
    print(json.dumps({"passed": bool(b7["differ"] == 0 and det and ok_ii),
                      "compared": "first token + top-20 logprobs, bitwise, fused QKV prep (60 layers: 50 sliding + "
                                  "10 K=V global) vs main's unfused chain in one process, host prefill plan off; "
                                  "mode (i) overlap on, mode (ii) overlap off + one request per call",
                      "b7_vs_main_bitwise": b7["differ"] == 0, "repeat_bitwise": det,
                      "summary": {k: v["total"] for k, v in rep.items() if isinstance(v, dict) and "total" in v},
                      "report": rep}), flush=True)


def _first_cmp(m, c):
    """(differ, flip, max top-20 |dlp| over common tokens, top-20 token sets equal) for one prompt."""
    tm, dm = _first(m)
    tc, dc = _first(c)
    common = set(dm) & set(dc)
    d = max((abs(dm[t] - dc[t]) for t in common), default=0.0)
    return m != c, tm != tc, d, set(dm) == set(dc)


def mode_item5():
    """Item 5 (plan 547 / ruling 914-1): in-process A,B,A',B' over own prompts, 1 token, top-20 bitwise.

    TRTLLM_FI_HOST_PREFILL_PLAN is set (default 1, main's default) before tensorrt_llm is imported and is
    never changed. Arms: main (fused prep off = main's unfused chain), b7 (fused prep on, all 60 layers),
    main_b, b7_b, plus the landing-3 OLD kernel (positive control) on the trigger set (a).
    Set (a): job 246's random-id prompts, Prompts(774): 10 x isl300@1 then 10 x isl1k@1 (sha256 printed).
    Set (b): natural text + random-id prompts across ISL x c, c=4 / c=16 groups sent together.
    Mode (ii) LLM (overlap off, one request per call, sync per request): main vs b7 on set (a) and on
    every prompt that diverged in mode (i).
    """
    import gc
    import hashlib

    print_dims()
    import tensorrt_llm._torch.attention.backends.flashinfer as fi
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    hp_env = os.environ.get("TRTLLM_FI_HOST_PREFILL_PLAN")
    log(f"TRTLLM_FI_HOST_PREFILL_PLAN={hp_env!r} (set before import) -> flashinfer._HOST_PREFILL_PLAN="
        f"{fi._HOST_PREFILL_PLAN}; TLLM_WORKER_USE_SINGLE_PROCESS={os.environ.get('TLLM_WORKER_USE_SINGLE_PROCESS')}")
    real = mg.gemma4_fused_qkv_norm_rope_quant
    real_enabled = mg.Gemma4Attention._fused_qkv_prep_enabled
    variant = {"cute": True}

    def wrapped(*a, **kw):
        kw["cute_norm_order"] = bool(variant["cute"] and kw.get("cute_norm_order", False))
        return real(*a, **kw)

    def enabled(self):
        ok = real_enabled(self)
        return ok and (variant["cute"] or not self._qk_only_proj)  # OLD kernel: sliding layers only

    mg.gemma4_fused_qkv_norm_rope_quant = wrapped
    mg.Gemma4Attention._fused_qkv_prep_enabled = enabled
    params = sp(logprobs=20)

    # ---- set (a): trigger set, regenerated from job 246's seed in job 246's order
    rnd = Prompts(774)
    set_a = [("a_rnd_isl300@1", [rnd.make(300)]) for _ in range(10)] + \
            [("a_rnd_isl1k@1", [rnd.make(1000)]) for _ in range(10)]
    ids_a = [g[1][0] for g in set_a]
    sha_a = hashlib.sha256(json.dumps(ids_a).encode()).hexdigest()
    sha_a300 = hashlib.sha256(json.dumps(ids_a[:10]).encode()).hexdigest()
    log(f"set (a): {len(ids_a)} prompts sha256(all)={sha_a} sha256(isl300 x10)={sha_a300} "
        f"first ids #0={ids_a[0][:6]} #2={ids_a[2][:6]}")

    # ---- set (b): 200+ own prompts
    nat = NaturalPrompts(9147)
    rb = Prompts(5470)
    sb = []
    for isl, lab, n in ((300, "isl300@1", 20), (1000, "isl1k@1", 20), (2000, "isl2k@1", 20),
                        (4000, "isl4k@1", 8), (8000, "isl8k@1", 16)):
        sb += [(lab, [nat.make(isl)]) for _ in range(n)]
    for isl, lab, n in ((300, "rnd_isl300@1", 10), (1000, "rnd_isl1k@1", 10), (2000, "rnd_isl2k@1", 10),
                        (4000, "rnd_isl4k@1", 5), (8000, "rnd_isl8k@1", 5)):
        sb += [(lab, [rb.make(isl)]) for _ in range(n)]
    for isl, lab, n in ((8000, "isl8k@4", 5), (1000, "isl1k@4", 5), (300, "isl300@4", 3), (2000, "isl2k@4", 3),
                        (4000, "isl4k@4", 2)):
        sb += [(lab, [nat.make(isl) for _ in range(4)]) for _ in range(n)]
    sb.append(("rnd_isl1k@4", [rb.make(1000) for _ in range(4)]))
    sb.append(("rnd_isl8k@4", [rb.make(8000) for _ in range(4)]))
    sb.append(("isl1k@16", [nat.make(1000) for _ in range(16)]))
    n_b = sum(len(g) for _, g in sb)
    n_rnd_b = sum(len(g) for lab, g in sb if lab.startswith("rnd"))
    sha_b = hashlib.sha256(json.dumps([g for _, g in sb]).encode()).hexdigest()
    log(f"set (b): {len(sb)} submissions, {n_b} prompts ({n_rnd_b} random-id, {n_b - n_rnd_b} natural), "
        f"sha256={sha_b}")

    def run(llm, tag, fused, cute, gs, sequential=False):
        mg._FUSED_QKV_PREP = fused
        variant["cute"] = cute
        res = []
        t0 = time.time()
        for label, ps in gs:
            if sequential:
                outs = []
                for p in ps:
                    outs.extend(llm.generate([p], sampling_params=params, use_tqdm=False))
                    torch.cuda.synchronize()
            else:
                outs = llm.generate(ps, sampling_params=params, use_tqdm=False)
            for o in outs:
                res.append((label, len(res), summarize(o)))
        log(f"arm {tag}: fused={fused} cute={cute} host_plan={fi._HOST_PREFILL_PLAN} seq={sequential} "
            f"n={len(res)} {time.time() - t0:.1f}s")
        return res

    def per_prompt(ref, cand, tag):
        rows = []
        for (lab, i, m), (_, _, c) in zip(ref, cand):
            differ, flip, d, same_set = _first_cmp(m, c)
            rows.append({"idx": i, "label": lab, "ref_tok": _first(m)[0], "tok": _first(c)[0],
                         "bitwise": not differ, "flip": flip, "max_top20_dlp": d, "top20_set_equal": same_set})
        return rows

    rep = {}
    t_start = time.time()
    batch_wait = float(os.environ.get("BATCH_WAIT_MS", "300"))
    llm = build_llm(batch_wait_timeout_ms=batch_wait)
    log(f"LLM 1 (mode i, overlap on, batch_wait_timeout_ms={batch_wait}) loaded in {time.time() - t_start:.0f}s")
    warm = Prompts(5)
    for isl in (300, 1000, 8000):
        llm.generate([warm.make(isl) for _ in range(2)], sampling_params=params, use_tqdm=False)
    llm.generate([warm.make(1000) for _ in range(4)], sampling_params=params, use_tqdm=False)
    out = {}
    # (a) first
    for tag, f, c in (("main", False, True), ("b7", True, True), ("main_b", False, True), ("b7_b", True, True),
                      ("old", True, False)):
        out["a_" + tag] = run(llm, "a_" + tag, f, c, set_a)
    a_rows = {t: per_prompt(out["a_main"], out["a_" + t], t) for t in ("b7", "main_b", "b7_b", "old")}
    rep["a_b7_vs_main"] = _cmp_arms(out["a_main"], out["a_b7"])
    rep["a_b7_b_vs_main"] = _cmp_arms(out["a_main"], out["a_b7_b"])
    rep["a_main_b_vs_main"] = _cmp_arms(out["a_main"], out["a_main_b"])
    rep["a_b7_b_vs_b7"] = _cmp_arms(out["a_b7"], out["a_b7_b"])
    rep["a_old_vs_main"] = _cmp_arms(out["a_main"], out["a_old"])
    for k in [k for k in rep if k.startswith("a_")]:
        log(f"{k}: {json.dumps(rep[k]['total'])}")
    named = {}
    for j in (2, 4, 7, 8):
        b, o = a_rows["b7"][j], a_rows["old"][j]
        named[f"#{j}"] = {"label": b["label"], "main_tok": b["ref_tok"], "b7_tok": b["tok"], "b7_bitwise": b["bitwise"],
                          "b7_max_dlp": b["max_top20_dlp"], "b7_b_bitwise": a_rows["b7_b"][j]["bitwise"],
                          "old_tok": o["tok"], "old_flip": o["flip"], "old_max_dlp": round(o["max_top20_dlp"], 4)}
    log("set (a) named isl300@1 prompts (0-based): " + json.dumps(named))
    for r_b7, r_old in zip(a_rows["b7"], a_rows["old"]):
        log(f"  a#{r_b7['idx']:2d} {r_b7['label']}: main={r_b7['ref_tok']} b7={r_b7['tok']} "
            f"b7_bitwise={r_b7['bitwise']} b7_dlp={r_b7['max_top20_dlp']} | old={r_old['tok']} "
            f"old_flip={r_old['flip']} old_dlp={r_old['max_top20_dlp']:.4f}")
    # (b)
    for tag, f in (("main", False), ("b7", True), ("main_b", False), ("b7_b", True)):
        out["b_" + tag] = run(llm, "b_" + tag, f, True, sb)
    rep["b_b7_vs_main"] = _cmp_arms(out["b_main"], out["b_b7"])
    rep["b_b7_b_vs_main"] = _cmp_arms(out["b_main"], out["b_b7_b"])
    rep["b_main_b_vs_main"] = _cmp_arms(out["b_main"], out["b_main_b"])
    rep["b_b7_b_vs_b7"] = _cmp_arms(out["b_b7"], out["b_b7_b"])
    for k in [k for k in rep if k.startswith("b_")]:
        log(f"{k}: {json.dumps(rep[k]['total'])}")
    flip_table = {lab: {"n": v["n"], "flips": v["flips"], "differ": v["differ"], "max_top20_dlp": v["max_top20_dlp"]}
                  for lab, v in rep["b_b7_vs_main"]["per_label"].items()}
    log("set (b) flip table b7 vs main per ISL x c: " + json.dumps(flip_table))
    # prompts that diverged anywhere in mode (i) between b7 arms and main
    div = []
    for key in ("b_b7", "b_b7_b"):
        for (lab, i, m), (_, _, c) in zip(out["b_main"], out[key]):
            if m != c and i not in div:
                div.append(i)
    flat_b = [(lab, p) for lab, ps in sb for p in ps]
    llm.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log(f"LLM 1 freed; mode (i) diverging set (b) prompts: {div}")
    # ---- mode (ii)
    try:
        llm = build_llm(batch_wait_timeout_ms=0.0, disable_overlap_scheduler=True)
        for isl in (300, 1000):
            llm.generate([warm.make(isl)], sampling_params=params, use_tqdm=False)
        o2 = {}
        o2["a_main"] = run(llm, "ii_a_main", False, True, set_a, sequential=True)
        o2["a_b7"] = run(llm, "ii_a_b7", True, True, set_a, sequential=True)
        rep["ii_a_b7_vs_main"] = _cmp_arms(o2["a_main"], o2["a_b7"])
        log(f"ii_a_b7_vs_main: {json.dumps(rep['ii_a_b7_vs_main']['total'])}")
        if div:
            gd = [(flat_b[i][0], [flat_b[i][1]]) for i in div]
            o2["d_main"] = run(llm, "ii_div_main", False, True, gd, sequential=True)
            o2["d_b7"] = run(llm, "ii_div_b7", True, True, gd, sequential=True)
            rep["ii_div_b7_vs_main"] = _cmp_arms(o2["d_main"], o2["d_b7"])
            log(f"ii_div_b7_vs_main: {json.dumps(rep['ii_div_b7_vs_main']['total'])}")
        llm.shutdown()
    except Exception as e:  # noqa: BLE001
        log(f"mode (ii) failed: {e!r}")
        rep["mode_ii_error"] = repr(e)
    mg._FUSED_QKV_PREP, variant["cute"] = True, True
    a_ok = rep["a_b7_vs_main"]["total"]["differ"] == 0 and rep["a_b7_b_vs_main"]["total"]["differ"] == 0
    b_ok = rep["b_b7_vs_main"]["total"]["differ"] == 0 and rep["b_b7_b_vs_main"]["total"]["differ"] == 0
    rep_ok = all(rep[k]["total"]["differ"] == 0 for k in ("a_main_b_vs_main", "a_b7_b_vs_b7", "b_main_b_vs_main",
                                                          "b_b7_b_vs_b7"))
    ii_ok = rep.get("ii_a_b7_vs_main", {"total": {"differ": 1}})["total"]["differ"] == 0
    pos_ctrl = rep["a_old_vs_main"]["total"]
    print(json.dumps({
        "passed": bool(a_ok and b_ok and rep_ok and ii_ok),
        "compared": "first token id + top-20 logprobs, bitwise, in one process: A=main (fused prep off, main's "
                    "unfused chain), B=b7 fused prep (60 layers, cute_norm_order), A'=main repeat, B'=b7 repeat; "
                    "1 token per prompt; c=4/c=16 groups submitted together (batch_wait_timeout_ms); "
                    "own prompts per ruling 914-1",
        "host_prefill_plan": {"env_TRTLLM_FI_HOST_PREFILL_PLAN": hp_env, "module_flag": bool(fi._HOST_PREFILL_PLAN),
                              "same_in_every_arm": True},
        "set_a": {"sha256_ids": sha_a, "sha256_isl300x10": sha_a300, "n": len(ids_a),
                  "b7_vs_main": rep["a_b7_vs_main"]["total"], "b7_b_vs_main": rep["a_b7_b_vs_main"]["total"],
                  "named_isl300": named, "positive_control_old_kernel_vs_main": pos_ctrl,
                  "per_prompt_b7": a_rows["b7"], "per_prompt_old": a_rows["old"]},
        "set_b": {"sha256_ids": sha_b, "n": n_b, "n_random_id": n_rnd_b, "flip_table_b7_vs_main": flip_table,
                  "b7_vs_main": rep["b_b7_vs_main"]["total"], "b7_b_vs_main": rep["b_b7_b_vs_main"]["total"]},
        "repeat_arms_bitwise": rep_ok, "mode_ii_set_a_bitwise": ii_ok, "mode_i_diverging_prompts": div,
        "summary": {k: v["total"] for k, v in rep.items() if isinstance(v, dict) and "total" in v},
        "elapsed_s": round(time.time() - t_start, 1)}), flush=True)


def mode_both(modes=("dump", "ab")):
    """dump then ab, each in its own process (one model instance at a time); combined JSON last."""
    import subprocess
    out = {}
    for m in modes:
        p = subprocess.Popen([sys.executable, __file__, m, CKPT], stdout=subprocess.PIPE, text=True)
        last = None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip().startswith("{"):
                last = line
        p.wait()
        try:
            out[m] = json.loads(last) if last else {"passed": False, "error": f"rc={p.returncode}"}
        except json.JSONDecodeError:
            out[m] = {"passed": False, "error": "bad json"}
    print(json.dumps({"passed": bool(all(out[m].get("passed") for m in modes)), **out}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        mode_dump()
    elif sys.argv[1] == "both":
        mode_both()
    elif sys.argv[1] == "routea":
        mode_routea()
    elif sys.argv[1] == "dump_routea":
        mode_both(("dump", "routea"))
    elif sys.argv[1] == "outcome0":
        mode_outcome0()
    elif sys.argv[1] == "item5":
        mode_item5()
    else:
        mode_ab()
