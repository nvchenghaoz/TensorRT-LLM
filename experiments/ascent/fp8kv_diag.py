#!/usr/bin/env python3
"""FP8-KV diagnostic for idea 12 (kv_cache_config.dtype='fp8').

Runs the real checkpoint in-process (TLLM_WORKER_USE_SINGLE_PROCESS=1) with the
benchmark's shared-prefix shape (32768-token prefix + 2048-token suffix) and:
  * records per-layer |V| amax (and post-k_norm |K| amax) of the full-attention
    layers during eager prefill via forward hooks on qkv_proj;
  * runs a prime request, then a prefix-reuse request (optionally bracketed by
    cudaProfilerStart/Stop for nsys), then 3 identical greedy requests and
    checks they are token- and logprob-identical;
  * dumps greedy tokens + logprobs of a fixed prompt set so a second arm (BF16
    KV) can be compared against it.
Prints one JSON line at the end.
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import torch  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def corpus_ids(tok, n_tokens, offset_files=0):
    """Real text (repo docs/markdown) tokenized, at least n_tokens long."""
    files = sorted(glob.glob("docs/source/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/_torch/**/*.md", recursive=True))
    files = files[offset_files:] + files[:offset_files]
    ids = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except OSError:
            continue
        ids.extend(tok.encode(txt, add_special_tokens=False))
        if len(ids) >= n_tokens:
            break
    assert len(ids) >= n_tokens, (len(ids), n_tokens)
    return ids[:n_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--llm-kwargs", default=None,
                    help="JSON file with extra LLM kwargs (server settings)")
    ap.add_argument("--prefix", type=int, default=32768)
    ap.add_argument("--suffix", type=int, default=2048)
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--n-compare", type=int, default=4)
    ap.add_argument("--compare-osl", type=int, default=128)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default="/tmp/fp8kv_diag.json")
    args = ap.parse_args()

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    kw = {}
    if args.llm_kwargs:
        with open(args.llm_kwargs) as fh:
            kw = json.load(fh)
    kvc = dict(kw.pop("kv_cache_config", {}) or {})
    kvc["dtype"] = args.kv_dtype
    kvc.setdefault("enable_block_reuse", True)
    kw.setdefault("max_seq_len", 36864)
    log("LLM kwargs:", json.dumps(kw), "kv_cache_config:", json.dumps(kvc))
    t0 = time.time()
    llm = LLM(model=args.ckpt, kv_cache_config=KvCacheConfig(**kvc), **kw)
    log(f"model loaded in {time.time() - t0:.1f}s")
    tok = llm.tokenizer

    # ---------------- hooks on full-attention qkv_proj ----------------
    model = llm._executor.engine.model_engine.model
    vamax, kamax = {}, {}
    hooks = []
    n_attn = 0
    for name, mod in model.named_modules():
        if hasattr(mod, "qkv_proj") and hasattr(mod, "kv_size") and hasattr(
                mod, "head_dim"):
            n_attn += 1
            lid = getattr(mod, "layer_idx", name)
            kv_size, hd = mod.kv_size, mod.head_dim
            vamax[lid] = torch.zeros((), device="cuda", dtype=torch.float32)
            kamax[lid] = torch.zeros((), device="cuda", dtype=torch.float32)
            knorm = getattr(mod, "k_norm", None)

            def hook(m, inp, out, lid=lid, kv_size=kv_size, hd=hd,
                     knorm=knorm):
                if torch.cuda.is_current_stream_capturing():
                    return
                o = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(o) or o.dim() < 2:
                    return
                v = o[..., -kv_size:]
                k = o[..., -2 * kv_size:-kv_size]
                vamax[lid] = torch.maximum(vamax[lid],
                                           v.abs().amax().float())
                if knorm is not None:
                    try:
                        kn = knorm(k.reshape(-1, hd).contiguous())
                        kn = kn[0] if isinstance(kn, tuple) else kn
                        kamax[lid] = torch.maximum(kamax[lid],
                                                   kn.abs().amax().float())
                    except Exception:  # noqa: BLE001
                        pass

            hooks.append(mod.qkv_proj.register_forward_hook(hook))
    log(f"hooked {n_attn} attention modules")

    prefix = corpus_ids(tok, args.prefix)
    suf_pool = corpus_ids(tok, args.suffix * 8, offset_files=37)

    def suffix(i):
        s = suf_pool[(i * args.suffix) % len(suf_pool):]
        s = (s + suf_pool)[:args.suffix]
        return s

    def gen(ids, osl, logprobs=True):
        sp = SamplingParams(max_tokens=osl, temperature=0.0, top_k=1,
                            ignore_eos=True,
                            logprobs=1 if logprobs else None)
        t = time.time()
        out = llm.generate([ids], sp, use_tqdm=False)[0]
        dt = time.time() - t
        c = out.outputs[0]
        lps = []
        if logprobs and c.logprobs:
            for tid, d in zip(c.token_ids, c.logprobs):
                lp = d.get(tid) if isinstance(d, dict) else None
                lps.append(float(lp.logprob) if lp is not None else None)
        cached = getattr(out, "cached_tokens", None)
        return list(c.token_ids), lps, dt, cached

    res = {"kv_dtype": args.kv_dtype, "n_attn_layers": n_attn}

    # prime the prefix
    log("prime request")
    _, _, dt, cached = gen(prefix + suffix(100), 1, logprobs=False)
    log(f"prime done {dt:.2f}s cached={cached}")
    # warm one reuse request (JIT/autotune for this context shape)
    _, _, dt, cached = gen(prefix + suffix(101), 4, logprobs=False)
    log(f"warm reuse {dt:.2f}s cached={cached}")

    # profiled reuse request (benchmark shape)
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    toks, _, dt, cached = gen(prefix + suffix(102), args.osl,
                              logprobs=False)
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    log(f"bench-shape request {dt:.2f}s cached={cached}")
    res["bench_request_s"] = dt
    res["bench_request_cached"] = cached

    # repeated-call determinism: 3 identical greedy requests
    rep = []
    for r in range(3):
        t, lp, dt, cached = gen(prefix + suffix(103), args.compare_osl)
        rep.append((t, lp))
        log(f"repeat {r}: {dt:.2f}s cached={cached} first={t[:8]}")
    same_tok = all(x[0] == rep[0][0] for x in rep)
    same_lp = all(x[1] == rep[0][1] for x in rep)
    res["repeat_identical_tokens"] = same_tok
    res["repeat_identical_logprobs"] = same_lp

    # fixed comparison prompts (for BF16 vs FP8 comparison across arms)
    cmp = []
    for i in range(args.n_compare):
        t, lp, dt, cached = gen(prefix + suffix(200 + i), args.compare_osl)
        cmp.append({"tokens": t, "logprobs": lp, "cached": cached})
        log(f"compare {i}: {dt:.2f}s cached={cached}")
    # a short no-prefix prompt set (cold path, non-reuse context FMHA)
    for i in range(2):
        ids = suf_pool[i * 1500:(i + 1) * 1500]
        t, lp, dt, cached = gen(ids, args.compare_osl)
        cmp.append({"tokens": t, "logprobs": lp, "cached": cached})
        log(f"short compare {i}: {dt:.2f}s")
    res["compare"] = cmp

    torch.cuda.synchronize()
    res["v_amax"] = {str(k): round(float(v), 4) for k, v in vamax.items()}
    res["k_amax_postnorm"] = {
        str(k): round(float(v), 4)
        for k, v in kamax.items()
    }
    res["v_amax_max"] = max(res["v_amax"].values()) if vamax else None
    res["k_amax_max"] = max(res["k_amax_postnorm"].values()) if kamax else None
    log("V amax per layer:", res["v_amax"])
    log("K(post-norm) amax per layer:", res["k_amax_postnorm"])
    for h in hooks:
        h.remove()
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    summary = {k: v for k, v in res.items() if k != "compare"}
    print(json.dumps(summary), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
