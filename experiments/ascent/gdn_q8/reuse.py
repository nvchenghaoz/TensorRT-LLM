# Prefix-reuse check for idea 37 (8-bit GDN state pool), acceptance item 4.
# Engine: experiments/ascent-config/engine.json (FP8 KV, block reuse, mamba
# snapshot at 32768, mamba_ssm_cache_dtype from the json) + the harness base
# engine (max_batch_size 128, decode CUDA graphs up to 128, max_num_tokens
# 8192, chunked prefill).
# Lifetime 1 (engine.json pool, int8):
#   A. prompt 0 = 32768-token prefix + 2048 unique tokens, alone, cold cache
#      (fresh prefill; the snapshot at 32768 is saved from the 4th chunk).
#   B. prompt 0 again (reuse hit: cached_tokens 32768, state restored from
#      the snapshot). Expect the same 40 greedy tokens and logprobs as A.
#   C. prompts 0..63 together (all reuse hits; mixed steps with riders),
#      40 greedy tokens each.
# Lifetime 2 (same engine, mamba_ssm_cache_dtype "auto" = main's bf16 pool):
#   A/B/C again; reports token agreement int8 vs bf16 (advisory only).
import glob
import json
import os
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

PREFIX, SUFFIX, OSL, NC = 32768, 2048, 40, 64


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def heartbeat():
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log("[heartbeat] %.0fs" % (time.time() - t0))

    threading.Thread(target=loop, daemon=True).start()


def corpus_ids(tok, n_tokens):
    files = sorted(glob.glob("docs/source/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/_torch/**/*.md", recursive=True))
    files += sorted(glob.glob("examples/**/*.md", recursive=True))
    files += sorted(glob.glob("tensorrt_llm/**/*.py", recursive=True))
    ids = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                ids.extend(tok.encode(fh.read(), add_special_tokens=False))
        except OSError:
            continue
        if len(ids) >= n_tokens:
            break
    assert len(ids) >= n_tokens, (len(ids), n_tokens)
    return ids[:n_tokens]


def lifetime(ckpt, ssm_dtype, ids_cache):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    if ssm_dtype is not None:
        kvc["mamba_ssm_cache_dtype"] = ssm_dtype
    log("kv_cache_config", kvc)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=128,
              max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=128),
              kv_cache_config=KvCacheConfig(**kvc))
    log("LLM ready in %.1fs" % (time.time() - t0))
    tok = llm.tokenizer
    if "ids" not in ids_cache:
        ids_cache["ids"] = corpus_ids(tok, PREFIX + NC * SUFFIX)
    ids = ids_cache["ids"]
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX]
               for i in range(NC)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1,
                        ignore_eos=True, logprobs=1)

    def run(idx, tag):
        t = time.time()
        outs = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            lps = []
            for tid, d in zip(c.token_ids, c.logprobs or []):
                lp = d.get(tid) if isinstance(d, dict) else None
                lps.append(float(lp.logprob) if lp is not None else None)
            res.append(dict(ids=list(c.token_ids), lp=lps,
                            cached=getattr(o, "cached_tokens", None)))
        log("%s: %d prompts %.1fs cached %s" %
            (tag, len(idx), time.time() - t,
             sorted(set(str(r["cached"]) for r in res))))
        return res

    a = run([0], "A fresh")[0]
    b = run([0], "B reuse")[0]
    c = run(list(range(NC)), "C reuse x64")
    llm.shutdown()
    del llm
    return dict(a=a, b=b, c=c)


def main(ckpt):
    heartbeat()
    ids_cache = dict()
    q8 = lifetime(ckpt, None, ids_cache)
    ref = lifetime(ckpt, "auto", ids_cache)

    def first_div(x, y):
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return i
        return len(x)

    def summarize(r):
        a, b = r["a"], r["b"]
        max_dlp = max(abs(p - q) for p, q in zip(a["lp"], b["lp"])
                      if p is not None and q is not None)
        return dict(
            fresh_cached=a["cached"], reuse_cached=b["cached"],
            reuse_same_ids=a["ids"] == b["ids"],
            reuse_same_logprobs=a["lp"] == b["lp"],
            reuse_max_abs_dlogprob=max_dlp,
            reuse_first_divergence=first_div(a["ids"], b["ids"]),
            c_cached=sorted(set(str(x["cached"]) for x in r["c"])),
            c_prompt0_same_as_fresh=r["c"][0]["ids"] == a["ids"])

    s_q8, s_ref = summarize(q8), summarize(ref)
    agree = [first_div(x["ids"], y["ids"]) for x, y in zip(q8["c"], ref["c"])]
    passed = (s_q8["reuse_same_ids"] and s_q8["reuse_cached"] == PREFIX
              and s_q8["fresh_cached"] in (0, None))
    res = dict(
        passed=bool(passed),
        compared="int8 pool: fresh prefill vs prefix-reuse hit (snapshot "
        "at 32768) of the same prompt, same lifetime; main bf16 pool alongside",
        int8=s_q8, main_bf16=s_ref,
        c64_int8_vs_bf16_first_divergence=agree,
        c64_int8_vs_bf16_full_match=sum(1 for x in agree if x == OSL),
        c64_int8_vs_bf16_mean_first_divergence=sum(agree) / len(agree))
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "reuse_tokens.json"), "w") as f:
            json.dump(dict(int8=q8, bf16=ref), f)
    print(json.dumps(res), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
