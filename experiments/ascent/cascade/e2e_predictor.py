import glob, hashlib, json, os, sys, threading, time
os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
# Worker-side two-pass PREDICTOR of the frozen landing check (not the gate), protocol shapes, one arm per process.
# BRACE-FREE on purpose: run_gpu_pair passes it inline (python -c) to both trees (the base tree lacks this file).
# 56 prompts = one 32768-token shared prefix (real text) + 2048 unique tokens; c8 = prompts 0-7 together, c16 = 8-23,
# c32 = 24-55; 40 greedy tokens with logprobs=1; max_batch_size 32, decode CUDA graphs to 32, max_num_tokens 8192,
# chunked prefill, kv_cache_config from experiments/ascent-config/engine.json; pass 2 reruns all 56 prompts in the same
# LLM lifetime (cached prefix + recurrent snapshot). Writes every token id and logprob to
# $AUTO_DEV_ARTIFACTS/e2e_pred.json and prints a per-prompt digest line. Usage: python -c "<this>" <checkpoint>
PREFIX, SUFFIX, OSL = 32768, 2048, 40
GROUPS = (("c8", 8), ("c16", 16), ("c32", 32))
WAIT_MS = float(os.environ.get("E2E_BATCH_WAIT_MS", "300"))


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
    files += sorted(glob.glob("tensorrt_llm/**/*.py", recursive=True))
    files += sorted(glob.glob("examples/**/*.md", recursive=True))
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


def main(ckpt):
    heartbeat()
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    log("engine.json kv_cache_config:", json.dumps(kvc), "batch_wait_ms", WAIT_MS)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=WAIT_MS)
    log("LLM ready in %.1fs" % (time.time() - t0))
    tok = llm.tokenizer
    n = sum(c for _, c in GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(n)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)

    def run_pass(tag):
        res = dict()
        base = 0
        for g, conc in GROUPS:
            idx = list(range(base, base + conc))
            base += conc
            t = time.time()
            outs = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(round(float(lp.logprob), 6) if lp is not None else None)
                res["%s-p%d" % (g, j)] = dict(ids=list(c.token_ids), lp=lps)
            log("%s %s: %d prompts in %.2fs" % (tag, g, conc, time.time() - t))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_pred.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    p2p1 = sorted(k for k in p1 if p1[k]["ids"] != p2[k]["ids"])
    per = dict()
    for k in sorted(p1):
        per[k] = hashlib.sha256(json.dumps([p1[k]["ids"], p2[k]["ids"]]).encode()).hexdigest()[:16]
    out = dict(pass2_vs_pass1_token_diverging=p2p1, n_prompts=len(p1), osl=OSL, batch_wait_ms=WAIT_MS,
               artifact="e2e_pred.json", per_prompt_ids_sha=per, pass1=p1, pass2=p2)
    print(json.dumps(out), flush=True)
    llm.shutdown()
    os._exit(0)


main(sys.argv[1])
