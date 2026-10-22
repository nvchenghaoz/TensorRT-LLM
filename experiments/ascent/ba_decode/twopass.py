# Worker-side two-pass replica of the frozen check (brace-free; run_gpu_pair runs it inline in the base tree).
# 56 prompts = one 32768-token shared prefix (repo docs text) + 2048 unique suffix tokens each; groups c8 (8 prompts,
# one generate call of 8), c16 (16 @ 16), c32 (32 @ 32); 40 greedy tokens (ignore_eos, logprobs=1).
# Engine: max_batch_size 32, decode CUDA graphs up to 32, max_num_tokens 8192, chunked prefill, and the tree's
# experiments/ascent-config/engine.json kv_cache_config (FP8 KV, block reuse, mamba snapshot at 32768).
# Pass 1 starts cold; pass 2 reruns all 56 in the same LLM lifetime (cached prefix + snapshot).
# Last stdout line: JSON with per-prompt ids and logprobs for both passes (compare across the pair arms).
import glob
import hashlib
import json
import os
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
PREFIX, SUFFIX, OSL = 32768, 2048, 40
GROUPS = (("c8", 8), ("c16", 16), ("c32", 32))
for _a in sys.argv[2:]:
    print(_a, flush=True)


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
    files += sorted(glob.glob("tensorrt_llm/_torch/modules/**/*.py", recursive=True))
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
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import small_n_gemv as SNG
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    log("engine.json:", json.dumps(engine_json), "small_n_gemv route", SNG._MAX_M_BY_N)
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc))
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
                    lps.append(float(lp.logprob) if lp is not None else None)
                res["%s-p%d" % (g, j)] = dict(ids=list(c.token_ids), lp=lps)
            log("%s %s: %d prompts %.2fs" % (tag, g, conc, time.time() - t))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    p2_vs_p1 = sorted(k for k in p1 if p1[k]["ids"] != p2[k]["ids"])
    sha = hashlib.sha256(json.dumps([p1[k]["ids"] for k in sorted(p1)]).encode()).hexdigest()
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    log("pass2-vs-pass1 token-diverging prompts:", len(p2_vs_p1), p2_vs_p1, "pass1 ids sha", sha)
    print(json.dumps(dict(passed=True, pass2_vs_pass1_diverging=p2_vs_p1, pass1_ids_sha=sha,
                          route=str(SNG._MAX_M_BY_N), pass1=p1, pass2=p2)), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
