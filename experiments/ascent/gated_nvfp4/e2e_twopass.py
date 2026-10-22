# End-to-end two-pass exactness predictor for branch 23 (quant-linear idea 11), one arm per process.
# BRACE-FREE on purpose: passed verbatim via python -c to both trees (the base tree lacks this file).
# Modelled on experiments/ascent/xqa_pdl/e2e_twopass.py, changed to this run's check protocol:
#   56 prompts = one 32768-token shared prefix (real text) + 2048 unique suffix tokens, 40 greedy tokens with
#   logprobs; groups c8@8 (prompts 0-7 in one call), c16@16 (8-23), c32@32 (24-55); max_batch_size 32, decode
#   CUDA graphs to 32, max_num_tokens 8192, chunked prefill, engine.json (FP8 KV, block reuse, snapshot).
#   pass 1 starts cold; pass 2 reruns all 56 reusing the cached prefix and recurrent snapshot.
# Compare the two arms' per-prompt digests: an exact change must match base token for token and logprob for
# logprob in both passes. Also counts the fused-kernel and quantize launches seen by the GDN/attn producers.
# Usage: python e2e_twopass.py <checkpoint>
import glob
import hashlib
import json
import os
import struct
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

PREFIX, SUFFIX = 32768, 2048
OSL = int(os.environ.get("E2E_OSL", "40"))
BATCH_WAIT_MS = float(os.environ.get("E2E_BATCH_WAIT_MS", "200"))
GROUPS = (("c8", 8), ("c16", 16), ("c32", 32))


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


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def digest_lp(lps):
    return hashlib.sha256(b"".join(struct.pack("<d", float("nan") if x is None else x) for x in lps)).hexdigest()


def main(ckpt):
    heartbeat()
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    from tensorrt_llm._torch.modules.mamba import layernorm_gated
    from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn

    cand = hasattr(layernorm_gated, "rms_norm_gated_token_major_nvfp4")
    replays, captures = dict(), set()
    cur = dict(phase="load")
    orig_replay, orig_capture = cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture

    def replay(self, key, *a, **kw):
        k = "%s:bs%d" % (cur["phase"], cgr.KeyType(*key).batch_size)
        replays[k] = replays.get(k, 0) + 1
        return orig_replay(self, key, *a, **kw)

    def capture(self, key, *a, **kw):
        captures.add(cgr.KeyType(*key).batch_size)
        return orig_capture(self, key, *a, **kw)

    cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture = replay, capture

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    log("cand", cand, "engine.json:", json.dumps(engine_json))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=BATCH_WAIT_MS)
    log("LLM ready in %.1fs; decode graphs captured at batch sizes %s" % (time.time() - t0, sorted(captures)))
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
            cur["phase"] = "%s:%s" % (tag, g)
            idx = [base + j for j in range(conc)]
            t = time.time()
            outs = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
            dt = time.time() - t
            names = []
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                name = "%s-p%d" % (g, j)
                names.append(name)
                res[name] = dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, "cached_tokens", None))
            log("%s %s %d prompts: %.2fs cached %s" % (tag, g, conc, dt, [res[x]["cached"] for x in names]))
            base += conc
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    p2_eq_p1 = True
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        p2_eq_p1 &= same
        per[k] = dict(ids_sha=digest_ids(p1[k]["ids"]), lp_sha=digest_lp(p1[k]["lp"]),
                      ids_sha_p2=digest_ids(p2[k]["ids"]), lp_sha_p2=digest_lp(p2[k]["lp"]),
                      first8=p1[k]["ids"][:8], cached_p1=p1[k]["cached"], cached_p2=p2[k]["cached"],
                      pass2_equals_pass1=same)
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays if k.startswith("pass")))
    res = dict(cand=cand, pass2_equals_pass1=p2_eq_p1, graphs_captured_batch_sizes=sorted(captures),
               graph_replays=replays, decode_graph_batch_sizes_replayed=dec_bs,
               prefill_nvfp4_stats=fpn.prefill_nvfp4_stats(),
               all_ids_sha_p1=digest_ids([p1[k]["ids"] for k in sorted(p1)]),
               all_lp_sha_p1=hashlib.sha256("".join(per[k]["lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               all_ids_sha_p2=digest_ids([p2[k]["ids"] for k in sorted(p2)]),
               all_lp_sha_p2=hashlib.sha256("".join(per[k]["lp_sha_p2"] for k in sorted(per)).encode()).hexdigest(),
               engine_json=engine_json, batch_wait_timeout_ms=BATCH_WAIT_MS, osl=OSL,
               prompts="%d x (%d shared prefix + %d unique), %d greedy tokens, logprobs=1; c8@8 c16@16 c32@32"
               % (n, PREFIX, SUFFIX, OSL), per_prompt=per,
               compared="compare all_*_sha and per_prompt across the pair arms (pass 1 and pass 2)")
    res["passed"] = True
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
