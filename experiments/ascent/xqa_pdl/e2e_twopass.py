# End-to-end two-pass exactness check for branch 65 (idea 133, XQA PDL early trigger), one arm per process.
# BRACE-FREE on purpose: run_gpu_pair embeds it in a bash heredoc in both trees (the base tree lacks this file).
# Compare the two arms' digests: an exact change must match base token for token and logprob for logprob.
#
# Mirrors the check adapter's two-pass protocol on the benchmark shape with main's FP8-KV engine settings
# (experiments/ascent-config/engine.json: kv_cache_config dtype fp8, block reuse, mamba snapshot at 32768) and the
# server defaults (max_batch_size 8, max_num_tokens 8192, chunked prefill, decode CUDA graphs up to batch 8):
#   24 prompts = one 32768-token shared prefix (real text) + 2048 unique suffix tokens, 256 greedy tokens (ignore_eos,
#   logprobs=1); c1 = prompts 0-7 one at a time, c2 = prompts 8-15 in pairs, c4 = prompts 16-23 in fours.
#   pass 1 starts from an empty cache (cold prefix on the first request); pass 2 reruns all 24 prompts reusing the
#   cached 32768-token prefix and recurrent snapshot. pass 2 must equal pass 1 (tokens + logprobs) in each arm.
# Decode CUDA-graph replays are counted per graph batch size (CUDAGraphRunner.replay) so the log shows the early-trigger
# path ran under graphs at batch 1/2/4. Last stdout line: JSON with per-prompt sha256 digests.
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

PREFIX, SUFFIX, OSL, N_PER_GROUP = 32768, 2048, 256, 8
GROUPS = (("c1", 1), ("c2", 2), ("c4", 4))


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
    log("engine.json:", json.dumps(engine_json))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    assert kvc.get("dtype") == "fp8", kvc
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=8, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=8), kv_cache_config=KvCacheConfig(**kvc))
    log("LLM ready in %.1fs; decode graphs captured at batch sizes %s" % (time.time() - t0, sorted(captures)))
    tok = llm.tokenizer
    n = N_PER_GROUP * len(GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(n)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)

    def run_pass(tag):
        res = dict()
        for gi, (g, conc) in enumerate(GROUPS):
            cur["phase"] = "%s:%s" % (tag, g)
            for s in range(0, N_PER_GROUP, conc):
                idx = [gi * N_PER_GROUP + s + j for j in range(conc)]
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
                    name = "%s-p%d" % (g, s + j)
                    names.append(name)
                    res[name] = dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, "cached_tokens", None))
                log("%s %s prompts %s: %.2fs cached %s" % (tag, g, idx, dt, [res[x]["cached"] for x in names]))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    p2_eq_p1 = True
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        p2_eq_p1 &= same
        per[k] = dict(ids_sha=digest_ids(p1[k]["ids"]), lp_sha=digest_lp(p1[k]["lp"]), n_tokens=len(p1[k]["ids"]),
                      first8=p1[k]["ids"][:8], cached_p1=p1[k]["cached"], cached_p2=p2[k]["cached"],
                      pass2_equals_pass1=same)
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays if k.startswith("pass")))
    warm_ok = all(per[k]["cached_p2"] is None or per[k]["cached_p2"] >= PREFIX for k in per)
    res = dict(passed=bool(p2_eq_p1 and all(b in dec_bs for b in (1, 2, 4))), pass2_equals_pass1=p2_eq_p1,
               graphs_captured_batch_sizes=sorted(captures), graph_replays=replays,
               decode_graph_batch_sizes_replayed=dec_bs, pass2_prefix_reused_all=warm_ok,
               all_ids_sha=digest_ids([p1[k]["ids"] for k in sorted(p1)]),
               all_lp_sha=hashlib.sha256("".join(per[k]["lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               engine_json=engine_json, prompts="%d x (%d shared prefix + %d unique), %d greedy tokens, logprobs=1; "
               "c1/c2/c4 groups of 8" % (n, PREFIX, SUFFIX, OSL), per_prompt=per,
               compared="pass2 vs pass1 in this arm; compare all_ids_sha/all_lp_sha/per_prompt across the pair arms")
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
