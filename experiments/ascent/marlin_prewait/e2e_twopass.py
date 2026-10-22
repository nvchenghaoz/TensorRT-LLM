# Two-pass end-to-end exactness predictor for branch 21 (idea 28), one arm per process (NOT the frozen check).
# BRACE-FREE on purpose: run_gpu_pair passes it verbatim via python -c to both trees (the base tree lacks this file).
# Protocol shapes: 56 prompts = one 32768-token shared prefix (real text) + 2048 unique suffix tokens each;
# groups c8 (prompts 0-7 in one call of 8), c16 (8-23, one call of 16), c32 (24-55, one call of 32);
# 40 greedy tokens with logprobs=1; max_batch_size 32, decode CUDA graphs up to 32, max_num_tokens 8192, chunked
# prefill, experiments/ascent-config/engine.json kv_cache_config (FP8 KV, block reuse, snapshot at 32768).
# pass 1 starts from an empty cache; pass 2 reruns all 56 prompts reusing the cached prefix and snapshot.
# Compare per-prompt ids/logprob digests across the pair arms (exact change: equal in pass 1 and in pass 2).
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
    log("TRTLLM_ENABLE_PDL=%s TRTLLM_MARLIN_PREWAIT_B=%s batch_wait_timeout_ms=%s osl=%d" % (
        os.environ.get("TRTLLM_ENABLE_PDL"), os.environ.get("TRTLLM_MARLIN_PREWAIT_B"), BATCH_WAIT_MS, OSL))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    assert kvc.get("dtype") == "fp8", kvc
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
            idx = list(range(base, base + conc))
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
            log("%s %s: %d prompts %.2fs" % (tag, g, conc, dt))
            base += conc
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    p2_eq_p1 = 0
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        p2_eq_p1 += int(same)
        per[k] = dict(p1_ids_sha=digest_ids(p1[k]["ids"])[:16], p1_lp_sha=digest_lp(p1[k]["lp"])[:16],
                      p2_ids_sha=digest_ids(p2[k]["ids"])[:16], p2_lp_sha=digest_lp(p2[k]["lp"])[:16],
                      n_tokens=len(p1[k]["ids"]), pass2_equals_pass1=same)
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays if k.startswith("pass")))
    res = dict(pass2_equals_pass1_prompts=p2_eq_p1, n_prompts=len(p1),
               graphs_captured_batch_sizes=sorted(captures), graph_replays=replays,
               decode_graph_batch_sizes_replayed=dec_bs,
               pass1_all_ids_sha=digest_ids([p1[k]["ids"] for k in sorted(p1)]),
               pass1_all_lp_sha=hashlib.sha256("".join(per[k]["p1_lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               pass2_all_ids_sha=digest_ids([p2[k]["ids"] for k in sorted(p2)]),
               pass2_all_lp_sha=hashlib.sha256("".join(per[k]["p2_lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               engine_json=engine_json, batch_wait_timeout_ms=BATCH_WAIT_MS, osl=OSL,
               prompts="%d x (%d shared prefix + %d unique), %d greedy tokens, logprobs=1; c8@8, c16@16, c32@32" % (
                   n, PREFIX, SUFFIX, OSL), per_prompt=per,
               compared="compare pass1/pass2 digests and per_prompt across the pair arms")
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
