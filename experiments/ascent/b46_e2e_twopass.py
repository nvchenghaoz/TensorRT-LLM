# Branch 46 (idea 80) end-to-end two-pass greedy parity replica at c8/c16/c32, one arm per process.
# BRACE-FREE on purpose: run_gpu_pair inlines it in the base tree via a bash heredoc (the base lacks this file).
# Engine: max_seq_len 36864, max_batch_size 32, decode CUDA graphs up to 32, max_num_tokens 8192, chunked prefill,
# kv_cache_config from experiments/ascent-config/engine.json (FP8 KV, block reuse, recurrent snapshot at 32768),
# overlap scheduler on (default), so the steady-decode fast prepare (_apply_steady_gen_fast_prepare) runs.
# Prompts: 56 = one 32768-token shared prefix (real text) + 2048 unique suffix tokens; greedy, ignore_eos,
# logprobs=1, E2E_OSL tokens (default 64). c8 = prompts 0-7 in one call, c16 = 8-23, c32 = 24-55.
# Pass 1 starts cold (the first request primes the prefix); pass 2 reruns all 56 reusing the cached prefix.
# Each arm: pass 2 must equal pass 1 (tokens + logprobs). Across arms: compare all_ids_sha / all_lp_sha / per_prompt.
# Counts fast-prepare calls per group and decode graph replays per batch size to show the changed path ran.
# Usage: python b46_e2e_twopass.py <checkpoint>
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
OSL = int(os.environ.get("E2E_OSL", "64"))
BATCH_WAIT_MS = float(os.environ.get("E2E_BATCH_WAIT_MS", "200"))
GROUPS = (("c8", 8), ("c16", 16), ("c32", 32))
RUNNER = "tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py"


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
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as drunner
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    tree_op = "torch.gather(cached int64)" if "previous_slots_i64" in open(RUNNER).read() else "index_select(main)"
    log("tree op:", tree_op)
    replays = dict()
    fast = dict()
    cur = dict(phase="load")
    orig_replay = cgr.CUDAGraphRunner.replay
    orig_fast = drunner.DecoderRunner._apply_steady_gen_fast_prepare

    def replay(self, key, *a, **kw):
        k = "%s:bs%d" % (cur["phase"], cgr.KeyType(*key).batch_size)
        replays[k] = replays.get(k, 0) + 1
        return orig_replay(self, key, *a, **kw)

    def fast_prepare(self, *a, **kw):
        fast[cur["phase"]] = fast.get(cur["phase"], 0) + 1
        return orig_fast(self, *a, **kw)

    cgr.CUDAGraphRunner.replay = replay
    drunner.DecoderRunner._apply_steady_gen_fast_prepare = fast_prepare

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    log("engine.json:", json.dumps(engine_json))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    assert kvc.get("dtype") == "fp8", kvc
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=BATCH_WAIT_MS)
    log("LLM ready in %.1fs" % (time.time() - t0))
    tok = llm.tokenizer
    n = sum(c for _, c in GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX] for i in range(n)]
    inputs_sha = digest_ids(prompts)
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)

    def run_pass(tag):
        res = dict()
        base = 0
        for g, conc in GROUPS:
            cur["phase"] = "%s:%s" % (tag, g)
            idx = list(range(base, base + conc))
            base += conc
            t = time.time()
            outs = llm.generate([prompts[i] for i in idx], sp, use_tqdm=False)
            dt = time.time() - t
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                res["%s-p%d" % (g, j)] = dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, "cached_tokens", None))
            log("%s %s: %.2fs fast_prepare calls %s" % (tag, g, dt, fast.get(cur["phase"], 0)))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    p2_eq_p1 = True
    n_diff = 0
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        p2_eq_p1 &= same
        n_diff += 0 if same else 1
        per[k] = dict(ids_sha=digest_ids(p1[k]["ids"]), lp_sha=digest_lp(p1[k]["lp"]),
                      ids2_sha=digest_ids(p2[k]["ids"]), n_tokens=len(p1[k]["ids"]), first8=p1[k]["ids"][:8],
                      cached_p1=p1[k]["cached"], cached_p2=p2[k]["cached"], pass2_equals_pass1=same)
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "b46_e2e_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays if k.startswith("pass")))
    fast_ok = all(fast.get("%s:%s" % (t, g), 0) > 0 for t in ("pass1", "pass2") for g, _ in GROUPS)
    res = dict(passed=bool(p2_eq_p1 and fast_ok), tree_op=tree_op, pass2_equals_pass1=p2_eq_p1,
               pass2_vs_pass1_diverging=n_diff, fast_prepare_calls=fast, fast_prepare_ran_every_group=fast_ok,
               graph_replays=replays, decode_graph_batch_sizes_replayed=dec_bs,
               all_ids_sha=digest_ids([p1[k]["ids"] for k in sorted(p1)]),
               all_ids2_sha=digest_ids([p2[k]["ids"] for k in sorted(p2)]),
               all_lp_sha=hashlib.sha256("".join(per[k]["lp_sha"] for k in sorted(per)).encode()).hexdigest(),
               inputs_sha=inputs_sha, engine_json=engine_json, batch_wait_timeout_ms=BATCH_WAIT_MS, osl=OSL,
               per_prompt=per,
               compared="pass2 vs pass1 in this arm; compare all_ids_sha/all_lp_sha/per_prompt across pair arms")
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
