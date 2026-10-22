# End-to-end two-pass exactness replica for idea 38 (GDN T1 + gated norm
# fusion), one arm per process. BRACE-FREE on purpose: passed inline with
# python -c to run_gpu_pair, so it runs unchanged in both trees.
# Mirrors the check adapter's two-pass protocol on the benchmark shape with
# the engine settings of experiments/ascent-config/engine.json (FP8 KV, block
# reuse, mamba snapshot at 32768) and the harness base engine (max_batch_size
# 32, decode CUDA graphs up to 32, max_num_tokens 8192, chunked prefill):
#   56 prompts = one 32768-token shared prefix (real text) + 2048 unique
#   suffix tokens, 40 greedy tokens (ignore_eos, logprobs=1);
#   c8 = prompts 0-7 together, c16 = 8-23 together, c32 = 24-55 together.
#   Pass 1 starts from an empty cache; pass 2 reruns all 56 in the same LLM
#   lifetime, reusing the cached prefix and recurrent snapshot.
# Counts decode CUDA-graph replays per batch size and how many times the fused
# T1 + norm wrapper was entered (at capture/eager) in this tree.
# Last stdout line: JSON with per-prompt sha256 digests (compare across arms).
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
BATCH_WAIT_MS = float(os.environ.get("E2E_BATCH_WAIT_MS", "300"))
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
    return hashlib.sha256(b"".join(
        struct.pack("<d", float("nan") if x is None else x)
        for x in lps)).hexdigest()


def main(ckpt):
    heartbeat()
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules.mamba import gdn_mixer
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    replays, captures = dict(), set()
    fused_calls = dict(n=0)
    cur = dict(phase="load")
    orig_replay, orig_capture = cgr.CUDAGraphRunner.replay, \
        cgr.CUDAGraphRunner.capture

    def replay(self, key, *a, **kw):
        k = "%s:bs%d" % (cur["phase"], cgr.KeyType(*key).batch_size)
        replays[k] = replays.get(k, 0) + 1
        return orig_replay(self, key, *a, **kw)

    def capture(self, key, *a, **kw):
        captures.add(cgr.KeyType(*key).batch_size)
        return orig_capture(self, key, *a, **kw)

    cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture = replay, capture
    fn = getattr(gdn_mixer, "fused_sigmoid_gating_delta_rule_update_t1_norm",
                 None)
    if fn is not None:

        def counted(*a, **kw):
            fused_calls["n"] += 1
            return fn(*a, **kw)

        gdn_mixer.fused_sigmoid_gating_delta_rule_update_t1_norm = counted

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    log("engine.json:", json.dumps(engine_json))
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    assert kvc.get("dtype") == "fp8", kvc
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32,
              max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=BATCH_WAIT_MS)
    log("LLM ready in %.1fs; decode graphs captured at batch sizes %s; "
        "fused T1+norm entries so far %d" %
        (time.time() - t0, sorted(captures), fused_calls["n"]))
    tok = llm.tokenizer
    n = sum(c for _, c in GROUPS)
    ids = corpus_ids(tok, PREFIX + n * SUFFIX)
    prefix = ids[:PREFIX]
    prompts = [prefix + ids[PREFIX + i * SUFFIX:PREFIX + (i + 1) * SUFFIX]
               for i in range(n)]
    sp = SamplingParams(max_tokens=OSL, temperature=0.0, top_k=1,
                        ignore_eos=True, logprobs=1)

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
            names = []
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                name = "%s-p%d" % (g, j)
                names.append(name)
                res[name] = dict(ids=list(c.token_ids), lp=lps,
                                 cached=getattr(o, "cached_tokens", None))
            log("%s %s: %.2fs cached %s" %
                (tag, g, dt, [res[x]["cached"] for x in names]))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    p2_diff = []
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        if not same:
            p2_diff.append(k)
        per[k] = dict(p1_ids_sha=digest_ids(p1[k]["ids"]),
                      p1_lp_sha=digest_lp(p1[k]["lp"]),
                      p2_ids_sha=digest_ids(p2[k]["ids"]),
                      p2_lp_sha=digest_lp(p2[k]["lp"]),
                      ids=p1[k]["ids"], lp=p1[k]["lp"], ids2=p2[k]["ids"],
                      lp2=p2[k]["lp"], cached_p2=p2[k]["cached"])
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays
                        if k.startswith("pass")))
    keys = sorted(per)
    res = dict(
        passed=True, fusion_in_tree=fn is not None,
        fused_t1_norm_entries=fused_calls["n"],
        pass2_vs_pass1_diverging=p2_diff,
        graphs_captured_batch_sizes=sorted(captures), graph_replays=replays,
        decode_graph_batch_sizes_replayed=dec_bs,
        pass1_all_ids_sha=digest_ids([p1[k]["ids"] for k in keys]),
        pass1_all_lp_sha=hashlib.sha256("".join(
            per[k]["p1_lp_sha"] for k in keys).encode()).hexdigest(),
        pass2_all_ids_sha=digest_ids([p2[k]["ids"] for k in keys]),
        pass2_all_lp_sha=hashlib.sha256("".join(
            per[k]["p2_lp_sha"] for k in keys).encode()).hexdigest(),
        engine_json=engine_json, batch_wait_timeout_ms=BATCH_WAIT_MS,
        osl=OSL, per_prompt=per,
        prompts="56 x (32768 shared prefix + 2048 unique), %d greedy tokens,"
        " logprobs=1; c8@8, c16@16, c32@32; pass 2 same lifetime" % OSL)
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
