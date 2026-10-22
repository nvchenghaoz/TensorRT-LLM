# End-to-end two-pass exactness replica for gdn-fusion idea 23 (decode post_attention gemma add+RMSNorm + NVFP4
# epilogue), one arm per process; run_gpu_pair runs it in both trees (the base arm via python -c).
# Protocol (plan 364 item 2): 56 prompts = one 32768-token shared prefix (real text) + 2048 unique tokens each;
# c8 = prompts 0-7 in one call, c16 = 8-23, c32 = 24-55; 40 greedy tokens, logprobs=1; max_batch_size 32, decode
# CUDA graphs up to 32, max_num_tokens 8192, chunked prefill, kv_cache_config from experiments/ascent-config/engine.json.
# Pass 1 starts from an empty cache; pass 2 reruns all 56 prompts in the same LLM lifetime.
# Also records, per decode CUDA-graph capture batch size, how many trtllm::fp4_quantize calls (one
# quantize_with_block_size launch each) and fused norm+NVFP4 calls were captured (= launches per replay), and the
# AutoTuner picks of the gate_up nvfp4_gemm (N=34816, K=5120).
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
    cur = dict(phase="load", cap_bs=None)
    cap_counts = dict()
    orig_replay, orig_capture = cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture

    def replay(self, key, *a, **kw):
        k = "%s:bs%d" % (cur["phase"], cgr.KeyType(*key).batch_size)
        replays[k] = replays.get(k, 0) + 1
        return orig_replay(self, key, *a, **kw)

    def capture(self, key, *a, **kw):
        bs = cgr.KeyType(*key).batch_size
        captures.add(bs)
        cur["cap_bs"] = bs
        try:
            return orig_capture(self, key, *a, **kw)
        finally:
            cur["cap_bs"] = None

    cgr.CUDAGraphRunner.replay, cgr.CUDAGraphRunner.capture = replay, capture

    def counted(name, fn):
        def wrapper(*a, **kw):
            if cur["cap_bs"] is not None and torch.cuda.is_current_stream_capturing():
                d = cap_counts.setdefault(str(cur["cap_bs"]), dict())
                d[name] = d.get(name, 0) + 1
            return fn(*a, **kw)
        return wrapper

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401
    torch.ops.trtllm.fp4_quantize = counted("fp4_quantize", torch.ops.trtllm.fp4_quantize)
    has_fused = False
    try:
        from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4 as fg
        if hasattr(fg, "flashinfer_nvfp4_launch"):
            fg.flashinfer_nvfp4_launch = counted("fused_norm_nvfp4", fg.flashinfer_nvfp4_launch)
            has_fused = True
    except Exception as e:  # noqa: BLE001
        log("no fused module:", e)

    engine_json = json.load(open("experiments/ascent-config/engine.json"))
    log("engine.json:", json.dumps(engine_json), "has_fused_op:", has_fused)
    kvc = dict(engine_json.get("kv_cache_config", dict()))
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32, max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32), kv_cache_config=KvCacheConfig(**kvc),
              batch_wait_timeout_ms=BATCH_WAIT_MS)
    log("LLM ready in %.1fs; decode graphs captured at batch sizes %s" % (time.time() - t0, sorted(captures)))
    log("capture counts:", json.dumps(cap_counts, sort_keys=True))
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
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                res["%s-p%d" % (g, j)] = dict(ids=list(c.token_ids), lp=lps,
                                             cached=getattr(o, "cached_tokens", None))
            log("%s %s: %d prompts %.2fs" % (tag, g, conc, dt))
            base += conc
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    torch.cuda.synchronize()
    per = dict()
    n_p2_ne_p1 = 0
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        n_p2_ne_p1 += 0 if same else 1
        per[k] = dict(p1_ids_sha=digest_ids(p1[k]["ids"]), p1_lp_sha=digest_lp(p1[k]["lp"]),
                      p2_ids_sha=digest_ids(p2[k]["ids"]), p2_lp_sha=digest_lp(p2[k]["lp"]),
                      pass2_equals_pass1=same, first6=p1[k]["ids"][:6])
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "e2e_twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    tactics = dict()
    try:
        from tensorrt_llm._torch.autotuner import AutoTuner
        for key, val in AutoTuner.get().profiling_cache.cache.items():
            ks = str(key)
            if "nvfp4_gemm" in ks and "34816" in ks:
                tactics[ks] = [str(val[0]), str(val[1])]
    except Exception as e:  # noqa: BLE001
        tactics["error"] = repr(e)
    dec_bs = sorted(set(int(k.split(":bs")[1]) for k in replays if k.startswith("pass")))
    res = dict(passed=True, has_fused_op=has_fused, pass2_vs_pass1_diverging=n_p2_ne_p1,
               graphs_captured_batch_sizes=sorted(captures), capture_counts_per_graph=cap_counts,
               decode_graph_batch_sizes_replayed=dec_bs, graph_replays=replays,
               all_p1_sha=hashlib.sha256("".join(per[k]["p1_ids_sha"] + per[k]["p1_lp_sha"]
                                                 for k in sorted(per)).encode()).hexdigest(),
               all_p2_sha=hashlib.sha256("".join(per[k]["p2_ids_sha"] + per[k]["p2_lp_sha"]
                                                 for k in sorted(per)).encode()).hexdigest(),
               gate_up_nvfp4_gemm_tactics=tactics, engine_json=engine_json, batch_wait_timeout_ms=BATCH_WAIT_MS,
               osl=OSL, prompts="56 x (32768 shared prefix + 2048 unique), c8/c16/c32 calls, %d greedy tokens" % OSL,
               per_prompt=per,
               compared="compare all_p1_sha/all_p2_sha/per_prompt across the pair arms (exact: equal)")
    print(json.dumps(res), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
