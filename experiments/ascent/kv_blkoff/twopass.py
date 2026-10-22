# Idea 30 exact-branch two-pass predictor (one LLM lifetime per arm), BRACE-FREE for run_gpu_pair python -c.
# 56 prompts = one 32768-token shared prefix + 2048 unique suffix tokens: c8 = 8 prompts at once, c16 = 16, c32 = 32.
# 40 greedy tokens with logprobs. batch_wait_timeout_ms (TP_BATCH_WAIT_MS, default 500) makes the executor admit each
# group in one iteration, so batch composition does not depend on host timing (it otherwise splits a group across
# iterations at random, which changes numerics independently of the code under test).
# Pass 1 starts with a cold prefix; pass 2 reruns all 56 reusing the cached prefix and recurrent snapshot; pass 3
# reruns pass 2 in the same lifetime with kv_cache_manager_v2._BLOCK_OFFSET_SNAPSHOT_COPY = False (in the base tree the
# flag does not exist, so pass 3 is an in-process A/A of pass 2). Prints per-prompt sha256 of ids and logprobs per pass;
# compare the two arms of the pair bitwise, and pass 3 vs pass 2 within each arm.
import hashlib
import json
import os
import random
import struct
import sys
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
T0 = time.time()
WAIT_MS = float(os.environ.get("TP_BATCH_WAIT_MS", "500"))


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def hb():
    while True:
        time.sleep(30)
        log("heartbeat")


threading.Thread(target=hb, daemon=True).start()


def sha_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def sha_lp(lps):
    return hashlib.sha256(b"".join(struct.pack("<d", float("nan") if x is None else x) for x in lps)).hexdigest()


def main():
    ckpt = sys.argv[1]
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    has_flag = hasattr(kvm, "_BLOCK_OFFSET_SNAPSHOT_COPY")
    eng = json.load(open("experiments/ascent-config/engine.json"))
    llm = LLM(model=ckpt, max_batch_size=32, max_num_tokens=8192, max_seq_len=36864, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=KvCacheConfig(**eng["kv_cache_config"]), batch_wait_timeout_ms=WAIT_MS)
    log("LLM ready; snapshot flag in tree:", has_flag, "batch_wait_timeout_ms", WAIT_MS)
    rng = random.Random(4242)
    prefix = [rng.randrange(1000, 100000) for _ in range(32768)]
    groups = (("c8", 8), ("c16", 16), ("c32", 32))
    prompts = dict()
    for g, n in groups:
        prompts[g] = [prefix + [rng.randrange(1000, 100000) for _ in range(2048)] for _ in range(n)]
    sp = SamplingParams(max_tokens=40, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)

    def run_pass(tag):
        res = dict()
        for g, n in groups:
            t = time.time()
            outs = llm.generate(prompts[g], sp, use_tqdm=False)
            for j, o in enumerate(outs):
                c = o.outputs[0]
                lps = []
                for tid, d in zip(c.token_ids, c.logprobs or []):
                    lp = d.get(tid) if isinstance(d, dict) else None
                    lps.append(float(lp.logprob) if lp is not None else None)
                res["%s-p%d" % (g, j)] = dict(ids=list(c.token_ids), lp=lps)
            log(tag, g, "%.1fs" % (time.time() - t))
        return res

    p1 = run_pass("pass1")
    p2 = run_pass("pass2")
    if has_flag:
        kvm._BLOCK_OFFSET_SNAPSHOT_COPY = False
    p3 = run_pass("pass3")
    if has_flag:
        kvm._BLOCK_OFFSET_SNAPSHOT_COPY = True

    def same(a, b, k):
        return a[k]["ids"] == b[k]["ids"] and a[k]["lp"] == b[k]["lp"]

    per = dict()
    for k in p1:
        per[k] = "|".join([sha_ids(p1[k]["ids"]), sha_lp(p1[k]["lp"]), sha_ids(p2[k]["ids"]), sha_lp(p2[k]["lp"]),
                           sha_ids(p3[k]["ids"]), sha_lp(p3[k]["lp"])])
    keys = sorted(per)
    p2_vs_p1 = [k for k in keys if not same(p1, p2, k)]
    p3_vs_p2 = [k for k in keys if not same(p2, p3, k)]
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2, pass3=p3), f)

    def digest(p, field):
        return hashlib.sha256(json.dumps([p[k][field] for k in keys]).encode()).hexdigest()

    res = dict(passed=len(p3_vs_p2) == 0, prompts=len(per), snapshot_flag_in_tree=has_flag,
               batch_wait_timeout_ms=WAIT_MS,
               pass2_vs_pass1_diverging=len(p2_vs_p1), pass2_vs_pass1_set=p2_vs_p1,
               pass3_vs_pass2_diverging=len(p3_vs_p2), pass3_vs_pass2_set=p3_vs_p2,
               p1_ids=digest(p1, "ids"), p1_lp=digest(p1, "lp"), p2_ids=digest(p2, "ids"), p2_lp=digest(p2, "lp"),
               p3_ids=digest(p3, "ids"), p3_lp=digest(p3, "lp"),
               per_prompt_p1ids_p1lp_p2ids_p2lp_p3ids_p3lp=per,
               compared="cross-arm: p1/p2 digests and per_prompt must be bitwise equal between base and candidate; "
                        "in-arm: pass3 (candidate: stock copy path) vs pass2 (candidate: snapshot path)")
    llm.shutdown()
    print(json.dumps(res), flush=True)
    os._exit(0)


main()
