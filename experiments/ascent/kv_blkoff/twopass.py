# Idea 30 exact-branch two-pass predictor (one LLM lifetime per arm), BRACE-FREE for run_gpu_pair python -c.
# 56 prompts = one 32768-token shared prefix + 2048 unique suffix tokens: c8 = 8 prompts at once, c16 = 16, c32 = 32.
# 40 greedy tokens with logprobs. Pass 1 starts with a cold prefix; pass 2 reruns all 56 reusing the cached prefix and
# recurrent snapshot. Prints per-prompt sha256 of token ids and logprobs; compare the two arms of the pair bitwise.
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
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    eng = json.load(open("experiments/ascent-config/engine.json"))
    llm = LLM(model=ckpt, max_batch_size=32, max_num_tokens=8192, max_seq_len=36864, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=KvCacheConfig(**eng["kv_cache_config"]))
    log("LLM ready")
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
    per = dict()
    n_p2_ne_p1 = 0
    for k in p1:
        same = p1[k]["ids"] == p2[k]["ids"] and p1[k]["lp"] == p2[k]["lp"]
        n_p2_ne_p1 += 0 if same else 1
        per[k] = dict(p1_ids=sha_ids(p1[k]["ids"]), p1_lp=sha_lp(p1[k]["lp"]), p2_ids=sha_ids(p2[k]["ids"]),
                      p2_lp=sha_lp(p2[k]["lp"]), n=len(p1[k]["ids"]))
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "twopass_tokens.json"), "w") as f:
            json.dump(dict(pass1=p1, pass2=p2), f)
    keys = sorted(per)
    res = dict(passed=True, prompts=len(per), pass2_vs_pass1_diverging=n_p2_ne_p1,
               all_p1_ids_sha=hashlib.sha256("".join(per[k]["p1_ids"] for k in keys).encode()).hexdigest(),
               all_p1_lp_sha=hashlib.sha256("".join(per[k]["p1_lp"] for k in keys).encode()).hexdigest(),
               all_p2_ids_sha=hashlib.sha256("".join(per[k]["p2_ids"] for k in keys).encode()).hexdigest(),
               all_p2_lp_sha=hashlib.sha256("".join(per[k]["p2_lp"] for k in keys).encode()).hexdigest(),
               per_prompt=per, compared="compare per_prompt and all_* digests across the pair arms (bitwise)")
    llm.shutdown()
    print(json.dumps(res), flush=True)
    os._exit(0)


main()
