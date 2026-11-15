"""Load Gemma4 with default engine settings, run a few prefill steps, print greedy tokens.

Used under profile_timeline to count per-layer QKV prep kernels in a real step,
and in a base/branch pair to compare greedy tokens and top logprobs.
"""
import json
import sys
import time

import torch


def main():
    ckpt = sys.argv[1]
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    t0 = time.time()
    llm = LLM(model=ckpt, max_num_tokens=8192, max_batch_size=64,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5,
                                            enable_block_reuse=False))
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
             "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
    prompts = []
    for i, n in enumerate((300, 1000, 2000)):
        body = " ".join(words[(i * 7 + j * 3) % len(words)] for j in range(n))
        prompts.append(f"Here is a list of words: {body}\nQuestion: what is the capital of France? "
                       f"Answer in one sentence.")
    prompts.append("Explain in two sentences why the sky is blue.")
    sp = SamplingParams(max_tokens=16, temperature=0.0, logprobs=3)
    # warmup
    llm.generate(prompts[:1], SamplingParams(max_tokens=2, temperature=0.0))
    torch.cuda.synchronize()
    print("warmup done", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    if len(sys.argv) > 2 and sys.argv[2] == "sequential":
        # One request per step: deterministic step composition across runs.
        outs = [llm.generate([p], sp)[0] for p in prompts]
    else:
        outs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    res = []
    for o in outs:
        c = o.outputs[0]
        lp = []
        for d in (c.logprobs or []):
            lp.append({str(k): round(float(v.logprob), 4) for k, v in d.items()})
        res.append({"tokens": list(c.token_ids), "text": c.text, "top_logprobs_first4": lp})
        print(json.dumps(res[-1]), flush=True)
    llm.shutdown()
    print(json.dumps({"outputs": [r["tokens"] for r in res],
                      "first_token_top3": [r["top_logprobs_first4"][:1] for r in res]}))


if __name__ == "__main__":
    main()
