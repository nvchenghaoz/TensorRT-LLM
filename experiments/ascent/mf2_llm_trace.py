"""Load the model with the LLM API and run a few 1-output-token prefills.

Used under profile_timeline to count per-forward kernel launches, and as a
quick end-to-end smoke test (greedy token per prompt printed).
"""
import json
import sys
import time

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

ck = sys.argv[1]
isl = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
llm = LLM(
    model=ck,
    max_num_tokens=8192,
    max_seq_len=9000,
    max_batch_size=64,
    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.8, enable_block_reuse=False),
)
print("loaded", flush=True)
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
words = "the quick brown fox jumps over a lazy dog while seven bright stars watch".split()


def prompt(seed):
    ids = [(seed * 7919 + i * 104729) % 9973 + 1000 for i in range(isl)]
    return {"prompt_token_ids": ids}


# warmup
llm.generate([prompt(99)], sp)
outs = []
for i in range(3):
    t = time.time()
    o = llm.generate([prompt(i)], sp)
    outs.append({"tok": o[0].outputs[0].token_ids, "lp": [list(d.items())[:5] if isinstance(d, dict) else str(d) for d in (o[0].outputs[0].logprobs or [])][:1], "s": time.time() - t})
    print(i, json.dumps(outs[-1], default=str), flush=True)
print(json.dumps({"passed": True, "outs": outs}, default=str))
