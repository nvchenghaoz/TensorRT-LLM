"""Load the model with the LLM API and run a few 1-output-token prefills.

Used under profile_timeline to count per-forward kernel launches, and as a
quick end-to-end smoke test (greedy token per prompt printed).
The __main__ guard matters: the LLM API spawns its worker by re-importing
this module, and an unguarded LLM() in the worker deadlocks start-up.
"""
import json
import os
import sys
import threading
import time


def _heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


def main():
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    threading.Thread(target=_heartbeat, daemon=True).start()

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    ck = sys.argv[1]
    isl = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    llm = LLM(
        model=ck,
        max_num_tokens=8192,
        max_seq_len=isl + 64,
        max_batch_size=16,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.6, enable_block_reuse=False),
    )
    print("loaded", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)

    def prompt(seed):
        ids = [(seed * 7919 + i * 104729) % 9973 + 1000 for i in range(isl)]
        return {"prompt_token_ids": ids}

    llm.generate([prompt(99)], sp)  # warmup
    outs = []
    for i in range(3):
        t = time.time()
        o = llm.generate([prompt(i)], sp)
        outs.append({"tok": o[0].outputs[0].token_ids, "s": time.time() - t})
        print(i, json.dumps(outs[-1], default=str), flush=True)
    llm.shutdown()
    print(json.dumps({"passed": True, "outs": outs}, default=str))


if __name__ == "__main__":
    main()
