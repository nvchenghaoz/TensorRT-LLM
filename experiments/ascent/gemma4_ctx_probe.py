"""One isl8k@1 context step of Gemma4 inside a cudaProfilerStart/Stop region (for profile_timeline).

Default engine settings (FA2 + FP8 KV on SM120), natural-text prompt in the chat template, max_tokens=1,
so the region holds exactly one 8000-token prefill forward (60 decoder layers). Prints the per-kernel-name
launch counts are read from the trace; stdout reports the first token and its top-5 logprobs.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ckpt = sys.argv[1]
    isl = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    os.environ["CKPT"] = ckpt
    import gemma4_prep_ab as ab  # noqa: E402
    ab.CKPT = ckpt
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    t0 = time.time()
    llm = LLM(model=ckpt, max_num_tokens=8192, max_batch_size=64,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.6, enable_block_reuse=False))
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    nat = ab.NaturalPrompts(81)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
    for _ in range(2):
        llm.generate([nat.make(isl)], sp, use_tqdm=False)
    torch.cuda.synchronize()
    print("warmup done", flush=True)
    p = nat.make(isl)
    torch.cuda.cudart().cudaProfilerStart()
    out = llm.generate([p], sp, use_tqdm=False)[0]
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    c = out.outputs[0]
    lp = {str(k): round(float(v.logprob), 5) for k, v in (c.logprobs[0] if c.logprobs else {}).items()}
    llm.shutdown()
    print(json.dumps({"isl": len(p), "token": list(c.token_ids), "top5": lp}), flush=True)


if __name__ == "__main__":
    main()
