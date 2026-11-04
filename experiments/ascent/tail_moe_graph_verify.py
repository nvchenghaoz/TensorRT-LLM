"""4-GPU VERIFY run for the DeepSeek-V4.1 128-row tail MoE-window graph (idea 156, stage 3).

Runs the real checkpoint with the LLM API (TP4/EP4, no attention DP, bounded replay on by the
model defaults) with TRTLLM_DSV4_TAIL_MOE_GRAPH=1 and TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY=N
(env_overrides, applied before the workers import the model). It sends STEPS prompts one at a
time, each a fresh ~2.2k-token context (one context-only EXTEND step whose L20-39 replay tail
has 128 rows) with max_tokens=2. Per layer the first eligible step warms eagerly, the second
captures, later ones replay; for the first N replays each rank also runs the eager window and
logs '[TAIL_MOE_GRAPH VERIFY] ... equal|MISMATCH' plus one '[TAIL_MOE_GRAPH VERIFY SUMMARY]'
line per rank. The workers log to the job's stdout, so read the job log for those lines.

usage: python tail_moe_graph_verify.py <checkpoint> [steps=10] [verify=8]
"""

import json
import os
import random
import sys
import time


def main():
    ckpt = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    verify = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    env = {"TRTLLM_DSV4_TAIL_MOE_GRAPH": "1", "TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY": str(verify)}
    os.environ.update(env)
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    from tensorrt_llm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        enable_attention_dp=False,
        env_overrides=env,
        max_num_tokens=8192,
        max_seq_len=40960,
    )
    print(f"[verify] LLM ready in {time.time() - t0:.0f}s", flush=True)
    tok = llm.tokenizer
    rng = random.Random(0)
    vocab = tok.vocab_size if hasattr(tok, "vocab_size") else 100000
    sp = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
    out = []
    for i in range(steps):
        ids = [rng.randrange(1000, min(vocab, 100000)) for _ in range(2200)]
        t1 = time.time()
        r = llm.generate([ids], sp)
        out.append({"step": i, "s": round(time.time() - t1, 3),
                    "tokens": list(r[0].outputs[0].token_ids)})
        print(f"[verify] step {i}: {out[-1]}", flush=True)
    llm.shutdown()
    print(json.dumps({"passed": True, "steps": out,
                      "note": "read the job log for [TAIL_MOE_GRAPH VERIFY SUMMARY] per rank"}),
          flush=True)


if __name__ == "__main__":
    main()
