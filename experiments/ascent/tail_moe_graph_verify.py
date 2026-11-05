"""4-GPU VERIFY run for the DeepSeek-V4.1 128-row tail MoE-window graph (idea 156/227).

Runs the real checkpoint with the LLM API under the frozen protocol's server settings
(TP4/EP4, no attention DP, bounded replay on by the model defaults) with
TRTLLM_DSV4_TAIL_MOE_GRAPH=1 and TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY=N. It sends STEPS prompts
one at a time, each a fresh 2200-token context (one context-only EXTEND step whose L20-39
replay tail has 128 rows) with max_tokens=2. Per layer the first eligible step warms eagerly,
the second captures, later ones replay; for the first N replays each rank also runs main's
eager window twice on the same inputs and logs graph-vs-eager and eager-vs-eager
(control) comparisons, then one '[TAIL_MOE_GRAPH VERIFY SUMMARY]' line per rank.

Modes:
  served  - the served MoE config (CUTLASS FC2 + fused finalize, whose epilogue scatters
            top-k rows with atomics, so main's eager window is not bitwise run-to-run).
  determ  - moe_config.disable_finalize_fusion=True (the documented deterministic mode):
            graph vs eager must be bitwise equal, eager-vs-eager too.
  both    - served then determ, each in its own subprocess (default).

usage: python tail_moe_graph_verify.py <checkpoint> [steps=10] [verify=8] [mode=both]
"""

import json
import os
import random
import subprocess
import sys
import time


def _layout():
    """MoE layout: env TAIL_VERIFY_MOE_TP / TAIL_VERIFY_MOE_EP (default: main's EP4, moe_tp unset)."""
    moe_tp = os.environ.get("TAIL_VERIFY_MOE_TP")
    moe_ep = int(os.environ.get("TAIL_VERIFY_MOE_EP", "4"))
    return (int(moe_tp) if moe_tp else None), moe_ep


def _mapping_receipt(llm, moe_tp, moe_ep):
    from tensorrt_llm.mapping import Mapping

    for key in ("moe_tensor_parallel_size", "moe_expert_parallel_size", "tensor_parallel_size"):
        print(f"[verify] llm.args.{key} = {getattr(llm.args, key, '?')}", flush=True)
    for r in range(4):
        m = Mapping(world_size=4, rank=r, tp_size=4, moe_tp_size=moe_tp if moe_tp else 1,
                    moe_ep_size=moe_ep)
        experts_per_rank = 384 // m.moe_ep_size
        print(f"[verify] MAPPING rank {r}: tp_rank={m.tp_rank} moe_tp_size={m.moe_tp_size} "
              f"moe_ep_size={m.moe_ep_size} moe_tp_rank={m.moe_tp_rank} "
              f"moe_ep_rank={m.moe_ep_rank} slot_start={experts_per_rank * m.moe_ep_rank} "
              f"local_experts={experts_per_rank} inter_per_rank={2304 // m.moe_tp_size}",
              flush=True)


def run_one(ckpt, steps, verify, mode):
    env = {"TRTLLM_DSV4_TAIL_MOE_GRAPH": "1", "TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY": str(verify)}
    os.environ.update(env)
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    moe_tp, moe_ep = _layout()
    print(f"[verify] mode={mode} steps={steps} verify={verify} "
          f"moe_tensor_parallel_size={moe_tp} moe_expert_parallel_size={moe_ep}", flush=True)
    t0 = time.time()
    extra = {}
    if moe_tp is not None:
        extra["moe_tensor_parallel_size"] = moe_tp
    if mode == "determ":
        extra["moe_config"] = MoeConfig(disable_finalize_fusion=True)
    # The frozen protocol's server settings (harness.server), so memory sizing matches serving.
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        moe_expert_parallel_size=moe_ep,
        enable_attention_dp=False,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        env_overrides=env,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.5,
                                      tokens_per_block=128),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
        max_batch_size=8,
        max_num_tokens=8192,
        max_seq_len=36864,
        **extra,
    )
    print(f"[verify] mode={mode} LLM ready in {time.time() - t0:.0f}s", flush=True)
    _mapping_receipt(llm, moe_tp, moe_ep)
    rng = random.Random(0)
    vocab = 100000  # tokenizer.vocab_size raises NotImplementedError here; ids < 100000 are valid
    sp = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
    out = []
    for i in range(steps):
        ids = [rng.randrange(1000, vocab) for _ in range(2200)]
        t1 = time.time()
        r = llm.generate([ids], sp)
        out.append({"step": i, "s": round(time.time() - t1, 3),
                    "tokens": list(r[0].outputs[0].token_ids)})
        print(f"[verify] mode={mode} step {i}: {out[-1]}", flush=True)
    llm.shutdown()
    return out


def main():
    ckpt = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    verify = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    mode = sys.argv[4] if len(sys.argv) > 4 else "both"
    if mode == "both":
        rcs = {}
        for m in ("served", "determ"):
            print(f"[verify] ===== starting mode {m} =====", flush=True)
            rcs[m] = subprocess.call([sys.executable, os.path.abspath(__file__), ckpt,
                                      str(steps), str(verify), m])
            print(f"[verify] ===== mode {m} exit {rcs[m]} =====", flush=True)
        print(json.dumps({"passed": all(v == 0 for v in rcs.values()), "exit_codes": rcs,
                          "note": "read the job log for [TAIL_MOE_GRAPH VERIFY SUMMARY] "
                                  "per rank and mode"}), flush=True)
        return
    out = run_one(ckpt, steps, verify, mode)
    print(json.dumps({"passed": True, "mode": mode, "steps": out}), flush=True)


if __name__ == "__main__":
    main()
