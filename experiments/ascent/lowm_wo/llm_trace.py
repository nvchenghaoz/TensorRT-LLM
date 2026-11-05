"""Idea 233 acceptance item 4: in-process Kineto decode replay at batch 1 and 4 (4 GPUs, TP4/EP4, protocol server
settings, CUDA graphs [1,2,4,8] padded), then the route audit of trace_job.py (K2'/K3'/Triton K1/b12x counts per
step on every rank, K2'->K3' PDL overlap on rank 0).

The frozen profile adapter is never on disk in a worker job (pitfall 1530), so this drives the same
PyExecutor runtime profiler (/start_profile's backend, LLM.start_profile) from the LLM API directly. Prompts are
synthetic token ids (decode M is what matters for the WO route, not the prefix length). Not timing evidence.
Usage: llm_trace.py <checkpoint>. Last stdout line: JSON summary.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEPS = 40


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ckpt = sys.argv[1]
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts")).resolve()
    root = art / "profile" / "traces"
    root.mkdir(parents=True, exist_ok=True)
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    t0 = time.time()
    llm = LLM(model=ckpt, tensor_parallel_size=4, moe_expert_parallel_size=4, pipeline_parallel_size=1,
              enable_attention_dp=False, enable_chunked_prefill=True,
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.5,
                                            tokens_per_block=128),
              cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
              max_batch_size=8, max_num_tokens=8192, max_seq_len=36864, trust_remote_code=True)
    log(f"LLM up in {time.time() - t0:.0f}s")
    import random
    rng = random.Random(233)

    def prompt(n=512):
        return [rng.randrange(1000, 60000) for _ in range(n)]

    sp_warm = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    for c in (1, 2, 4, 8):
        outs = llm.generate([prompt() for _ in range(c)], sp_warm)
        log(f"warmup c{c}: {[len(o.outputs[0].token_ids) for o in outs]}")
    control = {}
    for c in (1, 4):
        d = root / f"decode_c{c}"
        d.mkdir(parents=True, exist_ok=True)
        sp = SamplingParams(max_tokens=1200, temperature=0.0, ignore_eos=True)
        futs = [llm.generate_async(prompt(), sp) for _ in range(c)]
        time.sleep(3.0)  # prefill of 512-token prompts is done long before; all c rows decode
        llm.start_profile(output_dir=str(d), num_steps=STEPS, activities=["GPU"])
        log(f"decode_c{c}: profile started")
        res = [f.result() for f in futs]
        try:
            llm.stop_profile()
        except Exception as exc:  # noqa: BLE001  (num_steps already closed the window)
            log(f"stop_profile: {exc!r}")
        deadline = time.time() + 300
        while time.time() < deadline and len(list(d.glob("*.json*"))) < 4:
            time.sleep(5)
        time.sleep(3)
        files = sorted(str(p.name) for p in d.glob("*.json*"))
        control[c] = {"tokens": [len(r.outputs[0].token_ids) for r in res], "files": files}
        log(f"decode_c{c}: {control[c]}")
    llm.shutdown()
    for c in (1, 4):
        for p in (root / f"decode_c{c}").glob("*.json*"):
            log("trace", p.name, p.stat().st_size)
    import trace_job
    trace_job.STEPS = STEPS
    sys.argv = [sys.argv[0], ckpt, str(root)]
    print(json.dumps({"control": control}), flush=True)
    trace_job.main()


if __name__ == "__main__":
    main()
