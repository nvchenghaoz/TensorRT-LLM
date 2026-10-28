"""Idea 82: one steady served-config t2i request (T=80) under cudaProfilerStart/Stop for
profile_timeline capture=region (in-model nsys per arm, pitfall 1223).
Arm from argv[2]: TRTLLM_FLUX3_SM120_QKV_SCHED value ("pingpong" = main's 3-stream per-plane
QKV GEMMs, "batched" = one L=3 launch). Set before any TRT-LLM import, i.e. fixed at load.
Usage: python timeline.py <checkpoint> <pingpong|batched>
"""

import json
import os
import sys

os.environ["TRTLLM_FLUX3_SM120_QKV_SCHED"] = sys.argv[2]

import torch  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

CKPT = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
    **engine,
)
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
print("loaded; sched", strided.qkv_schedule(), flush=True)


def generate():
    with torch.inference_mode():
        ctx = pipe._encode_prompt("A red fox standing in fresh snow at sunrise", None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        pipe._prepare_t2i(ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=11)
    torch.cuda.synchronize()


generate()
generate()
torch.cuda.profiler.start()
generate()
torch.cuda.profiler.stop()
print(json.dumps({"passed": True, "sched": strided.qkv_schedule(), "note": "timeline only"}))
