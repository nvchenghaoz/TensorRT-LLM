"""Branch 71: one steady served-config t2i request (T=80) under cudaProfilerStart/Stop for
profile_timeline capture=region. Switch from argv[2] ("on"/"off", default on).
Usage: python b71_timeline.py <checkpoint> [on|off]
"""

import json
import os
import sys

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_fp8r as fa
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

CKPT = sys.argv[1]
fa.FP8R_ADALN_QUANT = (sys.argv[2] if len(sys.argv) > 2 else "on") == "on"
here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "ascent-config", "engine.json")))
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
print("loaded; switch", fa.FP8R_ADALN_QUANT, flush=True)


def generate():
    with torch.inference_mode():
        ctx = pipe._encode_prompt("A red fox standing in fresh snow at sunrise", None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=11
        )
    torch.cuda.synchronize()


generate()
generate()
torch.cuda.profiler.start()
generate()
torch.cuda.profiler.stop()
print(json.dumps({"passed": True, "note": "timeline only"}))
