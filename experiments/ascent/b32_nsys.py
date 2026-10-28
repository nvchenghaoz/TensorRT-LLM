"""Branch 32 nsys driver: one steady T=80 request with FLUX3_BROADCAST_ADALN on or off.

Usage: python b32_nsys.py <checkpoint> on|off
Sets the module switch before the pipeline loads (so warmup compiles that path), runs one
untimed request, then one request between cudaProfilerStart/Stop (capture=region).
"""

import json
import os
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

CKPT, ARM = sys.argv[1], sys.argv[2]
tf.FLUX3_BROADCAST_ADALN = ARM == "on"
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
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
print(f"loaded {time.time() - t0:.0f}s arm={ARM}", flush=True)

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)


def request(seed):
    with torch.inference_mode():
        ctx = pipe._encode_prompt("A red fox standing in fresh snow at sunrise", None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
        )
    torch.cuda.synchronize()


request(1)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
t1 = time.time()
request(2)
torch.cuda.cudart().cudaProfilerStop()
print(json.dumps({"arm": ARM, "denoise_wall_s": round(time.time() - t1, 3)}), flush=True)
