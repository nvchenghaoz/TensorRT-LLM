"""Branch 23: served-pipeline census helper (CUDA graphs ON, torch.compile + autotune, engine.json
overlay, warmup [[2048, 2048]] t2i + i2i, as the benchmark serves it). Usage:
  served_census.py <checkpoint> <t2i|i2i>
Runs two warm requests of that mode, then ONE request of that mode inside cudaProfilerStart/Stop
(profile_timeline capture=region). Kernel calls in the region / 16 = calls per denoise step.
"""

import glob
import json
import os
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

CKPT, MODE = sys.argv[1], sys.argv[2]
INPUTS_DIR = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/"
    "flux3-klein-nvfp4-2048/v1/inputs"
)


def log(*a):
    print(f"[b23c {time.strftime('%H:%M:%S')}]", *a, flush=True)


here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
engine.pop("env", None)
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT, pipeline_config=pipeline_config, parallel_config={"tp_size": 1},
    torch_compile_config={"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s graphs={runner is not None and runner.enabled}")
cond = [bool(getattr(getattr(b, "_orig_mod", b), "_sm120_attn_nvfp4_out", False))
        for b in pipe.transformer.content_mode_blocks["image_cond"]]
log("image_cond nfo flags", cond)
ref = open(sorted(glob.glob(os.path.join(INPUTS_DIR, "*.png")))[0], "rb").read()


def generate():
    with torch.inference_mode():
        if MODE == "t2i":
            ctx = pipe._encode_prompt("A red fox standing in fresh snow at sunrise", None)
            with autotune_context(pipe.device, pipe.rank, tune=False):
                pipe._prepare_t2i(ctx=ctx, height=2048, width=2048, num_inference_steps=16, seed=100)
        else:
            ctx = pipe._encode_prompt("Make it look like a watercolor painting", None)
            image = _load_image_pixels(ref, device=pipe.device)
            with autotune_context(pipe.device, pipe.rank, tune=False):
                pipe._prepare_i2i(ctx=ctx, image_cond=[image], height=2048, width=2048,
                                  num_inference_steps=16, seed=7)
    torch.cuda.synchronize()


generate()
generate()
torch.cuda.profiler.start()
generate()
torch.cuda.profiler.stop()
log("profiled one", MODE, "request")
print(json.dumps({"passed": True, "profiled": f"one graph-on {MODE} request (16 steps)",
                  "image_cond_nfo": cond}))
