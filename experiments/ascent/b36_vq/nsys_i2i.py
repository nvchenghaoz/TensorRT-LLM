"""Branch 36: one graph-on i2i request (and one t2i) inside cudaProfilerStart/Stop for nsys launch
counts of the fused QKV GEMM and v_quant_kernel per request. Loads + warms the served pipeline
(engine.json overlay) and runs one untraced i2i first so capture happens outside the region.
Usage: python nsys_i2i.py <checkpoint> [i2i|t2i]
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

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "i2i"
here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine.setdefault("torch_compile_config",
                  {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
engine.setdefault("parallel_config", {"tp_size": 1})
args = VisualGenArgs(model=CKPT, pipeline_config=pipeline_config,
                     compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1],
                                         "modes": ["t2i", "i2i"]}, **engine)
pipe = PipelineLoader(args, device="cuda:0").load()
store = "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/" \
        "flux3-klein-nvfp4-2048/v1/inputs"
sha = "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"
path = [p for p in glob.glob(os.path.join(store, "**", "*"), recursive=True) if sha in p][0]
ref_img = _load_image_pixels(open(path, "rb").read(), device=pipe.device)


def generate(mode):
    with torch.inference_mode():
        ctx = pipe._encode_prompt("Make it night time with a full moon", None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if mode == "t2i":
            lat = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048, num_inference_steps=16,
                                    seed=7)
        else:
            lat = pipe._prepare_i2i(ctx=ctx, image_cond=[ref_img], height=2048, width=2048,
                                    num_inference_steps=16, seed=7)
        pipe._decode_latents(lat)
    torch.cuda.synchronize()


generate(MODE)
print(f"[{time.strftime('%H:%M:%S')}] warm {MODE} done; profiling one request", flush=True)
torch.cuda.cudart().cudaProfilerStart()
generate(MODE)
torch.cuda.cudart().cudaProfilerStop()
print(json.dumps({"profiled_requests": 1, "mode": MODE}))
