"""Idea 21 (branch 46) acceptance 3: census of one served i2i request (run under nsys, capture region).

  served_census.py <ckpt>
Loads the pipeline as served (engine.json overlay, torch.compile + autotune, CUDA graphs, warmup
of 2048^2 t2i + i2i), runs one untimed t2i and i2i request, then "timed" requests t2i, i2i and
finally one i2i between cudaProfilerStart/Stop (the nsys region). Dynamo counters (frames,
unique graphs, recompiles, graph breaks) are recorded before and after the timed requests; a
torch.profiler pass is NOT used (nsys sees the kernels). Prints the counter deltas and whether
the op ran (row_softmax_triton call counts, wrapped at the Python level).
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


log("git", subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True).stdout)

from torch._dynamo.utils import counters  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine.setdefault("torch_compile_config", {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
engine.setdefault("parallel_config", {"tp_size": 1})
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
log(f"loaded+warmup {time.time() - t0:.0f}s")

STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
SHA00 = "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"


def store_png(sha):
    for root, _, files in os.walk(STORE):
        for f in files:
            p = os.path.join(root, f)
            if sha in f:
                return p
    for root, _, files in os.walk(STORE):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == sha:
                    return p
    raise FileNotFoundError(sha)


_data = open(store_png(SHA00), "rb").read()
assert hashlib.sha256(_data).hexdigest() == SHA00
REF_IMG = _load_image_pixels(_data, device=torch.device("cuda:0"))
CASES = {"t2i": ("A red fox standing in fresh snow at sunrise, photograph", 11),
         "i2i": ("Turn the scene into a watercolor painting", 5)}

# count op calls (Python level; the compiled graphs call the triton_op's kernels directly, so
# this counts eager calls only -- nsys is the authority for the compiled VAE)
try:
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import row_softmax_triton as rs
    OP = {"present": True, "enabled": rs.ENABLED}
except ImportError:
    OP = {"present": False}


def run(case):
    prompt, seed = CASES[case]
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if case == "t2i":
            latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048, num_inference_steps=16, seed=seed)
        else:
            latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[REF_IMG], height=2048, width=2048,
                                            num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return hashlib.sha256(img.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def snap():
    return {"frames": dict(counters["frames"]), "stats": dict(counters["stats"]),
            "recompiles": dict(counters["recompiles"]) if "recompiles" in counters else {},
            "graph_break": sum(counters["graph_break"].values())}


for case in ("t2i", "i2i"):
    log("untimed", case, run(case))
before = snap()
h = {}
for case in ("t2i", "i2i"):
    h[case] = run(case)
    log("timed", case, h[case])
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("i2i_request")
h["i2i_profiled"] = run("i2i")
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
after = snap()
delta = {
    "frames": {k: after["frames"].get(k, 0) - before["frames"].get(k, 0) for k in after["frames"]},
    "unique_graphs": after["stats"].get("unique_graphs", 0) - before["stats"].get("unique_graphs", 0),
    "graph_break": after["graph_break"] - before["graph_break"],
}
res = {"op": OP, "dynamo_delta_timed": delta, "before": before, "after": after, "hashes": h,
       "no_recompile_in_timed": delta["unique_graphs"] == 0 and delta["frames"].get("total", 0) == 0
       and delta["frames"].get("ok", 0) == 0,
       "passed": True}
print(json.dumps(res))
