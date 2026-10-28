"""Idea 9 pipeline-level check, one process (full FLUX.3 Klein nvfp4 pipeline, served defaults).

Loads the pipeline with PipelineLoader (torch_compile + autotune + warmup as served), then
A (T=80), B (T=160), A again: ctx from the branch path (graph replay at warmed buckets)
vs main's compiled graphs-off encode (two tokenizer calls, compiled fn, clone), then the
16-step latent and decoded uint8 image from each ctx -> bitwise equal, and A vs A' equal.
Reports load+warmup seconds, GPU memory, graphs, Dynamo counters after warmup and
after each request (TORCH_LOGS=recompiles shows any compile).
mode "profile": after the check, NVTX-marked requests graph vs off for nsys.
Usage: python te_pipe_check.py <checkpoint> [check|profile]
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import main_encode, prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "check"
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return dict(c["frames"]), dict(c["stats"])


from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

args = VisualGenArgs(model=CKPT, pipeline_config={"variant": "nvfp4"})
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
load_s = time.time() - t0
te = pipe.text_encoder
info = {
    "load_and_warmup_s": round(load_s, 1),
    "torch_compile": pipe.pipeline_config.torch_compile.enable,
    "cuda_graph_denoiser": pipe.pipeline_config.cuda_graph.enable,
    "te_graphs": sorted(getattr(te, "_graphs", {}).keys()),
    "mem_allocated_mb": torch.cuda.memory_allocated() / 2**20,
    "mem_reserved_mb": torch.cuda.memory_reserved() / 2**20,
}
print("after warmup", info, flush=True)
note(info["te_graphs"] == [80], "T=80 graph captured at warmup")
c_warm = counters()
print("dynamo after warmup", c_warm, flush=True)

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)


def generate(ctx, seed):
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(ctx=ctx, height=1024, width=1024,
                                        num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return latent.clone(), img.clone()


A = "A red fox standing in fresh snow at sunrise"
B = prompt_for_bucket(te, 2)
seq = [("A", A, 11), ("B", B, 22), ("A", A, 11)]
results, first = {}, {}
for name, p, seed in seq:
    t0 = time.time()
    with torch.inference_mode():
        ctx_g = pipe._encode_prompt(p, None)
    lat_g, img_g = generate(ctx_g, seed)
    t_req = time.time() - t0
    c_after_g = counters()
    ctx_o = [main_encode(te, p)]
    lat_o, img_o = generate(ctx_o, seed)
    r = {
        "T": ctx_g[0].shape[1],
        "req_s": round(t_req, 2),
        "ctx_equal": torch.equal(ctx_g[0], ctx_o[0]),
        "latent_equal": torch.equal(lat_g, lat_o),
        "image_equal": torch.equal(img_g, img_o),
        "image_shape": list(img_g.shape),
        "dynamo_after": c_after_g,
    }
    if name in first:
        r["repeat_equal_first"] = (torch.equal(first[name][0], lat_g)
                                   and torch.equal(first[name][1], img_g))
        note(r["repeat_equal_first"], f"{name} repeat equal")
    else:
        first[name] = (lat_g, img_g)
    for k in ("ctx_equal", "latent_equal", "image_equal"):
        note(r[k], f"{name} {k}")
    results[f"{len(results)}_{name}"] = r
    print(name, json.dumps(r, default=str), flush=True)

# T=80 requests after warmup must not compile anything.
c_a = results["0_A"]["dynamo_after"]
c_a2 = results["2_A"]["dynamo_after"]
note(c_a == c_warm, "no compile in first T=80 request")
note(c_a2 == results["1_B"]["dynamo_after"], "no compile in repeated T=80 request")
note(sorted(te._graphs) == [80], "no graph captured after warmup")

if MODE == "profile":
    import torch.cuda.nvtx as nvtx

    graphs = te._graphs

    def request(p, seed, tag):
        torch.cuda.synchronize()
        nvtx.range_push(f"request_{tag}")
        nvtx.range_push(f"text_encode_{tag}")
        with torch.inference_mode():
            ctx = pipe._encode_prompt(p, None)
        nvtx.range_pop()
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(ctx=ctx, height=1024, width=1024,
                                            num_inference_steps=16, seed=seed)
            pipe._package_output(pipe._decode_latents((latent, ids)))
        torch.cuda.synchronize()
        nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(3):
        te._graphs = graphs
        request(A, 5 + i, "graph")
        te._graphs = {}
        request(A, 5 + i, "off")
    te._graphs = graphs
    torch.cuda.cudart().cudaProfilerStop()

print(json.dumps({"passed": not fails, "failures": fails, "info": info,
                  "compared_against": "main's compiled graphs-off encode in the same process",
                  "results": results}, default=str))
