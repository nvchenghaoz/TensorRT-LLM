"""Full FLUX.3 pipeline check for the compiled VAE decoder (served engine settings).

Loads the pipeline as trtllm-serve does (VANILLA attention, per-block compile with
autotune, no CUDA graphs, warmup 1024x1024 t2i), then runs requests along the
infer() path. Reports: dynamo recompiles after warmup, bitwise repeat of two
same-seed requests, and compiled-vs-eager decoder error on the real latents.
Usage: TORCH_LOGS=recompiles python vae_pipeline_check.py <checkpoint_root>
"""

import json
import sys
import time

import torch
import torch._dynamo.utils as dyn_utils

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

root = sys.argv[1]
args = VisualGenArgs(
    model=root,
    pipeline_config={"variant": "nvfp4"},
    attention_config={"backend": "VANILLA"},
    compilation_config={"modes": ["t2i"], "num_frames": [1], "resolutions": [[1024, 1024]]},
    cuda_graph_config={"enable": False},
    torch_compile_config={"enable": True, "enable_autotune": True, "enable_fullgraph": False},
)
t0 = time.time()
pipe = PipelineLoader(args).load()
print(f"loaded+warmup in {time.time() - t0:.1f}s", flush=True)
print("vae dtype", next(pipe.vae.parameters()).dtype, flush=True)
compiled = getattr(pipe.vae.decoder, "_compiled_call_impl", None) is not None
print("decoder compiled:", compiled, flush=True)


def snap():
    return {k: dict(v) for k, v in dyn_utils.counters.items() if k in ("frames", "stats", "recompiles")}


width, height = pipe.pipeline_config.extra_attrs["flux3_checkpoint"].config["sampler"]["size"]
before = snap()
prompts = ["a red fox sitting in a snowy forest, photograph", "a red fox sitting in a snowy forest, photograph",
           "a bowl of ramen on a wooden table"]
images, latents, times = [], [], []
with torch.inference_mode():
    for i, p in enumerate(prompts):
        torch.cuda.synchronize()
        t = time.time()
        ctx = pipe._encode_prompt(p, None)
        with autotune_context(pipe.device, pipe.rank, tune=False):
            latent = pipe._prepare_t2i(ctx=ctx, height=height, width=width, seed=0)
        torch.cuda.synchronize()
        td = time.time()
        img = pipe._package_output(pipe._decode_latents(latent)).image
        torch.cuda.synchronize()
        times.append({"total_s": time.time() - t, "post_s": time.time() - td})
        images.append(img.clone())
        latents.append(latent)
        print(f"request {i} done {times[-1]}", flush=True)
after = snap()

repeat_bitwise = bool(torch.equal(images[0], images[1]))

# Compiled vs eager decoder on the real latents (eager = nn.Module.forward, bypassing compile).
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _as_image_grid  # noqa: E402

errs = []
with torch.inference_mode():
    for latent, ids in latents[1:]:
        grid = _as_image_grid(latent, ids).to(next(pipe.vae.parameters()).dtype)
        z = pipe.vae.inv_normalize(grid)
        from einops import rearrange

        z = rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)
        out_c = pipe.vae.decoder(z)
        out_e = pipe.vae.decoder.forward(z)
        d = (out_c.float() - out_e.float()).abs()
        u = (pipe._to_uint8_image(out_c).int() - pipe._to_uint8_image(out_e).int()).abs()
        errs.append({"max_abs": d.max().item(), "mean_abs": d.mean().item(),
                     "bitwise_rate": (d == 0).float().mean().item(),
                     "u8_max": u.max().item(), "u8_frac_diff": (u > 0).float().mean().item()})
recompiled = after != before
print(json.dumps({
    "passed": bool(compiled and repeat_bitwise and not recompiled),
    "compared": "same-seed request repeat (bitwise), dynamo counters across post-warmup requests, compiled vs eager decoder on real latents",
    "decoder_compiled": compiled,
    "no_recompile_after_warmup": not recompiled,
    "counters_before": before,
    "counters_after": after,
    "repeat_bitwise": repeat_bitwise,
    "times": times,
    "compiled_vs_eager_real_latents": errs,
}))
