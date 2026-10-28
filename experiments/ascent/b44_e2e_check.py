"""Branch 44 (idea 92) one-process e2e check on the full FLUX.3 Klein nvfp4 pipeline.

Switch: transformer_flux3.FLUX3_ROPE_TABLE (True = branch: per-request cos_sin table passed as
freqs[2]; False = main's path: 2-tuple freqs, inline cat in every fused QKV). It is read in
_packed_positions per request, so both arms run in one process.

Loads the pipeline as served (engine.json overlay, torch_compile + autotune, warmup
[[1024,1024]] t2i, no CUDA graphs: cuda_graph_config is not on main).
Phase SERVED (switch ON as loaded): A(T=80), B(T=160), A -> compiles per request, repeats.
Phase SHARED (same process, no reset, shared Dynamo history), interleaved per request:
    A_off, A_on, B_off, B_on, A_off, A_on   -> on vs off at the same position, bitwise.
Phase STATIC (torch._dynamo.reset(), automatic_dynamic_shapes=False for both arms):
    A_off, A_on, B_off, B_on               -> bitwise.
Phase PROFILE: torch.profiler (CUDA kernels) over one ON and one OFF T=80 request (no compile
inside either): kernels matching fused_cat.*qkv per request, total kernels.
Output class EXACT: pass needs every on/off pair bitwise (final latent and uint8 image),
repeats bitwise, no compile inside the ON T=80 requests after warmup, and 0 cat kernels on ON.
Usage: python b44_e2e_check.py <checkpoint>
"""

import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return {"frames": dict(c["frames"]), "unique_graphs": c["stats"].get("unique_graphs", 0)}


from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(here, "..", "ascent-config", "engine.json")
engine = json.load(open(engine_path)) if os.path.exists(engine_path) else {}
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
    **engine,
)
print("engine overlay", engine, pipeline_config, flush=True)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
print(f"loaded+warmup {time.time() - t0:.0f}s; switch={tf.FLUX3_ROPE_TABLE}", flush=True)
c_warm = counters()
print("dynamo after warmup", c_warm, flush=True)

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)

te = pipe.text_encoder


def generate(prompt, seed):
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
        )
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return ctx[0].shape[1], latent.clone(), img.clone()


PROMPTS = {
    "A": ("A red fox standing in fresh snow at sunrise", 11),
    "B": (prompt_for_bucket(te, 2), 22),
}


def run(label, name, on):
    tf.FLUX3_ROPE_TABLE = on
    prompt, seed = PROMPTS[name]
    prev = counters()["unique_graphs"]
    t1 = time.time()
    T, lat, img = generate(prompt, seed)
    rec = {
        "phase": label,
        "req": name,
        "on": on,
        "T": T,
        "s": round(time.time() - t1, 2),
        "new_graphs": counters()["unique_graphs"] - prev,
    }
    print(label, json.dumps(rec), flush=True)
    return rec, lat, img


def same(a, b):
    return torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def diff(a, b):
    return {
        "latent_bitwise": torch.equal(a[1], b[1]),
        "image_bitwise": torch.equal(a[2], b[2]),
        "latent_max_abs": (a[1].float() - b[1].float()).abs().max().item(),
        "image_max_abs": (a[2].int() - b[2].int()).abs().max().item(),
    }


results = {}

# SERVED: switch on as loaded.
served = [run("SERVED", n, True) for n in ("A", "B", "A")]
note(served[0][0]["new_graphs"] == 0, "no compile in first T=80 request after warmup (ON)")
note(served[2][0]["new_graphs"] == 0, "no compile in repeated T=80 request (ON)")
note(same(served[0], served[2]), "SERVED A repeat bitwise")
results["served"] = [r[0] for r in served]
results["served_A_repeat_bitwise"] = same(served[0], served[2])

# SHARED Dynamo history, interleaved per request.
seq = [("A", False), ("A", True), ("B", False), ("B", True), ("A", False), ("A", True)]
shared = [run("SHARED", n, on) for n, on in seq]
results["shared"] = [r[0] for r in shared]
results["shared_on_vs_off"] = []
for i in range(0, len(shared), 2):
    d = diff(shared[i], shared[i + 1])
    d["req"] = shared[i][0]["req"]
    results["shared_on_vs_off"].append(d)
    print("SHARED CMP", json.dumps(d), flush=True)
    note(d["latent_bitwise"] and d["image_bitwise"], f"SHARED {d['req']}#{i // 2} on==off")
note(same(shared[1], shared[5]), "SHARED A_on repeat bitwise")
note(same(shared[0], shared[4]), "SHARED A_off repeat bitwise")
results["shared_on_A_vs_served_A"] = diff(shared[1], served[0])

# STATIC: reset, no automatic dynamic shapes, both arms.
torch._dynamo.reset()
torch._dynamo.config.automatic_dynamic_shapes = False
static = [run("STATIC", n, on) for n, on in seq[:4]]
results["static"] = [r[0] for r in static]
results["static_on_vs_off"] = []
for i in range(0, len(static), 2):
    d = diff(static[i], static[i + 1])
    d["req"] = static[i][0]["req"]
    results["static_on_vs_off"].append(d)
    print("STATIC CMP", json.dumps(d), flush=True)
    note(d["latent_bitwise"] and d["image_bitwise"], f"STATIC {d['req']} on==off")

# PROFILE: kernels per request, ON vs OFF (both compiled already in STATIC).
pat = re.compile(r"fused_cat.*qkv|qkv.*fused_cat")
prof = {}
for on in (True, False):
    run("PROFILE_WARM", "A", on)
    tf.FLUX3_ROPE_TABLE = on
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        generate(*PROMPTS["A"])
    names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    cats = [n for n in names if pat.search(n)]
    allcat = [n for n in names if "fused_cat" in n]
    prof["on" if on else "off"] = {
        "kernels": len(names),
        "cat_qkv_kernels": len(cats),
        "cat_qkv_names": sorted(set(cats))[:5],
        "all_fused_cat_kernels": len(allcat),
        "all_fused_cat_names": sorted(set(allcat))[:10],
    }
    print("PROFILE", json.dumps(prof["on" if on else "off"]), flush=True)
results["profile"] = prof
note(prof["on"]["cat_qkv_kernels"] == 0, "ON has no cat-qkv kernels")

tf.FLUX3_ROPE_TABLE = True
print(
    json.dumps(
        {
            "passed": not fails,
            "failures": fails,
            "compared_against": "FLUX3_ROPE_TABLE=False (main's 2-tuple freqs, inline cat) in the "
            "same process, same prompts/seeds, final latent + uint8 image bitwise",
            "warmup_unique_graphs": c_warm["unique_graphs"],
            **results,
        },
        default=str,
    )
)
