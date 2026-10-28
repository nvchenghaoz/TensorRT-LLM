"""Branch 49 (idea 100) one-process e2e check on the full FLUX.3 Klein nvfp4 pipeline.

Switch: transformer_flux3.FLUX3_TEXT_STEP_BROADCAST (True = branch: flow_map_text_step returns
[1,1,D] text single_* rows, forward takes the [1,2,D] table + select path; False = main's code:
rows expanded to [1,T,D], forward falls back to joint() per-token [1,S,3072] mods). It is read
in flow_map_text_step per step, so both arms run in one process.

Loads the pipeline as served (engine.json overlay, torch_compile + autotune, warmup
[[1024,1024]] t2i). TRTLLM_FLUX3_TEXT_STEP_BATCH defaults to 1 (asserted).
STEP0: a pre-hook on single_blocks[0] logs the mod tuple (arity, shapes) each arm hands the
  single blocks.
SERVED (switch ON as loaded): A(T=80), B(T=160), A -> new Dynamo graphs per request, A==A.
SHARED (no reset, shared Dynamo history), interleaved: A_off, A_on, B_off, B_on, A_off, A_on.
  on vs off: max abs / rel error of the final latent, image PSNR; A_on==A_on, A_off==A_off.
PROFILE: per arm, after a warm request, (a) CUDA-event time of the 28 single blocks per
  request (pre-hook on block 0 .. post-hook on block 27, summed over 16 steps), (b)
  torch.profiler kernel table of one T=80 request: top kernels by total time, cat kernels.
usage: python b49_e2e_check.py <checkpoint>
"""

import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
fails = []
assert os.environ.get("TRTLLM_FLUX3_TEXT_STEP_BATCH", "1") == "1"


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
print(f"loaded+warmup {time.time() - t0:.0f}s; switch={tf.FLUX3_TEXT_STEP_BROADCAST}", flush=True)
c_warm = counters()
print("dynamo after warmup", c_warm, flush=True)

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)

te = pipe.text_encoder
blocks = pipe.transformer.single_blocks
print("single blocks", len(blocks), type(blocks[0]).__name__, flush=True)

# STEP0 / timing hooks (eager wrappers around the compiled blocks; no numeric effect).
mod_log = []
timing = {"on": False, "events": []}


def pre0(module, args):
    mod = args[2]
    mod_log.append([len(mod)] + [list(t.shape) for t in mod])
    if timing["on"]:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        timing["events"].append([e, None])


def post_last(module, args, out):
    if timing["on"]:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        timing["events"][-1][1] = e


blocks[0].register_forward_pre_hook(pre0)
blocks[-1].register_forward_hook(post_last)


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
    tf.FLUX3_TEXT_STEP_BROADCAST = on
    prompt, seed = PROMPTS[name]
    prev = counters()["unique_graphs"]
    mod_log.clear()
    t1 = time.time()
    T, lat, img = generate(prompt, seed)
    arity = sorted({m[0] for m in mod_log})
    rec = {
        "phase": label,
        "req": name,
        "on": on,
        "T": T,
        "s": round(time.time() - t1, 2),
        "new_graphs": counters()["unique_graphs"] - prev,
        "single_block_calls": len(mod_log),
        "mod_arity": arity,
        "mod_shapes_step0": mod_log[0] if mod_log else None,
    }
    print(label, json.dumps(rec), flush=True)
    return rec, lat, img


def same(a, b):
    return torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def psnr(a, b):
    x, y = a.float(), b.float()
    peak = 255.0 if a.dtype == torch.uint8 else max(1.0, y.abs().max().item())
    mse = (x - y).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(peak * peak / mse)).item()


def diff(a, b):
    la, lb = a[1].float(), b[1].float()
    d = (la - lb).abs()
    return {
        "latent_bitwise": torch.equal(a[1], b[1]),
        "image_bitwise": torch.equal(a[2], b[2]),
        "latent_max_abs": d.max().item(),
        "latent_max_rel": (d.max() / lb.abs().max()).item(),
        "latent_mean_abs": d.mean().item(),
        "image_max_abs": (a[2].float() - b[2].float()).abs().max().item(),
        "image_psnr_db": psnr(a[2], b[2]),
        "image_dtype": str(a[2].dtype),
    }


results = {}

served = [run("SERVED", n, True) for n in ("A", "B", "A")]
note(served[0][0]["new_graphs"] == 0, "no compile in first T=80 request after warmup (ON)")
note(served[2][0]["new_graphs"] == 0, "no compile in repeated T=80 request (ON)")
note(all(r[0]["mod_arity"] == [4] for r in served), "ON single blocks get the 4-tuple table+select")
note(same(served[0], served[2]), "SERVED A repeat bitwise")
results["served"] = [r[0] for r in served]
results["served_A_repeat_bitwise"] = same(served[0], served[2])

seq = [("A", False), ("A", True), ("B", False), ("B", True), ("A", False), ("A", True)]
shared = [run("SHARED", n, on) for n, on in seq]
results["shared"] = [r[0] for r in shared]
note(all(r[0]["mod_arity"] == [3] for r, (_, on) in zip(shared, seq) if not on),
     "OFF (main's code) single blocks get the per-token 3-tuple")
note(all(r[0]["mod_arity"] == [4] for r, (_, on) in zip(shared, seq) if on),
     "ON single blocks get the 4-tuple")
results["shared_on_vs_off"] = []
for i in range(0, len(shared), 2):
    d = diff(shared[i + 1], shared[i])
    d["req"] = shared[i][0]["req"]
    results["shared_on_vs_off"].append(d)
    print("SHARED CMP", json.dumps(d), flush=True)
results["shared_A_on_repeat_bitwise"] = same(shared[1], shared[5])
results["shared_A_off_repeat_bitwise"] = same(shared[0], shared[4])
note(results["shared_A_on_repeat_bitwise"], "SHARED A_on repeat bitwise")
note(results["shared_A_off_repeat_bitwise"], "SHARED A_off repeat bitwise")
# Informational: after the OFF arm has compiled, automatic-dynamic history is shared (knowledge
# 567), so ON graphs may be recompiled with other configs.
results["shared_A_on_vs_served_A_bitwise"] = same(shared[1], served[0])

# PROFILE
prof = {}
for on in (False, True):
    run("PROFILE_WARM", "A", on)
    tf.FLUX3_TEXT_STEP_BROADCAST = on
    timing["on"], timing["events"] = True, []
    reps = []
    for _ in range(2):
        timing["events"] = []
        generate(*PROMPTS["A"])
        reps.append(sum(a.elapsed_time(b) for a, b in timing["events"]))
    timing["on"] = False
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        generate(*PROMPTS["A"])
    agg = defaultdict(lambda: [0, 0.0])
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name][0] += 1
            agg[e.name][1] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])
    key = "on" if on else "off"
    prof[key] = {
        "single_blocks_ms_per_request": reps,
        "kernels": sum(v[0] for v in agg.values()),
        "kernel_us_total": sum(v[1] for v in agg.values()),
        "top": [[n[:110], v[0], round(v[1], 1), round(v[1] / v[0], 2)] for n, v in top[:30]],
        "cat_kernels": [[n[:110], v[0], round(v[1], 1)] for n, v in top if "cat" in n.lower()][:10],
        "norm_kernels": [[n[:110], v[0], round(v[1], 1), round(v[1] / v[0], 2)]
                         for n, v in top if "layer_norm" in n or "native_layer" in n][:10],
        "add_mul_kernels": [[n[:110], v[0], round(v[1], 1), round(v[1] / v[0], 2)]
                            for n, v in top if "add_mul" in n or "mul_add" in n
                            or "where" in n][:10],
    }
    print("PROFILE", key, json.dumps(prof[key]), flush=True)
results["profile"] = prof

tf.FLUX3_TEXT_STEP_BROADCAST = True
print(
    json.dumps(
        {
            "passed": not fails,
            "failures": fails,
            "output": {"class": "reordered"},
            "compared_against": "FLUX3_TEXT_STEP_BROADCAST=False (main's flow_map_text_step "
            "expand -> per-token joint() mods) in the same process, same prompts/seeds",
            "warmup_unique_graphs": c_warm["unique_graphs"],
            **results,
        },
        default=str,
    )
)
