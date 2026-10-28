"""Branch 32 (idea 74, adaLN Part A) one-process check on the full FLUX.3 Klein nvfp4 pipeline.

Loads the pipeline with PipelineLoader (served engine overlay from experiments/ascent-config/
engine.json, torch_compile + autotune, warmup [[1024,1024]] t2i, no CUDA graphs).

Phase ON  (FLUX3_BROADCAST_ADALN=True, as loaded and warmed up):
    A (T=80), B (T=160), C (T=320), D (T=480), A
Phase OFF (FLUX3_BROADCAST_ADALN=False = main's materialized path), after torch._dynamo.reset():
    W (T=80, stands in for warmup), A, B, C, D, A
Per request: final latent + decoded uint8 image, Dynamo counters (recompiles), wall time.
Then OFF2 (another reset, W, A) measures main's own compile-to-compile variation, and, if
torch._inductor.config.deterministic exists, ON vs OFF under it (W, A each, after resets).
Pass: A == A' within each phase, ON compiles no more than OFF inside the request sequence,
and ON == OFF bitwise under deterministic Inductor. ON vs OFF in default mode is reported
(bitwise flag, latent max abs, image PSNR) next to OFF vs OFF2.
Usage: python b32_adaln_check.py <checkpoint>
"""

import json
import os
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
print(f"loaded+warmup {time.time() - t0:.0f}s; switch={tf.FLUX3_BROADCAST_ADALN}", flush=True)
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


A = "A red fox standing in fresh snow at sunrise"
PROMPTS = {
    "A": (A, 11),
    "B": (prompt_for_bucket(te, 2), 22),
    "C": (prompt_for_bucket(te, 4), 33),
    "D": (prompt_for_bucket(te, 6), 44),
}


def phase(label, seq):
    out, first, comp = [], {}, []
    prev = counters()
    for name in seq:
        prompt, seed = PROMPTS["A" if name == "W" else name]
        t1 = time.time()
        T, lat, img = generate(prompt, seed)
        dt = time.time() - t1
        cur = counters()
        new_graphs = cur["unique_graphs"] - prev["unique_graphs"]
        prev = cur
        rec = {"req": name, "T": T, "s": round(dt, 2), "new_graphs": new_graphs}
        if name in first:
            rec["repeat_equal"] = torch.equal(first[name][0], lat) and torch.equal(
                first[name][1], img
            )
            note(rec["repeat_equal"], f"{label} {name} repeat bitwise")
        elif name != "W":
            first[name] = (lat, img)
        print(label, json.dumps(rec), flush=True)
        comp.append(rec)
        out.append((name, lat, img))
    return out, comp


tf.FLUX3_BROADCAST_ADALN = True
on, on_comp = phase("ON", ["A", "B", "C", "D", "A"])
torch._dynamo.reset()
tf.FLUX3_BROADCAST_ADALN = False
off, off_comp = phase("OFF", ["W", "A", "B", "C", "D", "A"])
off = [r for r in off if r[0] != "W"]
off_comp_req = [r for r in off_comp if r["req"] != "W"]

cmp = []
for (n1, l1, i1), (n2, l2, i2) in zip(on, off):
    assert n1 == n2
    d = (l1.float() - l2.float()).abs()
    rec = {
        "req": n1,
        "latent_bitwise": torch.equal(l1, l2),
        "image_bitwise": torch.equal(i1, i2),
        "latent_max_abs": d.max().item(),
        "image_max_abs": (i1.int() - i2.int()).abs().max().item(),
    }
    cmp.append(rec)
    print("CMP", json.dumps(rec), flush=True)

def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 99.0 if mse == 0 else round(10 * __import__("math").log10(255.0**2 / mse), 3)


for rec, (_, _, i1), (_, _, i2) in zip(cmp, on, off):
    rec["image_psnr_db"] = psnr(i1, i2)

# Compile-to-compile variation of main's own path: OFF again after another reset.
torch._dynamo.reset()
tf.FLUX3_BROADCAST_ADALN = False
off2, off2_comp = phase("OFF2", ["W", "A"])
a_off2 = off2[-1]
a_off = off[0]
variation = {
    "off_vs_off_after_reset_latent_bitwise": torch.equal(a_off2[1], a_off[1]),
    "off_vs_off_after_reset_image_bitwise": torch.equal(a_off2[2], a_off[2]),
    "off_vs_off_after_reset_latent_max_abs": (a_off2[1].float() - a_off[1].float()).abs().max().item(),
    "off_vs_off_after_reset_image_psnr_db": psnr(a_off2[2], a_off[2]),
}
print("VARIATION", json.dumps(variation), flush=True)

# Deterministic Inductor (no benchmark-picked reduction configs): ON vs OFF must be bitwise.
det = {"available": hasattr(torch._inductor.config, "deterministic")}
if det["available"]:
    torch._inductor.config.deterministic = True
    got = {}
    for flag in (False, True):
        torch._dynamo.reset()
        tf.FLUX3_BROADCAST_ADALN = flag
        out_d, _ = phase(f"DET_{'ON' if flag else 'OFF'}", ["W", "A"])
        got[flag] = out_d[-1]
    det["on_vs_off_latent_bitwise"] = torch.equal(got[True][1], got[False][1])
    det["on_vs_off_image_bitwise"] = torch.equal(got[True][2], got[False][2])
    det["on_vs_off_latent_max_abs"] = (got[True][1].float() - got[False][1].float()).abs().max().item()
    det["on_vs_off_image_psnr_db"] = psnr(got[True][2], got[False][2])
    torch._inductor.config.deterministic = False
print("DETERMINISTIC", json.dumps(det), flush=True)
if det["available"]:
    note(det["on_vs_off_latent_bitwise"] and det["on_vs_off_image_bitwise"],
         "ON vs OFF bitwise under deterministic Inductor")

on_graphs = sum(r["new_graphs"] for r in on_comp)
off_graphs = sum(r["new_graphs"] for r in off_comp_req)
note(on_graphs <= off_graphs, "ON compiles no more than OFF in the request sequence")
note(on_comp[0]["new_graphs"] == 0, "no compile in first T=80 request after warmup (ON)")
note(on_comp[-1]["new_graphs"] == 0, "no compile in repeated T=80 request (ON)")
print(
    json.dumps(
        {
            "passed": not fails,
            "failures": fails,
            "compared_against": "FLUX3_BROADCAST_ADALN=False (main's materialized per-token "
            "modulation path) in the same process after torch._dynamo.reset(), same prompts/seeds",
            "warmup_unique_graphs": c_warm["unique_graphs"],
            "on_requests": on_comp,
            "off_requests": off_comp,
            "on_vs_off": cmp,
            "compile_variation_of_main_path": variation,
            "deterministic_inductor": det,
        },
        default=str,
    )
)
