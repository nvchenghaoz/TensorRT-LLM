"""Branch 71 (idea 121) one-process e2e check on the full FLUX.3 Klein nvfp4 pipeline.

Switch: flux3_adaln_fp8r.FP8R_ADALN_QUANT (True = fused LN+adaLN+E4M3 head in the FP8R blocks;
False = main's head: Inductor LN+modulate + quantize_fp8r). Dynamo guards on it, so both arms
run in one process. Pipeline as served (engine.json overlay incl. cuda_graph_config, torch_compile
+ autotune, warmup [[1024,1024]] t2i), switch ON as loaded.

Sequence (A = T=80 prompt, B = T=160 prompt), each request a full 16-step t2i + VAE decode:
  G1  graphs on,  A on        (replays warmup capture)
  E1  graphs off, A on        -> graph-on == graph-off (G1 == E1)
  E2  graphs off, A off       -> ON vs OFF at T=80 (E1 == E2), same Dynamo history position
  E3  graphs off, B off
  E4  graphs off, B on        -> ON vs OFF at T=160 (E3 == E4)
  E5  graphs off, A off
  E6  graphs off, A on        -> ON vs OFF at T=80-after-T=160 (E5 == E6)
  G2  graphs on,  A on        -> repeat of G1 bitwise; 0 new captures, 0 new Dynamo graphs
  P   profile (graphs off, both compiled): kernel counts per request ON vs OFF
Pass: every comparison bitwise (final latent and uint8 image), G2 steady.
Usage: python b71_e2e_check.py <checkpoint>
"""

import collections
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


def unique_graphs():
    from torch._dynamo.utils import counters as c

    return c["stats"].get("unique_graphs", 0)


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_fp8r as fa  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

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
print("engine overlay", engine, pipeline_config, flush=True)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s switch={fa.FP8R_ADALN_QUANT} "
      f"runner={runner is not None and runner.enabled} "
      f"captures={len(runner.graphs) if runner else 0}", flush=True)
tr = pipe.transformer
heads = [
    name for name, m in tr.named_modules()
    if getattr(m, "_flux3_fp8r_adaln_head", None) is not None
]
print("fused heads on", heads, flush=True)
note(len(heads) == 5, f"5 FP8R blocks carry the fused head (got {len(heads)})")
warm_graphs = unique_graphs()

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
results = {"warmup_unique_graphs": warm_graphs, "fused_heads": heads}


def run(label, name, on, graphs):
    fa.FP8R_ADALN_QUANT = on
    if runner is not None:
        runner.enabled = graphs
    caps = len(runner.graphs) if runner else 0
    prev = unique_graphs()
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name])
    rec = {
        "label": label, "req": name, "on": on, "graphs": graphs, "T": T,
        "s": round(time.time() - t1, 2),
        "new_dynamo_graphs": unique_graphs() - prev,
        "new_captures": (len(runner.graphs) if runner else 0) - caps,
    }
    print(label, json.dumps(rec), flush=True)
    results.setdefault("requests", []).append(rec)
    return rec, lat, img


def cmp(a, b, what):
    d = {
        "what": what,
        "latent_bitwise": torch.equal(a[1], b[1]),
        "image_bitwise": torch.equal(a[2], b[2]),
        "latent_max_abs": (a[1].float() - b[1].float()).abs().max().item(),
        "image_max_abs": (a[2].int() - b[2].int()).abs().max().item(),
    }
    print("CMP", json.dumps(d), flush=True)
    results.setdefault("comparisons", []).append(d)
    note(d["latent_bitwise"] and d["image_bitwise"], what)


g1 = run("G1", "A", True, True)
e1 = run("E1", "A", True, False)
e2 = run("E2", "A", False, False)
e3 = run("E3", "B", False, False)
e4 = run("E4", "B", True, False)
e5 = run("E5", "A", False, False)
e6 = run("E6", "A", True, False)
g2 = run("G2", "A", True, True)
cmp(g1, e1, "graph-on == graph-off (A, switch ON)")
cmp(e1, e2, "ON == OFF at T=80 (first A)")
cmp(e3, e4, "ON == OFF at T=160")
cmp(e5, e6, "ON == OFF at T=80 after T=160")
cmp(g1, g2, "graphs-on A repeat (G2 == G1)")
note(g2[0]["new_captures"] == 0, "G2 steady: 0 new CUDA graph captures")
note(g2[0]["new_dynamo_graphs"] == 0, "G2 steady: 0 new Dynamo graphs")
note(g1[0]["new_dynamo_graphs"] == 0, "G1 (first request after warmup): 0 new Dynamo graphs")

# PROFILE: kernels per request ON vs OFF, graphs off (both arms compiled at this history).
prof = {}
for on in (True, False):
    run("PROFILE_WARM", "A", on, False)
    fa.FP8R_ADALN_QUANT = on
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        generate(*PROMPTS["A"])
    kern = collections.defaultdict(lambda: [0, 0.0])
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            k = kern[ev.name[:100]]
            k[0] += 1
            k[1] += ev.time_range.end - ev.time_range.start
    pick = {
        n: [v[0], round(v[1], 1)] for n, v in kern.items()
        if "PerTokenFp8Quant" in n or "adaln_fp8r" in n or "layer_norm" in n
    }
    prof["on" if on else "off"] = {
        "kernels": sum(v[0] for v in kern.values()),
        "kernel_sum_ms": round(sum(v[1] for v in kern.values()) / 1e3, 3),
        "selected": pick,
    }
    print("PROFILE", on, json.dumps(prof["on" if on else "off"]), flush=True)
results["profile_graphs_off"] = prof


def count(arm, sub):
    return sum(v[0] for n, v in prof[arm]["selected"].items() if sub in n)


q_on, q_off = count("on", "PerTokenFp8Quant"), count("off", "PerTokenFp8Quant")
results["warpPerTokenFp8Quant_per_request"] = {"off": q_off, "on": q_on}
results["fused_kernel_per_request_on"] = count("on", "adaln_fp8r")
note(q_off - q_on == 80, f"warpPerTokenFp8Quant drops by 80/request (off {q_off}, on {q_on})")
note(results["fused_kernel_per_request_on"] == 80, "80 fused head launches/request")
fa.FP8R_ADALN_QUANT = True
if runner is not None:
    runner.enabled = True
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "FP8R_ADALN_QUANT=False (main's Inductor LN+modulate + quantize_fp8r) "
    "in the same process at the same Dynamo history position; final latent + uint8 image "
    "bitwise; graphs on vs off; steady-request captures/recompiles",
    **results,
}, default=str))
