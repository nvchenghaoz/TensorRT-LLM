"""Branch 79 (idea 123) one-process e2e check on the full FLUX.3 Klein nvfp4 pipeline (as
served: engine.json overlay incl. CUDA graphs, torch_compile + autotune, warmup t2i 1024^2).

Arms, switched in one process by wrapping transformer.final_layer["image"]:
  copy  : tokens.clone() before the final layer == main's single-part torch.cat copy
          (a fresh contiguous [1,n,D], same strides as the branch's unsqueeze view)
  eager : the uncompiled Flux3LastLayer (main's final layer), else the compiled one
  main   = eager+copy   (main's code path)
  view_e = eager+view   (stage 1 alone)
  comp_c = compiled+copy
  branch = compiled+view (as loaded)
CUDA graphs stay ON; on an arm switch the runner's captures are dropped so the next request
re-captures with that arm (graph key is shapes only).
Sequence (A = T=80 prompt, B = T=160 prompt):
  SERVED: branch A, B, A  -> A repeat bitwise, 0 new final-layer cache entries
  ABA   : for req in A, B, A: main, view_e, comp_c, branch
          -> view_e == main bitwise (stage 1 vs main); branch == comp_c bitwise (stage 1 under
             compile); branch vs main: latent max abs/rel + image PSNR (compiled vs uncompiled)
  OP    : the first final-layer call's (tokens, mod) of an A request: compiled vs eager output
          max abs/rel, and each vs an fp32 reference.
Usage: python b79_e2e_check.py <checkpoint>
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf  # noqa: E402
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
tr = pipe.transformer
compiled = tr.final_layer["image"]
eager = getattr(compiled, "_orig_mod", None)
print(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
      f"captures={len(runner.graphs) if runner else 0} final layer type "
      f"{type(compiled).__name__}", flush=True)
note(eager is not None, "final_layer['image'] is compiled (OptimizedModule)")


def cache_entries():
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    return len(_debug_get_cache_entry_list(tf.Flux3LastLayer.forward.__code__))


MODE = {"copy": False, "eager": False, "capture": None}


class Switch(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_c = compiled

    @property
    def adaLN_modulation(self):
        return eager.adaLN_modulation

    def forward(self, x, vec, mod=None):
        if MODE["capture"] is not None and not MODE["capture"]:
            MODE["capture"].update(x=x.clone(), mod=None if mod is None else mod.clone(),
                                   stride=tuple(x.stride()))
        if MODE["copy"]:
            x = x.clone()
        return (eager if MODE["eager"] else compiled)(x, vec, mod=mod)


tr.final_layer["image"] = Switch()
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
ARMS = {"main": (True, True), "view_e": (False, True), "comp_c": (True, False),
        "branch": (False, False)}
results = {"cache_entries_after_warmup": cache_entries()}
last_arm = ["branch"]


def run(label, name, arm):
    MODE["copy"], MODE["eager"] = ARMS[arm]
    if runner is not None and arm != last_arm[0]:
        runner.clear()
    last_arm[0] = arm
    caps = len(runner.graphs) if runner else 0
    ce = cache_entries()
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name])
    rec = {"label": label, "req": name, "arm": arm, "T": T, "s": round(time.time() - t1, 2),
           "new_captures": (len(runner.graphs) if runner else 0) - caps,
           "final_layer_cache_entries": cache_entries(), "new_final_layer_entries":
           cache_entries() - ce}
    print(label, json.dumps(rec), flush=True)
    results.setdefault("requests", []).append(rec)
    return rec, lat, img


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def cmp(a, b, what, require_bitwise):
    da = (a[1].float() - b[1].float()).abs()
    d = {
        "what": what,
        "latent_bitwise": torch.equal(a[1], b[1]),
        "image_bitwise": torch.equal(a[2], b[2]),
        "latent_max_abs": da.max().item(),
        "latent_max_rel": (da / b[1].float().abs().clamp_min(1e-2)).max().item(),
        "image_max_abs": (a[2].int() - b[2].int()).abs().max().item(),
        "image_psnr_db": psnr(a[2], b[2]),
    }
    print("CMP", json.dumps(d), flush=True)
    results.setdefault("comparisons", []).append(d)
    if require_bitwise:
        note(d["latent_bitwise"] and d["image_bitwise"], what)
    return d


# SERVED (branch as loaded, graphs on)
s = [run("SERVED", n, "branch") for n in ("A", "B", "A")]
cmp(s[0], s[2], "SERVED branch A repeat (P3==P1)", True)
note(all(r[0]["new_final_layer_entries"] == 0 for r in s), "SERVED: 0 new final-layer entries")

# ABA, all four arms per request
aba = {}
for k, name in enumerate(("A", "B", "A")):
    for arm in ARMS:
        aba[(k, arm)] = run(f"ABA{k}", name, arm)
for k, name in enumerate(("A", "B", "A")):
    cmp(aba[(k, "view_e")], aba[(k, "main")], f"ABA{k} {name}: view_e == main (stage 1, eager)",
        True)
    cmp(aba[(k, "branch")], aba[(k, "comp_c")],
        f"ABA{k} {name}: branch == comp_c (stage 1, compiled)", True)
    results.setdefault("branch_vs_main", []).append(
        cmp(aba[(k, "branch")], aba[(k, "main")], f"ABA{k} {name}: branch vs main (compiled "
            "vs uncompiled final layer)", False))
cmp(aba[(0, "branch")], s[0], "ABA branch A == SERVED branch A", True)
cmp(aba[(2, "branch")], aba[(0, "branch")], "ABA branch A repeat after B", True)
cmp(aba[(2, "main")], aba[(0, "main")], "ABA main A repeat after B", True)

# OP-level: capture one call's inputs, compiled vs eager
MODE["capture"] = {}
if runner is not None:
    runner.enabled = False  # run the forward in Python so the hook sees the call
run("OPCAP", "A", "branch")
if runner is not None:
    runner.enabled = True
cap = MODE["capture"]
MODE["capture"] = None
with torch.inference_mode():
    x, mod = cap["x"], cap["mod"]
    oc = compiled(x, None, mod=mod)
    oe = eager(x, None, mod=mod)
    sh, sc = mod.float().chunk(2, dim=-1)
    xn = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)
    w = eager.linear.weight.float()
    o32 = F.linear((1 + sc) * xn + sh, w)
    d = (oc.float() - oe.float()).abs()
    op = {
        "tokens_shape": list(x.shape), "tokens_stride_seen": cap["stride"],
        "bitwise": torch.equal(oc, oe), "max_abs": d.max().item(),
        "max_rel": (d / oe.float().abs().clamp_min(1e-3)).max().item(),
        "frac_diff": (d > 0).float().mean().item(),
        "compiled_vs_fp32_max_abs": (oc.float() - o32).abs().max().item(),
        "eager_vs_fp32_max_abs": (oe.float() - o32).abs().max().item(),
    }
print("OP", json.dumps(op), flush=True)
results["op_compiled_vs_eager"] = op
results["cache_entries_end"] = cache_entries()
note(results["cache_entries_end"] == results["cache_entries_after_warmup"],
     "final layer cache entries unchanged across all phases")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "main's final-layer path (tokens copy == torch.cat([part]), uncompiled "
    "Flux3LastLayer) in the same process, graphs on, re-captured per arm switch",
    **results,
}, default=str))
