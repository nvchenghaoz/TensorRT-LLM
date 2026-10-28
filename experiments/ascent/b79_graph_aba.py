"""Branch 79: is the graphs-on A/B/A replay mismatch (job 1097 SERVED: T=80 replay after a T=160
capture != first T=80) caused by the compiled final layer, or present on main's path too?

One process, pipeline as served (engine.json overlay, CUDA graphs on). Arms by wrapping
transformer.final_layer["image"] (as b79_e2e_check.py): main = uncompiled layer + token copy
(== main's torch.cat), branch = compiled layer + view (as loaded).
For arm in (main, branch, main, branch): runner.clear(); A (capture), A (replay), B (capture),
A (replay). Reports per arm: A2==A1 (replay before B) and A3==A1 (replay after B capture).
Usage: python b79_graph_aba.py <checkpoint>
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
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
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
tr = pipe.transformer
fl = tr.final_layer["image"]
compiled = fl
eager = getattr(fl, "_orig_mod", fl)
print(f"loaded {time.time() - t0:.0f}s final layer {type(fl).__name__}", flush=True)
MODE = {"copy": False, "eager": False}


class Switch(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_c = compiled

    @property
    def adaLN_modulation(self):
        return eager.adaLN_modulation

    def forward(self, x, vec, mod=None):
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
    return latent.clone(), img.clone()


PROMPTS = {"A": ("A red fox standing in fresh snow at sunrise", 11),
           "B": (prompt_for_bucket(te, 2), 22)}
ARMS = {"main": (True, True), "branch": (False, False)}
out = {"runs": []}
first_A = {}
for arm in ("main", "branch", "main", "branch"):
    MODE["copy"], MODE["eager"] = ARMS[arm]
    runner.clear()
    seq = []
    for name in ("A", "A", "B", "A"):
        caps = len(runner.graphs)
        seq.append(generate(*PROMPTS[name]))
        print(arm, name, "new captures", len(runner.graphs) - caps, flush=True)
    a1, a2, a3 = seq[0], seq[1], seq[3]
    rec = {
        "arm": arm,
        "A2_eq_A1": torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1]),
        "A3_eq_A1_after_B": torch.equal(a1[0], a3[0]) and torch.equal(a1[1], a3[1]),
        "A3_max_abs_px": (a1[1].int() - a3[1].int()).abs().max().item(),
        "A1_eq_prev_same_arm": (torch.equal(first_A[arm][1], a1[1]) if arm in first_A
                                else None),
    }
    first_A.setdefault(arm, a1)
    print("ARM", json.dumps(rec), flush=True)
    out["runs"].append(rec)
main_ok = all(r["A3_eq_A1_after_B"] for r in out["runs"] if r["arm"] == "main")
br_ok = all(r["A3_eq_A1_after_B"] for r in out["runs"] if r["arm"] == "branch")
out.update({"main_A_after_B_bitwise": main_ok, "branch_A_after_B_bitwise": br_ok,
            "passed": br_ok or not main_ok})
print(json.dumps(out), flush=True)
