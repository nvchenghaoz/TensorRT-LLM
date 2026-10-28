"""Branch 32 diag: one denoiser forward, ON vs OFF, per-block output diffs (same inputs).

Usage: python b32_blockdiag.py <checkpoint>
"""

import json
import os
import sys

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

CKPT = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
tf.FLUX3_BROADCAST_ADALN = False  # warm up main's path
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
    **engine,
)
pipe = PipelineLoader(args, device="cuda:0").load()
model = pipe.transformer
print("loaded", flush=True)

captured = {}
orig_forward = model.forward


def capture(*a, **kw):
    if "kw" not in captured:
        captured["kw"] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    return orig_forward(*a, **kw)


model.forward = capture
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)

with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
    ctx = pipe._encode_prompt("A red fox standing in fresh snow at sunrise", None)
    pipe._prepare_t2i(ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=5)
model.forward = orig_forward
kw = captured["kw"]
print("captured keys", sorted(kw), flush=True)

names = []
for label, blocks in (
    ("txt", model.txt_mode_blocks),
    ("img", model.content_mode_blocks["image"]),
    ("single", model.single_blocks),
):
    for i, b in enumerate(blocks):
        names.append((f"{label}.{i}", b))
records = {}


def make_hook(name):
    def hook(mod, inp, out):
        records.setdefault(name, []).append(out.detach().clone())

    return hook


handles = [b.register_forward_hook(make_hook(n)) for n, b in names]


def run(flag, reps=2):
    tf.FLUX3_BROADCAST_ADALN = flag
    outs = []
    for _ in range(reps):
        records.clear()
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            o = model(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()})
        torch.cuda.synchronize()
        outs.append(({k: r[-1] for k, r in records.items()}, o["x_image"].clone()))
    return outs


off = run(False)
on = run(True)
off2 = run(False)
res = {"off_repeat_bitwise": torch.equal(off[0][1], off[1][1]),
       "on_repeat_bitwise": torch.equal(on[0][1], on[1][1]),
       "off_after_on_bitwise": torch.equal(off[0][1], off2[0][1]),
       "final_max_abs": (on[0][1].float() - off[0][1].float()).abs().max().item(),
       "final_ref_absmax": off[0][1].float().abs().max().item()}
blocks = []
for n, _ in names:
    a, b = off[0][0].get(n), on[0][0].get(n)
    if a is None or b is None:
        continue
    d = (a.float() - b.float()).abs()
    blocks.append({"block": n, "bitwise": torch.equal(a, b), "max_abs": d.max().item(),
                   "frac_diff": (d > 0).float().mean().item(),
                   "ref_absmax": a.float().abs().max().item()})
    print(json.dumps(blocks[-1]), flush=True)
first_bad = next((b["block"] for b in blocks if not b["bitwise"]), None)
print(json.dumps({"passed": first_bad is None, "first_diff_block": first_bad, **res,
                  "blocks": blocks}))
