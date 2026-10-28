"""Branch 71 (idea 121): dump main's real LN+adaLN-modulate Inductor kernels (FP8R heads included)
from the served pipeline, with the runtime-chosen launch config of every LN reduction kernel.

Requests: A(T=80), B(T=160), A(T=80). Usage: python b71_dump_real.py <checkpoint>
"""

import json
import os
import sys

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_LOGS_OUT"] = os.path.join(art, "b71_real_output_code.log")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
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
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
print("loaded", flush=True)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)


def generate(prompt, seed):
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
        )
    torch.cuda.synchronize()
    return ctx[0].shape[1]


te = pipe.text_encoder
for prompt, seed in (
    ("A red fox standing in fresh snow at sunrise", 11),
    (prompt_for_bucket(te, 2), 22),
    ("A red fox standing in fresh snow at sunrise", 11),
):
    print("T", generate(prompt, seed), flush=True)

from torch._inductor.codecache import PyCodeCache  # noqa: E402

mods = PyCodeCache.modules
mods = list(mods.values()) if isinstance(mods, dict) else list(mods)
seen = {}
for mod in mods:
    for name in dir(mod):
        obj = getattr(mod, name)
        if not hasattr(obj, "launchers") or "layer_norm" not in name:
            continue
        src = getattr(getattr(obj, "fn", None), "src", "") or ""
        cfgs = [
            {
                "config": str(getattr(launcher, "config", None)),
                "n_regs": getattr(launcher, "n_regs", None),
                "n_spills": getattr(launcher, "n_spills", None),
            }
            for launcher in obj.launchers
        ]
        key = (name, src)
        if key in seen:
            continue
        seen[key] = 1
        print("=" * 30, name, getattr(mod, "__file__", ""), flush=True)
        offered = getattr(obj, "configs", None) or []
        print("configs", json.dumps(cfgs), "offered", [str(c) for c in offered])
        print("inductor_meta num_load", obj.inductor_meta.get("num_load"),
              "num_reduction", obj.inductor_meta.get("num_reduction"))
        print(src)
print(json.dumps({"ln_kernels": len(seen)}))
