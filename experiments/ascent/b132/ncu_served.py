"""Branch 132 (idea 153) acceptance item 4: served-pipeline ncu, main vs candidate, ONE process.

Same setup as e2e_check.py: served engine overlay, PipelineLoader warmup, main's
_adaln_nvfp4_kernel (git 762d1fc6) patched into the module for the main arm.
Per arm: one untimed warm T=80 1024x1024 request, then one T=80 request in which the
cudaProfiler region is opened just before ONE chosen _adaln_nvfp4 launch (step >= 1) and
closed at the next _adaln_nvfp4 launch. So each window holds exactly one adaln launch plus the
block's following kernels (gated-residual add_mul_slice, quantizes) up to the next block.
Windows in order: main M=4096, main M=4176, cand M=4096, cand M=4176.
Graphs off (runner disabled) so the python hook runs at launch time, as in the CUPTI census.
Usage (under profile_kernels capture=region): python ncu_served.py <checkpoint>
"""

import json
import os
import subprocess
import sys
import time
import types

import torch

CKPT = sys.argv[1]
MAIN_REV = "762d1fc684da3037a392f48af7539a52b7250758"

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

PATH = "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_adaln_quant.py"
src = subprocess.check_output(["git", "show", f"{MAIN_REV}:{PATH}"], text=True)
src = src.split("@torch.library.custom_op")[0]
src = src.replace(
    "from .primsts import", "from tensorrt_llm._torch.visual_gen.models.flux3.primsts import"
)
main_mod = types.ModuleType("b132_main_faq")
main_mod.__file__ = "/tmp/b132_main_faq_ncu.py"
with open(main_mod.__file__, "w") as f:
    f.write(src)
exec(compile(src, main_mod.__file__, "exec"), main_mod.__dict__)
CAND_KERNEL = faq._adaln_nvfp4_kernel
MAIN_KERNEL = main_mod._adaln_nvfp4_kernel

here = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(here, "..", "..", "ascent-config", "engine.json")
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
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s; runner {runner is not None}", flush=True)
if runner is not None:
    runner.enabled = False

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)

PROMPT, SEED = "A red fox standing in fresh snow at sunrise", 11


def generate():
    with torch.inference_mode():
        ctx = pipe._encode_prompt(PROMPT, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=SEED
        )
        pipe._package_output(pipe._decode_latents((latent, ids)))
    torch.cuda.synchronize()


cudart = torch.cuda.cudart()
STATE = {"active": False, "open": False, "count": {}, "targets": {}, "windows": []}
_orig_launch = faq._launch


def _hooked(x, *a, **k):
    m = x.numel() // x.shape[-1]
    if STATE["open"]:
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        STATE["open"] = False
    if STATE["active"]:
        n = STATE["count"].get(m, 0)
        STATE["count"][m] = n + 1
        if STATE["targets"].get(m) == n:
            torch.cuda.synchronize()
            cudart.cudaProfilerStart()
            STATE["open"] = True
            STATE["windows"].append({"arm": STATE["arm"], "M": m, "call_index_in_M": n})
            print("WINDOW", json.dumps(STATE["windows"][-1]), flush=True)
    return _orig_launch(x, *a, **k)


faq._launch = _hooked

# M=4096: 4 calls/step -> index 6 is step 1. M=4176: 28 calls/step -> index 40 is step 1.
TARGETS = {4096: 6, 4176: 40}
for arm in ("main", "cand"):
    faq._adaln_nvfp4_kernel = MAIN_KERNEL if arm == "main" else CAND_KERNEL
    STATE.update(active=False, count={}, arm=arm)
    generate()  # warm (Triton JIT for this arm), not profiled
    STATE.update(active=True, count={}, targets=dict(TARGETS))
    generate()
    if STATE["open"]:
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        STATE["open"] = False
    STATE["active"] = False
    print("ARM", arm, "calls per M", json.dumps(STATE["count"]), flush=True)

faq._launch = _orig_launch
faq._adaln_nvfp4_kernel = CAND_KERNEL
print(json.dumps({"passed": len(STATE["windows"]) == 4, "windows": STATE["windows"],
                  "order": "main M4096, main M4176, cand M4096, cand M4176"}))
