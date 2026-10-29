"""Branch 193 one-process in-model check: LN_ID warp adaLN (exact) on vs off.

Arms switched in-process by flux3_sm120_adaln_nvfp4._LN_ID (read in launch(), host side); a test-only
CUDA-graph extra key gives each arm its own graphs. Both NVRTC modules are built before requests.
Checks: identity registry populated at prepare; graph-off request per arm counts warp launches and
how many took LN_ID=1 (ln1: 448/448, ln0: 0/448); capture request per arm; then ROUNDS alternating
steady requests per arm: latents+image bitwise vs the first request, 0 Dynamo graphs, 0 new captures,
0 new NVRTC modules after warmup; graph on == off.
Usage: python e2e.py <checkpoint> [rounds]
"""

import collections
import json
import os
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa
        return f"ERR {e}"


INFO = {"head": sh("git rev-parse HEAD"), "log": sh("git log --oneline -3"),
        "anc_9e499b22": sh("git merge-base --is-ancestor 9e499b22 HEAD; echo rc=$?"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sh("nvidia-smi --query-gpu=uuid,clocks.sm --format=csv,noheader")}
print("INFO", json.dumps(INFO), flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return {"frames": dict(c["frames"]), "unique_graphs": c["stats"].get("unique_graphs", 0)}


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

LAUNCHES = []
_orig_compile = wk._compile


def _spy(device, warps, use_pdl, extra_defines=()):
    LAUNCHES.append(tuple(extra_defines))
    return _orig_compile(device, warps, use_pdl, extra_defines)


wk._compile = _spy
ARM = ["ln1"]


def set_arm(a):
    ARM[0] = a
    wk._LN_ID = a == "ln1"


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
set_arm("ln1")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
c_warm = counters()
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
warm_defs = collections.Counter(LAUNCHES)
print(f"loaded+warmup {time.time() - t0:.0f}s; dynamo {c_warm}; runner {runner is not None}; "
      f"registry {len(wk._IDENTITY_AFFINE)}; warmup launches {dict(warm_defs)}; regs {wk.NUM_REGS}",
      flush=True)
note(len(wk._IDENTITY_AFFINE) > 0, "identity registry populated at prepare")
for d in (wk._DEFAULT_DEFINES, tuple(wk._DEFAULT_DEFINES) + ("LN_ID=1",)):
    _orig_compile(torch.device("cuda", 0), wk._DEFAULT_WARPS, False, d)
print("modules", {str(k): v for k, v in wk.NUM_REGS.items()}, flush=True)
if runner is not None:
    runner.clear()
    runner.register_extra_key_fn("b193_ln", lambda *a, **k: ARM[0])

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
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return ctx[0].shape[1], latent.clone(), img.clone()


def run(arm, tag):
    set_arm(arm)
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    n_mod = len(wk._MODULES)
    n_l = len(LAUNCHES)
    t1 = time.time()
    T, lat, img = generate()
    defs = LAUNCHES[n_l:]
    rec = {"arm": arm, "tag": tag, "s": round(time.time() - t1, 2), "T": T,
           "warp_launches": len(defs), "ln_id_launches": sum("LN_ID=1" in d for d in defs),
           "new_graphs": counters()["unique_graphs"] - prev,
           "new_captures": (len(runner.graphs) if runner is not None else 0) - prev_caps,
           "new_nvrtc_modules": len(wk._MODULES) - n_mod}
    print("REQ", json.dumps(rec), flush=True)
    return rec, lat, img


def same(a, b):
    return torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


results = {"info": INFO, "warm_unique_graphs": c_warm["unique_graphs"],
           "registry_entries": len(wk._IDENTITY_AFFINE)}
recs = []
# graph OFF first: eager launch counts per request (448 expected), latents per arm
goff = {}
if runner is not None:
    runner.enabled = False
for a in ("ln1", "ln0"):
    goff[a] = run(a, "graph_off")
    recs.append(goff[a][0])
if runner is not None:
    runner.enabled = True
note(goff["ln1"][0]["warp_launches"] == 448 and goff["ln1"][0]["ln_id_launches"] == 448,
     "graph off ln1: 448/448 warp launches LN_ID")
note(goff["ln0"][0]["warp_launches"] == 448 and goff["ln0"][0]["ln_id_launches"] == 0,
     "graph off ln0: 0/448 LN_ID")
ref = goff["ln0"]
note(same(goff["ln1"], ref), "graph off: ln1 vs ln0 latents+image bitwise")
caps = {}
for a in ("ln1", "ln0"):
    caps[a] = run(a, "capture")
    recs.append(caps[a][0])
    note(same(caps[a], ref), f"capture {a}: bitwise vs graph-off ln0")
    note(caps[a][0]["new_graphs"] == 0 and caps[a][0]["new_nvrtc_modules"] == 0,
         f"capture {a}: 0 Dynamo graphs / NVRTC modules")
note(caps["ln1"][0]["ln_id_launches"] == caps["ln1"][0]["warp_launches"]
     and caps["ln1"][0]["warp_launches"] > 0, "capture ln1: every warp launch in its graph key LN_ID")
note(caps["ln0"][0]["ln_id_launches"] == 0, "capture ln0: no LN_ID")
for r in range(ROUNDS):
    order = ("ln0", "ln1") if r % 2 == 0 else ("ln1", "ln0")
    for a in order:
        out = run(a, f"steady{r}")
        recs.append(out[0])
        note(same(out, ref), f"round {r} {a}: latent+image bitwise vs reference")
        note(out[0]["new_graphs"] == 0 and out[0]["new_captures"] == 0
             and out[0]["new_nvrtc_modules"] == 0, f"round {r} {a}: 0 compiles/captures/NVRTC")
results["requests"] = recs
print(json.dumps({"passed": not fails, "failures": fails, **results}, default=str))
