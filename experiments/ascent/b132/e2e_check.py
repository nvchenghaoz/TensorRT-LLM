"""Branch 132 (idea 153) one-process e2e check + in-model CUPTI census.

Arms (switched in ONE process; the kernel runs inside the opaque custom op, so no Dynamo change):
  cand : this branch's _adaln_nvfp4_kernel (x chunks kept in registers, selected-row loads)
  main : main's _adaln_nvfp4_kernel, loaded from git (762d1fc6) and patched into the module
  off  : FLUX3_ADALN_NVFP4=False (Inductor LN+mod + flux3_nvfp4_quantize_prepared), informational
A test-only CUDA-graph extra key on the arm gives each arm its own graphs.

SERVED : cand A(T=80), B(T=160), A; 0 compiles/captures in the repeat (P3 vs P1 reported).
SHARED : per request interleaved main/cand/off for A, B, A (same history position): cand == main.
GRAPHOFF: runner disabled: A and B, main vs cand, and vs graph-on.
CENSUS : graphs off, CUPTI kernel durations of _adaln_nvfp4_kernel per M class, main vs cand, x2.
Usage: python e2e_check.py <checkpoint>
"""

import collections
import json
import os
import statistics
import subprocess
import sys
import time
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
MAIN_REV = "762d1fc684da3037a392f48af7539a52b7250758"
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return {"frames": dict(c["frames"]), "unique_graphs": c["stats"].get("unique_graphs", 0)}


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

PATH = "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_adaln_quant.py"
src = subprocess.check_output(["git", "show", f"{MAIN_REV}:{PATH}"], text=True)
# Kernel + helpers only (the custom op is already registered by this tree's module).
src = src.split("@torch.library.custom_op")[0]
src = src.replace(
    "from .primsts import", "from tensorrt_llm._torch.visual_gen.models.flux3.primsts import"
)
main_mod = types.ModuleType("b132_main_faq")
main_mod.__file__ = "/tmp/b132_main_faq.py"
with open(main_mod.__file__, "w") as f:
    f.write(src)
exec(compile(src, main_mod.__file__, "exec"), main_mod.__dict__)
CAND_KERNEL = faq._adaln_nvfp4_kernel
MAIN_KERNEL = main_mod._adaln_nvfp4_kernel
note(hasattr(faq, "_row_cached") and not hasattr(main_mod, "_row_cached"), "arms differ")

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
c_warm = counters()
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s; dynamo {c_warm}; runner {runner is not None}",
      flush=True)
fused_blocks = [n for n, m in pipe.transformer.named_modules()
                if getattr(m, "_flux3_adaln_nvfp4", False) and not n.endswith("_orig_mod")]
print("fused blocks", len(fused_blocks), flush=True)

ARM = {"name": "cand"}
if runner is not None:
    runner.clear()
    runner.register_extra_key_fn("b132_arm", lambda *a, **k: ARM["name"])

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


def set_arm(arm):
    ARM["name"] = arm
    faq.FLUX3_ADALN_NVFP4 = arm != "off"
    faq._adaln_nvfp4_kernel = MAIN_KERNEL if arm == "main" else CAND_KERNEL


def run(label, name, arm):
    set_arm(arm)
    prompt, seed = PROMPTS[name]
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    t1 = time.time()
    T, lat, img = generate(prompt, seed)
    rec = {
        "phase": label, "req": name, "arm": arm, "T": T, "s": round(time.time() - t1, 2),
        "new_graphs": counters()["unique_graphs"] - prev,
        "new_captures": (len(runner.graphs) if runner is not None else 0) - prev_caps,
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
        "image_px_diff": int((a[2] != b[2]).sum().item()),
    }


results = {"fused_blocks": len(fused_blocks), "runner": runner is not None}
note(len(fused_blocks) >= 28, f">=28 fused NVFP4 blocks (got {len(fused_blocks)})")

served = [run("SERVED", n, "cand") for n in ("A", "B", "A")]
note(served[2][0]["new_graphs"] == 0, "no Dynamo compile in repeated T=80 request")
note(served[2][0]["new_captures"] == 0, "no CUDA-graph capture in repeated T=80 request")
results["served"] = [r[0] for r in served]
results["served_P3_vs_P1_info"] = diff(served[0], served[2])
print("SERVED P3 vs P1 (info, known main leak)", json.dumps(results["served_P3_vs_P1_info"]),
      flush=True)

seq = [(n, m) for n in ("A", "B", "A") for m in ("main", "cand", "off")]
shared = [run("SHARED", n, m) for n, m in seq]
results["shared"] = [r[0] for r in shared]
results["shared_cmp"] = []
for i in range(0, len(shared), 3):
    mn, cd, off = shared[i : i + 3]
    d = {"req": mn[0]["req"], "T": mn[0]["T"], "cand_vs_main": diff(cd, mn),
         "cand_vs_off_info": diff(cd, off)}
    results["shared_cmp"].append(d)
    print("SHARED CMP", json.dumps(d), flush=True)
    note(same(cd, mn), f"SHARED {d['req']}#{i // 3} cand == main (graphs on)")
note(shared[7][0]["new_graphs"] == 0 and shared[7][0]["new_captures"] == 0,
     "steady A cand: 0 compiles and 0 captures")
note(any(r[0]["new_captures"] > 0 for r in shared[:2]) or runner is None or
     len(runner.graphs) > 0, "CUDA-graph path captures")

if runner is not None:
    results["graph_count"] = len(runner.graphs)
    runner.enabled = False
    go = {}
    for n in ("A", "B", "A"):
        for arm in ("main", "cand"):
            go[(n, arm, len([k for k in go if k[0] == n]) // 2)] = run("GRAPHOFF", n, arm)
    a_main, a_cand = go[("A", "main", 0)], go[("A", "cand", 0)]
    b_main, b_cand = go[("B", "main", 0)], go[("B", "cand", 0)]
    a2_main, a2_cand = go[("A", "main", 1)], go[("A", "cand", 1)]
    results["graphoff"] = {
        "A_cand_vs_main": diff(a_cand, a_main),
        "B_cand_vs_main": diff(b_cand, b_main),
        "A2_cand_vs_main": diff(a2_cand, a2_main),
        "A_graphoff_vs_graphon_cand_same_history_info": diff(a_cand, shared[7]),
    }
    print("GRAPHOFF", json.dumps(results["graphoff"]), flush=True)
    note(same(a_cand, a_main), "graph-off A(T=80) cand == main")
    note(same(b_cand, b_main), "graph-off B(T=160) cand == main")
    note(same(a2_cand, a2_main), "graph-off A2(T=80) cand == main")

# CENSUS (graphs off: every kernel visible to CUPTI). M of each call recorded in launch order.
mlog = []
_orig_launch = faq._launch


def _tagged_launch(x, *a, **k):
    mlog.append(x.numel() // x.shape[-1])
    return _orig_launch(x, *a, **k)


faq._launch = _tagged_launch
if runner is not None:
    runner.enabled = False
census = collections.defaultdict(list)
for arm in ("main", "cand", "main", "cand"):
    set_arm(arm)
    generate(*PROMPTS["A"])
    mlog.clear()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        generate(*PROMPTS["A"])
    ks = sorted(
        (e for e in p.events()
         if e.device_type == torch.autograd.DeviceType.CUDA and "_adaln_nvfp4_kernel" in e.name),
        key=lambda e: e.time_range.start,
    )
    rec = {"arm": arm, "calls": len(ks), "logged": len(mlog)}
    if len(ks) == len(mlog):
        per = collections.defaultdict(list)
        for m, e in zip(mlog, ks):
            per[m].append(e.device_time)
        for m, v in sorted(per.items()):
            rec[f"M{m}"] = {"calls": len(v), "us_per_call": round(sum(v) / len(v), 3),
                            "us_median": round(statistics.median(v), 3)}
    else:
        rec["us_per_call_all"] = round(sum(e.device_time for e in ks) / max(len(ks), 1), 3)
    census[arm].append(rec)
    print("CENSUS", json.dumps(rec), flush=True)
faq._launch = _orig_launch
if runner is not None:
    runner.enabled = True
set_arm("cand")
results["census"] = census

print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "same process, same prompts/seeds/history position: main's "
    "_adaln_nvfp4_kernel (git 762d1fc6) patched in vs this branch's; final latent + uint8 image "
    "bitwise, graphs on and off; Inductor off path informational",
    "warmup_unique_graphs": c_warm["unique_graphs"],
    **results,
}, default=str))
