"""Branch 81 (idea 134) one-process e2e check: x_mod store skip vs store-forced vs OFF.

Modes (module-level switches in flux3_adaln_quant, Dynamo guards on both):
  skip  : FLUX3_ADALN_NVFP4=True,  FLUX3_ADALN_NVFP4_STORE_X_MOD=False (branch default)
  store : FLUX3_ADALN_NVFP4=True,  FLUX3_ADALN_NVFP4_STORE_X_MOD=True  (branch 66 / main)
  off   : FLUX3_ADALN_NVFP4=False (pre-66 Inductor LN+mod + flux3_nvfp4_quantize_prepared)
A test-only CUDA-graph extra key on (flag, store) gives each mode its own graphs.

SERVED: skip as loaded, A(T=80), B(T=160), A: P3 == P1, 0 compiles/captures in the repeat.
SHARED: per request interleaved off/store/skip for A, B, A: all three bitwise equal.
GRAPHOFF: runner disabled, A skip and A store vs graph-on.
PROFILE: torch.profiler (CUPTI) over one steady T=80 request each in skip and store (graphs off):
  _adaln_nvfp4_kernel calls and us/call.
Usage: python e2e_check.py <checkpoint>
"""

import collections
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

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

# Static predicate census: which fused blocks skip the store at the served shapes.
fused_blocks, skip_blocks = [], []
for name, mod in pipe.transformer.named_modules():
    if getattr(mod, "_flux3_adaln_nvfp4", False) and not name.endswith("_orig_mod"):
        fused_blocks.append(name)
        rows = 4176 if type(mod).__name__ == "Flux3SingleStreamBlock" else 4096
        x = torch.empty(1, rows, 3072, device="cuda", dtype=torch.bfloat16)
        cos = torch.empty(rows, 128, device="cuda")
        if faq._x_mod_unread(mod, x, (cos, cos.clone())):
            skip_blocks.append(name)
print("fused blocks", len(fused_blocks), "skip blocks", len(skip_blocks), skip_blocks[:3],
      [b for b in fused_blocks if b not in skip_blocks], flush=True)
if runner is not None:
    runner.clear()
    runner.register_extra_key_fn(
        "b81_adaln_mode",
        lambda *a, **k: (bool(faq.FLUX3_ADALN_NVFP4), bool(faq.FLUX3_ADALN_NVFP4_STORE_X_MOD)),
    )

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


def set_mode(mode):
    faq.FLUX3_ADALN_NVFP4 = mode != "off"
    faq.FLUX3_ADALN_NVFP4_STORE_X_MOD = mode == "store"


def run(label, name, mode):
    set_mode(mode)
    prompt, seed = PROMPTS[name]
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    t1 = time.time()
    T, lat, img = generate(prompt, seed)
    rec = {
        "phase": label, "req": name, "mode": mode, "T": T, "s": round(time.time() - t1, 2),
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


results = {"fused_blocks": len(fused_blocks), "skip_blocks": len(skip_blocks)}
note(len(fused_blocks) == 28, f"28 fused NVFP4 blocks (got {len(fused_blocks)})")
note(len(skip_blocks) == 28, f"28 skip blocks (got {len(skip_blocks)})")

served = [run("SERVED", n, "skip") for n in ("A", "B", "A")]
note(served[2][0]["new_graphs"] == 0, "no Dynamo compile in repeated T=80 request")
note(served[2][0]["new_captures"] == 0, "no CUDA-graph capture in repeated T=80 request")
results["served"] = [r[0] for r in served]
results["served_P3_eq_P1"] = diff(served[0], served[2])
note(same(served[0], served[2]), "SERVED P3 == P1")
print("SERVED P3 vs P1", json.dumps(results["served_P3_eq_P1"]), flush=True)

seq = [(n, m) for n in ("A", "B", "A") for m in ("off", "store", "skip")]
shared = [run("SHARED", n, m) for n, m in seq]
results["shared"] = [r[0] for r in shared]
results["shared_cmp"] = []
for i in range(0, len(shared), 3):
    off, store, skip = shared[i : i + 3]
    d = {"req": off[0]["req"], "skip_vs_off": diff(skip, off), "skip_vs_store": diff(skip, store),
         "store_vs_off": diff(store, off)}
    results["shared_cmp"].append(d)
    print("SHARED CMP", json.dumps(d), flush=True)
    note(same(skip, off) and same(skip, store) and same(store, off), f"SHARED {d['req']}#{i // 3}")
note(same(shared[2], served[0]), "SHARED A skip == SERVED A")
note(shared[8][0]["new_graphs"] == 0 and shared[8][0]["new_captures"] == 0,
     "steady A skip: 0 compiles and 0 captures")

if runner is not None:
    runner.enabled = False
    g_skip = run("GRAPHOFF", "A", "skip")
    g_store = run("GRAPHOFF", "A", "store")
    results["graphoff_skip_vs_graphon_skip"] = diff(g_skip, shared[8])
    results["graphoff_skip_vs_store"] = diff(g_skip, g_store)
    print("GRAPH", json.dumps({k: results[k] for k in results if k.startswith("graphoff")}),
          flush=True)
    note(same(g_skip, shared[8]), "graph-off skip == graph-on skip")
    note(same(g_skip, g_store), "graph-off skip == graph-off store")

# PROFILE (graphs off: every kernel visible to CUPTI).
prof = {}
for mode in ("store", "skip", "store", "skip"):
    set_mode(mode)
    generate(*PROMPTS["A"])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        generate(*PROMPTS["A"])
    cnt = collections.Counter()
    dur = collections.Counter()
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            cnt[e.name] += 1
            dur[e.name] += e.device_time
    k = [n for n in cnt if "_adaln_nvfp4_kernel" in n]
    calls = sum(cnt[n] for n in k)
    us = sum(dur[n] for n in k)
    rec = {"mode": mode, "adaln_calls": calls, "adaln_us_total": round(us, 1),
           "adaln_us_per_call": round(us / max(calls, 1), 2),
           "device_us_total": round(sum(dur.values()), 1)}
    prof.setdefault(mode, []).append(rec)
    print("PROFILE", json.dumps(rec), flush=True)
if runner is not None:
    runner.enabled = True
set_mode("skip")
results["profile"] = prof

print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "same process, same prompts/seeds: store-forced (branch 66 kernel) and "
    "FLUX3_ADALN_NVFP4=False (pre-66 Inductor LN+mod + flux3_nvfp4_quantize_prepared); final "
    "latent + uint8 image bitwise",
    "warmup_unique_graphs": c_warm["unique_graphs"],
    **results,
}, default=str))
