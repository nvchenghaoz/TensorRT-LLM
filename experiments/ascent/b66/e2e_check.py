"""Branch 66 (idea 14) one-process e2e check: fused LN+adaLN+NVFP4 quantize ON vs OFF.

Switch: flux3_adaln_quant.FLUX3_ADALN_NVFP4 (True = branch: one Triton kernel per NVFP4 block head;
False = main's path: Inductor LN+modulate + trtllm::flux3_nvfp4_quantize_prepared). A test-only
CUDA-graph extra key on the flag gives ON and OFF their own graphs in one process.

Loads the pipeline as served (engine.json overlay: CUDA graphs on, torch_compile + autotune,
warmup [[1024,1024]] t2i).
Phase SERVED (switch as loaded): A(T=80), B(T=160), A -> new Dynamo graphs / CUDA-graph captures
per request; P3 == P1.
Phase SHARED (same process, shared Dynamo history), interleaved per request:
    A_off, A_on, B_off, B_on, A_off, A_on  -> on vs off at the same position, bitwise.
Phase GRAPH: runner disabled (eager replay of the compiled blocks): A_on, A_off vs graph A_on.
Phase PROFILE (graph off so every kernel is visible): torch.profiler CUDA kernels over one ON and
one OFF T=80 request: per-name call counts and device us.
Usage: python e2e_check.py <checkpoint> [served_only]
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
SERVED_ONLY = len(sys.argv) > 2 and sys.argv[2] == "served_only"
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return {"frames": dict(c["frames"]), "unique_graphs": c["stats"].get("unique_graphs", 0)}


from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

try:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq
except ImportError:
    faq = None

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
print("engine overlay", engine, pipeline_config, "faq", faq is not None,
      "env", os.environ.get("TRTLLM_FLUX3_ADALN_QUANT"), flush=True)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
c_warm = counters()
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s; dynamo {c_warm}; runner {runner is not None}",
      flush=True)
fused_blocks = []
if faq is not None:
    tr = pipe.transformer
    for name, mod in tr.named_modules():
        if getattr(mod, "_flux3_adaln_nvfp4", False) and not name.endswith("_orig_mod"):
            fused_blocks.append(name)
print("fused blocks", len(fused_blocks), fused_blocks[:3], fused_blocks[-3:], flush=True)
if runner is not None and faq is not None:
    # Test-only key so ON and OFF get their own graphs. Keys must be registered before capture:
    # drop the warmup graphs (the first SERVED request recaptures, counted in its new_captures).
    runner.clear()
    runner.register_extra_key_fn("b66_adaln_nvfp4", lambda *a, **k: bool(faq.FLUX3_ADALN_NVFP4))

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


def set_flag(on):
    if faq is not None:
        faq.FLUX3_ADALN_NVFP4 = on


def run(label, name, on):
    set_flag(on)
    prompt, seed = PROMPTS[name]
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    t1 = time.time()
    T, lat, img = generate(prompt, seed)
    rec = {
        "phase": label,
        "req": name,
        "on": on,
        "T": T,
        "s": round(time.time() - t1, 2),
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
        "image_max_abs": (a[2].int() - b[2].int()).abs().max().item(),
        "image_px_diff": int((a[2] != b[2]).sum().item()),
    }


results = {"fused_blocks": len(fused_blocks), "env": os.environ.get("TRTLLM_FLUX3_ADALN_QUANT")}

# SERVED: switch as loaded (ON on the branch).
served = [run("SERVED", n, True) for n in ("A", "B", "A")]
note(served[0][0]["new_graphs"] == 0, "no Dynamo compile in first T=80 request after warmup")
note(served[2][0]["new_graphs"] == 0, "no Dynamo compile in repeated T=80 request")
note(served[2][0]["new_captures"] == 0, "no CUDA-graph capture in repeated T=80 request")
results["served"] = [r[0] for r in served]
results["served_P3_eq_P1"] = diff(served[0], served[2])
print("SERVED P3 vs P1", json.dumps(results["served_P3_eq_P1"]), flush=True)

if not SERVED_ONLY and faq is not None:
    note(len(fused_blocks) == 28, f"28 fused NVFP4 blocks (got {len(fused_blocks)})")
    # SHARED Dynamo history, interleaved per request.
    seq = [("A", False), ("A", True), ("B", False), ("B", True), ("A", False), ("A", True)]
    shared = [run("SHARED", n, on) for n, on in seq]
    results["shared"] = [r[0] for r in shared]
    results["shared_on_vs_off"] = []
    for i in range(0, len(shared), 2):
        d = diff(shared[i], shared[i + 1])
        d["req"] = shared[i][0]["req"]
        results["shared_on_vs_off"].append(d)
        print("SHARED CMP", json.dumps(d), flush=True)
        note(d["latent_bitwise"] and d["image_bitwise"], f"SHARED {d['req']}#{i // 2} on==off")
    note(same(shared[1], shared[5]), "SHARED A_on repeat bitwise")
    note(same(shared[0], shared[4]), "SHARED A_off repeat bitwise")
    note(shared[5][0]["new_graphs"] == 0 and shared[5][0]["new_captures"] == 0,
         "steady A_on (3rd A_on): 0 Dynamo compiles and 0 captures")

    # GRAPH OFF vs ON.
    if runner is not None:
        runner.enabled = False
        g_on = run("GRAPHOFF", "A", True)
        g_off = run("GRAPHOFF", "A", False)
        runner.enabled = True
        results["graphoff_on_vs_graphon_on"] = diff(g_on, shared[5])
        results["graphoff_off_vs_graphon_off"] = diff(g_off, shared[4])
        results["graphoff_on_vs_off"] = diff(g_on, g_off)
        print("GRAPH", json.dumps({k: results[k] for k in results if k.startswith("graphoff")}),
              flush=True)
        note(same(g_on, shared[5]), "graph-off ON == graph-on ON")
        note(same(g_on, g_off), "graph-off ON == graph-off OFF")

    # PROFILE (graph off so kernels are individually visible).
    prof = {}
    if runner is not None:
        runner.enabled = False
    for on in (True, False):
        set_flag(on)
        generate(*PROMPTS["A"])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            generate(*PROMPTS["A"])
        cnt = collections.Counter()
        dur = collections.Counter()
        for e in p.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                cnt[e.name] += 1
                dur[e.name] += e.device_time
        prof["on" if on else "off"] = (cnt, dur)
    if runner is not None:
        runner.enabled = True
    set_flag(True)
    on_c, on_d = prof["on"]
    off_c, off_d = prof["off"]
    names = set(on_c) | set(off_c)
    delta = sorted(
        ((n, on_c[n] - off_c[n], round(on_d[n] / max(on_c[n], 1), 2),
          round(off_d[n] / max(off_c[n], 1), 2), round(on_d[n] - off_d[n], 1)) for n in names
         if on_c[n] != off_c[n]),
        key=lambda t: t[1],
    )
    results["profile"] = {
        "kernels_on": sum(on_c.values()),
        "kernels_off": sum(off_c.values()),
        "device_us_on": round(sum(on_d.values()), 1),
        "device_us_off": round(sum(off_d.values()), 1),
        "count_deltas(name, on-off, us/call on, us/call off, total us on-off)": [
            (n[:120], *rest) for n, *rest in delta
        ][:20],
    }
    print("PROFILE", json.dumps(results["profile"]), flush=True)

print(
    json.dumps(
        {
            "passed": not fails,
            "failures": fails,
            "compared_against": "FLUX3_ADALN_NVFP4=False (main's Inductor LN+modulate + "
            "flux3_nvfp4_quantize_prepared) in the same process, same prompts/seeds, final latent "
            "+ uint8 image bitwise",
            "warmup_unique_graphs": c_warm["unique_graphs"],
            **results,
        },
        default=str,
    )
)
