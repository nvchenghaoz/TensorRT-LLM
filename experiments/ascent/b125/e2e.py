"""Branch 125 one-process e2e: warp-per-row adaLN NVFP4 kernel vs main's Triton kernel.

Switch: flux3_adaln_quant._IMPL ("warp" = branch, "triton" = main's kernel; the env
TRTLLM_FLUX3_ADALN_NVFP4_IMPL sets it at import). A test-only CUDA-graph extra key on the switch
gives each impl its own graphs in one process (the custom op is opaque to Dynamo, so both impls
share the compiled code).

Loads the pipeline as served (engine.json overlay). Sequence (T=80 = A, T=160 = B):
  STEADY  : A_w (warmup capture), then A_t, A_w, A_t, A_w  -> per impl repeats bitwise, A_w vs A_t
            diff reported (reordered class), steady requests 0 Dynamo compiles / 0 captures
  LONG    : B_t, B_w, B_t, B_w -> same at T=160
  GRAPHOFF: runner disabled: A_w, A_t (vs their graph-on results; reported, main's T80-after-T160
            drift applies)
  CENSUS  : torch.profiler (CUPTI, graph on) over one steady A request per impl: per-kernel
            us/call per grid (M class) for the adaLN NVFP4 kernel.
Usage: python e2e.py <checkpoint>
"""

import collections
import json
import os
import sys
import tempfile
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
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa: E402
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
print("engine overlay", engine, pipeline_config, "impl", faq._IMPL, flush=True)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
c_warm = counters()
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s; dynamo {c_warm}; runner {runner is not None}; "
      f"compile_seconds {wk.COMPILE_SECONDS}", flush=True)
if runner is not None:
    runner.clear()
    runner.register_extra_key_fn("b125_impl", lambda *a, **k: faq._IMPL)

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


def run(label, name, impl):
    faq._IMPL = impl
    prompt, seed = PROMPTS[name]
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    t1 = time.time()
    T, lat, img = generate(prompt, seed)
    rec = {
        "phase": label,
        "req": name,
        "impl": impl,
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
    la, lb = a[1].float(), b[1].float()
    return {
        "latent_bitwise": torch.equal(a[1], b[1]),
        "image_bitwise": torch.equal(a[2], b[2]),
        "latent_max_abs": (la - lb).abs().max().item(),
        "latent_rel_l2": ((la - lb).norm() / lb.norm().clamp_min(1e-12)).item(),
        "image_max_abs": (a[2].int() - b[2].int()).abs().max().item(),
        "image_px_diff": int((a[2] != b[2]).sum().item()),
    }


results = {"warm_unique_graphs": c_warm["unique_graphs"]}

# STEADY at T=80 (before any T=160 request: main's T80-after-T160 drift does not apply).
cap = run("STEADY", "A", "warp")  # first warp capture after runner.clear()
seq = [run("STEADY", "A", impl) for impl in ("triton", "warp", "triton", "warp")]
results["steady"] = [r[0] for r in [cap] + seq]
note(same(seq[0], seq[2]), "T80 triton repeat bitwise")
note(same(seq[1], seq[3]), "T80 warp repeat bitwise")
note(same(cap, seq[1]), "T80 warp first capture == steady warp")
for r in seq[2:]:
    note(r[0]["new_graphs"] == 0 and r[0]["new_captures"] == 0,
         f"steady T80 {r[0]['impl']}: 0 compiles, 0 captures")
results["T80_warp_vs_triton"] = diff(seq[1], seq[0])
results["T80_triton_repeat"] = diff(seq[2], seq[0])
results["T80_warp_repeat"] = diff(seq[3], seq[1])
print("T80 warp vs triton", json.dumps(results["T80_warp_vs_triton"]), flush=True)

# LONG at T=160.
seqb = [run("LONG", "B", impl) for impl in ("triton", "warp", "triton", "warp")]
results["long"] = [r[0] for r in seqb]
note(same(seqb[0], seqb[2]), "T160 triton repeat bitwise")
note(same(seqb[1], seqb[3]), "T160 warp repeat bitwise")
for r in seqb[2:]:
    note(r[0]["new_graphs"] == 0 and r[0]["new_captures"] == 0,
         f"steady T160 {r[0]['impl']}: 0 compiles, 0 captures")
results["T160_warp_vs_triton"] = diff(seqb[1], seqb[0])
print("T160 warp vs triton", json.dumps(results["T160_warp_vs_triton"]), flush=True)

# A after B (main's known drift: reported for both impls, not judged).
a3 = [run("A_AFTER_B", "A", impl) for impl in ("triton", "warp")]
results["A3_vs_A1_triton"] = diff(a3[0], seq[0])
results["A3_vs_A1_warp"] = diff(a3[1], seq[1])
results["A3_warp_vs_triton"] = diff(a3[1], a3[0])

# GRAPH OFF.
if runner is not None:
    runner.enabled = False
    g_w = run("GRAPHOFF", "A", "warp")
    g_t = run("GRAPHOFF", "A", "triton")
    runner.enabled = True
    results["graphoff_warp_vs_graphon_warp(A3)"] = diff(g_w, a3[1])
    results["graphoff_triton_vs_graphon_triton(A3)"] = diff(g_t, a3[0])
    results["graphoff_warp_vs_triton"] = diff(g_w, g_t)
    print("GRAPHOFF", json.dumps({k: v for k, v in results.items() if k.startswith("graphoff")}),
          flush=True)

# CENSUS: CUPTI kernels (graph on) over one steady T=80 request per impl.
census = {}
for impl in ("triton", "warp"):
    faq._IMPL = impl
    generate(*PROMPTS["A"])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        generate(*PROMPTS["A"])
    path = tempfile.mktemp(suffix=".json")
    p.export_chrome_trace(path)
    trace = json.load(open(path))
    os.unlink(path)
    rows = collections.defaultdict(list)
    total = 0.0
    for ev in trace.get("traceEvents", []):
        if ev.get("cat") != "kernel":
            continue
        total += ev.get("dur", 0.0)
        name = ev.get("name", "")
        if "adaln_nvfp4" in name:
            grid = tuple(ev.get("args", {}).get("grid", []))
            rows[(name[:60], grid)].append(ev.get("dur", 0.0))
    census[impl] = {
        "total_kernel_us": round(total, 1),
        "rows": [
            {"name": k[0], "grid": k[1], "calls": len(v), "us_per_call": round(sum(v) / len(v), 2),
             "total_us": round(sum(v), 1)}
            for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1]))
        ],
    }
    print("CENSUS", impl, json.dumps(census[impl]), flush=True)
faq._IMPL = "warp"
results["census"] = census
note(any("flux3_adaln_nvfp4_warp" in r["name"] for r in census["warp"]["rows"]),
     "warp kernel present in the warp request")
note(not any("_adaln_nvfp4_kernel" in r["name"] for r in census["warp"]["rows"]),
     "no Triton adaLN kernel in the warp request")

print(json.dumps({"passed": not fails, "failures": fails, **results}, default=str))
