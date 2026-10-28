"""Branch 82 (idea 117, stage A of 136): CUDAGraphRunner.replay copy_ skip for an explicit
allowlist of step-invariant Flux3 kwargs. ONE process, pipeline as served (engine.json overlay,
cuda_graph_config on, torch_compile + autotune, warmup [[1024,1024]] t2i).

Switch: pipeline_flux3.FLUX3_GRAPH_REUSE_INPUTS (read at every replay).
Numerics (each case: graph-on flag ON, graph-on flag OFF, graph-off, consecutively; final latent +
uint8 image bitwise across all three):
  A (T=80), B (T=160), A after B, then A2 (different T=80 prompt) right after A with the flag on:
  A2 != A and A2 == its graph-off result.
Timing: >=60 warm denoise-only requests (graphs on), then >=20 rounds of OFF / ON / OFF2 with CUDA
events around _run_denoise; median, p10, p90 of ON-OFF and OFF2-OFF.
Usage: python inproc.py <checkpoint> [warm] [rounds]
"""

import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what, flush=True)


def unique_graphs():
    from torch._dynamo.utils import counters as c

    return c["stats"].get("unique_graphs", 0)


from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as pf  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
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
print(
    f"loaded+warmup {time.time() - t0:.0f}s flag={pf.FLUX3_GRAPH_REUSE_INPUTS} "
    f"runner={runner is not None and runner.enabled} "
    f"captures={len(runner.graphs) if runner else 0} "
    f"reuse_fn={getattr(runner, '_reusable_input_fn', None) is not None}",
    flush=True,
)
note(runner is not None and runner.enabled, "graph runner present and enabled")
note(getattr(runner, "_reusable_input_fn", None) is not None, "reusable-input fn registered")
te = pipe.text_encoder

# Time _run_denoise with CUDA events (host runs ahead; events bracket the GPU work).
_orig_denoise = pipe._run_denoise
_times = []


def _timed_denoise(*a, **k):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = _orig_denoise(*a, **k)
    e.record()
    _times.append((s, e))
    return out


pipe._run_denoise = _timed_denoise

# Diagnostic: count copies the runner will skip (same object as last copied source).
_skips = [0]
_orig_replay = runner.replay


def _counting_replay(key, a, kw):
    if pf.FLUX3_GRAPH_REUSE_INPUTS and runner._reusable_input_fn is not None:
        for k, v in kw.items():
            if isinstance(v, torch.Tensor) and runner._reusable_input_fn(k):
                if runner._last_src.get((key, k)) is v:
                    _skips[0] += 1
    return _orig_replay(key, a, kw)


runner.replay = _counting_replay


def generate(prompt, seed, decode=True):
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
        )
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image if decode else None
    torch.cuda.synchronize()
    return ctx[0].shape[1], latent.clone(), (img.clone() if img is not None else None)


PROMPTS = {
    "A": ("A red fox standing in fresh snow at sunrise", 11),
    "A2": ("A blue sailboat on a calm lake at dusk, oil painting", 11),
    "B": (prompt_for_bucket(te, 2), 22),
}
results = {}


def run(label, name, flag, graphs, decode=True):
    pf.FLUX3_GRAPH_REUSE_INPUTS = flag
    runner.enabled = graphs
    caps = len(runner.graphs)
    prev = unique_graphs()
    sk0 = _skips[0]
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name], decode=decode)
    rec = {
        "label": label, "req": name, "flag": flag, "graphs": graphs, "T": T,
        "s": round(time.time() - t1, 2),
        "new_dynamo_graphs": unique_graphs() - prev,
        "new_captures": len(runner.graphs) - caps,
        "copy_skips": _skips[0] - sk0,
    }
    print(label, json.dumps(rec), flush=True)
    results.setdefault("requests", []).append(rec)
    return rec, lat, img


def cmp(a, b, what, expect_equal=True):
    d = {
        "what": what,
        "latent_bitwise": torch.equal(a[1], b[1]),
        "image_bitwise": torch.equal(a[2], b[2]),
        "latent_max_abs": (a[1].float() - b[1].float()).abs().max().item(),
        "image_max_abs": (a[2].int() - b[2].int()).abs().max().item(),
    }
    print("CMP", json.dumps(d), flush=True)
    results.setdefault("comparisons", []).append(d)
    if expect_equal:
        note(d["latent_bitwise"] and d["image_bitwise"], what)
    else:
        note(not d["latent_bitwise"] and not d["image_bitwise"], what)


def case(tag, name):
    on = run(f"{tag}_ON", name, True, True)
    off = run(f"{tag}_OFF", name, False, True)
    eag = run(f"{tag}_EAGER", name, True, False)
    cmp(on, off, f"{tag}: graph flag ON == graph flag OFF")
    cmp(on, eag, f"{tag}: graph flag ON == graph-off")
    cmp(off, eag, f"{tag}: graph flag OFF == graph-off")
    return on, off, eag


a1 = case("A_T80", "A")
note(a1[0][0]["T"] == 80, f"A is T=80 (got {a1[0][0]['T']})")
note(a1[0][0]["copy_skips"] > 0, "flag-on request skipped copies")
note(a1[1][0]["copy_skips"] == 0, "flag-off request skipped nothing")
b1 = case("B_T160", "B")
note(b1[0][0]["T"] == 160, f"B is T=160 (got {b1[0][0]['T']})")
a3 = case("A_T80_after_T160", "A")
note(a3[0][0]["new_captures"] == 0, "A after B: 0 new captures")
# Two different T=80 prompts back to back, flag on, then each one's graph-off result.
x = run("SEQ_A_ON", "A", True, True)
y = run("SEQ_A2_ON", "A2", True, True)
note(y[0]["T"] == 80, f"A2 is T=80 (got {y[0]['T']})")
xe = run("SEQ_A_EAGER", "A", True, False)
ye = run("SEQ_A2_EAGER", "A2", True, False)
cmp(x, y, "A vs A2 differ (no stale input replayed)", expect_equal=False)
cmp(x, xe, "SEQ A flag ON == A graph-off")
cmp(y, ye, "SEQ A2 flag ON (right after A) == A2 graph-off")
cmp(x, a1[2], "SEQ A == first A graph-off (repeat)")
steady = [r for r in results["requests"] if r["graphs"] and r["label"] != "B_T160_ON"]
note(all(r["new_captures"] == 0 for r in steady), "0 captures in steady graph-on requests")
# The first T=160 request (B_T160_ON) is a new shape bucket: it compiles and captures once.
note(all(r["new_dynamo_graphs"] == 0 for r in results["requests"] if r["label"] != "B_T160_ON"),
     "0 Dynamo recompiles in all requests after the first T=160 request")
results["first_T160_request"] = next(r for r in results["requests"] if r["label"] == "B_T160_ON")

# ---------------- timing: in-process interleaved OFF / ON / OFF2 ----------------
runner.enabled = True
pf.FLUX3_GRAPH_REUSE_INPUTS = True
print(f"warming {WARM} denoise-only requests", flush=True)
for i in range(WARM):
    generate(*PROMPTS["A"], decode=False)
    if i % 20 == 19:
        print("warm", i + 1, flush=True)
rows = []
for r in range(max(ROUNDS, 1)):
    ms = {}
    for arm, flag in (("OFF", False), ("ON", True), ("OFF2", False)):
        pf.FLUX3_GRAPH_REUSE_INPUTS = flag
        _times.clear()
        generate(*PROMPTS["A"], decode=False)
        s, e = _times[-1]
        ms[arm] = s.elapsed_time(e)
    rows.append(ms)
    print("round", r, json.dumps({k: round(v, 3) for k, v in ms.items()}), flush=True)
pf.FLUX3_GRAPH_REUSE_INPUTS = True


def pct(v, q):
    v = sorted(v)
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def summ(v):
    return {"median": statistics.median(v), "p10": pct(v, 0.1), "p90": pct(v, 0.9)}


d_on = [r["ON"] - r["OFF"] for r in rows]
d_aa = [r["OFF2"] - r["OFF"] for r in rows]
timing = {
    "rounds": ROUNDS, "warm_requests": WARM,
    "denoise_ms": {arm: summ([r[arm] for r in rows]) for arm in ("OFF", "ON", "OFF2")},
    "ON_minus_OFF_ms": summ(d_on),
    "OFF2_minus_OFF_ms": summ(d_aa),
    "ON_minus_mean_OFFs_ms": summ([r["ON"] - (r["OFF"] + r["OFF2"]) / 2 for r in rows]),
}
results["timing"] = timing
print("TIMING", json.dumps(timing), flush=True)
on_med = timing["denoise_ms"]["ON"]["median"]
offs = [r["OFF"] for r in rows] + [r["OFF2"] for r in rows]
noise = (pct(offs, 0.9) - pct(offs, 0.1)) / statistics.median(offs)
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "flag OFF (main's replay: copy_ every tensor kwarg) and graph-off eager, "
    "same process, same Dynamo history position; final latent + uint8 image bitwise",
    "batch": {"1": {"step_ms": on_med, "noise_floor": noise}},
    "output": {"class": "exact"},
    **results,
}, default=str))
