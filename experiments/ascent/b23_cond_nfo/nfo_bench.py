"""Branch 23 (idea 56): TRTLLM_FLUX3_COND_ATTN_NFO default flip for the 5 i2i image_cond blocks.

ONE process: the pipeline as served (engine.json overlay; torch.compile + autotune; warmup
[[2048, 2048]] i2i), CUDA graphs off (block-level timing). One i2i request (frozen 2048^2
reference PNG, 16 steps) captures each image_cond block's real step-0 inputs (+ image.1 control).

Modes (argv[2]):
  gate   in-process A/B: flips block._sm120_attn_nvfp4_out on the 5 cond blocks between the
         value prepare gives with COND_ATTN_NFO=0 (False) and =1 (eligibility recomputed with the
         env var set; asserted True). Interleaved passes OFF,ON,OFF,ON,OFF,ON (5 warmup + 50 timed
         per pass per block); A/A spread = spread of the per-arm 5-block sums across passes.
         Correctness: per block and arm, pristine inputs restored, call 0 then 100 repeated calls
         bitwise vs call 0; ON call 0 bitwise vs OFF call 0; inputs unchanged. Kernel names per
         arm from torch.profiler (one call). Last line: {"passed": ...}.
  bench  loaded route only (whatever prepare chose in this tree), interleaved 3 passes; last line
         is the evidence JSON with step_ms = sum of the 5 cond block medians (min over passes).
"""

import glob
import hashlib
import json
import os
import statistics
import sys
import time

import torch

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "bench"
WARMUP, ITERS, REPEATS = 5, 50, 100
INPUTS_DIR = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/"
    "flux3-klein-nvfp4-2048/v1/inputs"
)
fails = []


def log(*a):
    print(f"[b23 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


def loadavg():
    try:
        return open("/proc/loadavg").read().strip()
    except OSError:
        return None


from tensorrt_llm._torch.visual_gen.models.flux3 import primsts  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    _load_image_pixels,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
engine.pop("env", None)
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine["cuda_graph_config"] = {"enable": False}
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    parallel_config={"tp_size": 1},
    torch_compile_config={"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["i2i"]},
    **engine,
)
log("env", {k: v for k, v in os.environ.items() if k.startswith("TRTLLM_FLUX3")})
log("loadavg start", loadavg())
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
log(f"loaded+warmup {time.time() - t0:.0f}s")
tr = pipe.transformer
cond_blocks = list(tr.content_mode_blocks["image_cond"])
img_block1 = tr.content_mode_blocks["image"][1]


def orig(b):
    return getattr(b, "_orig_mod", b)


def route_of(b):
    b = orig(b)
    return {
        "cond_stream": bool(getattr(b, "_flux3_cond_stream", False)),
        "tail_pair": bool(getattr(b, "_sm120_tail_pair", False)),
        "side_gate_up": int(getattr(b, "_sm120_side_gate_up", 0)),
        "attn_nvfp4_out": bool(getattr(b, "_sm120_attn_nvfp4_out", False)),
        "pair_residual": bool(getattr(b, "_sm120_pair_residual", False)),
    }


routes = {f"image_cond.{j}": route_of(b) for j, b in enumerate(cond_blocks)}
routes["image.1"] = route_of(img_block1)
log("routes (as loaded)", json.dumps(routes))
for k, r in routes.items():
    note(r["pair_residual"], f"{k}: _sm120_pair_residual True")
    note(r["tail_pair"], f"{k}: _sm120_tail_pair True")
note(routes["image.1"]["attn_nvfp4_out"], "image.1: nfo True (control)")
cond_keys = [f"image_cond.{j}" for j in range(len(cond_blocks))]
note(all(routes[k]["cond_stream"] for k in cond_keys), "cond blocks have _flux3_cond_stream")

# Eligibility under each env value (recomputed exactly as prepare does).
elig = {}
old_env = os.environ.get("TRTLLM_FLUX3_COND_ATTN_NFO")
for val in ("0", "1"):
    os.environ["TRTLLM_FLUX3_COND_ATTN_NFO"] = val
    primsts._cond_attn_nfo_enabled.cache_clear()
    elig[val] = [bool(primsts._sm120_attn_nvfp4_out_eligible(orig(b))) for b in cond_blocks]
if old_env is None:
    os.environ.pop("TRTLLM_FLUX3_COND_ATTN_NFO", None)
else:
    os.environ["TRTLLM_FLUX3_COND_ATTN_NFO"] = old_env
primsts._cond_attn_nfo_enabled.cache_clear()
log("eligibility per env value", json.dumps(elig), "loaded default ->",
    primsts._cond_attn_nfo_enabled())
note(elig["0"] == [False] * len(cond_blocks), "env=0 -> nfo False for every cond block")
note(elig["1"] == [True] * len(cond_blocks), "env=1 -> nfo True for every cond block")

# ---------------------------------------------------------------- capture real inputs
captured = {}


def _clone(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, (tuple, list)):
        return type(v)(_clone(e) for e in v)
    if isinstance(v, dict):
        return {k: _clone(e) for k, e in v.items()}
    return v


def make_hook(key):
    def hook(mod, a, kw):
        if key not in captured:
            captured[key] = (_clone(a), _clone(kw))

    return hook


handles = [b.register_forward_pre_hook(make_hook(f"image_cond.{j}"), with_kwargs=True)
           for j, b in enumerate(cond_blocks)]
handles.append(img_block1.register_forward_pre_hook(make_hook("image.1"), with_kwargs=True))
pngs = sorted(glob.glob(os.path.join(INPUTS_DIR, "*.png")))
with open(pngs[0], "rb") as f:
    ref_bytes = f.read()
with torch.inference_mode():
    ctx = pipe._encode_prompt("Make it look like a watercolor painting", None)
    image = _load_image_pixels(ref_bytes, device=pipe.device)
    with autotune_context(pipe.device, pipe.rank, tune=False):
        t1 = time.time()
        pipe._prepare_i2i(ctx=ctx, image_cond=[image], height=2048, width=2048,
                          num_inference_steps=16, seed=7)
        torch.cuda.synchronize()
        log(f"capture request {time.time() - t1:.1f}s")
for h in handles:
    h.remove()
log("captured", sorted(captured), "x shape", list(captured[cond_keys[0]][0][0].shape))
pristine = {k: _clone(v) for k, v in captured.items()}


def tensors(v):
    if isinstance(v, torch.Tensor):
        yield v
    elif isinstance(v, (tuple, list)):
        for e in v:
            yield from tensors(e)
    elif isinstance(v, dict):
        for e in v.values():
            yield from tensors(e)


def inputs_unchanged(key):
    return all(torch.equal(a, b) for a, b in zip(tensors(captured[key]), tensors(pristine[key])))


def restore(key):
    with torch.inference_mode():
        for a, b in zip(tensors(captured[key]), tensors(pristine[key])):
            a.copy_(b)


def call(b, key):
    a, kw = captured[key]
    return b(*a, **kw)


def time_block(b, key):
    with torch.inference_mode():
        for _ in range(WARMUP):
            call(b, key)
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITERS):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            call(b, key)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
    ts.sort()
    med = statistics.median(ts)
    p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
    return {"median_ms": round(med, 4), "noise": round((p90 - p10) / med, 4)}


def set_nfo(on):
    for b in cond_blocks:
        orig(b)._sm120_attn_nvfp4_out = bool(on)


def pass_times(label):
    res = {}
    for j, key in enumerate(cond_keys):
        restore(key)
        res[key] = time_block(cond_blocks[j], key)
        note(inputs_unchanged(key), f"{label} {key}: inputs unchanged after timing")
    res["sum_ms"] = round(sum(res[k]["median_ms"] for k in cond_keys), 4)
    log("PASS", label, json.dumps(res))
    return res


def kernel_names(b, key):
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        call(b, key)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call(b, key)
            torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return names


def spread(xs):
    return (max(xs) - min(xs)) / statistics.median(xs)


loaded_nfo = [route_of(b)["attn_nvfp4_out"] for b in cond_blocks]

if MODE == "bench":
    # Measured quantity: S = sum over the 5 cond blocks of the per-block median (50 timed calls).
    # Repeated timings of this configuration = S of each of 5 passes; step_ms = median of them,
    # noise_floor = (max - min) / median of them (the per-call p90-p10 spread is also reported).
    passes = [pass_times(f"loaded{i}") for i in range(5)]
    per_block = {k: statistics.median(p[k]["median_ms"] for p in passes) for k in cond_keys}
    sums = [p["sum_ms"] for p in passes]
    total = statistics.median(sums)
    noise = spread(sums)
    per_call_noise = max(max(p[k]["noise"] for p in passes) for k in cond_keys)
    ctrl = time_block(img_block1, "image.1")
    log("loadavg end", loadavg())
    print(json.dumps({
        "batch": {"i2i-2048": {"step_ms": round(total, 4), "noise_floor": round(noise, 4)}},
        "what": "median over 5 passes of the sum over the 5 image_cond blocks of per-block median "
                "ms (50 timed calls), real step-0 i2i inputs, graphs off; noise_floor = "
                "(max-min)/median of the 5 pass sums",
        "pass_sums_ms": sums, "per_call_p90_p10_max": round(per_call_noise, 4),
        "per_block_ms": per_block, "loaded_cond_nfo": loaded_nfo,
        "control_image1_ms": ctrl["median_ms"], "routes": routes,
        "output": {"class": "exact"},
    }))
    sys.exit(0)

# ---------------------------------------------------------------- gate: interleaved A/B + check
times = {"off": [], "on": []}
for rnd in range(3):
    for arm in ("off", "on"):
        set_nfo(arm == "on")
        times[arm].append(pass_times(f"{arm}{rnd}"))
set_nfo(any(loaded_nfo))
sums = {arm: [p["sum_ms"] for p in ps] for arm, ps in times.items()}
per_block = {arm: {k: statistics.median(p[k]["median_ms"] for p in ps) for k in cond_keys}
             for arm, ps in times.items()}
med_sum = {arm: round(statistics.median(s), 4) for arm, s in sums.items()}
aa = {arm: round(spread(s), 4) for arm, s in sums.items()}
gain = (med_sum["off"] - med_sum["on"]) / med_sum["off"]
gate_pass = (med_sum["off"] - med_sum["on"]) > max(
    max(sums["off"]) - min(sums["off"]), max(sums["on"]) - min(sums["on"]))
log("GATE", json.dumps({"sums": sums, "median_sum": med_sum, "aa_spread": aa,
                        "rel_gain": round(gain, 4), "gate_pass": gate_pass,
                        "per_block": per_block}))

# kernels per arm (one call of image_cond.0)
kn = {}
for arm in ("off", "on"):
    set_nfo(arm == "on")
    restore(cond_keys[0])
    names = kernel_names(cond_blocks[0], cond_keys[0])
    kn[arm] = {"n": len(names),
               "fmha": [n[:160] for n in names if "fmha" in n.lower() or "attention" in n.lower()
                        or "g3" in n.lower() or "nfo" in n.lower()],
               "quantize": [n[:160] for n in names if "quant" in n.lower()],
               "all": [n[:120] for n in names]}
    log("KERNELS", arm, json.dumps(kn[arm]))
set_nfo(any(loaded_nfo))

# correctness: pristine inputs, call 0 + 100 repeats per arm, ON vs OFF bitwise
check = {}
for j, key in enumerate(cond_keys):
    b = cond_blocks[j]
    firsts = {}
    for arm in ("off", "on"):
        set_nfo(arm == "on")
        restore(key)
        note(inputs_unchanged(key), f"{key} {arm}: pristine inputs before sequence")
        with torch.inference_mode():
            first = call(b, key).clone()
            same = 0
            for _ in range(REPEATS):
                same += int(torch.equal(call(b, key), first))
            torch.cuda.synchronize()
        note(same == REPEATS, f"{key} {arm}: {REPEATS} repeats bitwise == call 0 ({same})")
        note(inputs_unchanged(key), f"{key} {arm}: inputs unchanged after sequence")
        firsts[arm] = first
    eq = bool(torch.equal(firsts["off"], firsts["on"]))
    d = (firsts["off"].float() - firsts["on"].float()).abs().max().item()
    note(eq, f"{key}: nfo=1 output bitwise == nfo=0 output (max abs {d})")
    check[key] = {"on_eq_off_bitwise": eq, "max_abs": d,
                  "sha_off": hashlib.sha256(firsts["off"].contiguous().view(torch.uint8)
                                            .cpu().numpy().tobytes()).hexdigest()[:16]}
    log("CHECK", key, json.dumps(check[key]))
set_nfo(any(loaded_nfo))
log("loadavg end", loadavg())
print(json.dumps({
    "passed": not fails, "failures": fails, "gate_pass": gate_pass,
    "compared_against": "per image_cond block on real step-0 i2i inputs: nfo=1 (G3 NVFP4-out) "
                        "vs nfo=0 (main default) bitwise; 100 repeats vs call 0 per arm",
    "median_sum_ms": med_sum, "pass_sums_ms": sums, "aa_spread": aa, "rel_gain": round(gain, 4),
    "eligibility": elig, "routes": routes, "check": check,
    "kernels": {a: {"fmha": v["fmha"], "quantize": v["quantize"], "n": v["n"]} for a, v in kn.items()},
}))
