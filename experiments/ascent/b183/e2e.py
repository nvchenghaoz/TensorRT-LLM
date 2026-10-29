"""Branch 183 one-process in-model census + A/B: served warp adaLN NVFP4 kernel launch bounds.

Arms (switched in-process by swapping flux3_sm120_adaln_nvfp4.launch's defaults; a test-only
CUDA-graph extra key gives each arm its own graphs; the custom op is opaque to Dynamo so both
arms share compiled code):
  old: warps=4, MINB=4  (main's build: 128 regs + 24 B spill)
  new: warps=1, MINB=12 (branch default: 167 regs, 0 B local)
Sequence (T=80 prompt, fixed seed): capture each arm once, then 8 rounds alternating order
(old,new / new,old), each request under torch.profiler (CUPTI, graph on): whole-request kernel
total, per-class rows (warp adaLN, Triton adaLN, everything else). Latents/images bitwise across
all requests (exact change). 0 Dynamo compiles / 0 graph captures in steady requests. Graph OFF:
one request per arm, latents vs graph-on; CPU-op twin counts for the attention side ops.
Usage: python e2e.py <checkpoint> [rounds]
"""

import collections
import json
import os
import statistics
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
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
        "anc_0761b30b": sh("git merge-base --is-ancestor 0761b30b HEAD; echo rc=$?"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sh("nvidia-smi --query-gpu=uuid,clocks.sm --format=csv,noheader")}
print("INFO", json.dumps(INFO), flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return {"frames": dict(c["frames"]), "unique_graphs": c["stats"].get("unique_graphs", 0)}


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

DEF = list(wk.launch.__defaults__)
names = wk.launch.__code__.co_varnames[: wk.launch.__code__.co_argcount]
dnames = names[len(names) - len(DEF):]
i_w, i_d = dnames.index("warps"), dnames.index("extra_defines")
ARMS = {"old": (4, ("MINB=4",)), "new": (1, ("MINB=12",))}
ARM = ["new"]


def set_arm(a):
    ARM[0] = a
    d = list(DEF)
    d[i_w], d[i_d] = ARMS[a]
    wk.launch.__defaults__ = tuple(d)


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
set_arm("new")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
c_warm = counters()
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s; dynamo {c_warm}; runner {runner is not None}; "
      f"compile_seconds {wk.COMPILE_SECONDS} regs {wk.NUM_REGS}", flush=True)
for _a, (_w, _d) in ARMS.items():  # build both NVRTC modules outside any capture/request
    wk._compile(torch.device("cuda", 0), _w, False, _d)
print("modules", {str(k): v for k, v in wk.NUM_REGS.items()}, flush=True)
if runner is not None:
    runner.clear()
    runner.register_extra_key_fn("b183_arm", lambda *a, **k: ARM[0])

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


def klass(n):
    if "flux3_adaln_nvfp4_warp" in n:
        return "warp_adaln"
    if "_adaln_nvfp4_kernel" in n:
        return "triton_adaln"
    return "other"


def run(arm, profile=True, cpu_ops=False):
    set_arm(arm)
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    n_mod = len(wk._MODULES)
    t1 = time.time()
    acts = [torch.profiler.ProfilerActivity.CUDA]
    if cpu_ops:
        acts.append(torch.profiler.ProfilerActivity.CPU)
    rec = {"arm": arm}
    if profile:
        with torch.profiler.profile(activities=acts) as p:
            T, lat, img = generate()
        rec["s"] = round(time.time() - t1, 2)
        cls = collections.defaultdict(lambda: [0, 0.0])
        total = 0.0
        ops = collections.Counter()
        for ev in p.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                if "Memcpy" in ev.name or "Memset" in ev.name:
                    continue
                d = ev.device_time_total
                total += d
                c = cls[klass(ev.name)]
                c[0] += 1
                c[1] += d
            elif cpu_ops and "side_gate_up" in ev.name:
                ops[ev.name] += 1
        rec["total_kernel_us"] = round(total, 1)
        rec["classes"] = {k: {"calls": v[0], "us": round(v[1], 1),
                              "us_per_call": round(v[1] / max(v[0], 1), 2)} for k, v in cls.items()}
        if cpu_ops:
            rec["side_gate_up_ops"] = dict(ops)
    else:
        T, lat, img = generate()
        rec["s"] = round(time.time() - t1, 2)
    rec["T"] = T
    rec["new_graphs"] = counters()["unique_graphs"] - prev
    rec["new_captures"] = (len(runner.graphs) if runner is not None else 0) - prev_caps
    rec["new_nvrtc_modules"] = len(wk._MODULES) - n_mod
    print("REQ", json.dumps(rec), flush=True)
    return rec, lat, img


def same(a, b):
    return torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


results = {"info": INFO, "warm_unique_graphs": c_warm["unique_graphs"]}
caps = {a: run(a, profile=False) for a in ("new", "old")}  # first capture per arm
ref = caps["new"]
note(same(caps["old"], ref), "capture requests: old vs new bitwise")
reqs = {"old": [], "new": []}
for r in range(ROUNDS):
    order = ("old", "new") if r % 2 == 0 else ("new", "old")
    for a in order:
        out = run(a)
        reqs[a].append(out[0])
        note(same(out, ref), f"round {r} {a}: latent+image bitwise vs first request")
        note(out[0]["new_graphs"] == 0 and out[0]["new_captures"] == 0
             and out[0]["new_nvrtc_modules"] == 0, f"round {r} {a}: 0 compiles/captures/NVRTC")


def summ(lst, key=lambda r: r["total_kernel_us"]):
    v = sorted(key(r) for r in lst)
    return {"median": statistics.median(v), "min": v[0], "max": v[-1],
            "spread_rel": (v[-1] - v[0]) / statistics.median(v), "n": len(v)}


census = {}
for a in ("old", "new"):
    census[a] = {"total_kernel_us": summ(reqs[a])}
    for c in ("warp_adaln", "triton_adaln", "other"):
        census[a][c] = {
            "us": summ(reqs[a], key=lambda r, c=c: r["classes"].get(c, {}).get("us", 0.0)),
            "calls": reqs[a][0]["classes"].get(c, {}).get("calls", 0),
            "us_per_call_median": statistics.median(
                r["classes"].get(c, {}).get("us_per_call", 0.0) for r in reqs[a]),
        }
paired = [n["total_kernel_us"] - o["total_kernel_us"] for o, n in zip(reqs["old"], reqs["new"])]
paired_warp = [n["classes"]["warp_adaln"]["us"] - o["classes"]["warp_adaln"]["us"]
               for o, n in zip(reqs["old"], reqs["new"])]
census["paired_new_minus_old_total_us"] = {"median": statistics.median(paired), "min": min(paired),
                                           "max": max(paired),
                                           "neg": sum(1 for d in paired if d < 0), "n": len(paired)}
census["paired_new_minus_old_warp_adaln_us"] = {"median": statistics.median(paired_warp),
                                                "min": min(paired_warp), "max": max(paired_warp)}
census["wall_s"] = {a: summ(reqs[a], key=lambda r: r["s"]) for a in reqs}
results["census"] = census
print("CENSUS", json.dumps(census), flush=True)

if runner is not None:
    runner.enabled = False
    goff = {a: run(a, profile=True, cpu_ops=True) for a in ("new", "old")}
    runner.enabled = True
    for a, out in goff.items():
        note(same(out, ref), f"graph off {a}: bitwise vs graph on")
    results["graph_off"] = {a: {"side_gate_up_ops": o[0].get("side_gate_up_ops"),
                                "warp_calls": o[0]["classes"].get("warp_adaln", {}).get("calls")}
                            for a, o in goff.items()}
    print("GRAPHOFF", json.dumps(results["graph_off"]), flush=True)

print(json.dumps({"passed": not fails, "failures": fails, **results}, default=str))
