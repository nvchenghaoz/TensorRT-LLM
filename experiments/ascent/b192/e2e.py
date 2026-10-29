"""Branch 192 (idea 273) one-process in-model A/B of the warp adaLN X_POLICY L2 hint.

Arms (switched in-process through flux3_sm120_adaln_nvfp4._X_POLICY, which launch() resolves at
call time; a test-only CUDA-graph extra key gives every arm its own graphs; the custom op is
opaque to Dynamo, so all arms share compiled code):
  A  : X_POLICY 0 (main's plain loads, main's module key/cubin)
  B  : X_POLICY 2 (evict_last fraction 0.5, the branch default)
  A2 : X_POLICY 0 again, separate graphs (A/A' twin)
Every arm is captured once in warmup. A per-graph-key capture record (wrapping _compile, which
launch() calls with its resolved defines) shows which defines each arm's captured graph used.

  e2e.py <ckpt> aba [rounds]    UNTRACED rotation (no profiler): rounds of A,B,A2 / A2,B,A;
                                per request the CUDA-event denoise span (_run_denoise) and the
                                wall time; NVML SM clock sampled every 20 ms. Decides.
  e2e.py <ckpt> census [rounds] CUPTI (torch.profiler, graph on) rotated W0/W2/W0' per round:
                                whole-request kernel total and per-class rows. Sizes only.
Both: latents+images bitwise across arms and repeats; 0 Dynamo compiles / 0 captures / 0 NVRTC
modules after warmup; graph-off request per arm (bitwise vs graph on) with CPU-op twin counts.
"""

import collections
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

import torch

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "aba"
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else (16 if MODE == "aba" else 10)
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


REL = "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_sm120_adaln_nvfp4.py"
INFO = {"head": sh("git rev-parse HEAD"), "log": sh("git log --oneline -3"),
        "anc_9e499b22": sh("git merge-base --is-ancestor 9e499b22 HEAD; echo rc=$?"),
        "anc_69eac46e": sh("git merge-base --is-ancestor 69eac46e HEAD; echo rc=$?"),
        "grep_minb12": sh(f"grep -c '_DEFAULT_DEFINES = (\"MINB=12\",)' {REL}"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": sh("hostname"),
        "gpu": sh("nvidia-smi --query-gpu=uuid,clocks.sm --format=csv,noheader")}
print("INFO", json.dumps(INFO), flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return {"frames": dict(c["frames"]), "unique_graphs": c["stats"].get("unique_graphs", 0)}


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

ARMS = {"A": 0, "B": 2, "A2": 0}
ARM = ["B"]
CAPREC = collections.defaultdict(collections.Counter)  # arm -> (capturing, defines) -> calls
_orig_compile = wk._compile


def _rec_compile(device, warps, use_pdl, extra_defines=()):
    CAPREC[ARM[0]][(bool(torch.cuda.is_current_stream_capturing()), tuple(extra_defines))] += 1
    return _orig_compile(device, warps, use_pdl, extra_defines)


wk._compile = _rec_compile


def set_arm(a):
    ARM[0] = a
    wk._X_POLICY = ARMS[a]


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
set_arm("B")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
c_warm = counters()
runner = getattr(pipe, "_cuda_graph_runners", {}).get("transformer")
print(f"loaded+warmup {time.time() - t0:.0f}s; dynamo {c_warm}; runner {runner is not None}; "
      f"compile_seconds {wk.COMPILE_SECONDS} regs {wk.NUM_REGS} xpol {wk._X_POLICY}", flush=True)
for _xp in (0, 2):  # build both NVRTC modules outside any capture/request
    _orig_compile(torch.device("cuda", 0), wk._DEFAULT_WARPS, False,
                  tuple(wk._DEFAULT_DEFINES) + ((f"X_POLICY={_xp}",) if _xp else ()))
print("modules", {str(k): v for k, v in wk.NUM_REGS.items()}, flush=True)
note(runner is not None, "transformer CUDA graph runner present")
if runner is not None:
    runner.clear()
    runner.register_extra_key_fn("b192_arm", lambda *a, **k: ARM[0])
CAPREC.clear()

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)

PROMPT, SEED = "A red fox standing in fresh snow at sunrise", 11
_DEN = {}
_orig_run_denoise = pipe._run_denoise


def _timed_run_denoise(*a, **k):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = _orig_run_denoise(*a, **k)
    e.record()
    _DEN["ev"] = (s, e)
    return out


pipe._run_denoise = _timed_run_denoise

# NVML SM clock sampler (20 ms)
CLK = {"on": False, "samples": []}
try:
    import pynvml

    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _sampler():
        while True:
            if CLK["on"]:
                try:
                    CLK["samples"].append(pynvml.nvmlDeviceGetClockInfo(_h, pynvml.NVML_CLOCK_SM))
                except Exception:  # noqa
                    pass
            time.sleep(0.02)

    threading.Thread(target=_sampler, daemon=True).start()
    HAVE_NVML = True
except Exception as _e:  # noqa
    print("no NVML sampler", _e, flush=True)
    HAVE_NVML = False


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
    if "add_mul_slice" in n:
        return "add_mul_slice"
    if "add_mul_view" in n:
        return "add_mul_view"
    if "GemmUniversal" in n:
        return "GemmUniversal"
    if re.search(r"gemm|Gemm|GEMM|nvjet|cutlass|cublas|xmma|_mma_|matmul", n):
        return "other_gemm"
    if "fmha" in n.lower():
        return "fmha"
    return "other"


def run(arm, profile=False, cpu_ops=False):
    set_arm(arm)
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs) if runner is not None else 0
    n_mod = len(wk._MODULES)
    rec = {"arm": arm}
    CLK["samples"] = []
    CLK["on"] = True
    t1 = time.perf_counter()
    if profile:
        acts = [torch.profiler.ProfilerActivity.CUDA]
        if cpu_ops:
            acts.append(torch.profiler.ProfilerActivity.CPU)
        with torch.profiler.profile(activities=acts) as p:
            T, lat, img = generate()
        rec["wall_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        cls = collections.defaultdict(lambda: [0, 0.0])
        total = 0.0
        ops = collections.Counter()
        warp_grids = collections.Counter()
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
        rec["wall_ms"] = round((time.perf_counter() - t1) * 1000, 2)
    CLK["on"] = False
    s, e = _DEN["ev"]
    rec["denoise_ms"] = round(s.elapsed_time(e), 3)
    if CLK["samples"]:
        rec["sm_clk_mean"] = round(statistics.mean(CLK["samples"]), 1)
        rec["sm_clk_min"] = min(CLK["samples"])
    rec["T"] = T
    rec["new_graphs"] = counters()["unique_graphs"] - prev
    rec["new_captures"] = (len(runner.graphs) if runner is not None else 0) - prev_caps
    rec["new_nvrtc_modules"] = len(wk._MODULES) - n_mod
    print("REQ", json.dumps(rec), flush=True)
    return rec, lat, img


def same(a, b):
    return torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


results = {"info": INFO, "mode": MODE, "rounds": ROUNDS, "warm_unique_graphs": c_warm["unique_graphs"]}
# warmup: capture every arm once (the first request per arm captures its graph)
caps = {}
for a in ("A", "B", "A2"):
    caps[a] = run(a)
ref = caps["A"]
for a in ("B", "A2"):
    note(same(caps[a], ref), f"capture request {a} vs A bitwise")
results["capture_record"] = {a: {f"capturing={k[0]} defines={list(k[1])}": n for k, n in c.items()}
                             for a, c in CAPREC.items()}
print("CAPREC", json.dumps(results["capture_record"]), flush=True)
cap_b = {k: n for k, n in CAPREC["B"].items() if k[0]}
cap_a = {k: n for k, n in CAPREC["A"].items() if k[0]}
want_b = tuple(wk._DEFAULT_DEFINES) + ("X_POLICY=2",)
want_a = tuple(wk._DEFAULT_DEFINES)
note(set(k[1] for k in cap_b) == {want_b}, "B graph captured only X_POLICY=2 launches")
note(set(k[1] for k in cap_a) == {want_a}, "A graph captured only main-define launches")
results["captured_warp_calls_per_graph"] = {"A": sum(cap_a.values()), "B": sum(cap_b.values())}
# one warm request per arm (untimed)
for a in ("A", "B", "A2"):
    out = run(a)
    note(same(out, ref), f"warm {a} bitwise")
    note(out[0]["new_captures"] == 0, f"warm {a}: 0 captures")

reqs = {a: [] for a in ARMS}
for r in range(ROUNDS):
    order = ("A", "B", "A2") if r % 2 == 0 else ("A2", "B", "A")
    for a in order:
        out = run(a, profile=(MODE == "census"))
        reqs[a].append(out[0])
        note(same(out, ref), f"round {r} {a}: latent+image bitwise vs first A request")
        note(out[0]["new_graphs"] == 0 and out[0]["new_captures"] == 0
             and out[0]["new_nvrtc_modules"] == 0, f"round {r} {a}: 0 compiles/captures/NVRTC")
    print("ROUND", r, {a: reqs[a][-1]["denoise_ms"] for a in ARMS}, flush=True)


def summ(v):
    v = sorted(v)
    return {"median": statistics.median(v), "mean": statistics.mean(v), "min": v[0], "max": v[-1],
            "n": len(v)}


def decide(key):
    a = [r[key] for r in reqs["A"]]
    a2 = [r[key] for r in reqs["A2"]]
    b = [r[key] for r in reqs["B"]]
    ma, ma2, mb = statistics.median(a), statistics.median(a2), statistics.median(b)
    aa = abs(ma - ma2)
    per = [bb - (x + y) / 2 for bb, x, y in zip(b, a, a2)]
    neg = sum(1 for d in per if d < 0)
    go = (mb < ma - aa) and (mb < ma2 - aa) and neg >= int(0.75 * len(per) + 0.999)
    return {"A": summ(a), "A2": summ(a2), "B": summ(b), "abs_A_minus_A2_median": aa,
            "B_minus_meanA_per_round": {"median": statistics.median(per), "mean": statistics.mean(per),
                                        "min": min(per), "max": max(per), "neg": neg, "n": len(per)},
            "B_minus_A_median": mb - ma, "B_minus_A2_median": mb - ma2, "GO": go}


summary = {"denoise_ms": decide("denoise_ms"), "wall_ms": decide("wall_ms")}
if MODE == "census":
    summary["total_kernel_us"] = decide("total_kernel_us")
    rows = {}
    classes = sorted({c for a in ARMS for r in reqs[a] for c in r["classes"]})
    for c in classes:
        rows[c] = {}
        for a in ARMS:
            us = [r["classes"].get(c, {}).get("us", 0.0) for r in reqs[a]]
            rows[c][a] = {"us_median": statistics.median(us),
                          "calls": reqs[a][0]["classes"].get(c, {}).get("calls", 0),
                          "us_per_call_median": statistics.median(
                              r["classes"].get(c, {}).get("us_per_call", 0.0) for r in reqs[a])}
        pb = [rb["classes"].get(c, {}).get("us", 0.0) - (ra["classes"].get(c, {}).get("us", 0.0)
              + ra2["classes"].get(c, {}).get("us", 0.0)) / 2
              for ra, rb, ra2 in zip(reqs["A"], reqs["B"], reqs["A2"])]
        paa = [ra2["classes"].get(c, {}).get("us", 0.0) - ra["classes"].get(c, {}).get("us", 0.0)
               for ra, ra2 in zip(reqs["A"], reqs["A2"])]
        rows[c]["W2_minus_meanW0_us_median"] = statistics.median(pb)
        rows[c]["W0p_minus_W0_us_median"] = statistics.median(paa)
    summary["class_rows"] = rows
    note(rows.get("warp_adaln", {}).get("B", {}).get("calls") == 448, "448 warp adaLN calls in B")
if HAVE_NVML:
    summary["sm_clk"] = {a: summ([r.get("sm_clk_mean", 0) for r in reqs[a]]) for a in ARMS}
results["summary"] = summary
print("SUMMARY", json.dumps(summary), flush=True)

# graph off: one request per arm, bitwise vs graph on, CPU-op twin counts
if runner is not None:
    runner.enabled = False
    goff = {a: run(a, profile=True, cpu_ops=True) for a in ("A", "B")}
    runner.enabled = True
    for a, out in goff.items():
        note(same(out, ref), f"graph off {a}: bitwise vs graph on")
    tw = {}
    for a, o in goff.items():
        ops = o[0].get("side_gate_up_ops") or {}
        tw[a] = {
            "nvfp4_out_side_gate_up": sum(n for k, n in ops.items() if "nvfp4_out_side_gate_up" in k),
            "prequant_side_gate_up": sum(n for k, n in ops.items()
                                         if "prequant_attention_side_gate_up" in k),
            "fp8_side_gate_up": sum(n for k, n in ops.items()
                                    if "fp8_attention_side_gate_up" in k and "prequant" not in k),
            "warp_calls": o[0]["classes"].get("warp_adaln", {}).get("calls"),
            "raw": ops,
        }
    results["graph_off_twins"] = tw
    print("GRAPHOFF", json.dumps(tw), flush=True)
    for a in tw:
        note(tw[a]["nvfp4_out_side_gate_up"] == 512 and tw[a]["prequant_side_gate_up"] == 0
             and tw[a]["fp8_side_gate_up"] == 0, f"twin counts 512/0/0 arm {a}")

print(json.dumps({"passed": not fails, "failures": fails, **results}, default=str))
