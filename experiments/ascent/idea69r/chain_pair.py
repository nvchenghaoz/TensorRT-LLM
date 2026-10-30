"""Branch 51 kernel-time pair on the served chain each warp adaLN+NVFP4 head sits in.

Tree-agnostic (runs on main and on the branch). Loads the served pipeline (engine.json, compile,
autotune, CUDA graphs, warmup), heat-soaks, then profiles REQS requests per case (alternating
t2i-2048 / i2i-2048) with torch.profiler (CUDA activity, kernels inside graphs). Per request, from
the time-ordered kernel trace:
  chain_ms = sum over every flux3_adaln_nvfp4_warp launch of
             (the kernel right before it: the x_new writer, fused pair+residual)
           + (the head itself)
           + (the kernel right after it: the QKV GEMM that reads the head's fp4/SF).
kernel_ms per case = median chain_ms over requests; noise_floor = (p90 - p10) / median.
Also reports the per-part split (writer / head / consumer per head grid) and the CUDA-event
denoise span, and sha256 of latent+image per request (repeat determinism in-process).

  chain_pair.py <ckpt> <reqs_per_case> <soak_rounds>
"""

import collections
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
REQS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SOAK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
HEAD = "flux3_adaln_nvfp4_warp"


def log(*a):
    print(*a, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa
        return f"ERR {e}"


log("GIT", sh("git log --oneline -3"))
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels  # noqa
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

log("ROW_REV", getattr(wk, "_ROW_REV", "absent"), "X_POLICY", wk._X_POLICY)
DEV = torch.device("cuda", 0)
engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine.setdefault("torch_compile_config",
                  {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
engine.setdefault("parallel_config", {"tp_size": 1})
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
log(f"loaded+warmup {time.time() - t0:.0f}s")


def store_png(sha):
    store = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
             "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
    for root, _, files in os.walk(store):
        for f in files:
            if sha in f:
                return os.path.join(root, f)
    for root, _, files in os.walk(store):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == sha:
                    return p
    raise FileNotFoundError(sha)


SHA00 = "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"
_data = open(store_png(SHA00), "rb").read()
assert hashlib.sha256(_data).hexdigest() == SHA00
REF_IMG = _load_image_pixels(_data, device=DEV)
CASES = {"t2i": ("A red fox standing in fresh snow at sunrise, photograph", 11),
         "i2i": ("Turn the scene into a watercolor painting", 5)}
PROF = {"on": False, "p": None}
_DEN = {}
_orig = pipe._run_denoise


def _timed(*a, **k):
    if PROF["on"]:
        torch.cuda.synchronize()
        p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        p.__enter__()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = _orig(*a, **k)
    e.record()
    if PROF["on"]:
        torch.cuda.synchronize()
        p.__exit__(None, None, None)
        PROF["p"] = p
    _DEN["ev"] = (s, e)
    return out


pipe._run_denoise = _timed


def generate(case, profile=False):
    prompt, seed = CASES[case]
    PROF["on"] = profile
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if case == "t2i":
            latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                            num_inference_steps=16, seed=seed)
        else:
            latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[REF_IMG], height=2048,
                                            width=2048, num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    PROF["on"] = False
    s, e = _DEN["ev"]
    h = hashlib.sha256()
    h.update(latent.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    h.update(img.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return s.elapsed_time(e), h.hexdigest()


def chain(prof):
    path = os.path.join(ART, f"_tmp_chain_{os.getpid()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    evs = sorted((e for e in tr.get("traceEvents", []) if e.get("ph") == "X"
                  and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    parts = collections.defaultdict(float)
    calls = collections.Counter()
    total = 0.0
    for i, e in enumerate(evs):
        if HEAD not in e["name"]:
            continue
        g = "x".join(str(v) for v in e.get("args", {}).get("grid", []))
        calls[g] += 1
        d = float(e["dur"])
        parts[f"head@{g}"] += d
        total += d
        if i > 0:
            parts[f"writer@{g}"] += float(evs[i - 1]["dur"])
            total += float(evs[i - 1]["dur"])
        if i + 1 < len(evs):
            parts[f"consumer@{g}"] += float(evs[i + 1]["dur"])
            total += float(evs[i + 1]["dur"])
    return total / 1000.0, {k: v / 1000.0 for k, v in parts.items()}, dict(calls)


for _ in range(max(SOAK, 1)):  # untimed: graph capture, warm, heat soak
    for case in ("t2i", "i2i"):
        ms, hx = generate(case)
        log("WARM", case, round(ms, 2), hx[:16])
res = {c: {"chain": [], "den": [], "parts": [], "hash": set(), "calls": None}
       for c in ("t2i", "i2i")}
for r in range(REQS):
    for case in (("t2i", "i2i") if r % 2 == 0 else ("i2i", "t2i")):
        ms, hx = generate(case, profile=True)
        tot, parts, calls = chain(PROF["p"])
        PROF["p"] = None
        d = res[case]
        d["chain"].append(tot)
        d["den"].append(ms)
        d["parts"].append(parts)
        d["hash"].add(hx)
        d["calls"] = calls
        log("REQ", case, r, json.dumps({"chain_ms": round(tot, 3), "denoise_ms": round(ms, 2),
                                        "calls": calls,
                                        "parts": {k: round(v, 3) for k, v in parts.items()}}))


def q(v, p):
    v = sorted(v)
    i = (len(v) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


out = {}
for case, label in (("t2i", "t2i-2048"), ("i2i", "i2i-2048")):
    d = res[case]
    m = statistics.median(d["chain"])
    keys = sorted({k for p in d["parts"] for k in p})
    out[label] = {"kernel_ms": round(m, 4),
                  "noise_floor": round((q(d["chain"], .9) - q(d["chain"], .1)) / m, 5),
                  "n": len(d["chain"]),
                  "parts_ms": {k: round(statistics.median([p.get(k, 0.0) for p in d["parts"]]), 3)
                               for k in keys},
                  "head_calls_per_grid": d["calls"],
                  "denoise_ms_profiled_median": round(statistics.median(d["den"]), 3),
                  "sha256": sorted(d["hash"])}
print(json.dumps({"batch": out, "output": {"class": "exact"},
                  "deterministic": all(len(res[c]["hash"]) == 1 for c in res),
                  "row_rev": getattr(wk, "_ROW_REV", "absent"),
                  "chain": "writer(kernel before) + head + consumer(kernel after) per "
                           "flux3_adaln_nvfp4_warp launch, all head forms"}))
