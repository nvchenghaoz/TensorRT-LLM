"""Branch 51 step-time pair: served pipeline (engine.json, compile, autotune, CUDA graphs, warmup),
then ROUNDS alternating t2i-2048 / i2i-2048 requests; per request CUDA-event 16-step denoise span.
Tree-agnostic (runs on main and on the branch). Prints step_ms = median denoise / 16 per case and
noise_floor = (p90 - p10) / median, plus sha256 of every request's latent+image (exact check across
arms and repeats).

  step_pair.py <ckpt> <rounds>
"""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SOAK = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # untimed t2i+i2i rounds (clock heat soak)
HERE = os.path.dirname(os.path.abspath(__file__))


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
_DEN = {}
_orig = pipe._run_denoise


def _timed(*a, **k):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = _orig(*a, **k)
    e.record()
    _DEN["ev"] = (s, e)
    return out


pipe._run_denoise = _timed


def generate(case):
    prompt, seed = CASES[case]
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
    s, e = _DEN["ev"]
    h = hashlib.sha256()
    h.update(latent.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    h.update(img.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return s.elapsed_time(e), h.hexdigest()


for case in ("t2i", "i2i") * max(SOAK, 2):  # untimed: graph capture + warm + heat soak
    ms, hx = generate(case)
    log("WARM", case, round(ms, 2), hx[:16])
den = {"t2i": [], "i2i": []}
hashes = {"t2i": set(), "i2i": set()}
for rd in range(ROUNDS):
    for case in (("t2i", "i2i") if rd % 2 == 0 else ("i2i", "t2i")):
        ms, hx = generate(case)
        den[case].append(ms)
        hashes[case].add(hx)
    log("ROUND", rd, {c: round(v[-1], 2) for c, v in den.items()})


def q(v, p):
    v = sorted(v)
    i = (len(v) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


out = {}
for case, label in (("t2i", "t2i-2048"), ("i2i", "i2i-2048")):
    m = statistics.median(den[case])
    out[label] = {"step_ms": round(m / 16, 4), "denoise_ms": round(m, 3),
                  "noise_floor": round((q(den[case], .9) - q(den[case], .1)) / m, 5),
                  "n": len(den[case]), "sha256": sorted(hashes[case])}
print(json.dumps({"batch": out, "output": {"class": "exact"},
                  "deterministic": all(len(h) == 1 for h in hashes.values()),
                  "row_rev": getattr(wk, "_ROW_REV", "absent")}))
