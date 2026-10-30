"""Branch 20 (idea 31) step-time pair arm: the served pipeline at its tree's X_POLICY default.

  pair.py <ckpt> [rounds]
Loads the pipeline as served (engine.json overlay, torch.compile + autotune, CUDA graphs, warmup
of 2048^2 t2i + i2i), runs one untimed request per case, then <rounds> rounds of t2i then i2i.
Per request: CUDA-event span of _run_denoise (16 steps) and sha256 of the final latent and uint8
image. Last stdout line: {"batch": {"t2i-2048": {"step_ms", "noise_floor"}, "i2i-2048": ...},
"output": {"class": "exact"}, "hashes": ...}; step_ms = median denoise span / 16, noise_floor =
(p90 - p10) / median of the per-request spans.
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
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:  # noqa
        return f"ERR {e}"


from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels  # noqa
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

INFO = {"log": sh("git log --oneline -4"), "x_policy": wk._X_POLICY,
        "env_x_policy": os.environ.get("TRTLLM_FLUX3_ADALN_WARP_X_POLICY"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
log("INFO", json.dumps(INFO))

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
REF_IMG = _load_image_pixels(_data, device=torch.device("cuda:0"))
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


def run(case):
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
    h = (hashlib.sha256(latent.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest(),
         hashlib.sha256(img.detach().contiguous().cpu().numpy().tobytes()).hexdigest())
    return s.elapsed_time(e), h


hashes = {"t2i": set(), "i2i": set()}
for case in ("t2i", "i2i"):
    ms, h = run(case)
    hashes[case].add(h)
    log("warm", case, round(ms, 2))
spans = {"t2i": [], "i2i": []}
for rd in range(ROUNDS):
    for case in ("t2i", "i2i"):
        ms, h = run(case)
        spans[case].append(ms)
        hashes[case].add(h)
    log("ROUND", rd, {c: round(spans[c][-1], 2) for c in spans})


def q(v, p):
    v = sorted(v)
    i = (len(v) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


batch = {}
for case, label in (("t2i", "t2i-2048"), ("i2i", "i2i-2048")):
    m = statistics.median(spans[case])
    batch[label] = {"step_ms": round(m / 16, 4), "denoise_ms_median": round(m, 3),
                    "noise_floor": round((q(spans[case], 0.9) - q(spans[case], 0.1)) / m, 5),
                    "n": len(spans[case]), "raw_denoise_ms": [round(x, 3) for x in spans[case]]}
print(json.dumps({"batch": batch, "output": {"class": "exact"}, "info": INFO,
                  "deterministic": {c: len(hashes[c]) == 1 for c in hashes},
                  "hashes": {c: sorted(list(x) for x in hashes[c]) for c in hashes}}))
