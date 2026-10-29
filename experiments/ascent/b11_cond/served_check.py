"""Branch 11: in-process served-pipeline check (CUDA graphs ON, torch.compile + autotune, the
engine.json overlay, warmup [[2048, 2048]] t2i + i2i, as the benchmark serves it).

Usage: served_check.py <checkpoint> [check|profile] [cond_off]
  cond_off: sets TRTLLM_FLUX3_COND_PRIMSTS=0 before import (kill switch; must equal main).
check: sequence T_A, I_B, T_A, I_B, I_C, T_A, I_B (t2i = text-to-image 2048^2 16 steps; i2i = the
  frozen 2048^2 reference PNG + an edit prompt) in ONE process; prints per-request sha256 of the
  final latent and the uint8 image and the denoise wall time, and checks every repeat is bitwise
  equal to its first run (state across t2i <-> i2i graphs, shapes and caches).
  Cross-tree comparison (t2i bitwise vs main; i2i vs main with cond_off) uses the printed hashes.
profile: warmup + one i2i request, then one i2i request inside cudaProfilerStart/Stop.
"""

import glob
import hashlib
import json
import os
import sys
import time

if "cond_off" in sys.argv[3:]:
    os.environ["TRTLLM_FLUX3_COND_PRIMSTS"] = "0"

import torch  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    _load_image_pixels,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "check"
INPUTS_DIR = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/"
    "flux3-klein-nvfp4-2048/v1/inputs"
)
fails = []


def log(*a):
    print(f"[b11s {time.strftime('%H:%M:%S')}]", *a, flush=True)


here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
engine.pop("env", None)
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    parallel_config={"tp_size": 1},
    torch_compile_config={"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
log("env", {k: v for k, v in os.environ.items() if k.startswith("TRTLLM_FLUX3_COND")})
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s graphs={runner is not None and runner.enabled}")
tr = pipe.transformer
cond_route = []
for j, b in enumerate(tr.content_mode_blocks["image_cond"]):
    b = getattr(b, "_orig_mod", b)
    cond_route.append({
        "block": f"image_cond.{j}",
        "prepared_qkv": getattr(b.attn, "_flux3_primsts_nvfp4_qkv", None) is not None,
        "adaln_nvfp4": bool(getattr(b, "_flux3_adaln_nvfp4", False)),
        "tail_pair": bool(getattr(b, "_sm120_tail_pair", False)),
        "side_gate_up": int(getattr(b, "_sm120_side_gate_up", 0)),
        "attn_nvfp4_out": bool(getattr(b, "_sm120_attn_nvfp4_out", False)),
        "static_qk": bool(getattr(b.attn, "_flux3_static_qk", True)),
    })
log("cond route", json.dumps(cond_route))

if "fixedref" in sys.argv[3:]:
    # Cross-process bitwise comparisons: the VAE (cuDNN frontend plans) is not bitwise stable
    # across processes, so replace the encoded reference latent by a fixed seeded tensor of the
    # same shape/dtype (deterministic on every process); compare denoise latents only.
    _orig_encode = pipe.vae.encode

    def _fixed_encode(image, *a, **k):
        out = _orig_encode(image, *a, **k)  # [B, C, h, w]
        g = torch.Generator(device=out.device).manual_seed(1234)
        fixed = torch.randn(out.shape, generator=g, device=out.device, dtype=torch.float32)
        return fixed.to(out.dtype)

    pipe.vae.encode = _fixed_encode
    log("fixedref: VAE-encoded reference latents replaced by a seeded tensor")

pngs = sorted(glob.glob(os.path.join(INPUTS_DIR, "*.png")))
REFS = {k: open(p, "rb").read() for k, p in zip(("B", "C"), pngs[:2])}
PROMPTS = {
    "A": ("A red fox standing in fresh snow at sunrise", 100),
    "B": ("Make it look like a watercolor painting", 7),
    "C": ("Add a small red balloon in the sky", 8),
}


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def generate(name):
    prompt, seed = PROMPTS[name]
    t1 = time.time()
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
        with autotune_context(pipe.device, pipe.rank, tune=False):
            if name == "A":
                latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                                num_inference_steps=16, seed=seed)
            else:
                image = _load_image_pixels(REFS[name], device=pipe.device)
                latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[image], height=2048,
                                                width=2048, num_inference_steps=16, seed=seed)
        torch.cuda.synchronize()
        den = time.time() - t1
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return {"req": name, "mode": "t2i" if name == "A" else "i2i", "latent_sha": sha(latent),
            "image_sha": sha(img), "denoise_wall_s": round(den, 3),
            "total_s": round(time.time() - t1, 3)}


if MODE == "profile":
    generate("B")
    generate("B")
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    generate("B")
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"passed": True, "profiled": "one graph-on i2i request", "cond_route": cond_route}))
    sys.exit(0)

order = ["A", "B", "A", "B", "C", "A", "B"]
recs = []
for name in order:
    r = generate(name)
    recs.append(r)
    log("REQ", json.dumps(r))
first = {}
for r in recs:
    if r["req"] in first:
        f = first[r["req"]]
        ok = r["latent_sha"] == f["latent_sha"] and r["image_sha"] == f["image_sha"]
        if not ok:
            fails.append(f"repeat of {r['req']} not bitwise equal to its first run")
    else:
        first[r["req"]] = r
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "repeated_execution_passed": not fails,
    "sequence": "t2i A -> i2i B -> t2i A -> i2i B -> i2i C -> t2i A -> i2i B, one process, graphs on",
    "first_hashes": {k: {"latent": v["latent_sha"][:16], "image": v["image_sha"][:16]} for k, v in first.items()},
    "denoise_wall_s": {k: [r["denoise_wall_s"] for r in recs if r["req"] == k] for k in first},
    "cond_route": cond_route,
    "cond_env": os.environ.get("TRTLLM_FLUX3_COND_PRIMSTS"),
}))
