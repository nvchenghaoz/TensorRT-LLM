"""FLUX.3 image VAE decode microbenchmark / correctness check (no DiT load).

Usage: python vae_decode_bench.py <checkpoint_root> bench|check|profile
Env: LAYOUT_OPT=0 disables inductor layout_optimization; COMPILE=0 forces eager.
Builds the served VAE (package config vae.dtype, real VAE weights), decodes a
fixed-seed latent at the 1024x1024 served shape (packed latent 1x128x64x64).
"""

import json
import os
import sys
import time

import torch
from safetensors.torch import load_file

from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
    Flux3ImageVAE,
    Flux3ImageVAEConfig,
)

root, mode = sys.argv[1], sys.argv[2]
WARMUP, ITERS = 3, 50
if os.environ.get("LAYOUT_OPT") == "0":
    import torch._inductor.config as icfg

    icfg.layout_optimization = False

pkg = Flux3Checkpoint(root, variant="nvfp4")
vcfg = pkg.config["vae"]
dtype = getattr(torch, vcfg["dtype"])
print(f"vae dtype={dtype} sampler.size={pkg.config['sampler'].get('size')}", flush=True)
sd = load_file(str(pkg.asset(vcfg["path"])))


def build(compiled):
    vae = Flux3ImageVAE(Flux3ImageVAEConfig())
    vae.load_state_dict(sd)
    vae = vae.to(device="cuda", dtype=dtype).eval()
    is_compiled = False
    if compiled and hasattr(vae, "compile_decoder") and os.environ.get("COMPILE", "1") != "0":
        vae.compile_decoder()
        is_compiled = True
    return vae, is_compiled


torch.manual_seed(0)
z0 = torch.randn(1, 128, 64, 64, device="cuda", dtype=torch.float32)
z_pristine = z0.clone()


def decode(vae, z):
    with torch.inference_mode():
        return vae.decode(z.to(dtype))


def to_u8(x):
    x = (x / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    return (x * 255).round().to(torch.uint8)


vae, is_compiled = build(True)
cls = "reordered" if is_compiled else "exact"
print(f"compiled={is_compiled}", flush=True)

t0 = time.time()
for i in range(WARMUP):
    out = decode(vae, z0)
    torch.cuda.synchronize()
    print(f"warmup {i} done at {time.time() - t0:.1f}s", flush=True)
warm_s = time.time() - t0

if mode == "bench":
    times = []
    for i in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        decode(vae, z0)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
        if i % 10 == 0:
            print(f"iter {i} {times[-1]:.2f} ms", flush=True)
    ts = sorted(times)
    med = ts[len(ts) // 2]
    p10, p90 = ts[int(0.1 * len(ts))], ts[int(0.9 * len(ts)) - 1]
    print(
        json.dumps(
            {
                "batch": {"1": {"kernel_ms": med, "noise_floor": (p90 - p10) / med}},
                "output": {"class": cls},
                "compiled": is_compiled,
                "warmup_s": warm_s,
                "p10": p10,
                "p90": p90,
                "layout_opt": os.environ.get("LAYOUT_OPT", "1"),
            }
        )
    )
elif mode == "profile":
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    decode(vae, z0)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"profiled": True, "compiled": is_compiled}))
else:
    import torch._dynamo.utils as dyn_utils

    frames_before = dict(dyn_utils.counters["frames"])
    # Protected reference: the unmodified eager module on the same inputs.
    ref_vae, _ = build(False)
    ref = decode(ref_vae, z0).clone()
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    strict = decode(ref_vae, z0).clone()
    torch.backends.cudnn.allow_tf32 = tf32
    del ref_vae
    base_vs_fp32 = (ref.float() - strict.float()).abs().max().item()
    tol = max(2e-2, 2 * base_vs_fp32)
    print(f"declared tol={tol:.3e} (base vs strict-fp32 max abs {base_vs_fp32:.3e})", flush=True)

    # Untimed sequence matching the timed one: WARMUP already ran, then ITERS calls.
    first = None
    max_abs, bitwise_repeat, inputs_ok = 0.0, True, True
    for i in range(ITERS):
        if not torch.equal(z0, z_pristine):
            inputs_ok = False
        out = decode(vae, z0)
        if first is None:
            first = out.clone()
        elif not torch.equal(out, first):
            bitwise_repeat = False
        max_abs = max(max_abs, (out.float() - ref.float()).abs().max().item())
    d = (first.float() - ref.float()).abs()
    u8 = (to_u8(first).int() - to_u8(ref).int()).abs()
    frames_after = dict(dyn_utils.counters["frames"])
    passed = bool(max_abs <= tol and bitwise_repeat and inputs_ok)
    print(
        json.dumps(
            {
                "passed": passed,
                "compared": "compiled decoder vs unmodified eager decoder (same weights/input)",
                "compiled": is_compiled,
                "max_abs": max_abs,
                "mean_abs": d.mean().item(),
                "bitwise_equal_rate": (d == 0).float().mean().item(),
                "base_vs_strict_fp32_max_abs": base_vs_fp32,
                "tol": tol,
                "u8_max_diff": u8.max().item(),
                "u8_frac_diff": (u8 > 0).float().mean().item(),
                "repeat_bitwise": bitwise_repeat,
                "inputs_unchanged": inputs_ok,
                "frames_before": frames_before,
                "frames_after": frames_after,
            }
        )
    )
