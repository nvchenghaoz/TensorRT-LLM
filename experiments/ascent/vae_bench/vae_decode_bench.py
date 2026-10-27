"""Shared FLUX.3 image VAE decode microbenchmark (vae_decode area, lead ruling).

The ruling names experiments/vae_bench/vae_decode_bench.py; that path is outside
worker edit scope, so the shared copy lives here (experiments/ascent/vae_bench/).

Builds the FLUX.3 image VAE (Flux3ImageVAEConfig defaults) with RANDOM weights
(torch.manual_seed(0), module default init, no checkpoint weight load) at the
served dtype (checkpoint package config vae.dtype; only config.json is read),
and decodes a fixed-seed 1x128x64x64 packed latent (the 1024x1024 output).

Usage: python vae_decode_bench.py <checkpoint_root> <mode>
  time   one arm, for run_gpu_pair: this tree's decoder (compiled when the tree
         provides Flux3ImageVAE.compile_decoder and VAE_COMPILE!=0). CUDA events,
         median of ITERS after WARMUP. Last line: pair evidence JSON.
  ab     in-process interleaved A/B/(C)/A': A = this tree's eager decoder, B = this
         tree's decoder (compiled if supported), C = compiled with Flux3GroupNorm's
         Triton kernels kept as extern calls (when supported), A' = a second eager
         instance (A/A noise). Same weights, same input, arms rotated every iteration.
  check  untimed repeated-execution pass with the same WARMUP/ITERS as timing:
         every iteration is compared with iteration 0 (bitwise) and with the
         eager reference (declared tolerance), and the input is checked unchanged.
  profile one warm decode between cudaProfilerStart/Stop (for nsys/ncu region capture).
  kernels launch-count census of one warm decode (torch.profiler), incl.
         RowwiseMomentsCUDAKernel and cuDNN nchwToNhwcKernel.
Env: VAE_COMPILE=0 forces eager; WARMUP, ITERS override counts.
"""

import json
import os
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
    Flux3ImageVAE,
    Flux3ImageVAEConfig,
)

root, mode = sys.argv[1], sys.argv[2]
WARMUP = int(os.environ.get("WARMUP", "3"))
ITERS = int(os.environ.get("ITERS", "50"))

pkg = Flux3Checkpoint(root, variant="nvfp4")
dtype = getattr(torch, pkg.config["vae"]["dtype"])
print(f"vae dtype={dtype} sampler.size={pkg.config['sampler'].get('size')}", flush=True)

torch.manual_seed(0)
_proto = Flux3ImageVAE(Flux3ImageVAEConfig())
STATE = {k: v.clone() for k, v in _proto.state_dict().items()}
del _proto


def build(compiled: bool, **compile_kwargs):
    vae = Flux3ImageVAE(Flux3ImageVAEConfig())
    vae.load_state_dict(STATE)
    vae = vae.to(device="cuda", dtype=dtype).eval()
    is_compiled = False
    if compiled and hasattr(vae, "compile_decoder") and os.environ.get("VAE_COMPILE", "1") != "0":
        vae.compile_decoder(**compile_kwargs)
        is_compiled = True
    return vae, is_compiled


def _supports_triton_gn_kwarg():
    import inspect

    fn = getattr(Flux3ImageVAE, "compile_decoder", None)
    return fn is not None and "triton_group_norm" in inspect.signature(fn).parameters


gen = torch.Generator(device="cuda").manual_seed(1234)
z0 = torch.randn(1, 128, 64, 64, device="cuda", dtype=torch.float32, generator=gen)
z_pristine = z0.clone()


def decode(vae, z):
    with torch.inference_mode():
        return vae.decode(z.to(dtype))


def to_u8(x):
    x = (x.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    return (x * 255).round().to(torch.uint8)


def timed(vae):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    decode(vae, z0)
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med}


def warm(vae, tag):
    t0 = time.time()
    for i in range(WARMUP):
        decode(vae, z0)
        torch.cuda.synchronize()
        print(f"[{tag}] warmup {i} done at {time.time() - t0:.1f}s", flush=True)
    return time.time() - t0


if mode == "time":
    vae, is_compiled = build(True)
    print(f"compiled={is_compiled}", flush=True)
    warm_s = warm(vae, "B" if is_compiled else "A")
    ts = []
    for i in range(ITERS):
        ts.append(timed(vae))
        if i % 10 == 0:
            print(f"iter {i} {ts[-1]:.2f} ms", flush=True)
    st = stats(ts)
    print(json.dumps({
        "batch": {"1": {"kernel_ms": st["median_ms"], "noise_floor": st["noise_floor"]}},
        "output": {"class": "reordered" if is_compiled else "exact"},
        "compiled": is_compiled, "warmup_s": warm_s, **st,
        "bench": "experiments/ascent/vae_bench/vae_decode_bench.py time (random weights, 1x128x64x64)",
    }))

elif mode == "ab":
    arms = {}
    arms["A"], _ = build(False)
    arms["B"], b_compiled = build(True)
    if _supports_triton_gn_kwarg():
        # C = compiled decoder keeping Flux3GroupNorm's Triton kernels as extern calls.
        arms["C"], _ = build(True, triton_group_norm=True)
    arms["A2"], _ = build(False)
    warm_s = {k: warm(v, k) for k, v in arms.items()}
    order = list(arms)
    ts = {k: [] for k in order}
    for i in range(ITERS):
        r = i % len(order)
        rot = order[r:] + order[:r]
        for k in rot:
            ts[k].append(timed(arms[k]))
        if i % 10 == 0:
            print(f"iter {i} " + " ".join(f"{k}={ts[k][-1]:.2f}" for k in order), flush=True)
    st = {k: stats(v) for k, v in ts.items()}
    a, b, a2 = st["A"]["median_ms"], st["B"]["median_ms"], st["A2"]["median_ms"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": b, "noise_floor": st["B"]["noise_floor"]}},
        "output": {"class": "reordered" if b_compiled else "exact"},
        "interleaved": True, "b_compiled": b_compiled, "arms": st,
        "speedup_A_over_B": a / b, "saving_ms": a - b, "aa_rel_diff": abs(a2 - a) / a,
        "C_triton_gn_compiled_ms": st["C"]["median_ms"] if "C" in st else None,
        "compile_warmup_s": warm_s["B"], "eager_warmup_s": warm_s["A"],
    }))

elif mode == "profile":
    vae, is_compiled = build(True)
    warm(vae, "P")
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    decode(vae, z0)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"profiled": True, "compiled": is_compiled}))

elif mode == "kernels":
    # Kernel census of one warm decode (torch.profiler / CUPTI): launch counts per kernel name.
    from collections import Counter

    from torch.profiler import ProfilerActivity, profile

    vae, is_compiled = build(True)
    warm(vae, "K")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        decode(vae, z0)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total
    def match(s):
        keys = [k for k in cnt if s in k]
        return {"launches": sum(cnt[k] for k in keys), "total_ms": sum(dur[k] for k in keys) / 1e3}
    print(json.dumps({
        "compiled": is_compiled,
        "total_launches": sum(cnt.values()),
        "RowwiseMoments": match("RowwiseMoments"),
        "nchwToNhwc": match("nchwToNhwc"),
        "group_norm_aten": match("group_norm"),
        "triton": match("triton_"),
        "top": [(k[:120], cnt[k], round(dur[k] / 1e3, 3)) for k, _ in dur.most_common(12)],
    }))

elif mode == "check":
    import torch._dynamo.utils as dyn_utils

    ref_vae, _ = build(False)
    ref = decode(ref_vae, z0).clone()
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    strict = decode(ref_vae, z0).clone()
    torch.backends.cudnn.allow_tf32 = tf32
    ref_repeat = bool(torch.equal(decode(ref_vae, z0), ref))
    del ref_vae
    base_vs_fp32 = (ref.float() - strict.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    # Declared before the comparison: compiled-vs-eager error must stay within 2x the
    # eager decoder's own error against strict-fp32 convs (TF32 accumulation order),
    # floored at 1e-4 of the output range.
    tol = max(2.0 * base_vs_fp32, 1e-4 * scale)
    print(f"declared tol={tol:.3e} (eager vs strict-fp32 max abs {base_vs_fp32:.3e}, |ref|max {scale:.3f})",
          flush=True)

    vae, is_compiled = build(True)
    warm(vae, "C")
    frames_before = dict(dyn_utils.counters["frames"])
    first = None
    max_abs, repeat_bitwise, inputs_ok = 0.0, True, True
    for i in range(ITERS):
        if not torch.equal(z0, z_pristine):
            inputs_ok = False
        out = decode(vae, z0)
        if first is None:
            first = out.clone()
        elif not torch.equal(out, first):
            repeat_bitwise = False
        max_abs = max(max_abs, (out.float() - ref.float()).abs().max().item())
    frames_after = dict(dyn_utils.counters["frames"])
    d = (first.float() - ref.float()).abs()
    u8 = (to_u8(first).int() - to_u8(ref).int()).abs()
    no_recompile = frames_after == frames_before
    passed = bool(max_abs <= tol and repeat_bitwise and inputs_ok and no_recompile and ref_repeat)
    print(json.dumps({
        "passed": passed,
        "compared": "this tree's decoder vs unmodified eager decoder (same random weights, same input); "
                    "every iteration vs iteration 0 bitwise",
        "compiled": is_compiled, "iters": ITERS, "warmup": WARMUP,
        "max_abs": max_abs, "mean_abs": d.mean().item(),
        "bitwise_equal_rate": (d == 0).float().mean().item(),
        "eager_vs_strict_fp32_max_abs": base_vs_fp32, "tol": tol,
        "u8_max_diff": u8.max().item(), "u8_frac_diff": (u8 > 0).float().mean().item(),
        "repeat_bitwise": repeat_bitwise, "eager_ref_repeat_bitwise": ref_repeat,
        "inputs_unchanged": inputs_ok,
        "no_recompile_after_warmup": no_recompile,
        "frames_before": frames_before, "frames_after": frames_after,
    }))
else:
    raise SystemExit(f"unknown mode {mode}")
