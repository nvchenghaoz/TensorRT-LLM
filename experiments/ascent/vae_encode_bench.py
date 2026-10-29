"""FLUX.3 VAE encoder microbenchmark / correctness check at the served 2048^2 i2i shape.

Loads ONLY the image VAE (fp32 package dtype) from the checkpoint and the frozen 2048^2
edit references (inputs.json store_dir). Modes:
  bench    : timing. The tree's served encoder (compiled channels_last when the tree has
             Flux3ImageVAE.compile_encoder, else eager) interleaved with an eager instance;
             CUDA events, warm. kernel_ms = served path median over the real reference.
  check    : untimed correctness (candidate tree): repeated-call bitwise, no stale state
             across two images, compiled vs eager vs fp32 no-TF32 reference, no recompile
             after the zeros warmup (torch._dynamo error_on_recompile), peak memory.
  profile  : one compiled encode of a real reference between cudaProfilerStart/Stop.
  stats64  : diagnostic: compiled with fp64 GN statistics vs fp32 GN statistics.
usage: python vae_encode_bench.py <mode> <checkpoint>
"""

import glob
import json
import math
import os
import statistics
import sys
import time

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
CKPT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CKPT")
ITERS = int(os.environ.get("ITERS", "24"))
STORE = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
    "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs"
)
DEV = torch.device("cuda")


def log(msg):
    print(f"[vae-enc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_vae():
    from safetensors.torch import load_file

    from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
        Flux3ImageVAE,
        Flux3ImageVAEConfig,
    )

    pkg = Flux3Checkpoint(CKPT, "nvfp4")
    cfg = pkg.config["vae"]
    vae = Flux3ImageVAE(Flux3ImageVAEConfig())
    vae.load_state_dict(load_file(str(pkg.asset(cfg["path"]))))
    return vae.to(device=DEV, dtype=getattr(torch, cfg["dtype"])).eval()


def load_images(n=2):
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels

    files = sorted(glob.glob(os.path.join(STORE, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(STORE, "**", "*"), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
    log(f"{len(files)} reference files in {STORE}; using {[os.path.basename(f) for f in files[:n]]}")
    # As _prepare_i2i: _load_image_pixels then .to(vae dtype) (fp32, a no-op).
    return [_load_image_pixels(f, device=DEV).to(torch.float32) for f in files[:n]]


def has_compile_encoder(vae):
    return hasattr(vae, "compile_encoder")


def timed(fn, x):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def spread(v):
    q = statistics.quantiles(sorted(v), n=10)
    return (q[-1] - q[0]) / statistics.median(v)


def psnr(a, ref):
    mse = (a.double() - ref.double()).pow(2).mean().item()
    rng = (ref.max() - ref.min()).item()
    return float("inf") if mse == 0 else 10 * math.log10(rng * rng / mse)


def err(a, ref):
    d = (a.double() - ref.double()).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "max_rel_to_absmax": d.max().item() / ref.abs().max().item(),
        "psnr_db": psnr(a, ref),
    }


def peak(fn, x):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(x)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


@torch.inference_mode()
def bench():
    served = load_vae()
    compiled = has_compile_encoder(served)
    if compiled:
        served.compile_encoder()
    eager = load_vae()
    (x,) = load_images(1)
    zeros = torch.zeros(1, 3, 2048, 2048, device=DEV)
    t0 = time.time()
    served.encode(zeros)  # the server's warmup encode (compiles when compiled)
    torch.cuda.synchronize()
    log(f"served warmup encode (compile={compiled}) {time.time() - t0:.1f} s")
    for _ in range(3):
        served.encode(x)
        eager.encode(x)
    torch.cuda.synchronize()
    ts, te, tz = [], [], []
    for i in range(ITERS):
        ts.append(timed(served.encode, x))
        te.append(timed(eager.encode, x))
        if i % 4 == 0:
            tz.append(timed(served.encode, zeros))
            log(f"iter {i}: served {ts[-1]:.2f} ms eager {te[-1]:.2f} ms")
    # A/A: the eager instance split into even/odd iterations.
    aa = abs(statistics.median(te[0::2]) - statistics.median(te[1::2])) / statistics.median(te)
    pk_s, pk_e = peak(served.encode, x), peak(eager.encode, x)
    res = {
        "batch": {
            "i2i-2048": {"kernel_ms": statistics.median(ts), "noise_floor": spread(ts)},
        },
        "output": {"class": "reordered" if compiled else "exact"},
        "served_compiled": compiled,
        "served_ms_median": statistics.median(ts),
        "eager_ms_median": statistics.median(te),
        "eager_noise_floor": spread(te),
        "aa_eager_even_odd_rel": aa,
        "served_zeros_ms_median": statistics.median(tz),
        "peak_gib_served": pk_s,
        "peak_gib_eager": pk_e,
        "iters": ITERS,
    }
    log("done")
    print(json.dumps(res))


@torch.inference_mode()
def check():
    import torch._dynamo
    from torch._dynamo.utils import counters

    comp = load_vae()
    assert has_compile_encoder(comp), "candidate tree needed"
    comp.compile_encoder()
    eager = load_vae()
    a, b = load_images(2)
    a0, b0 = a.clone(), b.clone()
    zeros = torch.zeros(1, 3, 2048, 2048, device=DEV)
    log(f"input strides: real {tuple(a.stride())}, zeros {tuple(zeros.stride())}")
    t0 = time.time()
    comp.encode(zeros)
    torch.cuda.synchronize()
    log(f"warmup compile {time.time() - t0:.1f} s; counters {dict(counters['stats'])}")
    frames_before = dict(counters["frames"])
    stats_before = dict(counters["stats"])
    torch._dynamo.config.error_on_recompile = True

    # Protected reference: the eager fp32 encoder (main's path), plus an fp32 no-TF32 run.
    ref_a, ref_b = eager.encode(a), eager.encode(b)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        r32_a = eager.encode(a)
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    eager_vs_fp32 = err(ref_a, r32_a)
    log(f"eager(TF32) vs fp32 no-TF32: {eager_vs_fp32}")

    ok = True
    seq = []
    first = {}
    t_recompile = None
    try:
        for i, (name, x, ref) in enumerate([("A", a, ref_a), ("A", a, ref_a), ("A", a, ref_a),
                                             ("B", b, ref_b), ("A", a, ref_a), ("B", b, ref_b)]):
            out = comp.encode(x)
            e = err(out, ref)
            bit = None
            if name in first:
                bit = bool(torch.equal(out, first[name]))
                ok &= bit
            else:
                first[name] = out.clone()
            inputs_ok = bool(torch.equal(a, a0) and torch.equal(b, b0))
            ok &= inputs_ok
            seq.append({"i": i, "img": name, "bitwise_vs_first": bit, "inputs_unchanged": inputs_ok,
                        "vs_eager": e, "vs_fp32_noTF32": err(out, r32_a) if name == "A" else None})
            log(f"iter {i} {name}: bitwise_vs_first={bit} vs_eager={e}")
    except Exception as exc:  # a recompile raises under error_on_recompile
        t_recompile = repr(exc)
        ok = False
        log(f"FAILED: {t_recompile}")
    frames_after = dict(counters["frames"])
    stats_after = dict(counters["stats"])
    torch._dynamo.config.error_on_recompile = False
    # Pre-registered tolerance (reordered: cuDNN NHWC TF32 engines + Inductor reductions):
    # compiled vs eager max abs <= 2x the eager TF32-vs-fp32 max abs, and PSNR >= 60 dB.
    comp_vs_eager = [s["vs_eager"] for s in seq]
    tol = 2 * eager_vs_fp32["max_abs"]
    tol_ok = bool(comp_vs_eager) and all(
        c["max_abs"] <= tol and c["psnr_db"] >= 60 for c in comp_vs_eager)
    a_vs_32 = [s["vs_fp32_noTF32"] for s in seq if s["vs_fp32_noTF32"]]
    pk_c, pk_e = peak(comp.encode, a), peak(eager.encode, a)
    res = {
        "passed": bool(ok and tol_ok and pk_c <= pk_e and len(seq) == 6),
        "compared": "compiled channels_last encoder vs eager fp32 encoder (main's path) on 2 real "
                    "2048^2 references; repeated calls bitwise vs first; fp32 no-TF32 for scale",
        "tolerance": {"max_abs": tol, "psnr_db_min": 60},
        "eager_vs_fp32_noTF32": eager_vs_fp32,
        "compiled_vs_fp32_noTF32": a_vs_32[0] if a_vs_32 else None,
        "compiled_vs_eager_worst_max_abs": max(c["max_abs"] for c in comp_vs_eager) if comp_vs_eager else None,
        "compiled_vs_eager_min_psnr": min(c["psnr_db"] for c in comp_vs_eager) if comp_vs_eager else None,
        "repeat_bitwise": all(s["bitwise_vs_first"] in (None, True) for s in seq),
        "recompile_error": t_recompile,
        "dynamo_frames_before_real_inputs": frames_before,
        "dynamo_frames_after": frames_after,
        "dynamo_stats_before": stats_before,
        "dynamo_stats_after": stats_after,
        "peak_gib_compiled": pk_c,
        "peak_gib_eager": pk_e,
        "seq": seq,
    }
    print(json.dumps(res))


@torch.inference_mode()
def profile():
    vae = load_vae()
    if has_compile_encoder(vae):
        vae.compile_encoder()
    (x,) = load_images(1)
    vae.encode(torch.zeros(1, 3, 2048, 2048, device=DEV))
    for _ in range(2):
        vae.encode(x)
    torch.cuda.synchronize()
    log("profiling one encode")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("vae_encode")
    vae.encode(x)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({"done": True}))


@torch.inference_mode()
def stats64():
    v32, v64 = load_vae(), load_vae()
    v32.compile_encoder(fp32_channel_stats=True)
    v64.compile_encoder(fp32_channel_stats=False)
    (x,) = load_images(1)
    z = torch.zeros(1, 3, 2048, 2048, device=DEV)
    for v in (v32, v64):
        t0 = time.time()
        v.encode(z)
        torch.cuda.synchronize()
        log(f"compile {time.time() - t0:.1f} s")
        v.encode(x)
    t32, t64 = [], []
    for i in range(ITERS):
        t32.append(timed(v32.encode, x))
        t64.append(timed(v64.encode, x))
        if i % 6 == 0:
            log(f"iter {i}: fp32 stats {t32[-1]:.2f} fp64 stats {t64[-1]:.2f}")
    o32, o64 = v32.encode(x), v64.encode(x)
    res = {"fp32_stats_ms": statistics.median(t32), "fp64_stats_ms": statistics.median(t64),
           "noise32": spread(t32), "noise64": spread(t64), "fp32_vs_fp64_stats": err(o32, o64)}
    print(json.dumps(res))


if __name__ == "__main__":
    {"bench": bench, "check": check, "profile": profile, "stats64": stats64}[MODE]()
