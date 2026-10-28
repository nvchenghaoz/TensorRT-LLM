"""Shared FLUX.3 image VAE decode microbenchmark (vae_decode area, plan #310).

Builds the REAL checkpoint VAE at the served dtype through gn_check.build_vae and
decodes a fixed-seed 1x128x64x64 packed latent (the 1024x1024 output) with
vae.decode. If the tree provides Flux3ImageVAE.compile_decoder (and VAE_COMPILE!=0)
the decoder is compiled first, as the pipeline does.

usage: python experiments/ascent/vae_decode_bench.py <checkpoint_root> [--mode M]
           [--warmup 5] [--iters 60] [--save PATH] [--ref PATH]
modes:
  time     (default) CUDA-event median of --iters decodes after --warmup, then an
           untimed repeat sequence with the same warmup/iteration count: every
           output bitwise == iteration 0, input latent unchanged. --save writes the
           output (base arm); --ref diffs this arm's output against a saved one.
           Last line: pair evidence JSON (batch/kernel_ms/noise_floor, passed).
  ab       in-process interleaved arms on the same weights/input: A = eager decoder,
           B = compile_decoder() (Inductor owns GroupNorm), C = compile_decoder(
           triton_group_norm=True) when supported, A2 = second eager instance (A/A).
  check    compiled vs eager decoder of this tree, in one process (declared
           tolerance: max abs <= 1e-2 and uint8 max diff <= 2), plus the repeat
           sequence and dynamo frame counters before/after the timed-length run.
  profile  one warm decode between cudaProfilerStart/Stop (nsys capture=region).
  kernels  launch-count census of one warm decode (torch.profiler), including
           RowwiseMomentsCUDAKernel and cuDNN nchwToNhwcKernel.
Run from the tree root (imports experiments/ascent/gn_check.py).
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--mode", default="time")
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=60)
p.add_argument("--save", default=None)
p.add_argument("--ref", default=None)
# BF16/FP16 decoder trunk (pipeline_config.vae_decode_dtype); applied before compile.
p.add_argument("--decode-dtype", default=None)
args = p.parse_args()
WARMUP, ITERS = args.warmup, args.iters
TOL_ABS, TOL_U8 = 1e-2, 2


def log(*a):
    print(*a, flush=True)


def build(compiled=True, decode_dtype="arg", **kw):
    vae, dt = gn_check.build_vae(args.root)
    decode_dtype = args.decode_dtype if decode_dtype == "arg" else decode_dtype
    if decode_dtype is not None:
        from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (
            set_vae_decoder_trunk_dtype,
        )

        set_vae_decoder_trunk_dtype(vae.decoder, getattr(torch, decode_dtype))
        log("decoder trunk dtype:", decode_dtype, "post_quant_conv", vae.decoder.post_quant_conv.weight.dtype,
            "conv_out", vae.decoder.conv_out.weight.dtype)
    is_compiled = False
    if compiled and hasattr(vae, "compile_decoder") and os.environ.get("VAE_COMPILE", "1") != "0":
        vae.compile_decoder(**kw)
        is_compiled = True
    return vae, is_compiled


def supports_triton_kw():
    import inspect

    fn = getattr(gn_check.iv.Flux3ImageVAE, "compile_decoder", None)
    return fn is not None and "triton_group_norm" in inspect.signature(fn).parameters


from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint  # noqa: E402

DTYPE = getattr(torch, Flux3Checkpoint(args.root, "nvfp4").config["vae"]["dtype"])
gen = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=gen).to(DTYPE)
Z0 = Z.clone()
log("served vae dtype:", DTYPE)


@torch.inference_mode()
def decode(vae):
    return vae.decode(Z)


def timed(vae):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    decode(vae)
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
        decode(vae)
        torch.cuda.synchronize()
        log(f"[{tag}] warmup {i} done at {time.time() - t0:.1f}s")
    return time.time() - t0


def repeat_seq(vae):
    """Untimed: same warmup + iteration count as the timed loop."""
    for _ in range(WARMUP):
        decode(vae)
    first, rep, inp = None, True, True
    for i in range(ITERS):
        inp &= torch.equal(Z, Z0)
        out = decode(vae)
        if first is None:
            first = out.clone()
        elif not torch.equal(out, first):
            rep = False
    torch.cuda.synchronize()
    return first, rep, inp


def diff(a, b):
    d = (a.float() - b.float()).abs()
    du = (gn_check.to_u8(a.float()).int() - gn_check.to_u8(b.float()).int()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "bitwise_frac": (d == 0).float().mean().item(),
            "u8_max_diff": du.max().item(), "u8_frac_diff": (du > 0).float().mean().item()}


def frames():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


def psnr(a, b):
    mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


if args.mode == "time":
    vae, comp = build()
    log("compiled:", comp)
    warm_s = warm(vae, "T")
    ts = []
    for i in range(ITERS):
        ts.append(timed(vae))
        if i % 10 == 0:
            log(f"iter {i} {ts[-1]:.2f} ms")
    st = stats(ts)
    f0 = frames()
    first, rep, inp = repeat_seq(vae)
    no_recompile = frames() == f0
    res = {"batch": {"1": {"kernel_ms": st["median_ms"], "noise_floor": st["noise_floor"]}},
           "output": {"class": "reordered" if comp else "exact"},
           "compiled": comp, "warmup_s": warm_s, **st, "repeat_bitwise": rep,
           "inputs_unchanged": inp, "no_recompile_after_warmup": no_recompile,
           "frames_after_timing": f0, "out_dtype": str(first.dtype),
           "bench": "experiments/ascent/vae_decode_bench.py time (real weights, 1x128x64x64)"}
    passed = rep and inp and no_recompile
    if args.decode_dtype is not None:
        # Mode switch: error vs this tree's served-dtype (fp32) decoder, compiled the
        # same way (main's code path; unchanged by this branch), seeds 0-2.
        res["decode_dtype"] = args.decode_dtype
        res["output"]["class"] = "mode_switch"
        ref_vae, _ = build(decode_dtype=None)
        acc = []
        with torch.inference_mode():
            for s in range(3):
                g = torch.Generator(device="cuda").manual_seed(s)
                z = torch.randn(1, 128, 64, 64, device="cuda", generator=g).to(DTYPE)
                r = ref_vae.decode(z)
                o = vae.decode(z)
                acc.append({"seed": s, "psnr_db": psnr(o, r), **diff(o, r)})
                log(json.dumps(acc[-1]))
        res["output"]["error_vs_fp32_compiled"] = {
            "psnr_min_db": min(a["psnr_db"] for a in acc),
            "max_abs": max(a["max_abs"] for a in acc),
            "u8_max_diff": max(a["u8_max_diff"] for a in acc),
            "u8_frac_diff_max": max(a["u8_frac_diff"] for a in acc)}
        res["accuracy_per_seed"] = acc
    if args.save:
        torch.save(first.cpu(), args.save)
        res["saved"] = args.save
    if args.ref:
        if os.path.exists(args.ref):
            res["vs_ref"] = diff(first, torch.load(args.ref).cuda())
            passed = passed and res["vs_ref"]["max_abs"] <= TOL_ABS and res["vs_ref"]["u8_max_diff"] <= TOL_U8
        else:
            res["vs_ref"] = "ref file not found (arms on different hosts/containers)"
    res["passed"] = bool(passed)
    log(json.dumps(res))

elif args.mode == "ab":
    arms = {"A": build(False)[0]}
    arms["B"], b_comp = build(True)
    if supports_triton_kw():
        arms["C"], _ = build(True, triton_group_norm=True)
    arms["A2"] = build(False)[0]
    warm_s = {k: warm(v, k) for k, v in arms.items()}
    order = list(arms)
    ts = {k: [] for k in order}
    for i in range(ITERS):
        r = i % len(order)
        for k in order[r:] + order[:r]:
            ts[k].append(timed(arms[k]))
        if i % 10 == 0:
            log(f"iter {i} " + " ".join(f"{k}={ts[k][-1]:.2f}" for k in order))
    st = {k: stats(v) for k, v in ts.items()}
    outs = {k: decode(v).clone() for k, v in arms.items()}
    a, b, a2 = st["A"]["median_ms"], st["B"]["median_ms"], st["A2"]["median_ms"]
    log(json.dumps({
        "batch": {"1": {"kernel_ms": b, "noise_floor": st["B"]["noise_floor"]}},
        "output": {"class": "reordered" if b_comp else "exact"},
        "interleaved": True, "arms": st, "saving_ms_A_minus_B": a - b, "speedup_A_over_B": a / b,
        "aa_rel_diff": abs(a2 - a) / a,
        "C_triton_gn_compiled_ms": st["C"]["median_ms"] if "C" in st else None,
        "B_vs_A": diff(outs["B"], outs["A"]),
        "C_vs_A": diff(outs["C"], outs["A"]) if "C" in outs else None,
        "warmup_s": warm_s,
    }))

elif args.mode == "check":
    ref_vae, _ = build(False)
    ref = decode(ref_vae).clone()
    ref_rep = torch.equal(decode(ref_vae), ref)
    del ref_vae
    log(f"declared tolerance: max_abs <= {TOL_ABS}, uint8 max diff <= {TOL_U8} vs this tree's eager decoder")
    vae, comp = build()
    warm(vae, "C")
    f0 = frames()
    first, rep, inp = repeat_seq(vae)
    f1 = frames()
    d = diff(first, ref)
    passed = rep and inp and ref_rep and f0 == f1 and d["max_abs"] <= TOL_ABS and d["u8_max_diff"] <= TOL_U8
    log(json.dumps({"passed": bool(passed),
                    "compared": "compiled decoder vs this tree's eager decoder (real weights, same latent); "
                                f"{WARMUP} warmup + {ITERS} decodes all bitwise == iteration 0",
                    "compiled": comp, **d, "repeat_bitwise": rep, "eager_repeat_bitwise": ref_rep,
                    "inputs_unchanged": inp, "no_recompile_after_warmup": f0 == f1,
                    "frames_before": f0, "frames_after": f1}))

elif args.mode == "profile":
    # One warm decode between cudaProfilerStart/Stop (nsys/ncu capture=region).
    vae, comp = build()
    warm(vae, "P")
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    decode(vae)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    log(json.dumps({"profiled": True, "compiled": comp}))

elif args.mode == "kernels":
    from collections import Counter

    from torch.profiler import ProfilerActivity, profile

    gn_calls = Counter()
    orig_can = gn_check.iv.can_use_fused_gn

    def counting_can(x, gn):
        ok = orig_can(x, gn)
        if not torch.compiler.is_compiling():
            gn_calls["fused" if ok else "aten"] += 1
            gn_calls[f"dtype_{str(x.dtype).split('.')[-1]}"] += 1
        return ok

    gn_check.iv.can_use_fused_gn = counting_can
    vae, comp = build()
    warm(vae, "K")
    f_warm = frames()
    gn_calls.clear()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        decode(vae)
        torch.cuda.synchronize()
    gn_one = dict(gn_calls)
    for _ in range(3):
        decode(vae)
    torch.cuda.synchronize()
    f_after3 = frames()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total

    def match(s):
        ks = [k for k in cnt if s in k]
        return {"launches": sum(cnt[k] for k in ks), "total_ms": sum(dur[k] for k in ks) / 1e3}

    conv_like = {k: (cnt[k], round(dur[k] / 1e3, 3)) for k in cnt
                 if any(s in k.lower() for s in ("conv", "xmma", "cudnn", "implicit", "gemm", "cutlass", "sm80", "sm90", "sm100", "sm120"))}
    tf32 = {k: v for k, v in conv_like.items() if "tf32" in k.lower() or "f32f32_f32f32" in k.lower()}
    log(json.dumps({"compiled": comp, "decode_dtype": args.decode_dtype, "total_launches": sum(cnt.values()),
                    "gpu_total_ms": sum(dur.values()) / 1e3,
                    "RowwiseMoments": match("RowwiseMoments"), "native_group_norm": match("group_norm"),
                    "nchwToNhwc": match("nchwToNhwc"),
                    "triton_gn_stats_apply": match("_gn_"), "inductor_triton": match("triton_"),
                    "sgemm_simt": match("sgemm"), "softmax": match("softmax"),
                    "gn_python_calls_one_decode": gn_one,
                    "frames_after_warmup": f_warm, "frames_after_3_more_decodes": f_after3,
                    "counters_unchanged_3_decodes": f_warm == f_after3,
                    "tf32_kernels": tf32,
                    "conv_gemm_kernels": sorted(((k[:140], *v) for k, v in conv_like.items()), key=lambda t: -t[2]),
                    "top": [(k[:140], cnt[k], round(dur[k] / 1e3, 3)) for k, _ in dur.most_common(25)]}))
else:
    raise SystemExit(f"unknown mode {args.mode}")
