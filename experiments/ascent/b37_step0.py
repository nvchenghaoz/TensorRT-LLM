"""Branch 37 STEP 0 (B0): FLUX.3 VAE decode, per-class kernel census + interleaved timing.

Arms (each a separately compiled decoder on the real checkpoint weights):
  i    bf16 trunk, compile options of main (layout_optimization=False)
  ii   bf16 trunk, layout_optimization=True
  iii  bf16 trunk, main options, torch.backends.cudnn.benchmark=True set/restored around
       every decode of this arm (warmup included)
  f32  fp32 decoder, main options              (optional, for question (b))
  f32L fp32 decoder, layout_optimization=True  (optional, for question (b))

usage: python experiments/ascent/b37_step0.py <ckpt> [--arms i,ii,iii,f32,f32L] [--iters 30]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--arms", default="i,ii,iii,f32,f32L")
p.add_argument("--iters", type=int, default=30)
p.add_argument("--warmup", type=int, default=3)
args = p.parse_args()


def log(*a):
    print(*a, flush=True)


from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    set_vae_decoder_trunk_dtype,
)

iv = gn_check.iv
gen = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=gen)

ARMS = {
    "i": dict(dt=torch.bfloat16, layout=False, bench=False),
    "ii": dict(dt=torch.bfloat16, layout=True, bench=False),
    "iii": dict(dt=torch.bfloat16, layout=False, bench=True),
    "f32": dict(dt=None, layout=False, bench=False),
    "f32L": dict(dt=None, layout=True, bench=False),
}


def build(cfg):
    vae, _ = gn_check.build_vae(args.root)
    if cfg["dt"] is not None:
        set_vae_decoder_trunk_dtype(vae.decoder, cfg["dt"])
    for m in vae.decoder.modules():
        if isinstance(m, iv.Flux3GroupNorm):
            m.triton_under_compile = False
    dtype = next(vae.decoder.parameters()).dtype
    options = {"layout_optimization": cfg["layout"]}
    if dtype != torch.float32:
        options["emulate_precision_casts"] = True
    vae.decoder.compile(dynamic=False, options=options)
    log("built", cfg, "trunk", dtype, "conv_out", vae.decoder.conv_out.weight.dtype,
        "upsample subpixel?", hasattr(iv.Flux3ImageUpsample, "_load_hook") or "phase" in
        iv.Flux3ImageUpsample.forward.__code__.co_names)
    return vae


@torch.inference_mode()
def decode(name):
    cfg, vae = ARMS[name], VAES[name]
    z = Z.to(vae.decoder.post_quant_conv.weight.dtype if cfg["dt"] is None else torch.float32)
    old = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = cfg["bench"] or old
    try:
        return vae.decode(z)
    finally:
        torch.backends.cudnn.benchmark = old


def timed(name):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    decode(name)
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def classify(k):
    kl = k.lower()
    if "nchwtonhwc" in kl:
        return "reformat_nchwToNhwc"
    if "nhwctonchw" in kl:
        return "reformat_nhwcToNchw"
    if kl.startswith("triton_red") or kl.startswith("triton_per"):
        return "inductor_reduction(gn_stats/softmax)"
    if kl.startswith("triton_poi"):
        return "inductor_pointwise"
    if kl.startswith("triton_"):
        return "inductor_other"
    if any(s in kl for s in ("flash", "fmha", "attention", "sgemm", "softmax", "gemv", "kernel2")):
        return "attention"
    if any(s in kl for s in ("copy", "catarray", "elementwise", "vectorized", "fill")):
        return "aten_copy_elementwise"
    if "tf32" in kl or "f32f32_f32f32" in kl:
        return "conv_fp32(tf32)"
    if any(s in kl for s in ("fprop", "xmma", "conv", "cutlass", "implicit", "cudnn", "gemm", "nvjet", "kernel")):
        return "conv_fprop"
    return "other"


def census(name):
    from torch.profiler import ProfilerActivity, profile

    import torch._dynamo.utils as du
    f0 = {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        decode(name)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total
    cls_c, cls_d = Counter(), Counter()
    for k in cnt:
        c = classify(k)
        cls_c[c] += cnt[k]
        cls_d[c] += dur[k]
    f1 = {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}
    return {
        "gpu_total_ms": round(sum(dur.values()) / 1e3, 3),
        "launches": sum(cnt.values()),
        "classes": {c: [cls_c[c], round(cls_d[c] / 1e3, 3)] for c in sorted(cls_d, key=lambda c: -cls_d[c])},
        "top": [(k[:150], cnt[k], round(dur[k] / 1e3, 3), classify(k)) for k, _ in dur.most_common(40)],
        "counters_unchanged_during_census": f0 == f1,
    }


arms = [a for a in args.arms.split(",") if a]
VAES = {}
for a in arms:
    t0 = time.time()
    VAES[a] = build(ARMS[a])
    for w in range(args.warmup):
        decode(a)
        torch.cuda.synchronize()
        log(f"[{a}] warmup {w} at {time.time() - t0:.1f}s")

ts = {a: [] for a in arms}
for i in range(args.iters):
    r = i % len(arms)
    for a in arms[r:] + arms[:r]:
        ts[a].append(timed(a))
    if i % 10 == 0:
        log(f"iter {i} " + " ".join(f"{a}={ts[a][-1]:.2f}" for a in arms))


def stats(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2]
    return {"median_ms": round(med, 3), "p10": round(v[int(0.1 * n)], 3), "p90": round(v[max(int(0.9 * n) - 1, 0)], 3),
            "noise_floor": round((v[max(int(0.9 * n) - 1, 0)] - v[int(0.1 * n)]) / med, 4)}


res = {"timing_interleaved": {a: stats(ts[a]) for a in arms}, "census": {}}
for a in arms:
    res["census"][a] = census(a)
    log(a, json.dumps(res["census"][a]))
# accuracy of bf16 arms vs f32 arm on the same latent (information)
with torch.inference_mode():
    outs = {a: decode(a).float().clone() for a in arms}
if "f32" in outs:
    res["vs_f32"] = {a: {"max_abs": (outs[a] - outs["f32"]).abs().max().item(),
                         "mean_abs": (outs[a] - outs["f32"]).abs().mean().item()} for a in arms}
if "i" in outs:
    res["vs_i"] = {a: {"max_abs": (outs[a] - outs["i"]).abs().max().item()} for a in arms}
log(json.dumps({k: v for k, v in res.items() if k != "census"}))
log(json.dumps({"passed": True, "timing": res["timing_interleaved"],
                "classes": {a: res["census"][a]["classes"] for a in arms},
                "totals": {a: res["census"][a]["gpu_total_ms"] for a in arms}}))
