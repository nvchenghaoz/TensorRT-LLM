"""Branch 37 B1 exploration: channels_last bf16 decoder with per-channel-stats GroupNorm.

Arms (bf16 trunk, conv_out fp32, real weights, each separately compiled):
  i   main options (layout_optimization=False, aten GN)
  B   layout_optimization=True + channel-stats GN
  C   channels_last params (decoder.to(memory_format=channels_last)), layout_optimization=False,
      channel-stats GN
  D   NCHW, layout_optimization=False, channel-stats GN
  R   fp32 main decoder (accuracy reference only; not timed)
Census per class reuses b37_step0.classify.

usage: python experiments/ascent/b37_b1.py <ckpt> [--arms i,B,C,D] [--iters 30]
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
p.add_argument("--arms", default="i,B,C,D")
p.add_argument("--iters", type=int, default=30)
p.add_argument("--warmup", type=int, default=3)
args = p.parse_args()


def log(*a):
    print(*a, flush=True)


from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    set_vae_decoder_trunk_dtype,
)

iv = gn_check.iv

ARMS = {
    "i": dict(dt=torch.bfloat16, layout=False, cl=False, cs=False),
    "B": dict(dt=torch.bfloat16, layout=True, cl=False, cs=True),
    "C": dict(dt=torch.bfloat16, layout=False, cl=True, cs=True),
    "D": dict(dt=torch.bfloat16, layout=False, cl=False, cs=True),
    "R": dict(dt=None, layout=False, cl=False, cs=False),
}


def classify(k):
    kl = k.lower()
    if "nchwtonhwc" in kl:
        return "reformat_nchwToNhwc"
    if "nhwctonchw" in kl:
        return "reformat_nhwcToNchw"
    if "converttensor" in kl:
        return "cudnn_convertTensor"
    if kl.startswith("triton_red") or kl.startswith("triton_per"):
        return "inductor_reduction"
    if kl.startswith("triton_poi"):
        return "inductor_pointwise"
    if kl.startswith("triton_"):
        return "inductor_other"
    if any(s in kl for s in ("flash", "fmha", "attention", "sgemm", "softmax", "gemv", "kernel2")):
        return "attention"
    if any(s in kl for s in ("copy", "catarray", "elementwise", "vectorized", "fill")):
        return "aten_copy_elementwise"
    if "tf32" in kl or "f32f32_f32f32" in kl:
        return "conv_out_fp32(tf32)"
    if any(s in kl for s in ("fprop", "xmma", "conv", "cutlass", "implicit", "cudnn", "gemm", "nvjet", "kernel")):
        return "conv_fprop"
    return "other"


def build(cfg):
    vae, _ = gn_check.build_vae(args.root)
    if cfg["dt"] is not None:
        set_vae_decoder_trunk_dtype(vae.decoder, cfg["dt"])
    if cfg["cl"]:
        vae.decoder.to(memory_format=torch.channels_last)
    for m in vae.decoder.modules():
        if isinstance(m, iv.Flux3GroupNorm):
            m.triton_under_compile = False
            m.channel_stats_under_compile = cfg["cs"]
    dtype = next(vae.decoder.parameters()).dtype
    options = {"layout_optimization": cfg["layout"]}
    if dtype != torch.float32:
        options["emulate_precision_casts"] = True
    vae.decoder.compile(dynamic=False, options=options)
    log("built", cfg, "trunk", dtype, "conv_out", vae.decoder.conv_out.weight.dtype)
    return vae


def latent(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


Z = latent(0)


@torch.inference_mode()
def decode(name, z=None):
    return VAES[name].decode(Z if z is None else z)


def timed(name):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    decode(name)
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def frames():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


def census(name):
    from torch.profiler import ProfilerActivity, profile

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
    return {
        "gpu_total_ms": round(sum(dur.values()) / 1e3, 3),
        "launches": sum(cnt.values()),
        "classes": {c: [cls_c[c], round(cls_d[c] / 1e3, 3)] for c in sorted(cls_d, key=lambda c: -cls_d[c])},
        "top": [(k[:150], cnt[k], round(dur[k] / 1e3, 3), classify(k)) for k, _ in dur.most_common(30)],
    }


def psnr(a, b):
    mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def diff(a, b):
    d = (a.float() - b.float()).abs()
    du = (gn_check.to_u8(a.float()).int() - gn_check.to_u8(b.float()).int()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "psnr": psnr(a, b),
            "u8_max": du.max().item(), "u8_frac": (du > 0).float().mean().item()}


arms = [a for a in args.arms.split(",") if a]
VAES = {}
for a in arms + ["R"]:
    t0 = time.time()
    VAES[a] = build(ARMS[a])
    for w in range(args.warmup):
        decode(a)
        torch.cuda.synchronize()
    log(f"[{a}] warm in {time.time() - t0:.1f}s; frames {frames()}")

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
    p10, p90 = v[int(0.1 * n)], v[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3), "noise_floor": round((p90 - p10) / med, 4)}


res = {"timing": {a: stats(ts[a]) for a in arms}, "census": {}, "acc": {}}
f0 = frames()
for a in arms:
    res["census"][a] = census(a)
    log(a, json.dumps(res["census"][a]))
# accuracy per seed vs the fp32 main decoder R (gate B style numbers, ungated here)
for s in range(3):
    z = latent(s)
    with torch.inference_mode():
        ref = decode("R", z).float().clone()
        outs = {a: decode(a, z).float().clone() for a in arms}
    for a in arms:
        res["acc"].setdefault(a, []).append({"seed": s, **diff(outs[a], ref)})
    if "i" in arms:
        for a in arms:
            res["acc"][a][-1]["vs_i_u8_max"] = diff(outs[a], outs["i"])["u8_max"]
res["no_recompile_after_warmup"] = frames() == f0
res["frames"] = frames()
log(json.dumps({k: v for k, v in res.items() if k != "census"}))
log(json.dumps({"passed": True, "timing": res["timing"],
                "classes": {a: res["census"][a]["classes"] for a in arms},
                "totals": {a: res["census"][a]["gpu_total_ms"] for a in arms},
                "no_recompile_after_warmup": res["no_recompile_after_warmup"]}))
