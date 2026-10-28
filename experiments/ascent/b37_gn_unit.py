"""Branch 37 GN unit test: group_norm_channel_stats vs aten GroupNorm, both compiled.

For each FLUX.3 decoder GroupNorm shape (1024^2 decode, 32 groups) and dtype (bf16, fp32):
  ref    fp64 F.group_norm of the same (rounded) input, weight and bias
  aten   torch.compile(F.group_norm) on NCHW input            (main's traced path)
  new    torch.compile(group_norm_channel_stats) on channels_last input (branch path)
  also   swish variants (GN + x*sigmoid(x)), as forward_silu traces them.
Declared gate (before any run): per case, mean_abs(new - ref) <= 1.05 x mean_abs(aten - ref)
and max_abs(new - ref) <= 1.05 x max_abs(aten - ref) + one output ulp at the max |ref|
(bf16 output rounding of a single element can flip by one ulp under any reordering).
Each compiled op is also run 5x: outputs bitwise identical across calls; input unchanged.
Timing (CUDA events, median of 30) reported for information.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
    group_norm_channel_stats,
    swish,
)

for _k in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, 64)
SHAPES = [(512, 64), (512, 128), (512, 256), (256, 256), (256, 512), (128, 512)]
G, EPS = 32, 1e-6


def log(*a):
    print(*a, flush=True)


def aten_gn(x, w, b, silu):
    y = F.group_norm(x, G, w, b, EPS)
    return swish(y) if silu else y


def new_gn(x, w, b, silu):
    y = group_norm_channel_stats(x, G, w, b, EPS)
    return swish(y) if silu else y


def med_ms(fn, *a):
    ts = []
    for _ in range(30):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*a)
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[15]


ok_all, recs = True, []
with torch.inference_mode():
    for dt in (torch.bfloat16, torch.float32):
        opts = {"emulate_precision_casts": True} if dt != torch.float32 else {}
        for silu in (False, True):
            ca = torch.compile(lambda x, w, b: aten_gn(x, w, b, silu), dynamic=False, options=dict(opts))
            cn = torch.compile(lambda x, w, b: new_gn(x, w, b, silu), dynamic=False, options=dict(opts))
            for c, hw in SHAPES:
                torch.manual_seed(c * 7 + hw)
                # per-channel offsets/scales like real activations (non-zero means stress var)
                x = (torch.randn(1, c, hw, hw, device="cuda") * (0.5 + torch.rand(1, c, 1, 1, device="cuda") * 3)
                     + torch.randn(1, c, 1, 1, device="cuda") * 2).to(dt)
                w = (1 + 0.3 * torch.randn(c, device="cuda")).to(dt)
                b = (0.2 * torch.randn(c, device="cuda")).to(dt)
                ref = aten_gn(x.double(), w.double(), b.double(), silu)
                xcl = x.contiguous(memory_format=torch.channels_last)
                x0, xcl0 = x.clone(), xcl.clone()
                oa = ca(x, w, b)
                on = cn(xcl, w, b)
                rep = True
                for _ in range(5):
                    rep &= torch.equal(ca(x, w, b), oa) and torch.equal(cn(xcl, w, b), on)
                inp = torch.equal(x, x0) and torch.equal(xcl, xcl0)
                ea = (oa.double() - ref).abs()
                en = (on.double() - ref).abs()
                ulp = (ref.abs().max() * (2.0 ** -8 if dt == torch.bfloat16 else 2.0 ** -23)).item()
                gate = (en.mean().item() <= 1.05 * ea.mean().item()
                        and en.max().item() <= 1.05 * ea.max().item() + ulp)
                t_aten = med_ms(ca, x, w, b)
                t_new = med_ms(cn, xcl, w, b)
                t_aten_cl = med_ms(ca, xcl, w, b)
                rec = {"dtype": str(dt).split(".")[-1], "silu": silu, "shape": [1, c, hw, hw],
                       "aten_mean_abs": ea.mean().item(), "new_mean_abs": en.mean().item(),
                       "ratio_mean": en.mean().item() / max(ea.mean().item(), 1e-30),
                       "aten_max_abs": ea.max().item(), "new_max_abs": en.max().item(),
                       "new_vs_aten_max_abs": (on.double() - oa.double()).abs().max().item(),
                       "out_is_channels_last": on.is_contiguous(memory_format=torch.channels_last),
                       "repeat_bitwise": bool(rep), "inputs_unchanged": bool(inp), "gate": bool(gate),
                       "ms_aten_nchw": round(t_aten, 4), "ms_new_nhwc": round(t_new, 4),
                       "ms_aten_on_nhwc": round(t_aten_cl, 4)}
                log(json.dumps(rec))
                recs.append(rec)
                ok_all &= gate and rep and inp
                del x, xcl, ref, oa, on, ea, en, x0, xcl0
                torch.cuda.empty_cache()

log(json.dumps({"passed": bool(ok_all), "n_cases": len(recs),
                "compared": "compiled group_norm_channel_stats (channels_last) and compiled aten group_norm (NCHW) "
                            "vs fp64 F.group_norm; 6 decoder shapes x {bf16, fp32} x {GN, GN+swish}",
                "max_ratio_mean": max(r["ratio_mean"] for r in recs),
                "failed": [r for r in recs if not r["gate"]]}))
