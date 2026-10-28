"""Idea 165 correctness check (untimed). usage: python b165_check.py <checkpoint_root>

1. Unit (eager, random weights): Flux3ImageUpsample.forward of this tree vs main's forward
   (copied verbatim below as OLD) at the three decoder shapes (512ch 64^2, 512ch 128^2,
   256ch 256^2): channels_last bf16 inputs -> torch.equal values, output channels_last;
   NCHW fp32 inputs -> torch.equal values and identical strides (main's path unchanged).
2. Real weights, eager fp32 NCHW decoder (the reference decoder path): this tree vs main's
   upsample forward, seeds 0-2 -> bitwise.
3. Real weights, compiled bf16 decoder (served path): this tree (NEW) vs a second VAE whose
   upsample modules run main's forward (OLD), both compiled; per seed 0-2 error vs each
   other and vs the compiled fp32 decoder. Declared tolerance (reordered: the GN stats
   reduction reads x in NHWC order instead of NCHW, fp32 accumulation): PSNR(NEW, OLD)
   >= 45 dB, and PSNR(NEW, fp32) >= PSNR(OLD, fp32) - 0.5 dB on every seed.
4. Repeated execution on NEW: 10 warmup + 60 decodes (the timed sequence), every output
   bitwise == iteration 0, input unchanged; dynamo frames total 1 / ok 1, no graph break,
   counters unchanged over the sequence.
Last line: {"passed": ...}.
"""

import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    set_vae_decoder_trunk_dtype,
)

root = sys.argv[1]
WARMUP, ITERS = 10, 60


def log(*a):
    print(*a, flush=True)


def old_forward(self, x):  # main ac3317ba Flux3ImageUpsample.forward, verbatim
    if not self._subpixel_ready:
        self.refresh_subpixel()
    w4, b4 = self.subpixel_weight, self.subpixel_bias
    if w4.dtype != x.dtype:
        w4, b4 = w4.to(x.dtype), b4.to(x.dtype)
    n, c, h, w = x.shape
    o = nn.functional.conv2d(x, w4, b4, padding=1)  # [n, 4C, h+1, w+1]
    o = o.view(n, 2, 2, c, h + 1, w + 1)
    rows = [
        torch.stack([o[:, a, b, :, a : a + h, b : b + w] for b in range(2)], dim=-1)
        for a in range(2)
    ]
    y = torch.stack(rows, dim=3)
    return y.reshape(n, c, 2 * h, 2 * w)


class OldUpsample(iv.Flux3ImageUpsample):
    forward = old_forward


def use_old(vae):
    k = 0
    for m in vae.decoder.modules():
        if type(m) is iv.Flux3ImageUpsample:
            m.__class__ = OldUpsample
            k += 1
    return k


def to_u8(x):
    return gn_check.to_u8(x.float())


def diff(a, b):
    d = (a.float() - b.float()).abs()
    du = (to_u8(a).int() - to_u8(b).int()).abs()
    mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()
    return {"psnr_db": psnr, "max_abs": d.max().item(), "bitwise": bool(torch.equal(a, b)),
            "u8_max_diff": du.max().item(), "u8_frac_diff": (du > 0).float().mean().item()}


ok = True
res = {}

# 1. unit
torch.manual_seed(0)
unit = []
with torch.inference_mode():
    for c, hw in ((512, 64), (512, 128), (256, 256)):
        m = iv.Flux3ImageUpsample(c).cuda()
        m.conv.weight.normal_(0, 0.02)
        m.conv.bias.normal_(0, 0.02)
        m.refresh_subpixel()
        for dt, cl in ((torch.bfloat16, True), (torch.float32, False)):
            mm = m.to(dt)
            x = torch.randn(1, c, hw, hw, device="cuda", dtype=dt)
            if cl:
                mm = mm.to(memory_format=torch.channels_last)
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                mm = mm.to(memory_format=torch.contiguous_format)
            new = iv.Flux3ImageUpsample.forward(mm, x)
            old = old_forward(mm, x)
            r = {"c": c, "hw": hw, "dtype": str(dt), "channels_last_in": cl,
                 "equal": bool(torch.equal(new, old)),
                 "new_channels_last": new.is_contiguous(memory_format=torch.channels_last),
                 "same_strides": new.stride() == old.stride()}
            good = r["equal"] and (r["new_channels_last"] if cl else r["same_strides"])
            r["ok"] = good
            ok &= good
            unit.append(r)
            log(json.dumps(r))
res["unit"] = unit

DT = None


def latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g).to(DT)


# 2. eager fp32 NCHW decoder
vae_e, dts = gn_check.build_vae(root)
DT = getattr(torch, dts)
vae_o, _ = gn_check.build_vae(root)
log("old upsample modules swapped:", use_old(vae_o))
eager = []
with torch.inference_mode():
    for s in range(3):
        z = latent(s)
        a, b = vae_e.decode(z), vae_o.decode(z)
        r = {"seed": s, "bitwise": bool(torch.equal(a, b)), "dtype": str(a.dtype)}
        ok &= r["bitwise"]
        eager.append(r)
        log("eager fp32", json.dumps(r))
res["eager_fp32_nchw_vs_main"] = eager
del vae_e, vae_o
torch.cuda.empty_cache()

# 3. compiled bf16 NEW vs OLD, and vs compiled fp32
import torch._dynamo.utils as du  # noqa: E402


def frames():
    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


vae_n, _ = gn_check.build_vae(root)
set_vae_decoder_trunk_dtype(vae_n.decoder, torch.bfloat16)
vae_n.compile_decoder()
Z = latent(0)
Z0 = Z.clone()
with torch.inference_mode():
    for _ in range(WARMUP):
        vae_n.decode(Z)
torch.cuda.synchronize()
f_new = frames()
log("frames after NEW warmup:", f_new)

# 4. repeat sequence on NEW (same warmup + iteration count as the timed loop)
rep, inp, first = True, True, None
with torch.inference_mode():
    for _ in range(WARMUP):
        vae_n.decode(Z)
    for i in range(ITERS):
        inp &= torch.equal(Z, Z0)
        out = vae_n.decode(Z)
        if first is None:
            first = out.clone()
        elif not torch.equal(out, first):
            rep = False
torch.cuda.synchronize()
f_new2 = frames()
fr = f_new.get("frames", {})
seq = {"repeat_bitwise": rep, "inputs_unchanged": inp, "counters_unchanged": f_new == f_new2,
       "one_frame": fr.get("total") == 1 and fr.get("ok") == 1,
       "no_graph_break": not f_new.get("graph_break"), "frames": f_new2}
log("repeat", json.dumps(seq))
ok &= rep and inp and seq["counters_unchanged"] and seq["one_frame"] and seq["no_graph_break"]
res["repeat_sequence"] = seq

vae_old, _ = gn_check.build_vae(root)
use_old(vae_old)
set_vae_decoder_trunk_dtype(vae_old.decoder, torch.bfloat16)
vae_old.compile_decoder()
vae_f, _ = gn_check.build_vae(root)
vae_f.compile_decoder()
comp = []
with torch.inference_mode():
    for s in range(3):
        z = latent(s)
        n_, o_, f_ = vae_n.decode(z), vae_old.decode(z), vae_f.decode(z)
        r = {"seed": s, "new_vs_main": diff(n_, o_), "new_vs_fp32": diff(n_, f_), "main_vs_fp32": diff(o_, f_)}
        good = (r["new_vs_main"]["psnr_db"] >= 45.0
                and r["new_vs_fp32"]["psnr_db"] >= r["main_vs_fp32"]["psnr_db"] - 0.5)
        r["ok"] = good
        ok &= good
        comp.append(r)
        log("compiled bf16", json.dumps(r))
res["compiled_bf16"] = comp
log(json.dumps({"passed": bool(ok),
                "compared": "unit upsample vs main forward (bitwise); eager fp32 NCHW decoder vs main "
                            "upsample (bitwise, seeds 0-2); compiled bf16 decoder vs main-upsample compiled "
                            "bf16 and vs compiled fp32 (declared tol PSNR>=45 dB vs main, PSNR vs fp32 within "
                            "0.5 dB of main's); 10+60 repeat sequence bitwise, 1 frame, no break/recompile",
                **res}))
