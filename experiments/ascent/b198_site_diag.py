"""Idea 279 diagnostic: bf16 compiled decoder (compile_decoder()), ON vs OFF per site, seeds 0-7.
Variants: U1 only (up[3]), U2 only (up[2]), both. Control: OFF on z + 1e-5 noise vs OFF (bf16 chaos floor).
Reports u8_max / PSNR per variant; not a gate job."""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

T0 = time.time()
root = sys.argv[1]
torch._dynamo.config.cache_size_limit = 64
SEEDS = list(range(8))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def lat(s, eps=0.0):
    g = torch.Generator(device="cuda").manual_seed(s)
    z = torch.randn(1, 128, 64, 64, device="cuda", generator=g)
    if eps:
        g2 = torch.Generator(device="cuda").manual_seed(1000 + s)
        z = z + eps * torch.randn(1, 128, 64, 64, device="cuda", generator=g2)
    return z


def build(levels):
    vae, _ = gn_check.build_vae(root)
    set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    for lvl in (2, 3):
        vae.decoder.up[lvl].upsample.use_fused_bias_op = lvl in levels
    vae.compile_decoder()
    return vae


def cmp(a, b):
    a, b = a.float(), b.float()
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    du = (gn_check.to_u8(a).int() - gn_check.to_u8(b).int()).abs()
    return {"psnr": round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3) if mse > 0 else 999.0,
            "u8_max": du.max().item(), "u8_gt2": int((du > 2).sum())}


def agg(rows):
    return {"psnr_min": min(r["psnr"] for r in rows), "u8_max": max(r["u8_max"] for r in rows),
            "u8_gt2_total": sum(r["u8_gt2"] for r in rows)}


out = {}
with torch.inference_mode():
    O = {}
    for name, levels in (("OFF", ()), ("U1", (3,)), ("U2", (2,)), ("both", (2, 3))):
        v = build(levels)
        O[name] = {s: v.decode(lat(s)).clone() for s in SEEDS}
        if name == "OFF":
            out["ctrl_eps1e-5_vs_OFF"] = agg([cmp(v.decode(lat(s, 1e-5)), O["OFF"][s]) for s in SEEDS])
            log("ctrl", out["ctrl_eps1e-5_vs_OFF"])
        else:
            out[f"{name}_vs_OFF"] = agg([cmp(O[name][s], O["OFF"][s]) for s in SEEDS])
            log(name, out[f"{name}_vs_OFF"])
        del v
        torch.cuda.empty_cache()
out["passed"] = True
print(json.dumps(out))
