"""Idea 227 accuracy diagnostic (branch tree only). Seeds 0-7, real checkpoint VAE.
Decoders: fp8 ON / fp8 OFF (up[1].upsample.use_fused_bias_op False = main's path) and bf16 ON / OFF
(compile_decoder() without fp8). Each vs the fp32 eager decoder R: PSNR, mean abs, max abs, u8 max diff,
count of u8 diffs > 8 / > 16. Control: fp8 OFF on z + 1e-5 * noise (a tiny upstream perturbation)
to size how the e4m3 quantizers amplify any small change into u8 tails. ON vs OFF direct diffs."""

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

T0 = time.time()
root = sys.argv[1]
torch._dynamo.config.cache_size_limit = 64


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
rc = subprocess.run(["git", "merge-base", "--is-ancestor", "fb730d34", "HEAD"]).returncode
log("git", r.stdout.strip().replace("\n", " | "), "ancestor_rc", rc)
SEEDS = list(range(8))


def lat(s, eps=0.0):
    g = torch.Generator(device="cuda").manual_seed(s)
    z = torch.randn(1, 128, 64, 64, device="cuda", generator=g)
    if eps:
        g2 = torch.Generator(device="cuda").manual_seed(1000 + s)
        z = z + eps * torch.randn(1, 128, 64, 64, device="cuda", generator=g2)
    return z


def build(bf16, kw, flag):
    vae, _ = gn_check.build_vae(root)
    if bf16:
        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    vae.decoder.up[1].upsample.use_fused_bias_op = flag
    if kw is not None:
        vae.compile_decoder(**kw)
    return vae


def cmp(a, b):
    a, b = a.float(), b.float()
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    du = (gn_check.to_u8(a).int() - gn_check.to_u8(b).int()).abs()
    return {"psnr": round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3) if mse > 0 else float("inf"),
            "mean_abs": round((a - b).abs().mean().item(), 7), "max_abs": round((a - b).abs().max().item(), 4),
            "u8_max": du.max().item(), "u8_gt8": int((du > 8).sum()), "u8_gt16": int((du > 16).sum())}


with torch.inference_mode():
    ref = build(False, None, False)
    R = {s: ref.decode(lat(s)).clone() for s in SEEDS}
    del ref
    torch.cuda.empty_cache()
    log("R built")
    out = {}
    O = {}
    for name, bf16, kw, flag, eps in (("fp8_ON", True, {"fp8_convs": True}, True, 0.0),
                                      ("fp8_OFF", True, {"fp8_convs": True}, False, 0.0),
                                      ("bf16_ON", True, {}, True, 0.0),
                                      ("bf16_OFF", True, {}, False, 0.0)):
        v = build(bf16, kw, flag)
        O[name] = {s: v.decode(lat(s)).clone() for s in SEEDS}
        out[name] = [cmp(O[name][s], R[s]) for s in SEEDS]
        if name == "fp8_OFF":
            out["fp8_OFF_ctrl_eps1e-5"] = [cmp(v.decode(lat(s, 1e-5)), R[s]) for s in SEEDS]
            out["fp8_OFF_ctrl_vs_OFF"] = [cmp(v.decode(lat(s, 1e-5)), O[name][s]) for s in SEEDS]
        log(name, json.dumps(out[name]))
        del v
        torch.cuda.empty_cache()
    out["fp8_ON_vs_OFF"] = [cmp(O["fp8_ON"][s], O["fp8_OFF"][s]) for s in SEEDS]
    out["bf16_ON_vs_OFF"] = [cmp(O["bf16_ON"][s], O["bf16_OFF"][s]) for s in SEEDS]


def agg(rows):
    return {"psnr_min": min(r["psnr"] for r in rows), "psnr_mean": round(sum(r["psnr"] for r in rows) / len(rows), 3),
            "mean_abs_mean": round(sum(r["mean_abs"] for r in rows) / len(rows), 7),
            "u8_max": max(r["u8_max"] for r in rows), "u8_gt8_total": sum(r["u8_gt8"] for r in rows),
            "u8_gt16_total": sum(r["u8_gt16"] for r in rows)}


out["summary"] = {k: agg(v) for k, v in out.items() if isinstance(v, list)}
log("summary", json.dumps(out["summary"]))
out["passed"] = True
print(json.dumps(out))
