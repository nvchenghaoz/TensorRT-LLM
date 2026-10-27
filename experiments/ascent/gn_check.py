"""Correctness + diagnostics for the fused Flux3 VAE GroupNorm(+SiLU) Triton kernel.

usage: python gn_check.py unit|decoder|bench [checkpoint_root]
Last stdout line is a JSON object.
"""

import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import group_norm_triton as gnt
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv

SHAPES = [(512, 128), (512, 256), (512, 512), (256, 512), (256, 1024), (128, 1024)]
G, EPS = 32, 1e-6


def log(*a):
    print(*a, flush=True)


def unit():
    results = []
    ok_all = True
    for dtype in (torch.float32, torch.bfloat16):
        for C, R in SHAPES:
            for silu in (True, False):
                g = torch.Generator(device="cuda").manual_seed(1234 + C + R)
                x = (torch.randn(1, C, R, R, device="cuda", generator=g) * 2.0 + 0.5).to(dtype)
                w = (1 + 0.1 * torch.randn(C, device="cuda", generator=g)).to(dtype)
                b = (0.1 * torch.randn(C, device="cuda", generator=g)).to(dtype)
                x0 = x.clone()
                ref = F.group_norm(x, G, w, b, EPS)
                if silu:
                    ref = ref * torch.sigmoid(ref)
                out = gnt.fused_group_norm(x, G, w, b, EPS, silu)
                out2 = gnt.fused_group_norm(x, G, w, b, EPS, silu)
                torch.cuda.synchronize()
                inputs_ok = torch.equal(x, x0)
                repeat_ok = torch.equal(out, out2)
                r64 = F.group_norm(x.double(), G, w.double(), b.double(), EPS)
                if silu:
                    r64 = r64 * torch.sigmoid(r64)
                d = (out.float() - ref.float()).abs()
                e_ours = (out.double() - r64).abs()
                e_aten = (ref.double() - r64).abs()
                bitwise = (out == ref).float().mean().item()
                rec = dict(
                    dtype=str(dtype).split(".")[-1], C=C, HW=R, silu=silu,
                    bitwise_frac=round(bitwise, 6), max_abs_vs_aten=d.max().item(),
                    ours_vs_fp64_max=e_ours.max().item(), aten_vs_fp64_max=e_aten.max().item(),
                    ours_vs_fp64_mean=e_ours.mean().item(), aten_vs_fp64_mean=e_aten.mean().item(),
                    repeat_bitwise=repeat_ok, inputs_unchanged=inputs_ok,
                )
                # Tolerance (stated before the run): stats are only reordered, so our
                # error vs an fp64 reference must stay at aten's level.
                if dtype == torch.float32:
                    ok = rec["ours_vs_fp64_max"] <= 2 * rec["aten_vs_fp64_max"] + 1e-6
                    ok = ok and rec["max_abs_vs_aten"] <= 1e-4
                else:
                    ok = rec["ours_vs_fp64_mean"] <= 1.05 * rec["aten_vs_fp64_mean"] + 1e-6
                    ok = ok and torch.allclose(out.float(), ref.float(), rtol=1.6e-2, atol=1e-2)
                ok = ok and repeat_ok and inputs_ok
                rec["ok"] = bool(ok)
                ok_all &= bool(ok)
                log(json.dumps(rec))
                results.append(rec)
                del x, x0, ref, out, out2, r64, d, e_ours, e_aten
                torch.cuda.empty_cache()
    return {"passed": ok_all, "check": "unit fused GN(+SiLU) vs torch.nn.functional.group_norm (+x*sigmoid(x)), 6 decoder shapes x fp32/bf16", "n_cases": len(results)}


def build_vae(root):
    from safetensors.torch import load_file

    from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint

    pkg = Flux3Checkpoint(root, "nvfp4")
    cfg = pkg.config["vae"]
    log("vae config:", json.dumps(cfg))
    vae = iv.Flux3ImageVAE(iv.Flux3ImageVAEConfig())
    vae.load_state_dict(load_file(str(pkg.asset(cfg["path"]))))
    vae = vae.to(device="cuda", dtype=getattr(torch, cfg["dtype"])).eval()
    return vae, cfg["dtype"]


def to_u8(x):
    x = (x / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    return (x * 255).round().to(torch.uint8)


@torch.inference_mode()
def decoder(root, n_lat=3):
    vae, dt = build_vae(root)
    dtype = next(vae.parameters()).dtype
    log("served vae dtype:", dtype)
    ok = True
    recs = []
    for s in range(n_lat):
        g = torch.Generator(device="cuda").manual_seed(s)
        z = torch.randn(1, 128, 64, 64, device="cuda", generator=g, dtype=torch.float32).to(dtype)
        gnt.set_enabled(False)
        ref = vae.decode(z)
        gnt.set_enabled(True)
        o1 = vae.decode(z)
        o2 = vae.decode(z)
        torch.cuda.synchronize()
        u_ref, u1 = to_u8(ref), to_u8(o1)
        du = (u_ref.int() - u1.int()).abs()
        rec = dict(
            seed=s, repeat_bitwise=torch.equal(o1, o2),
            bitwise_frac=(o1 == ref).float().mean().item(),
            max_abs=(o1.float() - ref.float()).abs().max().item(),
            mean_abs=(o1.float() - ref.float()).abs().mean().item(),
            u8_equal_frac=(du == 0).float().mean().item(), u8_max_diff=du.max().item(),
        )
        log(json.dumps(rec))
        recs.append(rec)
        ok &= rec["repeat_bitwise"] and rec["u8_max_diff"] <= 2
    return {"passed": bool(ok), "check": "decoder fused vs aten GN on real VAE weights, repeated decode bitwise", "vae_dtype": dt, "per_latent": recs}


@torch.inference_mode()
def repeat(root, warmup=5, iters=60):
    """Untimed sequence matching the timed microbenchmark: same warmup and iteration
    count, no resets needed (decode has no mutable state). Every iteration must be
    bitwise equal to iteration 0, within tolerance of the aten reference, and the
    input latent must be unchanged."""
    vae, dt = build_vae(root)
    dtype = next(vae.parameters()).dtype
    z = torch.randn(1, 128, 64, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0)).to(dtype)
    z0 = z.clone()
    gnt.set_enabled(False)
    ref_u8 = to_u8(vae.decode(z))
    gnt.set_enabled(True)
    for _ in range(warmup):
        vae.decode(z)
    first = None
    ok = True
    worst_u8 = 0
    for i in range(iters):
        out = vae.decode(z)
        if first is None:
            first = out.clone()
        same = torch.equal(out, first)
        du = (to_u8(out).int() - ref_u8.int()).abs().max().item()
        worst_u8 = max(worst_u8, du)
        inp = torch.equal(z, z0)
        ok &= same and inp and du <= 2
        if i % 10 == 0:
            log(i, same, du, inp)
    return {"passed": bool(ok), "check": f"{warmup} warmup + {iters} fused decodes: all bitwise == iter 0, uint8 max diff vs aten ref <= 2, input unchanged", "u8_max_diff_vs_aten": worst_u8, "vae_dtype": dt}


@torch.inference_mode()
def bench(root, iters=20):
    vae, dt = build_vae(root)
    dtype = next(vae.parameters()).dtype
    z = torch.randn(1, 128, 64, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0)).to(dtype)
    times = {True: [], False: []}
    for flag in (False, True, False, True):
        gnt.set_enabled(flag)
        vae.decode(z)
    torch.cuda.synchronize()
    for i in range(iters):
        for flag in (False, True):
            gnt.set_enabled(flag)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            vae.decode(z)
            e.record()
            e.synchronize()
            times[flag].append(s.elapsed_time(e))
        if i % 5 == 0:
            log(i, times[False][-1], times[True][-1])
    res = {}
    for flag, name in ((False, "aten"), (True, "fused")):
        t = sorted(times[flag])
        res[name] = dict(median=statistics.median(t), p10=t[len(t) // 10], p90=t[(9 * len(t)) // 10])
    return {"vae_dtype": dt, **res}


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "unit":
        out = unit()
    elif mode == "decoder":
        out = decoder(sys.argv[2])
    elif mode == "bench":
        out = bench(sys.argv[2])
    elif mode == "all":
        out = {"unit": unit(), "decoder": decoder(sys.argv[2]), "bench": bench(sys.argv[2])}
        out["passed"] = out["unit"]["passed"] and out["decoder"]["passed"]
    elif mode == "check":
        out = {"unit": unit(), "decoder": decoder(sys.argv[2]), "repeat": repeat(sys.argv[2])}
        out["passed"] = all(out[k]["passed"] for k in ("unit", "decoder", "repeat"))
    out["elapsed_s"] = time.time() - t0
    print(json.dumps(out))
