"""Idea 97: served FLUX.3 decoder at 2048^2, split attention on (B) vs landing 25 (A), one process.

usage: python vae_check.py <checkpoint_root>
Served VAE (fp32 package, bf16 decoder trunk + compile_decoder(fp8_convs=True), compile_encoder()).
Arms toggle split_attn_triton.ENABLED (TLLM_FLUX3_VAE_SPLIT_ATTN). Per arm: graph breaks, kernel
census of one decode, decode of latents seed 0/1. Alternating t2i(decode a) -> i2i(encode x +
decode b) sequence, ROUNDS rounds of B with A interleaved, error_on_recompile, bitwise vs each
arm's first. fp32 eager reference decoder R (no FP8, fp32 trunk): PSNR (u8) of A and B vs R.
DECLARED GATE: |PSNR(B, R) - PSNR(A, R)| <= 0.1 dB per latent; repeat bitwise; no recompile;
no graph breaks; inputs unchanged.
"""

import glob
import json
import math
import os
import sys
import time
from collections import Counter

import torch

ROOT = sys.argv[1]
ROUNDS = int(os.environ.get("ROUNDS", "50"))
STORE = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
    "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs"
)
T0 = time.time()
torch._dynamo.config.cache_size_limit = 64


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load(dtype=None):
    from safetensors.torch import load_file

    from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
        Flux3ImageVAE,
        Flux3ImageVAEConfig,
    )

    pkg = Flux3Checkpoint(ROOT, "nvfp4")
    cfg = pkg.config["vae"]
    vae = Flux3ImageVAE(Flux3ImageVAEConfig())
    vae.load_state_dict(load_file(str(pkg.asset(cfg["path"]))))
    dt = dtype or getattr(torch, cfg["dtype"])
    return vae.to(device="cuda", dtype=dt).eval(), getattr(torch, cfg["dtype"])


def build():
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

    vae, dt = load()
    set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    vae.compile_decoder(fp8_convs=True)
    vae.compile_encoder()
    return vae, dt


def latent(seed, dt):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, 128, 128, 128, device="cuda", generator=g).to(dt)


def image():
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels

    files = sorted(glob.glob(os.path.join(STORE, "*.png")))
    if not files:
        files = sorted(f for f in glob.glob(os.path.join(STORE, "**", "*"), recursive=True) if os.path.isfile(f))
    return _load_image_pixels(files[0], device=torch.device("cuda")).to(torch.float32)


def to_u8(x):
    x = (x.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    return (x * 255).round().to(torch.uint8)


def psnr_u8(a, b):
    mse = (to_u8(a).double() - to_u8(b).double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def census(fn, x):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total / 1e3
    return {"gpu_total_ms": round(sum(dur.values()), 3),
            "top": [(k[:100], cnt[k], round(dur[k], 3)) for k, _ in dur.most_common(10)]}


@torch.inference_mode()
def main():
    from torch._dynamo.utils import counters

    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

    vae, dt = build()
    za, zb = latent(0, dt), latent(1, dt)
    x = image()
    keep = [t.clone() for t in (za, zb, x)]
    res, gates = {"arms": {}}, {}
    vae.encode(torch.zeros(1, 3, 2048, 2048, device="cuda"))
    for arm, en in (("A", False), ("B", True)):
        sa.ENABLED = en
        gb0 = sum(counters["graph_break"].values())
        vae.decode(za)
        torch.cuda.synchronize()
        res["arms"][arm] = {"graph_breaks": sum(counters["graph_break"].values()) - gb0,
                            "census_decode": census(vae.decode, za)}
        log(arm, json.dumps(res["arms"][arm]))
    gates["no_graph_breaks"] = res["arms"]["A"]["graph_breaks"] == 0 and res["arms"]["B"]["graph_breaks"] == 0
    nb = " ".join(n for n, _, _ in res["arms"]["B"]["census_decode"]["top"])
    na = " ".join(n for n, _, _ in res["arms"]["A"]["census_decode"]["top"])
    gates["B_uses_split"] = "qk_tilemax" in nb and "pv_prologue" in nb and "_softmax_rows_e4m3" not in nb
    gates["A_is_landing25"] = "_softmax_rows_e4m3" in na and "qk_tilemax" not in na
    torch._dynamo.config.error_on_recompile = True
    first, rep, err = {}, True, None
    try:
        for i in range(ROUNDS):
            arms = ("B",) if i % 10 else (("A", "B") if i % 20 == 0 else ("B", "A"))
            for arm in arms:
                sa.ENABLED = arm == "B"
                outs = {"dec_a": vae.decode(za)}  # t2i
                outs["enc_x"] = vae.encode(x)  # i2i
                outs["dec_b"] = vae.decode(zb)
                for n, y in outs.items():
                    if (arm, n) in first:
                        rep &= bool(torch.equal(y, first[(arm, n)]))
                    else:
                        first[(arm, n)] = y.clone()
            if i % 10 == 0:
                ok = all(torch.equal(a, b) for a, b in zip((za, zb, x), keep))
                gates[f"inputs_unchanged_{i}"] = ok
                log(f"round {i}: repeat so far {rep}, inputs {ok}")
    except Exception as ex:  # noqa: BLE001
        err = repr(ex)[:500]
        log("FAILED", err)
    torch._dynamo.config.error_on_recompile = False
    gates["no_recompile"] = err is None
    gates["repeat_bitwise"] = rep
    gates["encode_unchanged_A_vs_B"] = bool(torch.equal(first[("A", "enc_x")], first[("B", "enc_x")]))
    del vae
    torch.cuda.empty_cache()
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import row_softmax_triton as rs

    ref, _ = load(torch.float32)
    sa.ENABLED, rs.ENABLED = False, False
    R = {"dec_a": ref.decode(za.float()).clone(), "dec_b": ref.decode(zb.float()).clone()}
    psnr = {}
    for n in ("dec_a", "dec_b"):
        pa, pb = psnr_u8(first[("A", n)], R[n]), psnr_u8(first[("B", n)], R[n])
        du = (to_u8(first[("A", n)]).int() - to_u8(first[("B", n)]).int()).abs()
        psnr[n] = {"A_vs_R_db": pa, "B_vs_R_db": pb, "delta_db": pb - pa, "B_vs_A_db": psnr_u8(first[("B", n)],
                   first[("A", n)]), "B_vs_A_u8_max": int(du.max().item()),
                   "B_vs_A_u8_frac": (du > 0).float().mean().item()}
        gates[f"psnr_{n}"] = abs(pb - pa) <= 0.1
    res["psnr"] = psnr
    res["recompile_error"] = err
    res["gates"] = gates
    res["passed"] = bool(all(gates.values()))
    res["output"] = {"class": "reordered"}
    log(json.dumps(psnr))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
