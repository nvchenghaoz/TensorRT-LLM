"""Idea 279 correctness (candidate tree): up[2]/up[3] upsample on flux3_vae::subpixel_up_conv ON vs
OFF (OFF = instances' use_fused_bias_op False at up[2], up[3] = main's aten dgrad path + fold).
Seeds 0-7, real checkpoint VAE, each decoder vs the fp32 eager decoder R and ON vs OFF directly.
Decoders: served FP8 (compile_decoder(fp8_convs=True)) and bf16 (compile_decoder()).
Repeated-execution sequence on FP8 ON: 10 warm + 30 decodes bitwise vs iteration 0, latent unchanged,
Dynamo counters and cuDNN FE graph counts unchanged after warmup.
Declared gates: bf16 ON vs OFF u8_max <= 2 and PSNR >= 60 dB (all seeds); FP8 ON vs R mean_abs <= 1.05x
FP8 OFF's (mean over seeds) and PSNR min >= OFF's - 0.3 dB; repeat bitwise; no recompile / FE build."""

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import subpixel_conv as spc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

T0 = time.time()
root = sys.argv[1]
torch._dynamo.config.cache_size_limit = 64


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
rcs = {c: subprocess.run(["git", "merge-base", "--is-ancestor", c, "HEAD"]).returncode
       for c in ("da43d58f", "59981b4e")}
log("git", r.stdout.strip().replace("\n", " | "), "ancestor_rcs", rcs, "UP_FE_DGRAD_EXTRA", spc.UP_FE_DGRAD_EXTRA)
SEEDS = list(range(8))


def lat(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


def build(bf16, kw, flag):
    vae, _ = gn_check.build_vae(root)
    if bf16:
        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    for lvl in (2, 3):
        vae.decoder.up[lvl].upsample.use_fused_bias_op = flag
    assert vae.decoder.up[1].upsample.use_fused_bias_op and not vae.decoder.up[1].upsample.use_transpose
    if kw is not None:
        vae.compile_decoder(**kw)
    return vae


def cmp(a, b):
    a, b = a.float(), b.float()
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    du = (gn_check.to_u8(a).int() - gn_check.to_u8(b).int()).abs()
    return {"psnr": round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3) if mse > 0 else 999.0,
            "mean_abs": round((a - b).abs().mean().item(), 7), "max_abs": round((a - b).abs().max().item(), 4),
            "u8_max": du.max().item()}


def counters():
    import torch._dynamo.utils as du
    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break", "stats")}


def agg(rows):
    return {"psnr_min": min(r["psnr"] for r in rows),
            "mean_abs_mean": round(sum(r["mean_abs"] for r in rows) / len(rows), 7),
            "u8_max": max(r["u8_max"] for r in rows)}


out = {"git": r.stdout.strip(), "ancestor_rcs": rcs}
with torch.inference_mode():
    ref = build(False, None, False)
    R = {s: ref.decode(lat(s)).clone() for s in SEEDS}
    del ref
    torch.cuda.empty_cache()
    log("R built")
    O = {}
    for name, bf16, kw, flag in (("fp8_ON", True, {"fp8_convs": True}, True),
                                 ("fp8_OFF", True, {"fp8_convs": True}, False),
                                 ("bf16_ON", True, {}, True),
                                 ("bf16_OFF", True, {}, False)):
        v = build(bf16, kw, flag)
        O[name] = {s: v.decode(lat(s)).clone() for s in SEEDS}
        out[name] = [cmp(O[name][s], R[s]) for s in SEEDS]
        log(name, json.dumps(agg(out[name])))
        if name == "fp8_ON":
            z = lat(0)
            z0 = z.clone()
            for _ in range(10):
                v.decode(z)
            torch.cuda.synchronize()
            c0, g0 = counters(), (len(fc._graphs), len(spc._graphs))
            first, rep, inp = None, True, True
            for _ in range(30):
                inp &= bool(torch.equal(z, z0))
                y = v.decode(z)
                if first is None:
                    first = y.clone()
                elif not torch.equal(y, first):
                    rep = False
            torch.cuda.synchronize()
            c1, g1 = counters(), (len(fc._graphs), len(spc._graphs))
            out["repeat"] = {"bitwise": rep, "inputs_unchanged": inp, "no_recompile": c1 == c0,
                             "no_new_fe_graphs": g1 == g0, "counters": c1, "fe_graphs": list(g1),
                             "subpixel_graphs": spc.graph_table()}
            log("repeat", json.dumps(out["repeat"])[:2000])
        del v
        torch.cuda.empty_cache()
    out["fp8_ON_vs_OFF"] = [cmp(O["fp8_ON"][s], O["fp8_OFF"][s]) for s in SEEDS]
    out["bf16_ON_vs_OFF"] = [cmp(O["bf16_ON"][s], O["bf16_OFF"][s]) for s in SEEDS]

out["summary"] = {k: agg(v) for k, v in out.items() if isinstance(v, list)}
S = out["summary"]
gates = {
    "bf16_on_vs_off_u8_le2": S["bf16_ON_vs_OFF"]["u8_max"] <= 2,
    "bf16_on_vs_off_psnr_ge60": S["bf16_ON_vs_OFF"]["psnr_min"] >= 60,
    "fp8_mean_abs_le_1.05x_off": S["fp8_ON"]["mean_abs_mean"] <= 1.05 * S["fp8_OFF"]["mean_abs_mean"],
    "fp8_psnr_min_ge_off_minus_0.3": S["fp8_ON"]["psnr_min"] >= S["fp8_OFF"]["psnr_min"] - 0.3,
    "repeat_bitwise": out["repeat"]["bitwise"], "inputs_unchanged": out["repeat"]["inputs_unchanged"],
    "no_recompile_after_warmup": out["repeat"]["no_recompile"],
    "no_cudnn_fe_graph_build_after_warmup": out["repeat"]["no_new_fe_graphs"],
}
out["gates"] = gates
log("summary", json.dumps(S), json.dumps(gates))
out["passed"] = bool(all(gates.values()))
print(json.dumps(out))
