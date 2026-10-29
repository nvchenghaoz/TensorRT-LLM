"""Branch 156 diagnostic 2 (not a gate): per-level bias fold, accuracy vs R (compiled fp32
decode) and in-process timing (30 decodes, CUDA events, median/p10/p90), one process:
  M     main code (transpose path disabled)
  C     transpose + fold at both nin levels (as committed)
  NF    transpose, no fold (Inductor materializes x + bias for the nin shortcut)
  F512  fold only into up[1].block[0] (512->256 @512^2)
  F256  fold only into up[0].block[0] (256->128 @1024^2)
usage: python experiments/ascent/b156_diag2.py <checkpoint_root>
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

iv = gn_check.iv
ROOT = sys.argv[1]


def log(*a):
    print(*a, flush=True)


def u8(x):
    return ((x / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)


def stat(a, R):
    d = (a.double() - R.double()).abs()
    mse = ((a.double().clamp(-1, 1) - R.double().clamp(-1, 1)) ** 2).mean().item()
    return {"mean_abs": d.mean().item(), "psnr_db": 10 * torch.log10(torch.tensor(4.0 / mse)).item(),
            "u8_max": (u8(a).int() - u8(R).int()).abs().max().item()}


def latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


@torch.inference_mode()
def run(variant):
    import torch._dynamo

    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

    torch._dynamo.reset()
    tp, cf = iv.Flux3ImageUpsample.__dict__["transpose_path"], iv._can_fold_input_bias
    try:
        if variant == "M":
            iv.Flux3ImageUpsample.transpose_path = staticmethod(lambda x: False)
        elif variant == "NF":
            iv._can_fold_input_bias = lambda *a: False
        elif variant in ("F512", "F256"):
            cin = int(variant[1:])
            iv._can_fold_input_bias = lambda blk, up, x: cf(blk, up, x) and blk.in_channels == cin
        vae, _ = gn_check.build_vae(ROOT)
        if variant != "R":
            set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
        vae.compile_decoder()
        outs = [vae.decode(latent(s)).clone() for s in range(3)]
        z = latent(0)
        for _ in range(10):
            vae.decode(z)
        ts = []
        for _ in range(30):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            vae.decode(z)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
    finally:
        iv.Flux3ImageUpsample.transpose_path = tp
        iv._can_fold_input_bias = cf
    del vae
    torch.cuda.empty_cache()
    return outs, {"median": ts[15], "p10": ts[3], "p90": ts[26]}


R, tR = run("R")
res = {}
for v in ("M", "C", "NF", "F512", "F256", "M"):
    outs, t = run(v)
    key = v if v not in res else v + "_again"
    res[key] = {"ms": t, "psnr": [round(stat(o, r)["psnr_db"], 3) for o, r in zip(outs, R)],
                "mean_abs": [stat(o, r)["mean_abs"] for o, r in zip(outs, R)],
                "u8_max": [stat(o, r)["u8_max"] for o, r in zip(outs, R)]}
    log(json.dumps({key: res[key]}))
print(json.dumps({"passed": True, "diagnostic": True, "fp32_ms": tR, "variants": res}))
