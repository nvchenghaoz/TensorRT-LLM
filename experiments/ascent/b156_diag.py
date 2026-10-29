"""Branch 156 diagnostic (not a gate): which part of the change moves bf16 accuracy vs
R = the compiled fp32 decode (fp32 path = main's code; cross-process bitwise in pair
1686/1687). In one process, bf16-trunk compiled decoders, seeds 0-2:
  M    main code (transpose path disabled)
  C32  branch as committed (transpose + bias fold, fold bias kept in fp32)
  C16  transpose + fold with the fold bias rounded to bf16 (pair 1687's version)
  NF   transpose, no fold (x + bias materialized for the nin shortcut)
usage: python experiments/ascent/b156_diag.py <checkpoint_root>
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

iv = gn_check.iv
ROOT = sys.argv[1]


def log(*a):
    print(*a, flush=True)


def fold_bf16(block, x_nb, b_in):
    x = x_nb + b_in.view(1, -1, 1, 1)
    h = block.norm1.forward_silu(x)
    h = block.conv1(h)
    h = block.norm2.forward_silu(h)
    h = block.conv2(h)
    nin = block.nin_shortcut
    w = nin.weight
    b_fold = (nin.bias.double() + torch.mv(w[:, :, 0, 0].double(), b_in.double())).to(nin.bias.dtype)
    return F.conv2d(x_nb, w, b_fold) + h


def u8(x):
    return ((x / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)


def stat(a, R):
    d = (a.double() - R.double()).abs()
    mse = ((a.double().clamp(-1, 1) - R.double().clamp(-1, 1)) ** 2).mean().item()
    return {"mean_abs": d.mean().item(), "max_abs": d.max().item(),
            "psnr_db": 10 * torch.log10(torch.tensor(4.0 / mse)).item(),
            "u8_max": (u8(a).int() - u8(R).int()).abs().max().item()}


def latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


@torch.inference_mode()
def run(variant):
    import torch._dynamo

    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

    torch._dynamo.reset()
    saved = (iv.Flux3ImageUpsample.__dict__["transpose_path"], iv._resnet_block_with_input_bias,
             iv._can_fold_input_bias)
    try:
        if variant == "M":
            iv.Flux3ImageUpsample.transpose_path = staticmethod(lambda x: False)
        elif variant == "C16":
            iv._resnet_block_with_input_bias = fold_bf16
        elif variant == "NF":
            iv._can_fold_input_bias = lambda *a: False
        vae, _ = gn_check.build_vae(ROOT)
        if variant != "R":
            set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
        vae.compile_decoder()
        outs = [vae.decode(latent(s)).clone() for s in range(3)]
        outs2 = [vae.decode(latent(s)) for s in range(3)]
        rep = all(torch.equal(a, b) for a, b in zip(outs, outs2))
    finally:
        iv.Flux3ImageUpsample.transpose_path = saved[0]
        iv._resnet_block_with_input_bias = saved[1]
        iv._can_fold_input_bias = saved[2]
    del vae
    torch.cuda.empty_cache()
    return outs, rep


R, _ = run("R")
res = {}
for v in ("M", "C32", "C16", "NF"):
    outs, rep = run(v)
    res[v] = {"repeat_bitwise": rep, "per_seed": [stat(o, r) for o, r in zip(outs, R)]}
    log(json.dumps({v: res[v]}))
summary = {v: {"psnr": [round(s["psnr_db"], 3) for s in r["per_seed"]],
               "mean_abs": [round(s["mean_abs"], 7) for s in r["per_seed"]]} for v, r in res.items()}
print(json.dumps({"passed": True, "diagnostic": True, "summary": summary, "detail": res}))
