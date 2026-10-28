"""Branch 31 / idea 58 correctness: tensor-core mid-block attention in the compiled decoder.

One process, real checkpoint VAE, latents 1x128x64x64 (1024^2 output), seeds 0..2:
  new  = this tree's compiled decoder (Flux3ImageAttnBlock._attention_tc on CUDA)
  main = the same compiled decoder with the attention block's original SDPA code
         (copied verbatim from main), patched onto the instance before compile
  fp64 = eager decoder in float64 with the original SDPA attention (protected math ref)
Reports max/mean abs, uint8 diff and PSNR (on [-1,1] images) for new and main vs fp64 and
new vs main; an untimed repeat sequence (warmup + iters decodes, bitwise == iteration 0,
input latent unchanged); dynamo frame / graph-break / recompile counters before and after.
--dtype bf16 applies main's served trunk recipe (set_vae_decoder_trunk_dtype, conv_out fp32)
to the new and main arms; the fp64 reference stays fully fp64.
Declared tolerance (precision class): PSNR(new vs fp64) >= 40 dB and within 3 dB of main's.
"""

import argparse
import json
import math
import os
import sys
import types

import torch
import torch.nn as nn
from einops import rearrange

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--dtype", default="served")
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=60)
p.add_argument("--seeds", default="0,1,2")
args = p.parse_args()


def log(*a):
    print(*a, flush=True)


def main_attention(self, h_):
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    b, c, h, w = q.shape
    q = rearrange(q, "b c h w -> b 1 (h w) c")
    k = rearrange(k, "b c h w -> b 1 (h w) c")
    v = rearrange(v, "b c h w -> b 1 (h w) c")
    h_ = nn.functional.scaled_dot_product_attention(q, k, v)

    return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)


def frames():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


def build(kind):
    vae, dt = gn_check.build_vae(args.root)
    if args.dtype == "bf16" and kind != "fp64":
        # main's served recipe (engine.json vae_decode_dtype=bfloat16): trunk bf16, conv_out fp32
        from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    if kind in ("main", "fp64"):
        a = vae.decoder.mid.attn_1
        a.attention = types.MethodType(main_attention, a)
    if kind == "fp64":
        vae = vae.double()
    else:
        vae.compile_decoder()
    return vae


def dec_dtype(vae):
    return next(vae.decoder.mid.parameters()).dtype


@torch.inference_mode()
def decode(vae, z):
    # as the pipeline: latent in the VAE package dtype (trunk casts happen in hooks)
    return vae.decode(z.to(next(vae.parameters()).dtype)).double()


def cmp(a, b):
    d = (a - b).abs()
    ua, ub = gn_check.to_u8(a.float()).int(), gn_check.to_u8(b.float()).int()
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "u8_max_diff": (ua - ub).abs().max().item(), "u8_frac_diff": (ua != ub).float().mean().item(),
            "psnr_db": 10 * math.log10(4.0 / mse) if mse > 0 else float("inf")}


seeds = [int(s) for s in args.seeds.split(",")]
lat = {}
for s in seeds:
    g = torch.Generator(device="cuda").manual_seed(s)
    lat[s] = torch.randn(1, 128, 64, 64, device="cuda", generator=g)

log("building fp64 reference")
v64 = build("fp64")
ref = {s: decode(v64, lat[s]).clone() for s in seeds}
del v64
torch.cuda.empty_cache()

log("building main-attention compiled decoder")
vm = build("main")
outm = {s: decode(vm, lat[s]).clone() for s in seeds}
del vm
torch.cuda.empty_cache()

torch._dynamo.utils.counters.clear()
log("building new compiled decoder")
vn = build("new")
log("decoder dtype:", dec_dtype(vn))
outn = {s: decode(vn, lat[s]).clone() for s in seeds}
f_first = frames()
log("frames after first decodes:", f_first)

# untimed repeat sequence: same warmup + iteration count as the timed bench; two passes
# (two consecutive 'requests') compared with each other and with the first decode
z = lat[seeds[0]].clone()
z0 = z.clone()
rep, inp = True, True
for _pass in range(2):
    for i in range(args.warmup + args.iters):
        inp &= torch.equal(z, z0)
        o = decode(vn, z)
        if not torch.equal(o, outn[seeds[0]]):
            rep = False
    log(f"repeat pass {_pass} done, bitwise so far: {rep}")
torch.cuda.synchronize()
f_after = frames()

res = {"decoder_dtype": str(dec_dtype(vn)), "per_seed": {}}
ok_acc = True
for s in seeds:
    r = {"new_vs_fp64": cmp(outn[s], ref[s]), "main_vs_fp64": cmp(outm[s], ref[s]),
         "new_vs_main": cmp(outn[s], outm[s])}
    ok_acc &= r["new_vs_fp64"]["psnr_db"] >= 40 and r["new_vs_fp64"]["psnr_db"] >= r["main_vs_fp64"]["psnr_db"] - 3
    res["per_seed"][str(s)] = r
    log(f"seed {s}: {json.dumps(r)}")
no_gb = not f_after.get("graph_break")
one_frame = f_first.get("frames", {}).get("total") == 1
passed = rep and inp and ok_acc and f_first == f_after and no_gb and one_frame
res.update({"passed": bool(passed),
            "compared": "compiled decoder with tensor-core attention vs fp64 eager decoder (original SDPA) "
                        "and vs the compiled decoder with main's SDPA attention; real weights",
            "class": "precision", "repeat_bitwise_2x(warmup+iters)": rep, "inputs_unchanged": inp,
            "accuracy_ok": bool(ok_acc), "one_dynamo_frame": one_frame, "no_graph_breaks": no_gb,
            "no_recompile_after_first": f_first == f_after, "frames_first": f_first, "frames_after": f_after})
log(json.dumps(res))
