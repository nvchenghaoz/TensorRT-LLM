"""Branch 156 (idea 226) checks: Flux3ImageUpsample as one stride-2 4x4 conv_transpose2d
(bf16/fp16 channels_last path) and the upsample-bias fold into the next nin_shortcut.

usage: python experiments/ascent/b156_check.py <checkpoint_root> [unit|decoder|all]
Declared gates (stated before any run, never relaxed):
 unit (eager, real checkpoint weights):
  U1 transpose_weight == W4 values at the mapped index, bitwise, fp32 and bf16 (after
     the module cast), for all 3 decoder upsamples;
  U2 bf16 channels_last: candidate forward vs fp64 reference (nearest2x + conv3x3 on
     the fp32 master weights, same bf16 input): max_abs <= 1.5x main's sub-pixel path
     (same module, transpose path disabled) at 1x64x40x56 (seeded weights) and the 3
     real shapes; repeated calls bitwise; input unchanged; output channels_last;
  U3 fold helper _resnet_block_with_input_bias(block, x_nb, b) vs fp64 block(x_nb + b):
     max_abs <= 1.5x main's block(bf16(x_nb + b)) error vs the same fp64 reference, for
     up[1].block[0] (512->256 @512^2) and up[0].block[0] (256->128 @1024^2).
 decoder (same process):
  A  fp32 decoder (served dtype), seeds 0-2: candidate vs main-emulated (transpose path
     disabled, i.e. main's code) bitwise, eager and compiled;
  R  bf16 compiled candidate: one Dynamo frame, 0 graph breaks, no recompiles after
     warmup across 10 warmup + 60 decodes (all bitwise == iteration 0, latent
     unchanged) and 2 consecutive requests (seed 1 then seed 0 bitwise == first).
Last stdout line: JSON with passed.
"""

import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

iv = gn_check.iv
ROOT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
CL = torch.channels_last
LEVELS = [(3, 512, 128), (2, 512, 256), (1, 256, 512)]  # (up index, C, H)
TAPS = ((1, 1), (0, 1), (1, 0), (0, 0))


def log(*a):
    print(*a, flush=True)


class main_path:
    """Context: disable the transpose path (main's Upsample/decoder code)."""

    def __enter__(self):
        self.orig = iv.Flux3ImageUpsample.__dict__["transpose_path"]  # the staticmethod object
        iv.Flux3ImageUpsample.transpose_path = staticmethod(lambda x: False)

    def __exit__(self, *a):
        iv.Flux3ImageUpsample.transpose_path = self.orig


def wt_matches(m):
    w4, wt = m.subpixel_weight, m.transpose_weight
    C = wt.shape[0]
    w6 = w4.reshape(2, 2, C, C, 2, 2)
    ok = True
    for ky, (a, p) in enumerate(TAPS):
        for kx, (b, q) in enumerate(TAPS):
            ok &= torch.equal(wt[:, :, ky, kx], w6[a, b, :, :, p, q].t())
    ok &= torch.equal(torch.sort(wt.flatten())[0], torch.sort(w4.flatten())[0])
    return bool(ok)


@torch.no_grad()
def unit():
    vae, _ = gn_check.build_vae(ROOT)
    recs, ok_all = [], True
    cases = [("seeded_64@40x56", None, 64, (40, 56))] + [(f"up{l}_{C}@{H}", l, C, (H, H)) for l, C, H in LEVELS]
    for tag, lvl, C, (H, W) in cases:
        m32 = iv.Flux3ImageUpsample(C).cuda()
        if lvl is None:
            g = torch.Generator(device="cuda").manual_seed(7)
            m32.conv.weight.copy_(torch.randn(C, C, 3, 3, device="cuda", generator=g) / (3 * C**0.5))
            m32.conv.bias.copy_(0.1 * torch.randn(C, device="cuda", generator=g))
            m32.refresh_subpixel()
        else:
            m32.load_state_dict(vae.decoder.up[lvl].upsample.state_dict())
        u1_fp32 = wt_matches(m32)
        Wm, Bm = m32.conv.weight.double().clone(), m32.conv.bias.double().clone()
        m = copy.deepcopy(m32).to(torch.bfloat16).to(memory_format=CL)
        u1_bf16 = wt_matches(m)
        g = torch.Generator(device="cuda").manual_seed(100 + C + H)
        x = torch.randn(1, C, H, W, device="cuda", generator=g).to(torch.bfloat16).contiguous(memory_format=CL)
        x0 = x.clone()
        assert m.transpose_path(x)
        yc = m(x)
        yc2 = m(x)
        with main_path():
            ym = m(x)
        r64 = F.conv2d(F.interpolate(x.double(), scale_factor=2.0, mode="nearest"), Wm, Bm, padding=1)
        ec = (yc.double() - r64).abs()
        em = (ym.double() - r64).abs()
        rec = {"case": tag, "U1_wt_eq_w4_fp32": u1_fp32, "U1_wt_eq_w4_bf16": u1_bf16,
               "cand_vs_fp64_max": ec.max().item(), "main_vs_fp64_max": em.max().item(),
               "cand_vs_fp64_mean": ec.mean().item(), "main_vs_fp64_mean": em.mean().item(),
               "cand_vs_main_max": (yc.double() - ym.double()).abs().max().item(),
               "repeat_bitwise": torch.equal(yc, yc2), "inputs_unchanged": torch.equal(x, x0),
               "out_channels_last": yc.is_contiguous(memory_format=CL), "out_shape": list(yc.shape)}
        rec["ok"] = bool(u1_fp32 and u1_bf16 and rec["cand_vs_fp64_max"] <= 1.5 * rec["main_vs_fp64_max"]
                         and rec["repeat_bitwise"] and rec["inputs_unchanged"] and rec["out_channels_last"]
                         and list(yc.shape) == list(ym.shape))
        ok_all &= rec["ok"]
        log(json.dumps(rec))
        recs.append(rec)
        del m, m32, x, x0, yc, yc2, ym, r64, ec, em
        torch.cuda.empty_cache()
    # U3: bias fold helper
    for lvl, C, H in ((1, 512, 512), (0, 256, 1024)):
        blk32 = vae.decoder.up[lvl].block[0]
        b_up32 = vae.decoder.up[lvl + 1].upsample.conv.bias
        blk = copy.deepcopy(blk32).to(torch.bfloat16).to(memory_format=CL)
        blk64 = copy.deepcopy(blk32).double()
        b_up = b_up32.to(torch.bfloat16)
        g = torch.Generator(device="cuda").manual_seed(300 + C)
        x_nb = torch.randn(1, C, H, H, device="cuda", generator=g).to(torch.bfloat16).contiguous(memory_format=CL)
        x0 = x_nb.clone()
        yf = iv._resnet_block_with_input_bias(blk, x_nb, b_up)
        yf2 = iv._resnet_block_with_input_bias(blk, x_nb, b_up)
        ym = blk(x_nb + b_up.view(1, -1, 1, 1))
        r64 = blk64(x_nb.double() + b_up.double().view(1, -1, 1, 1))
        ef, em = (yf.double() - r64).abs(), (ym.double() - r64).abs()
        up_bf16 = copy.deepcopy(vae.decoder.up[lvl + 1].upsample).to(torch.bfloat16)
        decoder_folds_here = bool(up_bf16.fold_bias_into_shortcut)
        up_bf16.fold_bias_into_shortcut = True  # helper math is checked at both levels
        rec = {"case": f"fold_up{lvl}.block0_{C}@{H}", "decoder_folds_here": decoder_folds_here,
               "can_fold": iv._can_fold_input_bias(blk, up_bf16, x_nb),
               "fold_vs_fp64_max": ef.max().item(), "main_vs_fp64_max": em.max().item(),
               "fold_vs_fp64_mean": ef.mean().item(), "main_vs_fp64_mean": em.mean().item(),
               "fold_vs_main_max": (yf.double() - ym.double()).abs().max().item(),
               "repeat_bitwise": torch.equal(yf, yf2), "inputs_unchanged": torch.equal(x_nb, x0)}
        rec["ok"] = bool(rec["can_fold"] and rec["fold_vs_fp64_max"] <= 1.5 * rec["main_vs_fp64_max"]
                         and rec["repeat_bitwise"] and rec["inputs_unchanged"])
        ok_all &= rec["ok"]
        log(json.dumps(rec))
        recs.append(rec)
        del blk, blk64, x_nb, x0, yf, yf2, ym, r64, ef, em
        torch.cuda.empty_cache()
    return {"unit_passed": bool(ok_all), "unit": recs}


def counters():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


def latent(seed, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g).to(dtype)


@torch.inference_mode()
def decoder():
    import torch._dynamo

    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

    res, ok = {}, True
    # A: fp32 (served package dtype), eager then compiled; candidate vs main-emulated
    vae, dt = gn_check.build_vae(ROOT)
    zs = [latent(s, getattr(torch, dt)) for s in range(3)]
    cand = [vae.decode(z).clone() for z in zs]
    with main_path():
        mainv = [vae.decode(z).clone() for z in zs]
    res["A_fp32_eager_bitwise"] = [bool(torch.equal(a, b)) for a, b in zip(cand, mainv)]
    del vae
    torch._dynamo.reset()
    vae, _ = gn_check.build_vae(ROOT)
    vae.compile_decoder()
    cand = [vae.decode(z).clone() for z in zs]
    del vae
    torch._dynamo.reset()
    with main_path():
        vae, _ = gn_check.build_vae(ROOT)
        vae.compile_decoder()
        mainv = [vae.decode(z).clone() for z in zs]
    res["A_fp32_compiled_bitwise"] = [bool(torch.equal(a, b)) for a, b in zip(cand, mainv)]
    res["A_fp32_compiled_max_abs"] = [(a - b).abs().max().item() for a, b in zip(cand, mainv)]
    ok &= all(res["A_fp32_eager_bitwise"]) and all(res["A_fp32_compiled_bitwise"])
    log(json.dumps(res))
    del vae
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    # R: bf16 compiled candidate, repeated execution + Dynamo counters (the counters are
    # process-cumulative and survive dynamo.reset(): clear them so A's fp32 frames do not count)
    import torch._dynamo.utils as dyn_utils

    dyn_utils.counters.clear()
    vae, dt = gn_check.build_vae(ROOT)
    set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    vae.compile_decoder()
    n_up_transpose = 0
    z = latent(0, getattr(torch, dt))
    z0 = z.clone()
    c0 = counters()
    for _ in range(10):
        vae.decode(z)
    torch.cuda.synchronize()
    c_warm = counters()
    first, rep, inp = None, True, True
    for i in range(60):
        inp &= torch.equal(z, z0)
        o = vae.decode(z)
        if first is None:
            first = o.clone()
        elif not torch.equal(o, first):
            rep = False
    vae.decode(latent(1, getattr(torch, dt)))
    again = torch.equal(vae.decode(z), first)
    torch.cuda.synchronize()
    c_end = counters()
    fr = c_end.get("frames", {})
    res.update({"R_repeat_bitwise_60": bool(rep), "R_inputs_unchanged": bool(inp),
                "R_second_request_bitwise": bool(again), "R_counters_before": c0,
                "R_counters_after_warmup": c_warm, "R_counters_end": c_end,
                "R_no_recompile_after_warmup": c_end == c_warm,
                "R_one_frame": fr.get("total", 0) == 1 and fr.get("ok", 0) == 1,
                "R_no_graph_break": not c_end.get("graph_break")})
    ok &= all(res[k] for k in ("R_repeat_bitwise_60", "R_inputs_unchanged", "R_second_request_bitwise",
                               "R_no_recompile_after_warmup", "R_one_frame", "R_no_graph_break"))
    res["decoder_passed"] = bool(ok)
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    if MODE in ("unit", "all"):
        out.update(unit())
    if MODE in ("decoder", "all"):
        out.update(decoder())
    out["passed"] = bool(out.get("unit_passed", True) and out.get("decoder_passed", True))
    out["check"] = "b156 unit U1-U3 + decoder A (fp32 bitwise vs main code, same process) + R (bf16 compiled repeat/dynamo)"
    out["elapsed_s"] = time.time() - t0
    print(json.dumps(out))
