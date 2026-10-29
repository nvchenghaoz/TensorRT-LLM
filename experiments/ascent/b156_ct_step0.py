"""Idea 226 STEP 0: sub-pixel upsample (main: 2x2 conv to 4C + interleave) vs one
stride-2 4x4 conv_transpose2d (cuDNN dgrad), at the 3 real FLUX.3 VAE decoder upsample
shapes, bf16 channels_last, fixed-seed weights (perf does not depend on values).

usage: python experiments/ascent/b156_ct_step0.py [--warmup 10] [--iters 60]
Last stdout line: JSON.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv

p = argparse.ArgumentParser()
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iters", type=int, default=60)
p.add_argument("--shapes", default="up1,up2,up3")
args = p.parse_args()

SHAPES = {"up1": (512, 128), "up2": (512, 256), "up3": (256, 512)}
CL = torch.channels_last


def log(*a):
    print(*a, flush=True)


def transpose_weight(w4):
    """w4 [4C_out, C_in, 2, 2] phase-major (a, b, o) -> Wt [C_in, C_out, 4, 4] (a permutation)."""
    four_c, ci = w4.shape[0], w4.shape[1]
    co = four_c // 4
    w6 = w4.reshape(2, 2, co, ci, 2, 2)  # [a, b, o, i, p, q]
    ap = [(1, 1), (0, 1), (1, 0), (0, 0)]  # k -> (phase, tap)
    wt = w4.new_empty(ci, co, 4, 4)
    for ky, (a, pp) in enumerate(ap):
        for kx, (b, q) in enumerate(ap):
            wt[:, :, ky, kx] = w6[a, b, :, :, pp, q].transpose(0, 1)
    return wt.contiguous()


def timeit(fn):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    return {"median": ts[n // 2], "p10": ts[int(0.1 * n)], "p90": ts[int(0.9 * n) - 1]}


def kernels(fn):
    from collections import Counter

    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name[:150]] += 1
            dur[ev.name[:150]] += ev.device_time_total / 1e3
    return [(k, cnt[k], round(dur[k], 4)) for k in sorted(cnt, key=lambda k: -dur[k])]


def comp(fn):
    return torch.compile(fn, dynamic=False,
                         options={"layout_optimization": False, "emulate_precision_casts": True})


@torch.no_grad()
def run_shape(tag, C, H):
    torch._dynamo.reset()
    g = torch.Generator(device="cuda").manual_seed(1000 + C + H)
    m = iv.Flux3ImageUpsample(C).cuda()
    m.conv.weight.copy_(torch.randn(C, C, 3, 3, device="cuda", generator=g) * (1.0 / (3 * C**0.5)))
    m.conv.bias.copy_(torch.randn(C, device="cuda", generator=g) * 0.1)
    m.refresh_subpixel()  # fp32 masters -> w4 (fp64 sums, one rounding)
    m = m.to(torch.bfloat16).to(memory_format=CL)
    w4, b4 = m.subpixel_weight, m.subpixel_bias
    wt = transpose_weight(w4)
    # value identity: Wt is a permutation of w4
    wt_vals_equal = torch.equal(torch.sort(wt.flatten().float())[0], torch.sort(w4.flatten().float())[0])
    wt_cl = wt.contiguous(memory_format=CL)
    b = m.conv.bias
    x = (torch.randn(1, C, H, H, device="cuda", generator=g)).to(torch.bfloat16).contiguous(memory_format=CL)
    x0 = x.clone()

    main_c = comp(m)
    ct_e = lambda: F.conv_transpose2d(x, wt_cl, None, 2, 1)  # noqa: E731
    ct_e_nclw = lambda: F.conv_transpose2d(x, wt, None, 2, 1)  # noqa: E731
    ct_c = comp(lambda xx, ww: F.conv_transpose2d(xx, ww, None, 2, 1))
    ctb_c = comp(lambda xx, ww, bb: F.conv_transpose2d(xx, ww, bb, 2, 1))
    sub_conv_only = lambda: F.conv2d(x, w4, None, padding=1)  # noqa: E731

    arms = {
        "i_main_compiled": lambda: main_c(x),
        "main_eager": lambda: m(x),
        "main_subpixel_conv_only_eager": sub_conv_only,
        "ii_ct_eager_wcl": ct_e,
        "ii_ct_eager_wnchw": ct_e_nclw,
        "ii_ct_compiled": lambda: ct_c(x, wt_cl),
        "iii_ct_bias_compiled": lambda: ctb_c(x, wt_cl, b),
    }
    rec = {"shape": tag, "C": C, "H": H, "wt_vals_equal_w4": wt_vals_equal}
    outs = {}
    for k, fn in arms.items():
        o = fn()
        torch.cuda.synchronize()
        outs[k] = o
        rec[f"{k}_out_channels_last"] = bool(o.is_contiguous(memory_format=CL))
        rec[f"{k}_ms"] = timeit(fn)
        rec[f"{k}_kernels"] = kernels(fn)
        log(json.dumps({tag: k, "ms": rec[f"{k}_ms"], "kernels": rec[f"{k}_kernels"][:6],
                        "cl": rec[f"{k}_out_channels_last"]}))
    rec["inputs_unchanged"] = torch.equal(x, x0)
    # numerics vs fp64 nearest2x + conv3x3 on fp32 master weights, same bf16 input
    rec["note_ref"] = "fp64 ref uses the bf16-cast conv weights' fp32 masters regenerated with the same seed"
    g2 = torch.Generator(device="cuda").manual_seed(1000 + C + H)
    W = (torch.randn(C, C, 3, 3, device="cuda", generator=g2) * (1.0 / (3 * C**0.5))).double()
    Bv = (torch.randn(C, device="cuda", generator=g2) * 0.1).double()
    r64 = F.conv2d(F.interpolate(x.double(), scale_factor=2.0, mode="nearest"), W, Bv, padding=1)
    bias_v = b.double().view(1, -1, 1, 1)
    errs = {}
    for k in ("i_main_compiled", "main_eager", "ii_ct_eager_wcl", "ii_ct_compiled", "iii_ct_bias_compiled"):
        o = outs[k].double()
        if k.startswith("ii_"):
            o = o + bias_v  # bias added in fp64 for the error comparison only
        d = (o - r64).abs()
        errs[k] = {"max_abs": d.max().item(), "mean_abs": d.mean().item()}
    rec["err_vs_fp64"] = errs
    rec["ct_vs_main_eager_max_abs"] = (outs["iii_ct_bias_compiled"].double()
                                       - outs["main_eager"].double()).abs().max().item()
    rec["repeat_bitwise_ct"] = torch.equal(ct_e(), ct_e())
    s = rec
    rec["saving_ms_i_minus_ii_compiled"] = s["i_main_compiled_ms"]["median"] - s["ii_ct_compiled_ms"]["median"]
    rec["saving_ms_i_minus_iii"] = s["i_main_compiled_ms"]["median"] - s["iii_ct_bias_compiled_ms"]["median"]
    flops = 2 * C * C * 9 * (2 * H) ** 2
    rec["conv3x3_equiv_gflop"] = flops / 1e9
    rec["subpixel_gflop"] = 2 * 4 * C * C * 4 * (H + 1) ** 2 / 1e9
    del outs, main_c, ct_c, ctb_c
    torch.cuda.empty_cache()
    return rec


t0 = time.time()
log("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
res = {"shapes": []}
for tag in args.shapes.split(","):
    C, H = SHAPES[tag]
    res["shapes"].append(run_shape(tag, C, H))
res["summary"] = {r["shape"]: {"main_compiled": r["i_main_compiled_ms"]["median"],
                               "ct_eager": r["ii_ct_eager_wcl_ms"]["median"],
                               "ct_compiled": r["ii_ct_compiled_ms"]["median"],
                               "ct_bias_compiled": r["iii_ct_bias_compiled_ms"]["median"],
                               "subpixel_conv_only": r["main_subpixel_conv_only_eager_ms"]["median"]}
                  for r in res["shapes"]}
res["sum_saving_ms_ii"] = sum(r["saving_ms_i_minus_ii_compiled"] for r in res["shapes"])
res["sum_saving_ms_iii"] = sum(r["saving_ms_i_minus_iii"] for r in res["shapes"])
res["elapsed_s"] = time.time() - t0
res["passed"] = True
log(json.dumps(res))
