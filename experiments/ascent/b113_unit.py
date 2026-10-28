"""Branch 113 (idea 52): unit test + config sweep of the fused VAE decoder tail kernel.

usage: python experiments/ascent/b113_unit.py [unit|sweep|all]

unit: conv_out_3x3 (M2) and norm_swish_conv_out_3x3 (M3) vs the module path (cuDNN fp32
  conv with the default TF32 setting, + bias; M3: fp32 affine + swish then that conv) and
  vs an fp64 reference, at 1x128x1024x1024 (NHWC and NCHW) and 1x128x40x56 (NHWC, NCHW).
  Tolerances declared before the run (fp32 FMA over 1152 terms):
    kernel vs fp64: max_abs <= 1e-4 and mean_abs <= module-path mean_abs vs fp64;
    kernel vs module path: max_abs <= 1e-2 (the module path is TF32).
  Repeated execution: 20 calls, every output bitwise == call 0, inputs unchanged.
sweep: CUDA-event median (50 iters) of tile configs at the served NHWC 1024^2 shape,
  next to the module path (conv only, and conv + the fp32 cast for M3's comparison).
Last stdout line: JSON.
"""

import itertools
import json
import sys

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import decoder_tail_triton as dt


def log(*a):
    print(*a, flush=True)


def make(C, H, W, cl, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    raw = (torch.randn(1, C, H, W, device="cuda", generator=g) * 1.3 + 0.2).to(torch.bfloat16)
    scale = (0.8 + 0.4 * torch.rand(C, device="cuda", generator=g)).float()
    shift = (0.2 * torch.randn(C, device="cuda", generator=g)).float()
    w = (torch.randn(3, C, 3, 3, device="cuda", generator=g) * 0.03).float()
    b = (0.05 * torch.randn(3, device="cuda", generator=g)).float()
    if cl:
        raw = raw.contiguous(memory_format=torch.channels_last)
        w = w.contiguous(memory_format=torch.channels_last)
    # M2 input: main's conv_out input = bf16 swish(GN) cast to fp32.
    a = raw.float() * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    x = (a * torch.sigmoid(a)).to(torch.bfloat16).float()
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)
    return raw, scale, shift, w, b, x


def err(a, r):
    d = (a.double() - r).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


@torch.inference_mode()
def unit():
    cases, ok_all = [], True
    for (C, H, W), cl in itertools.product([(128, 1024, 1024), (128, 40, 56)], [True, False]):
        raw, scale, shift, w, b, x = make(C, H, W, cl)
        # ---- M2
        mod = F.conv2d(x, w, b, padding=1)
        r64 = F.conv2d(x.double(), w.double(), b.double(), padding=1)
        x0 = x.clone()
        outs = [dt.conv_out_3x3(x, w, b) for _ in range(20)]
        torch.cuda.synchronize()
        rep = all(torch.equal(o, outs[0]) for o in outs) and torch.equal(x, x0)
        o = outs[0]
        layout_ok = o.stride() == mod.stride() and o.shape == mod.shape and o.dtype == mod.dtype
        k64, m64, km = err(o, r64), err(mod, r64), err(o, mod.double())
        ok = (k64["max_abs"] <= 1e-4 and k64["mean_abs"] <= m64["mean_abs"] and km["max_abs"] <= 1e-2
              and rep and layout_ok)
        rec = {"op": "M2 conv_out_3x3", "shape": [1, C, H, W], "nhwc": cl, "kernel_vs_fp64": k64,
               "module_vs_fp64": m64, "kernel_vs_module": km, "repeat_bitwise_inputs_unchanged": rep,
               "out_stride": list(o.stride()), "module_stride": list(mod.stride()), "ok": bool(ok)}
        log(json.dumps(rec))
        cases.append(rec)
        ok_all &= bool(ok)
        del outs, mod, r64
        # ---- M3
        a = raw.float() * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
        xm = a * torch.sigmoid(a)  # module path: fp32 affine + swish, then conv
        mod = F.conv2d(xm, w, b, padding=1)
        a64 = raw.double() * scale.double().view(1, -1, 1, 1) + shift.double().view(1, -1, 1, 1)
        r64 = F.conv2d(a64 * torch.sigmoid(a64), w.double(), b.double(), padding=1)
        main_like = F.conv2d(x, w, b, padding=1)  # main: bf16-rounded swish, then TF32 conv
        raw0 = raw.clone()
        outs = [dt.norm_swish_conv_out_3x3(raw, scale, shift, w, b) for _ in range(20)]
        torch.cuda.synchronize()
        rep = all(torch.equal(o, outs[0]) for o in outs) and torch.equal(raw, raw0)
        o = outs[0]
        k64, m64, km = err(o, r64), err(mod, r64), err(o, mod.double())
        main64 = err(main_like, r64)
        ok = (k64["max_abs"] <= 1e-4 and k64["mean_abs"] <= m64["mean_abs"] and km["max_abs"] <= 1e-2
              and rep and o.stride() == mod.stride())
        rec = {"op": "M3 norm_swish_conv_out_3x3", "shape": [1, C, H, W], "nhwc": cl,
               "kernel_vs_fp64": k64, "module_vs_fp64": m64, "kernel_vs_module": km,
               "main_bf16_swish_tf32_vs_fp64": main64, "repeat_bitwise_inputs_unchanged": rep,
               "ok": bool(ok)}
        log(json.dumps(rec))
        cases.append(rec)
        ok_all &= bool(ok)
        del outs, mod, r64, a64, a, xm, main_like
        torch.cuda.empty_cache()
    # ---- full tail: GroupNorm(32, eps 1e-6) stats -> fused affine+swish+conv (M3 as wired
    # in Flux3ImageDecoder) vs main's math (bf16 GN apply+swish, fp32 cast, cuDNN conv)
    # and vs fp64 GroupNorm -> swish -> conv.
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv

    for (C, H, W), cl in itertools.product([(128, 1024, 1024), (128, 40, 56)], [True, False]):
        xs, _, _, w, b, _ = make(C, H, W, cl, seed=7)
        rr = make(C, H, W, cl, seed=8)[0]
        g = torch.Generator(device="cuda").manual_seed(11)
        gw = (1 + 0.1 * torch.randn(C, device="cuda", generator=g)).to(torch.bfloat16)
        gb = (0.1 * torch.randn(C, device="cuda", generator=g)).to(torch.bfloat16)
        rb = (0.1 * torch.randn(C, device="cuda", generator=g)).to(torch.bfloat16)
        raw = xs + (rr + rb.view(1, -1, 1, 1))  # conv2 bias add, then residual add (bf16)
        sc, sh = dt.group_norm_channel_affine(raw, 32, gw, gb, 1e-6)
        o = dt.add_norm_swish_conv_out_3x3(xs, rr, rb, sc, sh, w, b)
        o_nores = dt.norm_swish_conv_out_3x3(raw, sc, sh, w, b)
        res_bitwise = torch.equal(o, o_nores)
        y = iv.group_norm_channel_stats(raw, 32, gw, gb, 1e-6)
        main_like = F.conv2d((y * torch.sigmoid(y)).float(), w, b, padding=1)
        y64 = F.group_norm(raw.double(), 32, gw.double(), gb.double(), 1e-6)
        r64 = F.conv2d(y64 * torch.sigmoid(y64), w.double(), b.double(), padding=1)
        k64, m64 = err(o, r64), err(main_like, r64)
        ok = (k64["max_abs"] <= 1e-3 and k64["mean_abs"] <= m64["mean_abs"] and o.stride() == main_like.stride()
              and res_bitwise)
        rec = {"op": "M3 full tail (x + r, GN stats, fused add+affine+swish+conv kernel)", "shape": [1, C, H, W],
               "nhwc": cl, "residual_op_bitwise_eq_materialized_sum_op": res_bitwise,
               "kernel_vs_fp64": k64, "main_math_vs_fp64": m64,
               "kernel_vs_main_math": err(o, main_like.double()), "ok": bool(ok)}
        log(json.dumps(rec))
        cases.append(rec)
        ok_all &= bool(ok)
        del raw, xs, rr, o, o_nores, y, main_like, y64, r64
        torch.cuda.empty_cache()
    return {"passed": bool(ok_all), "n_cases": len(cases),
            "check": "decoder_tail_triton M2/M3 vs module path (cuDNN TF32) and fp64 reference"}


def bench(fn, iters=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"median": ts[len(ts) // 2], "p10": ts[len(ts) // 10], "p90": ts[(9 * len(ts)) // 10]}


@torch.inference_mode()
def sweep():
    raw, scale, shift, w, b, x = make(128, 1024, 1024, True)
    res = {"module_conv_fp32": bench(lambda: F.conv2d(x, w, b, padding=1)),
           "module_cast_then_conv": bench(lambda: F.conv2d(raw.float(), w, b, padding=1))}
    log("module", json.dumps(res))
    best = {}
    r64 = F.conv2d(x.double(), w.double(), b.double(), padding=1)
    for prec in ("tf32x3", "tf32"):
        dt._PRECISION = prec
        res[f"err_{prec}_vs_fp64"] = err(dt.conv_out_3x3(x, w, b), r64)
        log(prec, json.dumps(res[f"err_{prec}_vs_fp64"]))
    res["err_module_vs_fp64"] = err(F.conv2d(x, w, b, padding=1), r64)
    del r64
    rr = make(128, 1024, 1024, True, seed=8)[0]
    b0 = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    space = [("tf32x3",) + c for c in itertools.product([True, False], [64, 128], [8, 16, 32], [32, 64], [4, 8])]
    for prec, ga, BW, R, BC, nw in space:
        ns = 2
        if BW * BC > 16384 or (not ga and nw == 8):
            continue
        dt._PRECISION, dt._BW, dt._R, dt._BC, dt._NUM_WARPS, dt._NUM_STAGES = prec, BW, R, BC, nw, ns
        dt._GATHER = ga
        key = f"{prec}_{'gather' if ga else 'shift'}_BW{BW}_R{R}_BC{BC}_w{nw}_s{ns}"
        try:
            t2 = bench(lambda: dt.conv_out_3x3(x, w, b), iters=20, warm=2)["median"]
            t3 = bench(lambda: dt.norm_swish_conv_out_3x3(raw, scale, shift, w, b), iters=20, warm=2)["median"]
            t3r = bench(lambda: dt.add_norm_swish_conv_out_3x3(raw, rr, b0, scale, shift, w, b), iters=20,
                        warm=2)["median"]
        except Exception as ex:  # noqa: BLE001
            log(key, "failed", repr(ex)[:200])
            continue
        res[key] = {"M2": t2, "M3": t3, "M3r": t3r}
        log(key, f"M2 {t2:.4f} ms  M3 {t3:.4f} ms  M3r {t3r:.4f} ms")
    cfgs = [k for k in res if k.startswith(("tf32", "ieee"))]
    for m in ("M2", "M3", "M3r"):
        best[m] = min(cfgs, key=lambda k: res[k][m])
    return {"sweep": res, "best": best, **{f"best_{m}_ms": res[best[m]][m] for m in best}}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if mode in ("unit", "all"):
        out["unit"] = unit()
    if mode in ("sweep", "all"):
        out["sweep"] = sweep()
    out["passed"] = out.get("unit", {"passed": True})["passed"]
    print(json.dumps(out))
