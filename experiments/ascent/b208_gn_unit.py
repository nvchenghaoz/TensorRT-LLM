"""Idea 218 / commit point A-ii unit check for fp8_conv.gn_swish_fp8 (GN + swish -> e4m3, s_x).

usage: python experiments/ascent/b208_gn_unit.py   (env B208_ITERS=50)

The 6 GroupNorm input shapes in front of the decoder's FP8 conv sites (N=1, bf16 channels_last,
fixed seed). Gates (all declared before the run):
  G1 q within 1 e4m3 ulp of an fp64 reference of the same formula (|q - ref| <= ulp_e4m3(|ref|));
  G2 s_x rel err vs the fp64 s_x <= 1e-5;
  G3 5x repeat bitwise (eager and compiled), inputs restored from pristine copies, unchanged after;
  G4 0 graph breaks under torch.compile(fullgraph=True);
  G5 compiled within 1 e4m3 ulp of eager.
Rows (not gates): s_x bound / true amax, Inductor kernel count of the compiled op, compiled
gn_swish_fp8 time vs main's compiled group_norm_channel_stats + swish (CUDA events, 10 warmup,
median of B208_ITERS, p10/p90). Last stdout line: {"passed": ...}.
"""

import json
import math
import os
import time

import torch

t_start = time.time()
torch.manual_seed(0)
dev = "cuda"
CL = torch.channels_last
FP8 = torch.float8_e4m3fn
QMAX = 448.0

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (  # noqa: E402
    Flux3GroupNorm,
    group_norm_channel_stats,
    swish,
)

ITERS = int(os.environ.get("B208_ITERS", "50"))
WARM = 10
NREP = 5
OPTS = {"layout_optimization": False, "emulate_precision_casts": True}  # compile_decoder's bf16 options
SHAPES = [("S1", 512, 128), ("S2", 512, 256), ("S3", 512, 512), ("S4", 256, 512), ("S5", 256, 1024),
          ("S6", 128, 1024)]


def emit(d):
    print(json.dumps(d), flush=True)


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def time_fn(f):
    for _ in range(WARM):
        f()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    for s, e in evs:
        s.record()
        f()
        e.record()
    torch.cuda.synchronize()
    v = [s.elapsed_time(e) for s, e in evs]
    return {"med": round(pct(v, 0.5), 4), "p10": round(pct(v, 0.1), 4), "p90": round(pct(v, 0.9), 4)}


def e4m3_ulp(v):
    # |v| in normal range: 2^(floor(log2|v|) - 3); subnormal (< 2^-6): 2^-9
    _, e = torch.frexp(v.clamp_min(2.0 ** -6))  # v = m*2^e, m in [0.5,1) -> floor(log2 v) = e-1
    return torch.ldexp(torch.ones_like(v), e - 4)


def ref64(x, norm):
    n, c, h, w = x.shape
    g = norm.num_groups
    xd = x.double()
    var_c, mean_c = torch.var_mean(xd, dim=(2, 3), correction=0)
    mean_c, var_c = mean_c.view(n, g, c // g), var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd = torch.rsqrt(var_g + norm.eps).expand(n, g, c // g).reshape(n, c)
    mean = mean_g.expand(n, g, c // g).reshape(n, c)
    a = norm.weight.double().view(1, c) * rstd
    b = norm.bias.double().view(1, c) - mean * a
    mn, mx = xd.amin(dim=(2, 3)), xd.amax(dim=(2, 3))
    hi = torch.maximum(a * mn + b, a * mx + b).clamp_min(0.0)
    s_x = (torch.clamp_min((hi * torch.sigmoid(hi)).amax(), fc.SWISH_NEG_BOUND) / QMAX).reshape(1)
    y = xd * a.view(n, c, 1, 1) + b.view(n, c, 1, 1)
    y = y * torch.sigmoid(y)
    true_amax = y.abs().amax()
    q = (y / s_x).clamp(-QMAX, QMAX)
    return q, s_x, true_amax


def kernel_count(f):
    from torch.profiler import ProfilerActivity, profile

    f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        f()
        torch.cuda.synchronize()
    names = [ev.name for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
    return len(names), sorted(set(n[:90] for n in names))


all_ok = True
rows = []
for name, c, hw in SHAPES:
    t0 = time.time()
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    sc = 0.5 + 2.5 * torch.rand(1, c, 1, 1, device=dev)
    off = 0.8 * torch.randn(1, c, 1, 1, device=dev)
    x = (torch.randn(1, c, hw, hw, device=dev) * sc + off).to(torch.bfloat16).contiguous(memory_format=CL)
    norm = Flux3GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True).to(dev)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.3 * torch.randn(c, device=dev))
        norm.weight[::17] *= -1.0  # some negative gammas (the hi_c/lo_c swap)
        norm.bias.copy_(0.2 * torch.randn(c, device=dev))
    norm = norm.to(torch.bfloat16)
    x0 = x.clone()
    w0, b0 = norm.weight.detach().clone(), norm.bias.detach().clone()

    def restore():
        x.copy_(x0)

    def unchanged():
        return torch.equal(x, x0) and torch.equal(norm.weight, w0) and torch.equal(norm.bias, b0)

    with torch.inference_mode():
        # fp64 reference of the same formula
        qr, sr, true_amax = ref64(x, norm)
        # eager: same warmup and iteration count as timing, reset points before the sequence
        restore()
        for _ in range(WARM):
            fc.gn_swish_fp8(x, norm)
        eo = [fc.gn_swish_fp8(x, norm) for _ in range(NREP)]
        for _ in range(ITERS - NREP):
            fc.gn_swish_fp8(x, norm)
        torch.cuda.synchronize()
        e_in = unchanged()
        e_rep = all(torch.equal(q.view(torch.uint8), eo[0][0].view(torch.uint8)) and torch.equal(s, eo[0][1])
                    for q, s in eo[1:])
        qe, se = eo[0]
        layout_ok = qe.dtype == FP8 and qe.is_contiguous(memory_format=CL) and se.shape == (1,) \
            and se.dtype == torch.float32

        fnew = torch.compile(lambda t: fc.gn_swish_fp8(t, norm), fullgraph=True, dynamic=False, options=OPTS)
        fmain = torch.compile(lambda t: swish(group_norm_channel_stats(t, 32, norm.weight, norm.bias, norm.eps)),
                              fullgraph=True, dynamic=False, options=OPTS)
        restore()
        for _ in range(WARM):
            fnew(x)
        co = [fnew(x) for _ in range(NREP)]
        for _ in range(ITERS - NREP):
            fnew(x)
        torch.cuda.synchronize()
        c_in = unchanged()
        c_rep = all(torch.equal(q.view(torch.uint8), co[0][0].view(torch.uint8)) and torch.equal(s, co[0][1])
                    for q, s in co[1:])
        qc, scx = co[0]
        c_layout = qc.dtype == FP8 and qc.is_contiguous(memory_format=CL)
        gb = sum(counters["graph_break"].values())

        # G1 / G2
        def ulp_err(q, ref):
            return float(((q.double() - ref).abs() / e4m3_ulp(ref.abs())).amax())

        g1_eager = ulp_err(qe.float(), qr)
        g1_comp = ulp_err(qc.float(), qr)
        g2_eager = float(((se.double() - sr).abs() / sr).amax())
        g2_comp = float(((scx.double() - sr).abs() / sr).amax())
        g5 = float(((qc.float() - qe.float()).abs() / e4m3_ulp(qe.float().abs())).amax())
        diff_frac = float((qc.view(torch.uint8) != qe.view(torch.uint8)).float().mean())
        sat_frac = float((qe.float().abs() >= QMAX).float().mean())
        del qr

        kn_new, kernels_new = kernel_count(lambda: fnew(x))
        kn_main, kernels_main = kernel_count(lambda: fmain(x))
        restore()
        t_new = time_fn(lambda: fnew(x))
        restore()
        t_main = time_fn(lambda: fmain(x))
        t_in = unchanged()

    ok = (g1_eager <= 1.0 and g1_comp <= 1.0 and g2_eager <= 1e-5 and g2_comp <= 1e-5 and e_rep and c_rep
          and e_in and c_in and t_in and gb == 0 and g5 <= 1.0 and layout_ok and c_layout)
    all_ok &= ok
    row = {"shape": name, "C": c, "HW": hw, "ok": ok,
           "G1_ulp_eager_vs_fp64": round(g1_eager, 4), "G1_ulp_compiled_vs_fp64": round(g1_comp, 4),
           "G2_sx_relerr_eager": g2_eager, "G2_sx_relerr_compiled": g2_comp,
           "G3_eager_repeat_bitwise": e_rep, "G3_compiled_repeat_bitwise": c_rep,
           "inputs_unchanged": e_in and c_in and t_in, "G4_graph_breaks": gb,
           "G5_ulp_compiled_vs_eager": round(g5, 4), "compiled_vs_eager_byte_diff_frac": diff_frac,
           "layout_ok": layout_ok and c_layout, "s_x": float(se), "s_x_fp64": float(sr),
           "true_amax": float(true_amax), "bound_over_true_amax": float(sr * QMAX / true_amax),
           "saturated_frac": sat_frac,
           "kernels_gn_swish_fp8": kn_new, "kernels_main_gn_swish": kn_main,
           "kernel_names_new": kernels_new, "kernel_names_main": kernels_main,
           "t_gn_swish_fp8_ms": t_new, "t_main_gn_swish_bf16_ms": t_main,
           "delta_ms": round(t_new["med"] - t_main["med"], 4), "wall_s": round(time.time() - t0, 1)}
    emit(row)
    rows.append({k: row[k] for k in ("shape", "ok", "G1_ulp_compiled_vs_fp64", "G2_sx_relerr_compiled",
                                     "G5_ulp_compiled_vs_eager", "bound_over_true_amax",
                                     "kernels_gn_swish_fp8", "kernels_main_gn_swish", "delta_ms")}
                | {"new_ms": t_new["med"], "main_ms": t_main["med"]})
    del x, x0, eo, co, qe, qc
    torch.cuda.empty_cache()

print(json.dumps({"passed": bool(all_ok), "rows": rows,
                  "sum_delta_ms_per_shape_once": round(sum(r["delta_ms"] for r in rows), 4),
                  "wall_s": round(time.time() - t_start, 1)}), flush=True)
