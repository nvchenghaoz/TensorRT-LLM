"""Idea 208 / commit point A-i unit check for flux3_vae::fp8_conv3x3 + prepare_fp8_convs.

usage: python experiments/ascent/b208_unit.py      (env B208_SHAPES=S5,S6 subset, B208_ITERS=50)

Per cuDNN key (6 shapes x {sb, sbr, s} as the decoder has them):
  * plan chosen (fixed eng7 k24 from #1431) logged, every plan timed once (table), fixed plan
    timed (median of B208_ITERS CUDA-event iterations) next to aten bf16 conv (main's op).
  * correctness: out vs fp32 conv (TF32 off) on the dequantized operands, + bias + res in fp32:
    |out-ref| <= 2 bf16 ulps of max(|ref|, 2^-10 max|ref|); also reported vs fp64.
  * untimed sequence (same warmup and iteration count as timing, same reset points):
    5 calls bitwise identical to call 0, inputs restored from pristine copies before timing and
    before correctness and unchanged after.
  * torch.compile(fullgraph=True) with 0 graph breaks, compiled == eager bitwise.
prepare_fp8_convs on a synthetic module of Flux3ImageResnetBlocks: KRSC e4m3 weights,
state_dict keys/values unchanged, no-op on fp32 and on non-channels_last.
Last stdout line: {"passed": ...}.
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

t_start = time.time()
torch.manual_seed(0)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
CL = torch.channels_last
FP8 = torch.float8_e4m3fn
QMAX = 448.0

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (  # noqa: E402
    Flux3ImageResnetBlock,
)

ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b208")
os.makedirs(ART, exist_ok=True)
LOG = open(os.path.join(ART, "b208_unit.jsonl"), "w")


def emit(d):
    s = json.dumps(d)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


import cudnn  # noqa: E402

emit({"versions": {"cudnn_backend": cudnn.backend_version(), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name()}})

SHAPES = [  # name, cin, cout, hw, epis
    ("S1", 512, 512, 128, ["sb", "sbr"]),
    ("S2", 512, 512, 256, ["sb", "sbr"]),
    ("S3", 512, 256, 512, ["sb"]),
    ("S4", 256, 256, 512, ["sb", "sbr"]),
    ("S5", 256, 128, 1024, ["sb"]),
    ("S6", 128, 128, 1024, ["sb", "sbr", "s"]),
]
only = os.environ.get("B208_SHAPES")
if only:
    SHAPES = [s for s in SHAPES if s[0] in only.split(",")]
ITERS = int(os.environ.get("B208_ITERS", "50"))
WARM = 5
NREP = 5


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def time_fn(f, iters=ITERS, warm=WARM):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in evs:
        s.record()
        f()
        e.record()
    torch.cuda.synchronize()
    v = [s.elapsed_time(e) for s, e in evs]
    return {"med": pct(v, 0.5), "p10": pct(v, 0.1), "p90": pct(v, 0.9)}


def bf16_ulp(v):
    # v > 0: v = m*2^e, m in [0.5,1) -> bf16 ulp = 2^(e-8)
    _, e = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), e - 8)


all_ok = True
table = []

for name, cin, cout, hw, epis in SHAPES:
    t0 = time.time()
    x = (torch.randn(1, cin, hw, hw, device=dev) * 1.5 + 0.3)
    h_bf = (x * torch.sigmoid(x)).to(torch.bfloat16).contiguous(memory_format=CL)
    s_x = (h_bf.float().abs().amax() / QMAX).reshape(1).contiguous()
    xq = torch.div(h_bf.float(), s_x).clamp(-QMAX, QMAX).to(FP8).contiguous(memory_format=CL)
    w = (torch.randn(cout, cin, 3, 3, device=dev) / math.sqrt(9 * cin)).to(torch.bfloat16)
    w = w.contiguous(memory_format=CL)
    wq, s_w = fc.quantize_conv3x3_weight(w)
    bias = (0.1 * torch.randn(cout, device=dev)).to(torch.bfloat16)
    res = torch.randn(1, cout, hw, hw, device=dev).to(torch.bfloat16).contiguous(memory_format=CL)
    inputs = {"xq": xq, "s_x": s_x, "wq": wq, "s_w": s_w, "bias": bias, "res": res, "h_bf": h_bf, "w": w}
    pristine = {k: v.clone() for k, v in inputs.items()}

    def restore():
        for k, v in inputs.items():
            v.copy_(pristine[k])

    def unchanged():
        return all(torch.equal(v.view(torch.uint8) if v.dtype == FP8 else v,
                               pristine[k].view(torch.uint8) if v.dtype == FP8 else pristine[k])
                   for k, v in inputs.items())

    # fp32 / fp64 references of the dequantized operands (shared conv, epilogue per key)
    xd = xq.float() * s_x
    wd = wq.permute(0, 3, 1, 2).float() * s_w.view(-1, 1, 1, 1)
    conv32 = F.conv2d(xd.contiguous(), wd.contiguous(), None, padding=1)
    conv64 = F.conv2d(xd.double().contiguous(), wd.double().contiguous(), None, padding=1)
    del xd, wd

    for epi in epis:
        b = bias if "b" in epi else None
        r = res if "r" in epi else None

        def call():
            return torch.ops.flux3_vae.fp8_conv3x3(xq, s_x, wq, s_w, b, r)

        restore()
        out0 = call()  # build (warmup) call
        torch.cuda.synchronize()
        key = (1, cin, cout, hw, hw, b is not None, r is not None)
        gr = fc._graphs[key]
        # every plan timed once on a scratch output (does not change the fixed plan)
        scratch = torch.empty_like(out0)
        plan_times = gr.autotune(gr.variant_pack(xq, s_x, wq, s_w, b, r, scratch))
        del scratch
        # timing, fixed plan, vs aten bf16 conv (main's op: conv with bias; +res add for sbr)
        restore()
        t_fp8 = time_fn(call)
        if r is not None:
            t_aten = time_fn(lambda: F.conv2d(h_bf, w, b, padding=1).add_(r))
        else:
            t_aten = time_fn(lambda: F.conv2d(h_bf, w, b, padding=1))
        # untimed correctness sequence: same warmup and iteration count, same reset points
        restore()
        for _ in range(WARM):
            call()
        outs = [call() for _ in range(max(NREP, 1))]
        for _ in range(ITERS - NREP):
            call()
        torch.cuda.synchronize()
        inputs_ok = unchanged()
        rep_ok = all(torch.equal(o, outs[0]) for o in outs[1:]) and torch.equal(outs[0], out0)
        ref32 = conv32.clone()
        ref64 = conv64.clone()
        if b is not None:
            ref32 += b.float().view(1, -1, 1, 1)
            ref64 += b.double().view(1, -1, 1, 1)
        if r is not None:
            ref32 += r.float()
            ref64 += r.double()
        o = outs[0].float()
        mx = ref32.abs().amax()
        mag = torch.maximum(ref32.abs(), mx * 2.0 ** -10)
        tol = 2 * bf16_ulp(mag)
        err = (o - ref32).abs()
        tol_ok = bool((err <= tol).all())
        worst = float((err / tol).amax())
        err64 = float((o.double() - ref64).abs().amax())
        layout_ok = outs[0].is_contiguous(memory_format=CL) and outs[0].dtype == torch.bfloat16
        ok = tol_ok and rep_ok and inputs_ok and layout_ok
        all_ok &= ok
        row = {"shape": name, "epi": epi, **gr.describe(), "fp8_ms": t_fp8, "aten_bf16_ms": t_aten,
               "speedup": t_aten["med"] / t_fp8["med"], "max_abs_err_vs_fp32": float(err.amax()),
               "max_err_over_tol": worst, "max_abs_err_vs_fp64": err64, "max_abs_ref": float(mx),
               "tol_ok": tol_ok, "repeat_bitwise": rep_ok, "inputs_unchanged": inputs_ok,
               "layout_ok": layout_ok, "ok": ok,
               "plan_times_ms": {k: round(v, 4) for k, v in plan_times.items()}}
        emit(row)
        table.append({k: row[k] for k in ("shape", "epi", "plan", "plan_name", "build_s", "speedup", "ok")}
                     | {"fp8_med": round(t_fp8["med"], 4), "aten_med": round(t_aten["med"], 4)})
        del outs, out0, ref32, ref64, o, err, tol, mag
    del conv32, conv64
    torch.cuda.empty_cache()
    print(f"--- {name} done in {time.time() - t0:.1f}s", flush=True)

# ---------------------------------------------------------------- torch.compile fullgraph
compile_ok = False
try:
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    cin, cout, hw = 128, 128, 1024
    hx = torch.randn(1, cin, hw, hw, device=dev)
    hx = (hx * torch.sigmoid(hx)).to(torch.bfloat16).contiguous(memory_format=CL)
    sx = (hx.float().abs().amax() / QMAX).reshape(1)
    w = (torch.randn(cout, cin, 3, 3, device=dev) / math.sqrt(9 * cin)).to(torch.bfloat16)
    wq, s_w = fc.quantize_conv3x3_weight(w)
    bias = (0.1 * torch.randn(cout, device=dev)).to(torch.bfloat16)
    res = torch.randn(1, cout, hw, hw, device=dev).to(torch.bfloat16).contiguous(memory_format=CL)

    q = torch.div(hx.float(), sx).clamp(-QMAX, QMAX).to(FP8).contiguous(memory_format=CL)

    def f(q, sx, wq, s_w, bias, res):
        y = torch.ops.flux3_vae.fp8_conv3x3(q, sx, wq, s_w, bias, res)
        y2 = torch.ops.flux3_vae.fp8_conv3x3(q, sx, wq, s_w, None, None)
        return y, y2, (y.float() + y2.float()).sum(dim=(2, 3))

    eager = f(q, sx, wq, s_w, bias, res)
    cf = torch.compile(f, fullgraph=True, dynamic=False)
    comp = cf(q, sx, wq, s_w, bias, res)
    comp2 = cf(q, sx, wq, s_w, bias, res)
    torch.cuda.synchronize()
    gb = sum(counters["graph_break"].values())
    same = torch.equal(eager[0], comp[0]) and torch.equal(eager[1], comp[1])
    same2 = all(torch.equal(a, b) for a, b in zip(comp, comp2))
    strides = comp[0].stride() == eager[0].stride()
    compile_ok = gb == 0 and same and same2 and strides
    emit({"compile_fullgraph": True, "graph_breaks": gb, "compiled_eq_eager_bitwise": same,
          "compiled_repeat_bitwise": same2, "strides_match": strides, "ok": compile_ok})
except Exception as e:  # noqa: BLE001
    emit({"compile_fullgraph": True, "error": repr(e)[:800], "ok": False})
all_ok &= compile_ok

# ---------------------------------------------------------------- prepare_fp8_convs
prep_ok = False
try:
    class Synth(nn.Module):
        def __init__(self):
            super().__init__()
            self.mid = nn.Module()
            self.mid.block_1 = Flux3ImageResnetBlock(256, 256)
            self.up = nn.ModuleList([nn.Module()])
            self.up[0].block = nn.ModuleList([Flux3ImageResnetBlock(256, 128), Flux3ImageResnetBlock(128, 128)])
            self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    m32 = Synth().to(dev)
    n32 = fc.prepare_fp8_convs(m32)  # fp32 -> no-op
    mbf_nocl = Synth().to(dev).to(torch.bfloat16)
    n_nocl = fc.prepare_fp8_convs(mbf_nocl)  # bf16 but not channels_last -> no-op
    m = Synth().to(dev).to(torch.bfloat16).to(memory_format=CL)
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    n = fc.prepare_fp8_convs(m)
    sd1 = m.state_dict()
    keys_same = list(sd0.keys()) == list(sd1.keys())
    vals_same = all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    conv_checks = []
    blocks = [m.mid.block_1, m.up[0].block[0], m.up[0].block[1]]
    for blk in blocks:
        for c in (blk.conv1, blk.conv2):
            wq, sw = c.fp8_wq, c.fp8_sw
            W = c.weight.float()
            deq = wq.permute(0, 3, 1, 2).float() * sw.view(-1, 1, 1, 1)
            bound = W.abs() * 2.0 ** -4 + sw.view(-1, 1, 1, 1) * 2.0 ** -9 * 1.0001
            conv_checks.append(
                wq.dtype == FP8 and tuple(wq.shape) == (c.out_channels, 3, 3, c.in_channels)
                and wq.is_contiguous() and sw.dtype == torch.float32 and tuple(sw.shape) == (c.out_channels,)
                and bool(((deq - W).abs() <= bound).all())
                and torch.allclose(sw, W.abs().amax(dim=(1, 2, 3)) / QMAX)
                and float(wq.float().abs().amax()) <= QMAX)
    fp8_flags = all(getattr(b, "_fp8", False) for b in blocks)
    no_flag_32 = not any(getattr(b, "_fp8", False) for b in [m32.mid.block_1])
    m.to(torch.bfloat16)  # plain attributes survive dtype casts
    still_fp8 = m.mid.block_1.conv1.fp8_wq.dtype == FP8
    # a prepared conv through the op matches the dequantized fp32 conv
    c = m.up[0].block[1].conv1
    hx = torch.randn(1, 128, 64, 64, device=dev)
    hx = (hx * torch.sigmoid(hx)).to(torch.bfloat16).contiguous(memory_format=CL)
    sx = (hx.float().abs().amax() / QMAX).reshape(1)
    q = torch.div(hx.float(), sx).clamp(-QMAX, QMAX).to(FP8).contiguous(memory_format=CL)
    y = torch.ops.flux3_vae.fp8_conv3x3(q, sx, c.fp8_wq, c.fp8_sw, c.bias, None)
    ref = F.conv2d(q.float() * sx, c.fp8_wq.permute(0, 3, 1, 2).float() * c.fp8_sw.view(-1, 1, 1, 1),
                   c.bias.float(), padding=1)
    mag = torch.maximum(ref.abs(), ref.abs().amax() * 2.0 ** -10)
    op_ok = bool(((y.float() - ref).abs() <= 2 * bf16_ulp(mag)).all())
    prep_ok = (n32 == 0 and n_nocl == 0 and n == 6 and keys_same and vals_same and all(conv_checks)
               and fp8_flags and no_flag_32 and still_fp8 and op_ok)
    emit({"prepare_fp8_convs": True, "n_fp32": n32, "n_not_cl": n_nocl, "n_prepared": n,
          "state_dict_keys_same": keys_same, "state_dict_values_same": vals_same,
          "conv_checks": conv_checks, "block_fp8_flags": fp8_flags, "fp8_survives_to_bf16": still_fp8,
          "prepared_conv_op_ok": op_ok, "ok": prep_ok})
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    emit({"prepare_fp8_convs": True, "error": repr(e)[:800], "ok": False})
all_ok &= prep_ok

emit({"table": table})
emit({"graph_table": fc.graph_table()})
print(json.dumps({"passed": bool(all_ok), "n_keys": len(table), "compile_ok": compile_ok,
                  "prepare_ok": prep_ok, "table": table, "wall_s": round(time.time() - t_start, 1)}))
sys.exit(0)
