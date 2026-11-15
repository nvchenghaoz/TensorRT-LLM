"""Probe: which flashinfer rmsnorm backend runs on this GPU for [M, 5376] bf16,
its PTX reduction order (CuTe DSL dump), and which Triton reduction-order
replica matches it bitwise."""
import glob
import json
import os
import re
import sys
import tempfile

DUMP = tempfile.mkdtemp(prefix="cute_dump_")
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", DUMP)

import torch
import triton
import triton.language as tl

ckpt = sys.argv[1] if len(sys.argv) > 1 else None
H = 5376
EPS = 1e-6
if ckpt:
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = tc["hidden_size"]
    EPS = tc["rms_norm_eps"]
    print("config hidden", H, "eps", EPS, "vocab", tc["vocab_size"], flush=True)

import flashinfer
import flashinfer.norm as fn

dev = torch.device("cuda")
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
print("use_cuda_norm", fn._use_cuda_norm(dev), flush=True)


@triton.jit
def _norm_variant(x_ptr, w_ptr, o_ptr, EPS, H: tl.constexpr, TPR: tl.constexpr,
                  VEC: tl.constexpr, NVB: tl.constexpr, ORDER: tl.constexpr,
                  DIV: tl.constexpr, MULORD: tl.constexpr, BF: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, TPR)
    base = x_ptr + row * H
    acc = tl.zeros([TPR], dtype=tl.float32)
    if ORDER == 0:  # sequential, v fastest then vb
        for vb in tl.static_range(NVB):
            for v in tl.static_range(VEC):
                c = vb * TPR * VEC + lane * VEC + v
                x = tl.load(base + c, mask=c < H, other=0.0).to(tl.float32)
                acc = acc + x * x
    elif ORDER == 1:  # sequential, vb fastest then v
        for v in tl.static_range(VEC):
            for vb in tl.static_range(NVB):
                c = vb * TPR * VEC + lane * VEC + v
                x = tl.load(base + c, mask=c < H, other=0.0).to(tl.float32)
                acc = acc + x * x
    elif ORDER == 2:  # per-vb pairwise tree over 8 v, then sequential over vb
        for vb in tl.static_range(NVB):
            c = vb * TPR * VEC + lane * VEC
            x0 = tl.load(base + c + 0, mask=c + 0 < H, other=0.0).to(tl.float32)
            x1 = tl.load(base + c + 1, mask=c + 1 < H, other=0.0).to(tl.float32)
            x2 = tl.load(base + c + 2, mask=c + 2 < H, other=0.0).to(tl.float32)
            x3 = tl.load(base + c + 3, mask=c + 3 < H, other=0.0).to(tl.float32)
            x4 = tl.load(base + c + 4, mask=c + 4 < H, other=0.0).to(tl.float32)
            x5 = tl.load(base + c + 5, mask=c + 5 < H, other=0.0).to(tl.float32)
            x6 = tl.load(base + c + 6, mask=c + 6 < H, other=0.0).to(tl.float32)
            x7 = tl.load(base + c + 7, mask=c + 7 < H, other=0.0).to(tl.float32)
            p = ((x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3)) + ((x4 * x4 + x5 * x5) +
                                                               (x6 * x6 + x7 * x7))
            acc = acc + p
    elif ORDER == 3:  # per-vb sequential over 8, then sequential over vb
        for vb in tl.static_range(NVB):
            p = tl.zeros([TPR], dtype=tl.float32)
            for v in tl.static_range(VEC):
                c = vb * TPR * VEC + lane * VEC + v
                x = tl.load(base + c, mask=c < H, other=0.0).to(tl.float32)
                p = p + x * x
            acc = acc + p
    elif ORDER == 4:  # sequential reverse of ORDER 0
        for vbr in tl.static_range(NVB):
            for vr in tl.static_range(VEC):
                c = (NVB - 1 - vbr) * TPR * VEC + lane * VEC + (VEC - 1 - vr)
                x = tl.load(base + c, mask=c < H, other=0.0).to(tl.float32)
                acc = acc + x * x
    elif ORDER == 5:  # vb ascending, but within each vb the 4 bf16 pairs reversed
        for vb in tl.static_range(NVB):
            for pr in tl.static_range(VEC // 2):
                for e in tl.static_range(2):
                    c = vb * TPR * VEC + lane * VEC + (VEC - 2 - 2 * pr) + e
                    x = tl.load(base + c, mask=c < H, other=0.0).to(tl.float32)
                    acc = acc + x * x
    elif ORDER == 6:  # vb descending, pairs reversed (lo then hi)
        for vbr in tl.static_range(NVB):
            for pr in tl.static_range(VEC // 2):
                for e in tl.static_range(2):
                    c = (NVB - 1 - vbr) * TPR * VEC + lane * VEC + (VEC - 2 - 2 * pr) + e
                    x = tl.load(base + c, mask=c < H, other=0.0).to(tl.float32)
                    acc = acc + x * x
    # butterfly within warps, then the two warps of the row.
    t = tl.reshape(acc, [TPR // 32, 2, 2, 2, 2, 2])
    if BF == 0:  # offsets 1, 2, 4, 8, 16 (lane bit 0 first), as the CuTe PTX shows
        t = tl.sum(t, axis=5)
        t = tl.sum(t, axis=4)
        t = tl.sum(t, axis=3)
        t = tl.sum(t, axis=2)
        t = tl.sum(t, axis=1)
    else:
        t = tl.sum(t, axis=1)
        t = tl.sum(t, axis=1)
        t = tl.sum(t, axis=1)
        t = tl.sum(t, axis=1)
        t = tl.sum(t, axis=1)
    ssq = tl.sum(t, axis=0)
    if DIV == 0:
        mean = tl.math.div_rn(ssq, H * 1.0)
    elif DIV == 1:
        mean = ssq / H
    elif DIV == 2:
        mean = tl.inline_asm_elementwise("div.approx.f32 $0, $1, $2;", "=f,f,f",
                                         [ssq, tl.full([], H, tl.float32)],
                                         dtype=tl.float32, is_pure=True, pack=1)
    else:
        mean = tl.inline_asm_elementwise("div.full.f32 $0, $1, $2;", "=f,f,f",
                                         [ssq, tl.full([], H, tl.float32)],
                                         dtype=tl.float32, is_pure=True, pack=1)
    rcp = tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [mean + EPS],
                                    dtype=tl.float32, is_pure=True, pack=1)
    BH: tl.constexpr = TPR * VEC * NVB
    cols = tl.arange(0, 8192)
    cm = cols < H
    xx = tl.load(base + cols, mask=cm, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=cm, other=0.0).to(tl.float32)
    if MULORD == 0:
        y = (xx * rcp) * w
    else:
        y = xx * (rcp * w)
    tl.store(o_ptr + row * H + cols, y.to(tl.bfloat16), mask=cm)


def run_variant(x, w, order, div, mulord, bf=0):
    out = torch.empty_like(x)
    _norm_variant[(x.shape[0],)](x, w, out, EPS, H=H, TPR=64, VEC=8,
                                 NVB=triton.cdiv(H, 512), ORDER=order, DIV=div,
                                 MULORD=mulord, BF=bf, num_warps=8)
    return out


g = torch.Generator(device="cuda").manual_seed(0)
M = 8192
cases = {
    "randn": torch.randn(M, H, device=dev, generator=g),
    "emb_scaled": (torch.randn(M, H, device=dev, generator=g) * 0.03).bfloat16().float() * 73.5,
    "heavy": torch.randn(M, H, device=dev, generator=g) * torch.exp(
        torch.randn(M, 1, device=dev, generator=g) * 2),
}
w = (torch.randn(H, device=dev, generator=g) * 0.3 + 1.0).bfloat16()
results = {}
for name, xf in cases.items():
    x = xf.bfloat16().contiguous()
    ref = flashinfer.rmsnorm(x, w, EPS)
    torch.cuda.synchronize()
    best = None
    for order in range(7):
        for bf in range(2):
            div, mulord = 0, 0
            o = run_variant(x, w, order, div, mulord, bf)
            bad_rows = int((o.view(torch.int16) != ref.view(torch.int16)).any(1).sum())
            bad_el = int((o.view(torch.int16) != ref.view(torch.int16)).sum())
            results[f"{name}/o{order}bf{bf}"] = (bad_rows, bad_el)
            if best is None or bad_el < best[1]:
                best = (f"o{order}bf{bf}", bad_el)
    print(name, "best", best, flush=True)
for k, v in results.items():
    print(k, v)

# PTX of the CuTe kernel, if dumped
files = glob.glob(os.path.join(DUMP, "**", "*"), recursive=True)
print("dump files", files[:20])
for f in files:
    if f.endswith(".ptx"):
        txt = open(f).read()
        lines = txt.splitlines()
        print("PTX", f, len(lines))
        first_fma = next(i for i, l in enumerate(lines) if "fma.rn" in l)
        keep = [l for l in lines[:first_fma + 4] if ("%rs" in l or "ld.shared" in l)]
        for l in keep[-140:]:
            print(l)
print(json.dumps({"passed": True, "done": True}))
