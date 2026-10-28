"""Branch 18 diagnostic: where does CUTLASS GEMM + qknorm_rope pass differ from main's fused CuTe QKV op?

Prints: fused-vs-fused determinism, per-plane mismatch count / max abs, mismatch row/col spread,
and for mismatching elements which fp32 emulation of norm+RoPE (FMA contraction variants)
reproduces the fused ref and which reproduces the pass.
"""

import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
    qknorm_rope,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
    _sm120_fp4_gemm_runner,
    nvfp4_quantize,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b18_qkv_step0 import norm_weights, quant  # noqa: E402

H, D, K = 24, 128, 3072
W = H * D


def log(*a):
    print(*a, flush=True)


def f32(x):
    return x.to(torch.float32)


def emulate(pre, w, cs, variant):
    """pre: [n,128] bf16 prenorm head rows; w: [128] bf16; cs: [n,128] f32. Returns bf16 [n,128]."""
    x = f32(pre)
    # lane-order sum of squares with fma: lane l: ((0 + x_l^2) + x_{32+l}^2)... each step fma
    xs = x.view(-1, 4, 32).double()
    s = torch.zeros(x.shape[0], 32, dtype=torch.float64, device=x.device)
    for g in range(4):
        s = (s + xs[:, g] * xs[:, g]).float().double()  # fma: exact product, one rounding
    for d in (16, 8, 4, 2, 1):
        idx = torch.arange(32, device=x.device) ^ d
        s = (s + s[:, idx]).float().double()
    tot = s[:, 0].float()
    mean = tot / 128.0
    inv = torch.rsqrt(mean + 1e-6) if variant["rsqrt"] == "torch" else (1.0 / torch.sqrt((mean + 1e-6).double())).float()
    inv = inv[:, None]
    wf = f32(w)[None]
    scaled = (x * inv) * wf
    e = scaled[:, 0::2]
    o = scaled[:, 1::2]
    c = cs[:, :64]
    sn = cs[:, 64:]
    ed, od, cd, sd = e.double(), o.double(), c.double(), sn.double()
    v = variant["rope"]
    if v == "nofma":
        ne = (e * c) - (o * sn)
        no = (o * c) + (e * sn)
    elif v == "fma_first":  # fma(a,b,-(c*d)) : first product fused
        ne = (ed * cd - (o * sn).double()).float()
        no = (od * cd + (e * sn).double()).float()
    else:  # fma second product fused
        ne = ((e * c).double() - od * sd).float()
        no = ((o * c).double() + ed * sd).float()
    out = torch.stack([ne, no], -1).reshape(x.shape[0], 128)
    return out.to(torch.bfloat16)


def main():
    ckpt = sys.argv[1]
    M = int(os.environ.get("B18_M", "4176"))
    torch.manual_seed(0)
    q_norm, k_norm, _ = norm_weights(ckpt)
    runner = _sm120_fp4_gemm_runner()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(W, K, device="cuda", dtype=torch.bfloat16) * s for s in (0.02, 0.035, 0.012)]
    a, a_sf, a_gs = quant(x)
    qw = [quant(w) for w in ws]
    weight = torch.cat([q[0] for q in qw], 0).contiguous()
    w_sf = torch.cat([q[1] for q in qw], 0).contiguous()
    seg_sf = w_sf.numel() // 3
    qkv_scale = torch.tensor([a_gs * q[2] for q in qw], device="cuda", dtype=torch.float32)
    theta = torch.rand(M, 64, device="cuda") * 6.28
    cos_sin = torch.cat([theta.cos(), theta.sin()], -1).float().contiguous()
    positions = torch.arange(M, device="cuda", dtype=torch.int64)

    def fused():
        return flux3_sm120_nvfp4.run(
            "qkv", a, weight, a_sf, w_sf, a_gs, qw[0][2], q_norm=q_norm, k_norm=k_norm,
            cos_sin=cos_sin, positions=positions, qkv_scale=qkv_scale,
        )

    refs = [fused().clone() for _ in range(5)]
    res = {"fused_self_equal": [bool(torch.equal(refs[0], r)) for r in refs[1:]]}
    res["fused_self_nmismatch"] = [int((refs[0] != r).sum()) for r in refs[1:]]
    ref = refs[0]
    pre = [runner.run_gemm(a, weight[i * W:(i + 1) * W], a_sf, w_sf[i * seg_sf:(i + 1) * seg_sf],
                           qkv_scale[i:i + 1], 0, 0, None, None) for i in range(3)]
    out = torch.empty(M, 3 * W, device="cuda", dtype=torch.bfloat16)
    qknorm_rope(pre[0], pre[1], pre[2], out, q_norm, k_norm, cos_sin, positions)
    torch.cuda.synchronize()
    for p, name in enumerate("qkv"):
        r = ref[:, p * W:(p + 1) * W]
        o = out[:, p * W:(p + 1) * W]
        mm = r != o
        res[name] = {"nmismatch": int(mm.sum()), "maxabs": float((f32(r) - f32(o)).abs().max())}
        if mm.any():
            idx = mm.nonzero()
            res[name]["rows_distinct"] = int(idx[:, 0].unique().numel())
            res[name]["heads_distinct"] = int((idx[:, 1] // 128).unique().numel())
            res[name]["row_tiles"] = sorted(set((idx[:, 0] // 128).tolist()))[:40]
            res[name]["examples"] = [
                [int(i), int(j), float(r[i, j]), float(o[i, j])] for i, j in idx[:10].tolist()
            ]
    # Emulation on all (row, head) groups that contain a mismatch, q and k planes
    variants = [{"rsqrt": rs, "rope": rp} for rs in ("torch", "exact") for rp in ("nofma", "fma_first", "fma_second")]
    emu = {}
    for p, name, wn in ((0, "q", q_norm), (1, "k", k_norm)):
        r = ref[:, p * W:(p + 1) * W].reshape(M, H, D)
        o = out[:, p * W:(p + 1) * W].reshape(M, H, D)
        g = (r != o).any(-1).nonzero()
        if g.numel() == 0:
            continue
        g = g[:2000]
        prev = pre[p].reshape(M, H, D)[g[:, 0], g[:, 1]]
        cs = cos_sin[positions[g[:, 0]]]
        rg = r[g[:, 0], g[:, 1]]
        og = o[g[:, 0], g[:, 1]]
        for var in variants:
            e = emulate(prev, wn, cs, var)
            key = f"{name}:{var['rsqrt']}:{var['rope']}"
            emu[key] = {
                "eq_ref_groups": int((e == rg).all(-1).sum()),
                "eq_pass_groups": int((e == og).all(-1).sum()),
                "groups": int(g.shape[0]),
            }
    res["emulation"] = emu
    log(json.dumps(res, indent=1))
    print(json.dumps({"passed": True, "diag": res}))


if __name__ == "__main__":
    main()
