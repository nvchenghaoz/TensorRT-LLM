"""Find the Triton arithmetic that is bitwise equal to main's SM120 q/k/v prep chain.

Main's chain per head (modeling_gemma4 unfused apply_rope on SM120 FA2):
  RMSNorm.forward -> flashinfer_rmsnorm -> CuTe-DSL RMSNormKernel (2-D [N*H, hd]),
  flashinfer BatchQKApplyRotaryPosIdsCosSinCacheKernel (in place, fp32 math, bf16 store),
  torch .to(float8_e4m3fn) of K/V.
This probe runs Triton variants of each stage on the same synthetic inputs and counts
bitwise mismatches against main's ops. Synthetic tensors only; no model load.
  norm LOCAL: 0 tl.sum (old kernel), 1 sequential add, 2 sequential fma, 3 adjacent-pair tree,
              4 halving tree; LANE: 0 butterfly offsets 1,2,4,8,16; 1 offsets 16,8,4,2,1.
  rope R: 0 plain expression with fp fusion (old kernel), 1 fma(x,cos,-(p*sin)) / fma(x,cos,p*sin),
          2 fma(-p,sin,x*cos) / fma(p,sin,x*cos), 3 no fma.
"""
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


@triton.jit
def _split8(t):
    # t: [BN, L, 8] -> the 8 tensors t[..., v], v = 0..7
    BN: tl.constexpr = t.shape[0]
    L: tl.constexpr = t.shape[1]
    t = tl.reshape(t, (BN, L, 2, 2, 2))  # v = 4*a2 + 2*a1 + a0, a0 last
    e0, e1 = tl.split(t)  # a0
    e00, e01 = tl.split(e0)  # a1 (a0=0)
    e10, e11 = tl.split(e1)  # a1 (a0=1)
    v0, v4 = tl.split(e00)
    v2, v6 = tl.split(e01)
    v1, v5 = tl.split(e10)
    v3, v7 = tl.split(e11)
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _local8(x0, x1, x2, x3, x4, x5, x6, x7, LOCAL: tl.constexpr):
    if LOCAL == 1:
        s = x0 * x0
        s = s + x1 * x1
        s = s + x2 * x2
        s = s + x3 * x3
        s = s + x4 * x4
        s = s + x5 * x5
        s = s + x6 * x6
        s = s + x7 * x7
    elif LOCAL == 2:
        s = x0 * x0
        s = tl.fma(x1, x1, s)
        s = tl.fma(x2, x2, s)
        s = tl.fma(x3, x3, s)
        s = tl.fma(x4, x4, s)
        s = tl.fma(x5, x5, s)
        s = tl.fma(x6, x6, s)
        s = tl.fma(x7, x7, s)
    elif LOCAL == 3:
        s = ((x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3)) + ((x4 * x4 + x5 * x5) + (x6 * x6 + x7 * x7))
    else:
        s = ((x0 * x0 + x4 * x4) + (x2 * x2 + x6 * x6)) + ((x1 * x1 + x5 * x5) + (x3 * x3 + x7 * x7))
    return s


@triton.jit
def _local16(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, LOCAL: tl.constexpr):
    # value order v + 8*b (block 0 then block 1)
    if LOCAL == 1:
        s = _local8(a0, a1, a2, a3, a4, a5, a6, a7, 1)
        s = s + b0 * b0
        s = s + b1 * b1
        s = s + b2 * b2
        s = s + b3 * b3
        s = s + b4 * b4
        s = s + b5 * b5
        s = s + b6 * b6
        s = s + b7 * b7
    elif LOCAL == 2:
        s = _local8(a0, a1, a2, a3, a4, a5, a6, a7, 2)
        s = tl.fma(b0, b0, s)
        s = tl.fma(b1, b1, s)
        s = tl.fma(b2, b2, s)
        s = tl.fma(b3, b3, s)
        s = tl.fma(b4, b4, s)
        s = tl.fma(b5, b5, s)
        s = tl.fma(b6, b6, s)
        s = tl.fma(b7, b7, s)
    elif LOCAL == 3:
        s = _local8(a0, a1, a2, a3, a4, a5, a6, a7, 3) + _local8(b0, b1, b2, b3, b4, b5, b6, b7, 3)
    else:
        # halving over 16: (a_i + b_i) first, then halving over the 8 partials
        p0 = a0 * a0 + b0 * b0
        p1 = a1 * a1 + b1 * b1
        p2 = a2 * a2 + b2 * b2
        p3 = a3 * a3 + b3 * b3
        p4 = a4 * a4 + b4 * b4
        p5 = a5 * a5 + b5 * b5
        p6 = a6 * a6 + b6 * b6
        p7 = a7 * a7 + b7 * b7
        s = ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))
    return s


@triton.jit
def _lane_reduce(p, LANE: tl.constexpr):
    # p: [BN, 32] per-lane partials -> [BN]
    BN: tl.constexpr = p.shape[0]
    if LANE == 0:
        a, b = tl.split(tl.reshape(p, (BN, 16, 2)))
        p = a + b
        a, b = tl.split(tl.reshape(p, (BN, 8, 2)))
        p = a + b
        a, b = tl.split(tl.reshape(p, (BN, 4, 2)))
        p = a + b
        a, b = tl.split(tl.reshape(p, (BN, 2, 2)))
        p = a + b
        a, b = tl.split(p)
        s = a + b
    else:
        a, b = tl.split(tl.permute(tl.reshape(p, (BN, 2, 16)), (0, 2, 1)))
        p = a + b
        a, b = tl.split(tl.permute(tl.reshape(p, (BN, 2, 8)), (0, 2, 1)))
        p = a + b
        a, b = tl.split(tl.permute(tl.reshape(p, (BN, 2, 4)), (0, 2, 1)))
        p = a + b
        a, b = tl.split(tl.permute(tl.reshape(p, (BN, 2, 2)), (0, 2, 1)))
        p = a + b
        a, b = tl.split(p)
        s = a + b
    return s


@triton.jit
def _ssq(x, HD: tl.constexpr, LOCAL: tl.constexpr, LANE: tl.constexpr):
    # x: [BN, HD] fp32 -> sum of squares [BN] in the chosen order
    BN: tl.constexpr = x.shape[0]
    if LOCAL == 0:
        s = tl.sum(x * x, axis=1)
    elif HD == 256:
        x0, x1, x2, x3, x4, x5, x6, x7 = _split8(tl.reshape(x, (BN, 32, 8)))
        s = _lane_reduce(_local8(x0, x1, x2, x3, x4, x5, x6, x7, LOCAL), LANE)
    else:
        t = tl.permute(tl.reshape(x, (BN, 2, 256)), (0, 2, 1))
        xa, xb = tl.split(t)
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(tl.reshape(xa, (BN, 32, 8)))
        b0, b1, b2, b3, b4, b5, b6, b7 = _split8(tl.reshape(xb, (BN, 32, 8)))
        s = _lane_reduce(_local16(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, LOCAL), LANE)
    return s


@triton.jit
def _norm_kernel(x_ptr, w_ptr, y_ptr, M, EPS, HD: tl.constexpr, BN: tl.constexpr, LOCAL: tl.constexpr,
                 LANE: tl.constexpr, RSQ: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
    rmask = rows < M
    cols = tl.arange(0, HD)
    x = tl.load(x_ptr + rows[:, None] * HD + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    s = _ssq(x, HD, LOCAL, LANE)
    mean = s * (1.0 / HD)
    if RSQ == 0:
        r = tl.math.rsqrt(mean + EPS)
    else:
        r = tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [mean + EPS],
                                      dtype=tl.float32, is_pure=True, pack=1)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = x * r[:, None] * w[None, :]
    tl.store(y_ptr + rows[:, None] * HD + cols[None, :], y.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _rope_kernel(x_ptr, y_ptr, pos_ptr, cs_ptr, M, NH, HD: tl.constexpr, BN: tl.constexpr, R: tl.constexpr):
    # x: [M, NH*HD] bf16 (normed); row = token
    HALF: tl.constexpr = HD // 2
    h = tl.program_id(0)
    rows = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN).to(tl.int64)
    rmask = rows < M
    offs = tl.arange(0, HALF)
    base = x_ptr + rows[:, None] * (NH * HD) + h * HD + offs[None, :]
    y1 = tl.load(base, mask=rmask[:, None], other=0.0).to(tl.float32)
    y2 = tl.load(base + HALF, mask=rmask[:, None], other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cs = cs_ptr + pos[:, None] * HD + offs[None, :]
    cos = tl.load(cs, mask=rmask[:, None], other=0.0)
    sin = tl.load(cs + HALF, mask=rmask[:, None], other=0.0)
    if R == 0 or R == 3:
        o1 = y1 * cos - y2 * sin
        o2 = y2 * cos + y1 * sin
    elif R == 1:
        o1 = tl.fma(y1, cos, -(y2 * sin))
        o2 = tl.fma(y2, cos, y1 * sin)
    else:
        o1 = tl.fma(-y2, sin, y1 * cos)
        o2 = tl.fma(y1, sin, y2 * cos)
    ob = y_ptr + rows[:, None] * (NH * HD) + h * HD + offs[None, :]
    tl.store(ob, o1.to(tl.bfloat16), mask=rmask[:, None])
    tl.store(ob + HALF, o2.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _cast_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(x_ptr + o, mask=m, other=0.0)
    tl.store(y_ptr + o, x.to(tl.float8e4nv), mask=m)


def make_x(m, hd, seed, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, hd, generator=g)
    # heavy-tailed rows/channels like real projections
    x = x * torch.exp(0.8 * torch.randn(m, 1, generator=g)) * torch.exp(0.5 * torch.randn(1, hd, generator=g))
    return x.to(torch.bfloat16).to(dev)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT")
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    eps = float(tc.get("rms_norm_eps", 1e-6))
    hd_s = int(tc["head_dim"])
    hd_g = int(tc.get("global_head_dim") or hd_s)
    log("device", torch.cuda.get_device_name(), "eps", eps, "head_dims", hd_s, hd_g,
        "triton", triton.__version__)
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    res = {"norm": {}, "rope": {}, "cast": {}}
    M = int(os.environ.get("PROBE_ROWS", str(1 << 20)))
    for hd in sorted({hd_s, hd_g}):
        for has_w in (True, False):
            mod = RMSNorm(hidden_size=hd, eps=eps, dtype=torch.bfloat16, device=dev, has_weights=has_w)
            if has_w:
                g = torch.Generator(device="cpu").manual_seed(7)
                with torch.no_grad():
                    mod.weight.copy_((1.0 + 0.3 * torch.randn(hd, generator=g)).to(torch.bfloat16))
            x = make_x(M, hd, 11 + hd, dev)
            ref = mod(x)
            torch.cuda.synchronize()
            bn = max(1, 4096 // hd)
            for local in (0, 1, 2, 3, 4):
                for lane in ((0,) if local == 0 else (0, 1)):
                    for rsq in ((0, 1) if local in (0, 1) else (0,)):
                        y = torch.empty_like(x)
                        try:
                            _norm_kernel[(triton.cdiv(M, bn),)](x, mod.weight, y, M, eps, HD=hd, BN=bn,
                                                                LOCAL=local, LANE=lane, RSQ=rsq,
                                                                num_warps=8, enable_fp_fusion=False)
                            torch.cuda.synchronize()
                            diff = (y.view(torch.int16) != ref.view(torch.int16))
                            ne = int(diff.sum().item())
                            nr = int(diff.any(dim=1).sum().item())
                        except Exception as e:  # noqa: BLE001
                            ne, nr = -1, -1
                            log("variant failed", hd, local, lane, rsq, repr(e)[:300])
                        key = f"hd{hd}_w{int(has_w)}_L{local}_lane{lane}_rsq{rsq}"
                        res["norm"][key] = {"elem_mismatch": ne, "row_mismatch": nr,
                                            "frac": ne / x.numel() if ne >= 0 else None}
                        log(key, res["norm"][key])
    # RoPE vs flashinfer, on bf16 normed-like input
    from tensorrt_llm._torch.custom_ops import flashinfer_apply_rope_with_cos_sin_cache_inplace
    for hd in sorted({hd_s, hd_g}):
        nh = 8
        ntok = 32768
        half = hd // 2
        max_pos = 16384
        inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        s = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
        cs = torch.stack([torch.cos(s), torch.sin(s)], dim=1).to(torch.float32).contiguous().to(dev)
        g = torch.Generator(device="cpu").manual_seed(3)
        pos = torch.randint(0, max_pos, (ntok,), generator=g, dtype=torch.int32).to(dev)
        q = (torch.randn(ntok, nh * hd, generator=g) * 3).to(torch.bfloat16).to(dev)
        k = (torch.randn(ntok, nh * hd, generator=g) * 3).to(torch.bfloat16).to(dev)
        qr, kr = q.clone(), k.clone()
        flashinfer_apply_rope_with_cos_sin_cache_inplace(pos, qr, kr, hd, cs.view(max_pos, -1), True)
        torch.cuda.synchronize()
        bn = max(1, 4096 // hd)
        for r in (0, 1, 2, 3):
            y = torch.empty_like(q)
            _rope_kernel[(nh, triton.cdiv(ntok, bn))](q, y, pos, cs, ntok, nh, HD=hd, BN=bn, R=r, num_warps=8,
                                                      enable_fp_fusion=(r == 0))
            torch.cuda.synchronize()
            ne = int((y.view(torch.int16) != qr.view(torch.int16)).sum().item())
            key = f"hd{hd}_R{r}"
            res["rope"][key] = {"elem_mismatch": ne, "frac": ne / q.numel()}
            log(key, res["rope"][key])
    # bf16 -> fp8 cast: all 65536 bf16 bit patterns that are finite
    allb = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)
    fin = torch.isfinite(allb)
    t_ref = allb.to(torch.float8_e4m3fn)
    t_tri = torch.empty_like(t_ref)
    _cast_kernel[(triton.cdiv(allb.numel(), 1024),)](allb, t_tri, allb.numel(), BLOCK=1024)
    torch.cuda.synchronize()
    d = (t_ref.view(torch.uint8) != t_tri.view(torch.uint8)) & fin
    small = allb.float().abs() < 464.0
    res["cast"] = {"mismatch_all_finite": int(d.sum().item()),
                   "mismatch_abs_lt_464": int((d & small).sum().item()),
                   "first_mismatch_values": allb[d].float()[:8].tolist()}
    log("cast", res["cast"])
    best = {}
    for hd in sorted({hd_s, hd_g}):
        ok = [k for k, v in res["norm"].items() if k.startswith(f"hd{hd}_") and v["elem_mismatch"] == 0]
        best[f"norm_hd{hd}_exact_variants"] = ok
        best[f"rope_hd{hd}_exact_variants"] = [k for k, v in res["rope"].items()
                                               if k.startswith(f"hd{hd}_") and v["elem_mismatch"] == 0]
    log(json.dumps(best))
    print(json.dumps({"passed": True, "probe": res, "exact_variants": best}))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
