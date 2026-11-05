"""Idea 254 prototype v2 of tensorrt_llm/_torch/modules/fp8_shared_decode.py (same API).

Lower per-byte ALU, to contend less with the persistent routed-MoE kernels sharing the SMs:
* K0 (_xq_kernel): x quantized ONCE (per token, 1x128 UE8M0, as per_token_quant_and_transform)
  into e4m3 xq [TM, K] (rows >= M zero) and fp32 xs [TM, K/128].
* K1 (_gu_kernel): per 128-K block an e4m3 x e4m3 tensor-core dot (no operand conversion),
  accumulator scaled by sw[kb] * xs[:, kb] ([BN, TM] values only); SUB blocks per loop
  iteration for bytes in flight. Writes bf16 gate/up pre-activations.
* K2 (_dn_kernel): SwiGLU + 1x128 requant of the in-range h block, e4m3 dot, scaled.
PDL chain K0 -> K1 -> K2 (each waits before reading its producer's output).
"""

import os
from typing import Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

MAX_TOKENS = 8
_TM = 16
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_MIN = -_FP8_MAX

_GU_BN = int(os.environ.get("V2_GU_BN", "16"))
_GU_SUB = int(os.environ.get("V2_GU_SUB", "4"))
_GU_WARPS = int(os.environ.get("V2_GU_WARPS", "4"))
_GU_STAGES = int(os.environ.get("V2_GU_STAGES", "2"))
_DN_BN = int(os.environ.get("V2_DN_BN", "64"))
_DN_WARPS = int(os.environ.get("V2_DN_WARPS", "4"))
_DN_STAGES = int(os.environ.get("V2_DN_STAGES", "2"))
_FP8_DOT = os.environ.get("V2_FP8_DOT", "1") == "1"
_USE_PDL = True


@triton.jit(do_not_specialize=["M"])
def _xq_kernel(x_ptr, xq_ptr, xs_ptr, M, stride_xm, fp8_max, fp8_min, K: tl.constexpr,
               TM: tl.constexpr, USE_PDL: tl.constexpr):
    kb = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, 128)
    msk = offs_m[:, None] < M
    v = tl.load(x_ptr + offs_m[:, None] * stride_xm + kb * 128 + offs_k[None, :], mask=msk,
                other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-10)
    s = amax / fp8_max
    s = tl.exp2(tl.ceil(tl.log2(tl.abs(s))))
    q = tl.clamp(v / s[:, None], fp8_min, fp8_max).to(tl.float8e4nv)
    tl.store(xq_ptr + offs_m[:, None] * K + kb * 128 + offs_k[None, :], q)
    tl.store(xs_ptr + offs_m * (K // 128) + kb, s)


@triton.jit(do_not_specialize=["M"])
def _gu_kernel(xq_ptr, xs_ptr, w_ptr, ws_ptr, rng_ptr, g_ptr, M, stride_wn, stride_wsn, stride_gm,
               K: tl.constexpr, I: tl.constexpr, TM: tl.constexpr, BN: tl.constexpr,
               SUB: tl.constexpr, NUM_STAGES: tl.constexpr, FP8_DOT: tl.constexpr,
               USE_PDL: tl.constexpr):
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    b0 = tl.load(rng_ptr)
    b1 = tl.load(rng_ptr + 1)
    span = (b1 - b0) * 128
    j = pid * BN
    if j < 2 * span:
        row0 = b0 * 128 + j
        if j >= span:
            row0 = I + b0 * 128 + (j - span)
        sblk = row0 // 128
        offs_n = row0 + tl.arange(0, BN)
        offs_k = tl.arange(0, 128)
        offs_m = tl.arange(0, TM)
        w_ptrs = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + offs_k[None, :]
        q_ptrs = xq_ptr + offs_m[None, :] * K + offs_k[:, None]  # [128, TM]
        acc = tl.zeros([BN, TM], dtype=tl.float32)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        for k0 in tl.range(0, K // 128, SUB, num_stages=NUM_STAGES):
            for u in tl.static_range(SUB):
                kb = k0 + u
                w = tl.load(w_ptrs + kb * 128)
                q = tl.load(q_ptrs + kb * 128)
                xs = tl.load(xs_ptr + offs_m * (K // 128) + kb)
                sw = tl.load(ws_ptr + sblk * stride_wsn + kb)
                if FP8_DOT:
                    p = tl.dot(w, q)
                else:
                    p = tl.dot(w.to(tl.float16), q.to(tl.float16))
                acc += p * (xs * sw)[None, :]
        tl.store(g_ptr + offs_m[None, :] * stride_gm + offs_n[:, None], acc.to(tl.bfloat16),
                 mask=offs_m[None, :] < M)


@triton.jit(do_not_specialize=["M"])
def _dn_kernel(g_ptr, w_ptr, ws_ptr, rng_ptr, out_ptr, M, stride_gm, stride_wn, stride_wsn,
               stride_om, fp8_max, fp8_min, I: tl.constexpr, TM: tl.constexpr, BN: tl.constexpr,
               HAS_LIMIT: tl.constexpr, LIMIT: tl.constexpr, NUM_STAGES: tl.constexpr,
               FP8_DOT: tl.constexpr, USE_PDL: tl.constexpr):
    pid = tl.program_id(0)
    n0 = pid * BN
    nblk = n0 // 128
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    offs_m = tl.arange(0, TM)
    b0 = tl.load(rng_ptr)
    b1 = tl.load(rng_ptr + 1)
    w_ptrs = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + offs_k[None, :]
    g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_k[None, :]
    gmask = offs_m[:, None] < M
    acc = tl.zeros([BN, TM], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    for kb in tl.range(b0, b1, num_stages=NUM_STAGES):
        w = tl.load(w_ptrs + kb * 128)
        a = tl.load(g_ptrs + kb * 128, mask=gmask, other=0.0).to(tl.float32)
        b = tl.load(g_ptrs + I + kb * 128, mask=gmask, other=0.0).to(tl.float32)
        if HAS_LIMIT:
            a = tl.minimum(a, LIMIT)
            b = tl.clamp(b, -LIMIT, LIMIT)
        hv = (a * tl.sigmoid(1.0 * a) * (b + 0.0)).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(hv), axis=1), 1e-10)
        s = amax / fp8_max
        s = tl.exp2(tl.ceil(tl.log2(tl.abs(s))))
        q = tl.clamp(hv / s[:, None], fp8_min, fp8_max).to(tl.float8e4nv)
        sw = tl.load(ws_ptr + nblk * stride_wsn + kb)
        if FP8_DOT:
            p = tl.dot(w, tl.trans(q))
        else:
            p = tl.dot(w.to(tl.float16), tl.trans(q.to(tl.float16)))
        acc += p * (s * sw)[None, :]
    tl.store(out_ptr + offs_m[None, :] * stride_om + offs_n[:, None], acc.to(tl.bfloat16),
             mask=offs_m[None, :] < M)


def shared_decode_mlp(x, gate_up_weight, gate_up_scale, down_weight, down_scale, block_range,
                      swiglu_limit: Optional[float], out=None, use_pdl=None):
    m, k = x.shape
    assert 0 < m <= MAX_TOKENS and k % (128 * _GU_SUB) == 0
    two_i = gate_up_weight.shape[0]
    i = two_i // 2
    n = down_weight.shape[0]
    has_limit = swiglu_limit is not None and swiglu_limit > 0.0
    xq = torch.empty((_TM, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((_TM, k // 128), dtype=torch.float32, device=x.device)
    g = torch.empty((m, two_i), dtype=torch.bfloat16, device=x.device)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _xq_kernel[(k // 128, )](x, xq, xs, m, x.stride(0), _FP8_MAX, _FP8_MIN, K=k, TM=_TM,
                             USE_PDL=_USE_PDL, num_warps=4)
    _gu_kernel[(two_i // _GU_BN, )](xq, xs, gate_up_weight, gate_up_scale, block_range, g, m,
                                    gate_up_weight.stride(0), gate_up_scale.stride(0), g.stride(0),
                                    K=k, I=i, TM=_TM, BN=_GU_BN, SUB=_GU_SUB,
                                    NUM_STAGES=_GU_STAGES, FP8_DOT=_FP8_DOT, USE_PDL=_USE_PDL,
                                    num_warps=_GU_WARPS, launch_pdl=True)
    _dn_kernel[(n // _DN_BN, )](g, down_weight, down_scale, block_range, out, m, g.stride(0),
                                down_weight.stride(0), down_scale.stride(0), out.stride(0),
                                _FP8_MAX, _FP8_MIN, I=i, TM=_TM, BN=_DN_BN, HAS_LIMIT=has_limit,
                                LIMIT=float(swiglu_limit) if has_limit else 0.0,
                                NUM_STAGES=_DN_STAGES, FP8_DOT=_FP8_DOT, USE_PDL=_USE_PDL,
                                num_warps=_DN_WARPS, launch_pdl=True)
    return out
