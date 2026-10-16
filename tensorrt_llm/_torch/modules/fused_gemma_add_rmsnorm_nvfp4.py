# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Triton kernel for Gemma fused add + RMSNorm + NVFP4 (group-16) quantize.

Replaces FlashInfer ``gemma_fused_add_rmsnorm`` (CuTe-DSL
``FusedAddRMSNormKernel`` with ``weight_bias=1.0``, BF16 out) followed by
``torch.ops.trtllm.fp4_quantize(y, global_scale, 16, False)`` with
bitwise-identical residual, FP4 and swizzled (128x4) SF bytes, including the
zeroed SF padding rows up to a multiple of 128. Only H = 5120 is supported.

Bitwise contract (H = 5120 BF16, the DSL kernel's geometry there: 64 threads
per row, 16-byte vectors, thread t owns columns 512*j + 8*t + v):
  * h = f32(x) + f32(r); the residual is stored as bf16(h) in place.
  * Per thread t: acc = fma.rn(h, h, acc), sequential over j = 0..9, v = 0..7.
  * The 64 per-thread partials are summed by an adjacent-pair tree (the DSL's
    butterfly warp reduce with offsets 1..16, then warp 0 + warp 1).
  * rstd = rsqrt.approx(div.rn(sum, 5120) + eps); y = bf16(h * rstd * (w + 1)).
  * The NVFP4 tail is fused_silu_mul_nvfp4's bitwise copy of
    cvt_warp_fp16_to_fp4, applied to the BF16 y.

Logical layout: a [32, 2] tensor element (k, p) stands for thread t = 2k + p,
so one row of the tensor holds a whole 16-element SF block (columns
512*j + 16*k .. +15) and the block max needs no cross-element shuffle.
"""

import os
from typing import Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from .fused_silu_mul_nvfp4 import (_cvt_e2m1x8, _cvt_e4m3_bits, _mul_rn,
                                   _rcp_approx_ftz, _sf_offset)

SUPPORTED_HIDDEN_SIZE = 5120
_SF_VEC = 16


@triton.jit
def _gdc_wait_with_memory_clobber():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _div_rn(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;",
                                     "=f,f,f", [a, b],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _rsqrt_approx(x):
    return tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;",
                                     "=f,f", [x],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _lo(u):
    # Low BF16 of a packed pair, exactly widened to fp32.
    return (u << 16).to(tl.float32, bitcast=True)


@triton.jit
def _hi(u):
    # High BF16 of a packed pair, exactly widened to fp32.
    return (u & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_bits(v):
    return v.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.int32)


@triton.jit
def _bf16_round(v):
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _load_w(w_ptr, tt, j: tl.constexpr):
    # Gamma for thread tt, column block j: 4 packed BF16 pairs (v = 0..7).
    base = j * 256 + tt * 4
    return (tl.load(w_ptr + base), tl.load(w_ptr + base + 1),
            tl.load(w_ptr + base + 2), tl.load(w_ptr + base + 3))


@triton.jit
def _sumsq_j(x_row, r_row, tt, j: tl.constexpr, acc):
    base = j * 256 + tt * 4
    for p in tl.static_range(4):
        xu = tl.load(x_row + base + p)
        ru = tl.load(r_row + base + p)
        h0 = _lo(xu) + _lo(ru)
        h1 = _hi(xu) + _hi(ru)
        acc = tl.fma(h0, h0, acc)
        acc = tl.fma(h1, h1, acc)
    return acc


@triton.jit
def _load_xr(x_row, r_row, tt, j: tl.constexpr):
    # The 4 packed x and r words of thread tt for column block j.
    base = j * 256 + tt * 4
    return (tl.load(x_row + base), tl.load(r_row + base),
            tl.load(x_row + base + 1), tl.load(r_row + base + 1),
            tl.load(x_row + base + 2), tl.load(r_row + base + 2),
            tl.load(x_row + base + 3), tl.load(r_row + base + 3))


@triton.jit
def _norm_pair(r_row, off, xu, ru, g, rstd):
    h0 = _lo(xu) + _lo(ru)
    h1 = _hi(xu) + _hi(ru)
    # Residual = bf16(h), in place (its x/r words were loaded earlier).
    tl.store(r_row + off, _bf16_bits(h0) | (_bf16_bits(h1) << 16))
    y0 = _bf16_round(h0 * rstd * (_lo(g) + 1.0))
    y1 = _bf16_round(h1 * rstd * (_hi(g) + 1.0))
    return y0, y1


@triton.jit
def _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, j: tl.constexpr, x0, r0,
                  x1, r1, x2, r2, x3, r3, g0, g1, g2, g3, rstd, s, r6, inv_s,
                  NUM_K_TILES: tl.constexpr):
    # Pass 2 for column block j. Its x/r words come from the caller, which
    # issues block j + 1's loads before this block's in-place residual
    # stores, so the 10 blocks do not serialize on store -> load ordering.
    base = j * 256 + tt * 4
    y0, y1 = _norm_pair(r_row, base, x0, r0, g0, rstd)
    y2, y3 = _norm_pair(r_row, base + 1, x1, r1, g1, rstd)
    y4, y5 = _norm_pair(r_row, base + 2, x2, r2, g2, rstd)
    y6, y7 = _norm_pair(r_row, base + 3, x3, r3, g3, rstd)

    a = tl.maximum(tl.maximum(tl.maximum(tl.abs(y0), tl.abs(y1)),
                              tl.maximum(tl.abs(y2), tl.abs(y3))),
                   tl.maximum(tl.maximum(tl.abs(y4), tl.abs(y5)),
                              tl.maximum(tl.abs(y6), tl.abs(y7))))
    vmax = tl.max(a, axis=1)  # [32]: one 16-element SF block per k (exact)

    sf_val = _mul_rn(s, _mul_rn(vmax, r6))
    sf_u8 = _cvt_e4m3_bits(sf_val).to(tl.uint8)
    sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    out_scale = tl.where(vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, inv_s)),
                         0.0)
    os2 = tl.broadcast_to(out_scale[:, None], [32, 2])

    packed = _cvt_e2m1x8(_mul_rn(y0, os2), _mul_rn(y1, os2),
                         _mul_rn(y2, os2), _mul_rn(y3, os2),
                         _mul_rn(y4, os2), _mul_rn(y5, os2),
                         _mul_rn(y6, os2), _mul_rn(y7, os2))
    tl.store(fp4_row + j * 64 + tt, packed)
    tl.store(sf_ptr + _sf_offset(row, j * 32 + kk, NUM_K_TILES), sf_u8)


@triton.jit
def _pair_sum(v, N: tl.constexpr):
    # [N] -> [N // 2]: (v[2i] + v[2i + 1]).
    return tl.sum(tl.reshape(v, [N // 2, 2]), axis=1)


@triton.jit
def fused_gemma_add_rmsnorm_nvfp4_kernel(fp4_ptr, sf_ptr, sfw_ptr, x_ptr, r_ptr,
                                         w_ptr, scale_ptr, M, x_stride,
                                         r_stride, eps,
                                         NUM_K_TILES: tl.constexpr,
                                         USE_PDL: tl.constexpr,
                                         EARLY_TRIGGER: tl.constexpr) -> None:
    # x/r/w/fp4 pointers are int32 views (packed BF16 pairs / 8 FP4 codes);
    # sfw_ptr is the SF buffer viewed as int32 words (padding rows only).
    tl.static_assert(NUM_K_TILES % 8 == 0)
    row = tl.program_id(axis=0).to(tl.int64)
    kk = tl.arange(0, 32)
    tt = kk[:, None] * 2 + tl.arange(0, 2)[None, :]  # [32, 2] thread ids

    if row < M:
        # Gamma is a weight: load it before waiting on the producer.
        w00, w01, w02, w03 = _load_w(w_ptr, tt, 0)
        w10, w11, w12, w13 = _load_w(w_ptr, tt, 1)
        w20, w21, w22, w23 = _load_w(w_ptr, tt, 2)
        w30, w31, w32, w33 = _load_w(w_ptr, tt, 3)
        w40, w41, w42, w43 = _load_w(w_ptr, tt, 4)
        w50, w51, w52, w53 = _load_w(w_ptr, tt, 5)
        w60, w61, w62, w63 = _load_w(w_ptr, tt, 6)
        w70, w71, w72, w73 = _load_w(w_ptr, tt, 7)
        w80, w81, w82, w83 = _load_w(w_ptr, tt, 8)
        w90, w91, w92, w93 = _load_w(w_ptr, tt, 9)
        if USE_PDL:
            if EARLY_TRIGGER:
                tl.extra.cuda.gdc_launch_dependents()
            _gdc_wait_with_memory_clobber()

        x_row = x_ptr + row * x_stride
        r_row = r_ptr + row * r_stride
        acc = tl.zeros([32, 2], dtype=tl.float32)
        for j in tl.static_range(10):
            acc = _sumsq_j(x_row, r_row, tt, j, acc)
        # Adjacent-pair tree over the 64 per-thread partials.
        s32 = tl.sum(acc, axis=1)
        s16 = _pair_sum(s32, 32)
        s8 = _pair_sum(s16, 16)
        s4 = _pair_sum(s8, 8)
        s2 = _pair_sum(s4, 4)
        total = _pair_sum(s2, 2)  # [1]
        tot = tl.broadcast_to(total[:, None], [32, 2])
        rstd = _rsqrt_approx(
            _div_rn(tot, tl.full([32, 2], 5120.0, tl.float32)) + eps)

        s = tl.load(scale_ptr)
        s_v = tl.full([32], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([32], 6.0, tl.float32))
        inv_s = _rcp_approx_ftz(s_v)
        fp4_row = fp4_ptr + row * 640
        # Pass 2 with a 2-block load lookahead (three rotating buffers): the
        # x/r words of block j are loaded two blocks before block j's
        # in-place residual stores, so loads never queue behind stores.
        a0, a1, a2, a3, a4, a5, a6, a7 = _load_xr(x_row, r_row, tt, 0)
        b0, b1, b2, b3, b4, b5, b6, b7 = _load_xr(x_row, r_row, tt, 1)
        c0, c1, c2, c3, c4, c5, c6, c7 = _load_xr(x_row, r_row, tt, 2)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 0, a0, a1, a2, a3,
                      a4, a5, a6, a7, w00, w01, w02, w03, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        a0, a1, a2, a3, a4, a5, a6, a7 = _load_xr(x_row, r_row, tt, 3)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 1, b0, b1, b2, b3,
                      b4, b5, b6, b7, w10, w11, w12, w13, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        b0, b1, b2, b3, b4, b5, b6, b7 = _load_xr(x_row, r_row, tt, 4)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 2, c0, c1, c2, c3,
                      c4, c5, c6, c7, w20, w21, w22, w23, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        c0, c1, c2, c3, c4, c5, c6, c7 = _load_xr(x_row, r_row, tt, 5)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 3, a0, a1, a2, a3,
                      a4, a5, a6, a7, w30, w31, w32, w33, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        a0, a1, a2, a3, a4, a5, a6, a7 = _load_xr(x_row, r_row, tt, 6)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 4, b0, b1, b2, b3,
                      b4, b5, b6, b7, w40, w41, w42, w43, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        b0, b1, b2, b3, b4, b5, b6, b7 = _load_xr(x_row, r_row, tt, 7)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 5, c0, c1, c2, c3,
                      c4, c5, c6, c7, w50, w51, w52, w53, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        c0, c1, c2, c3, c4, c5, c6, c7 = _load_xr(x_row, r_row, tt, 8)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 6, a0, a1, a2, a3,
                      a4, a5, a6, a7, w60, w61, w62, w63, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        a0, a1, a2, a3, a4, a5, a6, a7 = _load_xr(x_row, r_row, tt, 9)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 7, b0, b1, b2, b3,
                      b4, b5, b6, b7, w70, w71, w72, w73, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 8, c0, c1, c2, c3,
                      c4, c5, c6, c7, w80, w81, w82, w83, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
        _norm_quant_j(r_row, fp4_row, sf_ptr, row, tt, kk, 9, a0, a1, a2, a3,
                      a4, a5, a6, a7, w90, w91, w92, w93, rstd, s_v, r6, inv_s,
                      NUM_K_TILES)
    else:
        if USE_PDL:
            if EARLY_TRIGGER:
                tl.extra.cuda.gdc_launch_dependents()
            _gdc_wait_with_memory_clobber()
        # One extra program zeroes the SF padding rows [M, ceil128(M)), as
        # quantize_with_block_size does. In the 128x4 swizzle every 32-bit
        # word holds the 4 SF bytes of one row (c % 4 = 0..3), so the last
        # m-tile is NUM_K_TILES * 128 words: word w -> k-tile w // 128,
        # row (w % 4) * 32 + (w % 128) // 4. Contiguous int32 stores of the
        # padding rows' words never touch a real row's bytes.
        m0 = (M // 128) * 128
        tile_w = sfw_ptr + (M // 128) * (NUM_K_TILES * 128)
        for c in tl.static_range(NUM_K_TILES // 8):
            w = c * 1024 + tl.arange(0, 1024)
            within = w % 128
            rr = (within % 4) * 32 + within // 4
            tl.store(tile_w + w, tl.zeros([1024], tl.int32),
                     mask=(m0 + rr) >= M)


_NUM_SMS: Optional[int] = None


def _num_sms(device: torch.device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            device).multi_processor_count
    return _NUM_SMS


_PDL_ENABLED = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _launch(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
            eps: float, global_scale: torch.Tensor, fp4: torch.Tensor,
            sf: torch.Tensor) -> None:
    m, h = x.shape
    assert h == SUPPORTED_HIDDEN_SIZE
    num_k_tiles = (h // _SF_VEC + 3) // 4
    grid_rows = m + (1 if m % 128 != 0 else 0)
    if grid_rows == 0:
        return
    use_pdl = _PDL_ENABLED
    # Trigger the consumer early only while the whole grid fits in one wave.
    early = use_pdl and grid_rows < _num_sms(x.device)
    fused_gemma_add_rmsnorm_nvfp4_kernel[(grid_rows, )](
        fp4.view(torch.int32),
        sf,
        sf.view(torch.int32),
        x.view(torch.int32),
        residual.view(torch.int32),
        weight.view(torch.int32),
        global_scale,
        m,
        x.stride(0) // 2,
        residual.stride(0) // 2,
        eps,
        NUM_K_TILES=num_k_tiles,
        USE_PDL=use_pdl,
        EARLY_TRIGGER=early,
        num_warps=2,
        launch_pdl=use_pdl,
    )


def _check_args(x: torch.Tensor, residual: torch.Tensor,
                weight: torch.Tensor) -> None:
    assert x.dim() == 2 and x.shape == residual.shape
    assert x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert x.shape[1] == SUPPORTED_HIDDEN_SIZE == weight.numel()
    assert x.stride(1) == 1 and residual.stride(1) == 1
    assert x.stride(0) % 2 == 0 and residual.stride(0) % 2 == 0


def fused_gemma_add_rmsnorm_nvfp4_impl(x: torch.Tensor,
                                       residual: torch.Tensor,
                                       weight: torch.Tensor, eps: float,
                                       global_scale: torch.Tensor):
    _check_args(x, residual, weight)
    m, h = x.shape
    fp4 = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (h // _SF_VEC + 3) // 4
    sf = torch.empty((m_pad * num_k_tiles * 4, ),
                     dtype=torch.uint8,
                     device=x.device)
    _launch(x, residual, weight, eps, global_scale, fp4, sf)
    return fp4, sf


@torch.library.custom_op("trtllm::fused_gemma_add_rmsnorm_nvfp4",
                         mutates_args=("residual", ))
def fused_gemma_add_rmsnorm_nvfp4(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """residual += x (in place, BF16); returns NVFP4 (FP4 bytes [M, H/2],
    swizzled E4M3 SF) of gemma_rmsnorm(residual) * (weight + 1)."""
    return fused_gemma_add_rmsnorm_nvfp4_impl(x, residual, weight, eps,
                                              global_scale)


@fused_gemma_add_rmsnorm_nvfp4.register_fake
def _(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
      eps: float, global_scale: torch.Tensor):
    m, h = x.shape
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (h // _SF_VEC + 3) // 4
    return (x.new_empty((m, h // 2), dtype=torch.uint8),
            x.new_empty((m_pad * num_k_tiles * 4, ), dtype=torch.uint8))
