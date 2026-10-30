# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton LayerNorm + adaLN modulate + NVFP4 quantize, one program per row (D = 3072).

The adaln_nvfp4 entry's kernel for row counts up to 4096 with a [1, 1, D] row or a [1, 2, D]
table + bool select (the 1024^2 mode blocks); the warp-per-row CUDA kernel takes the rest.

Exactness contract:
- LayerNorm + modulate is Inductor's own kernel for this block (per-block compile with
  emulate_precision_casts=True): the same looped Welford reduction
  (triton_helpers.welford_reduce + welford) over one [1, 1024] block per chunk (here the
  three chunks of the row unrolled and kept in registers), num_warps 8, the same bf16
  rounding points and enable_fp_fusion=False. The LayerNorm affine is the identity
  (FLUX.3's pre_norm has no weight and no bias): the weight multiply is dropped (exact) and
  the bias add kept as + 0.0 (it maps -0 to +0).
- The quantize is FlashInfer's fast-math path on the bf16-rounded x_mod: scale = gs * (amax
  * rcp.approx.ftz(6)), cvt.rn.satfinite.e4m3, output scale rcp.approx.ftz(f32(e4m3) *
  rcp.approx.ftz(gs)), values * output scale with cvt.rn.satfinite.e2m1x2 (even element in
  the low nibble), 128x4 swizzled scale layout with zero-filled padding rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

D = 3072
_SF_COLS = D // 16
_R0_BLOCK = 1024
_NUM_WARPS = 8


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _cvt_e4m3(x):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b16 fp8_pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
            cvt.u32.u16 $0, fp8_pair;
        }
        """,
        "=r,f",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _output_scale(sf_bits, gs):
    # FlashInfer nvfp4_compute_output_scale: rcp(f32(e4m3) * rcp(gs)), 0 when the scale is 0.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred p_zero;
            .reg .b16 fp8_pair;
            .reg .b32 h2_32;
            .reg .b16 h_lo, h_hi;
            .reg .f32 scale_f32, rcp_gs, product, result;
            cvt.u16.u32 fp8_pair, $1;
            cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
            mov.b32 {h_lo, h_hi}, h2_32;
            cvt.f32.f16 scale_f32, h_lo;
            rcp.approx.ftz.f32 rcp_gs, $2;
            mul.f32 product, scale_f32, rcp_gs;
            rcp.approx.ftz.f32 result, product;
            setp.eq.f32 p_zero, scale_f32, 0f00000000;
            selp.f32 $0, 0f00000000, result, p_zero;
        }
        """,
        "=f,r,f",
        [sf_bits, gs],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e2m1x2(lo, hi, out_scale):
    # One packed byte (lo nibble = lo) replicated in all four bytes of the u32; caller masks 0xFF.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 a, b;
            .reg .b8 byte0;
            mul.f32 a, $1, $3;
            mul.f32 b, $2, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, b, a;
            mov.b32 $0, {byte0, byte0, byte0, byte0};
        }
        """,
        "=r,f,f,f",
        [lo, hi, out_scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _modq_chunk(
    tmp29,  # this chunk of x as f32 (from a bf16 load)
    r0_offset,
    r0_1,
    r0_mask,
    x0,
    xindex,
    xmask,
    tmp5,  # row mean
    tmp9,  # row M2
    tmp11,  # row select (HAS_SELECT; unused otherwise)
    in_ptr2,
    in_ptr5,
    fp4_ptr,
    sf_ptr,
    gs,
    fp4_max_rcp,
    padded_xnumel,
    scale_row_stride,
    shift_row_stride,
    r0_numel: tl.constexpr,
    EPS: tl.constexpr,
    HAS_SELECT: tl.constexpr,
    XBLOCK: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    SF_COLS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Pointwise modulate + NVFP4 quantize of one Q_BLOCK chunk (Inductor's pointwise math).
    if HAS_SELECT:
        # Inductor loads both table rows and selects the value (where(sel, row1, row0) after the
        # same bf16->f32 casts); loading only the selected row gives the same value.
        sel_row = tmp11.to(tl.int64)
        tmp12 = tl.load(
            in_ptr2 + (sel_row * scale_row_stride + r0_1),
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp13 = tmp12.to(tl.bfloat16)
        tmp18 = tmp13.to(tl.float32)
    else:
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0).to(
            tl.float32
        )
    tmp19 = tmp18.to(tl.bfloat16)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20.to(tl.bfloat16)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.full([1, 1], 1.0, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24.to(tl.bfloat16)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26.to(tl.bfloat16)
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp29.to(tl.bfloat16)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 - tmp5
    tmp34 = tl.full([1, 1], r0_numel, tl.float32)
    tmp35 = tmp9 / tmp34
    tmp36 = tl.full([1, 1], EPS, tl.float32)
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    # Identity LayerNorm affine: weight 1.0 (an exact multiply, dropped), bias +0.0.
    tmp48 = tl.full([1, 1], 0.0, tl.float32)
    tmp49 = tmp39 + tmp48
    tmp50 = tmp49.to(tl.bfloat16)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51.to(tl.bfloat16)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53.to(tl.bfloat16)
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp28 * tmp55
    tmp57 = tmp56.to(tl.bfloat16)
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58.to(tl.bfloat16)
    tmp60 = tmp59.to(tl.float32)
    if HAS_SELECT:
        tmp61 = tl.load(
            in_ptr5 + (sel_row * shift_row_stride + r0_1),
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp62 = tmp61.to(tl.bfloat16)
        tmp67 = tmp62.to(tl.float32)
    else:
        tmp67 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0).to(
            tl.float32
        )
    tmp68 = tmp67.to(tl.bfloat16)
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp69.to(tl.bfloat16)
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp60 + tmp71
    tmp73 = tmp72.to(tl.bfloat16)
    tmp74 = tmp73.to(tl.float32)

    # ---- NVFP4 quantize of the bf16-rounded x_mod (FlashInfer fast-math path) ----
    # Pairs stay in one thread (4 elements/thread): one packed byte per pair, and the
    # 16-element amax is an in-thread max plus a 4-lane max (exact: bf16 magnitudes).
    pairs = tl.reshape(tmp74, [XBLOCK, Q_BLOCK // 2, 2])
    lo, hi = tl.split(pairs)
    m2 = tl.maximum(tl.abs(lo), tl.abs(hi))
    amax = tl.max(tl.reshape(m2, [XBLOCK, GROUPS, 8]), axis=2)  # [XBLOCK, GROUPS]
    scale_float = gs * (amax * fp4_max_rcp)
    sf_bits = _cvt_e4m3(scale_float)
    out_scale = _output_scale(sf_bits, gs)
    os_p = tl.reshape(
        tl.broadcast_to(out_scale[:, :, None], [XBLOCK, GROUPS, 8]),
        [XBLOCK, Q_BLOCK // 2],
    )
    pbyte = (_e2m1x2(lo, hi, os_p) & 0xFF).to(tl.uint8)
    byte_idx = (r0_offset // 2) + tl.arange(0, Q_BLOCK // 2)[None, :]
    tl.store(
        fp4_ptr + (byte_idx + (r0_numel // 2) * x0),
        pbyte,
        (byte_idx < r0_numel // 2) & xmask,
    )
    col = (r0_offset // 16) + tl.arange(0, GROUPS)[None, :]
    sf_off = (
        (col % 4)
        + (col // 4) * 512
        + (x0 % 32) * 16
        + ((x0 % 128) // 32) * 4
        + (x0 // 128) * (128 * SF_COLS)
    )
    sf_val = tl.where(xmask, sf_bits, 0).to(tl.uint8)
    tl.store(sf_ptr + sf_off, sf_val, (col < r0_numel // 16) & (xindex < padded_xnumel))


@triton.jit
def _adaln_nvfp4_kernel(
    in_ptr0,  # x [M, D] bf16
    in_ptr1,  # select [M] bool (HAS_SELECT)
    in_ptr2,  # scale: table row 0 (HAS_SELECT) or the [D] row
    in_ptr5,  # shift: table row 0 (HAS_SELECT) or the [D] row
    gs_ptr,  # [1] f32 NVFP4 global scale
    fp4_ptr,  # [M, D // 2] uint8
    sf_ptr,  # [padded_M * SF_COLS] uint8, 128x4 swizzled
    xnumel,
    padded_xnumel,
    scale_row_stride,
    shift_row_stride,
    r0_numel: tl.constexpr,
    EPS: tl.constexpr,
    HAS_SELECT: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    tl.static_assert(XBLOCK == 1)
    # The row is three register-resident R0_BLOCK chunks (no second x read); SF_COLS == D/16.
    tl.static_assert(r0_numel == 3 * R0_BLOCK)
    SF_COLS: tl.constexpr = r0_numel // 16
    GROUPS: tl.constexpr = R0_BLOCK // 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex.to(tl.int64)
    gs = tl.load(gs_ptr)
    fp4_max_rcp = _rcp_approx(tl.full([1, 1], 6.0, tl.float32))
    if xoffset < xnumel:
        # Inductor's LN reduction unrolled over the three chunks: the same per-element
        # welford_reduce sequence, the same R0_BLOCK accumulators and final combine.
        r0_mask = r0_base < r0_numel
        xa = tl.load(
            in_ptr0 + (r0_base + r0_numel * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        xb = tl.load(
            in_ptr0 + ((R0_BLOCK + r0_base) + r0_numel * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        xc = tl.load(
            in_ptr0 + ((2 * R0_BLOCK + r0_base) + r0_numel * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp4 = tl.broadcast_to(xa.to(tl.bfloat16).to(tl.float32).to(tl.float32), [XBLOCK, R0_BLOCK])
        mn, m2n, wn = triton_helpers.welford_reduce(tmp4, tmp5_mean, tmp5_m2, tmp5_weight, True)
        tmp5_mean = tl.where(r0_mask & xmask, mn, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, m2n, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, wn, tmp5_weight)
        tmp4 = tl.broadcast_to(xb.to(tl.bfloat16).to(tl.float32).to(tl.float32), [XBLOCK, R0_BLOCK])
        mn, m2n, wn = triton_helpers.welford_reduce(tmp4, tmp5_mean, tmp5_m2, tmp5_weight, False)
        tmp5_mean = tl.where(r0_mask & xmask, mn, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, m2n, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, wn, tmp5_weight)
        tmp4 = tl.broadcast_to(xc.to(tl.bfloat16).to(tl.float32).to(tl.float32), [XBLOCK, R0_BLOCK])
        mn, m2n, wn = triton_helpers.welford_reduce(tmp4, tmp5_mean, tmp5_m2, tmp5_weight, False)
        tmp5_mean = tl.where(r0_mask & xmask, mn, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, m2n, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, wn, tmp5_weight)
        tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
        # XBLOCK == 1: the row's mean/M2 as scalars (a max over one element is an exact copy),
        # decoupling the reduction's register layout from the pointwise pass.
        tmp5 = tl.max(tmp6, axis=0)
        tmp9 = tl.max(tmp7, axis=0)
        if HAS_SELECT:
            tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy="evict_last").to(tl.int1)
        else:
            tmp11 = xmask
        _modq_chunk(
            xa, 0, r0_base, r0_mask, x0, xindex, xmask, tmp5, tmp9, tmp11,
            in_ptr2, in_ptr5, fp4_ptr, sf_ptr, gs, fp4_max_rcp, padded_xnumel,
            scale_row_stride, shift_row_stride, r0_numel, EPS, HAS_SELECT, XBLOCK, R0_BLOCK,
            SF_COLS, GROUPS,
        )  # fmt: skip
        _modq_chunk(
            xb, R0_BLOCK, R0_BLOCK + r0_base, r0_mask, x0, xindex, xmask, tmp5, tmp9, tmp11,
            in_ptr2, in_ptr5, fp4_ptr, sf_ptr, gs, fp4_max_rcp, padded_xnumel,
            scale_row_stride, shift_row_stride, r0_numel, EPS, HAS_SELECT, XBLOCK, R0_BLOCK,
            SF_COLS, GROUPS,
        )  # fmt: skip
        _modq_chunk(
            xc, 2 * R0_BLOCK, 2 * R0_BLOCK + r0_base, r0_mask, x0, xindex, xmask, tmp5, tmp9,
            tmp11, in_ptr2, in_ptr5, fp4_ptr, sf_ptr, gs, fp4_max_rcp, padded_xnumel,
            scale_row_stride, shift_row_stride, r0_numel, EPS, HAS_SELECT, XBLOCK, R0_BLOCK,
            SF_COLS, GROUPS,
        )  # fmt: skip
    else:
        # Padding rows of the 128x4 layout: zero scale factors.
        pcol = tl.arange(0, 256)[None, :]
        for c0 in tl.range(0, SF_COLS, 256):
            c = c0 + pcol
            poff = (
                (c % 4)
                + (c // 4) * 512
                + (x0 % 32) * 16
                + ((x0 % 128) // 32) * 4
                + (x0 // 128) * (128 * SF_COLS)
            )
            tl.store(
                sf_ptr + poff,
                tl.zeros([1, 256], tl.uint8),
                (c < SF_COLS) & (xindex < padded_xnumel),
            )


@torch.library.custom_op(
    "trtllm::flux3_klein_adaln_nvfp4_triton", mutates_args=(), device_types="cuda"
)
def adaln_nvfp4_triton(
    x: torch.Tensor,
    select: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp4 [M, D/2] uint8, swizzled sf [ceil128(M) * D/16] uint8) of
    ``(1 + scale) * LayerNorm(x) + shift``; x contiguous [..., D] bf16, scale / shift [1, R,
    D] bf16 (unit last stride), select [..., 1] bool or None (row 0)."""
    m = x.numel() // D
    padded = (m + 127) // 128 * 128
    fp4 = torch.empty((m, D // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((padded * _SF_COLS,), dtype=torch.uint8, device=x.device)
    if m == 0:
        return fp4, sf
    has_select = select is not None
    _adaln_nvfp4_kernel[(padded,)](
        x,
        select if has_select else x,
        scale,
        shift,
        global_scale,
        fp4,
        sf,
        m,
        padded,
        scale.stride(1) if has_select else 0,
        shift.stride(1) if has_select else 0,
        r0_numel=D,
        EPS=eps,
        HAS_SELECT=has_select,
        XBLOCK=1,
        R0_BLOCK=_R0_BLOCK,
        num_warps=_NUM_WARPS,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return fp4, sf


@adaln_nvfp4_triton.register_fake
def _adaln_nvfp4_triton_fake(x, select, scale, shift, global_scale, eps):
    m = x.numel() // D
    padded = (m + 127) // 128 * 128
    return (
        x.new_empty((m, D // 2), dtype=torch.uint8),
        x.new_empty((padded * _SF_COLS,), dtype=torch.uint8),
    )
