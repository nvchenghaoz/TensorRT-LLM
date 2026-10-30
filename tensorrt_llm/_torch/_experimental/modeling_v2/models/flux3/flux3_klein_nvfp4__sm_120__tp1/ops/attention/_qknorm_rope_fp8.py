# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel of ``qknorm_rope_fp8``: Q/K RMSNorm + RoPE of a BF16 QKV GEMM output straight
to per-tensor E4M3 Q/K/V with a dynamic amax.

Two launches of one kernel (12 warps: 0-3 the q heads, 4-7 the k heads, 8-11 v; grid-stride
row loop, one wave of 4 CTAs/SM):

- mode 1: per (row, head) the RMSNorm + RoPE arithmetic below, each result rounded to BF16,
  folded into a per-warp abs-max (``max.f32`` of ``|x|``, NaN ignored) -- no store. Warps 8-11
  fold ``|v|``. Each CTA writes one FP32 partial per plane into ``parts`` ``[3, n_part]``
  (``n_part`` = mode 1's grid, <= 1024).
- mode 2: every CTA re-reduces the three planes' partials, derives
  ``scale = bf16(max(div_rn(amax, 448), 1 / (448 * 512)))`` and ``factor = div_rn(1, scale)``
  (CTA 0 writes ``scales[0..7]``), recomputes the identical BF16 values, multiplies them by the
  plane's factor (``mul.rn.f32``) and converts pairs with ``cvt.rn.satfinite.e4m3x2.f32`` into
  three contiguous ``[M, W]`` E4M3 planes.

Per row and head (128 dims, pairs (2i, 2i + 1)): ``inv_rms = rsqrt(sum(x^2) / 128 + 1e-6)``
(the sum over the four 32-lane column groups, then a butterfly across the warp);
``even = x[2i] * inv_rms * w[2i]``, ``odd = x[2i+1] * inv_rms * w[2i+1]``;
``rot_even = fma(even, cos, -(odd * sin))``, ``rot_odd = fma(odd, cos, even * sin)`` with
``cos = cos_sin[pos, i]``, ``sin = cos_sin[pos, 64 + i]``.

Data movement: q/k head pairs come in with 16-B cp.async into a per-warp 3-stage smem ring (row
r + grid's pair j is in flight while row r's pairs j + 1.. compute), E4M3 pairs are staged in
smem and stored 8 B per lane; v is 16-B loads and 8-B stores. 40 registers (4 CTAs/SM).

When every warp's row slice starts on a 128-B line, mode 2 drops each consumed source line from
L2 (``discard.global.L2``: no DRAM writeback) right after reading it, so the q, k and v columns
of the GEMM output are undefined after the call. The values written are unchanged.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import numpy as np
import torch
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

# 1 / (448 * 512) in FP32: the smallest per-tensor scale.
_MIN_SCALING_FACTOR = float(np.float32(1.0) / (np.float32(448.0) * np.float32(512.0)))
# The amax every fold starts from (an all-NaN plane gives 0).
_AMAX_INIT = 0.0
# The most mode-1 partials per plane: mode 2 reads <= 3 per thread (384 threads).
_MAX_PARTIALS = 1024
# Grid of both passes: one wave of 4 CTAs/SM (the kernel is capped at 40 registers).
_CTAS_PER_SM = 4

_COMPILED: dict = {}
_SM_COUNT: dict = {}


def _grid(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNT.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = count
    return _CTAS_PER_SM * count


@dsl_user_op
def _max_abs(acc: cutlass.Float32, x: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(acc).ir_value(loc=loc, ip=ip),
                cutlass.Float32(x).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .f32 t;\nabs.f32 t, $2;\nmax.f32 $0, $1, t;\n}",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _max_f32(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _e4m3x2(
    lo: cutlass.Float32, hi: cutlass.Float32, factor: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Int16:
    """Int16 holding e4m3(lo * factor) in the low byte and e4m3(hi * factor) in the high byte."""
    return cutlass.Int16(
        llvm.inline_asm(
            T.i16(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(factor).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .f32 a, b;\nmul.rn.f32 a, $1, $3;\nmul.rn.f32 b, $2, $3;\n"
            "cvt.rn.satfinite.e4m3x2.f32 $0, a, b;\n}",
            "=h,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _div_rn(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_rn(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _scale_factor(amax, min_scale):
    """scale = bf16(max(div_rn(amax, 448), min_scale)), factor = div_rn(1, scale)."""
    scale = _max_f32(_div_rn(amax, cutlass.Float32(448.0)), min_scale)
    scale = cutlass.BFloat16(scale).to(cutlass.Float32)
    return scale, _div_rn(cutlass.Float32(1.0), scale)


@dsl_user_op
def _cp_async16(saddr: cutlass.Int32, gaddr: cutlass.Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(saddr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _discard_l2(gaddr: cutlass.Int64, *, loc=None, ip=None) -> None:
    """discard.global.L2 of the 128-B line at gaddr (128-B aligned): dropped, no writeback."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip)],
        "discard.global.L2 [$0], 128;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _asm_void(text: str, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        text,
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _cp_async_commit():
    _asm_void("cp.async.commit_group;")


def _cp_async_wait(n: int):
    _asm_void(f"cp.async.wait_group {n};")


def _syncwarp():
    _asm_void("bar.warp.sync -1;")


@dsl_user_op
def _ld_global_v4(gaddr: cutlass.Int64, *, loc=None, ip=None):
    """16-B global load (read-only data) as four Int32 words."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_global_v2(
    gaddr: cutlass.Int64, a: cutlass.Int32, b: cutlass.Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_b64(gaddr: cutlass.Int64, a: cutlass.Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip), cutlass.Int64(a).ir_value(loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _fma_of_mul(
    a: cutlass.Float32,
    b: cutlass.Float32,
    c: cutlass.Float32,
    d: cutlass.Float32,
    neg_t: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    """fma.rn(a, b, (+-t)) with t = mul.rn(c, d): one explicit contraction of a*b +- c*d."""
    text = "{\n.reg .f32 t, u;\nmul.rn.f32 t, $3, $4;\n"
    text += "neg.f32 t, t;\n" if neg_t else ""
    text += "mov.f32 u, $1;\n"
    text += "fma.rn.f32 $0, u, $2, t;\n}"
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
                cutlass.Float32(d).ir_value(loc=loc, ip=ip),
            ],
            text,
            "=f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _rope(even, odd, cosine, sine):
    """(bf16(fma(even, cos, -(odd * sin))), bf16(fma(odd, cos, even * sin))): the contraction pinned
    explicitly (a DSL ``even * cos - odd * sin`` leaves which product is fused to LLVM)."""
    rot_even = _fma_of_mul(even, cosine, odd, sine, True)
    rot_odd = _fma_of_mul(odd, cosine, even, sine, False)
    return cutlass.BFloat16(rot_even), cutlass.BFloat16(rot_odd)


@dsl_user_op
def _bf16_lo(w: cutlass.Int32, *, loc=None, ip=None) -> cutlass.Float32:
    """FP32 of the BF16 in the low half of w (exact, as BFloat16.to(Float32))."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 t;\nshl.b32 t, $1, 16;\nmov.b32 $0, t;\n}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bf16_hi(w: cutlass.Int32, *, loc=None, ip=None) -> cutlass.Float32:
    """FP32 of the BF16 in the high half of w (exact)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 t;\nand.b32 t, $1, 0xFFFF0000;\nmov.b32 $0, t;\n}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _e4m3x4(
    w0: cutlass.Int32, w1: cutlass.Int32, factor: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Four BF16 (two words, element order) -> four E4M3 bytes in order: per pair, exactly
    _e4m3x2(lo, hi, factor) (mul.rn.f32 by factor, cvt.rn.satfinite.e4m3x2.f32)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
                cutlass.Int32(w1).ir_value(loc=loc, ip=ip),
                cutlass.Float32(factor).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b32 x0, x1, x2, x3;\n.reg .f32 f0, f1, f2, f3;\n.reg .b16 h0, h1;\n"
            "shl.b32 x0, $1, 16;\nand.b32 x1, $1, 0xFFFF0000;\n"
            "shl.b32 x2, $2, 16;\nand.b32 x3, $2, 0xFFFF0000;\n"
            "mov.b32 f0, x0;\nmov.b32 f1, x1;\nmov.b32 f2, x2;\nmov.b32 f3, x3;\n"
            "mul.rn.f32 f0, f0, $3;\nmul.rn.f32 f1, f1, $3;\n"
            "mul.rn.f32 f2, f2, $3;\nmul.rn.f32 f3, f3, $3;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h0, f1, f0;\ncvt.rn.satfinite.e4m3x2.f32 h1, f3, f2;\n"
            "mov.b32 $0, {h0, h1};\n}",
            "=r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def _qknorm_rope_fp8_kernel(
    src: cute.Pointer,
    k_off: cutlass.Int64,
    v_off: cutlass.Int64,
    dst: cute.Pointer,
    parts: cute.Pointer,
    scales: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    rows: cutlass.Int32,
    n_part: cutlass.Int32,
    sm_scale: cutlass.Float32,
    min_scale: cutlass.Float32,
    heads: cutlass.Constexpr,
    src_ld: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    discard: cutlass.Constexpr,
):
    """Mode 1 (amax partials) or mode 2 (scales + E4M3), see the module docstring.

    Warp w < 8 (plane w // 4, heads (w % 4) * per_warp ..): its row slice is per_warp // 2 head
    pairs of 512 B, one 16-B cp.async per lane each, into a per-warp ring of per_warp // 2 slots;
    slot j of row r + grid is re-filled right after row r's head pair j is consumed. Warps 8-11
    (v): per_warp // 2 16-B loads per lane per row (chunks lane, 32 + lane, ..), elementwise.
    discard (mode 2): drop each consumed q/k/v source line from L2.
    """
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = tidx // 32
    lane = tidx % 32
    width = heads * 128
    per_warp = heads // 4
    span = per_warp * 128
    stages = per_warp // 2
    smem = cutlass.utils.SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(3 * 12), byte_alignment=16)
    ring = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(8 * span), byte_alignment=16)
    ring32 = cute.make_tensor(
        cute.recast_ptr(ring.iterator, dtype=cutlass.Int32), cute.make_layout(4 * span)
    )
    stage = smem.allocate_tensor(cutlass.Int16, cute.make_layout(8 * 128), byte_alignment=16)
    stage64 = cute.make_tensor(
        cute.recast_ptr(stage.iterator, dtype=cutlass.Int64), cute.make_layout(8 * 32)
    )
    plane = warp // 4
    head0 = (warp % 4) * per_warp
    plane_off = cutlass.Int64(0)
    if plane == 1:
        plane_off = k_off
    if plane == 2:
        plane_off = v_off
    # Byte addresses of this lane's 16-B chunk in head pair 0 of row 0 (+ j * 512 per pair /
    # chunk group, + row * src_ld * 2 per row), and of this warp's E4M3 slice of row 0.
    g_lane = cutlass.Int64(src.toint()) + (plane_off + head0 * 128) * 2 + lane * 16
    d_warp = (
        cutlass.Int64(dst.toint())
        + cutlass.Int64(plane) * cutlass.Int64(rows) * width
        + head0 * 128
    )
    ring_base = warp * span  # BF16 index of this warp's ring (q/k warps only)
    s_lane = cutlass.Int32(ring.iterator.toint()) + ring_base * 2 + lane * 16
    if warp < 8:
        # Row cta's head pairs go in flight before the partial re-reduce below.
        for s0 in cutlass.range_constexpr(stages):
            _cp_async16(s_lane + s0 * 512, g_lane + cutlass.Int64(cta) * (src_ld * 2) + s0 * 512)
            _cp_async_commit()
    factor = cutlass.Float32(0.0)
    if cutlass.const_expr(mode == 2):
        # Plane by plane: load this plane's <= 3 partials, fold them in the same order, reduce the
        # warp, then the next plane (keeps only 3 loaded values live; max is order-free anyway).
        for p in cutlass.range_constexpr(3):
            loaded = []
            for i in cutlass.range_constexpr(3):
                pidx = tidx + i * 384
                x = cutlass.Float32(_AMAX_INIT)
                if pidx < n_part:
                    x = (parts + p * n_part + pidx).load()
                loaded.append(x)
            part_amax = cutlass.Float32(_AMAX_INIT)
            for i in cutlass.range_constexpr(3):
                part_amax = _max_f32(part_amax, loaded[i])
            for delta in (16, 8, 4, 2, 1):
                part_amax = _max_f32(
                    part_amax,
                    cute.arch.shuffle_sync_op(part_amax, delta, kind=nvvm.ShflKind.bfly),
                )
            if lane == 0:
                red[p * 12 + warp] = part_amax
        cute.arch.sync_threads()
        plane_amax = cutlass.Float32(_AMAX_INIT)
        for w in cutlass.range_constexpr(12):
            plane_amax = _max_f32(plane_amax, red[plane * 12 + w])
        _, factor = _scale_factor(plane_amax, min_scale)
        if cta == 0:
            # scales[0..7] written by 5 threads (same expressions as one thread would use):
            # thread p < 3 -> scales[p], scales[4 + p]; thread 3 -> scales[3]; thread 4 -> [7].
            out_scales = cute.make_tensor(scales, cute.make_layout(8))
            for p in cutlass.range_constexpr(3):
                if tidx == p:
                    amax_p = cutlass.Float32(_AMAX_INIT)
                    for w in cutlass.range_constexpr(12):
                        amax_p = _max_f32(amax_p, red[p * 12 + w])
                    s_p, f_p = _scale_factor(amax_p, min_scale)
                    out_scales[p] = s_p
                    out_scales[4 + p] = f_p
            if tidx == 3:
                descales = []
                for p in cutlass.range_constexpr(2):
                    amax_p = cutlass.Float32(_AMAX_INIT)
                    for w in cutlass.range_constexpr(12):
                        amax_p = _max_f32(amax_p, red[p * 12 + w])
                    s_p, _ = _scale_factor(amax_p, min_scale)
                    descales.append(s_p)
                out_scales[3] = _mul_rn(_mul_rn(descales[0], descales[1]), sm_scale)
            if tidx == 4:
                out_scales[7] = cutlass.Float32(0.0)
    amax = cutlass.Float32(_AMAX_INIT)
    if warp < 8:
        norm = q_norm if plane == 0 else k_norm
        weights = []
        for pair_group in cutlass.range_constexpr(2):
            even_col = 2 * (pair_group * 32 + lane)
            weights.append((norm + even_col).load().to(cutlass.Float32))
            weights.append((norm + even_col + 1).load().to(cutlass.Float32))
        for row in cutlass.range(cta, rows, grid):
            position = (positions + row).load()
            cs_row = cos_sin + position * 128
            cosines = [(cs_row + lane).load(), (cs_row + 32 + lane).load()]
            sines = [(cs_row + 64 + lane).load(), (cs_row + 96 + lane).load()]
            next_row = row + grid
            d_row = d_warp + cutlass.Int64(row) * width
            for sj in cutlass.range_constexpr(stages):
                _cp_async_wait(stages - 1)
                _syncwarp()
                for hh in cutlass.range_constexpr(2):
                    base = ring_base + sj * 256 + hh * 128
                    sum_squares = cutlass.Float32(0.0)
                    for group in cutlass.range_constexpr(4):
                        value = ring[base + group * 32 + lane].to(cutlass.Float32)
                        sum_squares += value * value
                    for delta in (16, 8, 4, 2, 1):
                        sum_squares += cute.arch.shuffle_sync_op(
                            sum_squares, delta, kind=nvvm.ShflKind.bfly
                        )
                    inv_rms = cute.rsqrt(
                        sum_squares / cutlass.Float32(128) + cutlass.Float32(1.0e-6)
                    )
                    for pair_group in cutlass.range_constexpr(2):
                        word = ring32[base // 2 + pair_group * 32 + lane]
                        even = _bf16_lo(word) * inv_rms * weights[2 * pair_group]
                        odd = _bf16_hi(word) * inv_rms * weights[2 * pair_group + 1]
                        rot_even, rot_odd = _rope(even, odd, cosines[pair_group], sines[pair_group])
                        if cutlass.const_expr(mode == 1):
                            amax = _max_abs(amax, rot_even.to(cutlass.Float32))
                            amax = _max_abs(amax, rot_odd.to(cutlass.Float32))
                        else:
                            stage[warp * 128 + hh * 64 + pair_group * 32 + lane] = _e4m3x2(
                                rot_even.to(cutlass.Float32), rot_odd.to(cutlass.Float32), factor
                            )
                _syncwarp()
                if cutlass.const_expr(mode == 2 and discard):
                    # Every lane's chunk of this head pair landed (wait_group + syncwarp): drop
                    # its 4 lines, one lane per 128 B.
                    if lane % 8 == 0:
                        _discard_l2(g_lane + cutlass.Int64(row) * (src_ld * 2) + sj * 512)
                if next_row < rows:
                    _cp_async16(
                        s_lane + sj * 512,
                        g_lane + cutlass.Int64(next_row) * (src_ld * 2) + sj * 512,
                    )
                _cp_async_commit()
                if cutlass.const_expr(mode == 2):
                    _st_global_b64(d_row + sj * 256 + lane * 8, stage64[warp * 32 + lane])
    else:
        for row in cutlass.range(cta, rows, grid):
            g_row = g_lane + cutlass.Int64(row) * (src_ld * 2)
            words = []
            for c in cutlass.range_constexpr(stages):
                words.append(_ld_global_v4(g_row + c * 512))
            if cutlass.const_expr(mode == 1):
                for c in cutlass.range_constexpr(stages):
                    for i in cutlass.range_constexpr(4):
                        amax = _max_abs(amax, _bf16_lo(words[c][i]))
                        amax = _max_abs(amax, _bf16_hi(words[c][i]))
            else:
                d_row = d_warp + cutlass.Int64(row) * width
                for c in cutlass.range_constexpr(stages):
                    _st_global_v2(
                        d_row + c * 256 + lane * 8,
                        _e4m3x4(words[c][0], words[c][1], factor),
                        _e4m3x4(words[c][2], words[c][3], factor),
                    )
                if cutlass.const_expr(discard):
                    # The stores above consumed every lane's loaded words: drop the v lines.
                    _syncwarp()
                    if lane % 8 == 0:
                        for c in cutlass.range_constexpr(stages):
                            _discard_l2(g_row + c * 512)
    if cutlass.const_expr(mode == 1):
        for delta in (16, 8, 4, 2, 1):
            amax = _max_f32(amax, cute.arch.shuffle_sync_op(amax, delta, kind=nvvm.ShflKind.bfly))
        if lane == 0:
            red[warp] = amax
        cute.arch.sync_threads()
        cta_amax = cutlass.Float32(_AMAX_INIT)
        if tidx < 3:
            for w in cutlass.range_constexpr(4):
                cta_amax = _max_f32(cta_amax, red[tidx * 4 + w])
            partial = cute.make_tensor(parts + tidx * n_part + cta, cute.make_layout(1))
            partial[0] = cta_amax


@cute.jit
def _launch(
    src: cute.Pointer,
    k_off: cutlass.Int64,
    v_off: cutlass.Int64,
    dst: cute.Pointer,
    parts: cute.Pointer,
    scales: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    rows: cutlass.Int32,
    n_part: cutlass.Int32,
    grid: cutlass.Int32,
    sm_scale: cutlass.Float32,
    min_scale: cutlass.Float32,
    heads: cutlass.Constexpr,
    src_ld: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    discard: cutlass.Constexpr,
    stream,
):
    _qknorm_rope_fp8_kernel(
        src,
        k_off,
        v_off,
        dst,
        parts,
        scales,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        rows,
        n_part,
        sm_scale,
        min_scale,
        heads,
        src_ld,
        mode,
        discard,
    ).launch(
        grid=[grid, 1, 1],
        block=[12 * 32, 1, 1],
        stream=stream,
        min_blocks_per_mp=_CTAS_PER_SM,
    )


def _pass(mode, heads, src_ld, ptrs, rows, n_part, grid, sm_scale, discard, device_index):
    key = (device_index, heads, src_ld, mode, discard)
    stream = cutlass_torch.current_stream()
    rt = (
        cutlass.Int32(rows),
        cutlass.Int32(n_part),
        cutlass.Int32(grid),
        cutlass.Float32(sm_scale),
        cutlass.Float32(_MIN_SCALING_FACTOR),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(_launch, *ptrs, *rt, heads, src_ld, mode, discard, stream)
        _COMPILED[key] = compiled
    compiled(*ptrs, *rt, stream)


def launch_qknorm_rope_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(qkv8 [3, M, W] float8_e4m3fn, scales [8] float32)``; see the module docstring and the
    ``qknorm_rope_fp8`` entry. q, k, v: BF16 ``[M, W]`` column views sharing one row stride."""
    rows, width = q.shape[0], q.shape[1]
    heads = width // 128
    if heads % 8 or width % 128 or rows <= 0:
        raise ValueError(f"unsupported QKV shape {tuple(q.shape)} (W = 128 * H, H % 8 == 0)")
    if tuple(k.shape) != (rows, width) or tuple(v.shape) != (rows, width):
        raise ValueError("q, k and v must have the same [M, W] shape")
    src_ld = q.stride(0)
    if k.stride(0) != src_ld or v.stride(0) != src_ld or q.stride(1) != 1:
        raise ValueError("q, k and v must share a row stride")
    if src_ld % 8 or any(t.data_ptr() % 16 for t in (q, k, v)):
        raise ValueError("q, k and v rows must be 16-B aligned")
    device = q.device
    qkv8 = torch.empty((3, rows, width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(8, dtype=torch.float32, device=device)
    grid = _grid(device)
    n_part = max(1, min(rows, grid, _MAX_PARTIALS))
    parts = torch.empty((3, n_part), dtype=torch.float32, device=device)
    k_off = (k.data_ptr() - q.data_ptr()) // q.element_size()
    v_off = (v.data_ptr() - q.data_ptr()) // q.element_size()
    ptrs = (
        make_ptr(cutlass.BFloat16, q.data_ptr(), assumed_align=16),
        cutlass.Int64(k_off),
        cutlass.Int64(v_off),
        make_ptr(cutlass.Int16, qkv8.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float32, parts.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float32, scales.data_ptr(), assumed_align=16),
        make_ptr(cutlass.BFloat16, q_norm.data_ptr(), assumed_align=16),
        make_ptr(cutlass.BFloat16, k_norm.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float32, cos_sin.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Int64, positions.data_ptr(), assumed_align=16),
    )
    # Every warp's row slice (heads // 4 heads of 256 B) starts on a 128-B line in all planes.
    discard = (src_ld * 2) % 128 == 0 and all(t.data_ptr() % 128 == 0 for t in (q, k, v))
    index = device.index if device.index is not None else torch.cuda.current_device()
    # Mode 1 runs n_part CTAs, so it writes every partial mode 2 reads.
    _pass(1, heads, src_ld, ptrs, rows, n_part, n_part, sm_scale, False, index)
    _pass(2, heads, src_ld, ptrs, rows, n_part, max(1, min(rows, grid)), sm_scale, discard, index)
    return qkv8, scales


# Opaque to Dynamo, as every other launch in the compiled blocks: traced, the launcher's host
# code (CuTe DSL pointers, the current-stream lookup) guards on the stream and recompiles under
# CUDA graph capture.
@torch.library.custom_op(
    "trtllm::flux3_klein_qknorm_rope_fp8", mutates_args=(), device_types="cuda"
)
def qknorm_rope_fp8_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return launch_qknorm_rope_fp8(q, k, v, q_norm, k_norm, cos_sin, positions, sm_scale)


@qknorm_rope_fp8_op.register_fake
def _qknorm_rope_fp8_fake(q, k, v, q_norm, k_norm, cos_sin, positions, sm_scale):
    rows, width = q.shape
    return (
        q.new_empty((3, rows, width), dtype=torch.float8_e4m3fn),
        q.new_empty((8,), dtype=torch.float32),
    )
