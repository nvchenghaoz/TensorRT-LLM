# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 SM120 NVFP4 QKV straight to per-tensor E4M3: Q/K RMSNorm + RoPE -> E4M3, no BF16 q/k.

Main's chain after the planar QKV GEMMs is: qknorm_rope (flux3_sm120_qknorm_rope.py) writes BF16
norm+RoPE q/k planes in place, then fp8_sm12x's K1 (_qkv_amax_partials_kernel) reads q/k/v for the
per-tensor amax and K2 (_qkv_quantize_kernel) reads them again to write E4M3 q/k/v. Here:

- pass 1 (mode 1): the same per-(row, head) norm+RoPE arithmetic as _qknorm_rope_kernel (copied
  expression for expression), rounding each result to BF16 exactly as main stores it, but only
  folding |value| into a per-warp abs-max (max.f32, NaN ignored like Triton's maxnumf). Warps 8-11
  fold |v| (the GEMM's v plane, untouched by main's pass). One FP32 partial per (plane, row, warp)
  is written; no BF16 store.
- reduce (Triton, one program): the max of each plane's partials (max is order-free, so the amax is
  exactly main's), then scale/factor exactly as fp8_sm12x._scale_of:
  scale = bf16(max(div_rn(amax, 448), 1/(448*512))), factor = div_rn(1, scale); scales[0..2] are
  the descales, scales[3] = (q_descale * k_descale) * sm_scale, scales[4..6] the factors.
- pass 2 (mode 2): recomputes the identical BF16 values, multiplies by the plane's factor
  (mul.rn.f32, as K2's x.to(f32) * factor) and converts pairs with cvt.rn.satfinite.e4m3x2.f32 (the
  instruction Triton's .to(tl.float8e4nv) emits), stored as Int16 into three contiguous [M, W]
  E4M3 planes (q, k, v).

So the E4M3 q/k/v and the four FP32 scalars are bitwise what quantize_qkv_e4m3 gives on main's BF16
planes. One CTA per row, 12 warps (0-3 q heads, 4-7 k heads, 8-11 v), runtime row count.
"""

import os

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

# FP32 constant as the C++ kernel computes it: 1.0f / (448.f * 512.f) (= fp8_sm12x's).
from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR

_COMPILED = {}
_PARTS_PER_ROW = 4  # one partial per warp of a plane
_REDUCE_BLOCK = 4096
_REDUCE_WARPS = 16

# Folded scale reduce (idea 182). TRTLLM_FLUX3_QKV_FP8_FOLD=0 restores landing 54's three launches
# (mode 1 per-(row, warp) partials, one-program Triton reduce, mode 2) byte for byte.
# Fold: mode 1 runs a grid-stride row loop and writes ONE partial per (plane, CTA) into an FP32
# [3, n_part] buffer (n_part = its grid, <= 1024); mode 2's CTAs each re-reduce all 3 planes'
# partials (n_part <= 1024 per plane, a runtime arg) and compute scale/factor in-kernel exactly as
# _scale_from_amax; CTA 0 writes scales[0..7]. No reduce launch, no persistent state.
FOLD = os.environ.get("TRTLLM_FLUX3_QKV_FP8_FOLD", "1").strip() != "0"
_AMAX_INIT = 0.0  # landing 54 mode 1's amax init; every fold starts from it (all-NaN plane -> 0)
_MAX_PARTIALS = 1024
_FOLD_GRID1 = int(os.environ.get("TRTLLM_FLUX3_QKV_FP8_FOLD_G1", "752"))
_FOLD_GRID2 = int(os.environ.get("TRTLLM_FLUX3_QKV_FP8_FOLD_G2", "564"))


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


@cute.kernel
def _qknorm_rope_fp8_kernel(
    src: cute.Pointer,
    k_off: cutlass.Int64,
    v_off: cutlass.Int64,
    dst: cute.Pointer,
    aux: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    rows: cutlass.Int32,
    heads: cutlass.Constexpr,
    src_ld: cutlass.Constexpr,
    mode: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    row64 = cutlass.Int64(row)
    width = heads * 128
    per_warp = heads // 4
    span = per_warp * 128
    # mode 2: Int16 (two E4M3) offsets of this row in a plane and between planes.
    dst_row = row64 * (width // 2)
    plane_stride = cutlass.Int64(rows) * (width // 2)
    parts = cutlass.Int64(rows) * _PARTS_PER_ROW
    amax = cutlass.Float32(0.0)
    if warp < 8:
        plane = warp // 4
        head0 = (warp % 4) * per_warp
        norm = q_norm if plane == 0 else k_norm
        offset = k_off if plane == 1 else cutlass.Int64(0)
        source = cute.make_tensor(
            src + offset + row64 * src_ld + head0 * 128, cute.make_layout(span)
        )
        output = cute.make_tensor(
            dst + cutlass.Int64(plane) * plane_stride + dst_row + head0 * 64,
            cute.make_layout(span // 2),
        )
        factor = (aux + 4 + plane).load() if cutlass.const_expr(mode == 2) else amax
        position = (positions + row).load()
        values = []
        pairs = []
        for h in cutlass.range_constexpr(per_warp):
            for group in cutlass.range_constexpr(4):
                values.append(source[h * 128 + group * 32 + lane])
            for pair_group in cutlass.range_constexpr(2):
                even_col = 2 * (pair_group * 32 + lane)
                pairs.append(source[h * 128 + even_col])
                pairs.append(source[h * 128 + even_col + 1])
        for h in cutlass.range_constexpr(per_warp):
            sum_squares = cutlass.Float32(0.0)
            for group in cutlass.range_constexpr(4):
                value = values[h * 4 + group].to(cutlass.Float32)
                sum_squares += value * value
            for delta in (16, 8, 4, 2, 1):
                sum_squares += cute.arch.shuffle_sync_op(
                    sum_squares, delta, kind=nvvm.ShflKind.bfly
                )
            inv_rms = cute.rsqrt(sum_squares / cutlass.Float32(128) + cutlass.Float32(1.0e-6))
            for pair_group in cutlass.range_constexpr(2):
                pair = pair_group * 32 + lane
                even_col = 2 * pair
                even = (
                    pairs[h * 4 + 2 * pair_group].to(cutlass.Float32)
                    * inv_rms
                    * (norm + even_col).load().to(cutlass.Float32)
                )
                odd = (
                    pairs[h * 4 + 2 * pair_group + 1].to(cutlass.Float32)
                    * inv_rms
                    * (norm + even_col + 1).load().to(cutlass.Float32)
                )
                cosine = (cos_sin + position * 128 + pair).load()
                sine = (cos_sin + position * 128 + 64 + pair).load()
                rot_even = cutlass.BFloat16(even * cosine - odd * sine)
                rot_odd = cutlass.BFloat16(odd * cosine + even * sine)
                if cutlass.const_expr(mode == 1):
                    amax = _max_abs(amax, rot_even.to(cutlass.Float32))
                    amax = _max_abs(amax, rot_odd.to(cutlass.Float32))
                else:
                    output[h * 64 + pair] = _e4m3x2(
                        rot_even.to(cutlass.Float32), rot_odd.to(cutlass.Float32), factor
                    )
        if cutlass.const_expr(mode == 1):
            for delta in (16, 8, 4, 2, 1):
                amax = _max_f32(
                    amax, cute.arch.shuffle_sync_op(amax, delta, kind=nvvm.ShflKind.bfly)
                )
            if lane == 0:
                partial = cute.make_tensor(
                    aux + cutlass.Int64(plane) * parts + row64 * _PARTS_PER_ROW + warp % 4,
                    cute.make_layout(1),
                )
                partial[0] = amax
    else:
        head0 = (warp - 8) * per_warp
        source = cute.make_tensor(
            src + v_off + row64 * src_ld + head0 * 128, cute.make_layout(span)
        )
        if cutlass.const_expr(mode == 1):
            loaded = []
            for i in cutlass.range_constexpr(per_warp * 4):
                loaded.append(source[i * 32 + lane])
            for i in cutlass.range_constexpr(per_warp * 4):
                amax = _max_abs(amax, loaded[i].to(cutlass.Float32))
            for delta in (16, 8, 4, 2, 1):
                amax = _max_f32(
                    amax, cute.arch.shuffle_sync_op(amax, delta, kind=nvvm.ShflKind.bfly)
                )
            if lane == 0:
                partial = cute.make_tensor(
                    aux + 2 * parts + row64 * _PARTS_PER_ROW + (warp - 8),
                    cute.make_layout(1),
                )
                partial[0] = amax
        else:
            factor_v = (aux + 6).load()
            output_v = cute.make_tensor(
                dst + 2 * plane_stride + dst_row + head0 * 64, cute.make_layout(span // 2)
            )
            loaded = []
            for i in cutlass.range_constexpr(per_warp * 2):
                j = i * 32 + lane
                loaded.append(source[2 * j])
                loaded.append(source[2 * j + 1])
            for i in cutlass.range_constexpr(per_warp * 2):
                output_v[i * 32 + lane] = _e4m3x2(
                    loaded[2 * i].to(cutlass.Float32),
                    loaded[2 * i + 1].to(cutlass.Float32),
                    factor_v,
                )


@cute.jit
def _launch(
    src: cute.Pointer,
    k_off: cutlass.Int64,
    v_off: cutlass.Int64,
    dst: cute.Pointer,
    aux: cute.Pointer,
    q_norm: cute.Pointer,
    k_norm: cute.Pointer,
    cos_sin: cute.Pointer,
    positions: cute.Pointer,
    rows: cutlass.Int32,
    heads: cutlass.Constexpr,
    src_ld: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    stream,
):
    _qknorm_rope_fp8_kernel(
        src, k_off, v_off, dst, aux, q_norm, k_norm, cos_sin, positions, rows, heads, src_ld,
        mode,
    ).launch(grid=[rows, 1, 1], block=[12 * 32, 1, 1], stream=stream)


# ----------------------------------------------------------------------------- folded reduce (182)
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
    """_scale_from_amax: scale = bf16(max(div_rn(amax, 448), min_scale)), factor = div_rn(1, scale)."""
    scale = _max_f32(_div_rn(amax, cutlass.Float32(448.0)), min_scale)
    scale = cutlass.BFloat16(scale).to(cutlass.Float32)
    return scale, _div_rn(cutlass.Float32(1.0), scale)


@cute.kernel
def _qknorm_rope_fp8_fold_kernel(
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
):
    """Landing 54's per-row arithmetic (instruction for instruction) in a grid-stride row loop.

    mode 1: per-warp |x| max over all of this CTA's rows, then one partial per (plane, CTA):
            parts[plane * n_part + cta] (n_part == grid size).
    mode 2: re-reduce parts [3, n_part] (n_part <= 1024, any producer), scale/factor in-kernel,
            CTA 0 writes scales[0..7]; then the rows' E4M3 stores.
    """
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = tidx // 32
    lane = tidx % 32
    width = heads * 128
    per_warp = heads // 4
    span = per_warp * 128
    plane_stride = cutlass.Int64(rows) * (width // 2)
    smem = cutlass.utils.SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(3 * 12), byte_alignment=16)
    factor = cutlass.Float32(0.0)
    if cutlass.const_expr(mode == 2):
        # Every CTA folds all three planes' partials (coalesced, <= 3 loads per thread per plane).
        loaded = []
        for p in cutlass.range_constexpr(3):
            for i in cutlass.range_constexpr(3):
                j = tidx + i * 384
                x = cutlass.Float32(_AMAX_INIT)
                if j < n_part:
                    x = (parts + p * n_part + j).load()
                loaded.append(x)
        for p in cutlass.range_constexpr(3):
            acc = cutlass.Float32(_AMAX_INIT)
            for i in cutlass.range_constexpr(3):
                acc = _max_f32(acc, loaded[p * 3 + i])
            for delta in (16, 8, 4, 2, 1):
                acc = _max_f32(acc, cute.arch.shuffle_sync_op(acc, delta, kind=nvvm.ShflKind.bfly))
            if lane == 0:
                red[p * 12 + warp] = acc
        cute.arch.sync_threads()
        my_plane = warp // 4
        plane_amax = cutlass.Float32(_AMAX_INIT)
        for w in cutlass.range_constexpr(12):
            plane_amax = _max_f32(plane_amax, red[my_plane * 12 + w])
        _, factor = _scale_factor(plane_amax, min_scale)
        if cta == 0:
            if tidx == 0:
                out_scales = cute.make_tensor(scales, cute.make_layout(8))
                descales = []
                factors = []
                for p in cutlass.range_constexpr(3):
                    amax_p = cutlass.Float32(_AMAX_INIT)
                    for w in cutlass.range_constexpr(12):
                        amax_p = _max_f32(amax_p, red[p * 12 + w])
                    s_p, f_p = _scale_factor(amax_p, min_scale)
                    descales.append(s_p)
                    factors.append(f_p)
                out_scales[0] = descales[0]
                out_scales[1] = descales[1]
                out_scales[2] = descales[2]
                out_scales[3] = _mul_rn(_mul_rn(descales[0], descales[1]), sm_scale)
                out_scales[4] = factors[0]
                out_scales[5] = factors[1]
                out_scales[6] = factors[2]
                out_scales[7] = cutlass.Float32(0.0)
    amax = cutlass.Float32(_AMAX_INIT)
    if warp < 8:
        plane = warp // 4
        head0 = (warp % 4) * per_warp
        norm = q_norm if plane == 0 else k_norm
        offset = k_off if plane == 1 else cutlass.Int64(0)
        for row in cutlass.range(cta, rows, grid):
            row64 = cutlass.Int64(row)
            source = cute.make_tensor(
                src + offset + row64 * src_ld + head0 * 128, cute.make_layout(span)
            )
            output = cute.make_tensor(
                dst + cutlass.Int64(plane) * plane_stride + row64 * (width // 2) + head0 * 64,
                cute.make_layout(span // 2),
            )
            position = (positions + row).load()
            values = []
            pairs = []
            for h in cutlass.range_constexpr(per_warp):
                for group in cutlass.range_constexpr(4):
                    values.append(source[h * 128 + group * 32 + lane])
                for pair_group in cutlass.range_constexpr(2):
                    even_col = 2 * (pair_group * 32 + lane)
                    pairs.append(source[h * 128 + even_col])
                    pairs.append(source[h * 128 + even_col + 1])
            for h in cutlass.range_constexpr(per_warp):
                sum_squares = cutlass.Float32(0.0)
                for group in cutlass.range_constexpr(4):
                    value = values[h * 4 + group].to(cutlass.Float32)
                    sum_squares += value * value
                for delta in (16, 8, 4, 2, 1):
                    sum_squares += cute.arch.shuffle_sync_op(
                        sum_squares, delta, kind=nvvm.ShflKind.bfly
                    )
                inv_rms = cute.rsqrt(sum_squares / cutlass.Float32(128) + cutlass.Float32(1.0e-6))
                for pair_group in cutlass.range_constexpr(2):
                    pair = pair_group * 32 + lane
                    even_col = 2 * pair
                    even = (
                        pairs[h * 4 + 2 * pair_group].to(cutlass.Float32)
                        * inv_rms
                        * (norm + even_col).load().to(cutlass.Float32)
                    )
                    odd = (
                        pairs[h * 4 + 2 * pair_group + 1].to(cutlass.Float32)
                        * inv_rms
                        * (norm + even_col + 1).load().to(cutlass.Float32)
                    )
                    cosine = (cos_sin + position * 128 + pair).load()
                    sine = (cos_sin + position * 128 + 64 + pair).load()
                    rot_even = cutlass.BFloat16(even * cosine - odd * sine)
                    rot_odd = cutlass.BFloat16(odd * cosine + even * sine)
                    if cutlass.const_expr(mode == 1):
                        amax = _max_abs(amax, rot_even.to(cutlass.Float32))
                        amax = _max_abs(amax, rot_odd.to(cutlass.Float32))
                    else:
                        output[h * 64 + pair] = _e4m3x2(
                            rot_even.to(cutlass.Float32), rot_odd.to(cutlass.Float32), factor
                        )
    else:
        head0 = (warp - 8) * per_warp
        for row in cutlass.range(cta, rows, grid):
            row64 = cutlass.Int64(row)
            source = cute.make_tensor(
                src + v_off + row64 * src_ld + head0 * 128, cute.make_layout(span)
            )
            if cutlass.const_expr(mode == 1):
                loaded_v = []
                for i in cutlass.range_constexpr(per_warp * 4):
                    loaded_v.append(source[i * 32 + lane])
                for i in cutlass.range_constexpr(per_warp * 4):
                    amax = _max_abs(amax, loaded_v[i].to(cutlass.Float32))
            else:
                output_v = cute.make_tensor(
                    dst + 2 * plane_stride + row64 * (width // 2) + head0 * 64,
                    cute.make_layout(span // 2),
                )
                loaded_v = []
                for i in cutlass.range_constexpr(per_warp * 2):
                    j = i * 32 + lane
                    loaded_v.append(source[2 * j])
                    loaded_v.append(source[2 * j + 1])
                for i in cutlass.range_constexpr(per_warp * 2):
                    output_v[i * 32 + lane] = _e4m3x2(
                        loaded_v[2 * i].to(cutlass.Float32),
                        loaded_v[2 * i + 1].to(cutlass.Float32),
                        factor,
                    )
    if cutlass.const_expr(mode == 1):
        for delta in (16, 8, 4, 2, 1):
            amax = _max_f32(amax, cute.arch.shuffle_sync_op(amax, delta, kind=nvvm.ShflKind.bfly))
        if lane == 0:
            red[warp] = amax
        cute.arch.sync_threads()
        if tidx < 3:
            acc = cutlass.Float32(_AMAX_INIT)
            for w in cutlass.range_constexpr(4):
                acc = _max_f32(acc, red[tidx * 4 + w])
            partial = cute.make_tensor(parts + tidx * n_part + cta, cute.make_layout(1))
            partial[0] = acc


@cute.jit
def _launch_fold(
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
    stream,
):
    _qknorm_rope_fp8_fold_kernel(
        src, k_off, v_off, dst, parts, scales, q_norm, k_norm, cos_sin, positions, rows, n_part,
        sm_scale, min_scale, heads, src_ld, mode,
    ).launch(grid=[grid, 1, 1], block=[12 * 32, 1, 1], stream=stream)


def _fold_pass(mode, key_base, args, rows, n_part, grid, sm_scale):
    key = key_base + ("fold", mode)
    stream = cutlass_torch.current_stream()
    compiled = _COMPILED.get(key)
    src, k_off, v_off, dst, parts, scales, qn, kn, cs, pos = args
    heads, src_ld = key_base[1], key_base[2]
    rt = (
        cutlass.Int32(rows), cutlass.Int32(n_part), cutlass.Int32(grid),
        cutlass.Float32(sm_scale), cutlass.Float32(_MIN_SCALING_FACTOR),
    )
    if compiled is None:
        compiled = cute.compile(
            _launch_fold, src, k_off, v_off, dst, parts, scales, qn, kn, cs, pos, *rt, heads,
            src_ld, mode, stream,
        )
        _COMPILED[key] = compiled
    compiled(src, k_off, v_off, dst, parts, scales, qn, kn, cs, pos, *rt, stream)


@triton.jit
def _plane_amax(part_ptr, base, n_part, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_part, BLOCK):
        p = start + offs
        acc = tl.maximum(acc, tl.load(part_ptr + base + p, mask=p < n_part, other=0.0))
    return tl.max(acc, axis=0)


@triton.jit
def _scale_from_amax(amax, min_scale):
    # fp8_sm12x._scale_of after its amax reduction.
    scale = tl.maximum(tl.math.div_rn(amax, tl.full([], 448.0, tl.float32)), min_scale)
    scale = scale.to(tl.bfloat16).to(tl.float32)
    factor = tl.math.div_rn(tl.full([], 1.0, tl.float32), scale)
    return scale, factor


@triton.jit
def _fp8_scales_kernel(part_ptr, scales_ptr, n_part, sm_scale, min_scale, BLOCK: tl.constexpr):
    sq, fq = _scale_from_amax(_plane_amax(part_ptr, 0, n_part, BLOCK), min_scale)
    sk, fk = _scale_from_amax(_plane_amax(part_ptr, n_part, n_part, BLOCK), min_scale)
    sv, fv = _scale_from_amax(_plane_amax(part_ptr, 2 * n_part, n_part, BLOCK), min_scale)
    tl.store(scales_ptr + 0, sq)
    tl.store(scales_ptr + 1, sk)
    tl.store(scales_ptr + 2, sv)
    tl.store(scales_ptr + 3, (sq * sk) * sm_scale)
    tl.store(scales_ptr + 4, fq)
    tl.store(scales_ptr + 5, fk)
    tl.store(scales_ptr + 6, fv)
    tl.store(scales_ptr + 7, 0.0)


def _pass(mode, key_base, args, rows):
    key = key_base + (mode,)
    stream = cutlass_torch.current_stream()
    compiled = _COMPILED.get(key)
    src, k_off, v_off, dst, aux, qn, kn, cs, pos = args
    heads, src_ld = key_base[1], key_base[2]
    if compiled is None:
        compiled = cute.compile(
            _launch, src, k_off, v_off, dst, aux, qn, kn, cs, pos, cutlass.Int32(rows), heads,
            src_ld, mode, stream,
        )
        _COMPILED[key] = compiled
    compiled(src, k_off, v_off, dst, aux, qn, kn, cs, pos, cutlass.Int32(rows), stream)


def qknorm_rope_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    sm_scale: float,
    external_partials: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 of (norm+RoPE(q), norm+RoPE(k), v) from the BF16 GEMM output.

    external_partials: optional FP32 [3, n_part] amax partials from another producer (plan 1233,
    e.g. a GEMM epilogue); mode 1 is then skipped and only the folded mode 2 runs.

    q, k, v: BF16 [M, W] row-major views sharing one row stride (W = H*128, H % 4 == 0).
    Returns (qkv8, scales): qkv8 float8_e4m3fn [3, M, W] (contiguous q, k, v planes) and FP32
    scales[8] = [q_descale, k_descale, v_descale, q_descale*k_descale*sm_scale, q_factor,
    k_factor, v_factor, 0]. Bitwise equal to qknorm_rope (BF16) followed by
    fp8_sm12x.quantize_qkv_e4m3 on the same rows. q/k/v are only read.
    """
    rows, width = q.shape[0], q.shape[1]
    heads = width // 128
    if heads % 4 or width % 128 or rows <= 0:
        raise ValueError(f"unsupported QKV shape {tuple(q.shape)}")
    if tuple(k.shape) != (rows, width) or tuple(v.shape) != (rows, width):
        raise ValueError("q, k and v must have the same [M, W] shape")
    src_ld = q.stride(0)
    if k.stride(0) != src_ld or v.stride(0) != src_ld or q.stride(1) != 1:
        raise ValueError("q, k and v must share a row stride")
    device = q.device
    qkv8 = torch.empty((3, rows, width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(8, dtype=torch.float32, device=device)
    k_off = (k.data_ptr() - q.data_ptr()) // q.element_size()
    v_off = (v.data_ptr() - q.data_ptr()) // q.element_size()
    src = make_ptr(cutlass.BFloat16, q.data_ptr(), assumed_align=16)
    dst = make_ptr(cutlass.Int16, qkv8.data_ptr(), assumed_align=16)
    qn = make_ptr(cutlass.BFloat16, q_norm.data_ptr(), assumed_align=16)
    kn = make_ptr(cutlass.BFloat16, k_norm.data_ptr(), assumed_align=16)
    cs = make_ptr(cutlass.Float32, cos_sin.data_ptr(), assumed_align=16)
    pos = make_ptr(cutlass.Int64, positions.data_ptr(), assumed_align=16)
    key_base = (device.index, heads, src_ld)
    common = (cutlass.Int64(k_off), cutlass.Int64(v_off))
    scale_ptr = make_ptr(cutlass.Float32, scales.data_ptr(), assumed_align=16)
    if FOLD or external_partials is not None:
        if external_partials is None:
            # Mode 1 emits the plan-1233 contract: FP32 [3, n_part], n_part = its grid <= 1024.
            n_part = max(1, min(rows, _FOLD_GRID1, _MAX_PARTIALS))
            fold_parts = torch.empty((3, n_part), dtype=torch.float32, device=device)
        else:
            fold_parts = external_partials
            n_part = fold_parts.shape[1]
            if (fold_parts.dim() != 2 or fold_parts.shape[0] != 3 or not 0 < n_part <= _MAX_PARTIALS
                    or fold_parts.dtype != torch.float32 or not fold_parts.is_contiguous()):
                raise ValueError("partials must be contiguous FP32 [3, n_part], n_part <= 1024")
        part_ptr = make_ptr(cutlass.Float32, fold_parts.data_ptr(), assumed_align=16)
        ptrs = (src, *common, dst, part_ptr, scale_ptr, qn, kn, cs, pos)
        if external_partials is None:
            _fold_pass(1, key_base, ptrs, rows, n_part, n_part, sm_scale)
        _fold_pass(2, key_base, ptrs, rows, n_part, max(1, min(rows, _FOLD_GRID2)), sm_scale)
        return qkv8, scales
    partials = torch.empty(3 * rows * _PARTS_PER_ROW, dtype=torch.float32, device=device)
    part_ptr = make_ptr(cutlass.Float32, partials.data_ptr(), assumed_align=16)
    scale_ptr = make_ptr(cutlass.Float32, scales.data_ptr(), assumed_align=16)
    _pass(1, key_base, (src, *common, dst, part_ptr, qn, kn, cs, pos), rows)
    n_part = rows * _PARTS_PER_ROW
    _fp8_scales_kernel[(1,)](
        partials, scales, n_part, sm_scale, _MIN_SCALING_FACTOR, BLOCK=_REDUCE_BLOCK,
        num_warps=_REDUCE_WARPS,
    )
    _pass(2, key_base, (src, *common, dst, scale_ptr, qn, kn, cs, pos), rows)
    return qkv8, scales
