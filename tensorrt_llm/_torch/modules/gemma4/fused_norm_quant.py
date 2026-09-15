# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma4 post-attention norm/add and pre-MLP norm/FP4 preparation."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout


@gluon.jit
def _gemma4_norm_add_fp4_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    next_weight_ptr,
    global_scale_ptr,
    output_ptr,
    fp4_ptr,
    sf_ptr,
    eps,
    next_eps,
    SX: gl.constexpr,
    SR: gl.constexpr,
    H: gl.constexpr,
    NORM_THREADS: gl.constexpr,
    BLOCK_H: gl.constexpr,
):
    gl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    row = gl.program_id(0)
    # Match the existing Triton post-attention norm's reduction layout.
    cols = gl.arange(
        0, BLOCK_H, layout=BlockedLayout([max(1, min(8, BLOCK_H // 128))], [32], [4], [0])
    )
    x = gl.load(x_ptr + row * SX + cols, cols < H, 0).to(gl.float32)
    weight = gl.load(weight_ptr + cols, cols < H, 0).to(gl.float32)
    residual = gl.load(residual_ptr + row * SR + cols, cols < H, 0).to(gl.float32)
    sum_sq = gl.sum(x * x, axis=0)
    rstd = gl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [sum_sq / H + eps],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    normed = (x * rstd * weight).to(gl.bfloat16)
    output = (residual + normed.to(gl.float32)).to(gl.bfloat16)
    gl.store(output_ptr + row * H + cols, output, cols < H)

    next_input = output.to(gl.float32)
    # CuTe uses eight adjacent elements per thread vector,
    # sequential local accumulation, then increasing-offset butterfly sums.
    samples = gl.reshape(
        gl.permute(
            gl.reshape(next_input, (BLOCK_H // (NORM_THREADS * 8), NORM_THREADS, 8)), (0, 2, 1)
        ),
        (BLOCK_H // NORM_THREADS, NORM_THREADS),
    )
    samples = gl.convert_layout(samples, BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
    lane_layout: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    lanes = gl.arange(0, NORM_THREADS, layout=lane_layout)
    partial = gl.full((NORM_THREADS,), 0.0, gl.float32, layout=lane_layout)
    for j in gl.static_range(triton.cdiv(H, NORM_THREADS * 8) * 8):
        index = gl.full(
            (1, NORM_THREADS),
            j,
            gl.int32,
            layout=BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]),
        )
        value = gl.reshape(gl.gather(samples, index, axis=0), (NORM_THREADS,))
        value = gl.convert_layout(value, lane_layout)
        partial = gl.fma(value, value, partial)
    for shift in gl.static_range(5):
        if (1 << shift) < NORM_THREADS:
            other = gl.inline_asm_elementwise(
                "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
                "=f,f,r",
                [partial, 1 << shift],
                dtype=gl.float32,
                is_pure=True,
                pack=1,
            )
            partial = partial + other
    next_sum_sq = gl.sum(gl.where(lanes % 32 == 0, partial, 0.0), axis=0)
    # Preserve CuTe's rounded division and separate epsilon addition.
    # Multiplying by a reciprocal and fusing epsilon can change FP4 codes.
    mean_with_eps = gl.inline_asm_elementwise(
        "{ .reg .f32 t; div.rn.f32 t, $1, $2; add.rn.f32 $0, t, $3; }",
        "=f,f,f,f",
        [next_sum_sq, H * 1.0, next_eps],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    next_rstd = gl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [mean_with_eps],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    next_weight = gl.load(next_weight_ptr + cols, cols < H, 0).to(gl.float32)
    next_weight = gl.convert_layout(next_weight, next_input.type.layout)
    # CuTe's BF16 path rounds input*weight before applying the norm factor.
    weighted = (next_input * next_weight).to(gl.bfloat16).to(gl.float32)
    groups = gl.reshape(weighted, (BLOCK_H // 16, 16))
    max_abs = gl.max(gl.abs(groups), axis=1) * next_rstd
    global_scale = gl.load(global_scale_ptr)
    reciprocal_six = gl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [6.0],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    scale = gl.minimum(global_scale * max_abs * reciprocal_six, 448.0).to(gl.float8e4nv)
    inverse_scale = (
        gl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [scale.to(gl.float32)],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        * global_scale
    )
    # Match FlashInfer: a zero FP8 scale has reciprocal zero, not infinity.
    inverse_scale = gl.where(scale.to(gl.float32) == 0, 0.0, inverse_scale)
    values = (groups * next_rstd) * inverse_scale[:, None]
    pairs = gl.convert_layout(
        gl.reshape(values, (BLOCK_H // 2, 2)),
        BlockedLayout([1, 2], [32, 1], [4, 1], [0, 1]),
    )
    low, high = gl.split(pairs)
    packed = gl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,f,f",
        [low, high],
        dtype=gl.uint16,
        is_pure=True,
        pack=1,
    ).to(gl.uint8)
    output_cols = gl.arange(0, BLOCK_H // 2, layout=packed.type.layout)
    gl.store(fp4_ptr + row * (H // 2) + output_cols, packed, output_cols < H // 2)
    scale_cols = gl.arange(0, BLOCK_H // 16, layout=scale.type.layout)
    # Existing 128x4 swizzled activation-scale format, including partial tiles.
    sf_offsets = ((row // 128) * triton.cdiv(H // 16, 4) + scale_cols // 4) * 512
    sf_offsets += (row % 32) * 16 + ((row % 128) // 32) * 4 + scale_cols % 4
    gl.store(sf_ptr + sf_offsets, scale.to(gl.uint8, bitcast=True), scale_cols < H // 16)
    gl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // dummy $0",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


def gemma4_fused_norm_add_fp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    next_weight: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float,
    next_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 residual [M, H], packed FP4 [M, H/2] and swizzled scales.

    Inputs and norm weights are BF16, with contiguous inner dimensions.
    The scalar FP32 global_scale follows the consumer's static NVFP4 scale.
    Each row preserves the two existing kernels' intermediate BF16 rounding
    and FP32 reduction order. H is a multiple of 32 between 64 and 16384;
    the upper bound keeps the reference norm reduction within one CTA.
    """
    m, h = x.shape
    assert x.dtype == residual.dtype == weight.dtype == next_weight.dtype == torch.bfloat16
    assert x.stride(-1) == residual.stride(-1) == 1
    assert residual.shape == x.shape and weight.shape == next_weight.shape == (h,)
    assert 64 <= h <= 16384 and h % 32 == 0
    output = torch.empty((m, h), dtype=x.dtype, device=x.device)
    fp4 = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty(
        (triton.cdiv(m, 128) * triton.cdiv(h // 16, 4) * 512,),
        dtype=torch.uint8,
        device=x.device,
    )
    if m == 0:
        return output, fp4, scales
    # FlashInfer CuTe's per-row reduction layout; independent of token count.
    norm_threads = next(
        t for limit, t in ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)) if h <= limit
    )
    _gemma4_norm_add_fp4_kernel[(m,)](
        x,
        residual,
        weight,
        next_weight,
        global_scale,
        output,
        fp4,
        scales,
        eps,
        next_eps,
        SX=x.stride(0),
        SR=residual.stride(0),
        H=h,
        NORM_THREADS=norm_threads,
        BLOCK_H=triton.next_power_of_2(h),
        num_warps=4,
        launch_pdl=True,
    )
    return output, fp4, scales
