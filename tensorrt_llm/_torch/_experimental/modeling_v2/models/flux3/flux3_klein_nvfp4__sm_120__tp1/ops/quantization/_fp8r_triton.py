# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton per-token E4M3 quantizers of the FP8 row-wise block (content_mode_blocks.image.0),
both the warp_fp8_per_token_quant recipe: ``scale = max(amax / 448, 1e-12)`` with a
round-to-nearest division, ``q = clamp(x / scale, +-448)`` rounded to E4M3.

* `one_pass_fp8_quant`: one program per row, the row in registers (attn_out's input at
  2048^2: 16384 rows of 3072).
* `swiglu_fp8r_quant`: SwiGLU over the gate_up output's ``[gate | up]`` halves, rounded to
  BF16 as ``silu_and_mul`` stores it, then the same quantize, one program per row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SWIGLU_FP8R_NUM_WARPS = 16


@triton.jit
def _one_pass_fp8(x, output, scales, stride, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, block)
    values = tl.load(x + row * stride + col, col < width, other=0).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), 0), 448.0), 1.0e-12)
    quantized = tl.minimum(tl.maximum(tl.div_rn(values, scale), -448.0), 448.0)
    tl.store(output + row * width + col, quantized, col < width)
    tl.store(scales + row, scale)


@torch.library.custom_op("trtllm::flux3_klein_one_pass_fp8_quant", mutates_args=())
def one_pass_fp8_quant(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 [M, K], fp32 scales [M]) of a [M, K] bf16 matrix with a unit last stride."""
    output = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows.shape[0], device=rows.device, dtype=torch.float32)
    if rows.shape[0]:
        block = triton.next_power_of_2(rows.shape[1])
        _one_pass_fp8[(rows.shape[0],)](
            rows,
            output,
            scales,
            rows.stride(0),
            rows.shape[1],
            block,
            num_warps=min(max(block // 512, 4), 16),
        )
    return output, scales


@one_pass_fp8_quant.register_fake
def _one_pass_fp8_quant_fake(rows):
    return (
        torch.empty_like(rows, dtype=torch.float8_e4m3fn),
        rows.new_empty((rows.shape[0],), dtype=torch.float32),
    )


@triton.jit
def _swiglu_fp8r_part(x_row, col, width: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr):
    mask = col < width
    a = tl.load(x_row + col, mask=mask, other=0).to(tl.float32)
    b = tl.load(x_row + col + width, mask=mask, other=0).to(tl.float32)
    # Same expression as swiglu.silu_and_mul_kernel, rounded to its BF16 output.
    result = a * tl.sigmoid(alpha * a) * (b + beta)
    return result.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _swiglu_fp8r(
    x,
    output,
    scales,
    x_stride,
    width: tl.constexpr,
    block1: tl.constexpr,
    block2: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    """SwiGLU over [gate | up] then per-row E4M3 quantization, one row per program. The row
    is split into two power-of-two parts kept in registers (9216 = 8192 + 1024)."""
    row = tl.program_id(0).to(tl.int64)
    x_row = x + row * x_stride
    out_row = output + row * width
    col1 = tl.arange(0, block1)
    h1 = _swiglu_fp8r_part(x_row, col1, width, alpha, beta)
    amax = tl.max(tl.abs(h1), 0)
    if block2 > 0:
        col2 = block1 + tl.arange(0, block2)
        h2 = _swiglu_fp8r_part(x_row, col2, width, alpha, beta)
        amax = tl.maximum(amax, tl.max(tl.abs(h2), 0))
    scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0e-12)
    q1 = tl.minimum(tl.maximum(tl.div_rn(h1, scale), -448.0), 448.0)
    tl.store(out_row + col1, q1.to(tl.float8e4nv), col1 < width)
    if block2 > 0:
        q2 = tl.minimum(tl.maximum(tl.div_rn(h2, scale), -448.0), 448.0)
        tl.store(out_row + col2, q2.to(tl.float8e4nv), col2 < width)
    tl.store(scales + row, scale)


def _swiglu_fp8r_blocks(width: int) -> tuple[int, int]:
    block1 = 1 << (width.bit_length() - 1)
    rest = width - block1
    return block1, (triton.next_power_of_2(rest) if rest else 0)


@torch.library.custom_op("trtllm::flux3_klein_swiglu_fp8r_quant", mutates_args=())
def swiglu_fp8r_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """silu_and_mul(x) (alpha 1, beta 0, no limit) -> BF16 -> per-token E4M3, fused."""
    rows = x.reshape(-1, x.shape[-1])
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    width = rows.shape[1] // 2
    output = torch.empty((rows.shape[0], width), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows.shape[0], device=x.device, dtype=torch.float32)
    if rows.shape[0] and width:
        block1, block2 = _swiglu_fp8r_blocks(width)
        _swiglu_fp8r[(rows.shape[0],)](
            rows,
            output,
            scales,
            rows.stride(0),
            width,
            block1,
            block2,
            1.0,
            0.0,
            num_warps=_SWIGLU_FP8R_NUM_WARPS,
        )
    return output, scales


@swiglu_fp8r_quant.register_fake
def _swiglu_fp8r_quant_fake(x):
    rows = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]
    return (
        x.new_empty((rows, x.shape[-1] // 2), dtype=torch.float8_e4m3fn),
        x.new_empty((rows,), dtype=torch.float32),
    )
