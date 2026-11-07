# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused per-layer glue around CSA2's FlashInfer FA2 selected-row attention.

The reference adapter (``CSA2FlashInfer._forward``) runs ~20 small ATen ops per
layer: ``torch.cat`` of the two staging pools, a dense-prefix ``valid`` mask,
``pack_query_masks`` (pad/reshape/cast/shift/sum/cast), a ``torch.where`` staging
copy, and a sink epilogue (``sigmoid``/``mul``/``any``/``where``/cast plus an
output copy). These two kernels produce bitwise identical results:

* ``stage_rows_and_mask`` writes FlashInfer's little-endian packed custom mask
  (bit ``j`` of query ``i`` set iff ``j < prepared_lens[i]``) straight into the
  plan's persistent mask buffer and the ragged ``[count * width, 1, 512]`` KV
  rows from the SWA/extra pools, with zeros past each query's length.
* ``sink_epilogue`` computes ``bf16(float(o) * sigmoid(lse * ln2 - sink))`` (zero
  for empty selections) directly into the destination, using PyTorch's sigmoid
  form ``1 / (1 + expf(-x))`` with libdevice ``expf``, IEEE division and no FMA
  contraction, so every rounding matches the ATen sequence.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

try:  # Triton >= 3.0
    from triton.language.extra import libdevice as _libdevice
except ImportError:  # pragma: no cover - older Triton layouts
    from triton.language.extra.cuda import libdevice as _libdevice

# ``lse.float() * math.log(2.0)`` multiplies by the scalar rounded to FP32.
_LN2_F32 = float(torch.tensor(math.log(2.0), dtype=torch.float32).item())


@triton.jit
def _stage_rows_and_mask_kernel(
    swa_ptr,
    extra_ptr,
    lens_ptr,
    rows_ptr,
    mask_ptr,
    swa_stride0,
    swa_stride1,
    extra_stride0,
    extra_stride1,
    width,
    swa_width,
    mask_bytes,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_BLOCK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    query = tl.program_id(0)
    block = tl.program_id(1)
    length = tl.load(lens_ptr + query)
    length = tl.minimum(tl.maximum(length, 0), width)
    if block == 0:
        byte = tl.arange(0, MASK_BLOCK)
        first = byte * 8
        partial = tl.maximum(tl.minimum(length - first, 8), 0)
        bits = (tl.full([MASK_BLOCK], 1, tl.int32) << partial) - 1
        tl.store(
            mask_ptr + query.to(tl.int64) * mask_bytes + byte,
            bits.to(tl.uint8),
            mask=byte < mask_bytes,
        )
    rows = block * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, DIM)
    in_width = rows < width
    live = rows < length
    in_swa = rows < swa_width
    swa_offsets = (
        query.to(tl.int64) * swa_stride0
        + rows.to(tl.int64)[:, None] * swa_stride1
        + cols[None, :]
    )
    values = tl.load(swa_ptr + swa_offsets, mask=(live & in_swa)[:, None], other=0.0)
    if HAS_EXTRA:
        extra_rows = tl.maximum(rows - swa_width, 0)
        extra_offsets = (
            query.to(tl.int64) * extra_stride0
            + extra_rows.to(tl.int64)[:, None] * extra_stride1
            + cols[None, :]
        )
        extra = tl.load(
            extra_ptr + extra_offsets, mask=(live & (rows >= swa_width))[:, None], other=0.0
        )
        values = tl.where(in_swa[:, None], values, extra)
    out_offsets = (query.to(tl.int64) * width + rows.to(tl.int64))[:, None] * DIM + cols[None, :]
    tl.store(rows_ptr + out_offsets, values, mask=in_width[:, None])


def stage_rows_and_mask(
    swa_pool: torch.Tensor,
    extra_pool: torch.Tensor | None,
    lengths: torch.Tensor,
    rows: torch.Tensor,
    packed_mask: torch.Tensor,
) -> None:
    """Write ragged rows ``[count * width, 1, dim]`` and the packed custom mask."""
    count, swa_width, dim = swa_pool.shape
    extra_width = 0 if extra_pool is None else extra_pool.shape[1]
    width = swa_width + extra_width
    mask_bytes = (width + 7) // 8
    if (
        swa_pool.stride(2) != 1
        or (extra_pool is not None and (extra_pool.stride(2) != 1 or extra_pool.shape[0] < count))
        or lengths.dtype != torch.int32
        or lengths.numel() < count
        or not rows.is_contiguous()
        or rows.numel() != count * width * dim
        or packed_mask.dtype != torch.uint8
        or not packed_mask.is_contiguous()
        or packed_mask.numel() != count * mask_bytes
        or dim & (dim - 1)
    ):
        raise ValueError("CSA2 fused staging received unexpected buffer geometry")
    rows_per_program = 16
    extra = swa_pool if extra_pool is None else extra_pool
    grid = (count, triton.cdiv(width, rows_per_program))
    _stage_rows_and_mask_kernel[grid](
        swa_pool,
        extra,
        lengths,
        rows,
        packed_mask,
        swa_pool.stride(0),
        swa_pool.stride(1),
        extra.stride(0),
        extra.stride(1),
        width,
        swa_width,
        mask_bytes,
        DIM=dim,
        ROWS=rows_per_program,
        MASK_BLOCK=triton.next_power_of_2(mask_bytes),
        HAS_EXTRA=extra_pool is not None,
        num_warps=4,
    )


@triton.jit
def _stage_shared_indices_and_mask_kernel(
    indices_ptr,
    plan_indices_ptr,
    mask_ptr,
    flag_ptr,
    indices_stride0,
    width,
    mask_bytes,
    MASK_BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    byte = tl.arange(0, MASK_BLOCK)
    bit = tl.arange(0, 8)
    cols = byte[:, None] * 8 + bit[None, :]
    in_width = cols < width
    idx = tl.load(
        indices_ptr + query.to(tl.int64) * indices_stride0 + cols, mask=in_width, other=-1
    )
    valid = (idx >= 0) & in_width
    tl.store(
        plan_indices_ptr + query.to(tl.int64) * width + cols, tl.maximum(idx, 0), mask=in_width
    )
    # Integer sums are exact in any order.
    packed = tl.sum(valid.to(tl.int32) << bit[None, :], axis=1)
    tl.store(
        mask_ptr + query.to(tl.int64) * mask_bytes + byte,
        packed.to(tl.uint8),
        mask=byte < mask_bytes,
    )
    any_valid = tl.max(tl.max(valid.to(tl.int32), axis=1), axis=0)
    tl.store(flag_ptr + query, any_valid)


def stage_shared_indices_and_mask(
    indices: torch.Tensor,
    plan_indices: torch.Tensor,
    packed_mask: torch.Tensor,
    valid_flag: torch.Tensor,
) -> None:
    """One launch: ``plan_indices = clamp_min(indices, 0).flatten()``, FlashInfer's
    little-endian packed mask (bit ``j`` of query ``i`` iff ``indices[i, j] >= 0``) and
    ``valid_flag[i] = any(indices[i] >= 0)``. Holes may sit anywhere in a row."""
    count, width = indices.shape
    mask_bytes = (width + 7) // 8
    if (
        indices.dtype != torch.int32
        or indices.stride(1) != 1
        or plan_indices.dtype != torch.int32
        or not plan_indices.is_contiguous()
        or plan_indices.numel() != count * width
        or packed_mask.dtype != torch.uint8
        or not packed_mask.is_contiguous()
        or packed_mask.numel() != count * mask_bytes
        or valid_flag.dtype != torch.int32
        or valid_flag.numel() < count
    ):
        raise ValueError("CSA2 shared staging received unexpected buffer geometry")
    if count == 0:
        return
    _stage_shared_indices_and_mask_kernel[(count,)](
        indices,
        plan_indices,
        packed_mask,
        valid_flag,
        indices.stride(0),
        width,
        mask_bytes,
        MASK_BLOCK=triton.next_power_of_2(mask_bytes),
        num_warps=4,
    )


@triton.jit
def _sink_epilogue_kernel(
    o_ptr,
    lse_ptr,
    sink_ptr,
    lens_ptr,
    out_ptr,
    heads,
    o_stride0,
    o_stride1,
    out_stride0,
    out_stride1,
    LN2: tl.constexpr,
    DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    query = pid // heads
    head = pid % heads
    cols = tl.arange(0, DIM)
    zero = cols * 0
    # Broadcast loads keep every step a plain FP32 vector op.
    lse = tl.load(lse_ptr + pid + zero)
    sink = tl.load(sink_ptr + head + zero).to(tl.float32)
    # Two separately rounded ops (fp fusion is disabled at launch).
    x = lse * LN2
    x = x - sink
    one = tl.full([DIM], 1.0, tl.float32)
    correction = tl.math.div_rn(one, one + _libdevice.exp(-x))
    length = tl.load(lens_ptr + query)
    o = tl.load(o_ptr + query.to(tl.int64) * o_stride0 + head * o_stride1 + cols)
    value = o.to(tl.float32) * correction
    value = tl.where(length > 0, value, 0.0)
    tl.store(
        out_ptr + query.to(tl.int64) * out_stride0 + head * out_stride1 + cols,
        value.to(tl.bfloat16),
    )


def sink_epilogue(
    o: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = where(len > 0, bf16(float(o) * sigmoid(lse * ln2 - sink)), 0)``."""
    count, heads, dim = o.shape
    if (
        o.stride(2) != 1
        or out.shape != o.shape
        or out.stride(2) != 1
        or out.dtype != torch.bfloat16
        or lse.shape != (count, heads)
        or not lse.is_contiguous()
        or lse.dtype != torch.float32
        or sink.numel() != heads
        or not sink.is_contiguous()
        or lengths.dtype != torch.int32
        or lengths.numel() < count
        or dim & (dim - 1)
    ):
        raise ValueError("CSA2 fused sink epilogue received unexpected buffer geometry")
    if count == 0:
        return out
    _sink_epilogue_kernel[(count * heads,)](
        o,
        lse,
        sink,
        lengths,
        out,
        heads,
        o.stride(0),
        o.stride(1),
        out.stride(0),
        out.stride(1),
        LN2=_LN2_F32,
        DIM=dim,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
