# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch query replication for the SM120 split-key decode indexer.

``CSA2Indexer._split_key_decode_logits`` feeds DeepGEMM's fp8_fp4_mqa_logits
``rows * reps`` query rows, where every replica ``j`` of row ``r`` (output row
``r * reps + j``) is a full copy of row ``r``'s packed FP4 query data, its
UE8M0 scales (int32 per head) and its FP32 head weights. The reference builds
them with three ``x[:, None].expand(...).contiguous()`` copies; this kernel
writes all three contiguous replicas in one launch. Pure data movement: every
element is loaded and stored as raw integer bits, so the replicas are
byte-identical to the expand copies.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _replicate_split_queries_kernel(
    q_ptr,
    q_stride_r,
    q_stride_h,
    q_stride_d,
    s_ptr,
    s_stride_r,
    s_stride_h,
    w_ptr,
    w_stride_r,
    w_stride_h,
    q_out_ptr,
    s_out_ptr,
    w_out_ptr,
    reps,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    out_row = tl.program_id(0)
    src = (out_row // reps).to(tl.int64)
    dst = out_row.to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    hm = h < HEADS
    d = tl.arange(0, DIM)
    qv = tl.load(
        q_ptr + src * q_stride_r + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        mask=hm[:, None],
    )
    tl.store(q_out_ptr + dst * (HEADS * DIM) + h[:, None] * DIM + d[None, :], qv, mask=hm[:, None])
    sv = tl.load(s_ptr + src * s_stride_r + h * s_stride_h, mask=hm)
    tl.store(s_out_ptr + dst * HEADS + h, sv, mask=hm)
    wv = tl.load(w_ptr + src * w_stride_r + h * w_stride_h, mask=hm)
    tl.store(w_out_ptr + dst * HEADS + h, wv, mask=hm)


def replicate_split_queries(
    q: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    reps: int,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous ``[rows * reps, ...]`` replicas of ``q`` (int8 ``[rows, heads, dim]``),
    ``scale`` (int32 ``[rows, heads]``) and ``weights`` (fp32 ``[rows, heads]``).

    Inputs are read through their strides. ``out`` optionally supplies the three
    destination tensors (contiguous, matching shapes and dtypes).
    """
    rows, heads, dim = q.shape
    total = rows * reps
    if out is None:
        q_out = torch.empty((total, heads, dim), dtype=q.dtype, device=q.device)
        s_out = torch.empty((total, heads), dtype=scale.dtype, device=q.device)
        w_out = torch.empty((total, heads), dtype=weights.dtype, device=q.device)
    else:
        q_out, s_out, w_out = out
    if total == 0:
        return q_out, s_out, w_out
    # Raw 32-bit moves for the weights: bit patterns pass through unchanged.
    w_bits = weights.view(torch.int32)
    _replicate_split_queries_kernel[(total,)](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        scale,
        scale.stride(0),
        scale.stride(1),
        w_bits,
        w_bits.stride(0),
        w_bits.stride(1),
        q_out,
        s_out,
        w_out.view(torch.int32),
        reps,
        HEADS=heads,
        DIM=dim,
        BLOCK_H=triton.next_power_of_2(heads),
        num_warps=4,
    )
    return q_out, s_out, w_out


def replicate_supported(q: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor) -> bool:
    dim = q.shape[-1]
    return (
        q.is_cuda
        and q.dim() == 3
        and q.dtype == torch.int8
        and scale.dtype == torch.int32
        and scale.dim() == 2
        and weights.dtype == torch.float32
        and weights.dim() == 2
        and dim > 0
        and (dim & (dim - 1)) == 0
        and scale.shape == weights.shape == q.shape[:2]
    )
