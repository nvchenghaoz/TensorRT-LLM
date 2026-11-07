# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch query replication for the SM120 CSA2 split-key decode indexer.

``CSA2Indexer._split_key_decode_raw`` feeds DeepGEMM ``reps`` copies of every
decode row: the packed E2M1 query ``[rows, heads, 64]`` (int8), its UE8M0 scale
``[rows, heads]`` (int32) and the head weights ``[rows, heads]`` (fp32). Main
builds them with three ATen ``expand().contiguous()`` copies. This kernel writes
the same three buffers in one launch: output row ``p`` holds source row
``p // reps``, every element copied as raw bits (int32 words), so the bytes are
identical to the ATen copies.

Used only while a CUDA graph is being captured and only once the kernel has been
compiled by an eager call (``warm``); every other call keeps main's copies.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Replicas written per program (``reps`` is always a multiple of BLOCK_Q = 4).
_REPLICAS_PER_PROGRAM = 4

_WARM: set = set()


@triton.jit(do_not_specialize=["reps", "q_stride", "s_stride", "w_stride"])
def _replicate_split_queries_kernel(
    q_ptr,
    s_ptr,
    w_ptr,
    q_out_ptr,
    s_out_ptr,
    w_out_ptr,
    reps,
    q_stride,
    s_stride,
    w_stride,
    QWORDS: tl.constexpr,
    HEADS: tl.constexpr,
    RB: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    qcols = tl.arange(0, QWORDS)
    hcols = tl.arange(0, HEADS)
    src = row.to(tl.int64)
    qv = tl.load(q_ptr + src * q_stride + qcols)
    sv = tl.load(s_ptr + src * s_stride + hcols)
    wv = tl.load(w_ptr + src * w_stride + hcols)
    base = src * reps + block * RB
    for i in tl.static_range(RB):
        dst = base + i
        tl.store(q_out_ptr + dst * QWORDS + qcols, qv)
        tl.store(s_out_ptr + dst * HEADS + hcols, sv)
        tl.store(w_out_ptr + dst * HEADS + hcols, wv)


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _words(
    q_data: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, reps: int
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """int32 word views of the three sources, or None when the kernel does not apply."""
    if not (q_data.is_cuda and q_data.dim() == 3 and scale.dim() == 2 and weights.dim() == 2):
        return None
    rows, heads, dim = q_data.shape
    q = q_data.view(torch.int8)
    if (
        heads not in (16, 32, 64)
        or q.shape[2] != 64
        or q.stride(2) != 1
        or q.stride(1) != 64
        or q.stride(0) % 16 != 0
        or reps % _REPLICAS_PER_PROGRAM != 0
        or weights.dtype != torch.float32
        or scale.dtype != torch.int32
        or tuple(weights.shape) != (rows, heads)
        or tuple(scale.shape) != (rows, heads)
        or weights.stride(1) != 1
        or scale.stride(1) != 1
        or weights.stride(0) % 4 != 0
        or scale.stride(0) % 4 != 0
        or not (_aligned(q) and _aligned(scale) and _aligned(weights))
    ):
        return None
    return q.view(torch.int32), scale, weights.view(torch.int32)


def _launch(q32, s32, w32, reps):
    rows, heads = s32.shape
    q_out = torch.empty((rows * reps, heads, 64), dtype=torch.int8, device=q32.device)
    s_out = torch.empty((rows * reps, heads), dtype=torch.int32, device=q32.device)
    w_out = torch.empty((rows * reps, heads), dtype=torch.float32, device=q32.device)
    grid = (rows, reps // _REPLICAS_PER_PROGRAM)
    _replicate_split_queries_kernel[grid](
        q32,
        s32,
        w32,
        q_out.view(torch.int32),
        s_out,
        w_out.view(torch.int32),
        reps,
        q32.stride(0),
        s32.stride(0),
        w32.stride(0),
        QWORDS=heads * 16,
        HEADS=heads,
        RB=_REPLICAS_PER_PROGRAM,
        num_warps=4,
    )
    return q_out, s_out, w_out


def warm(q_data: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, reps: int) -> None:
    """Eager only: compile (one throwaway launch) the kernel for this head count."""
    if rows_empty(q_data) or not q_data.is_cuda:
        return
    key = (q_data.device.index, q_data.shape[1])
    if key in _WARM or torch.cuda.is_current_stream_capturing():
        return
    words = _words(q_data, scale, weights, reps)
    if words is None:
        return
    _launch(*words, reps)
    _WARM.add(key)


def rows_empty(q_data: torch.Tensor) -> bool:
    return q_data.shape[0] == 0


def replicate_split_queries(
    q_data: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor, reps: int
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(q [rows*reps, heads, 64] int8, scale [rows*reps, heads] int32, w [rows*reps, heads])``.

    Returns None (caller keeps the ATen copies) unless capturing a CUDA graph with
    a kernel an eager call has already compiled.
    """
    if rows_empty(q_data) or not torch.cuda.is_current_stream_capturing():
        return None
    if (q_data.device.index, q_data.shape[1]) not in _WARM:
        return None
    words = _words(q_data, scale, weights, reps)
    if words is None:
        return None
    return _launch(*words, reps)
