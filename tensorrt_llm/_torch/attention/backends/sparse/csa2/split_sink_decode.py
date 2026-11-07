# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-launch split-KV sink attention for CSA2 decode (generation-only) rows.

Replaces ``stage_rows_and_mask`` + FlashInfer FA2 split-KV (+ merge) + ``sink_epilogue``
for decode queries. Query ``i`` attends to the dense prefix ``[0, lens[i])`` of
``swa[i] ++ extra[i]`` (one shared KV head, ``K == V``), with an attention sink:

    out = sum_j exp(s_j) v_j / (sum_j exp(s_j) + exp(sink)),  s_j = scale * q . k_j

which equals ``softmax(s) @ V * sigmoid(lse - sink)``.

* Kernel A (grid ``(count, splits)``) reads the pools in place. Each program owns a
  contiguous ``CHUNK`` of the logical rows; every ``BLOCK_N`` tile is loaded once and
  used as both K and V (base-2 online softmax, FP32 accumulator, P cast to BF16 for
  P @ V as FA2 does). It writes unnormalized partials (O, m, l). Rows at or past
  ``lens[i]`` are masked out of both the logits and V, whatever the pools hold there.
  Programs whose chunk starts at or past ``lens[i]`` exit without writing.
* Kernel B (grid ``count * heads``) merges the partials of the chunks that start
  before ``lens[i]`` (exactly those kernel A wrote this launch, so the workspace
  carries no state between calls) with the sink and stores BF16 output, zero for
  empty selections.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def _split_sink_main_kernel(
    q_ptr,
    swa_ptr,
    extra_ptr,
    lens_ptr,
    o_ws_ptr,
    ml_ws_ptr,
    heads,
    q_stride0,
    q_stride1,
    swa_stride0,
    swa_stride1,
    extra_stride0,
    extra_stride1,
    width,
    swa_width,
    qk_scale,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DIM: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    query = tl.program_id(0)
    split = tl.program_id(1)
    length = tl.load(lens_ptr + query)
    length = tl.minimum(tl.maximum(length, 0), width)
    start = split * CHUNK
    if start >= length:
        return
    heads_idx = tl.arange(0, BLOCK_H)
    cols = tl.arange(0, DIM)
    head_ok = heads_idx < heads
    q64 = query.to(tl.int64)
    q = tl.load(
        q_ptr + q64 * q_stride0 + heads_idx[:, None] * q_stride1 + cols[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, DIM], tl.float32)
    end = tl.minimum(start + CHUNK, length)
    for tile in range(start, end, BLOCK_N):
        rows = tile + tl.arange(0, BLOCK_N)
        live = rows < length
        if HAS_EXTRA:
            if tile < swa_width:
                ptrs = (
                    swa_ptr
                    + q64 * swa_stride0
                    + rows.to(tl.int64)[:, None] * swa_stride1
                    + cols[None, :]
                )
            else:
                ptrs = (
                    extra_ptr
                    + q64 * extra_stride0
                    + (rows - swa_width).to(tl.int64)[:, None] * extra_stride1
                    + cols[None, :]
                )
        else:
            ptrs = (
                swa_ptr + q64 * swa_stride0 + rows.to(tl.int64)[:, None] * swa_stride1 + cols[None, :]
            )
        kv = tl.load(ptrs, mask=live[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * qk_scale
        s = tl.where(live[None, :], s, float("-inf"))
        # Row ``tile`` is live, so m_new is finite and alpha = 2^-inf = 0 on the first tile.
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = m_new
    slot = q64 * SPLITS + split
    tl.store(o_ws_ptr + (slot * BLOCK_H + heads_idx)[:, None] * DIM + cols[None, :], acc)
    tl.store(ml_ws_ptr + (slot * 2) * BLOCK_H + heads_idx, m_i)
    tl.store(ml_ws_ptr + (slot * 2 + 1) * BLOCK_H + heads_idx, l_i)


@triton.jit
def _split_sink_merge_kernel(
    o_ws_ptr,
    ml_ws_ptr,
    sink_ptr,
    lens_ptr,
    out_ptr,
    heads,
    width,
    out_stride0,
    out_stride1,
    LOG2E: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_P2: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    query = pid // heads
    head = pid % heads
    length = tl.load(lens_ptr + query)
    length = tl.minimum(tl.maximum(length, 0), width)
    splits = tl.arange(0, SPLITS_P2)
    # Exactly the chunks kernel A wrote in this launch.
    written = (splits < SPLITS) & (splits * CHUNK < length)
    slot = query.to(tl.int64) * SPLITS + splits
    m = tl.load(ml_ws_ptr + (slot * 2) * BLOCK_H + head, mask=written, other=float("-inf"))
    l = tl.load(ml_ws_ptr + (slot * 2 + 1) * BLOCK_H + head, mask=written, other=0.0)
    sink2 = tl.load(sink_ptr + head).to(tl.float32) * LOG2E
    big = tl.maximum(tl.max(m, 0), sink2)
    w = tl.exp2(m - big)
    w = tl.where(written, w, 0.0)
    denom = tl.sum(l * w, 0) + tl.exp2(sink2 - big)
    cols = tl.arange(0, DIM)
    o = tl.load(
        o_ws_ptr + (slot * BLOCK_H + head)[:, None] * DIM + cols[None, :],
        mask=written[:, None],
        other=0.0,
    )
    value = tl.sum(o * w[:, None], 0) / denom
    value = tl.where(length > 0, value, 0.0)
    tl.store(
        out_ptr + query.to(tl.int64) * out_stride0 + head * out_stride1 + cols,
        value.to(tl.bfloat16),
    )


# (CHUNK rows per program, BLOCK_N rows per tile, num_warps) by decode count.
SPLIT_CONFIGS: dict[int, tuple[int, int, int]] = {}
DEFAULT_CONFIG = (32, 32, 4)
MAX_COUNT = 16


def split_config(count: int) -> tuple[int, int, int]:
    return SPLIT_CONFIGS.get(count, DEFAULT_CONFIG)


def supported(q, swa, extra) -> bool:
    count, heads, dim = q.shape
    if count > MAX_COUNT or dim != 512 or heads > 64:
        return False
    if q.stride(2) != 1 or swa.stride(2) != 1 or (extra is not None and extra.stride(2) != 1):
        return False
    chunk, block_n, _ = split_config(count)
    return swa.shape[1] % block_n == 0 and chunk % block_n == 0


def workspace_shapes(count: int, heads: int, width: int, dim: int):
    chunk, _, _ = split_config(count)
    splits = -(-width // chunk)
    block_h = max(16, triton.next_power_of_2(heads))
    return (count, splits, block_h, dim), (count, splits, 2, block_h)


def split_sink_decode(q, swa, extra, lengths, sink, scale, out, o_ws, ml_ws):
    """``out[i] = sink attention of q[i] over (swa[i] ++ extra[i])[:lengths[i]]``."""
    count, heads, dim = q.shape
    swa_width = swa.shape[1]
    width = swa_width + (0 if extra is None else extra.shape[1])
    chunk, block_n, num_warps = split_config(count)
    splits = -(-width // chunk)
    block_h = max(16, triton.next_power_of_2(heads))
    if (
        o_ws.shape != (count, splits, block_h, dim)
        or ml_ws.shape != (count, splits, 2, block_h)
        or not o_ws.is_contiguous()
        or not ml_ws.is_contiguous()
        or out.shape != q.shape
        or out.stride(2) != 1
        or out.dtype != torch.bfloat16
        or lengths.dtype != torch.int32
        or lengths.numel() < count
        or sink.numel() != heads
        or (extra is not None and extra.shape[0] < count)
        or swa.shape[0] < count
    ):
        raise ValueError("CSA2 split sink decode received unexpected buffer geometry")
    ext = swa if extra is None else extra
    _split_sink_main_kernel[(count, splits)](
        q,
        swa,
        ext,
        lengths,
        o_ws,
        ml_ws,
        heads,
        q.stride(0),
        q.stride(1),
        swa.stride(0),
        swa.stride(1),
        ext.stride(0),
        ext.stride(1),
        width,
        swa_width,
        float(scale) * _LOG2E,
        SPLITS=splits,
        CHUNK=chunk,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        DIM=dim,
        HAS_EXTRA=extra is not None,
        num_warps=num_warps,
        num_stages=1,
    )
    _split_sink_merge_kernel[(count * heads,)](
        o_ws,
        ml_ws,
        sink,
        lengths,
        out,
        heads,
        width,
        out.stride(0),
        out.stride(1),
        LOG2E=_LOG2E,
        SPLITS=splits,
        SPLITS_P2=triton.next_power_of_2(splits),
        CHUNK=chunk,
        BLOCK_H=block_h,
        DIM=dim,
        num_warps=4,
    )
    return out


__all__ = ["split_sink_decode", "supported", "workspace_shapes", "split_config"]
