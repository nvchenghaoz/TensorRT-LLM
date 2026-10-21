# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GDN prefill in one pass: causal conv1d + SiLU + q/k/v split + q/k L2 norm + gates.

Replaces ``extract_transpose_prefill_slice`` + ``causal_conv1d_fwd`` (or the
fused transpose+conv kernel) + ``fused_gdn_post_conv`` with one Triton kernel
that reads the token-major ``in_proj_qkvz`` column view and writes the
contiguous token-major q/k/v, g and beta that ``fused_gdn_post_conv`` returns,
plus the final conv states.

Conv math is the C++ ``causal_conv1d_fwd_kernel``'s (see gdn_prefill_conv.py):
fma.rn chain from bias/+0, non-ftz SiLU, RNE to the input dtype. Each program
owns one (sequence-aligned token tile, 128-channel head) block, so the q/k L2
norm over a head is done in registers. ``REDUCE`` selects how the squared norm
is summed: 0 = tl.sum (reordered), 1 / 3 = an explicit add tree that replays
the reduction order of ``_fused_gdn_post_conv_kernel``'s layout for pure
prefill with 16-aligned token counts / mixed steps (bitwise).
"""

import os

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.gdn_prefill_conv import (
    _final_state_tap,
    _first_tile_tap,
    _fma_rn,
    _silu_ref,
)


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _drop_bit(x, BT: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """x[BT, M]; add the two halves selected by index bit N (lo + hi) -> [BT, M // 2]."""
    y = tl.reshape(x, [BT, M // (2 * N), 2, N])
    y = tl.permute(y, (0, 1, 3, 2))
    lo, hi = tl.split(y)
    return tl.reshape(lo + hi, [BT, M // 2])


@triton.jit
def _split_bit(x, BT: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """x[BT, M] -> (bit N clear, bit N set), each [BT, M // 2]."""
    y = tl.reshape(x, [BT, M // (2 * N), 2, N])
    y = tl.permute(y, (0, 1, 3, 2))
    lo, hi = tl.split(y)
    return tl.reshape(lo, [BT, M // 2]), tl.reshape(hi, [BT, M // 2])


@triton.jit
def _seq8(x, BT: tl.constexpr, M: tl.constexpr, NB: tl.constexpr, TREE: tl.constexpr):
    """x[BT, M]; 8 slices by index bits (NB*4, NB*2, NB) -> slice sum in element order."""
    h0, h1 = _split_bit(x, BT, M, NB * 4)
    q0, q1 = _split_bit(h0, BT, M // 2, NB * 2)
    q2, q3 = _split_bit(h1, BT, M // 2, NB * 2)
    a0, a1 = _split_bit(q0, BT, M // 4, NB)
    a2, a3 = _split_bit(q1, BT, M // 4, NB)
    a4, a5 = _split_bit(q2, BT, M // 4, NB)
    a6, a7 = _split_bit(q3, BT, M // 4, NB)
    if TREE:
        acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
    else:
        acc = a0 + a1
        acc = acc + a2
        acc = acc + a3
        acc = acc + a4
        acc = acc + a5
        acc = acc + a6
        acc = acc + a7
    return acc


@triton.jit
def _warps4(s, BT: tl.constexpr, WARP_SEQ: tl.constexpr):
    """s[BT, 4] indexed by warp id -> [BT]."""
    if WARP_SEQ:
        w01, w23 = _split_bit(s, BT, 4, 2)
        w0, w1 = _split_bit(w01, BT, 2, 1)
        w2, w3 = _split_bit(w23, BT, 2, 1)
        t = ((w0 + w1) + w2) + w3
    else:
        s = _drop_bit(s, BT, 4, 2)
        t = _drop_bit(s, BT, 2, 1)
    return tl.reshape(t, [BT])


@triton.jit
def _sumsq(y, BT: tl.constexpr, D: tl.constexpr, REDUCE: tl.constexpr, WARP_SEQ: tl.constexpr,
           TREE: tl.constexpr):
    if REDUCE == 4:
        # Mixed-step layout of _fused_gdn_post_conv_kernel (sizePerThread [1, 8],
        # threadsPerWarp [2, 16]): column c = l*8 + e. Per thread the squares
        # (not fused into the adds) are summed ((e0+e2)+(e4+e6)) + ((e1+e3)+(e5+e7)),
        # then lanes xor 8, 4, 2, 1.
        sq = _mul_rn(y, y)
        s = _drop_bit(sq, BT, 128, 2)  # e bit 1
        s = _drop_bit(s, BT, 64, 2)  # e bit 2
        s = _drop_bit(s, BT, 32, 1)  # e bit 0 -> [BT, 16], index l
        s = _drop_bit(s, BT, 16, 8)
        s = _drop_bit(s, BT, 8, 4)
        s = _drop_bit(s, BT, 4, 2)
        s = _drop_bit(s, BT, 2, 1)
        return tl.reshape(s, [BT])
    if REDUCE == 5:
        # Pure-prefill layout for T % 16 != 0 (sizePerThread [1, 1],
        # threadsPerWarp [16, 2], warpsPerCTA [1, 4]): column c = rep*8 + w*2 + l
        # (rep = c bits 3..6). Per thread (PTX of main's kernel): f32x2 pairs of
        # fma(r, r, rn(r'^2)) for reps (r, r + 2), then ((f0 + f4) + (f8 + f12))
        # per parity, even + odd; lanes xor 16 (l); then the 4 warp partials.
        A, B = _split_bit(y, BT, 128, 16)  # rep bit 1
        f = _fma_rn(A, A, _mul_rn(B, B))  # [BT, 64]
        f = _drop_bit(f, BT, 64, 16)  # rep bit 2
        f = _drop_bit(f, BT, 32, 16)  # rep bit 3
        f = _drop_bit(f, BT, 16, 8)  # rep bit 0 (parity) -> [BT, 8], j = w*2 + l
        s = _drop_bit(f, BT, 8, 1)  # lane bit -> [BT, 4], index w
        return _warps4(s, BT, WARP_SEQ)
    sq = y * y
    if REDUCE == 0:
        return tl.sum(sq, axis=1)
    elif REDUCE == 1:
        # layout sizePerThread [8,1], threadsPerWarp [2,16], warpsPerCTA [1,4]:
        # column c = r*64 + w*16 + l. thread: sq[r0] + sq[r1]; lanes: xor 8,4,2,1;
        # then the 4 warp partials.
        s = _drop_bit(sq, BT, 128, 64)  # [BT, 64], j = w*16 + l
        s = _drop_bit(s, BT, 64, 8)
        s = _drop_bit(s, BT, 32, 4)
        s = _drop_bit(s, BT, 16, 2)
        s = _drop_bit(s, BT, 8, 1)  # [BT, 4], j = w
        return _warps4(s, BT, WARP_SEQ)
    elif REDUCE == 2:
        # layout sizePerThread [1,1], threadsPerWarp [16,2], warpsPerCTA [1,4]:
        # column c = rep*8 + w*2 + l. thread: over 16 reps; lanes: xor 1; warps.
        h0, h1 = _split_bit(sq, BT, 128, 64)  # rep bit 3
        lo = _seq8(h0, BT, 64, 8, TREE)  # reps 0..7 -> [BT, 8]
        hi = _seq8(h1, BT, 64, 8, TREE)  # reps 8..15
        if TREE:
            acc = lo + hi
        else:
            # sequential over 16: (((r0 + r1) + ...) + r7) + r8 + ... + r15
            g0, g1 = _split_bit(h1, BT, 64, 32)
            b0, b1 = _split_bit(g0, BT, 32, 16)
            b2, b3 = _split_bit(g1, BT, 32, 16)
            c0, c1 = _split_bit(b0, BT, 16, 8)
            c2, c3 = _split_bit(b1, BT, 16, 8)
            c4, c5 = _split_bit(b2, BT, 16, 8)
            c6, c7 = _split_bit(b3, BT, 16, 8)
            acc = lo + c0
            acc = acc + c1
            acc = acc + c2
            acc = acc + c3
            acc = acc + c4
            acc = acc + c5
            acc = acc + c6
            acc = acc + c7
        s = _drop_bit(acc, BT, 8, 1)  # lanes -> [BT, 4], j = w
        return _warps4(s, BT, WARP_SEQ)
    else:
        # layout sizePerThread [1,8], threadsPerWarp [2,16], warpsPerCTA [4,1]:
        # column c = l*8 + e. thread: over e; lanes: xor 8,4,2,1.
        y2 = tl.reshape(tl.permute(tl.reshape(sq, [BT, 16, 8]), (0, 2, 1)), [BT, 128])
        # y2 index = e*16 + l
        acc = _seq8(y2, BT, 128, 16, TREE)  # [BT, 16], index l
        s = _drop_bit(acc, BT, 16, 8)
        s = _drop_bit(s, BT, 8, 4)
        s = _drop_bit(s, BT, 4, 2)
        s = _drop_bit(s, BT, 2, 1)
        return tl.reshape(s, [BT])


@triton.jit
def _gdn_prefill_onepass_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    state_ptr,
    qsl_ptr,
    cache_idx_ptr,
    has_init_ptr,
    dec_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    num_seqs,
    num_prefill_tokens,
    num_decode_tokens,
    stride_x_t,
    stride_dec_t,
    stride_w_c,
    stride_w_k,
    stride_s_b,
    stride_s_c,
    stride_s_k,
    a_stride_t,
    a_stride_h,
    b_stride_t,
    b_stride_h,
    pad_slot_id,
    l2_norm_eps,
    softplus_threshold,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    HAS_CACHE_IDX: tl.constexpr,
    HAS_INIT_PTR: tl.constexpr,
    HAS_DECODE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    REDUCE: tl.constexpr,
    WARP_SEQ: tl.constexpr,
    TREE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    ii = tl.arange(0, BLOCK_T)
    cc = tl.arange(0, HEAD_DIM)
    c_first = pid_h * HEAD_DIM
    max_p_tiles = tl.cdiv(num_prefill_tokens, BLOCK_T) + num_seqs

    if pid_t < max_p_tiles:
        offs_n = tl.arange(0, BLOCK_N)
        n_mask = offs_n < num_seqs
        starts = tl.load(qsl_ptr + offs_n, mask=n_mask, other=0)
        ends = tl.load(qsl_ptr + offs_n + 1, mask=n_mask, other=0)
        ntiles = tl.where(n_mask, tl.maximum((ends - starts + BLOCK_T - 1) // BLOCK_T, 1), 0)
        cum = tl.cumsum(ntiles, 0)
        # Spare tiles (seq == num_seqs) clamp to the last sequence with a tile
        # index past its end: every token is masked and tile != 0.
        seq = tl.minimum(tl.sum((cum <= pid_t).to(tl.int32), 0), num_seqs - 1)
        sel = offs_n == seq
        seq_start = tl.sum(tl.where(sel, starts, 0), 0)
        seq_end = tl.sum(tl.where(sel, ends, 0), 0)
        tile = pid_t - tl.sum(tl.where(sel, cum - ntiles, 0), 0)

        if HAS_CACHE_IDX:
            slot = tl.load(cache_idx_ptr + seq)
        else:
            slot = seq
        is_pad = slot == pad_slot_id
        if HAS_INIT_PTR:
            has_init = tl.load(has_init_ptr + seq) != 0
        else:
            has_init = seq < 0
        has_init = has_init & (~is_pad)

        t_first = seq_start + tile * BLOCK_T
        t_mask = (t_first + ii) < seq_end
        c_mask = cc < HEAD_DIM
        mask = t_mask[:, None] & c_mask[None, :]

        x_base = x_ptr + t_first.to(tl.int64) * stride_x_t + c_first
        x_offs = ii[:, None] * stride_x_t + cc[None, :]
        s_base = state_ptr + slot.to(tl.int64) * stride_s_b + c_first.to(tl.int64) * stride_s_c
        s_offs_c = cc * stride_s_c

        x0_raw = tl.load(x_base + x_offs, mask=mask, other=0.0)
        if tile == 0:
            x1 = _first_tile_tap(x_base, x_offs, s_base, s_offs_c, ii, mask, has_init, stride_x_t,
                                 stride_s_k, 1, WIDTH)
            x2 = _first_tile_tap(x_base, x_offs, s_base, s_offs_c, ii, mask, has_init, stride_x_t,
                                 stride_s_k, 2, WIDTH)
            x3 = _first_tile_tap(x_base, x_offs, s_base, s_offs_c, ii, mask, has_init, stride_x_t,
                                 stride_s_k, 3, WIDTH)
        else:
            x1 = tl.load(x_base + x_offs - stride_x_t, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_base + x_offs - 2 * stride_x_t, mask=mask, other=0.0).to(tl.float32)
            x3 = tl.load(x_base + x_offs - 3 * stride_x_t, mask=mask, other=0.0).to(tl.float32)

        w_base = w_ptr + c_first * stride_w_c
        w0 = tl.load(w_base + cc * stride_w_c + 0 * stride_w_k).to(tl.float32)
        w1 = tl.load(w_base + cc * stride_w_c + 1 * stride_w_k).to(tl.float32)
        w2 = tl.load(w_base + cc * stride_w_c + 2 * stride_w_k).to(tl.float32)
        w3 = tl.load(w_base + cc * stride_w_c + 3 * stride_w_k).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + c_first + cc).to(tl.float32)
            acc = tl.broadcast_to(bias[None, :], [BLOCK_T, HEAD_DIM])
        else:
            acc = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)
        acc = _fma_rn(tl.broadcast_to(w0[None, :], [BLOCK_T, HEAD_DIM]), x3, acc)
        acc = _fma_rn(tl.broadcast_to(w1[None, :], [BLOCK_T, HEAD_DIM]), x2, acc)
        acc = _fma_rn(tl.broadcast_to(w2[None, :], [BLOCK_T, HEAD_DIM]), x1, acc)
        acc = _fma_rn(tl.broadcast_to(w3[None, :], [BLOCK_T, HEAD_DIM]), x0_raw.to(tl.float32),
                      acc)
        if SILU:
            acc = _silu_ref(acc)
        res = acc.to(x_ptr.dtype.element_ty)
        # A padded sequence is skipped by the C++ conv, which leaves the input.
        res = tl.where(is_pad, x0_raw, res)
        rows = t_first + ii
        row_mask = t_mask

        if (tile == 0) & (~is_pad):
            offs_c = c_first + cc
            s0 = _final_state_tap(x_ptr, s_base, s_offs_c, offs_c, c_mask, seq_start, seq_end,
                                  has_init, stride_x_t, stride_s_k, 0, WIDTH)
            s1 = _final_state_tap(x_ptr, s_base, s_offs_c, offs_c, c_mask, seq_start, seq_end,
                                  has_init, stride_x_t, stride_s_k, 1, WIDTH)
            s2 = _final_state_tap(x_ptr, s_base, s_offs_c, offs_c, c_mask, seq_start, seq_end,
                                  has_init, stride_x_t, stride_s_k, 2, WIDTH)
            # All reads of this sequence's initial state for these channels
            # happen in this CTA; finish them before overwriting the state.
            tl.debug_barrier()
            tl.store(s_base + s_offs_c + 0 * stride_s_k, s0, mask=c_mask)
            tl.store(s_base + s_offs_c + 1 * stride_s_k, s1, mask=c_mask)
            tl.store(s_base + s_offs_c + 2 * stride_s_k, s2, mask=c_mask)
    else:
        d_first = (pid_t - max_p_tiles) * BLOCK_T
        d_rows = d_first + ii
        row_mask = d_rows < num_decode_tokens
        res = tl.load(
            dec_ptr + d_rows[:, None].to(tl.int64) * stride_dec_t + c_first + cc[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        rows = num_prefill_tokens + d_rows

    rows64 = rows.to(tl.int64)
    st_mask = row_mask[:, None]
    if pid_h < 2 * NUM_K_HEADS:
        y = res.to(tl.float32)
        inv = 1.0 / tl.sqrt(_sumsq(y, BLOCK_T, HEAD_DIM, REDUCE, WARP_SEQ, TREE) + l2_norm_eps)
        y *= inv[:, None]
        if pid_h < NUM_K_HEADS:
            out_ptr = q_ptr
            hh = pid_h
        else:
            out_ptr = k_ptr
            hh = pid_h - NUM_K_HEADS
        offs = rows64[:, None] * (NUM_K_HEADS * HEAD_DIM) + hh * HEAD_DIM + cc[None, :]
        tl.store(out_ptr + offs, y, mask=st_mask)
    else:
        vh = pid_h - 2 * NUM_K_HEADS
        offs = rows64[:, None] * (NUM_V_HEADS * HEAD_DIM) + vh * HEAD_DIM + cc[None, :]
        tl.store(v_ptr + offs, res, mask=st_mask)

        a_values = tl.load(a_ptr + rows64 * a_stride_t + vh * a_stride_h, mask=row_mask,
                           other=0.0).to(tl.float32)
        b_values = tl.load(b_ptr + rows64 * b_stride_t + vh * b_stride_h, mask=row_mask,
                           other=0.0).to(tl.float32)
        A_log = tl.load(A_log_ptr + vh).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + vh).to(tl.float32)
        gate_input = a_values + dt_bias
        softplus = tl.where(
            gate_input <= softplus_threshold,
            tl.log(1.0 + tl.exp(gate_input)),
            gate_input,
        )
        g_values = -tl.exp(A_log) * softplus
        beta_values = tl.sigmoid(b_values)
        gate_offs = rows64 * NUM_V_HEADS + vh
        tl.store(g_ptr + gate_offs, g_values, mask=row_mask)
        tl.store(beta_ptr + gate_offs, beta_values, mask=row_mask)


def gdn_prefill_onepass_enabled() -> bool:
    """Kill switch: TRTLLM_GDN_ONEPASS_PREFILL=0 restores the multi-kernel path."""
    return os.environ.get("TRTLLM_GDN_ONEPASS_PREFILL", "1") != "0"


def _main_reduce_layout(num_prefill_tokens: int, has_decode: bool) -> int:
    """Reduce layout Triton picks for _fused_gdn_post_conv_kernel's q/k tile.

    The channel-major prefill input has token stride 1 and channel stride T;
    Triton specializes that stride (== 1, or divisible by 16) and the load
    vectorization decides the blocked layout of the [16, 128] tile.
    """
    if has_decode or num_prefill_tokens == 1:
        return 4  # sizePerThread [1, 8], threadsPerWarp [2, 16], warpsPerCTA [4, 1]
    if num_prefill_tokens % 16 == 0:
        return 1  # sizePerThread [8, 1], threadsPerWarp [2, 16], warpsPerCTA [1, 4]
    return 5  # sizePerThread [1, 1], threadsPerWarp [16, 2], warpsPerCTA [1, 4]


_BLOCK_T = int(os.environ.get("TRTLLM_GDN_ONEPASS_BLOCK_T", "16"))
_NUM_WARPS = int(os.environ.get("TRTLLM_GDN_ONEPASS_WARPS", "4"))


def gdn_prefill_onepass(
    x: torch.Tensor,
    decode: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    activation: str | None = "silu",
    l2_norm_eps: float = 1e-6,
    softplus_threshold: float = 20.0,
    pad_slot_id: int = PAD_SLOT_ID,
    reduce: int | None = None,
    warp_seq: bool = False,
    tree: bool = False,
    block_t: int | None = None,
    num_warps: int | None = None,
):
    """One-pass GDN prefill front end.

    Args:
        x: token-major prefill activations ``[num_prefill_tokens, conv_dim]``
            (unit channel stride, any row stride).
        decode: optional convolved decode rows ``[num_decode_tokens, conv_dim]``.
        Others as in ``gdn_prefill_transpose_conv`` and ``fused_gdn_post_conv``.
    Returns:
        q, k ``[1, T, num_k_heads, D]``, v ``[1, T, num_v_heads, D]``,
        g, beta ``[1, T, num_v_heads]`` fp32 (T = prefill + decode tokens);
        ``conv_states`` is updated in place.
    """
    assert head_k_dim == head_v_dim == 128
    assert x.dim() == 2 and x.stride(1) == 1
    num_prefill_tokens, dim = x.shape
    assert dim == (2 * num_k_heads + num_v_heads) * head_k_dim
    width = weight.shape[-1]
    assert width == 4
    weight2d = weight.view(dim, width) if weight.dim() != 2 else weight
    num_decode_tokens = 0 if decode is None else decode.shape[0]
    num_tokens = num_prefill_tokens + num_decode_tokens
    dev = x.device
    q = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=x.dtype, device=dev)
    k = torch.empty_like(q)
    v = torch.empty((1, num_tokens, num_v_heads, head_v_dim), dtype=x.dtype, device=dev)
    g = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=dev)
    beta = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=dev)
    num_seqs = query_start_loc.shape[0] - 1
    if num_tokens == 0 or num_seqs <= 0:
        return q, k, v, g, beta
    bt = block_t or _BLOCK_T
    nw = num_warps or _NUM_WARPS
    if reduce is None:
        reduce = _main_reduce_layout(num_prefill_tokens, decode is not None)
    block_n = max(16, triton.next_power_of_2(num_seqs))
    grid_t = triton.cdiv(num_prefill_tokens, bt) + num_seqs + triton.cdiv(num_decode_tokens, bt)
    dec = decode if decode is not None else x
    _gdn_prefill_onepass_kernel[(grid_t, 2 * num_k_heads + num_v_heads)](
        x,
        weight2d,
        bias if bias is not None else weight2d,
        conv_states,
        query_start_loc,
        cache_indices if cache_indices is not None else query_start_loc,
        has_initial_state if has_initial_state is not None else query_start_loc,
        dec,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        num_seqs,
        num_prefill_tokens,
        num_decode_tokens,
        x.stride(0),
        dec.stride(0),
        weight2d.stride(0),
        weight2d.stride(1),
        conv_states.stride(-3) if conv_states.dim() >= 3 else 0,
        conv_states.stride(-2),
        conv_states.stride(-1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        pad_slot_id,
        l2_norm_eps,
        softplus_threshold,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_DIM=head_k_dim,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        HAS_CACHE_IDX=cache_indices is not None,
        HAS_INIT_PTR=has_initial_state is not None,
        HAS_DECODE=decode is not None,
        WIDTH=width,
        BLOCK_N=block_n,
        BLOCK_T=bt,
        REDUCE=reduce,
        WARP_SEQ=warp_seq,
        TREE=tree,
        num_warps=nw,
    )
    return q, k, v, g, beta
