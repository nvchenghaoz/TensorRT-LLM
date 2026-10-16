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
"""GDN prefill: fused token-major -> channel-major transpose + causal conv1d.

Replaces ``extract_transpose_prefill_slice`` followed by the in-place C++
``causal_conv1d_fwd`` (channel-major kernel) with one Triton kernel that reads
the token-major ``in_proj_qkvz`` column view directly and writes the same
contiguous channel-major ``[conv_dim, num_tokens]`` output and the same final
conv states, bit for bit:

* fp32 math, ``acc = bias (or +0)`` then ``acc = fma(w[k], x[t-3+k], acc)`` for
  k = 0..3 (``__fmaf_rn`` chain of ``causal_conv1d_fwd_kernel``);
* SiLU as ``v * rcp.rn(1 + ex2.approx((-v) * log2e))``, i.e. the non-ftz
  expansion of ``v * __frcp_rn(1.0f + __expf(-v))`` (FAST_MATH is off);
* fp32 -> bf16 round to nearest even;
* halo before a sequence's first token from ``conv_states[slot]`` when
  ``has_initial_state`` else 0; final state taken from the input tail
  (with the ``seqlen < width - 1`` shift of the old state);
* ``cache_indices[seq] == pad_slot_id``: no conv, no state access; the output
  keeps the transposed input (what the in-place C++ kernel leaves behind).

Tiles never straddle sequences (sequence-aligned token tiles), so the only
program that reads a sequence's initial state is its first tile, which is also
the one that writes the final state; a CTA barrier orders the two.

Kill switch: ``TRTLLM_GDN_FUSED_TRANSPOSE_CONV=0`` restores the two-kernel path.
"""

import os

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID


def gdn_fused_transpose_conv_enabled(num_tokens: int | None = None) -> bool:
    """Whether GDN prefill uses the fused transpose + causal-conv kernel.

    Below ``TRTLLM_GDN_FUSED_TRANSPOSE_CONV_MIN_TOKENS`` (default 4096) prefill
    tokens the two-kernel path stays: at 1x2048 its transposed copy stays
    L2-resident on SM120 (128 MB L2) and the pair is as fast as the fused kernel
    (cold L2) or faster (warm L2); from 2x2048 the fused kernel wins.
    """
    if os.environ.get("TRTLLM_GDN_FUSED_TRANSPOSE_CONV", "1") == "0":
        return False
    if num_tokens is None:
        return True
    return num_tokens >= int(os.environ.get("TRTLLM_GDN_FUSED_TRANSPOSE_CONV_MIN_TOKENS", "4096"))


@triton.jit
def _silu_ref(v):
    # Bitwise twin of causal_conv1d_fwd_kernel's v * __frcp_rn(1.0f + __expf(-v))
    # compiled without ftz: __expf(x) = ex2.approx(x * log2e) in libdevice.
    return tl.inline_asm_elementwise(
        "{\n"
        ".reg .f32 t;\n"
        "neg.f32 t, $1;\n"
        "mul.rn.f32 t, t, 0f3FB8AA3B;\n"
        "ex2.approx.f32 t, t;\n"
        "add.rn.f32 t, t, 0f3F800000;\n"
        "rcp.rn.f32 t, t;\n"
        "mul.rn.f32 $0, $1, t;\n"
        "}",
        "=f,f",
        [v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _first_tile_tap(
    x_base,
    x_offs,
    s_base,
    s_offs_c,
    ii,
    mask,
    has_init,
    stride_x_t,
    stride_s_k,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """x[t - K] in a sequence's first tile: the input, or the initial conv state."""
    in_seq = ii >= K
    xv = tl.load(x_base + x_offs - K * stride_x_t, mask=mask & in_seq[:, None], other=0.0)
    w_idx = (WIDTH - 1) + ii - K
    sv = tl.load(
        s_base + s_offs_c[None, :] + w_idx[:, None] * stride_s_k,
        mask=mask & (~in_seq)[:, None] & has_init,
        other=0.0,
    )
    return tl.where(in_seq[:, None], xv, sv).to(tl.float32)


@triton.jit
def _final_state_tap(
    x_ptr,
    s_base,
    s_offs_c,
    offs_c,
    c_mask,
    seq_start,
    seq_end,
    has_init,
    stride_x_t,
    stride_s_k,
    W: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Final conv_states[slot, c, W] (input dtype), computed from the input."""
    seqlen = seq_end - seq_start
    full = seqlen >= WIDTH - 1
    from_x = full | (W >= (WIDTH - 1) - seqlen)
    src_t = tl.where(full, seq_end - (WIDTH - 1) + W, seq_start + W - ((WIDTH - 1) - seqlen))
    xv = tl.load(
        x_ptr + src_t.to(tl.int64) * stride_x_t + offs_c,
        mask=c_mask & from_x,
        other=0.0,
    )
    sv = tl.load(
        s_base + s_offs_c + (W + seqlen) * stride_s_k,
        mask=c_mask & (~from_x) & has_init,
        other=0.0,
    )
    return tl.where(from_x, xv, sv)


@triton.jit
def _gdn_prefill_transpose_conv_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    state_ptr,
    qsl_ptr,
    cache_idx_ptr,
    has_init_ptr,
    num_seqs,
    num_tokens,
    dim,
    stride_x_t,
    stride_w_c,
    stride_w_k,
    stride_s_b,
    stride_s_c,
    stride_s_k,
    pad_slot_id,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    HAS_CACHE_IDX: tl.constexpr,
    HAS_INIT_PTR: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Map the flat tile id to (sequence, tile within sequence). Every sequence
    # gets max(cdiv(len, BLOCK_T), 1) tiles so that tile 0 always exists and
    # owns the sequence's state read/write.
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_seqs
    starts = tl.load(qsl_ptr + offs_n, mask=n_mask, other=0)
    ends = tl.load(qsl_ptr + offs_n + 1, mask=n_mask, other=0)
    ntiles = tl.where(n_mask, tl.maximum((ends - starts + BLOCK_T - 1) // BLOCK_T, 1), 0)
    cum = tl.cumsum(ntiles, 0)
    seq = tl.sum((cum <= pid_t).to(tl.int32), 0)
    if seq >= num_seqs:
        return
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
    c_first = pid_c * BLOCK_C
    ii = tl.arange(0, BLOCK_T)
    cc = tl.arange(0, BLOCK_C)
    t_mask = (t_first + ii) < seq_end
    c_mask = (c_first + cc) < dim
    mask = t_mask[:, None] & c_mask[None, :]

    # 64-bit scalar bases, 32-bit in-tile offsets (channel stride is 1).
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
        # t - 3 >= seq_start for every token past the first tile (BLOCK_T >= 3).
        x1 = tl.load(x_base + x_offs - stride_x_t, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + x_offs - 2 * stride_x_t, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_base + x_offs - 3 * stride_x_t, mask=mask, other=0.0).to(tl.float32)

    w_base = w_ptr + c_first * stride_w_c
    w0 = tl.load(w_base + cc * stride_w_c + 0 * stride_w_k, mask=c_mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_base + cc * stride_w_c + 1 * stride_w_k, mask=c_mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_base + cc * stride_w_c + 2 * stride_w_k, mask=c_mask, other=0.0).to(tl.float32)
    w3 = tl.load(w_base + cc * stride_w_c + 3 * stride_w_k, mask=c_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + c_first + cc, mask=c_mask, other=0.0).to(tl.float32)
        acc = tl.broadcast_to(bias[None, :], [BLOCK_T, BLOCK_C])
    else:
        acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    acc = _fma_rn(tl.broadcast_to(w0[None, :], [BLOCK_T, BLOCK_C]), x3, acc)
    acc = _fma_rn(tl.broadcast_to(w1[None, :], [BLOCK_T, BLOCK_C]), x2, acc)
    acc = _fma_rn(tl.broadcast_to(w2[None, :], [BLOCK_T, BLOCK_C]), x1, acc)
    acc = _fma_rn(tl.broadcast_to(w3[None, :], [BLOCK_T, BLOCK_C]), x0_raw.to(tl.float32), acc)
    if SILU:
        acc = _silu_ref(acc)
    res = acc.to(out_ptr.dtype.element_ty)
    # A padded sequence is skipped by the C++ kernel, which leaves the
    # transposed input in place.
    res = tl.where(is_pad, x0_raw, res)

    out_base = out_ptr + c_first.to(tl.int64) * num_tokens + t_first
    out_offs = cc[:, None] * num_tokens + ii[None, :]
    tl.store(out_base + out_offs, tl.trans(res), mask=c_mask[:, None] & t_mask[None, :])

    if (tile == 0) & (~is_pad):
        offs_c = c_first + cc
        s0 = _final_state_tap(x_ptr, s_base, s_offs_c, offs_c, c_mask, seq_start, seq_end,
                              has_init, stride_x_t, stride_s_k, 0, WIDTH)
        s1 = _final_state_tap(x_ptr, s_base, s_offs_c, offs_c, c_mask, seq_start, seq_end,
                              has_init, stride_x_t, stride_s_k, 1, WIDTH)
        s2 = _final_state_tap(x_ptr, s_base, s_offs_c, offs_c, c_mask, seq_start, seq_end,
                              has_init, stride_x_t, stride_s_k, 2, WIDTH)
        # Every read of this sequence's initial state (halo taps above and the
        # short-sequence shift here) happens in this CTA; finish them all
        # before any thread overwrites the state.
        tl.debug_barrier()
        tl.store(s_base + s_offs_c + 0 * stride_s_k, s0, mask=c_mask)
        tl.store(s_base + s_offs_c + 1 * stride_s_k, s1, mask=c_mask)
        tl.store(s_base + s_offs_c + 2 * stride_s_k, s2, mask=c_mask)


_BLOCK_T = int(os.environ.get("TRTLLM_GDN_TRANSPOSE_CONV_BLOCK_T", "32"))
_BLOCK_C = int(os.environ.get("TRTLLM_GDN_TRANSPOSE_CONV_BLOCK_C", "64"))
_NUM_WARPS = int(os.environ.get("TRTLLM_GDN_TRANSPOSE_CONV_WARPS", "4"))


def gdn_prefill_transpose_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_t: int | None = None,
    block_c: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Causal conv1d over token-major prefill activations, channel-major output.

    Bitwise equivalent of::

        xt = extract_transpose_prefill_slice(x, x.shape[0], 0, x.shape[1])
        causal_conv1d_fn(xt, weight, bias, activation=activation,
                         conv_states=conv_states, has_initial_state=has_initial_state,
                         cache_indices=cache_indices, query_start_loc=query_start_loc)

    Args:
        x: ``[num_tokens, dim]`` with unit channel stride (any row stride).
        weight: ``[dim, width]`` (width 4).
        bias: optional ``[dim]``.
        conv_states: ``[..., dim, width - 1]``, updated in place.
        query_start_loc: ``[num_seqs + 1]`` int32 cumulative token offsets.
        cache_indices: ``[num_seqs]`` int32 conv-state slots (``pad_slot_id`` skips).
        has_initial_state: ``[num_seqs]`` bool.
    Returns:
        Contiguous ``[dim, num_tokens]`` conv output.
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    assert x.dim() == 2 and x.stride(1) == 1
    num_tokens, dim = x.shape
    width = weight.shape[-1]
    assert width == 4, "fused GDN transpose+conv supports width 4"
    weight2d = weight.view(dim, width) if weight.dim() != 2 else weight
    out = torch.empty(dim, num_tokens, dtype=x.dtype, device=x.device)
    num_seqs = query_start_loc.shape[0] - 1
    if num_seqs <= 0:
        return out
    bt = block_t or _BLOCK_T
    bc = block_c or _BLOCK_C
    nw = num_warps or _NUM_WARPS
    assert bt >= width - 1
    block_n = max(16, triton.next_power_of_2(num_seqs))
    grid = (triton.cdiv(num_tokens, bt) + num_seqs, triton.cdiv(dim, bc))
    _gdn_prefill_transpose_conv_kernel[grid](
        x,
        weight2d,
        bias if bias is not None else weight2d,
        out,
        conv_states,
        query_start_loc,
        cache_indices if cache_indices is not None else query_start_loc,
        has_initial_state if has_initial_state is not None else query_start_loc,
        num_seqs,
        num_tokens,
        dim,
        x.stride(0),
        weight2d.stride(0),
        weight2d.stride(1),
        conv_states.stride(-3) if conv_states.dim() >= 3 else 0,
        conv_states.stride(-2),
        conv_states.stride(-1),
        pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        HAS_CACHE_IDX=cache_indices is not None,
        HAS_INIT_PTR=has_initial_state is not None,
        WIDTH=width,
        BLOCK_N=block_n,
        BLOCK_T=bt,
        BLOCK_C=bc,
        num_warps=nw,
    )
    return out
