# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused greedy sampling: argmax over the vocabulary, cast and scatter, in one launch.

``torch.argmax(logits, -1).to(int32)`` followed by ``scatter_`` runs a single-CTA
reduce per row plus two small kernels. This kernel splits every row over many
CTAs. Each CTA packs its elements into an orderable int64 key

    key = (orderable(value) << 31) | (2**31 - 1 - index)

so the maximum key carries the maximum value and, among equal maxima, the
smallest index -- exactly torch.argmax's rule. NaN maps to the largest value
(torch returns the first NaN) and -0.0 to +0.0 (they compare equal in torch).
CTAs ``atomic_max`` their key into a per-row slot; the last CTA of a row (per-row
arrival counter) decodes the index, writes the token to ``out`` and to
``new_tokens[dest[row], :]``, and resets slot and counter to zero for the next
call, so no memset is needed.
"""

import os

import torch
import triton
import triton.language as tl

__all__ = ["fused_greedy_argmax_scatter", "fused_greedy_argmax_enabled"]

_BLOCK = int(os.environ.get("TRTLLM_FUSED_GREEDY_ARGMAX_BLOCK", "2048"))
_NUM_WARPS = int(os.environ.get("TRTLLM_FUSED_GREEDY_ARGMAX_WARPS", "4"))
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def fused_greedy_argmax_enabled() -> bool:
    return os.environ.get("TRTLLM_FUSED_GREEDY_ARGMAX", "1") != "0"


@triton.jit
def _greedy_argmax_scatter_kernel(
    logits_ptr,
    stride_logits_row,
    dest_ptr,
    new_tokens_ptr,
    stride_tokens_row,
    stride_tokens_beam,
    out_ptr,
    slot_ptr,
    count_ptr,
    vocab_size,
    num_chunks,
    BEAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab_size
    x = tl.load(
        logits_ptr + row.to(tl.int64) * stride_logits_row + offs, mask=mask, other=0.0
    ).to(tl.float32)
    s = x.to(tl.int32, bitcast=True)
    # Signed-orderable int32: flip the magnitude bits of negative floats.
    o = tl.where(s < 0, s ^ 0x7FFFFFFF, s)
    # -0.0 (bits 0x80000000 -> -1 after the flip) compares equal to +0.0.
    o = tl.where(s == (-2147483647 - 1), 0, o)
    # Every NaN is the maximum (torch.argmax returns the first NaN).
    o = tl.where(x != x, 2147483647, o)
    key = ((o.to(tl.int64) + 2147483647 + 1) << 31) | (2147483647 - offs.to(tl.int64))
    key = tl.where(mask, key, 0)
    kmax = tl.max(key, axis=0)
    tl.atomic_max(slot_ptr + row, kmax)
    prev = tl.atomic_add(count_ptr + row, 1)
    if prev == num_chunks - 1:
        final = tl.atomic_xchg(slot_ptr + row, 0)
        tl.atomic_xchg(count_ptr + row, 0)
        idx = 2147483647 - (final & 2147483647)
        tok = idx.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row, tok)
        dest = tl.load(dest_ptr + row).to(tl.int64)
        for beam in tl.static_range(BEAM):
            tl.store(new_tokens_ptr + dest * stride_tokens_row + beam * stride_tokens_beam, tok)


_scratch: dict = {}


def _get_scratch(device: torch.device, rows: int):
    """Per-device slot/counter scratch, zero between calls (the kernel resets it)."""
    key = device.index
    cur = _scratch.get(key)
    if cur is None or cur[0].numel() < rows:
        if torch.cuda.is_current_stream_capturing():
            return None
        cap = max(64, triton.next_power_of_2(rows))
        cur = (
            torch.zeros(cap, dtype=torch.int64, device=device),
            torch.zeros(cap, dtype=torch.int32, device=device),
        )
        _scratch[key] = cur
    return cur


def fused_greedy_argmax_scatter(
    logits: torch.Tensor,
    new_tokens: torch.Tensor,
    dest_indices: torch.Tensor,
    max_beam_width: int,
):
    """Return int tokens [B] (new_tokens.dtype) and scatter them, or None when unsupported."""
    if (
        max_beam_width != 1
        or not logits.is_cuda
        # Never JIT/load the Triton module inside an open CUDA-graph capture.
        or torch.cuda.is_current_stream_capturing()
        or logits.dim() != 2
        or logits.dtype not in _SUPPORTED_DTYPES
        or logits.stride(1) != 1
        or logits.shape[0] == 0
        or logits.shape[1] >= 2**31 - 1
        or dest_indices.dim() != 1
        or dest_indices.shape[0] != logits.shape[0]
        or dest_indices.device != logits.device
        or new_tokens.device != logits.device
        or new_tokens.dtype not in (torch.int32, torch.int64)
    ):
        return None
    tokens_2d = new_tokens.view(-1, *new_tokens.shape[2:])
    if tokens_2d.dim() != 2:
        return None
    rows, vocab = logits.shape
    scratch = _get_scratch(logits.device, rows)
    if scratch is None:
        return None
    slot, count = scratch
    out = torch.empty(rows, dtype=new_tokens.dtype, device=logits.device)
    num_chunks = triton.cdiv(vocab, _BLOCK)
    _greedy_argmax_scatter_kernel[(rows, num_chunks)](
        logits,
        logits.stride(0),
        dest_indices,
        tokens_2d,
        tokens_2d.stride(0),
        tokens_2d.stride(1),
        out,
        slot,
        count,
        vocab,
        num_chunks,
        BEAM=max_beam_width,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out
