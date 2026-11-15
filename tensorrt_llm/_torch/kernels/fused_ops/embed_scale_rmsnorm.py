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
"""Fused scaled embedding lookup (+ the first decoder layer's input RMSNorm).

One Triton launch replaces the non-parallel masked embedding chain of a scaled
word embedding (Gemma-style) followed by the first layer's standalone norm:

    get_masked_input_and_mask(ids, 0, V)   (4-5 small int kernels)
    F.embedding(ids, W)                    (read + write [M, H])
    masked_fill_(mask, 0)                  (read + write [M, H])
    * embed_scale (0-d bf16 tensor)        (read + write [M, H])
    flashinfer rmsnorm(h, w, eps)          (read + write [M, H])

with one row read and two row writes (h and rmsnorm(h)).

Numerics are bitwise those of the chain:

* out-of-range ids give zero rows (the masked lookup's masked_fill), in-range
  ids copy the table row;
* the scale multiply follows aten's bf16 * bf16-scalar recipe: fp32 opmath
  (the bf16 x bf16 product is exact in fp32) and one bf16 round;
* the norm replays flashinfer's CuTe-DSL ``RMSNormKernel`` (the backend
  ``flashinfer.norm.rmsnorm`` dispatches 2-D bf16 inputs to): each of the
  ``threads_per_row`` lanes accumulates its ``num_vec_blocks x vec_size``
  elements sequentially (block-major, element-minor, read from the PTX:
  88 chained ``fma.rn.f32.bf16`` at H=5376), the lanes combine with the
  warp butterfly at offsets 1, 2, 4, 8, 16, the row's warps combine the same
  way, then ``div.rn.f32`` by H, ``+ eps``, ``rsqrt.approx.ftz.f32`` and
  ``(x * rstd) * (w + 0)`` with a bf16 round. Every reduction step below is an
  explicit two-operand add, so the order cannot be changed by Triton's layout.

Only the CuTe configuration the replay was validated against is accepted
(``threads_per_row == 64``, ``vec_size == 8``, one CTA per row cluster, i.e.
bf16 with 3072 < H <= 6144 and H % 8 == 0); callers keep the unfused norm for
anything else (``rmsnorm_replay_supported``).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _cute_rmsnorm_layout(hidden: int, elem_bytes: int) -> Tuple[int, int, int]:
    """(threads_per_row, vec_size, num_vec_blocks) of flashinfer's CuTe
    RMSNormKernel for a single-CTA row (flashinfer/norm/kernels/rmsnorm.py)."""
    if hidden <= 64:
        tpr = 8
    elif hidden <= 128:
        tpr = 16
    elif hidden <= 3072:
        tpr = 32
    elif hidden <= 6144:
        tpr = 64
    elif hidden <= 16384:
        tpr = 128
    else:
        tpr = 256
    h_align = hidden & (-hidden)
    vec = min(h_align, 16 // elem_bytes)
    nvb = max(1, (hidden // vec + tpr - 1) // tpr)
    return tpr, vec, nvb


def rmsnorm_replay_supported(hidden: int, dtype: torch.dtype,
                             device: torch.device) -> bool:
    """Whether the fused norm replays the rmsnorm flashinfer runs here."""
    if dtype != torch.bfloat16 or hidden % 8 != 0:
        return False
    tpr, vec, _ = _cute_rmsnorm_layout(hidden, 2)
    if tpr != 64 or vec != 8:
        return False
    try:
        from flashinfer.norm import _use_cuda_norm
        # The CUDA-JIT fallback kernel reduces in a different order.
        return not _use_cuda_norm(device)
    except Exception:
        return False


@triton.jit
def _embed_scale_rmsnorm_kernel(
    ids_ptr,  # [M] int token ids
    w_ptr,  # [V, H] bf16 embedding table (contiguous)
    nw_ptr,  # [H] bf16 norm weight (unused when HAS_NORM == False)
    h_ptr,  # [M, H] bf16 scaled embedding out
    n_ptr,  # [M, H] bf16 rmsnorm(h) out (unused when HAS_NORM == False)
    num_embeddings,
    scale,  # fp32 value of the bf16 embed scale
    EPS,
    H: tl.constexpr,
    BH: tl.constexpr,  # next_power_of_2(H)
    TPR: tl.constexpr,  # CuTe threads_per_row
    VEC: tl.constexpr,  # CuTe vec_size (8)
    NVB: tl.constexpr,  # CuTe num_vec_blocks
    HAS_NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    tok = tl.load(ids_ptr + t).to(tl.int64)
    ok = (tok >= 0) & (tok < num_embeddings)
    row = tl.where(ok, tok, 0)
    src = w_ptr + row * H

    cols = tl.arange(0, BH)
    cmask = cols < H
    e = tl.load(src + cols, mask=cmask & ok, other=0.0).to(tl.float32)
    h = (e * scale).to(tl.bfloat16)
    tl.store(h_ptr + t * H + cols, h, mask=cmask)

    if HAS_NORM:
        # Sum of squares in the CuTe kernel's order. Lane l of the row owns
        # columns vb * TPR * VEC + l * VEC + v; it accumulates vb-major,
        # v-minor. The tiles are re-read from the (L1-hot) table row and
        # re-scaled, giving the same bf16 values as h.
        lane = tl.arange(0, TPR)
        sub = tl.arange(0, VEC)
        acc = tl.zeros([TPR], dtype=tl.float32)
        for vb in tl.static_range(NVB):
            c = vb * TPR * VEC + lane[:, None] * VEC + sub[None, :]
            x = tl.load(src + c, mask=(c < H) & ok, other=0.0).to(tl.float32)
            x = (x * scale).to(tl.bfloat16).to(tl.float32)
            # v = 4a + 2b + c: peel the three bits to get v0..v7 in order.
            x = tl.reshape(x, [TPR, 2, 2, 2])
            p0, p1 = tl.split(x)  # c = 0 / 1 -> [TPR, a, b]
            p00, p01 = tl.split(p0)  # b = 0 / 1 -> [TPR, a]
            p10, p11 = tl.split(p1)
            v0, v4 = tl.split(p00)
            v2, v6 = tl.split(p01)
            v1, v5 = tl.split(p10)
            v3, v7 = tl.split(p11)
            acc = acc + v0 * v0
            acc = acc + v1 * v1
            acc = acc + v2 * v2
            acc = acc + v3 * v3
            acc = acc + v4 * v4
            acc = acc + v5 * v5
            acc = acc + v6 * v6
            acc = acc + v7 * v7
        # Warp butterfly at offsets 1, 2, 4, 8, 16 (lane bit 0 first), then
        # the row's two warps (block_reduce_multirow: w0 + w1).
        s = tl.reshape(acc, [TPR // 32, 2, 2, 2, 2, 2])
        s = tl.sum(s, axis=5)
        s = tl.sum(s, axis=4)
        s = tl.sum(s, axis=3)
        s = tl.sum(s, axis=2)
        s = tl.sum(s, axis=1)
        ssq = tl.sum(s, axis=0)
        mean = tl.math.div_rn(ssq, tl.full([], H, tl.float32))
        rstd = tl.inline_asm_elementwise(
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            [mean + EPS],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        w = tl.load(nw_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        y = (h.to(tl.float32) * rstd) * w
        tl.store(n_ptr + t * H + cols, y.to(tl.bfloat16), mask=cmask)


def embed_scale_rmsnorm(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``h = bf16(masked_embedding(ids, weight) * scale)`` and, with
    ``norm_weight``, ``rmsnorm(h, norm_weight, eps)``, in one launch.

    Args:
        input_ids: int token ids, any shape; ids outside [0, V) give zero rows.
        weight: [V, H] bf16 contiguous embedding table (the full,
            non-parallel table).
        scale: the embed scale as stored (pass ``float`` of the bf16 tensor
            the unfused chain multiplies by).
        norm_weight: optional [H] bf16 RMSNorm weight (plain multiplier); only
            valid where ``rmsnorm_replay_supported`` holds.
        eps: RMSNorm epsilon.

    Returns:
        ``(h, normed)``; h is ``[*input_ids.shape, H]``, normed is [M, H] or None.
    """
    assert weight.dim() == 2 and weight.is_contiguous()
    assert weight.dtype == torch.bfloat16
    ids = input_ids.reshape(-1)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    num_embeddings, hidden = weight.shape
    m = ids.numel()
    h = torch.empty((m, hidden), dtype=weight.dtype, device=weight.device)
    has_norm = norm_weight is not None
    normed = torch.empty_like(h) if has_norm else None
    if has_norm:
        assert norm_weight.shape == (hidden, ) and norm_weight.dtype == torch.bfloat16
        tpr, vec, nvb = _cute_rmsnorm_layout(hidden, 2)
        assert tpr == 64 and vec == 8, "unsupported norm replay layout"
    else:
        tpr, vec, nvb = 64, 8, 1
    if m > 0:
        _embed_scale_rmsnorm_kernel[(m, )](
            ids,
            weight,
            norm_weight if has_norm else weight,
            h,
            normed if has_norm else h,
            num_embeddings,
            float(scale),
            float(eps),
            H=hidden,
            BH=triton.next_power_of_2(hidden),
            TPR=tpr,
            VEC=vec,
            NVB=nvb,
            HAS_NORM=has_norm,
            # Measured on SM120 at H=5376: 2 warps 4.1/16.6/180 us at
            # M=300/2000/8000 vs 5.1/20.8/181 for 4; outputs bitwise equal
            # for 2-16 warps (every reduction step is an explicit add).
            num_warps=2,
        )
    return h.view(*input_ids.shape, hidden), normed


__all__ = ["embed_scale_rmsnorm", "rmsnorm_replay_supported"]
