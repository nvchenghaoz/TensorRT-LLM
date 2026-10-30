# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton LayerNorm + adaLN modulate + per-token E4M3 quantize, one program per row (D = 3072).

The head of the FP8 row-wise block (content_mode_blocks.image.0, a mode block: [1, 1, D]
shift / scale rows, no select). Bitwise contract:

- the modulated activation is Inductor's LN + modulate kernel for the block, copied
  verbatim (looped Welford with the same [XBLOCK=1, R0_BLOCK=1024] layout and num_warps,
  the bf16 round points of emulate_precision_casts, libdevice.rsqrt,
  enable_fp_fusion=False). The LayerNorm affine is the identity (no weight, no bias): the
  weight multiply is dropped (exact) and the bias add kept as + 0.0.
- data / scales: the per-token E4M3 recipe on the bf16-rounded activation (scale =
  max(amax / 448, 1e-12) round-to-nearest; q = clamp(x / scale, +-448) RN satfinite).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

D = 3072
_R0_BLOCK = 1024
_NUM_WARPS = 8


@triton.jit
def _modulated(
    x_ptr, shift_ptr, scale_ptr, x0, r0_index, r0_mask, mean, m2, r0_numel: tl.constexpr,
    EPS: tl.constexpr,
):  # fmt: skip
    # Same op order and round points as Inductor's fused LN + modulate kernel.
    sc = tl.load(scale_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
    one = tl.full([1, 1], 1.0, tl.float32)
    one_plus = (sc + one).to(tl.bfloat16).to(tl.float32)
    xv = tl.load(x_ptr + (r0_index + r0_numel * x0), r0_mask, other=0.0).to(tl.float32)
    centered = xv - mean
    var = m2 / tl.full([1, 1], r0_numel, tl.float32)
    rstd = libdevice.rsqrt(var + tl.full([1, 1], EPS, tl.float32))
    normed = ((centered * rstd) + tl.full([1, 1], 0.0, tl.float32)).to(tl.bfloat16).to(tl.float32)
    prod = (one_plus * normed).to(tl.bfloat16).to(tl.float32)
    return (prod + sh).to(tl.bfloat16)


@triton.jit
def _adaln_fp8r(
    x_ptr,
    shift_ptr,
    scale_ptr,
    q_ptr,
    s_ptr,
    r0_numel: tl.constexpr,
    EPS: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """One row per program: the modulated row's E4M3 data and fp32 scale."""
    tl.static_assert(r0_numel == 3 * R0_BLOCK)
    xoffset = tl.program_id(0) * XBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    # Welford statistics, verbatim from Inductor's looped reduction.
    t_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    t_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    t_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        v = tl.load(
            x_ptr + (r0_index + r0_numel * x0),
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        v = tl.broadcast_to(v.to(tl.bfloat16).to(tl.float32), [XBLOCK, R0_BLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            v, t_mean, t_m2, t_weight, r0_offset == 0
        )
        t_mean = tl.where(r0_mask, mean_next, t_mean)
        t_m2 = tl.where(r0_mask, m2_next, t_m2)
        t_weight = tl.where(r0_mask, weight_next, t_weight)
    mean_r, m2_r, _ = triton_helpers.welford(t_mean, t_m2, t_weight, 1)
    mean = mean_r[:, None]
    m2 = m2_r[:, None]
    # The row is three register-resident chunks: modulate, amax, quantize.
    c0 = r0_base
    c1 = R0_BLOCK + r0_base
    c2 = 2 * R0_BLOCK + r0_base
    m0 = c0 < r0_numel
    m1 = c1 < r0_numel
    m2_ = c2 < r0_numel
    h0 = _modulated(x_ptr, shift_ptr, scale_ptr, x0, c0, m0, mean, m2, r0_numel, EPS)
    h1 = _modulated(x_ptr, shift_ptr, scale_ptr, x0, c1, m1, mean, m2, r0_numel, EPS)
    h2 = _modulated(x_ptr, shift_ptr, scale_ptr, x0, c2, m2_, mean, m2, r0_numel, EPS)
    f0 = h0.to(tl.float32)
    f1 = h1.to(tl.float32)
    f2 = h2.to(tl.float32)
    row_amax = tl.maximum(
        tl.maximum(tl.max(tl.abs(f0), 1), tl.max(tl.abs(f1), 1)), tl.max(tl.abs(f2), 1)
    )[:, None]
    scale = tl.maximum(tl.div_rn(row_amax, 448.0), 1.0e-12)
    q0 = tl.minimum(tl.maximum(tl.div_rn(f0, scale), -448.0), 448.0)
    q1 = tl.minimum(tl.maximum(tl.div_rn(f1, scale), -448.0), 448.0)
    q2 = tl.minimum(tl.maximum(tl.div_rn(f2, scale), -448.0), 448.0)
    tl.store(q_ptr + (c0 + r0_numel * x0), q0.to(tl.float8e4nv), m0)
    tl.store(q_ptr + (c1 + r0_numel * x0), q1.to(tl.float8e4nv), m1)
    tl.store(q_ptr + (c2 + r0_numel * x0), q2.to(tl.float8e4nv), m2_)
    tl.store(s_ptr + x0, scale)


@torch.library.custom_op("trtllm::flux3_klein_fp8r_adaln_quant", mutates_args=())
def fp8r_adaln_quant_op(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 data [M, D], fp32 row scales [M]) of ``(1 + scale) * LayerNorm(x) + shift``;
    x [1, S, D] contiguous bf16, shift / scale [1, 1, D] rows with a unit last stride."""
    rows = x.numel() // D
    data = torch.empty((rows, D), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows,), device=x.device, dtype=torch.float32)
    if rows:
        _adaln_fp8r[(rows,)](
            x,
            shift,
            scale,
            data,
            scales,
            D,
            eps,
            1,
            _R0_BLOCK,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
    return data, scales


@fp8r_adaln_quant_op.register_fake
def _fp8r_adaln_quant_fake(x, shift, scale, eps):
    rows = x.numel() // D
    return (
        x.new_empty((rows, D), dtype=torch.float8_e4m3fn),
        x.new_empty((rows,), dtype=torch.float32),
    )
