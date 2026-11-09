# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 wo_a input glue for ``project_sparse_attn_output`` (prefill / any M off the decode route).

Main's SM120 chain ahead of the grouped 128x128-block FP8 ``o_a_proj`` BMM is

1. ``torch.ops.trtllm.mla_rope_inplace`` (BF16 inverse RoPE on the 64 rope dims of each 512-dim head,
   in place), then
2. ``fp8_utils.per_token_quant_and_transform(x.view(M, G, -1), need_permute102=True)`` (1x128 UE8M0
   quant, fp8 ``[G, M, gw]`` + int32 scale words ``[G, gw/512, align(M, 4)]`` viewed as
   ``transpose(1, 2)[:, :M, :]``).

``inv_rope_quant_permute102`` does both in one Triton launch and returns exactly what step 2 returns
(same bytes, same scale words, same view/strides), without writing the rotated BF16 back. The RoPE
arithmetic is ``csa2/b12x_wo.py::_inv_rope_quant128_kernel`` ROPE_VARIANT=0 (bitwise equal to
``mla_rope_inplace``); the quant arithmetic is ``_per_token_quant_and_transform_kernel``'s.
"""


import torch
import triton
import triton.language as tl

from tensorrt_llm.quantization.utils.fp8_utils import check_sf_layout

# Graph-replay sweep on RTX PRO 6000 (experiments/ascent/prefill_wo_quant/bench.py sweep, jobs
# 1925/1928): the full [BLOCK_M, 4, 128] tile, BLOCK_M=4, 4 warps is best or within ~2 us of best
# at M=16..4096.
_BLOCK_M = 4
_NUM_WARPS = 4
_HEAD_DIM = 512
_WORD = 512  # 4 x 128-column quant groups per int32 scale word
# per_token_quant_and_transform passes torch.finfo(float8_e4m3fn).max (448.0) the same way.
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# Set after the first eager launch; a CUDA-graph capture never JIT-compiles this kernel.
_compiled = False


@triton.jit(do_not_specialize=["positions", "m", "s_stride_g", "s_stride_w"])
def _inv_rope_quant_permute102_kernel(
    o,
    positions,
    cos_sin,
    values,
    scales,
    m,
    cs_stride_pos,
    s_stride_g,
    s_stride_w,
    fp8_max,
    fp8_min,
    O_STRIDE: tl.constexpr,
    GROUP_WIDTH: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    w = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < m
    rows64 = rows.to(tl.int64)
    # Row strides are compile-time multiples of 512 elements, so loads/stores vectorize.
    in_base = o + rows64[:, None] * O_STRIDE + group * GROUP_WIDTH + w * 512
    v_stride_g = m.to(tl.int64) * GROUP_WIDTH
    out_base = values + group.to(tl.int64) * v_stride_g + rows64[:, None] * GROUP_WIDTH + w * 512
    # [BLOCK_M, 4, 128]: one 512-wide word (= one head) per row in one shot.
    d = tl.arange(0, 4)[:, None] * 128 + tl.arange(0, 128)[None, :]
    d3 = d[None, :, :]
    in3 = in_base[:, :, None]
    ok3 = row_ok[:, None, None]
    src = tl.load(in3 + d3, mask=ok3, other=0.0).to(tl.float32)
    is_rope = d3 >= NOPE_DIM
    rmask = ok3 & is_rope
    partner = tl.load(in3 + (d3 ^ 1), mask=rmask, other=0.0).to(tl.float32)
    pos = tl.load(positions + rows, mask=row_ok, other=0)
    ci = tl.maximum((d3 - NOPE_DIM) >> 1, 0)
    cbase = cos_sin + pos.to(tl.int64)[:, None, None] * cs_stride_pos
    c = tl.load(cbase + ci, mask=rmask, other=1.0)
    s = tl.load(cbase + HALF_ROPE + ci, mask=rmask, other=0.0)
    # mla_rope_inplace (interleaved, inverse): coef = (cos, -sin),
    # x' = coef.x * x - coef.y * y, y' = coef.x * y + coef.y * x, rounded to BF16.
    ns = -s
    r_even = tl.fma(c, src, -(ns * partner))
    r_odd = tl.fma(c, src, ns * partner)
    rot = tl.where((d3 & 1) == 0, r_even, r_odd)
    rot = rot.to(tl.bfloat16).to(tl.float32)
    v = tl.where(is_rope, rot, src)
    # fp8_utils._per_token_quant_and_transform_kernel (SCALE_UE8M0) arithmetic.
    amax = tl.maximum(tl.max(tl.abs(v), axis=2), 1e-10)
    sc = amax / fp8_max
    sc = tl.exp2(tl.ceil(tl.log2(tl.abs(sc))))
    q = tl.clamp(v / sc[:, :, None], fp8_min, fp8_max).to(values.dtype.element_ty)
    tl.store(out_base[:, :, None] + d3, q, mask=ok3)
    shift = tl.arange(0, 4) * 8
    word = tl.sum((sc.to(tl.int32, bitcast=True) >> 23) << shift[None, :], axis=1)
    tl.store(scales + group * s_stride_g + w * s_stride_w + rows, word, mask=row_ok)


def supports(
    o3: torch.Tensor, cos_sin: torch.Tensor, groups: int, nope_dim: int, rope_dim: int
) -> bool:
    """Whether ``inv_rope_quant_permute102`` covers this call (else main's chain runs)."""
    if o3.dim() != 3 or o3.dtype != torch.bfloat16 or o3.shape[0] == 0:
        return False
    m, heads, head_dim = o3.shape
    if head_dim != _HEAD_DIM or nope_dim + rope_dim != head_dim or rope_dim % 2:
        return False
    if nope_dim % 2 or groups <= 0 or heads % groups:
        return False
    if o3.stride(-1) != 1 or o3.stride(-2) != head_dim or o3.stride(0) != heads * head_dim:
        return False
    if cos_sin.dtype != torch.float32 or cos_sin.stride(-1) != 1:
        return False
    if cos_sin.dim() == 3 and (
        cos_sin.shape[1] != 2
        or cos_sin.shape[2] != rope_dim // 2
        or cos_sin.stride(1) != rope_dim // 2
    ):
        return False
    if torch.cuda.is_current_stream_capturing() and not _compiled:
        return False
    return True


def inv_rope_quant_permute102(
    o3: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    groups: int,
    nope_dim: int,
    rope_dim: int,
):
    """Inverse RoPE + 1x128 UE8M0 quant + permute102 of ``o3`` [M, heads, 512] (not modified).

    Returns ``(fp8 [G, M, gw], scale int32 [G, M, gw/512] view)`` exactly as
    ``per_token_quant_and_transform(rope(o3).view(M, G, -1), need_permute102=True)``.
    """
    global _compiled
    m, heads, head_dim = o3.shape
    gw = heads // groups * head_dim
    words = gw // _WORD
    m_padded = (m + 3) // 4 * 4
    values = torch.empty((groups, m, gw), dtype=torch.float8_e4m3fn, device=o3.device)
    scale = torch.empty((groups, words, m_padded), dtype=torch.int32, device=o3.device)
    _inv_rope_quant_permute102_kernel[(triton.cdiv(m, _BLOCK_M), groups, words)](
        o3,
        positions,
        cos_sin,
        values,
        scale,
        m,
        cos_sin.stride(0),
        scale.stride(0),
        scale.stride(1),
        _FP8_MAX,
        -_FP8_MAX,
        O_STRIDE=o3.stride(0),
        GROUP_WIDTH=gw,
        NOPE_DIM=nope_dim,
        HALF_ROPE=rope_dim // 2,
        BLOCK_M=_BLOCK_M,
        num_warps=_NUM_WARPS,
    )
    if not torch.cuda.is_current_stream_capturing():
        _compiled = True
    scale = scale.transpose(1, 2)[:, :m, :]
    check_sf_layout(scale, m, gw, (1, 128), num_groups=groups, tma_stride_check=True)
    return values, scale
