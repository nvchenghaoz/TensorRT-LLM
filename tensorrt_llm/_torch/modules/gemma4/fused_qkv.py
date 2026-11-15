# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused QKV prep for Gemma4: per-head RMSNorm + RoPE in one kernel.

Replaces the unfused QKV preparation executed per layer per step. On the
FlashInfer FP8-KV path, that chain is:

    split_qkv (strided views) -> reshape copies (3x direct_copy)
    -> q_norm / k_norm / v_norm (3x rmsnorm)
    -> flashinfer rope in-place on q, k (1 kernel)
    -> q/k/v .to(float8_e4m3fn) in the backend (3x float8_copy)

with a single Triton kernel that reads the packed QKV GEMM output directly
(per-head strided access, no contiguous intermediate), normalizes each head,
applies neox-style RoPE to Q/K heads from the module's fp32 cos/sin table,
and writes separate FP8 Q/K/V or packed BF16 QKV. FlashInfer consumes the
FP8 outputs directly; native TRTLLM consumes the packed BF16 output before
its existing KV-cache quantization and append preprocessing.

Numerics deliberately replicate the unfused path: fp32 accumulation with a
round to bf16 after the norm and again after RoPE, so the final FP8 values
match the reference chain (see tests/unittest/_torch/modules/
test_gemma4_fused_qkv_prep.py).

The kernel processes BLOCK_N tokens per program (2D [BLOCK_N, HALF] tiles)
so each thread issues wide vectorized loads/stores instead of one scalar
element; tile size and warp count are fixed host-side as a pure function of
head_dim (no runtime autotuning, so launches stay deterministic under
CUDA-graph capture). Tuned on B200: 363 -> ~75 us/call at ~6.5k tokens for
the 64-head hd=256 shape (~4.3 TB/s effective), 2.8x on the graph-replayed
decode shape.

The unfused path is kept only for configurations this kernel does not
support (KV-shared layers, non-FP8 KV cache, custom-mask multimodal
prefill, torch.compile).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _split8(t):
    """[BN, L, 8] -> the eight [BN, L] tensors t[..., j], j = 0..7."""
    BN: tl.constexpr = t.shape[0]
    L: tl.constexpr = t.shape[1]
    t = tl.reshape(t, (BN, L, 2, 2, 2))  # j = 4*a2 + 2*a1 + a0, a0 innermost
    e0, e1 = tl.split(t)
    e00, e01 = tl.split(e0)
    e10, e11 = tl.split(e1)
    t0, t4 = tl.split(e00)
    t2, t6 = tl.split(e01)
    t1, t5 = tl.split(e10)
    t3, t7 = tl.split(e11)
    return t0, t1, t2, t3, t4, t5, t6, t7


@triton.jit
def _lane_sq_seq(s, x, FIRST: tl.constexpr):
    """Per-lane sequential sum of squares over 8 contiguous columns.

    x: [BN, L*8] fp32, lane l owns columns [8l, 8l+8). Returns [BN, L]: s plus
    x[8l]^2, x[8l+1]^2, ... added left to right (bf16 squares are exact in fp32).
    """
    BN: tl.constexpr = x.shape[0]
    L: tl.constexpr = x.shape[1] // 8
    x0, x1, x2, x3, x4, x5, x6, x7 = _split8(tl.reshape(x, (BN, L, 8)))
    if FIRST:
        s = x0 * x0
    else:
        s = s + x0 * x0
    s = s + x1 * x1
    s = s + x2 * x2
    s = s + x3 * x3
    s = s + x4 * x4
    s = s + x5 * x5
    s = s + x6 * x6
    s = s + x7 * x7
    return s


@triton.jit
def _pair_tree16(p):
    """[BN, 16] -> [BN]: adjacent pairwise tree (warp butterfly, xor 1, 2, 4, 8)."""
    BN: tl.constexpr = p.shape[0]
    a, b = tl.split(tl.reshape(p, (BN, 8, 2)))
    p = a + b
    a, b = tl.split(tl.reshape(p, (BN, 4, 2)))
    p = a + b
    a, b = tl.split(tl.reshape(p, (BN, 2, 2)))
    p = a + b
    a, b = tl.split(p)
    return a + b


@triton.jit
def _cute_rmsnorm_ssq(x1, x2, HD: tl.constexpr):
    """Sum of squares of [x1 | x2] in FlashInfer's CuTe-DSL RMSNormKernel order.

    For HD 256 and 512 that kernel (flashinfer/norm/kernels/rmsnorm.py) runs one
    warp per row with vec_size 8: lane t owns columns [8t, 8t+8) and, at HD 512,
    [256+8t, 256+8t+8); it sums its squares left to right, then reduces lanes
    with a butterfly over xor offsets 1, 2, 4, 8, 16.
    """
    if HD == 256:
        # x1 holds lanes 0..15, x2 lanes 16..31.
        return _pair_tree16(_lane_sq_seq(0.0, x1, True)) + _pair_tree16(_lane_sq_seq(0.0, x2, True))
    else:
        # x1 holds every lane's first 8 columns, x2 its second 8.
        BN: tl.constexpr = x1.shape[0]
        p = _lane_sq_seq(_lane_sq_seq(0.0, x1, True), x2, False)  # [BN, 32]
        a, b = tl.split(tl.reshape(p, (BN, 16, 2)))
        return _pair_tree16(a + b)


@triton.jit
def _gemma4_qkv_norm_rope_quant_kernel(
    qkv_ptr,  # [N, W] bf16, row stride SW; heads packed [q | k | v], head h at col h*HD
    q_out_ptr,  # [N, NQ*HD] fp8/bf16, possibly a view of packed QKV
    k_out_ptr,  # [N, NK*HD], possibly a view of packed QKV
    v_out_ptr,  # [N, NK*HD], possibly a view of packed QKV
    pos_ptr,  # [N] int positions
    cos_sin_ptr,  # [max_pos, 2, HALF] fp32: [:, 0, :]=cos, [:, 1, :]=sin
    qw_ptr,  # [HD] bf16 q_norm weight
    kw_ptr,  # [HD] bf16 k_norm weight
    SW,  # qkv row stride (elements)
    Q_OUT_SW: tl.constexpr,
    K_OUT_SW: tl.constexpr,
    V_OUT_SW: tl.constexpr,
    NQ,  # num q heads
    NK,  # num kv heads
    EPS,
    N,  # num tokens (rows)
    HD: tl.constexpr,  # head dim
    HALF: tl.constexpr,  # HD // 2 (power of two)
    Q_OUT_FP8: tl.constexpr,
    KV_OUT_FP8: tl.constexpr,
    CUTE_NORM_ORDER: tl.constexpr,  # sum of squares in FlashInfer CuTe RMSNorm order
    K_EQ_V: tl.constexpr,  # qkv is the [q | k] GEMM output; V = v_norm(raw K)
    BLOCK_N: tl.constexpr,  # tokens per program (power of two)
):
    h = tl.program_id(0)
    rows = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    rmask = rows < N
    offs = tl.arange(0, HALF)

    # Head h occupies columns [h*HD, (h+1)*HD) of the packed qkv rows. With
    # K_EQ_V there is no V block: V head j reads the raw K head j at column
    # (NQ + j) * HD (only inputs are read, so K's norm/RoPE never reach V).
    src_h = h
    if K_EQ_V:
        src_h = tl.where(h >= NQ + NK, h - NK, h)
    base = qkv_ptr + rows[:, None] * SW + src_h * HD + offs[None, :]
    x1 = tl.load(base, mask=rmask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF, mask=rmask[:, None], other=0.0).to(tl.float32)

    if CUTE_NORM_ORDER:
        # Bitwise equal to the unfused chain's FlashInfer RMSNorm (SM120 FA2).
        ssq = _cute_rmsnorm_ssq(x1, x2, HD)
    else:
        ssq = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
    rms = tl.math.rsqrt(ssq / HD + EPS)  # [BLOCK_N]

    if h < NQ + NK:
        # Q/K head: norm (with weight) -> bf16 round -> rope -> bf16 round.
        wp = qw_ptr if h < NQ else kw_ptr
        w1 = tl.load(wp + offs).to(tl.float32)
        w2 = tl.load(wp + HALF + offs).to(tl.float32)
        y1 = (x1 * rms[:, None] * w1[None, :]).to(tl.bfloat16).to(tl.float32)
        y2 = (x2 * rms[:, None] * w2[None, :]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
        cs = cos_sin_ptr + pos[:, None] * (2 * HALF) + offs[None, :]
        cos = tl.load(cs, mask=rmask[:, None], other=0.0)
        sin = tl.load(cs + HALF, mask=rmask[:, None], other=0.0)
        if CUTE_NORM_ORDER:
            # FlashInfer RoPE (pos_enc.cuh vec_apply_llama_rope_cos_sin) as
            # nvcc contracts it: x*cos + (+-x_other*sin). The launch disables
            # fp fusion so Triton does not pick another contraction.
            o1 = tl.fma(y1, cos, -(y2 * sin)).to(tl.bfloat16)
            o2 = tl.fma(y2, cos, y1 * sin).to(tl.bfloat16)
        else:
            o1 = (y1 * cos - y2 * sin).to(tl.bfloat16)
            o2 = (y2 * cos + y1 * sin).to(tl.bfloat16)
        if h < NQ:
            q_out = q_out_ptr + rows[:, None] * Q_OUT_SW + h * HD + offs[None, :]
            if Q_OUT_FP8:
                tl.store(q_out, o1.to(tl.float8e4nv), mask=rmask[:, None])
                tl.store(q_out + HALF, o2.to(tl.float8e4nv), mask=rmask[:, None])
            else:
                tl.store(q_out, o1, mask=rmask[:, None])
                tl.store(q_out + HALF, o2, mask=rmask[:, None])
        else:
            k_out = k_out_ptr + rows[:, None] * K_OUT_SW + (h - NQ) * HD + offs[None, :]
            if KV_OUT_FP8:
                tl.store(k_out, o1.to(tl.float8e4nv), mask=rmask[:, None])
                tl.store(k_out + HALF, o2.to(tl.float8e4nv), mask=rmask[:, None])
            else:
                tl.store(k_out, o1, mask=rmask[:, None])
                tl.store(k_out + HALF, o2, mask=rmask[:, None])
    else:
        # V head: weightless norm, no rope (v_norm is applied to the raw v
        # slice; reads happen before any write, matching HF's
        # v_norm-before-k_norm ordering for K=V layers).
        o1 = (x1 * rms[:, None]).to(tl.bfloat16)
        o2 = (x2 * rms[:, None]).to(tl.bfloat16)
        out = v_out_ptr + rows[:, None] * V_OUT_SW + (h - NQ - NK) * HD + offs[None, :]
        if KV_OUT_FP8:
            tl.store(out, o1.to(tl.float8e4nv), mask=rmask[:, None])
            tl.store(out + HALF, o2.to(tl.float8e4nv), mask=rmask[:, None])
        else:
            tl.store(out, o1, mask=rmask[:, None])
            tl.store(out + HALF, o2, mask=rmask[:, None])


def gemma4_fused_qkv_norm_rope_quant(
    qkv: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    out_fp8: bool,
    packed_output: bool = False,
    kv_out_fp8: Optional[bool] = None,
    cute_norm_order: bool = False,
    k_eq_v: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the fused per-head norm + rope + quant over a packed QKV tensor.

    Args:
        qkv: [N, num_q_heads*head_dim + 2*num_kv_heads*head_dim] bf16, packed
            [q | k | v]; may have a row stride larger than its width.
        position_ids: flattenable to [N]; integer rope positions per token.
        cos_sin: [max_positions, 2, head_dim // 2] fp32 contiguous table
            (RotaryEmbedding.rotary_cos_sin layout).
        q_weight/k_weight: [head_dim] q_norm/k_norm weights.
        eps: shared RMSNorm epsilon.
        out_fp8: emit float8_e4m3fn outputs (KV-cache dtype) when True,
            bf16 otherwise.
        packed_output: emit one contiguous BF16 packed-QKV tensor instead of
            separate Q/K/V tensors. Packed FP8 is not supported because the
            native TRTLLM preprocessing contract consumes BF16 packed QKV.
        kv_out_fp8: K/V output dtype override. None (default) follows
            ``out_fp8``; ``out_fp8=False, kv_out_fp8=True`` emits BF16 Q with
            FP8 K/V (FlashInfer FA2 with an FP8 paged KV cache).
        cute_norm_order: sum the squares in the order of FlashInfer's CuTe-DSL
            RMSNormKernel (per-lane sequential over 8 columns, then a
            xor-1,2,4,8,16 butterfly), so Q/K/V are bitwise equal to the
            unfused RMSNorm -> FlashInfer RoPE -> .to(fp8) chain. Needs
            head_dim 256 or 512.
        k_eq_v: ``qkv`` is the [q | k] output of a K=V layer's Q/K-only GEMM
            (width num_q_heads*head_dim + num_kv_heads*head_dim). V is
            v_norm of the raw K projection, taken before k_norm or RoPE.
            Separate outputs only (packed_output is refused).

    Returns:
        Separate mode returns contiguous ``(q, k, v)`` tensors in the
        requested output dtype. Packed mode returns ``(qkv, None, None)``
        with contiguous BF16 QKV. In both modes q/k are roped and v is
        norm-only.
    """
    if kv_out_fp8 is None:
        kv_out_fp8 = out_fp8
    if packed_output and (out_fp8 or kv_out_fp8):
        raise ValueError("packed_output requires BF16 output")

    assert qkv.dim() == 2 and qkv.stride(-1) == 1
    assert qkv.dtype == torch.bfloat16
    assert cos_sin.is_contiguous() and cos_sin.dtype == torch.float32
    half = head_dim // 2
    assert cos_sin.dim() == 3 and cos_sin.shape[1] == 2 and cos_sin.shape[2] == half
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    if k_eq_v:
        if packed_output:
            raise ValueError("k_eq_v does not support packed_output")
        assert qkv.shape[-1] == q_size + kv_size
    else:
        assert qkv.shape[-1] == q_size + 2 * kv_size
    if cute_norm_order and head_dim not in (256, 512):
        raise ValueError("cute_norm_order supports head_dim 256 and 512")

    n = qkv.shape[0]
    out_dtype = torch.float8_e4m3fn if out_fp8 else torch.bfloat16
    kv_out_dtype = torch.float8_e4m3fn if kv_out_fp8 else torch.bfloat16
    packed_qkv = None
    if packed_output:
        packed_qkv = torch.empty(
            (n, q_size + 2 * kv_size),
            dtype=out_dtype,
            device=qkv.device,
        )
        q_out, k_out, v_out = packed_qkv.split(
            [q_size, kv_size, kv_size],
            dim=-1,
        )
    else:
        q_out = torch.empty((n, q_size), dtype=out_dtype, device=qkv.device)
        k_out = torch.empty((n, kv_size), dtype=kv_out_dtype, device=qkv.device)
        v_out = torch.empty((n, kv_size), dtype=kv_out_dtype, device=qkv.device)
    if n == 0:
        if packed_output:
            assert packed_qkv is not None
            return packed_qkv, None, None
        return q_out, k_out, v_out

    # ~2k-element tiles keep every thread on wide vectorized accesses without
    # spilling: 16 tokens/program for hd=256, 8 for hd=512 (B200-tuned, see
    # module docstring). head_dim is a power of two on every gated shape, so
    # block_n is too (required by tl.arange).
    block_n = max(1, min(32, 4096 // head_dim))
    grid = (num_q_heads + 2 * num_kv_heads, triton.cdiv(n, block_n))
    _gemma4_qkv_norm_rope_quant_kernel[grid](
        qkv,
        q_out,
        k_out,
        v_out,
        position_ids.view(-1),
        cos_sin,
        q_weight,
        k_weight,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        v_out.stride(0),
        num_q_heads,
        num_kv_heads,
        eps,
        n,
        HD=head_dim,
        HALF=half,
        Q_OUT_FP8=out_fp8,
        KV_OUT_FP8=kv_out_fp8,
        CUTE_NORM_ORDER=cute_norm_order,
        K_EQ_V=k_eq_v,
        BLOCK_N=block_n,
        num_warps=8,
        # Exact mode spells out every FMA; keep LLVM from contracting others.
        enable_fp_fusion=not cute_norm_order,
    )
    if packed_output:
        assert packed_qkv is not None
        return packed_qkv, None, None
    return q_out, k_out, v_out
