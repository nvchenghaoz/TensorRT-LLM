# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py
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
"""Fused Qwen3.5 full-attention preprocessing and output gating.

The projection produces interleaved ``[q0, gate0, q1, gate1, ..., K, V]``.
The preprocessing kernel reads that layout directly and writes packed
``[Q, K, V]`` while applying per-head Gemma RMSNorm and NeoX RoPE to Q/K.
Gate deinterleave is fused into the same pass.  Keeping the checkpoint layout
unchanged avoids coupling the optimization to weight loading, quantization,
LoRA, or attention-backend fallback behavior.
"""

import functools
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ...modules.fused_silu_mul_nvfp4 import (_cvt_e2m1x8, _cvt_e4m3_bits,
                                             _gdc_wait_with_memory_clobber,
                                             _mul_rn, _rcp_approx_ftz,
                                             _sf_offset)


@functools.lru_cache(maxsize=None)
def _sm_at_least_90(device: torch.device) -> bool:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor >= 90


@triton.jit
def _fused_qkv_gemma_rmsnorm_rope_gate_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    num_tokens,
    positions_stride_section,
    max_positions,
    qkv_stride_token,
    qkv_out_stride_token,
    gate_out_stride_token,
    cos_sin_stride_position,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    head_block: tl.constexpr,
    rotary_block: tl.constexpr,
    eps: tl.constexpr,
    output_fp16: tl.constexpr,
    has_pass_through: tl.constexpr,
    use_mrope: tl.constexpr,
    mrope_section1: tl.constexpr,
    mrope_section2: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    output_dtype = tl.float16 if output_fp16 else tl.bfloat16

    if head < num_q_heads + num_kv_heads:
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)
        if is_k:
            input_base = qkv_ptr + token * qkv_stride_token + 2 * q_size + local_head * head_dim
            output_base = (
                qkv_out_ptr + token * qkv_out_stride_token + q_size + local_head * head_dim
            )
            weight_ptr = k_weight_ptr
        else:
            input_base = qkv_ptr + token * qkv_stride_token + local_head * 2 * head_dim
            output_base = qkv_out_ptr + token * qkv_out_stride_token + local_head * head_dim
            weight_ptr = q_weight_ptr

        head_offsets = tl.arange(0, head_block)
        head_mask = head_offsets < head_dim
        x = tl.load(input_base + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / head_dim + eps)
        normalized = (x * inverse_rms * (weight + 1.0)).to(output_dtype).to(tl.float32)

        if has_pass_through:
            pass_mask = head_mask & (head_offsets >= rotary_dim)
            tl.store(output_base + head_offsets, normalized, mask=pass_mask)

        rotary_offsets = tl.arange(0, rotary_block)
        rotary_mask = rotary_offsets < half_rotary
        x_first = tl.load(input_base + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        x_second = tl.load(
            input_base + half_rotary + rotary_offsets, mask=rotary_mask, other=0.0
        ).to(tl.float32)
        weight_first = tl.load(weight_ptr + rotary_offsets, mask=rotary_mask, other=0.0).to(
            tl.float32
        )
        weight_second = tl.load(
            weight_ptr + half_rotary + rotary_offsets, mask=rotary_mask, other=0.0
        ).to(tl.float32)
        x_first = (x_first * inverse_rms * (weight_first + 1.0)).to(output_dtype).to(tl.float32)
        x_second = (x_second * inverse_rms * (weight_second + 1.0)).to(output_dtype).to(tl.float32)

        if use_mrope:
            section = tl.where(
                (rotary_offsets % 3 == 1) & (rotary_offsets < mrope_section1 * 3),
                1,
                tl.where(
                    (rotary_offsets % 3 == 2) & (rotary_offsets < mrope_section2 * 3),
                    2,
                    0,
                ),
            )
            position = tl.load(
                positions_ptr + section * positions_stride_section + token,
                mask=rotary_mask,
                other=0,
            ).to(tl.int64)
        else:
            position = tl.load(positions_ptr + token).to(tl.int64)
        position_is_valid = (position >= 0) & (position < max_positions)
        tl.device_assert(position_is_valid, "position is outside the RoPE table")
        safe_position = tl.where(position_is_valid, position, 0)
        cos_sin_base = cos_sin_ptr + safe_position * cos_sin_stride_position
        cos = tl.load(cos_sin_base + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        sin = tl.load(cos_sin_base + half_rotary + rotary_offsets, mask=rotary_mask, other=0.0).to(
            tl.float32
        )
        cos = tl.where(position_is_valid, cos, float("nan"))
        sin = tl.where(position_is_valid, sin, float("nan"))
        tl.store(
            output_base + rotary_offsets,
            x_first * cos - x_second * sin,
            mask=rotary_mask,
        )
        tl.store(
            output_base + half_rotary + rotary_offsets,
            x_second * cos + x_first * sin,
            mask=rotary_mask,
        )

        if not is_k:
            gate_input = input_base + head_dim
            gate_output = gate_out_ptr + token * gate_out_stride_token + local_head * head_dim
            gate = tl.load(gate_input + head_offsets, mask=head_mask, other=0.0)
            tl.store(gate_output + head_offsets, gate, mask=head_mask)
    else:
        local_head = head - num_q_heads - num_kv_heads
        head_offsets = tl.arange(0, head_block)
        head_mask = head_offsets < head_dim
        input_base = (
            qkv_ptr + token * qkv_stride_token + 2 * q_size + kv_size + local_head * head_dim
        )
        output_base = (
            qkv_out_ptr + token * qkv_out_stride_token + q_size + kv_size + local_head * head_dim
        )
        value = tl.load(input_base + head_offsets, mask=head_mask, other=0.0)
        tl.store(output_base + head_offsets, value, mask=head_mask)


@torch.library.custom_op(
    "trtllm::fused_qkv_gemma_rmsnorm_rope_gate",
    mutates_args=(),
)
def _fused_qkv_gemma_rmsnorm_rope_gate(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    use_mrope: bool,
    mrope_section1: int,
    mrope_section2: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert qkv.dim() == 2 and qkv.stride(-1) == 1
    assert qkv.dtype in (torch.bfloat16, torch.float16)
    assert q_weight.shape == (head_dim,) and k_weight.shape == (head_dim,)
    assert q_weight.device == qkv.device and k_weight.device == qkv.device
    assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
    assert cos_sin.device == qkv.device and positions.device == qkv.device
    assert rotary_dim > 0 and rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert cos_sin.shape == (cos_sin.shape[0], 2, rotary_dim // 2)
    assert cos_sin.shape[0] > 0

    num_tokens = qkv.shape[0]
    if use_mrope:
        assert mrope_section1 > 0 and mrope_section2 > 0
        assert positions.numel() == 3 * num_tokens
        # reshape keeps a [3, 1, max_tokens] buffer slice a strided view; the
        # kernel reads section rows through positions_stride_section.
        positions = positions.reshape(3, num_tokens)
        if num_tokens > 1 and positions.stride(-1) != 1:
            positions = positions.contiguous()
        positions_stride_section = positions.stride(0)
    else:
        assert positions.numel() == num_tokens
        positions = positions.reshape(-1)
        assert positions.is_contiguous()
        positions_stride_section = 0
    assert positions.dtype in (torch.int32, torch.int64)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    assert qkv.shape[1] == 2 * q_size + 2 * kv_size

    qkv_out = torch.empty((num_tokens, q_size + 2 * kv_size), dtype=qkv.dtype, device=qkv.device)
    gate_out = torch.empty((num_tokens, num_q_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    if num_tokens == 0:
        return qkv_out, gate_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rotary_block = triton.next_power_of_2(half_rotary)
    grid = (num_tokens, num_q_heads + 2 * num_kv_heads)
    _fused_qkv_gemma_rmsnorm_rope_gate_kernel[grid](
        qkv,
        qkv_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        num_tokens,
        positions_stride_section,
        cos_sin.shape[0],
        qkv.stride(0),
        qkv_out.stride(0),
        gate_out.stride(0),
        cos_sin.stride(0),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rotary=half_rotary,
        head_block=head_block,
        rotary_block=rotary_block,
        eps=eps,
        output_fp16=qkv.dtype == torch.float16,
        has_pass_through=rotary_dim < head_dim,
        use_mrope=use_mrope,
        mrope_section1=mrope_section1,
        mrope_section2=mrope_section2,
        num_warps=4,
    )
    return qkv_out, gate_out


@_fused_qkv_gemma_rmsnorm_rope_gate.register_fake
def _(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    use_mrope: bool,
    mrope_section1: int,
    mrope_section2: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    qkv_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    return (
        qkv.new_empty((num_tokens, qkv_width)),
        qkv.new_empty((num_tokens, num_q_heads, head_dim)),
    )


def fused_qkv_gemma_rmsnorm_rope_gate(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: Optional[Tuple[int, int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare packed QKV and gate with one Triton kernel.

    Args:
        qkv: ``[num_tokens, 2 * q_size + 2 * kv_size]`` BF16/FP16 projection
            output in per-head interleaved Q/G layout.
        q_weight: ``[head_dim]`` raw Gemma RMSNorm Q weights.
        k_weight: ``[head_dim]`` raw Gemma RMSNorm K weights.
        cos_sin: ``[max_positions, 2, rotary_dim // 2]`` FP32 NeoX table.
        positions: Flattenable ``[num_tokens]`` plain-RoPE positions, or
            ``[3, ..., num_tokens]`` interleaved-MRoPE positions.
        eps: RMSNorm epsilon.
        num_q_heads: Local query-head count.
        num_kv_heads: Local key/value-head count.
        head_dim: Per-head Q/K/V dimension.
        rotary_dim: Prefix dimension receiving NeoX RoPE.
        mrope_section: Temporal/height/width rotary-half dimensions. ``None``
            selects plain RoPE; a tuple selects Qwen-style interleaved MRoPE.

    Returns:
        Packed QKV ``[num_tokens, q_size + 2 * kv_size]`` and gate
        ``[num_tokens, num_q_heads, head_dim]``.
    """
    use_mrope = mrope_section is not None
    if use_mrope:
        assert len(mrope_section) == 3
        assert sum(mrope_section) == rotary_dim // 2
        mrope_section1 = mrope_section[1]
        mrope_section2 = mrope_section[2]
    else:
        mrope_section1 = 0
        mrope_section2 = 0
    return _fused_qkv_gemma_rmsnorm_rope_gate(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        use_mrope,
        mrope_section1,
        mrope_section2,
    )


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attention_ptr,
    gate_ptr,
    output_stride_token,
    attention_stride_token,
    gate_stride_token,
    gate_stride_head,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < hidden_size
    head = offsets // head_dim
    dim = offsets - head * head_dim

    attention = tl.load(
        attention_ptr + token * attention_stride_token + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + token * gate_stride_token + head * gate_stride_head + dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + token * output_stride_token + offsets,
        attention * tl.sigmoid(gate),
        mask=mask,
    )


@torch.library.custom_op(
    "trtllm::fused_sigmoid_mul_inplace",
    mutates_args=("attention_output",),
)
def fused_sigmoid_mul_inplace(
    attention_output: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """Apply ``attention_output *= sigmoid(gate)`` with a Triton kernel."""
    assert attention_output.dim() == 2 and attention_output.stride(-1) == 1
    num_tokens, hidden_size = attention_output.shape
    if gate.dim() == 3:
        assert gate.shape[0] == num_tokens
        assert gate.shape[1] * gate.shape[2] == hidden_size
        assert gate.stride(-1) == 1
        head_dim = gate.shape[2]
        gate_stride_token = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        assert gate.dim() == 2 and gate.shape == attention_output.shape
        assert gate.stride(-1) == 1
        head_dim = hidden_size
        gate_stride_token = gate.stride(0)
        gate_stride_head = hidden_size

    if num_tokens == 0:
        return

    max_block_size = 1024 if num_tokens < 1024 else 2048
    block_size = min(triton.next_power_of_2(hidden_size), max_block_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    _fused_sigmoid_mul_kernel[grid](
        attention_output,
        attention_output,
        gate,
        attention_output.stride(0),
        attention_output.stride(0),
        gate_stride_token,
        gate_stride_head,
        hidden_size=hidden_size,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )


@triton.jit
def _fused_sigmoid_mul_nvfp4_kernel(
    fp4_ptr,  # int32 view of uint8 [num_tokens, hidden_size / 2]
    sf_ptr,  # uint8 swizzled SF, ceil128(num_tokens) * NUM_K_TILES * 4
    attention_ptr,
    gate_ptr,
    scale_ptr,  # 1-element fp32 global scale (6 / input_scale)
    num_tokens,
    attention_stride_token,
    gate_stride_token,
    gate_stride_head,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """_fused_sigmoid_mul_kernel's expression, rounded to BF16 as stored,
    then fp4_quantize(r, scale, 16) (fused_silu_mul_nvfp4's quantize tail).
    Program row num_tokens zeroes the SF padding rows. With USE_PDL all
    accesses stay behind griddepcontrol.wait and dependents are released
    after the stores."""
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    i = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    NSF: tl.constexpr = BLOCK_SIZE // 16
    sf_cols = j * NSF + tl.arange(0, NSF)
    num_sf_cols: tl.constexpr = hidden_size // 16
    if i < num_tokens:
        offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        head = offsets // head_dim
        dim = offsets - head * head_dim
        attention = tl.load(
            attention_ptr + i * attention_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + i * gate_stride_token + head * gate_stride_head + dim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        r = (attention * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)

        h2 = tl.reshape(r, [NSF, 16])
        vmax = tl.max(tl.abs(h2), axis=1)
        s = tl.load(scale_ptr)
        s_v = tl.full([NSF], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([NSF], 6.0, tl.float32))
        sf_val = _mul_rn(s_v, _mul_rn(vmax, r6))
        sf_u8 = _cvt_e4m3_bits(sf_val).to(tl.uint8)
        sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        out_scale = tl.where(
            vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, _rcp_approx_ftz(s_v))),
            0.0)
        y = _mul_rn(h2, tl.broadcast_to(out_scale[:, None], [NSF, 16]))
        y8 = tl.reshape(y, [BLOCK_SIZE // 8, 2, 2, 2])
        p, q = tl.split(y8)
        p0, p1 = tl.split(p)
        q0, q1 = tl.split(q)
        p00, p01 = tl.split(p0)
        p10, p11 = tl.split(p1)
        q00, q01 = tl.split(q0)
        q10, q11 = tl.split(q1)
        packed = _cvt_e2m1x8(p00, q00, p10, q10, p01, q01, p11, q11)
        w_offsets = j * (BLOCK_SIZE // 8) + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(fp4_ptr + i * (hidden_size // 8) + w_offsets,
                 packed,
                 mask=w_offsets < hidden_size // 8)
        sf_off = _sf_offset(i, sf_cols, NUM_K_TILES)
        tl.store(sf_ptr + sf_off, sf_u8, mask=sf_cols < num_sf_cols)
    else:
        m0 = (num_tokens // 128) * 128
        rows = m0 + tl.arange(0, 128).to(tl.int64)
        off = _sf_offset(rows[:, None], sf_cols[None, :], NUM_K_TILES)
        pmask = (rows >= num_tokens)[:, None] & (sf_cols
                                                 < num_sf_cols)[None, :]
        tl.store(sf_ptr + off, tl.zeros([128, NSF], tl.uint8), mask=pmask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


_SIGMOID_MUL_NVFP4_BLOCK = 512
# TRTLLM_ATTN_GATE_NVFP4_PDL=0 launches the fused gate kernel without PDL.
_SIGMOID_MUL_NVFP4_PDL = (
    os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    and os.environ.get("TRTLLM_ATTN_GATE_NVFP4_PDL", "1") != "0")


def fused_sigmoid_mul_nvfp4_eligible(attention_output: torch.Tensor,
                                     gate: torch.Tensor) -> bool:
    """Shapes the NVFP4 sigmoid-gate kernel takes (same contract as
    fused_sigmoid_mul_inplace, plus BF16 and whole 64-column SF tiles)."""
    if (attention_output.dim() != 2 or attention_output.stride(-1) != 1
            or attention_output.dtype != torch.bfloat16
            or gate.dtype != torch.bfloat16 or gate.stride(-1) != 1):
        return False
    hidden_size = attention_output.shape[1]
    if hidden_size % 64 != 0:
        return False
    if gate.dim() == 3:
        return (gate.shape[0] == attention_output.shape[0]
                and gate.shape[1] * gate.shape[2] == hidden_size)
    return gate.dim() == 2 and gate.shape == attention_output.shape


@torch.library.custom_op("trtllm::fused_sigmoid_mul_nvfp4", mutates_args=())
def fused_sigmoid_mul_nvfp4(
        attention_output: torch.Tensor, gate: torch.Tensor,
        global_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fused_sigmoid_mul_inplace then fp4_quantize(r, global_scale, 16):
    FP4 uint8 [num_tokens, hidden / 2] and swizzled uint8 SF, bitwise equal
    to the two-kernel path. attention_output is not modified."""
    assert fused_sigmoid_mul_nvfp4_eligible(attention_output, gate)
    num_tokens, hidden_size = attention_output.shape
    if gate.dim() == 3:
        head_dim = gate.shape[2]
        gate_stride_token = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        head_dim = hidden_size
        gate_stride_token = gate.stride(0)
        gate_stride_head = hidden_size
    num_k_tiles = (hidden_size // 16 + 3) // 4
    m_pad = (num_tokens + 127) // 128 * 128
    fp4 = torch.empty((num_tokens, hidden_size // 2),
                      dtype=torch.uint8,
                      device=attention_output.device)
    sf = torch.empty((m_pad * num_k_tiles * 4, ),
                     dtype=torch.uint8,
                     device=attention_output.device)
    grid_rows = num_tokens + (1 if num_tokens % 128 != 0 else 0)
    if grid_rows == 0:
        return fp4, sf
    grid = (grid_rows, triton.cdiv(hidden_size, _SIGMOID_MUL_NVFP4_BLOCK))
    use_pdl = _SIGMOID_MUL_NVFP4_PDL and _sm_at_least_90(
        attention_output.device)
    _fused_sigmoid_mul_nvfp4_kernel[grid](
        fp4.view(torch.int32),
        sf,
        attention_output,
        gate,
        global_scale,
        num_tokens,
        attention_output.stride(0),
        gate_stride_token,
        gate_stride_head,
        hidden_size=hidden_size,
        head_dim=head_dim,
        BLOCK_SIZE=_SIGMOID_MUL_NVFP4_BLOCK,
        NUM_K_TILES=num_k_tiles,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )
    return fp4, sf


@fused_sigmoid_mul_nvfp4.register_fake
def _(attention_output: torch.Tensor, gate: torch.Tensor,
      global_scale: torch.Tensor):
    num_tokens, hidden_size = attention_output.shape
    num_k_tiles = (hidden_size // 16 + 3) // 4
    m_pad = (num_tokens + 127) // 128 * 128
    return (attention_output.new_empty((num_tokens, hidden_size // 2),
                                       dtype=torch.uint8),
            attention_output.new_empty((m_pad * num_k_tiles * 4, ),
                                       dtype=torch.uint8))
