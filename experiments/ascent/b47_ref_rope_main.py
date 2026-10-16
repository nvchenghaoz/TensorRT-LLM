# Verbatim copy of main 2c795b56's _fused_qkv_gemma_rmsnorm_rope_gate kernel
# (tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py) and its
# launcher, used as the bitwise reference for branch 47.
import torch
import triton
import triton.language as tl


@triton.jit
def _main_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    num_tokens,
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
                positions_ptr + section * num_tokens + token,
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


def main_rope_gate(qkv, q_weight, k_weight, cos_sin, positions, eps, num_q_heads,
                   num_kv_heads, head_dim, rotary_dim, mrope_section=None):
    """Main's caller (qk_norm_attention.py: positions.contiguous()) + op."""
    use_mrope = mrope_section is not None
    m1 = mrope_section[1] if use_mrope else 0
    m2 = mrope_section[2] if use_mrope else 0
    positions = positions.contiguous()
    num_tokens = qkv.shape[0]
    if use_mrope:
        positions = positions.reshape(3, num_tokens)
    else:
        positions = positions.reshape(-1)
    assert positions.is_contiguous()
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_out = torch.empty((num_tokens, q_size + 2 * kv_size), dtype=qkv.dtype, device=qkv.device)
    gate_out = torch.empty((num_tokens, num_q_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    if num_tokens == 0:
        return qkv_out, gate_out
    half_rotary = rotary_dim // 2
    grid = (num_tokens, num_q_heads + 2 * num_kv_heads)
    _main_kernel[grid](
        qkv, qkv_out, gate_out, q_weight, k_weight, cos_sin, positions,
        num_tokens, cos_sin.shape[0], qkv.stride(0), qkv_out.stride(0),
        gate_out.stride(0), cos_sin.stride(0),
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        rotary_dim=rotary_dim, half_rotary=half_rotary,
        head_block=triton.next_power_of_2(head_dim),
        rotary_block=triton.next_power_of_2(half_rotary), eps=eps,
        output_fp16=qkv.dtype == torch.float16,
        has_pass_through=rotary_dim < head_dim, use_mrope=use_mrope,
        mrope_section1=m1, mrope_section2=m2, num_warps=4,
    )
    return qkv_out, gate_out


def main_embedding(input_, weight, lo, hi):
    """Main's TP1 pre_comm_embedding_ops body (9 kernels)."""
    vocab_mask = (input_ >= lo) & (input_ < hi)
    valid_offset = lo * vocab_mask
    masked = vocab_mask * (input_ - valid_offset)
    output = torch.nn.functional.embedding(masked, weight)
    output.masked_fill_(~vocab_mask.unsqueeze(-1), 0)
    return output
