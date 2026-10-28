# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 FP8R (BF16 checkpoint-excluded) blocks: fused QKV + per-head QK-RMSNorm + RoPE.

The FP8R blocks' QKV projection is one rowwise-scaled FP8 GEMM (torch._scaled_mm, exactly as
PreparedFP8Sm120.gemm runs it, so V is bitwise unchanged). Without this module, their QK-norm and
RoPE run eagerly / under Inductor in FP32 with several BF16 roundings. Here the CuTe-DSL
qknorm_rope pass (flux3_sm120_qknorm_rope.py, the same pass the NVFP4 CUTLASS QKV path uses)
normalizes and rotates q and k in place in the packed [M, 3*W] GEMM output (copy_v False):
one op, one extra launch, no FP32 intermediates. Q/K are reordered (FP32 math rounded once to
BF16); V and the GEMM are bit-identical.

primsts.prepare_sm120_fp8_excluded_block sets attn._flux3_sm120_fp8r_qknorm_rope at prepare
time; primsts.qkv_forward returns _sm120_fp8r_qkv_forward(...) when the flag is set.
"""

import os

import torch
from torch import nn

from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor


def _sm120_fp8r_qknorm_rope_enabled() -> bool:
    """TRTLLM_FLUX3_SM120_FP8R_QKNORM_ROPE=0 keeps the eager QK-norm/RoPE (read at prepare)."""
    return os.environ.get("TRTLLM_FLUX3_SM120_FP8R_QKNORM_ROPE", "1") != "0"


def _sm120_fp8r_qknorm_rope_eligible(attn: nn.Module) -> bool:
    """Static prepare-time check for an FP8R attention whose fused PreparedFP8Sm120 QKV is set."""
    from .primsts import _local_projection, _qkv_epilogue_eligible

    if not _sm120_fp8r_qknorm_rope_enabled():
        return False
    fused = getattr(attn, "_flux3_sm120_fp8_qkv", None)
    if fused is None or not _qkv_epilogue_eligible(attn):
        return False
    projections = [getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")]
    if any(layer is None for layer in projections):
        return False
    heads = attn.local_num_attention_heads
    for layer in projections:
        mapping = getattr(layer, "mapping", None)
        if mapping is not None and mapping.tp_size != 1:
            return False
        if not _local_projection(layer):
            return False
    return heads % 4 == 0 and fused.weight.shape[0] == 3 * heads * 128 and not fused.swiglu


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8r_qkv_qknorm_rope", mutates_args=(), device_types="cuda"
)
def fp8r_qkv_qknorm_rope(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
) -> torch.Tensor:
    """BF16 [M, 3*W] packed q|k|v rows: rowwise FP8 GEMM, then QK-RMSNorm+RoPE of q/k in place.

    activation E4M3 [M, K], activation_scale FP32 [M]; weight E4M3 [3*W, K], weight_scale FP32
    [3*W]; cos_sin FP32 [M, 128] indexed by the row's token position (identity).
    """
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
        qknorm_rope,
    )

    from .flux3_primsts import _token_positions

    out = torch._scaled_mm(
        activation,
        weight.t(),
        scale_a=activation_scale.reshape(-1, 1),
        scale_b=weight_scale.reshape(1, -1),
        out_dtype=torch.bfloat16,
    )
    width = out.shape[1] // 3
    positions = _token_positions(out.shape[0], out.device)
    return qknorm_rope(
        out[:, :width],
        out[:, width:2 * width],
        out[:, 2 * width:],
        out,
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions,
    )


@fp8r_qkv_qknorm_rope.register_fake
def _fp8r_qkv_qknorm_rope_fake(
    activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin
):
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def _sm120_fp8r_qkv_forward(
    attn: nn.Module,
    x,
    original: torch.Tensor,
    freqs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Q, K (normed + rotated) and V for an FP8R attention, each [..., W] (views of one packed
    [M, 3*W] buffer), or None for inputs the op does not cover (caller runs the eager path).
    x is the BF16 input or the block's shared Fp8RowwiseQuantizedTensor."""
    adapter = attn._flux3_sm120_fp8_qkv
    if not isinstance(x, (torch.Tensor, Fp8RowwiseQuantizedTensor)):
        return None
    cos, sin = freqs[0], freqs[1]
    if not adapter.can_run(original) or cos.numel() != original.numel() // original.shape[-1] * 128:
        return None
    if sin.shape != cos.shape:
        return None
    data, scales = adapter.activation(x)
    if len(freqs) > 2:
        cos_sin = freqs[2]
    else:
        cos_sin = torch.cat((cos.reshape(-1, 128)[:, ::2], sin.reshape(-1, 128)[:, ::2]), dim=-1)
    output = fp8r_qkv_qknorm_rope(
        data,
        adapter.weight,
        scales,
        adapter.scale,
        attn.norm_q.weight.contiguous(),
        attn.norm_k.weight.contiguous(),
        cos_sin.float().contiguous(),
    )
    width = output.shape[-1] // 3
    q, k, v = output.split(width, -1)
    return tuple(t.reshape(*original.shape[:-1], width) for t in (q, k, v))
