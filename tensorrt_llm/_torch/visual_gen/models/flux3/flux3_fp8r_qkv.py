# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 FP8R (BF16 checkpoint-excluded) blocks: fused QKV + per-head QK-RMSNorm + RoPE.

The FP8R blocks' QKV projection is one rowwise-scaled FP8 GEMM (torch._scaled_mm, exactly as
PreparedFP8Sm120.gemm runs it, so V is bitwise unchanged). Without this module, their QK-norm and
RoPE run eagerly / under Inductor in FP32 with several BF16 roundings. Here the CuTe-DSL
qknorm_rope pass (flux3_sm120_qknorm_rope.py, the same pass the NVFP4 CUTLASS QKV path uses)
normalizes and rotates q and k in place in the packed [M, 3*W] GEMM output (copy_v False):
one op, one extra launch, no FP32 intermediates. Q/K are reordered (FP32 math rounded once to
BF16); V and the GEMM are bit-identical. With _FP8R_QKV_PLANAR (default) the pass instead writes
q, k, v as three contiguous [M, W] planes of a fresh buffer (planar=True, v copied in the pass),
so attention gets contiguous q/k/v without three extra copies; values are the same bitwise.

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


def _fp8r_qkv_gemm(activation, weight, activation_scale, weight_scale) -> torch.Tensor:
    return torch._scaled_mm(
        activation,
        weight.t(),
        scale_a=activation_scale.reshape(-1, 1),
        scale_b=weight_scale.reshape(1, -1),
        out_dtype=torch.bfloat16,
    )


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
    planar: bool = False,
) -> torch.Tensor:
    """BF16 [M, 3*W]: rowwise FP8 GEMM, then QK-RMSNorm+RoPE of q/k.

    planar=False: packed q|k|v rows, q/k normed in place in the GEMM output.
    planar=True: a fresh contiguous buffer holding three contiguous [M, W] planes q, k, v
    (``out.view(3, M, W)``); same q/k values, v copied by the pass.
    activation E4M3 [M, K], activation_scale FP32 [M]; weight E4M3 [3*W, K], weight_scale FP32
    [3*W]; cos_sin FP32 [M, 128] indexed by the row's token position (identity).
    """
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
        qknorm_rope,
    )

    from .flux3_primsts import _token_positions

    out = _fp8r_qkv_gemm(activation, weight, activation_scale, weight_scale)
    width = out.shape[1] // 3
    positions = _token_positions(out.shape[0], out.device)
    dst = torch.empty_like(out) if planar else out
    return qknorm_rope(
        out[:, :width],
        out[:, width:2 * width],
        out[:, 2 * width:],
        dst,
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions,
        planar=planar,
    )


@fp8r_qkv_qknorm_rope.register_fake
def _fp8r_qkv_qknorm_rope_fake(
    activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin, planar=False
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
        _FP8R_QKV_PLANAR,
    )
    width = output.shape[-1] // 3
    if _FP8R_QKV_PLANAR:
        q, k, v = output.view(3, output.shape[0], width).unbind(0)
    else:
        q, k, v = output.split(width, -1)
        if _FP8R_QKV_CONTIGUOUS:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return tuple(t.reshape(*original.shape[:-1], width) for t in (q, k, v))


# Strided (both False): attention gets q/k/v as row-stride-3W column views of the packed GEMM
# output; fp8_sm12x (accepts_strided_qkv) reads them in place, so neither the planar v copy nor
# .contiguous() copies run. Bitwise equal to planar/contiguous, ~33 us/call faster than planar
# (branch 62). The served A(T=80)/B(T=160)/A history effect once blamed on this route (job 829)
# is intermittent on every layout, contiguous included (knowledge 847), not a strided-route bug.
# Module-level for A/B in one process.
_FP8R_QKV_CONTIGUOUS = False
# Planar: the qknorm_rope pass writes q, k, v as three contiguous [M, W] planes of a fresh buffer
# (v copied inside the pass), so the three .contiguous() copies above go. Same values (bitwise).
# Takes precedence over _FP8R_QKV_CONTIGUOUS; module-level for A/B in one process.
_FP8R_QKV_PLANAR = False

# Idea 244 stage R (idea 238): the FP8R QKV straight to per-tensor E4M3. Main runs the GEMM, the
# in-place BF16 qknorm_rope pass, then fp8_sm12x's K1 amax + K2 quantize on the strided q/k/v
# views inside visual_gen_fp8_attention_sm120. Here the same GEMM output goes to
# flux3_sm120_qknorm_rope_fp8 on its packed q|k|v column views (bitwise qknorm_rope + K1/K2) and
# the same fmha_v2 call runs through visual_gen_fp8_attention_sm120_prequant. v stays the GEMM's
# BF16 column view (for PAG). TRTLLM_FLUX3_FP8R_QKV_FP8_DIRECT=0 keeps main's path; module-level
# (Dynamo guards on it), read at trace time.
FP8R_QKV_FP8_DIRECT = os.environ.get("TRTLLM_FLUX3_FP8R_QKV_FP8_DIRECT", "1").strip() != "0"


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8r_qkv_rope_fp8", mutates_args=(), device_types="cuda"
)
def fp8r_qkv_rope_fp8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    sm_scale: float,
    keep_v: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(qkv8 [3, M, W] E4M3, scales [8] FP32, gemm [M, 3*W] BF16 packed GEMM output).

    The FP8R QKV GEMM (_fp8r_qkv_gemm, as fp8r_qkv_qknorm_rope runs it), then qknorm_rope_fp8
    on the packed q, k, v column views (row stride 3*W). gemm[:, 2W:] is v; gemm's q/k columns
    are never read again (qknorm_rope_fp8 may drop them from L2, contents undefined), nor v's
    when keep_v is False.
    """
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope_fp8 import (
        qknorm_rope_fp8,
    )

    from .flux3_primsts import _token_positions

    out = _fp8r_qkv_gemm(activation, weight, activation_scale, weight_scale)
    width = out.shape[1] // 3
    qkv8, scales = qknorm_rope_fp8(
        out[:, :width],
        out[:, width:2 * width],
        out[:, 2 * width:],
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        _token_positions(out.shape[0], out.device),
        sm_scale,
        discard_qk=True,
        keep_v=keep_v,
    )
    return qkv8, scales, out


@fp8r_qkv_rope_fp8.register_fake
def _fp8r_qkv_rope_fp8_fake(
    activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin, sm_scale,
    keep_v=True,
):
    rows, columns = activation.shape[0], weight.shape[0]
    return (
        activation.new_empty((3, rows, columns // 3), dtype=torch.float8_e4m3fn),
        activation.new_empty((8,), dtype=torch.float32),
        activation.new_empty((rows, columns), dtype=torch.bfloat16),
    )


def sm120_fp8r_qkv_fp8_attention(
    attn: nn.Module,
    x,
    freqs: tuple[torch.Tensor, ...],
    full_sequence: bool,
    v_live: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(attention output [B, S, H*D] BF16, v [B, S, W] BF16 view of the GEMM output) or None.

    Covers exactly the calls where main runs primsts.qkv_forward's _sm120_fp8r_qkv_forward
    (packed, non-planar) and then one FlashInferAttention FP8 (non-causal, no LSE) call on the
    whole [B, S] q/k/v with S >= the backend's FP8 minimum; anything else returns None.
    """
    if not FP8R_QKV_FP8_DIRECT or not full_sequence or _FP8R_QKV_PLANAR or _FP8R_QKV_CONTIGUOUS:
        return None
    if getattr(attn, "_flux3_primsts_nvfp4_qkv", None) is not None:
        return None
    if not getattr(attn, "_flux3_sm120_fp8r_qknorm_rope", False):
        return None
    if not isinstance(x, (torch.Tensor, Fp8RowwiseQuantizedTensor)):
        return None
    from .primsts import _fp8_direct_backend, original_input

    backend = _fp8_direct_backend(attn)
    if backend is None:
        return None
    adapter = attn._flux3_sm120_fp8_qkv
    original = original_input(x)
    if original.dim() != 3:
        return None
    batch, seq_len = original.shape[0], original.shape[1]
    cos, sin = freqs[0], freqs[1]
    if not adapter.can_run(original) or cos.numel() != original.numel() // original.shape[-1] * 128:
        return None
    if sin.shape != cos.shape:
        return None
    if seq_len < backend._fp8_min_seq_len or seq_len <= 0:
        return None
    heads = adapter.weight.shape[0] // 3 // 128
    if heads != backend.num_heads:
        return None
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (
        visual_gen_fp8_attention_sm120_prequant,
    )

    data, scales = adapter.activation(x)
    if len(freqs) > 2:
        cos_sin = freqs[2]
    else:
        cos_sin = torch.cat((cos.reshape(-1, 128)[:, ::2], sin.reshape(-1, 128)[:, ::2]), dim=-1)
    qkv8, fp8_scales, gemm = fp8r_qkv_rope_fp8(
        data,
        adapter.weight,
        scales,
        adapter.scale,
        attn.norm_q.weight.contiguous(),
        attn.norm_k.weight.contiguous(),
        cos_sin.float().contiguous(),
        backend.scale,
        bool(v_live),
    )
    width = gemm.shape[-1] // 3
    v = gemm[:, 2 * width:].reshape(batch, seq_len, width)
    output = visual_gen_fp8_attention_sm120_prequant(qkv8, fp8_scales, batch, heads)
    return output.flatten(2), v
