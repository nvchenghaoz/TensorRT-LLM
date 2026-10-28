# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 JointSingleSeq Transformer (Native TRT-LLM).

FLUX.3 image-small is a joint image/text DiT:

- Per-modality "early" blocks (Flux3ModeBlock) are self-attentive/siloed per
  modality; cross-modality fusion happens only in the joint
  Flux3SingleStreamBlock stage (image + reference images + text per sample).
- Modulation is stage-scoped: one Flux3Modulation per modality per stage,
  reused by every block in that stage.
- Timesteps are per-token/per-modality (conditioning tensors may sit at a
  different noise level than the target), not a single scalar.
- Ordinary checkpoints use CFG in the sampling loop. Flow-map checkpoints
  condition every modality on the target time via ref_time_in and run without CFG.
- 4-axis RoPE built as 2x2 rotation matrices (flux3_rope), bridged to the
  (freqs_cos, freqs_sin) the shared/fused attention path expects via
  flux3_pe_to_cos_sin.
"""

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import (
    FP8RowwiseLinearMethod,
    Fp8RowwiseQuantizedTensor,
    Linear,
    NVFP4LinearMethod,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode, apply_rotary_emb
from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import FluxJointAttnMLPProj
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import Bounds, Layout, bounds
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import (
    Flux3FP8RowwiseLinearMethod,
    quantize_fp8r,
)
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import (
    NVFP4Activation,
    convert_excluded_blocks_to_nvfp4,
    nvfp4_mlp_forward,
    original_input,
    prepare_fp8_block,
    prepare_nvfp4_block,
    prepare_sm120_fp8_excluded_block,
    prepare_sm120_tail_pair,
    rope_cos_sin_table,
    separate_qkv_forward,
    shared_nvfp4_input,
    sm120_fp8_mlp_forward,
    sm120_fp8_shared_input,
    sm120_side_gate_up_args,
    sm120_tail_pair_forward,
    verify_nvfp4_coverage,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import fp8_attention_side_gate_up
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import (
    linear_forward as _primsts_linear_forward,
)
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import qkv_forward as _primsts_qkv_forward
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import (
    qkv_fp8_attention_forward as _primsts_qkv_fp8_attention,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_adaln_quant import (
    adaln_nvfp4_input,
    prepare_adaln_nvfp4,
)
from tensorrt_llm._torch.visual_gen.models.flux3.utils_flux3 import (
    Flux3PerTokenAdaLN,
    Flux3PerTokenAdaLNRuntime,
    _pertoken_adaln_arch_ok,
)
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

DEFAULT_IN_CHANNELS: Dict[str, int] = {
    "image": 128,
    "image_cond": 128,
}

DEFAULT_SEQUENCE: Dict[str, str] = {
    "x_image": "image",
    "x_image_cond": "image_cond",
}

# Flux3Attention._attend: a single full-range segment calls the backend directly
# instead of filling a zero buffer and slice-copying the result into it (same
# backend inputs, same output values). Module-level so an A/B can toggle it.
_ATTEND_SINGLE_SEGMENT_FAST_PATH = True


# =============================================================================
# RoPE (FLUX.3-local: 2x2 rotation-matrix convention)
# =============================================================================


def flux3_rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def flux3_pe_to_cos_sin(pe: Tensor) -> Tuple[Tensor, Tensor]:
    """Bridge FLUX.3's 2x2 rotation-matrix pe (..., head_dim/2, 2, 2) into the
    (freqs_cos, freqs_sin) the shared RoPE expects: cos = pe[..., 0, 0],
    sin = pe[..., 1, 0], each repeat-interleaved to head_dim."""
    cos = pe[..., 0, 0]
    sin = pe[..., 1, 0]
    freqs_cos = cos.repeat_interleave(2, dim=-1)
    freqs_sin = sin.repeat_interleave(2, dim=-1)
    return freqs_cos, freqs_sin


def _pe_to_block_freqs(pe: Tensor) -> Tuple[Tensor, Tensor]:
    """pe (B, 1, S, head_dim/2, 2, 2) -> (cos, sin) each (B, S, 1, head_dim),
    broadcastable against the blocks' [B, S, H, head_dim] q/k for the shared
    apply_rotary_emb."""
    cos, sin = flux3_pe_to_cos_sin(pe)
    return cos.transpose(1, 2).contiguous(), sin.transpose(1, 2).contiguous()


# Build the fused QK-norm+RoPE FP32 [rows, 128] cos/sin table once per request in
# _packed_positions and hand it to every block as freqs[2], instead of every fused QKV
# re-concatenating it from cos/sin. Same values (a copy), so outputs are unchanged.
FLUX3_ROPE_TABLE = True


def _block_freqs(prepared: dict, prefix: str) -> Tuple[Tensor, ...]:
    """(cos, sin), plus the prebuilt cos_sin table when _packed_positions made one."""
    cos, sin = prepared[prefix + "cos"], prepared[prefix + "sin"]
    table = prepared.get(prefix + "cos_sin")
    return (cos, sin) if table is None else (cos, sin, table)


def apply_flux3_pag(
    attn: Tensor,
    v: Tensor,
    pag_strength: float,
    sample_bounds: Bounds | None = None,
    pag_samples: tuple[int, ...] = (0,),
) -> Tensor:
    if pag_strength <= 0.0:
        return attn
    if sample_bounds is not None:
        attn = attn.clone()
        values = v.transpose(1, 2).flatten(2)
        for sample in pag_samples:
            start, end = sample_bounds[sample]
            attn[:, start:end] = (1 - pag_strength) * attn[:, start:end] + pag_strength * values[
                :, start:end
            ]
        return attn
    B = attn.shape[0]
    half = B // 2 if B > 1 else B  # B=1 in CFG-parallel uncond rank: treat whole batch as uncond
    b, h, n, d = v.shape
    v_passthrough = v[:half].transpose(1, 2).reshape(half, n, h * d)
    attn = attn.clone()
    if pag_strength >= 1.0:
        attn[:half] = v_passthrough
    else:
        attn[:half] = (1 - pag_strength) * attn[:half] + pag_strength * v_passthrough
    return attn


class Flux3PosEmbed(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [
                flux3_rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(len(self.axes_dim))
            ],
            dim=-3,
        )
        return emb.unsqueeze(1)


def flux3_timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(out_dtype if out_dtype is not None else t.dtype)
    return embedding


# =============================================================================
# Checkpoint key remap
# =============================================================================


# Shared with flux3_lora.py so both encode gate_up_proj's layout once.
GATE_UP_ROLES: Tuple[str, str] = ("gate", "up")


def _remap_checkpoint_keys(weights: dict, fuse_qkv: bool = True) -> dict:
    """Remap checkpoint block-attention keys onto our shared-Attention layout
    (q/k/v_proj -> attn.*, norm.{query,key}_norm.scale -> attn.norm_{q,k}.weight).
    fuse_qkv picks the target Linear name: qkv_proj's params_map sub-keys
    (fused) vs Attention's own to_q/to_k/to_v (SEPARATE_QKV)."""
    q_name, k_name, v_name = (
        ("q_proj", "k_proj", "v_proj") if fuse_qkv else ("to_q", "to_k", "to_v")
    )
    remapped = {}
    for key, value in weights.items():
        new_key = key
        new_key = new_key.replace(".norm.query_norm.scale", ".attn.norm_q.weight")
        new_key = new_key.replace(".norm.key_norm.scale", ".attn.norm_k.weight")
        new_key = new_key.replace(".q_proj.", f".attn.{q_name}.")
        new_key = new_key.replace(".k_proj.", f".attn.{k_name}.")
        new_key = new_key.replace(".v_proj.", f".attn.{v_name}.")
        new_key = new_key.replace(".mlp_out.", ".ff.down_proj.")
        remapped[new_key] = value
    return remapped


def _use_fused_qkv(config: DiffusionModelConfig) -> bool:
    if config.extra_attrs.get("force_unfused_lora_topology"):
        # LoRA into a TP-sharded fused qkv_proj isn't supported yet; TP==1 stays fused.
        tp_size = config.mapping.tp_size if config.mapping else 1
        if tp_size > 1:
            return False
    quant_algo = config.quant_config.quant_algo if config.quant_config else None
    if quant_algo is None or config.dynamic_weight_quant:
        return True  # unquantized, or fuse-then-quantize-once is safe
    return quant_algo != QuantAlgo.NVFP4  # static NVFP4 may have mismatched per-shard scales


def _use_fused_proj_out(config: DiffusionModelConfig) -> bool:
    if config.extra_attrs.get("force_unfused_lora_topology"):
        return False
    quant_algo = config.quant_config.quant_algo if config.quant_config else None
    return quant_algo is None or config.dynamic_weight_quant


def _gated_mlp_in_tp_kwargs(mlp_hidden_dim: int, mapping, tp_size: int) -> dict:
    """TP kwargs for mlp_in (fused gate+up, feeds proj_out); mirrors
    GatedMLP.gate_up_proj's COLUMN + FUSED_GATE_UP_LINEAR sharding."""
    if tp_size <= 1:
        return {}
    tp_rank = mapping.tp_rank
    local_start = Linear._calc_shard(mlp_hidden_dim, tp_size, tp_rank)
    local_end = Linear._calc_shard(mlp_hidden_dim, tp_size, tp_rank + 1)
    local_size = local_end - local_start
    return dict(
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
        fused_weight_shard_indices_mapping={
            "gate": (0, local_size),
            "up": (local_size, local_size),
        },
        override_tp_sharding={"gate": (local_start, local_end), "up": (local_start, local_end)},
        reduce_output=False,
    )


def _fuse_out_proj_weights(weights: dict) -> dict:
    """Concatenate attn_out + mlp_out into one proj_out.weight (bias-free, unquantized only)."""
    out = {}
    consumed = set()
    for key, value in weights.items():
        if key.endswith(".attn_out.weight"):
            prefix = key[: -len(".attn_out.weight")]
            mlp_key = f"{prefix}.ff.down_proj.weight"
            if mlp_key not in weights:
                raise KeyError(f"{key} has no matching {mlp_key} to fuse into proj_out")
            out[f"{prefix}.proj_out.weight"] = torch.cat([value, weights[mlp_key]], dim=1)
            consumed.update([key, mlp_key])
        elif key.endswith(".attn_out.bias") or key.endswith(".ff.down_proj.bias"):
            raise NotImplementedError(
                f"Fused proj_out does not handle out-proj bias ({key}); "
                "FLUX.3 out-projections are bias-free."
            )
    for key, value in weights.items():
        if key not in consumed and key not in out:
            out[key] = value
    return out


def _split_mlp_in_weights(weights: dict) -> dict:
    """Split each mlp_in.weight [gate|up] into ff.gate_up_proj.{gate,up}.weight
    so GatedMLP's FUSED_GATE_UP_LINEAR loader can shard gate and up
    independently and then interleave them per rank."""
    out = {}
    for key, value in weights.items():
        if key.endswith(".mlp_in.weight"):
            prefix = key[: -len(".mlp_in.weight")]
            half = value.shape[0] // 2
            out[f"{prefix}.ff.gate_up_proj.gate.weight"] = value[:half]
            out[f"{prefix}.ff.gate_up_proj.up.weight"] = value[half:]
        elif key.endswith(".mlp_in.weight_scale"):
            prefix = key[: -len(".mlp_in.weight_scale")]
            half = value.shape[0] // 2
            out[f"{prefix}.ff.gate_up_proj.gate.weight_scale"] = value[:half]
            out[f"{prefix}.ff.gate_up_proj.up.weight_scale"] = value[half:]
        elif key.endswith(".mlp_in.bias"):
            prefix = key[: -len(".mlp_in.bias")]
            half = value.shape[0] // 2
            out[f"{prefix}.ff.gate_up_proj.gate.bias"] = value[:half]
            out[f"{prefix}.ff.gate_up_proj.up.bias"] = value[half:]
        elif key.endswith(
            (".mlp_in.weight_scale_2", ".mlp_in.input_scale", ".mlp_in.pre_quant_scale")
        ):
            prefix, suffix = key.rsplit(".mlp_in.", 1)
            out[f"{prefix}.ff.gate_up_proj.gate.{suffix}"] = value
            out[f"{prefix}.ff.gate_up_proj.up.{suffix}"] = value
        else:
            out[key] = value
    return out


def _split_mlp_in_weights_for_fused_proj_out(weights: dict) -> dict:
    """Split mlp_in.weight [gate|up] into mlp_in.{gate,up}.weight for TP>1
    fused-proj_out configs (see _gated_mlp_in_tp_kwargs)."""
    out = {}
    for key, value in weights.items():
        if key.endswith(".mlp_in.weight"):
            prefix = key[: -len(".mlp_in.weight")]
            half = value.shape[0] // 2
            out[f"{prefix}.mlp_in.gate.weight"] = value[:half]
            out[f"{prefix}.mlp_in.up.weight"] = value[half:]
        else:
            out[key] = value
    return out


# =============================================================================
# Building blocks
# =============================================================================


class Flux3MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, config: DiffusionModelConfig):
        super().__init__()
        self.in_layer = Linear(
            in_dim,
            hidden_dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )
        self.silu = nn.SiLU()
        self.out_layer = Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Flux3Modulation(nn.Module):
    """Single modulation: projects a (possibly per-token) vector to shift/scale/gate."""

    def __init__(self, dim: int, config: DiffusionModelConfig):
        super().__init__()
        self.multiplier = 3
        self.lin = Linear(
            dim,
            self.multiplier * dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )
        self.act = nn.SiLU()

    def forward(self, vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.lin(self.act(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        shift, scale, gate = out.chunk(self.multiplier, dim=-1)
        return shift, scale, gate


class Flux3LastLayer(nn.Module):
    def __init__(
        self, hidden_size: int, out_channels: int, config: DiffusionModelConfig, eps: float = 1e-6
    ):
        super().__init__()
        dtype = config.torch_dtype
        self.norm_final = LayerNorm(
            hidden_size=hidden_size, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )
        self.linear = Linear(
            hidden_size,
            out_channels,
            bias=False,
            dtype=dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            Linear(
                hidden_size,
                2 * hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
            ),
        )

    def forward(self, x: Tensor, vec: Tensor | None, mod: Tensor | None = None) -> Tensor:
        # `mod` is adaLN_modulation(vec) precomputed by the caller (step-batched flow-map rows).
        if mod is None:
            mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        return self.linear(x)


class Flux3Attention(Attention):
    """FLUX.3 self-attention on the shared TRT-LLM Attention: fused QKV, per-head
    QK-norm, FLUX.3 RoPE via the flux3_pe_to_cos_sin bridge. attn_pre_out returns
    the pre-out-projection result (and optionally v) so the block can apply PAG
    and its own output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: DiffusionModelConfig,
        eps: float = 1e-6,
        bias: bool = False,
        module_name: Optional[str] = None,
    ):
        head_dim = hidden_size // num_heads
        tp_size = config.mapping.tp_size if config.mapping else 1
        fuse_qkv = _use_fused_qkv(config)
        qkv_mode = QKVMode.FUSE_QKV if fuse_qkv else QKVMode.SEPARATE_QKV
        # Fused QK-norm+RoPE kernel needs the packed QKV tensor, head_dim in
        # {64, 128}, and no TP; else eager path.
        fuse_qk_norm_rope = fuse_qkv and head_dim in (64, 128) and tp_size == 1
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            qkv_mode=qkv_mode,
            separate_qkv_is_self_attention=True,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=eps,
            bias=bias,
            interleave=True,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            config=config,
            module_name=module_name,
        )
        # The block applies its own output projection (fused proj_out, or
        # separate attn_out/mlp_out), so drop the base to_out.
        del self.to_out

    def apply_qk_norm(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        q, k = q.float(), k.float()
        q = (
            q
            * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_q.variance_epsilon)
            * self.norm_q.weight.float()
        )
        k = (
            k
            * torch.rsqrt(k.square().mean(-1, keepdim=True) + self.norm_k.variance_epsilon)
            * self.norm_k.weight.float()
        )
        return q, k

    def attn_pre_out(
        self,
        hidden_states: Tensor,
        freqs: Tuple[Tensor, Tensor],
        need_v: bool = False,
        quantized: Fp8RowwiseQuantizedTensor | NVFP4Activation | None = None,
        sample_bounds: Bounds | None = None,
        timestep: Tensor | None = None,
        side_gate_up: tuple | None = None,
    ):
        """QKV -> QK-norm -> RoPE -> attention, returning the pre-out-projection
        result [B, S, H*D] (and v [B, H, S, D] when need_v, for PAG). Fused path
        does QK-norm+RoPE in one kernel; unfused uses eager rms_norm + rotary.
        side_gate_up (primsts.sm120_side_gate_up_args): also return the block's packed
        gate_up (packed, scales), run on a side stream beside attention, or None when the
        attention call is not the covered FP8 SM12x one (the caller then runs gate_up)."""
        b, s = hidden_states.shape[:2]
        cos, sin = freqs[0], freqs[1]
        projection_input = hidden_states if quantized is None else quantized

        direct = _primsts_qkv_fp8_attention(
            self,
            projection_input,
            freqs,
            sample_bounds is None
            or (
                _ATTEND_SINGLE_SEGMENT_FAST_PATH
                and b == 1
                and len(sample_bounds) == 1
                and sample_bounds[0][0] == 0
                and sample_bounds[0][1] == s
            ),
        )
        if direct is not None:
            # NVFP4 QKV straight to E4M3 + the same fmha_v2 call (bitwise to the path below).
            attn, gemm = direct
            if need_v:
                v = gemm.view(3, b * s, -1)[2]
                v = v.view(b, s, self.local_num_key_value_heads, self.head_dim).transpose(1, 2)
                return attn, v  # [B, H, S, D] for apply_flux3_pag
            return attn

        qkv = _primsts_qkv_forward(self, projection_input, freqs)
        if self.fuse_qk_norm_rope or qkv is not None:
            if qkv is None:
                qkv = _primsts_linear_forward(self.qkv_proj, projection_input)
                fc = cos.reshape(-1, self.head_dim).contiguous()
                fs = sin.reshape(-1, self.head_dim).contiguous()
                self.apply_packed_qk_norm_rope(qkv, fc, fs)
                qkv = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)
            q, k, v = qkv
            side = None
            if side_gate_up is not None:
                side = self._attend_side_gate_up(q, k, v, sample_bounds, side_gate_up)
            if side is not None:
                attn, gate_up = side
            else:
                attn, gate_up = self._attend(q, k, v, sample_bounds, timestep), None
            if need_v:
                v = v.view(b, s, self.local_num_key_value_heads, self.head_dim).transpose(1, 2)
                if side_gate_up is not None:
                    return attn, v, gate_up
                return attn, v  # [B, H, S, D] for apply_flux3_pag
            if side_gate_up is not None:
                return attn, gate_up
            return attn

        separate = (
            separate_qkv_forward(self, projection_input)
            if self.qkv_mode == QKVMode.SEPARATE_QKV
            else None
        )
        native_input = (
            original_input(projection_input)
            if isinstance(projection_input, NVFP4Activation)
            else projection_input
        )
        q, k, v = separate if separate is not None else self.get_qkv(native_input)
        q = q.view(b, -1, self.local_num_attention_heads, self.head_dim)
        k = k.view(b, -1, self.local_num_attention_heads, self.head_dim)
        v = v.view(b, -1, self.local_num_attention_heads, self.head_dim)
        q, k = self.apply_qk_norm(q, k)
        q = apply_rotary_emb(q, cos, sin).to(v.dtype)
        k = apply_rotary_emb(k, cos, sin).to(v.dtype)
        attn = self._attend(q.flatten(2), k.flatten(2), v.flatten(2), sample_bounds, timestep)
        if need_v:
            if side_gate_up is not None:
                return attn, v.transpose(1, 2), None
            return attn, v.transpose(1, 2)  # v as [B, H, S, D] for apply_flux3_pag
        if side_gate_up is not None:
            return attn, None
        return attn

    def _attend_side_gate_up(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sample_bounds: Bounds | None,
        side_gate_up: tuple,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]] | None:
        """_attend's FP8 SM12x call (no-LSE, non-causal, one full segment) fused with the
        block's side-stream gate_up into one op; None when _attend would take another path."""
        backend = self.attn
        if not (isinstance(backend, FlashInferAttention) and backend._fp8_sm12x):
            return None
        if backend.preferred_layout != AttentionTensorLayout.NHD:
            return None
        if sample_bounds is not None and not (
            _ATTEND_SINGLE_SEGMENT_FAST_PATH
            and len(sample_bounds) == 1
            and sample_bounds[0][0] == 0
            and sample_bounds[0][1] == q.shape[1]
            and q.shape[1] > 0
        ):
            return None
        if q.dtype != torch.bfloat16 or q.shape[1] < backend._fp8_min_seq_len:
            return None
        batch_size = q.shape[0]
        q = q.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
        k = k.view(batch_size, -1, self.local_num_key_value_heads, self.head_dim)
        v = v.view(batch_size, -1, self.local_num_key_value_heads, self.head_dim)
        if q.shape != k.shape or q.shape != v.shape:
            return None
        backend._validate_inputs(q, k, v, None)
        data, weight, scales, weight_scale, inverse_scale, original, source, mode = side_gate_up
        out, packed, packed_scales = fp8_attention_side_gate_up(
            q, k, v, backend.scale, False, data, weight, scales, weight_scale, inverse_scale,
            original, source, mode,
        )
        return out.flatten(2), (packed, packed_scales)

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sample_bounds: Bounds | None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        if sample_bounds is None:
            return self._attn_impl(q, k, v, timestep=timestep)
        if (
            _ATTEND_SINGLE_SEGMENT_FAST_PATH
            and len(sample_bounds) == 1
            and sample_bounds[0][0] == 0
            and sample_bounds[0][1] == q.shape[1]
            and q.shape[1] > 0
        ):
            # One full-range segment: the backend's [B, S, H*D] result is the whole
            # output, so skip the zero-filled buffer and the slice copy into it.
            # Keep the same (contiguous) inputs the general path hands the backend.
            if not isinstance(self.attn, VanillaAttention) and not getattr(
                self.attn, "accepts_strided_qkv", False
            ):
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            return self._attn_impl(q, k, v, timestep=timestep)
        if q.shape[0] == 1 and len(sample_bounds) > 1:
            length = sample_bounds[0][1]
            if (
                length > 0
                and sample_bounds[-1][1] == q.shape[1]
                and all(
                    (start, end) == (i * length, (i + 1) * length)
                    for i, (start, end) in enumerate(sample_bounds)
                )
            ):
                # Equal packed segments are independent batch rows for every backend.
                return self._attn_impl(
                    *(x.view(len(sample_bounds), length, x.shape[-1]) for x in (q, k, v)),
                    timestep=timestep,
                ).reshape_as(q)
        output = torch.zeros_like(q)
        for start, end in sample_bounds:
            if end > start:
                q_sample, k_sample, v_sample = q[:, start:end], k[:, start:end], v[:, start:end]
                # SDPA (and backends that read strided views) take QKV without copying.
                if not isinstance(self.attn, VanillaAttention) and not getattr(
                    self.attn, "accepts_strided_qkv", False
                ):
                    q_sample, k_sample, v_sample = (
                        q_sample.contiguous(),
                        k_sample.contiguous(),
                        v_sample.contiguous(),
                    )
                output[:, start:end] = self._attn_impl(
                    q_sample, k_sample, v_sample, timestep=timestep
                )
        return output


def _flux3_shared_fp8_input(
    attn: Flux3Attention, mlp_in: Linear, x: Tensor
) -> Fp8RowwiseQuantizedTensor | NVFP4Activation | None:
    """Share quantized QKV/MLP inputs only when their loaded quantization recipes match."""
    nvfp4 = shared_nvfp4_input(attn, mlp_in, x)
    if nvfp4 is not None:
        return nvfp4
    fp8_excluded = sm120_fp8_shared_input(attn, mlp_in, x)
    if fp8_excluded is not None:
        return fp8_excluded
    qkv = getattr(attn, "qkv_proj", None)
    if not all(
        isinstance(layer, Linear) and isinstance(layer.quant_method, FP8RowwiseLinearMethod)
        for layer in (qkv, mlp_in)
    ):
        return None
    flat = x.reshape(-1, x.shape[-1])
    if isinstance(qkv.quant_method, Flux3FP8RowwiseLinearMethod):
        data, scales = quantize_fp8r(flat)
    elif FP8RowwiseLinearMethod.use_tunable_quantize:
        data, scales = torch.ops.trtllm.tunable_fp8_per_token_quant(flat)
    else:
        data, scales = torch.ops.tensorrt_llm.quantize_e4m3_activation(flat)
    return Fp8RowwiseQuantizedTensor(x, data, scales)


def _flux3_gated_mlp_forward(
    ff: GatedMLP, x: torch.Tensor | Fp8RowwiseQuantizedTensor | NVFP4Activation
) -> torch.Tensor:
    """Dispatch PrimsTS or native MLP fusion with the downstream scale contract."""
    nvfp4_output = nvfp4_mlp_forward(ff, x)
    if nvfp4_output is not None:
        return nvfp4_output
    fp8r_output = sm120_fp8_mlp_forward(ff, x)
    if fp8r_output is not None:
        return fp8r_output
    prepared = getattr(ff.gate_up_proj, "_flux3_primsts_fp8", None)
    if prepared is None:
        prepared = getattr(ff.gate_up_proj, "_flux3_primsts_nvfp4", None)
    original = original_input(x)
    if prepared is not None and prepared.swiglu and prepared.can_run(original):
        return _primsts_linear_forward(ff.down_proj, prepared(x))
    if isinstance(x, NVFP4Activation):
        x = original
    return ff(x)


# Broadcast adaLN modulation (compact, single-segment streams): mode blocks take [1, 1, D]
# shift/scale/gate rows and single blocks take [1, 2, D] tables plus a [1, S, 1] text-row
# select mask, so no per-token [1, S, D] modulation is materialized. Same values and math;
# Inductor picks another reduction config for the single-block LN+modulate kernel, so the
# LayerNorm sums are reordered (mode blocks stay bitwise).
FLUX3_BROADCAST_ADALN = True
# The step-batched flow-map text stream (flow_map_text_step) hands forward its single_* rows
# as broadcast [1, 1, D] rows too, so the single blocks keep the table + select path.
FLUX3_TEXT_STEP_BROADCAST = True


def _flux3_select_mod(mod: tuple) -> Tuple[Tensor, Tensor, Tensor]:
    """Return per-token-broadcastable (shift, scale, gate) from a 3- or 4-tuple.

    A 4-tuple is (shift_table, scale_table, gate_table, select) with [1, 2, D] tables and a
    bool [1, S, 1] select (True picks row 1). torch.where copies the bf16 values exactly and
    Inductor fuses it as a pointwise load into the consumer kernels.
    """
    if len(mod) == 3:
        return mod
    shift, scale, gate, select = mod
    return tuple(torch.where(select, value[:, 1:2], value[:, 0:1]) for value in (shift, scale, gate))


class Flux3ModeBlock(nn.Module):
    """Per-modality siloed self-attention block ("early"/"late" stage)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: DiffusionModelConfig,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        eps: float = 1e-6,
        module_name: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        dtype = config.torch_dtype
        mapping = getattr(config, "mapping", None)
        tp_size = getattr(mapping, "tp_size", 1)

        self.attn = Flux3Attention(
            hidden_size,
            num_heads,
            config,
            eps=eps,
            bias=qkv_bias,
            module_name=None if module_name is None else f"{module_name}.attn",
        )
        self._fuse_proj_out = _use_fused_proj_out(config)
        if self._fuse_proj_out:
            self.mlp_in = Linear(
                hidden_size,
                self.mlp_hidden_dim * 2,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **_gated_mlp_in_tp_kwargs(self.mlp_hidden_dim, mapping, tp_size),
            )
            self.proj_out = FluxJointAttnMLPProj(
                attn_dim=hidden_size,
                mlp_dim=self.mlp_hidden_dim,
                out_dim=hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                config=config,
            )
        else:
            use_cute_dsl_swiglu = (
                torch.cuda.is_available()
                and get_sm_version() in (100, 103)
                and config.quant_config is not None
                and config.quant_config.quant_algo == QuantAlgo.NVFP4
            )
            self.ff = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=self.mlp_hidden_dim,
                bias=False,
                dtype=dtype,
                config=config,
                reduce_output=False,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_swiglu,
            )
            out_kwargs = {}
            self._allreduce = None
            if tp_size > 1:
                out_kwargs = dict(
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    reduce_output=False,
                )
                self._allreduce = AllReduce(mapping, strategy=config.allreduce_strategy)
            self.attn_out = Linear(
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **out_kwargs,
            )
        self.pre_norm = LayerNorm(
            hidden_size=hidden_size, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )
        self._pertoken_adaln = Flux3PerTokenAdaLN(hidden_size, dtype)
        # Set by prepare_sm120_tail_pair after weight preparation (static, read in forward).
        self._sm120_tail_pair = False
        self._sm120_side_gate_up = 0

    def forward(
        self,
        x: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Bounds | None = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        fused = adaln_nvfp4_input(self, x, mod, self.pre_norm.variance_epsilon, freqs=freqs)
        if fused is not None:
            x_mod, quantized = fused
        else:
            x_mod = self._pertoken_adaln.apply(
                x, mod_shift, mod_scale, self.pre_norm, self.pre_norm.variance_epsilon
            )
            mlp_in = self.mlp_in if self._fuse_proj_out else self.ff.gate_up_proj
            quantized = _flux3_shared_fp8_input(self.attn, mlp_in, x_mod)
        side_args = None
        if self._sm120_side_gate_up and isinstance(quantized, NVFP4Activation):
            side_args = sm120_side_gate_up_args(
                self, quantized._replace(original=x_mod.reshape(-1, x_mod.shape[-1]))
            )
        gate_up = None
        if side_args is not None:
            attn, gate_up = self.attn.attn_pre_out(
                x_mod,
                freqs,
                quantized=quantized,
                sample_bounds=sample_bounds,
                timestep=timestep,
                side_gate_up=side_args,
            )
        else:
            attn = self.attn.attn_pre_out(
                x_mod, freqs, quantized=quantized, sample_bounds=sample_bounds, timestep=timestep
            )

        if self._fuse_proj_out:
            mlp = self.mlp_in(x_mod if quantized is None else quantized)
            x1, x2 = mlp.chunk(2, dim=-1)
            mlp_act = F.silu(x1) * x2
            output = self.proj_out(attn, mlp_act)
        else:
            mlp_shape = x_mod.shape
            mlp_input = x_mod.reshape(-1, mlp_shape[-1])
            if isinstance(quantized, NVFP4Activation):
                mlp_input = quantized._replace(original=mlp_input)
            elif quantized is not None:
                mlp_input = Fp8RowwiseQuantizedTensor(
                    mlp_input, quantized.quantized, quantized.scales
                )
            pair = (
                sm120_tail_pair_forward(self, attn, mlp_input, gate_up)
                if self._sm120_tail_pair
                else None
            )
            if pair is not None:
                attn_y, mlp_y = pair
                output = attn_y.reshape(*attn.shape[:-1], -1) + mlp_y.reshape(
                    *mlp_shape[:-1], -1
                )
            else:
                mlp_out = _flux3_gated_mlp_forward(self.ff, mlp_input).reshape(*mlp_shape[:-1], -1)
                output = _primsts_linear_forward(self.attn_out, attn) + mlp_out
            if self._allreduce is not None:
                output = self._allreduce(output)
        return x + mod_gate * output


class Flux3SingleStreamBlock(nn.Module):
    """Joint image, reference-image, and text attention within each packed sample."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: DiffusionModelConfig,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        module_name: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        dtype = config.torch_dtype
        mapping = getattr(config, "mapping", None)
        tp_size = getattr(mapping, "tp_size", 1)

        self.attn = Flux3Attention(
            hidden_size,
            num_heads,
            config,
            eps=eps,
            bias=False,
            module_name=None if module_name is None else f"{module_name}.attn",
        )
        self._fuse_proj_out = _use_fused_proj_out(config)
        if self._fuse_proj_out:
            self.mlp_in = Linear(
                hidden_size,
                self.mlp_hidden_dim * 2,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **_gated_mlp_in_tp_kwargs(self.mlp_hidden_dim, mapping, tp_size),
            )
            self.proj_out = FluxJointAttnMLPProj(
                attn_dim=hidden_size,
                mlp_dim=self.mlp_hidden_dim,
                out_dim=hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                config=config,
            )
        else:
            use_cute_dsl_swiglu = (
                torch.cuda.is_available()
                and get_sm_version() in (100, 103)
                and config.quant_config is not None
                and config.quant_config.quant_algo == QuantAlgo.NVFP4
            )
            self.ff = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=self.mlp_hidden_dim,
                bias=False,
                dtype=dtype,
                config=config,
                reduce_output=False,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_swiglu,
            )
            out_kwargs = {}
            self._allreduce = None
            if tp_size > 1:
                out_kwargs = dict(
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    reduce_output=False,
                )
                self._allreduce = AllReduce(mapping, strategy=config.allreduce_strategy)
            self.attn_out = Linear(
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **out_kwargs,
            )
        self.pre_norm = LayerNorm(
            hidden_size=hidden_size, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )
        self._pertoken_adaln = Flux3PerTokenAdaLN(hidden_size, dtype)
        # Set by prepare_sm120_tail_pair after weight preparation (static, read in forward).
        self._sm120_tail_pair = False
        self._sm120_side_gate_up = 0

    def forward(
        self,
        combined: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        pag_strength: float = 0.0,
        sample_bounds: Bounds | None = None,
        pag_samples: tuple[int, ...] = (0,),
        timestep: Tensor | None = None,
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = _flux3_select_mod(mod)
        residual = combined

        eps = self.pre_norm.variance_epsilon
        fused = adaln_nvfp4_input(self, combined, mod, eps, freqs=freqs)
        if fused is not None:
            combined, quantized = fused
        else:
            combined = self._pertoken_adaln.apply(combined, mod_shift, mod_scale, self.pre_norm, eps)
            mlp_in = self.mlp_in if self._fuse_proj_out else self.ff.gate_up_proj
            quantized = _flux3_shared_fp8_input(self.attn, mlp_in, combined)
        side_args = None
        if self._sm120_side_gate_up and isinstance(quantized, NVFP4Activation):
            side_args = sm120_side_gate_up_args(
                self, quantized._replace(original=combined.reshape(-1, combined.shape[-1]))
            )
        gate_up = None
        if side_args is not None:
            attn, v, gate_up = self.attn.attn_pre_out(
                combined,
                freqs,
                need_v=True,
                quantized=quantized,
                sample_bounds=sample_bounds,
                timestep=timestep,
                side_gate_up=side_args,
            )
        else:
            attn, v = self.attn.attn_pre_out(
                combined,
                freqs,
                need_v=True,
                quantized=quantized,
                sample_bounds=sample_bounds,
                timestep=timestep,
            )
        attn = apply_flux3_pag(attn, v, pag_strength, sample_bounds, pag_samples)

        if self._fuse_proj_out:
            mlp = self.mlp_in(combined if quantized is None else quantized)
            x1, x2 = mlp.chunk(2, dim=-1)
            mlp_act = F.silu(x1) * x2
            output = self.proj_out(attn, mlp_act)
        else:
            combined_shape = combined.shape
            mlp_input = combined.reshape(-1, combined_shape[-1])
            if isinstance(quantized, NVFP4Activation):
                mlp_input = quantized._replace(original=mlp_input)
            elif quantized is not None:
                mlp_input = Fp8RowwiseQuantizedTensor(
                    mlp_input, quantized.quantized, quantized.scales
                )
            pair = (
                sm120_tail_pair_forward(self, attn, mlp_input, gate_up)
                if self._sm120_tail_pair
                else None
            )
            if pair is not None:
                attn_y, mlp_y = pair
                output = attn_y.reshape(*attn.shape[:-1], -1) + mlp_y.reshape(
                    *combined_shape[:-1], -1
                )
            else:
                mlp_out = _flux3_gated_mlp_forward(self.ff, mlp_input).reshape(
                    *combined_shape[:-1], -1
                )
                output = _primsts_linear_forward(self.attn_out, attn) + mlp_out
            if self._allreduce is not None:
                output = self._allreduce(output)
        return residual + mod_gate * output


class _Flux3FP8RModeBlock(Flux3ModeBlock):
    """A BF16 checkpoint-excluded mode block switched to the SM120 FP8R path.

    The override gives the block its own forward code object, so its Dynamo cache entries,
    guards and automatic-dynamic state stay separate from the NVFP4 and BF16 blocks of the
    same class. Sharing them made the NVFP4 blocks' outputs depend on request history
    (bits changed after a request with another text length).
    """

    # Set by prepare_fp8r_adaln_quant (flux3_adaln_fp8r.py): the fused LN + adaLN-modulate +
    # per-token E4M3 head. Returns (x_mod, quantized) or None for the base forward.
    _flux3_fp8r_adaln_head = None

    def forward(
        self,
        x: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Bounds | None = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        head = self._flux3_fp8r_adaln_head
        fused = None if head is None else head(self, x, mod, freqs)
        if fused is None:
            return super().forward(x, freqs, mod, sample_bounds=sample_bounds, timestep=timestep)
        # Flux3ModeBlock.forward after its head (FP8R blocks never fuse proj_out).
        mod_gate = mod[2]
        x_mod, quantized = fused
        attn = self.attn.attn_pre_out(
            x_mod, freqs, quantized=quantized, sample_bounds=sample_bounds, timestep=timestep
        )
        mlp_shape = x_mod.shape
        mlp_input = x_mod.reshape(-1, mlp_shape[-1])
        mlp_input = Fp8RowwiseQuantizedTensor(mlp_input, quantized.quantized, quantized.scales)
        pair = sm120_tail_pair_forward(self, attn, mlp_input) if self._sm120_tail_pair else None
        if pair is not None:
            attn_y, mlp_y = pair
            output = attn_y.reshape(*attn.shape[:-1], -1) + mlp_y.reshape(*mlp_shape[:-1], -1)
        else:
            mlp_out = _flux3_gated_mlp_forward(self.ff, mlp_input).reshape(*mlp_shape[:-1], -1)
            output = _primsts_linear_forward(self.attn_out, attn) + mlp_out
        if self._allreduce is not None:
            output = self._allreduce(output)
        return x + mod_gate * output


class _Flux3FP8RSingleStreamBlock(Flux3SingleStreamBlock):
    """Single-stream counterpart of _Flux3FP8RModeBlock."""

    _flux3_fp8r_adaln_head = None

    def forward(
        self,
        combined: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        pag_strength: float = 0.0,
        sample_bounds: Bounds | None = None,
        pag_samples: tuple[int, ...] = (0,),
        timestep: Tensor | None = None,
    ) -> Tensor:
        head = self._flux3_fp8r_adaln_head
        fused = None if head is None else head(self, combined, mod, freqs)
        if fused is None:
            return super().forward(
                combined,
                freqs,
                mod,
                pag_strength=pag_strength,
                sample_bounds=sample_bounds,
                pag_samples=pag_samples,
                timestep=timestep,
            )
        # Flux3SingleStreamBlock.forward after its head (FP8R blocks never fuse proj_out).
        mod_gate = _flux3_select_mod(mod)[2]
        residual = combined
        combined, quantized = fused
        attn, v = self.attn.attn_pre_out(
            combined,
            freqs,
            need_v=True,
            quantized=quantized,
            sample_bounds=sample_bounds,
            timestep=timestep,
        )
        attn = apply_flux3_pag(attn, v, pag_strength, sample_bounds, pag_samples)
        combined_shape = combined.shape
        mlp_input = combined.reshape(-1, combined_shape[-1])
        mlp_input = Fp8RowwiseQuantizedTensor(mlp_input, quantized.quantized, quantized.scales)
        pair = sm120_tail_pair_forward(self, attn, mlp_input) if self._sm120_tail_pair else None
        if pair is not None:
            attn_y, mlp_y = pair
            output = attn_y.reshape(*attn.shape[:-1], -1) + mlp_y.reshape(
                *combined_shape[:-1], -1
            )
        else:
            mlp_out = _flux3_gated_mlp_forward(self.ff, mlp_input).reshape(
                *combined_shape[:-1], -1
            )
            output = _primsts_linear_forward(self.attn_out, attn) + mlp_out
        if self._allreduce is not None:
            output = self._allreduce(output)
        return residual + mod_gate * output


_FP8R_BLOCK_CLASSES = {
    Flux3ModeBlock: _Flux3FP8RModeBlock,
    Flux3SingleStreamBlock: _Flux3FP8RSingleStreamBlock,
}


def _set_fp8r_block_class(block: nn.Module, switched: bool) -> None:
    """Swap an (uncompiled) block between its base class and its FP8R subclass."""
    for base, fp8r in _FP8R_BLOCK_CLASSES.items():
        if type(block) in (base, fp8r):
            block.__class__ = fp8r if switched else base
            return


# =============================================================================
# Main model
# =============================================================================


class Flux3Model(BaseDiffusionModel):
    """FLUX.3 image DiT with packed modality and joint attention."""

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)

        mapping = getattr(model_config, "mapping", None)
        self._tp_size = mapping.tp_size if mapping is not None else 1

        pretrained_config = model_config.pretrained_config
        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        in_channels = dict(getattr(pretrained_config, "in_channels", DEFAULT_IN_CHANNELS))
        sequence = dict(getattr(pretrained_config, "sequence", DEFAULT_SEQUENCE))
        context_in_dim = getattr(pretrained_config, "context_in_dim", 20480)
        hidden_size = getattr(pretrained_config, "hidden_size", 3072)
        num_heads = getattr(pretrained_config, "num_heads", 24)
        depth = getattr(pretrained_config, "depth", 5)
        depth_single_blocks = getattr(pretrained_config, "depth_single_blocks", 28)
        axes_dim = list(getattr(pretrained_config, "axes_dim", [32, 32, 32, 32]))
        theta = getattr(pretrained_config, "theta", 10000)
        mlp_ratio = getattr(pretrained_config, "mlp_ratio", 3.0)
        qkv_bias = getattr(pretrained_config, "qkv_bias", False)
        eps = getattr(pretrained_config, "eps", 1e-6)

        if getattr(pretrained_config, "vec_in_dim", None) is not None or getattr(
            pretrained_config, "depth_late_blocks", 0
        ):
            raise ValueError("FLUX.3 image-small requires no pooled vector or late blocks")

        for _, mod_key in sequence.items():
            assert mod_key in in_channels, (sequence, in_channels)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.in_channels = in_channels
        self.sequence = sequence
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks

        block_kwargs = dict(
            config=model_config,
            mlp_ratio=mlp_ratio,
            eps=eps,
        )

        self.pe_embedder = Flux3PosEmbed(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.emb_in = nn.ModuleDict(
            {
                m: Linear(
                    c,
                    hidden_size,
                    bias=False,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights_in_init=skip_create_weights,
                    force_dynamic_quantization=force_dynamic_quant,
                )
                for m, c in in_channels.items()
            }
        )
        self.txt_in = Linear(
            context_in_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
        )

        self.time_in = Flux3MLPEmbedder(256, hidden_size, config=model_config)
        self.flow_map = getattr(pretrained_config, "flow_map", False)
        if self.flow_map:
            self.ref_time_in = Flux3MLPEmbedder(256, hidden_size, config=model_config)
        content_modalities = sorted(in_channels.keys())

        def _make_modulations() -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    **{
                        m: Flux3Modulation(hidden_size, config=model_config)
                        for m in content_modalities
                    },
                    "txt": Flux3Modulation(hidden_size, config=model_config),
                }
            )

        self.early_stream_modulations = _make_modulations()
        self.single_stream_modulations = _make_modulations()

        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [
                        Flux3ModeBlock(hidden_size, num_heads, qkv_bias=qkv_bias, **block_kwargs)
                        for _ in range(depth)
                    ]
                )
                for m in content_modalities
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                Flux3ModeBlock(hidden_size, num_heads, qkv_bias=qkv_bias, **block_kwargs)
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                Flux3SingleStreamBlock(hidden_size, num_heads, **block_kwargs)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = nn.ModuleDict(
            {
                m: Flux3LastLayer(hidden_size, c, config=model_config, eps=eps)
                for m, c in in_channels.items()
                if m in getattr(pretrained_config, "out_modes", ["image"])
            }
        )

        vgm = getattr(model_config, "visual_gen_mapping", None)
        self.sharder = SequenceSharder.from_vgm(vgm, num_attention_heads=num_heads)

        self._early_adaln_runtime = {
            name: Flux3PerTokenAdaLNRuntime([block._pertoken_adaln for block in blocks])
            for name, blocks in {
                "text": self.txt_mode_blocks,
                **self.content_mode_blocks,
            }.items()
        }
        self._single_adaln_runtime = Flux3PerTokenAdaLNRuntime(
            [block._pertoken_adaln for block in self.single_blocks]
        )

        self.__post_init__()

    def __post_init__(self):
        quant = self.model_config.quant_config
        if quant is not None and quant.exclude_modules:
            excluded = _remap_checkpoint_keys(
                {name + ".weight": None for name in quant.exclude_modules},
                fuse_qkv=_use_fused_qkv(self.model_config),
            )
            quant.exclude_modules = sorted(
                {
                    re.sub(r"\.attn\.[qkv]_proj\.", ".attn.qkv_proj.", name)
                    .replace(".mlp_in.", ".ff.gate_up_proj.")
                    .removesuffix(".weight")
                    for name in excluded
                }
            )
        self.apply_quant_config_exclude_modules()
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def apply_quant_config_exclude_modules(self):
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        kv_cache_quant_algo = quant_config.kv_cache_quant_algo if quant_config else None
        no_quant_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_excluded = quant_config.is_module_excluded_from_quantization(name)
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = no_quant_config

    def register_cuda_graph_extra_key_fns(self, runner) -> None:
        """Include sample boundaries and PAG settings in graph keys."""
        super().register_cuda_graph_extra_key_fns(runner)

        def _pag_key(*args, **kwargs):
            layers = kwargs.get("pag_single_layers")
            strength = kwargs.get("pag_strength", 1.0)
            # None when PAG is inactive -> runner omits this key part.
            if not layers or not strength:
                return None
            return (tuple(layers), float(strength))

        runner.register_extra_key_fn("flux3_pag", _pag_key)

        quant_attention = self.model_config.attention.quant_attention_config
        if (
            quant_attention is not None
            and quant_attention.vc_attention
            and not quant_attention.vc_repair_budget
        ):
            # The tile-mean recipe follows the V-Smooth step schedule, so each phase gets its own graph.
            from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import vc_smooth_phase
            from tensorrt_llm._torch.visual_gen.denoise_step import get_denoise_step

            runner.register_extra_key_fn(
                "flux3_vc_phase", lambda *args, **kwargs: vc_smooth_phase(get_denoise_step())
            )
        runner.register_extra_key_fn(
            "flux3_layout",
            lambda *args, **kwargs: (kwargs.get("layout"), kwargs.get("pag_samples", (0,))),
        )

    def _packed_inputs(
        self, inputs: dict, layout: Layout | None
    ) -> tuple[dict, Layout, int | None]:
        if layout is not None:
            return inputs, layout, None
        # Dense component callers enter the same packed execution path.
        ctx = inputs["ctx"]
        batch = ctx.shape[0]
        packed = {key: value for key, value in inputs.items() if key.startswith("_flux3_")}
        lengths = {"text": (ctx.shape[1],) * batch}
        for field, value in (
            ("", ctx),
            ("_ids", inputs["ctx_ids"]),
            ("_timesteps", inputs["timesteps_ctx"]),
        ):
            packed["x_text" + field] = value.flatten(0, 1)
        for name in self.in_channels:
            key = "x_" + name
            length = inputs[key].shape[1] if key in inputs else 0
            lengths[name] = (length,) * batch
            if key in inputs:
                for suffix in ("", "_ids", "_timesteps"):
                    packed[key + suffix] = inputs[key + suffix].flatten(0, 1)
            else:
                packed[key] = ctx.new_empty(0, self.in_channels[name])
                packed[key + "_ids"] = inputs["ctx_ids"].new_empty(0, 4)
                packed[key + "_timesteps"] = ctx.new_empty(0)
        return packed, tuple(sorted(lengths.items())), batch

    def _local_layout(self, layout: Layout) -> Layout:
        size = self.sharder.size
        if any(length % size for _, lengths in layout for length in lengths):
            raise ValueError(
                "FLUX.3 sequence parallelism requires each sample's modality length "
                "to be divisible by the sequence-parallel size"
            )
        return tuple(
            (name, tuple(length // size for length in lengths)) for name, lengths in layout
        )

    def _shard_packed(self, value: Tensor, lengths: tuple[int, ...], dim: int) -> Tensor:
        if not self.sharder.is_active or not sum(lengths):
            return value
        return torch.cat(
            [
                self.sharder.shard(value.narrow(dim, start, end - start), dim=dim)
                for start, end in bounds(lengths)
                if end > start
            ],
            dim=dim,
        )

    def _broadcast_adaln_ok(self, x: Tensor) -> bool:
        """Broadcast modulation rows instead of per-token views (see FLUX3_BROADCAST_ADALN).

        Off under sequence sharding and where the fused per-token adaLN kernel (SM100) needs
        materialized [B, S, D] views.
        """
        return (
            FLUX3_BROADCAST_ADALN
            and not self.sharder.is_active
            and not _pertoken_adaln_arch_ok(x.device)
        )

    def _packed_stream(
        self, name: str, inputs: dict, lengths: tuple[int, ...], ref: Tensor | None = None
    ) -> dict[str, Tensor]:
        tokens, ts = inputs["x_" + name], inputs["x_" + name + "_timesteps"]
        if not tokens.shape[0]:
            tokens, ts = tokens.new_zeros(1, tokens.shape[-1]), ts.new_zeros(1)
        # Compact only when the pipeline guarantees one timestep per sample/stream.
        compact = inputs.get("_flux3_uniform_timesteps", False) and any(lengths)
        segments = [
            (index, start, end) for index, (start, end) in enumerate(bounds(lengths)) if end > start
        ]
        timestep_rows = torch.stack([ts[start] for _, start, _ in segments]) if compact else ts
        vec = self.time_in(
            flux3_timestep_embedding(timestep_rows, 256, out_dtype=tokens.dtype)
        ).unsqueeze(0)
        if ref is not None and any(lengths):
            ref_rows = (
                torch.stack([ref[index] for index, _, _ in segments])
                if compact
                else torch.cat(
                    [
                        ref[index].expand(length, -1)
                        for index, length in enumerate(lengths)
                        if length
                    ]
                )
            )
            vec = vec + ref_rows.unsqueeze(0)
        key = "txt" if name == "text" else name
        embedded = (self.txt_in if name == "text" else self.emb_in[name])(tokens).unsqueeze(0)
        state = {"tokens": embedded, "vec": vec}
        for stage, modules in (
            ("early", self.early_stream_modulations),
            ("single", self.single_stream_modulations),
        ):
            for field, value in zip(("shift", "scale", "gate"), modules[key](vec)):
                state[stage + "_" + field] = value
        broadcast = compact and len(segments) == 1 and self._broadcast_adaln_ok(tokens)
        if compact:
            for field, value in state.items():
                if broadcast and field not in ("tokens", "vec"):
                    # One segment: keep the [1, 1, D] row; the blocks broadcast it.
                    state[field] = value[:, 0:1]
                elif field != "tokens":
                    state[field] = torch.cat(
                        [
                            value[:, row : row + 1].expand(-1, end - start, -1)
                            for row, (_, start, end) in enumerate(segments)
                        ],
                        dim=1,
                    )
        return {
            field: value if field == "vec" else self._shard_packed(value, lengths, 1)
            for field, value in state.items()
        }

    @staticmethod
    def _joint_chunks(layout: Layout) -> tuple[list[tuple[str, int, int]], Bounds]:
        ranges = {name: bounds(lengths) for name, lengths in layout}
        chunks, lengths = [], []
        for sample in range(len(layout[0][1])):
            length = 0
            for name, _ in layout:
                start, end = ranges[name][sample]
                if end > start:
                    chunks.append((name, start, end))
                    length += end - start
            lengths.append(length)
        return chunks, bounds(tuple(lengths))

    def _packed_positions(self, inputs: dict, layout: Layout) -> dict[str, Tensor]:
        prepared, positions = {}, {}
        for name, lengths in layout:
            if name.endswith("_cond") and not any(lengths):
                continue
            ids = inputs["x_" + name + "_ids"]
            if not ids.shape[0]:
                ids = ids.new_zeros(1, 4)
            pe = self.pe_embedder(ids.unsqueeze(0))
            pe = self._shard_packed(pe, lengths, 2)
            positions[name] = pe
            prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"] = _pe_to_block_freqs(pe)
        chunks, _ = self._joint_chunks(self._local_layout(layout))
        pe = torch.cat([positions[name][:, :, start:end] for name, start, end in chunks], dim=2)
        prepared["_flux3_joint_cos"], prepared["_flux3_joint_sin"] = _pe_to_block_freqs(pe)
        if FLUX3_ROPE_TABLE:
            for prefix in [f"_flux3_{name}_" for name in positions] + ["_flux3_joint_"]:
                table = rope_cos_sin_table(prepared[prefix + "cos"], prepared[prefix + "sin"])
                prepared[prefix + "cos_sin"] = table.float().contiguous()
        if len(chunks) == 2:
            prepared["_flux3_joint_select"] = self._joint_select(chunks, pe.device)
        return prepared

    @staticmethod
    def _joint_select(chunks: list[tuple[str, int, int]], device: torch.device) -> Tensor:
        """Bool [1, S, 1]: True on the second chunk's rows (built on device, no H2D copy)."""
        first = chunks[0][2] - chunks[0][1]
        total = first + chunks[1][2] - chunks[1][1]
        return (torch.arange(total, device=device) >= first).view(1, total, 1)

    def prepare_static_inputs(self, *, layout: Layout | None = None, **inputs) -> dict[str, Tensor]:
        if torch.is_grad_enabled():
            return {}
        inputs, layout, _ = self._packed_inputs(inputs, layout)
        local_lengths = dict(self._local_layout(layout))
        prepared = self._packed_positions(inputs, layout)
        if self.flow_map:
            return prepared
        for name, lengths in layout:
            if name != "text" and not name.endswith("_cond"):
                continue
            if name.endswith("_cond") and not any(lengths):
                continue
            state = self._packed_stream(name, inputs, lengths)
            blocks = self.txt_mode_blocks if name == "text" else self.content_mode_blocks[name]
            freqs = _block_freqs(prepared, f"_flux3_{name}_")
            mod = tuple(state["early_" + field] for field in ("shift", "scale", "gate"))
            state["tokens"] = self._early_adaln_runtime[name].prepare(state["tokens"], *mod[:2])
            for block in blocks:
                state["tokens"] = block(
                    state["tokens"], freqs, mod, sample_bounds=bounds(local_lengths[name])
                )
            prepared.update(
                {
                    f"_flux3_{name}_{field}": value
                    for field, value in state.items()
                    if not field.startswith("early_")
                }
            )
        return prepared

    # Text attention below this per-sample length runs unquantized BF16 attention in every
    # backend (FlashInfer's FP8 path starts at 1024), so a batch of N steps keeps each step's
    # math. At or above it, per-tensor FP8 scales would span all N steps: keep per-step.
    _TEXT_STEP_BATCH_MAX_LEN = 1024

    def _text_step_batch_max_len(self) -> int:
        """Longest text length (exclusive) the step-batched text stream may run at; 0 disables it.

        Only backends that ignore `timestep` (the batched blocks run without one) and keep
        per-row math for equal packed segments qualify: VANILLA (SDPA) and FLASHINFER. The
        bound is the lowest FP8 switch-over length of any text-block backend (env
        TRTLLM_VISUAL_GEN_FP8_ATTN_MIN_SEQ), so a batch never shares one per-tensor FP8 scale.
        """
        limit = self._TEXT_STEP_BATCH_MAX_LEN
        for block in self.txt_mode_blocks:
            backend = block.attn.attn
            if not (
                isinstance(backend, VanillaAttention)
                or type(backend).__name__ == "FlashInferAttention"
            ):
                return 0
            fp8_min = getattr(backend, "_fp8_min_seq_len", None)
            if fp8_min is not None:
                limit = min(limit, int(fp8_min))
        return limit

    def prepare_flow_map_text(
        self, targets: Tensor, *, layout: Layout | None = None, **inputs
    ) -> dict[str, Tensor] | None:
        """Run a flow-map text stream once for every denoising step.

        A flow-map text stream depends only on the context and the step's target time
        (timesteps_ref), never on the image latents, so step i's text tokens and single_*
        modulation are f(ctx, targets[i]). This runs the stream over N equal segments, one
        per target, so txt_mode_blocks run at M=N*T instead of N times at M=T.

        Returns [1, N*T, D] tokens and [1, N, D] single_* rows, or None when the per-step
        path must run (non-flow-map, sequence parallel, conditioning image streams,
        non-uniform timesteps, long text, or TRTLLM_FLUX3_TEXT_STEP_BATCH=0).
        """
        if (
            not self.flow_map
            or torch.is_grad_enabled()
            or self.sharder.is_active
            or "text" in self.final_layer
            or os.environ.get("TRTLLM_FLUX3_TEXT_STEP_BATCH", "1") == "0"
        ):
            return None
        inputs, layout, _ = self._packed_inputs(inputs, layout)
        if not inputs.get("_flux3_uniform_timesteps", False):
            return None
        if any(name.endswith("_cond") and any(lengths) for name, lengths in layout):
            return None
        text_lengths = dict(layout).get("text", ())
        if len(text_lengths) != 1 or targets.ndim != 1 or not targets.shape[0]:
            return None
        length, steps = text_lengths[0], targets.shape[0]
        if not 0 < length < self._text_step_batch_max_len():
            return None
        if "_flux3_text_cos" not in inputs:
            inputs = {**inputs, **self._packed_positions(inputs, layout)}
        tokens, ts = inputs["x_text"], inputs["x_text_timesteps"]
        # Same rows as the per-step compact path: one time_in row for the text stream plus
        # one ref_time_in row per step.
        target = targets.to(ts)
        ref = self.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
        vec = self.time_in(flux3_timestep_embedding(ts[:1], 256, out_dtype=tokens.dtype))
        vec = (vec + ref).unsqueeze(0)  # [1, N, D]
        mods = {}
        for stage, modules in (
            ("early", self.early_stream_modulations),
            ("single", self.single_stream_modulations),
        ):
            for field, value in zip(("shift", "scale", "gate"), modules["txt"](vec)):
                mods[stage + "_" + field] = value

        def per_token(rows: Tensor) -> Tensor:
            return rows[0].unsqueeze(1).expand(steps, length, rows.shape[-1]).reshape(
                1, steps * length, rows.shape[-1]
            )

        # txt_in does not depend on the step: run it once at M=T and repeat.
        embedded = self.txt_in(tokens).unsqueeze(0).repeat(1, steps, 1)
        mod = tuple(per_token(mods["early_" + field]) for field in ("shift", "scale", "gate"))
        freqs = tuple(
            inputs["_flux3_text_" + kind].repeat(1, steps, 1, 1) for kind in ("cos", "sin")
        )
        sample_bounds = bounds((length,) * steps)
        embedded = self._early_adaln_runtime["text"].prepare(embedded, *mod[:2])
        for block in self.txt_mode_blocks:
            embedded = block(embedded, freqs, mod, sample_bounds=sample_bounds)
        prepared = {"_flux3_text_steps_tokens": embedded}
        for field in ("shift", "scale", "gate"):
            prepared["_flux3_text_steps_single_" + field] = mods["single_" + field]
        return prepared

    @staticmethod
    def flow_map_text_step(prepared: dict[str, Tensor], step: int) -> dict[str, Tensor]:
        """Step `step`'s text-stream slice of prepare_flow_map_text's result, as forward kwargs."""
        tokens = prepared["_flux3_text_steps_tokens"]
        steps = prepared["_flux3_text_steps_single_shift"].shape[1]
        length = tokens.shape[1] // steps
        out = {"_flux3_text_step_tokens": tokens[:, step * length : (step + 1) * length]}
        # Keep each single_* row as a broadcast [1, 1, D] row (like _packed_stream's compact
        # single-segment stream), so forward takes the [1, 2, D] table + select path for the
        # single blocks; where it cannot, forward's rows() expands it to the text length.
        for field in ("shift", "scale", "gate"):
            rows = prepared["_flux3_text_steps_single_" + field][:, step : step + 1]
            if not FLUX3_TEXT_STEP_BROADCAST:
                rows = rows.expand(-1, length, -1)
            out["_flux3_text_step_single_" + field] = rows
        return out

    def prepare_flow_map_image(
        self, image_timesteps: Tensor, targets: Tensor, *, layout: Layout | None = None, **inputs
    ) -> dict[str, Tensor] | None:
        """Run a flow-map image stream's per-step M=1 vec GEMVs once for every step.

        The image vec of step i is time_in(emb(image_timesteps[i])) + ref_time_in(emb(
        targets[i])); its early/single modulations and the final layer's adaLN row depend on
        nothing else. This runs them at M=N once per request instead of N times at M=1.

        Returns [N, 1, 1, 3D] early/single and [N, 1, 1, 2D] final rows (flow_map_image_step
        hands forward step i's [1, 1, *] row, laid out like the per-step module outputs), or
        None when the per-step path must run (non-flow-map, sequence parallel, conditioning
        image streams, non-uniform timesteps, more than one image segment, no broadcast adaLN
        rows, or TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH=0).
        """
        if (
            not self.flow_map
            or torch.is_grad_enabled()
            or self.sharder.is_active
            or set(self.final_layer) != {"image"}
            or os.environ.get("TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH", "1") == "0"
        ):
            return None
        inputs, layout, _ = self._packed_inputs(inputs, layout)
        if not inputs.get("_flux3_uniform_timesteps", False):
            return None
        if any(name.endswith("_cond") and any(lengths) for name, lengths in layout):
            return None
        image_lengths = dict(layout).get("image", ())
        if len(image_lengths) != 1 or not image_lengths[0]:
            return None
        if (
            image_timesteps.ndim != 1
            or not image_timesteps.shape[0]
            or targets.shape != image_timesteps.shape
        ):
            return None
        tokens = inputs["x_image"]
        if not self._broadcast_adaln_ok(tokens):
            return None
        steps = image_timesteps.shape[0]
        # Same rows as the per-step compact path (_packed_stream + forward's ref), at M=N.
        target = targets.to(inputs["x_text_timesteps"])
        ref = self.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
        ts = image_timesteps.to(inputs["x_image_timesteps"])
        vec = self.time_in(flux3_timestep_embedding(ts, 256, out_dtype=tokens.dtype))
        vec = (vec + ref).unsqueeze(0)  # [1, N, D]
        prepared = {}
        for stage, modules in (
            ("early", self.early_stream_modulations),
            ("single", self.single_stream_modulations),
        ):
            module = modules["image"]
            out = module.lin(module.act(vec))  # [1, N, 3D]
            prepared["_flux3_image_steps_" + stage] = out.reshape(steps, 1, 1, out.shape[-1])
        mod = self.final_layer["image"].adaLN_modulation(vec)  # [1, N, 2D]
        prepared["_flux3_image_steps_final"] = mod.reshape(steps, 1, 1, mod.shape[-1])
        return prepared

    @staticmethod
    def flow_map_image_step(prepared: dict[str, Tensor], step: int) -> dict[str, Tensor]:
        """Step `step`'s [1, 1, *] image rows from prepare_flow_map_image, as forward kwargs."""
        return {
            "_flux3_image_step_" + kind: prepared["_flux3_image_steps_" + kind][step]
            for kind in ("early", "single", "final")
        }

    def _image_step_stream(self, inputs: dict, lengths: tuple[int, ...]) -> dict[str, Tensor]:
        """The image stream state from flow_map_image_step's rows: only emb_in runs per step."""
        embedded = self.emb_in["image"](inputs["x_image"]).unsqueeze(0)
        state = {"tokens": self._shard_packed(embedded, lengths, 1)}
        for stage in ("early", "single"):
            rows = inputs["_flux3_image_step_" + stage].chunk(3, dim=-1)
            for field, value in zip(("shift", "scale", "gate"), rows):
                # One segment: the [1, 1, D] row; the blocks broadcast it.
                state[stage + "_" + field] = value
        state["final_mod"] = inputs["_flux3_image_step_final"]
        return state

    def forward(
        self,
        *,
        layout: Layout | None = None,
        timesteps_ref: Tensor | None = None,
        timestep: Tensor | None = None,
        pag_single_layers: list[int] | None = None,
        pag_strength: float = 1.0,
        pag_samples: tuple[int, ...] = (0,),
        **inputs,
    ) -> dict[str, Tensor]:
        inputs, layout, batch = self._packed_inputs(inputs, layout)
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref is required by, and only by, a flow-map DiT")
        ref = None
        image_rows = self.flow_map and "_flux3_image_step_early" in inputs
        if timesteps_ref is not None:
            num_samples = len(layout[0][1])
            if timesteps_ref.shape != (num_samples,):
                raise ValueError(f"timesteps_ref must have shape ({num_samples},)")
            # Only streams built per step need ref; step-batched text/image rows include it.
            if not (
                "_flux3_text_step_tokens" in inputs
                and image_rows
                and {name for name, lengths in layout if any(lengths)} <= {"text", "image"}
            ):
                target = timesteps_ref.to(inputs["x_text_timesteps"])
                ref = self.ref_time_in(
                    flux3_timestep_embedding(target, 256, out_dtype=target.dtype)
                )
        local_layout = self._local_layout(layout)
        local_lengths = dict(local_layout)
        prepared = (
            inputs if "_flux3_joint_cos" in inputs else self._packed_positions(inputs, layout)
        )
        streams = {}
        for name, lengths in layout:
            if name.endswith("_cond") and not any(lengths):
                continue
            prefix = f"_flux3_{name}_"
            if not self.flow_map and prefix + "tokens" in inputs:
                streams[name] = {
                    field: inputs[prefix + field]
                    for field in ("tokens", "vec", "single_shift", "single_scale", "single_gate")
                }
                continue
            if self.flow_map and name == "text" and "_flux3_text_step_tokens" in inputs:
                # Precomputed for all steps by prepare_flow_map_text; the text vec only feeds
                # final layers, which exist for output modalities (image) alone.
                streams[name] = {
                    field: inputs["_flux3_text_step_" + field]
                    for field in ("tokens", "single_shift", "single_scale", "single_gate")
                }
                continue
            if image_rows and name == "image":
                # Precomputed for all steps by prepare_flow_map_image.
                state = self._image_step_stream(inputs, lengths)
            else:
                state = self._packed_stream(name, inputs, lengths, ref)
            blocks = self.txt_mode_blocks if name == "text" else self.content_mode_blocks[name]
            freqs = _block_freqs(prepared, prefix)
            mod = tuple(state["early_" + field] for field in ("shift", "scale", "gate"))
            state["tokens"] = self._early_adaln_runtime[name].prepare(state["tokens"], *mod[:2])
            for block in blocks:
                state["tokens"] = block(
                    state["tokens"],
                    freqs,
                    mod,
                    sample_bounds=bounds(local_lengths[name]),
                    timestep=timestep,
                )
            streams[name] = state

        chunks, sample_bounds = self._joint_chunks(local_layout)

        def rows(name: str, field: str, start: int, end: int) -> Tensor:
            value = streams[name][field]
            # A broadcast [1, 1, D] modulation row stands for every row of its stream.
            if value.shape[1] == 1:
                return value.expand(-1, end - start, -1)
            return value[:, start:end]

        def joint(field: str) -> Tensor:
            return torch.cat([rows(name, field, start, end) for name, start, end in chunks], dim=1)

        combined = joint("tokens")
        mod_fields = tuple("single_" + field for field in ("shift", "scale", "gate"))
        if len(chunks) == 2 and all(
            streams[name][field].shape[1] == 1 for name, _, _ in chunks for field in mod_fields
        ):
            # [1, 2, D] tables + a select mask; the compiled blocks pick the row per token.
            select = prepared.get("_flux3_joint_select")
            if select is None or select.shape[1] != combined.shape[1]:
                select = self._joint_select(chunks, combined.device)
            mod = tuple(
                torch.cat([streams[name][field] for name, _, _ in chunks], dim=1)
                for field in mod_fields
            ) + (select,)
        else:
            mod = tuple(joint(field) for field in mod_fields)
        freqs = _block_freqs(prepared, "_flux3_joint_")
        combined = self._single_adaln_runtime.prepare(combined, *mod[:2])
        for index, block in enumerate(self.single_blocks):
            combined = block(
                combined,
                freqs,
                mod,
                pag_strength=pag_strength
                if pag_single_layers and index in pag_single_layers
                else 0.0,
                sample_bounds=sample_bounds,
                pag_samples=pag_samples,
                timestep=timestep,
            )
        outputs = {}
        for name in self.final_layer:
            parts, offset = [], 0
            for mode, start, end in chunks:
                if mode == name:
                    part = combined[:, offset : offset + end - start].contiguous()
                    if self.sharder.is_active:
                        part = self.sharder.gather(part, dim=1)
                    parts.append(part)
                offset += end - start
            if len(parts) == 1 and parts[0].shape[0] == 1:
                # No single-part cat copy. Restride the B=1 view so its batch stride is
                # n*D, not the joint length's: compiled final-layer guards then ignore T.
                tokens = parts[0][0].unsqueeze(0)
            else:
                tokens = torch.cat(parts, dim=1)
            if "final_mod" in streams[name]:
                output = self.final_layer[name](tokens, None, mod=streams[name]["final_mod"])[0]
            else:
                vec = self._final_layer_vec(streams[name]["vec"], inputs, dict(layout)[name])
                output = self.final_layer[name](tokens, vec)[0]
            outputs["x_" + name] = (
                output.reshape(batch, -1, output.shape[-1]) if batch is not None else output
            )
        return outputs

    def _final_layer_vec(self, vec: Tensor, inputs: dict, lengths: tuple[int, ...]) -> Tensor:
        """The vec the final layer's adaLN needs: one row when the stream is a single
        compact segment (every row of the expanded vec is that segment's timestep row,
        and the final layer broadcasts it), else the per-token vec unchanged."""
        if (
            inputs.get("_flux3_uniform_timesteps", False)
            and not self.sharder.is_active
            and sum(1 for length in lengths if length) == 1
        ):
            return vec[:, :1]
        return vec

    def load_weights(self, weights: dict) -> None:
        fuse_qkv = _use_fused_qkv(self.model_config)
        weights = _remap_checkpoint_keys(weights, fuse_qkv=fuse_qkv)
        if _use_fused_proj_out(self.model_config):
            weights = _fuse_out_proj_weights(weights)

        params_map = {}
        if fuse_qkv:
            # Checkpoint stores q/k/v separately; fuse into qkv_proj.
            params_map["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]
        tp_size = getattr(self.model_config.mapping, "tp_size", 1)
        if not _use_fused_proj_out(self.model_config):
            # GatedMLP.gate_up_proj always uses FUSED_GATE_UP_LINEAR mode, so
            # split mlp_in.weight [gate|up] into ff.gate_up_proj.{gate,up}.
            weights = _split_mlp_in_weights(weights)
            params_map["gate_up_proj"] = [f"gate_up_proj.{role}" for role in GATE_UP_ROLES]
        elif tp_size > 1:
            # mlp_in stays column-parallel FUSED_GATE_UP_LINEAR under TP even
            # when proj_out is fused (see _gated_mlp_in_tp_kwargs).
            weights = _split_mlp_in_weights_for_fused_proj_out(weights)
            params_map["mlp_in"] = [f"mlp_in.{role}" for role in GATE_UP_ROLES]

        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)
        managed_prefixes: set = set()

        for name, module in self.named_modules():
            if any(name.startswith(p) for p in managed_prefixes):
                continue

            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

            if isinstance(module, FluxJointAttnMLPProj):
                managed_prefixes.add(name + ".")
                module_weights = loader.filter_weights(name, weights)
                module.load_weights(module_weights, loader)
                continue

            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    if type(module.quant_method) is NVFP4LinearMethod and name.startswith(
                        ("content_mode_blocks.image.", "single_blocks.")
                    ):
                        module._flux3_nvfp4_missing_input_scale = all(
                            "input_scale" not in shard for shard in weight_dicts
                        )
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                module_weights = loader.filter_weights(name, weights)
                target_dtype = self.model_config.torch_dtype
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(target_dtype))

    def post_load_weights(self) -> None:
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
                if type(module.quant_method) is FP8RowwiseLinearMethod:
                    module.quant_method = Flux3FP8RowwiseLinearMethod()
        self.prepare_primsts_weights()

    def prepare_primsts_weights(self) -> None:
        """Refresh image/joint PrimsTS layouts after loading or merging LoRA weights."""
        quant = self.model_config.quant_config
        if quant is not None and quant.quant_algo == QuantAlgo.NVFP4:
            # PRECISION: load-time NVFP4 W4A4 for the selected BF16 checkpoint-excluded blocks
            # (TRTLLM_FLUX3_EXCLUDED_NVFP4), so they take the NVFP4 block route below.
            named = (
                [
                    (f"content_mode_blocks.image.{i}", block)
                    for i, block in enumerate(self.content_mode_blocks["image"])
                ]
                if "image" in self.content_mode_blocks
                else []
            )
            named.extend((f"single_blocks.{i}", block) for i, block in enumerate(self.single_blocks))
            knob, converted = convert_excluded_blocks_to_nvfp4(named, quant)
            if converted:
                logger.info(
                    "FLUX.3 load-time NVFP4 W4A4 (precision) for BF16 checkpoint-excluded "
                    f"blocks (TRTLLM_FLUX3_EXCLUDED_NVFP4={knob}): {converted}"
                )
        if "image" in self.content_mode_blocks:
            for block in self.content_mode_blocks["image"]:
                prepare_fp8_block(block)
                prepare_nvfp4_block(block)
        for block in self.single_blocks:
            prepare_fp8_block(block)
            prepare_nvfp4_block(block)
        blocks = (
            [
                (f"content_mode_blocks.image.{i}", block)
                for i, block in enumerate(self.content_mode_blocks["image"])
            ]
            if "image" in self.content_mode_blocks
            else []
        )
        blocks.extend((f"single_blocks.{i}", block) for i, block in enumerate(self.single_blocks))
        verify_nvfp4_coverage(blocks)
        fused_adaln = prepare_adaln_nvfp4(blocks, Flux3SingleStreamBlock, Flux3ModeBlock)
        if fused_adaln:
            logger.info(f"FLUX.3 fused LN+adaLN+NVFP4 quantize (exact): {fused_adaln}")
        quant = self.model_config.quant_config
        if quant is not None and quant.quant_algo == QuantAlgo.NVFP4:
            # BF16 checkpoint-excluded image/joint blocks run with the fp8r recipe on SM120.
            switched = []
            for name, block in blocks:
                target = getattr(block, "_orig_mod", block)
                enabled = prepare_sm120_fp8_excluded_block(target)
                _set_fp8r_block_class(target, enabled)
                if enabled:
                    switched.append(name)
            if switched:
                logger.info(
                    "FLUX.3 SM120 FP8R for BF16 checkpoint-excluded blocks "
                    "(QKV, attn_out, MLP-up+SwiGLU, MLP-down; per-channel weights "
                    f"bitwise equal to variants/fp8r): {switched}"
                )
        paired = [
            name
            for name, block in blocks
            if prepare_sm120_tail_pair(getattr(block, "_orig_mod", block))
        ]
        if paired:
            logger.info(f"FLUX.3 SM120 down || attn_out two-stream tail pair: {paired}")
        from tensorrt_llm._torch.visual_gen.models.flux3.flux3_adaln_fp8r import (
            prepare_fp8r_adaln_quant,
        )

        fused_heads = prepare_fp8r_adaln_quant(blocks)
        if fused_heads:
            logger.info(f"FLUX.3 FP8R fused LN+adaLN+E4M3 quantize head: {fused_heads}")
