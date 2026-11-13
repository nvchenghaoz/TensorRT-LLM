# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""MiniMax-H3 packed audio-video diffusion transformer.

The architecture follows the Diffusers implementations while using TRT-LLM's
linear, attention, normalization, and dynamic-weight-loading modules
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from tqdm import tqdm

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.models.hf_parameter_utils import get_parameter_device
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_rope import (
    apply_minimax_h3_qk_norm_rope_bf16,
)
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .fused_adaln import h3_adaln_supported, h3_gate_res_norm_mod, h3_mod_out, h3_norm_mod
from .vsa import get_h3_vsa_context, h3_vsa_attention

MINIMAX_H3_MODALITY_NUM = 3


@dataclass
class MiniMaxH3TransformerOutput:
    """Velocity predictions for packed video and audio rows."""

    sample: torch.Tensor
    audio_sample: torch.Tensor


@dataclass
class MiniMaxH3StaticContext:
    """Lossless request-static transformer inputs reusable across denoise steps."""

    text_embeds: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]

    @property
    def sequence_length(self) -> int:
        """Return the packed sequence length represented by the cached RoPE."""

        return self.rotary_emb[0].shape[0]


def apply_minimax_h3_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply MiniMax-H3's partial split-half RoPE to ``[B, S, H, D]`` (eager path)."""

    rotary_dim = cos.shape[-1]
    if rotary_dim > hidden_states.shape[-1] or rotary_dim % 2:
        raise ValueError(
            f"Invalid rotary dimension {rotary_dim} for head dimension {hidden_states.shape[-1]}."
        )

    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


def _dense_attention_model_config(model_config: DiffusionModelConfig) -> DiffusionModelConfig:
    """Return a shallow config copy with VSA disabled for non-video attention."""

    sparse_config = getattr(model_config.attention, "sparse_attention_config", None)
    if sparse_config is None or getattr(sparse_config, "algorithm", None) != "vsa":
        return model_config

    dense_config = model_config.model_copy(deep=False)
    dense_config.attention = model_config.attention.model_copy(deep=True)
    dense_config.attention.sparse_attention_config = None
    return dense_config


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis RoPE over packed ``(time, height, width)`` coordinates."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                "position_ids must have shape [sequence_length, 3], got "
                f"{list(position_ids.shape)}."
            )
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3Attention(Attention):
    """Full non-causal attention with per-head RMSNorm and partial 3D RoPE."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
        module_name: str,
    ) -> None:
        vgm = model_config.visual_gen_mapping
        ulysses_size = vgm.ulysses_size if vgm is not None else 1
        # With ulysses_size > 1 and parallel.async_ulysses, use separate Q/K/V projections so
        # each one is pushed to the peers while the next is computed (forward_async).
        use_async_ulysses = bool(ulysses_size > 1 and model_config.parallel.async_ulysses)
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_mode=QKVMode.SEPARATE_QKV if use_async_ulysses else QKVMode.FUSE_QKV,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=qk_norm_eps,
            bias=False,
            # H3 rotates a leading split-half region and passes the tail through, which
            # the shared fused kernel does not implement; the flag drives H3's own kernel
            # (fused_rope.py) through the override of forward() below.
            fuse_qk_norm_rope=True,
            interleave=False,
            config=_dense_attention_model_config(model_config),
            layer_idx=layer_idx,
            module_name=module_name,
            async_ulysses=use_async_ulysses,
            separate_qkv_is_self_attention=True,
        )
        self.use_async_ulysses = use_async_ulysses
        sparse = model_config.attention.sparse_attention_config
        self.to_gate_compress = None
        self._vsa_backend = model_config.attention.backend
        if sparse is not None and sparse.algorithm == "vsa":
            if self._vsa_backend not in ("TRTLLM", "CUTEDSL"):
                raise ValueError("MiniMax-H3 VSA requires the TRTLLM or CUTEDSL backend.")
            if model_config.mapping.world_size != 1:
                raise ValueError("MiniMax-H3 VSA currently supports a single GPU.")
            self.to_gate_compress = Linear(
                hidden_size,
                num_attention_heads * attention_head_dim,
                bias=False,
                dtype=model_config.torch_dtype,
                mapping=model_config.mapping,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )

    def _use_fused_qk_norm_rope(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        """Decide between the fused per-head RMSNorm + RoPE kernel and the eager path.

        Only what selects the path is checked here: the per-instance flag, the packed-QKV
        layout the kernel reads (equal Q/K head counts, head dimension 128), an un-sharded
        plain RMSNorm with BF16 weights, CUDA BF16 activations, a rotary width that is a
        multiple of 32 and no autograd. ``apply_minimax_h3_qk_norm_rope_bf16`` raises on
        anything else.
        """
        if not self.fuse_qk_norm_rope or not self.qk_norm or self.qkv_mode != QKVMode.FUSE_QKV:
            return False
        if self.local_num_attention_heads != self.local_num_key_value_heads or self.head_dim != 128:
            return False
        for norm in (self.norm_q, self.norm_k):
            if norm.allreduce is not None or norm.use_gemma or norm.weight.dtype != torch.bfloat16:
                return False
        cos, sin = rotary_emb
        if (
            not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 3
            or cos.shape != (hidden_states.shape[1], cos.shape[-1])
            or sin.shape != cos.shape
            or cos.shape[-1] % 32
            or cos.shape[-1] > self.head_dim
        ):
            return False
        if torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (hidden_states, cos, sin, self.norm_q.weight, self.norm_k.weight)
        ):
            return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
        project_out: bool = True,
    ) -> torch.Tensor:
        """``project_out=False`` returns the attention output before ``to_out`` so the
        caller can run the row-parallel output projection (and its all-reduce) itself."""
        if self.use_async_ulysses:
            return self.forward_async(
                hidden_states,
                rotary_emb,
                key_padding_mask=key_padding_mask,
                timestep=timestep,
                valid_seq_len=valid_seq_len,
                project_out=project_out,
            )
        batch_size, sequence_length = hidden_states.shape[:2]
        if rotary_emb is not None and self._use_fused_qk_norm_rope(hidden_states, rotary_emb):
            # One launch normalizes and rotates Q and K from the packed projection
            # output; V stays a view of the same buffer.
            qkv = self.qkv_proj(hidden_states)
            value = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)[2]
            cos, sin = rotary_emb
            query, key = apply_minimax_h3_qk_norm_rope_bf16(
                qkv,
                self.norm_q.weight,
                self.norm_k.weight,
                cos,
                sin,
                self.norm_q.variance_epsilon,
                self.local_num_attention_heads,
                self.head_dim,
            )
        else:
            query, key, value = self.get_qkv(hidden_states)
            query = query.view(
                batch_size, sequence_length, self.local_num_attention_heads, self.head_dim
            )
            key = key.view(
                batch_size, sequence_length, self.local_num_key_value_heads, self.head_dim
            )
            query, key = self.apply_qk_norm(query, key)
            if rotary_emb is not None:
                query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
                key = apply_minimax_h3_rotary_emb(key, *rotary_emb)
            query = query.flatten(2)
            key = key.flatten(2)

        if self.to_gate_compress is not None:
            query = query.view(
                batch_size, sequence_length, self.local_num_attention_heads, self.head_dim
            )
            key = key.view_as(query)
            metadata = get_h3_vsa_context()
            gate = self.to_gate_compress(hidden_states).view_as(query)
            output = h3_vsa_attention(
                query,
                key,
                value.view_as(query),
                gate,
                metadata,
                self._attn_impl,
                backend=self._vsa_backend,
            )
            output = output.flatten(2)
            return self.to_out[0](output) if project_out else output

        hidden_states = self._attn_impl(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            discard_padded_queries=True,
            valid_seq_len=valid_seq_len,
            timestep=timestep,
        )
        return self.to_out[0](hidden_states) if project_out else hidden_states

    def forward_async(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
        project_out: bool = True,
    ) -> torch.Tensor:
        """``forward`` with each projection pushed to the peers as soon as it is
        computed: V, then Q and K with their per-head norm and RoPE, run on the
        compute stream while the previous tensor's push runs on the side stream
        (``UlyssesAttention.forward_async``)."""
        batch_size, sequence_length = hidden_states.shape[:2]
        head_dim = self.head_dim

        def project(linear: Linear, heads: int, norm: RMSNorm) -> torch.Tensor:
            states = norm(linear(hidden_states).view(batch_size, sequence_length, heads, head_dim))
            if rotary_emb is not None:
                states = apply_minimax_h3_rotary_emb(states, *rotary_emb)
            return states

        def compute_q() -> torch.Tensor:
            return project(self.to_q, self.local_num_attention_heads, self.norm_q)

        def compute_k() -> torch.Tensor:
            return project(self.to_k, self.local_num_key_value_heads, self.norm_k)

        def compute_v() -> torch.Tensor:
            return self.to_v(hidden_states).view(
                batch_size, sequence_length, self.local_num_key_value_heads, head_dim
            )

        out_4d = self.attn.forward_async(
            compute_q,
            compute_k,
            compute_v,
            key_padding_mask=key_padding_mask,
            discard_padded_queries=True,
            valid_seq_len=valid_seq_len,
            timestep=timestep,
        )
        batch_size, local_seq_len = out_4d.shape[:2]
        output = out_4d.reshape(batch_size, local_seq_len, -1)
        return self.to_out[0](output) if project_out else output


def _norm_2d(norm: RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply an RMSNorm over a packed ``[B, S, D]`` tensor.

    FlashInfer reads a 3-D input as ``(batch, heads, head_dim)`` and dispatches
    the per-head QK-norm kernel, which runs one warp per row.  Flattening to
    ``[B * S, D]`` keeps the CTA-per-row kernel, matching how the feed-forward
    call sites already reshape.  The reduction is over the last dim either way,
    so the result is unchanged.
    """
    shape = hidden_states.shape
    return norm(hidden_states.reshape(-1, shape[-1])).view(shape)


def _pad_tensor_dim(
    tensor: torch.Tensor,
    dim: int,
    pad: int,
    value: float | int | bool = 0,
) -> torch.Tensor:
    if pad == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad
    return torch.cat([tensor, tensor.new_full(pad_shape, value)], dim=dim)


def _padding_to_multiple(seq_len: int, multiple: int) -> int:
    if multiple <= 1:
        return 0
    return (multiple - seq_len % multiple) % multiple


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Produce six AdaLN vectors for every ``(timestep, modality)`` pair."""

    def __init__(
        self,
        time_embed_dim: int,
        hidden_size: int,
        model_config: DiffusionModelConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = Linear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=model_config.quant_config,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            tensor_parallel_mode=None,
            reduce_output=False,
        )

    def table(self, temb: torch.Tensor) -> torch.Tensor:
        """Return the modulation table ``[n_t * 3, 6 * hidden_size]``.

        One row per ``(timestep, modality)`` pair; the six column blocks are
        shift, scale and gate for attention, then shift, scale and gate for the MLP.
        """
        temb = self.linear(F.silu(temb).to(self.linear.dtype))
        return temb.view(-1, 6 * self.hidden_size)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.table(temb).chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final per-timestep shift/scale RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        eps: float,
        model_config: DiffusionModelConfig,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            hidden_size=hidden_size, eps=eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.linear = Linear(
            time_embed_dim,
            2 * hidden_size,
            bias=True,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=model_config.quant_config,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            tensor_parallel_mode=None,
            reduce_output=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(self.linear.dtype)).chunk(2, dim=-1)
        hidden_states = _norm_2d(self.norm, hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )

    def forward_to(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """``forward(...).to(out_dtype)`` for BF16 input, bitwise equal, in one modulation kernel.

        The kernel applies eager's three bf16 roundings and writes ``out_dtype`` directly, so the
        FP32 output projections then skip their casts.
        """
        table = self.linear(F.silu(temb).to(self.linear.dtype))
        hidden_states = _norm_2d(self.norm, hidden_states)
        return h3_mod_out(hidden_states, table, timestep_indices, 1, 0, out_dtype)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
        fuse_bf16_gate_up_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=_dense_attention_model_config(model_config),
            layer_idx=layer_idx,
            module_name=f"token_refiner.refiner_blocks.{layer_idx}.attn",
        )
        self.norm2 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=model_config.mapping.tp_size > 1,
            fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            _norm_2d(self.norm1, hidden_states),
            key_padding_mask=key_padding_mask,
            valid_seq_len=valid_seq_len,
        )
        residual = hidden_states
        hidden_states = _norm_2d(self.norm2, hidden_states)
        hidden_states = self.ff(hidden_states.reshape(-1, hidden_states.shape[-1])).reshape_as(
            hidden_states
        )
        return residual + hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        model_config: DiffusionModelConfig,
        fuse_bf16_gate_up_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self._attn_backend = model_config.attention.backend.upper()
        self._supports_key_padding_mask = self._attn_backend in (
            "VANILLA",
            "FA4",
            "TRTLLM",
            "FLASHINFER",
        )
        self.sharder = SequenceSharder.from_vgm(
            model_config.visual_gen_mapping,
            num_attention_heads=num_attention_heads,
        )
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                    fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=final_norm_eps,
            dtype=model_config.torch_dtype,
            has_weights=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        pad = _padding_to_multiple(seq_len, self.sharder.size)
        key_padding_mask = None
        if pad:
            if not self._supports_key_padding_mask:
                raise NotImplementedError(
                    "Padded token-refiner sequences require key_padding_mask support; "
                    "use VANILLA, FA4, TRTLLM or FLASHINFER."
                )
            if self._attn_backend != "TRTLLM":
                # Attention caches use the mask version to detect changes.
                with torch.inference_mode(False):
                    valid = hidden_states.new_ones(
                        (hidden_states.shape[0], seq_len), dtype=torch.bool
                    )
                    key_padding_mask = _pad_tensor_dim(valid, 1, pad, False)
            hidden_states = _pad_tensor_dim(hidden_states, 1, pad)
        hidden_states = self.sharder.shard(hidden_states, dim=1)
        for block in self.refiner_blocks:
            hidden_states = block(
                hidden_states,
                key_padding_mask=key_padding_mask,
                valid_seq_len=seq_len if pad else None,
            )
        hidden_states = _norm_2d(self.final_norm, hidden_states)
        return self.sharder.gather(hidden_states, dim=1, unpad_to=seq_len)


def _tp_overlap_num_chunks() -> int:
    """Row chunks for the TP all-reduce/compute overlap in the transformer blocks.

    ``TLLM_H3_TP_OVERLAP_CHUNKS`` overrides the default of 4; 0 or 1 restores the
    serialized path (to_out -> all-reduce -> ... -> down -> all-reduce).
    """
    try:
        return max(0, int(os.environ.get("TLLM_H3_TP_OVERLAP_CHUNKS", "4")))
    except ValueError:
        return 4


def _row_chunks(
    rows: int,
    num_chunks: int,
    align: int = 128,
    weights: Optional[tuple[float, ...]] = None,
) -> list[tuple[int, int]]:
    """Split ``rows`` into at most ``num_chunks`` ranges whose starts are ``align``-aligned.

    ``weights`` (one per chunk) gives tapered chunk sizes: chunk i gets about
    ``rows * weights[i] / sum(weights)`` rows, rounded to ``align``; the last chunk takes the rest.
    """
    if weights:
        total = float(sum(weights))
        bounds, start, acc = [], 0, 0.0
        for w in weights[:-1]:
            acc += w
            end = min(rows, int(round(rows * acc / total / align)) * align)
            if end > start:
                bounds.append((start, end))
                start = end
        if start < rows:
            bounds.append((start, rows))
        return bounds
    step = -(-rows // num_chunks)
    step = -(-step // align) * align
    return [(start, min(rows, start + step)) for start in range(0, rows, step)]


# Dedicated NCCL communicator for the chunked TP all-reduces (read once at model build).
# Its NCCL stream runs at high priority, so each chunk's all-reduce kernel gets SMs as soon as its
# GEMM finishes instead of queueing behind the next chunks' GEMMs (2-GPU PCIe pair: ~0.8 ms median
# launch-to-start delay per all-reduce and ~7 ms exposed per block on the shared normal-priority
# group, ~2.7% of the block). _TP_OVERLAP_PG_CTAS > 0 pins NCCL's CTA (channel) count through the
# communicator config (0 = NCCL's default; 8 measured no better). Both off keep the shared TP group.
_TP_OVERLAP_PG_CTAS = 0
_TP_OVERLAP_PG_HIGH_PRIORITY = True
# Per-chunk row weights for the chunked overlap (None = uniform _tp_overlap_num_chunks() chunks).
_TP_OVERLAP_CHUNK_WEIGHTS: Optional[tuple[float, ...]] = None
_H3_TP_GROUPS: dict = {}


def h3_dedicated_tp_group(
    base_group_name: str, ctas: int = 0, high_priority: bool = False
) -> Optional["dist.ProcessGroup"]:
    """A dedicated NCCL process group over the ranks of ``base_group_name``, built once per setting.

    The communicator config pins NCCL's CTA (channel) count to ``ctas`` when > 0 and can put its
    stream at high priority, without touching any other communicator or the NCCL env. ``new_group``
    is collective over the world, so the group is only built when the TP group spans the whole
    world (every rank then asks for the same ranks in the same order); otherwise None (callers
    keep the shared group). The all-reduce math is unchanged: a two-rank sum per element.
    """
    key = (base_group_name, int(ctas), bool(high_priority))
    if key in _H3_TP_GROUPS:
        return _H3_TP_GROUPS[key]
    pg = None
    if dist.is_available() and dist.is_initialized() and dist.is_nccl_available():
        base = dist.distributed_c10d._resolve_process_group(base_group_name)
        ranks = sorted(dist.get_process_group_ranks(base))
        if len(ranks) == dist.get_world_size():
            opts = dist.ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = bool(high_priority)
            if ctas > 0:
                opts.config.min_ctas = int(ctas)
                opts.config.max_ctas = int(ctas)
            pg = dist.new_group(ranks, backend="nccl", pg_options=opts)
            from tensorrt_llm.logger import logger

            logger.info(
                f"[rank {dist.get_rank()}] MiniMax-H3 TP overlap: dedicated NCCL group over ranks "
                f"{ranks}, min_ctas=max_ctas={ctas or 'default'}, high_priority_stream="
                f"{bool(high_priority)}; NCCL_MIN_NCHANNELS="
                f"{os.environ.get('NCCL_MIN_NCHANNELS')} NCCL_MAX_NCHANNELS="
                f"{os.environ.get('NCCL_MAX_NCHANNELS')}"
            )
    _H3_TP_GROUPS[key] = pg
    return pg


def _linear_quantized(linear: Linear) -> bool:
    """Quant mode from quant_config (the weights may not exist yet at build time)."""
    quant_config = getattr(linear, "quant_config", None)
    return quant_config is not None and quant_config.layer_quant_mode.has_any_quant(
        exclude_kv_cache=True
    )


def _h3_sm120_cublas_pin(model_config: DiffusionModelConfig) -> bool:
    """Whether the DiT block Linears use trtllm::cublas_mm's SM120 BF16 tile pins.

    ``TLLM_H3_SM120_CUBLAS_PIN=0`` keeps F.linear.
    """
    if os.environ.get("TLLM_H3_SM120_CUBLAS_PIN", "1") == "0":
        return False
    if model_config.torch_dtype != torch.bfloat16 or not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import get_sm_version

    return get_sm_version() == 120


def _pad_rotary_emb(
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rotary_emb
    return (
        _pad_tensor_dim(cos, 0, pad, 1.0),
        _pad_tensor_dim(sin, 0, pad, 0.0),
    )


class MiniMaxH3TransformerBlock(nn.Module):
    # TP all-reduce/compute overlap (see setup_tp_overlap and _post_attn_overlap); off until
    # the owning model enables it.
    _tp_overlap_chunks = 0
    _tp_overlap_weights = None
    _tp_overlap_group_name = None
    _tp_overlap_pg = None

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
        fuse_bf16_gate_up_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"transformer_blocks.{layer_idx}.attn",
        )
        self.norm2 = RMSNorm(
            hidden_size=hidden_size, eps=norm_eps, dtype=model_config.torch_dtype, has_weights=True
        )
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=model_config.mapping.tp_size > 1,
            fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
        )
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim,
            hidden_size=hidden_size,
            model_config=model_config,
        )
        # Fused RMSNorm + gathered modulation (and the gated residual before norm2).
        self._fused_adaln = h3_adaln_supported(hidden_size, model_config.torch_dtype)
        # The row-parallel outputs these all-reduce are private GEMM results: reduce them in place
        # (no [M, D] input clone per all-reduce on the ProcessGroup NCCL path).
        for linear in (self.attn.to_out[0], self.ff.down_proj):
            if getattr(linear, "all_reduce", None) is not None:
                linear.all_reduce.inplace_private_input = True
        # SM120: run the four BF16 DiT Linears through trtllm::cublas_mm, whose SM120 LUT
        # (cublasScaledMMLut.h sm120_bf16_algo_list) pins split-K-free nvjet tiles that are
        # bitwise identical to the F.linear heuristic kernel but faster at H3's per-rank shapes.
        # Shapes without a LUT entry fall back to the cuBLASLt heuristic. Output buffers stay
        # ordinary private allocations (no NCCL window: H3 runs with MPI disabled).
        if _h3_sm120_cublas_pin(model_config):
            for linear in (
                getattr(self.attn, "qkv_proj", None),
                self.attn.to_out[0],
                getattr(self.ff, "gate_up_proj", None),
                self.ff.down_proj,
            ):
                if isinstance(linear, Linear) and not _linear_quantized(linear):
                    linear.use_custom_cublas_mm = True
            if layer_idx == 0:
                from tensorrt_llm.logger import logger

                logger.info(
                    "[MiniMax-H3 SM120 cublas pin] block 0: use_custom_cublas_mm qkv/to_out/"
                    "gate_up/down = "
                    f"{[getattr(m, 'use_custom_cublas_mm', None) for m in (getattr(self.attn, 'qkv_proj', None), self.attn.to_out[0], getattr(self.ff, 'gate_up_proj', None), self.ff.down_proj)]}"
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self._fused_adaln and hidden_states.dtype == torch.bfloat16:
            return self._forward_fused_adaln(
                hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len,
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = _norm_2d(self.norm1, hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * self.attn(
            norm_hidden_states,
            rotary_emb,
            key_padding_mask,
            timestep,
            valid_seq_len,
        )

        residual = hidden_states
        norm_hidden_states = _norm_2d(self.norm2, hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        ff_output = self.ff(
            norm_hidden_states.reshape(-1, norm_hidden_states.shape[-1])
        ).reshape_as(norm_hidden_states)
        return residual + gate_mlp.index_select(0, adaln_indices) * ff_output

    def _forward_fused_adaln(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        valid_seq_len: Optional[int],
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        mod = self.adaln_proj.table(temb)
        norm_hidden_states = h3_norm_mod(
            hidden_states, self.norm1.weight, mod, adaln_indices, self.norm1.variance_epsilon, 1, 0
        )
        if self._tp_overlap_chunks > 1 and hidden_states.shape[0] == 1:
            attn_hidden = self.attn(
                norm_hidden_states,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len,
                project_out=False,
            )
            residual_chunks, ff_chunks = self._post_attn_overlap(
                attn_hidden, hidden_states, mod, adaln_indices
            )
            gate_mlp = mod[:, 5 * hidden_size : 6 * hidden_size]
            return torch.cat(residual_chunks, dim=1) + gate_mlp.index_select(
                0, adaln_indices
            ) * torch.cat(ff_chunks, dim=1)
        attn_output = self.attn(
            norm_hidden_states, rotary_emb, key_padding_mask, timestep, valid_seq_len
        )
        hidden_states, norm_hidden_states = h3_gate_res_norm_mod(
            hidden_states,
            attn_output,
            self.norm2.weight,
            mod,
            adaln_indices,
            self.norm2.variance_epsilon,
            2,
            4,
            3,
        )
        ff_output = self.ff(
            norm_hidden_states.reshape(-1, norm_hidden_states.shape[-1])
        ).reshape_as(norm_hidden_states)
        gate_mlp = mod[:, 5 * hidden_size : 6 * hidden_size]
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * ff_output

    def setup_tp_overlap(self, model_config: DiffusionModelConfig) -> None:
        """Enable the token-chunked TP all-reduce/compute overlap when it keeps the numerics.

        Needs tensor parallelism with both row-parallel projections (attention ``to_out`` and
        ``ff.down_proj``) reducing through the same c10d process group (TLLM_DISABLE_MPI), and
        the fused AdaLN path. Quantized Linears are excluded: a dynamic per-tensor activation
        scale (FP8/NVFP4 amax over all rows) would become a per-chunk scale.
        """
        self._tp_overlap_chunks = 0
        self._tp_overlap_weights = None
        self._tp_overlap_group_name = None
        self._tp_overlap_pg = None
        to_out = self.attn.to_out[0]
        to_out_ar = to_out.all_reduce
        down_ar = self.ff.down_proj.all_reduce
        num_chunks = _tp_overlap_num_chunks()
        # Read the quant mode from quant_config: Linear.has_any_quant asserts the weights
        # exist, and the serving loader builds the model with skip_create_weights_in_init.
        def _quantized(linear: Linear) -> bool:
            quant_config = getattr(linear, "quant_config", None)
            return quant_config is not None and quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True
            )

        unquantized = not any(
            _quantized(linear)
            for linear in [to_out] + [m for m in self.ff.modules() if isinstance(m, Linear)]
        )
        if (
            num_chunks > 1
            and unquantized
            and self._fused_adaln
            and model_config.mapping.tp_size > 1
            and to_out_ar is not None
            and down_ar is not None
            and getattr(to_out_ar, "_group_name", None) is not None
            and to_out_ar._group_name == getattr(down_ar, "_group_name", None)
        ):
            weights = _TP_OVERLAP_CHUNK_WEIGHTS
            self._tp_overlap_chunks = len(weights) if weights else num_chunks
            self._tp_overlap_weights = tuple(weights) if weights else None
            self._tp_overlap_group_name = to_out_ar._group_name
            if _TP_OVERLAP_PG_CTAS > 0 or _TP_OVERLAP_PG_HIGH_PRIORITY:
                self._tp_overlap_pg = h3_dedicated_tp_group(
                    to_out_ar._group_name, _TP_OVERLAP_PG_CTAS, _TP_OVERLAP_PG_HIGH_PRIORITY
                )
            if "logged" not in _H3_TP_GROUPS:
                from tensorrt_llm.logger import logger

                _H3_TP_GROUPS["logged"] = True
                logger.info(
                    f"[rank {dist.get_rank() if dist.is_initialized() else 0}] MiniMax-H3 TP "
                    f"overlap: chunks {self._tp_overlap_chunks} weights {self._tp_overlap_weights}, "
                    f"dedicated group {self._tp_overlap_pg is not None} (ctas "
                    f"{_TP_OVERLAP_PG_CTAS}, high priority {_TP_OVERLAP_PG_HIGH_PRIORITY})"
                )

    @torch.compiler.disable
    def _post_attn_overlap(
        self,
        attn_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        mod: torch.Tensor,
        adaln_indices: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Token-chunked post-attention half with the TP all-reduces off the critical path.

        Everything after attention is row-local, so the rows are split into chunks:
        every chunk's ``to_out`` partial is all-reduced asynchronously (in place, on the
        process group's NCCL stream) while later chunks' GEMMs run; each chunk's gated
        residual + norm2 + feed-forward waits only for its own reduction, and its
        ``down_proj`` partial is all-reduced asynchronously under the next chunks' work.
        Each row's math is unchanged (same kernels; a two-rank sum is order-free), and
        the all-reduces need no ``clone`` because their inputs are private GEMM outputs.

        Runs eagerly (one graph break per block); the caller's compiled code fuses the
        final gated residual. Returns the per-chunk ``[1, rows, D]`` residual stream after
        attention and the reduced feed-forward outputs.
        """
        pg = self._tp_overlap_pg
        if pg is None:
            pg = dist.distributed_c10d._resolve_process_group(self._tp_overlap_group_name)
            self._tp_overlap_pg = pg
        no_reduce = AllReduceParams(enable_allreduce=False)
        hidden_size = hidden_states.shape[-1]
        rows = hidden_states.shape[1]
        bounds = _row_chunks(rows, self._tp_overlap_chunks, weights=self._tp_overlap_weights)
        attn_rows = attn_hidden.reshape(rows, attn_hidden.shape[-1])
        to_out = self.attn.to_out[0]

        attn_outputs, attn_works = [], []
        for start, end in bounds:
            out = to_out(attn_rows[start:end], all_reduce_params=no_reduce)
            attn_works.append(dist.all_reduce(out, group=pg, async_op=True))
            attn_outputs.append(out)

        residual_chunks, ff_chunks, ff_works = [], [], []
        eps = self.norm2.variance_epsilon
        for i, (start, end) in enumerate(bounds):
            attn_works[i].wait()
            residual, norm_hidden = h3_gate_res_norm_mod(
                hidden_states[:, start:end],
                attn_outputs[i].view(1, end - start, hidden_size),
                self.norm2.weight,
                mod,
                adaln_indices[start:end],
                eps,
                2,
                4,
                3,
            )
            ff_out = self.ff(norm_hidden.view(end - start, hidden_size), final_all_reduce_params=no_reduce)
            ff_works.append(dist.all_reduce(ff_out, group=pg, async_op=True))
            residual_chunks.append(residual)
            ff_chunks.append(ff_out.view(1, end - start, hidden_size))
        for work in ff_works:
            work.wait()
        return residual_chunks, ff_chunks


class MiniMaxH3Transformer3DModel(BaseDiffusionModel):
    """MiniMax-H3's dense single-stream joint audio-video transformer.

    Args:
        model_config: VisualGen configuration containing the checkpoint
            architecture, dtype, distributed mapping, and loading settings.
    """

    def __init__(self, model_config: DiffusionModelConfig) -> None:
        super().__init__(model_config)
        self._attn_backend = getattr(model_config.attention, "backend", "VANILLA")
        # Only VANILLA supports arbitrary padding between packed modalities.
        self._supports_key_padding_mask = self._attn_backend in (
            "VANILLA",
            "FA4",
            "TRTLLM",
            "FLASHINFER",
        )
        quant_algo = model_config.quant_config.quant_algo
        # TODO: Support calibrated checkpoints in load_weights before enabling static quantization.
        if quant_algo is not None and not (
            quant_algo in (QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4)
            and model_config.dynamic_weight_quant
        ):
            raise NotImplementedError(
                "MiniMax-H3 supports BF16, or dynamically quantized FP8 / FP8_BLOCK_SCALES / NVFP4 weights only."
            )
        # The released checkpoint keeps proj_in / audio_proj_in / proj_out /
        # audio_proj_out in FP32 and feeds them FP32 activations.  NVFP4
        # quantizes them like every other Linear (see ``post_load_weights``),
        # but ``fp4_quantize`` only accepts fp16/bf16/e4m3, so those FP32
        # activations would be rejected at the kernel boundary.  Hand those
        # projections the model dtype instead; FP8 has no such restriction and
        # keeps its validated FP32 activations.
        self._projection_act_dtype = (
            model_config.torch_dtype if quant_algo == QuantAlgo.NVFP4 else torch.float32
        )
        # The dynamic loader does not initialize NVFP4's static activation scale.
        if quant_algo == QuantAlgo.NVFP4 and not model_config.force_dynamic_quantization:
            raise NotImplementedError(
                f"MiniMax-H3 {quant_algo.name} requires force_dynamic_quantization=True; "
                "the dynamic loader does not calibrate Linear.input_scale, so the "
                "static activation-scale path would read an uninitialized scale."
            )

        # GatedMLP's BF16 gate/up GEMM with SwiGLU in its epilogue is opt-in
        # because the fused GEMM is slightly slower than cuBLAS at small M. H3's
        # feed-forward sees M = every packed video and audio token of the clip
        # (about 19k at 544x960, 38k at the default 768x1344), where dropping the
        # [M, 2I] intermediate wins, so the BF16 configuration opts in for all
        # blocks here. The token refiner runs once per request on the text
        # tokens, so its cost either way is negligible. Quantized
        # configurations run their own fused GEMMs.
        fuse_bf16_gate_up_swiglu = quant_algo is None

        cfg = model_config.pretrained_config
        num_attention_heads = cfg.num_attention_heads
        attention_head_dim = cfg.attention_head_dim
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        num_refiner_layers = cfg.num_refiner_layers
        ffn_dim = cfg.ffn_dim
        in_channels = cfg.in_channels
        audio_in_channels = cfg.audio_in_channels
        patch_size = tuple(cfg.patch_size)
        text_dim = cfg.text_dim
        freq_dim = cfg.freq_dim
        time_embed_hidden_dim = cfg.time_embed_hidden_dim
        time_embed_dim = cfg.time_embed_dim
        rope_freq_dim = cfg.rope_freq_dim
        rope_theta = float(cfg.rope_theta)
        norm_eps = float(cfg.norm_eps)
        qk_norm_eps = float(cfg.qk_norm_eps)
        final_norm_eps = float(cfg.final_norm_eps)
        self.sharder = SequenceSharder.from_vgm(
            model_config.visual_gen_mapping,
            num_attention_heads=num_attention_heads,
        )

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        # Block-scaled GEMMs require aligned dimensions. Keep the checkpoint's
        # small FP32 media projections unquantized, including their output heads.
        if quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            excluded = list(model_config.quant_config.exclude_modules or [])
            for name, input_dim, output_dim in (
                ("proj_in", video_patch_dim, hidden_size),
                ("audio_proj_in", audio_in_channels, hidden_size),
                ("proj_out", hidden_size, video_patch_dim),
                ("audio_proj_out", hidden_size, audio_in_channels),
            ):
                if input_dim % 128 or output_dim % 128:
                    if name not in excluded:
                        excluded.append(name)
            model_config.quant_config.exclude_modules = excluded
        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        fp32_linear_kwargs = {
            "dtype": torch.float32,
            "mapping": model_config.mapping,
            "quant_config": model_config.quant_config,
            "skip_create_weights_in_init": model_config.skip_create_weights_in_init,
            "force_dynamic_quantization": model_config.force_dynamic_quantization,
            "tensor_parallel_mode": None,
            "reduce_output": False,
        }
        bf16_linear_kwargs = {
            "dtype": model_config.torch_dtype,
            "mapping": model_config.mapping,
            "quant_config": model_config.quant_config,
            "skip_create_weights_in_init": model_config.skip_create_weights_in_init,
            "force_dynamic_quantization": model_config.force_dynamic_quantization,
            "tensor_parallel_mode": None,
            "reduce_output": False,
        }

        self.proj_in = Linear(video_patch_dim, hidden_size, bias=True, **fp32_linear_kwargs)
        self.audio_proj_in = Linear(audio_in_channels, hidden_size, bias=True, **fp32_linear_kwargs)
        self.context_embedder = Linear(text_dim, hidden_size, bias=True, **bf16_linear_kwargs)

        self.time_proj = Timesteps(
            num_channels=freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=time_embed_hidden_dim,
            out_dim=time_embed_dim,
        )
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim, rope_theta)

        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            model_config=model_config,
            fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                    fuse_bf16_gate_up_swiglu=fuse_bf16_gate_up_swiglu,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size,
            time_embed_dim,
            final_norm_eps,
            model_config,
        )
        self.proj_out = Linear(hidden_size, video_patch_dim, bias=True, **fp32_linear_kwargs)
        self.audio_proj_out = Linear(
            hidden_size, audio_in_channels, bias=True, **fp32_linear_kwargs
        )

        self.apply_quant_config_exclude_modules()
        for block in self.transformer_blocks:
            block.setup_tp_overlap(model_config)

    def apply_quant_config_exclude_modules(self) -> None:
        """Drop quantization from Linears the quant config excludes.

        Every Linear is constructed with the model-wide ``quant_config``, so
        without this an excluded module still builds quantized buffers while
        ``DynamicLinearWeightLoader`` hands it a high-precision weight -- an FP8
        buffer left at its default scale of 1.0, or an NVFP4 shape mismatch.

        Reassigning ``quant_config`` alone is not enough. ``Linear.create_weights``
        caches ``quant_method`` and allocates the quantized parameters, and it
        early-returns once ``_weights_created`` is set. With the default
        ``skip_create_weights_in_init=False`` that has already happened by the time
        this runs, so an excluded module would keep its quantized buffers and
        method. Rebuild those modules so the layout matches the no-quant config.
        """
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        no_quant_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                if quant_config.is_module_excluded_from_quantization(name):
                    if getattr(module, "quant_config", None) is not None:
                        module.quant_config = no_quant_config
                        if getattr(module, "_weights_created", False):
                            module._weights_created = False
                            module._parameters.clear()
                            module._buffers.clear()
                            module.create_weights()

    @property
    def device(self) -> torch.device:
        return get_parameter_device(self)

    def prepare_static_context(
        self,
        encoder_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> MiniMaxH3StaticContext:
        """Project/refine text and build RoPE once for an entire request."""

        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.dtype))
        text_embeds = self.token_refiner(text_embeds)
        return MiniMaxH3StaticContext(
            text_embeds=text_embeds,
            rotary_emb=self.rope(position_ids),
        )

    def _validate_layout(
        self,
        sequence_length: int,
        token_tags: torch.Tensor,
        timestep_indices: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        """Range-check the packed layout; the fused AdaLN kernels gather ``mod[idx]`` without bounds checks."""
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "token_tags and timestep_indices must match the packed sequence length."
            )
        if token_tags.numel() and (
            token_tags.min().item() < -1 or token_tags.max().item() >= MINIMAX_H3_MODALITY_NUM
        ):
            raise ValueError("token_tags values must be one of -1, 0, 1, or 2.")
        if timestep_indices.numel() and (
            timestep_indices.min().item() < 0 or timestep_indices.max().item() >= timestep.shape[0]
        ):
            raise ValueError("timestep_indices contains an index outside the timestep table.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        timestep: torch.Tensor,
        conditioning_timesteps: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: Optional[dict[str, object]] = None,
        return_dict: bool = True,
        static_context: Optional[MiniMaxH3StaticContext] = None,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Predict packed velocities using native conditioning and normalized attention time.

        ``conditioning_timesteps`` is the unique H3 time table for AdaLN;
        ``timestep`` is descending noise in [0, 1] for attention/graph scheduling.
        """
        del attention_kwargs
        if position_ids is None:
            if static_context is None:
                raise ValueError("position_ids is required when static_context is not provided.")
            sequence_length = static_context.sequence_length
        else:
            if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
                raise ValueError(
                    f"position_ids must have shape [sequence_length, 3], got {list(position_ids.shape)}."
                )
            sequence_length = position_ids.shape[0]
        self._validate_layout(sequence_length, token_tags, timestep_indices, conditioning_timesteps)

        if static_context is None:
            if encoder_hidden_states is None or position_ids is None:
                raise ValueError(
                    "encoder_hidden_states and position_ids are required to prepare static context."
                )
            static_context = self.prepare_static_context(
                encoder_hidden_states,
                position_ids,
            )

        if static_context.sequence_length != sequence_length:
            raise ValueError("static_context RoPE length must match the packed sequence length.")

        if static_context.text_embeds.shape[1] != text_indices.numel():
            raise ValueError("text_indices must contain one entry for every cached text embedding.")

        video_embeds = self.proj_in(hidden_states.to(self._projection_act_dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self._projection_act_dtype))
        text_embeds = static_context.text_embeds
        packed_hidden_states = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        # Fill the fresh buffer in place (the out-of-place index_copy cloned [1, M, D] three times).
        packed_hidden_states.index_copy_(1, text_indices, text_embeds)
        packed_hidden_states.index_copy_(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed_hidden_states.index_copy_(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(conditioning_timesteps)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
        timestep_indices_for_norm = timestep_indices
        rotary_emb = static_context.rotary_emb
        pad = _padding_to_multiple(sequence_length, self.sharder.size)
        has_existing_padding = bool((token_tags < 0).any())
        valid_seq_len = None
        if self._attn_backend == "TRTLLM" and (pad or has_existing_padding):
            valid_seq_len = sequence_length
            if has_existing_padding:
                valid_seq_len = int((token_tags >= 0).sum().item())
        token_tags_for_mask = token_tags
        if pad:
            packed_hidden_states = _pad_tensor_dim(packed_hidden_states, 1, pad)
            adaln_indices = _pad_tensor_dim(adaln_indices, 0, pad)
            timestep_indices_for_norm = _pad_tensor_dim(timestep_indices, 0, pad)
            token_tags_for_mask = _pad_tensor_dim(token_tags, 0, pad, -1)
            rotary_emb = _pad_rotary_emb(rotary_emb, pad)

        key_padding_mask = None
        if pad or has_existing_padding:
            if not self._supports_key_padding_mask:
                raise NotImplementedError(
                    "Padded packed sequences (negative token_tags) need an attention "
                    "backend that honours key_padding_mask; the "
                    f"{self._attn_backend} backend cannot express it. Use "
                    "VANILLA, FA4, TRTLLM or FLASHINFER."
                )
            # FA4 seqused_k and TRTLLM prefix trimming require trailing padding.
            # Ulysses appends padding in this layout.
            if self._attn_backend in ("FA4", "TRTLLM") and bool(
                ((token_tags_for_mask[:-1] < 0) & (token_tags_for_mask[1:] >= 0)).any()
            ):
                raise NotImplementedError(
                    f"{self._attn_backend} requires trailing packed-row padding; use VANILLA for interior padding."
                )
            if self._attn_backend != "TRTLLM":
                # Attention caches use the mask version to detect changes.
                with torch.inference_mode(False):
                    key_padding_mask = (
                        (token_tags_for_mask >= 0)
                        .unsqueeze(0)
                        .expand(packed_hidden_states.shape[0], -1)
                    )

        padded_sequence_length = packed_hidden_states.shape[1]
        packed_hidden_states = self.sharder.shard(packed_hidden_states, dim=1)
        adaln_indices = self.sharder.shard(adaln_indices, dim=0)
        timestep_indices_for_norm = self.sharder.shard(timestep_indices_for_norm, dim=0)
        rotary_emb = self.sharder.shard_rope(
            rotary_emb,
            seq_len=padded_sequence_length,
            seq_dim=0,
        )

        for block in self.transformer_blocks:
            packed_hidden_states = block(
                packed_hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len=valid_seq_len,
            )

        if (
            packed_hidden_states.dtype == torch.bfloat16
            and packed_hidden_states.is_cuda
            and not self.sharder.is_active
            and not getattr(self.sharder, "token_sharded_tp", False)
        ):
            # No row gather follows (token-sharded TP reports an inactive sharder but still
            # all-gathers rows): write the projections' activation dtype straight from the
            # modulation kernel; the .to() calls below are then no-ops.
            packed_hidden_states = self.norm_out.forward_to(
                packed_hidden_states,
                temb,
                timestep_indices_for_norm,
                self._projection_act_dtype,
            )
        else:
            packed_hidden_states = self.norm_out(
                packed_hidden_states,
                temb,
                timestep_indices_for_norm,
            )
        packed_hidden_states = self.sharder.gather(
            packed_hidden_states,
            dim=1,
            unpad_to=sequence_length,
        )
        video_output = self.proj_out(
            packed_hidden_states.to(self._projection_act_dtype)
        ).index_select(1, video_indices)
        audio_output = self.audio_proj_out(
            packed_hidden_states.to(self._projection_act_dtype)
        ).index_select(1, audio_indices)

        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
        )

    @staticmethod
    def _remap_feed_forward_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        remapped = {}
        for key, value in weights.items():
            if ".ff.net.0.proj." in key:
                prefix, suffix = key.split("net.0.proj.", maxsplit=1)
                # Diffusers' SwiGLU stores ``[up, gate]`` while TRT-LLM's
                # fused GatedMLP consumes ``[gate, up]``.
                up, gate = value.chunk(2, dim=0)
                remapped[f"{prefix}gate.{suffix}"] = gate
                remapped[f"{prefix}up.{suffix}"] = up
            else:
                remapped[key.replace(".ff.net.2.", ".ff.down_proj.")] = value
        return remapped

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load the official converted Diffusers checkpoint layout."""

        weights = self._remap_feed_forward_weights(weights)
        loader = DynamicLinearWeightLoader(
            self.model_config,
            params_map={
                "qkv_proj": ["to_q", "to_k", "to_v"],
                "gate_up_proj": ["gate", "up"],
            },
        )
        device = self.device
        missing = []
        for name, module in tqdm(
            self.named_modules(),
            desc="Loading MiniMax-H3 weights",
        ):
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()
                module.to(device)
            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                required_parameters = ["weight"]
                if module.bias is not None:
                    required_parameters.append("bias")
                missing_parameters = [
                    parameter_name
                    for parameter_name in required_parameters
                    if not weight_dicts
                    or any(parameter_name not in entry for entry in weight_dicts)
                ]
                if missing_parameters:
                    missing.extend(
                        f"{name}.{parameter_name}" for parameter_name in missing_parameters
                    )
                    continue
                loader.load_linear_weights(module, name, weight_dicts)
                continue

            module_weights = loader.filter_weights(name, weights)
            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                if param_name not in module_weights:
                    missing.append(f"{name}.{param_name}")
                    continue
                param.data.copy_(module_weights[param_name].to(param.dtype))

        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise ValueError(
                f"MiniMax-H3 checkpoint is missing {len(missing)} required tensors: "
                f"{preview}{suffix}"
            )

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        # The released BF16 checkpoint deliberately keeps these modules in FP32.
        for module in (
            self.proj_in,
            self.audio_proj_in,
            self.proj_out,
            self.audio_proj_out,
        ):
            if not module.has_any_quant:
                module.to(torch.float32)
        self.time_embedder.to(torch.float32)
