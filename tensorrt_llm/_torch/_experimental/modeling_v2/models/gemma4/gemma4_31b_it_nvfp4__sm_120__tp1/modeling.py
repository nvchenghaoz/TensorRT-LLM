# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Gemma-4-31B-IT-NVFP4 / sm_120 / tp1 — self-contained modeling code.

A dense Gemma4 text tower: 60 decoder layers, five sliding-window layers (window 1024, 32 q /
16 kv heads of 256, RoPE theta 10k) per full-attention layer (32 q / 4 kv heads of 512 with
K = V, proportional RoPE theta 1M over a quarter of each head), each followed by a GELU-tanh
MLP of 21504. Every layer ends in a scaled residual (`layer_scalar`) and the logits are
soft-capped at 30. The attention projections are bf16; the MLP is NVFP4. The core holds the
weights `weights.py` declares; `forward` runs a step over them, passing every argument to every
op at the call site.

**Prefill** is the optimized path (the target was tuned for 300-8000-token prompts producing
one token). Per layer:

* qkv: the bf16 projection routed by the step's row count (`ops/gemm/bf16_proj_gemm`: the
  SM120 sequential-K GEMM at about 300 rows, cuBLASLt's measured per-range algorithms where
  they are pinned, F.linear elsewhere; all bitwise equal);
* one fused q/k/v RMSNorm + RoPE + FP8 kernel (`ops/attention/qkv_norm_rope_quant`) that, on an
  eager step, also stores K/V straight into the paged FP8 pool;
* FlashInfer FA2 over the FP8 pool on the target's own prefill module (`kernels/flashinfer/`:
  a head-512 64-row tile, longest-first CTA order, hardware FP8->BF16); the last layer computes
  attention for the rows the logits read only;
* o_proj, then one kernel for post-attention norm + residual add + pre-FFN norm + NVFP4
  quantize (`ops/norm/rmsnorm_residual_add_fp4_quant`);
* gate_up + GELU-tanh * up + NVFP4 quantize (`ops/gemm/gate_up_gelu_nvfp4`: the fused-epilogue
  GEMM where the tuned tactic is its mainloop, an M-chunked chain at 1664+ rows, else the tuned
  GEMM and the fused activation kernel), down_proj (`ops/gemm/nvfp4_gemm`: the shared NVFP4
  GEMM with dense tuning buckets and tactic pins for these shapes);
* one kernel for post-FFN norm + residual add + layer scalar + the next layer's input norm.

The first layer's input norm runs fused with the embedding lookup; the final norm and the
lm_head run on the rows the sampler reads only. Prefill CUDA graphs (breakable) replay steps
at a captured token count; attention is the eager break.

**Decode** steps run the same code (FlashInfer decode wrapper, captured in the engine's decode
CUDA graphs).

Numerics: every fused kernel is bitwise equal to the unfused chain of the built-in model at
the same settings, and every GEMM route computes the same bits as the default dispatch.
The accuracy gate in
`tests/integration/defs/accuracy/modeling_v2/gemma4/gemma4_31b_it_nvfp4__sm_120__tp1/test_accuracy.py`
certifies the whole.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph,
)
from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttentionMetadata
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata, RopeParams
from tensorrt_llm._torch.model_config import KVCacheLayerSpec, ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm.functional import RotaryScalingType

from . import weights as _weights
from .ops.attention.paged_attention import paged_attention
from .ops.attention.qkv_norm_rope_quant import qkv_norm_rope_quant
from .ops.embedding.embed_scale_rmsnorm import embed_scale_rmsnorm, rmsnorm
from .ops.gemm.bf16_proj_gemm import BF16_PROJ_ROUTES, bf16_proj_gemm
from .ops.gemm.gate_up_gelu_nvfp4 import (
    MLP_CHUNK_MIN_M,
    MLP_CHUNK_ROWS,
    gate_up_gelu_nvfp4,
    prepare_gate_up_weights,
)
from .ops.gemm.nvfp4_gemm import nvfp4_gemm
from .ops.norm.rmsnorm_residual_add_fp4_quant import rmsnorm_residual_add_fp4_quant
from .ops.norm.rmsnorm_residual_add_scale import rmsnorm_residual_add_scale

_SM = (12, 0)

_TEXT_ONLY = (
    "the gemma4 modeling_v2 target serves text only; image requests need modeling_v2='off'"
)


class _AttnGeometry(NamedTuple):
    """One attention kind's geometry (sliding-window or full)."""

    head_dim: int
    num_heads: int
    num_kv_heads: int
    q_width: int
    kv_width: int
    #: TRTLLM's exclusive window, or None.
    window: int | None
    #: K = V: the projection is [q | k] and V is the raw K block.
    k_eq_v: bool


class _Layer(NamedTuple):
    i: int
    sliding: bool
    geom: _AttnGeometry
    cos_sin: torch.Tensor
    is_last: bool


def _sliding_cos_sin(theta: float, head_dim: int, max_positions: int) -> torch.Tensor:
    """The sliding-window layers' RoPE table: default RoPE over the whole head, built by the
    engine's RoPE constants ([max_positions, 2, head_dim / 2] fp32)."""
    rope = RopeParams()
    rope.max_positions = max_positions
    rope.theta = theta
    rope.scale_type = RotaryScalingType.none
    rope.scale = 1.0
    rope.dim = head_dim
    return rope.create_rope_const_params(interleave=False)[1].reshape(max_positions, 2, -1)


def _full_cos_sin(theta: float, head_dim: int, partial: float, max_positions: int) -> torch.Tensor:
    """The full-attention layers' proportional RoPE table, as HF builds it: head_dim // 2
    frequency pairs of which only the first `partial * head_dim / 2` rotate (the rest have zero
    frequency: cos 1, sin 0), and `rotate_half` pairs dim i with dim i + head_dim / 2."""
    rope_angles = int(partial * head_dim // 2)
    half_dim = head_dim // 2
    inv_freq = np.concatenate(
        [
            1.0
            / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / head_dim)),
            np.zeros(half_dim - rope_angles, dtype=np.float32),
        ]
    )
    positions = np.arange(max_positions, dtype=np.float32)
    sinusoid = np.einsum("i,j->ij", positions, inv_freq)
    return torch.tensor(np.stack([np.cos(sinusoid), np.sin(sinusoid)], axis=1), dtype=torch.float32)


def kv_cache_layer_specs(text) -> list[KVCacheLayerSpec]:
    """Per-layer KV geometry for the engine's V2 manager: sliding-window layers keep a 1024-token
    window of 16 heads of 256, full-attention layers every token of 4 heads of 512."""
    specs = []
    for t in text.layer_types:
        sliding = t == "sliding_attention"
        specs.append(
            KVCacheLayerSpec(
                head_dim=text.head_dim if sliding else text.global_head_dim,
                num_kv_heads=text.num_key_value_heads if sliding else text.num_global_key_value_heads,
                num_q_heads=text.num_attention_heads,
                attention_window=int(text.sliding_window) if sliding else None,
            )
        )
    return specs


class Gemma431BTp1ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config

        # This target IS this geometry — assert, never adapt.
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        mapping = model_config.mapping
        assert mapping.world_size == 1 and not mapping.enable_attention_dp, "tp1 target"
        assert model_config.torch_dtype == torch.bfloat16, model_config.torch_dtype
        assert getattr(model_config, "spec_config", None) is None, (
            "this target implements no speculative decoding"
        )
        assert cfg.tie_word_embeddings, "the lm_head is the embedding"
        assert not cfg.attention_bias, "no q/k/v/o bias"
        assert cfg.hidden_activation == "gelu_pytorch_tanh", cfg.hidden_activation

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps
        self.softcap = float(cfg.final_logit_softcapping)
        # The embedding scale sqrt(hidden), rounded to bf16 as the built-in model's 0-dim bf16
        # tensor holds it.
        self.embed_scale = float(torch.tensor(math.sqrt(self.hidden), dtype=torch.bfloat16))

        self.is_sliding = [t == "sliding_attention" for t in cfg.layer_types]
        assert not self.is_sliding[-1], "the final layer is a full-attention layer"
        self.sliding = _AttnGeometry(
            head_dim=cfg.head_dim,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            q_width=cfg.num_attention_heads * cfg.head_dim,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            window=int(cfg.sliding_window),
            k_eq_v=False,
        )
        self.full = _AttnGeometry(
            head_dim=cfg.global_head_dim,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_global_key_value_heads,
            q_width=cfg.num_attention_heads * cfg.global_head_dim,
            kv_width=cfg.num_global_key_value_heads * cfg.global_head_dim,
            window=None,
            k_eq_v=True,
        )
        assert cfg.attention_k_eq_v
        rope = cfg.rope_parameters
        self._rope_sliding = rope["sliding_attention"]
        self._rope_full = rope["full_attention"]
        assert self._rope_full.get("rope_type") == "proportional", self._rope_full
        self.max_positions = cfg.max_position_embeddings

        # The NVFP4 GEMM backends the built-in Linear offers its auto-selection.
        self.nvfp4_backends = list(model_config.nvfp4_gemm_allowed_backends)

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))
        # The engine ties lm_head to `embed_tokens.weight`.
        self.embed_tokens = _EmbeddingTable(self.w["embed"])
        self.gate_up_permuted = False
        self.layers_view: list[_Layer] = []
        self._derived = False

    def post_load_weights(self) -> None:
        """Build the RoPE tables and the per-layer views, and put gate_up in the fused-epilogue
        row order. Runs once: the loader calls every module's hook and the shell calls this
        one too."""
        if self._derived:
            return
        self._derived = True
        device = self.w["embed"].device
        cos_sin_sliding = _sliding_cos_sin(
            self._rope_sliding.get("rope_theta", 10_000.0), self.sliding.head_dim, self.max_positions
        ).to(device)
        cos_sin_full = _full_cos_sin(
            self._rope_full.get("rope_theta", 1_000_000.0),
            self.full.head_dim,
            self._rope_full.get("partial_rotary_factor", 0.25),
            self.max_positions,
        ).to(device)
        self.layers_view = [
            _Layer(
                i=i,
                sliding=self.is_sliding[i],
                geom=self.sliding if self.is_sliding[i] else self.full,
                cos_sin=cos_sin_sliding if self.is_sliding[i] else cos_sin_full,
                is_last=i == self.num_layers - 1,
            )
            for i in range(self.num_layers)
        ]
        self.gate_up_permuted = all(
            prepare_gate_up_weights(self.w[f"l{i}_gate_up_w"], self.w[f"l{i}_gate_up_sf"], self.hidden)
            for i in range(self.num_layers)
        )

    # ------------------------------------------------------------------ the step ---

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        read_rows_last_layer: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Hidden states after the last layer's tail (the final norm is the caller's)."""
        md = attn_metadata
        assert isinstance(md, FlashInferAttentionMetadata), type(md)
        assert inputs_embeds is None, "the target embeds token ids itself"
        w = self.w
        # Embedding lookup * sqrt(hidden) and layer 0's input norm in one launch.
        h, x = embed_scale_rmsnorm(input_ids, w["embed"], self.embed_scale, w["l0_in_norm"], self.eps)
        if x is None:
            x = rmsnorm(h, w["l0_in_norm"], self.eps)
        positions = position_ids.view(-1)
        for L in self.layers_view:
            i = L.i
            attn = self._attention(L, x, positions, md, read_rows_last_layer and L.is_last)
            o = bf16_proj_gemm(attn, w[f"l{i}_o_w"], BF16_PROJ_ROUTES.get(tuple(w[f"l{i}_o_w"].shape[::-1])))
            # Post-attention norm + residual add, and the pre-FFN norm + gate_up NVFP4 quantize.
            h, fp4, sf = rmsnorm_residual_add_fp4_quant(
                o,
                h,
                w[f"l{i}_post_attn_norm"],
                self.eps,
                w[f"l{i}_pre_ffn_norm"],
                self.eps,
                w[f"l{i}_gate_up_is"],
            )
            mlp = self._mlp(i, fp4, sf)
            # Post-FFN norm + residual add + layer scalar, and the next layer's input norm.
            h, x = rmsnorm_residual_add_scale(
                mlp,
                h,
                w[f"l{i}_post_ffn_norm"],
                w[f"l{i}_layer_scalar"],
                self.eps,
                None if L.is_last else w[f"l{i + 1}_in_norm"],
                self.eps,
            )
        return h

    def _attention(self, L: _Layer, x, positions, md, read_rows: bool) -> torch.Tensor:
        g = L.geom
        i = L.i
        w = self.w
        qkv_w = w[f"l{i}_qkv_w"]
        qkv = bf16_proj_gemm(x, qkv_w, BF16_PROJ_ROUTES.get((qkv_w.shape[1], qkv_w.shape[0])))
        q, k, v = qkv_norm_rope_quant(
            qkv,
            positions,
            L.cos_sin,
            w[f"l{i}_q_norm"],
            w[f"l{i}_k_norm"],
            self.eps,
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            g.k_eq_v,
            md,
            i,
        )
        window_left = g.window - 1 if g.window is not None else None
        q_scaling = 1.0 / math.sqrt(g.head_dim)  # Gemma4 attends with scale 1
        if is_in_breakable_cuda_graph():
            output = torch.empty((q.shape[0], g.q_width), dtype=torch.bfloat16, device=q.device)
            _attention_break(
                q, k, v, i, g.num_heads, g.num_kv_heads, g.head_dim, q_scaling, window_left, output
            )
            return output
        output = torch.empty((q.shape[0], g.q_width), dtype=torch.bfloat16, device=q.device)
        paged_attention(
            q,
            k,
            v,
            md,
            i,
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            q_scaling,
            window_left,
            output,
            read_rows,
        )
        return output

    def _mlp(self, i: int, fp4, sf) -> torch.Tensor:
        w = self.w
        m = fp4.shape[0]
        act_fp4, act_sf = gate_up_gelu_nvfp4(
            fp4,
            sf,
            w[f"l{i}_gate_up_w"],
            w[f"l{i}_gate_up_sf"],
            w[f"l{i}_gate_up_alpha"],
            w[f"l{i}_down_is"],
            self.nvfp4_backends,
            MLP_CHUNK_ROWS if m >= MLP_CHUNK_MIN_M else None,
            self.gate_up_permuted,
        )
        return nvfp4_gemm(
            act_fp4,
            w[f"l{i}_down_w"],
            act_sf,
            w[f"l{i}_down_sf"],
            w[f"l{i}_down_alpha"],
            torch.bfloat16,
            self.nvfp4_backends,
        )


def _bcg_attention(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_scaling: float,
    window_left: int | None,
    output: torch.Tensor,
) -> None:
    """The breakable prefill graph's eager break: one layer's attention for the step being
    replayed. The per-step metadata is read here, never passed in (a break's arguments are
    frozen at capture)."""
    ref = get_model_extra_attrs()["attention_metadata"]
    paged_attention(
        q,
        k,
        v,
        ref(),
        layer_idx,
        num_heads,
        num_kv_heads,
        head_dim,
        q_scaling,
        window_left,
        output,
        False,
    )


def _attention_break(*args) -> None:
    eager_on_graph(*args, inner=_bcg_attention)


class _EmbeddingTable(nn.Module):
    """The embedding the engine ties lm_head to (`tie_word_embeddings`): it reads
    `embed_tokens.weight` and its tensor-parallel mode."""

    tp_mode = TensorParallelMode.COLUMN
    tp_size = 1

    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = weight


class _TiedLMHead(nn.Module):
    """lm_head = the bf16 embedding, as F.linear (the built-in tied head's dispatch)."""

    def __init__(self, core: Gemma431BTp1ModelingV2) -> None:
        super().__init__()
        self._core = (core,)  # not a submodule: the core is the shell's `model`
        self.vocab_size_padded = core.vocab
        self.num_embeddings = core.vocab

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self._core[0].w["embed"])


def _carries_media(data) -> bool:
    data = data or {}
    return (
        data.get("image", {}).get("pixel_values") is not None
        or data.get("video", {}).get("pixel_values_videos") is not None
        or data.get("audio") is not None
        or data.get("multimodal_embedding") is not None
    )


class ModelingV2Gemma431BItNvfp4Sm120Tp1(
    DecoderModelForCausalLM[Gemma431BTp1ModelingV2, PretrainedConfig]
):
    #: Prefill CUDA graphs replay only steps at a captured token count: a padded replay would run
    #: the NVFP4 GEMMs at another M, whose tuned tactic may differ from the eager step's.
    prefill_cuda_graph_exact_sizes = True

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """The sliding-window and full-attention pools are the V2 manager's per-layer pools."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        text = getattr(cfg, "text_config", None)
        assert text is not None, (
            "this target is the text tower of the composite Gemma4 checkpoint; "
            "pretrained_config carries no text_config"
        )
        # Everything that sizes the pools reads `model_config.pretrained_config`; the engine
        # keeps its own reference to this ModelConfig, so the swap is in place.
        model_config.pretrained_config = text
        model_config.set_kv_cache_layer_specs(kv_cache_layer_specs(text))
        super().__init__(
            Gemma431BTp1ModelingV2(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        self.lm_head = _TiedLMHead(self.model)
        self.logits_processor = LogitsProcessor()

    def validate_request(self, request) -> None:
        """Admission check: an image request fails here as its own error."""
        if _carries_media(getattr(request, "py_multimodal_data", None)):
            raise ValueError(_TEXT_ONLY)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        assert spec_metadata is None, "this target implements no speculative decoding"
        assert lora_params is None, "this target implements no LoRA"
        if any(
            _carries_media(getattr(p, "multimodal_data", None))
            for p in kwargs.get("multimodal_params") or ()
        ):
            raise ValueError(_TEXT_ONLY)
        md = attn_metadata
        core = self.model
        # Only each request's last row reaches the lm_head unless context logits are asked for:
        # the final norm and the head run on those rows, and the last layer's context attention
        # may run on them alone (bitwise; it falls back to the full-row call on any guard).
        gather_rows = not return_context_logits and getattr(md, "seq_lens_cuda", None) is not None
        read_rows = (
            gather_rows
            and md.num_contexts > 0
            and md.num_generations == 0
            and md.padded_num_tokens is None
            and not md.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
        )
        hidden = core(
            attn_metadata=md,
            input_ids=input_ids,
            position_ids=position_ids,
            read_rows_last_layer=read_rows,
        )
        if md.padded_num_tokens is not None:
            hidden = hidden[: md.num_tokens]
        if gather_rows:
            last_rows = torch.cumsum(md.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            normed = rmsnorm(hidden[last_rows], core.w["norm"], core.eps)
            logits = self.lm_head(normed).float()
        else:
            normed = rmsnorm(hidden, core.w["norm"], core.eps)
            logits = self.logits_processor.forward(normed, self.lm_head, md, return_context_logits)
        return torch.tanh(logits / core.softcap) * core.softcap

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()


