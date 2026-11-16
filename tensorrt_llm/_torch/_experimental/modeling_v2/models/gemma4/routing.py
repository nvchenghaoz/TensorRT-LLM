# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Gemma4ForConditionalGeneration.

`route` compares the facts that fix which code runs against the one deployment the target was
written for, and raises `NoTarget` stating them otherwise (`auto` falls back to the built-in
implementation; `require` raises). Shape criteria are read off `text_config`: the dense
Gemma-4-31B text tower is what the target implements. It serves text only and refuses a request
carrying images, which needs `modeling_v2='off'`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


class TextShape(NamedTuple):
    """The text tower's geometry. Shape-sniffing also matches fine-tunes; the accuracy gate
    pins identity."""

    num_layers: int
    hidden: int
    intermediate: int
    heads: int
    sliding_kv_heads: int
    sliding_head_dim: int
    full_kv_heads: int
    full_head_dim: int
    k_eq_v: bool
    sliding_window: int
    vocab: int


_TEXT_SHAPE = TextShape(60, 5376, 21504, 32, 16, 256, 4, 512, True, 1024, 262144)


def text_shape(text) -> TextShape:
    return TextShape(
        text.num_hidden_layers,
        text.hidden_size,
        text.intermediate_size,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        getattr(text, "num_global_key_value_heads", None),
        getattr(text, "global_head_dim", None),
        bool(getattr(text, "attention_k_eq_v", False)),
        getattr(text, "sliding_window", None),
        text.vocab_size,
    )


def _check_text(text) -> None:
    """Dense, no per-layer embeddings, no shared-KV layers, every sixth layer full attention
    ending on one: the layout the target's constructor builds."""
    if getattr(text, "enable_moe_block", False):
        raise NoTarget("the text tower has an MoE block; the target is the dense 31B model")
    if getattr(text, "hidden_size_per_layer_input", 0):
        raise NoTarget("the text tower has per-layer embeddings; the target implements none")
    if getattr(text, "num_kv_shared_layers", 0):
        raise NoTarget("the text tower shares KV across layers; the target implements none")
    if getattr(text, "use_double_wide_mlp", False):
        raise NoTarget("use_double_wide_mlp is set; the target's MLP is 21504 wide everywhere")
    expected = [
        "full_attention" if i % 6 == 5 else "sliding_attention" for i in range(text.num_hidden_layers)
    ]
    if list(text.layer_types) != expected:
        raise NoTarget("layer_types is not five sliding-window layers per full-attention layer")
    if getattr(text, "final_logit_softcapping", None) is None:
        raise NoTarget("final_logit_softcapping is unset; the target caps logits at 30")


def _nvfp4_mlp_recipe(model_config: ModelConfig) -> bool:
    """NVFP4 MLP with every layer's self_attn excluded (bf16): the ModelOpt export weights.py
    reads."""
    qc = model_config.quant_config
    if qc is None or qc.quant_algo != QuantAlgo.NVFP4 or qc.group_size != 16:
        return False
    n = model_config.pretrained_config.text_config.num_hidden_layers
    return all(
        qc.is_module_excluded_from_quantization(f"model.language_model.layers.{i}.self_attn.q_proj")
        for i in range(n)
    ) and not qc.is_module_excluded_from_quantization("model.language_model.layers.0.mlp.gate_proj")


def _check_runtime(llm_args: TorchLlmArgs) -> None:
    """Decline engine features the target does not implement, so ``auto`` falls back."""
    if llm_args.attn_backend != "FLASHINFER":
        raise NoTarget(
            f"attn_backend={llm_args.attn_backend!r}; the target's attention reads "
            "FlashInferAttentionMetadata (attn_backend='FLASHINFER', the sm_120 default)"
        )
    if llm_args.lora_config is not None:
        raise NoTarget("LoRA is configured; the target implements no LoRA")
    transceiver = llm_args.cache_transceiver_config
    if transceiver is not None and transceiver.backend is not None:
        raise NoTarget("disaggregated serving (cache_transceiver_config.backend is set)")
    if llm_args.kv_cache_config.use_kv_cache_manager_v2 is False:
        raise NoTarget(
            "use_kv_cache_manager_v2=False; the target's sliding-window and full-attention pools "
            "are the V2 manager's per-layer pools"
        )


def _check_engine(kv_cache_quant_algo, model_config: ModelConfig) -> None:
    if kv_cache_quant_algo != QuantAlgo.FP8:
        raise NoTarget(
            f"a {kv_cache_quant_algo or 'bf16'} KV cache; the target's QKV prep and attention "
            "read and write an fp8-e4m3 pool (the checkpoint's kv_cache_quant_algo)"
        )
    if model_config.torch_dtype != torch.bfloat16:
        raise NoTarget(f"the engine runs the model in {model_config.torch_dtype}; the target is bf16")


@routes("Gemma4ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    text = getattr(pretrained_config, "text_config", None)
    if text is None:
        raise NoTarget(
            f"{type(pretrained_config).__name__} carries no text_config; the gemma4 target "
            "reads the composite checkpoint's text tower"
        )
    shape = text_shape(text)
    recipe = _nvfp4_mlp_recipe(model_config)

    # Gemma-4-31B-IT-NVFP4 on one RTX PRO 6000 Blackwell (sm120), no speculative decoding.
    if (
        compute_capability == (12, 0)
        and shape == _TEXT_SHAPE
        and recipe
        and mapping.world_size == 1
        and spec_config is None
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .gemma4_31b_it_nvfp4__sm_120__tp1.modeling import ModelingV2Gemma431BItNvfp4Sm120Tp1
        # isort: on
        # fmt: on

        _check_text(text)
        _check_runtime(llm_args)
        _check_engine(engine_kv_cache_quant_algo(model_config, llm_args), model_config)
        return ModelingV2Gemma431BItNvfp4Sm120Tp1

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {tuple(shape)}, "
        f"{'the' if recipe else 'not the'} NVFP4-MLP / bf16-attention recipe, "
        f"{mapping.world_size} ranks, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}; "
        "the one gemma4 target is Gemma-4-31B-IT-NVFP4 on sm120, tp1, without speculative "
        "decoding"
    )
