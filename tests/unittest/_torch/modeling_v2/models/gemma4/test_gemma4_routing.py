# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma4 routing, driven by synthetic configs on CPU.

`route` claims Gemma-4-31B-IT-NVFP4 on one sm_120 GPU and declines every single deviation with
a `NoTarget` naming it. The claim branch imports the target's `modeling` module, whose package
loads the family's native library and FlashInfer; the tests put a stand-in module under that
name, so routing is checked on its own, on any host (the real class is exercised by the
target's accuracy test).
"""

import sys
from types import ModuleType

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    registered_routes,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4 import routing
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    KvCacheConfig,
    LoraConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM120 = (12, 0)
_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET_PACKAGE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1"
)
_TARGET_CLASS = "ModelingV2Gemma431BItNvfp4Sm120Tp1"
_LAYERS = 60


class _Claimed:
    """Stands in for the target class the claim branch imports."""


@pytest.fixture(autouse=True)
def _target_module(monkeypatch):
    """`from .<target>.modeling import <Class>` resolves to the stand-in."""
    package = ModuleType(_TARGET_PACKAGE)
    package.__path__ = []
    modeling = ModuleType(f"{_TARGET_PACKAGE}.modeling")
    setattr(modeling, _TARGET_CLASS, _Claimed)
    package.modeling = modeling
    monkeypatch.setitem(sys.modules, _TARGET_PACKAGE, package)
    monkeypatch.setitem(sys.modules, f"{_TARGET_PACKAGE}.modeling", modeling)


def _layer_types(n=_LAYERS):
    return ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(n)]


def _gemma4_config(**text_overrides) -> PretrainedConfig:
    """The composite config Gemma-4-31B-IT-NVFP4's config.json declares (text tower only)."""
    text = dict(
        num_hidden_layers=_LAYERS,
        hidden_size=5376,
        intermediate_size=21504,
        num_attention_heads=32,
        num_key_value_heads=16,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        attention_k_eq_v=True,
        sliding_window=1024,
        vocab_size=262144,
        layer_types=_layer_types(),
        final_logit_softcapping=30.0,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        torch_dtype=torch.bfloat16,
    )
    text.update(text_overrides)
    return PretrainedConfig(
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="gemma4",
        torch_dtype=torch.bfloat16,
        text_config=PretrainedConfig(**text),
    )


def _quant_config(attention_excluded=True, kv_cache_quant_algo=QuantAlgo.FP8) -> QuantConfig:
    """ModelOpt NVFP4 (group 16) with every layer's self_attn excluded, as the checkpoint's
    `ignore` list writes it."""
    exclude = ["lm_head", "model.embed_vision*"]
    if attention_excluded:
        exclude += [f"model.language_model.layers.{i}.self_attn*" for i in range(_LAYERS)]
    return QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        group_size=16,
        kv_cache_quant_algo=kv_cache_quant_algo,
        exclude_modules=exclude,
    )


def _model_config(pretrained_config=None, quant_config=None, spec_config=None, **mapping):
    return ModelConfig(
        pretrained_config=_gemma4_config() if pretrained_config is None else pretrained_config,
        mapping=Mapping(**mapping) if mapping else Mapping(),
        spec_config=spec_config,
        quant_config=_quant_config() if quant_config is None else quant_config,
    )


def _llm_args(kv=None, **kwargs) -> TorchLlmArgs:
    kwargs.setdefault("attn_backend", "FLASHINFER")
    return TorchLlmArgs(
        model=_DUMMY_MODEL,
        kv_cache_config=KvCacheConfig(**(dict(dtype="fp8") if kv is None else kv)),
        **kwargs,
    )


def test_the_route_is_registered_for_the_architecture():
    assert registered_routes()["Gemma4ForConditionalGeneration"] is routing.route


def test_the_served_shape_is_the_target_shape():
    text = _gemma4_config().text_config
    assert routing.text_shape(text) == routing._TEXT_SHAPE


def test_gemma4_31b_nvfp4_on_sm120_tp1_is_claimed():
    assert routing.route(_model_config(), _llm_args(), _SM120) is _Claimed


@pytest.mark.parametrize(
    "model_config, capability, missed",
    [
        (lambda: _model_config(), (10, 3), r"compute capability \(10, 3\)"),
        (
            lambda: _model_config(_gemma4_config(hidden_size=4096)),
            _SM120,
            r"text shape \(60, 4096,",
        ),
        (
            lambda: _model_config(_gemma4_config(num_global_key_value_heads=8)),
            _SM120,
            r"text shape \(60, .*, 256, 8, 512,",
        ),
        (
            lambda: _model_config(quant_config=_quant_config(attention_excluded=False)),
            _SM120,
            "not the NVFP4-MLP / bf16-attention recipe",
        ),
        (lambda: _model_config(world_size=2, tp_size=2), _SM120, "2 ranks"),
        (
            lambda: _model_config(spec_config=MTPDecodingConfig(max_draft_len=3)),
            _SM120,
            "speculative decoding MTP",
        ),
    ],
    ids=["sm103", "other-hidden", "other-full-kv-heads", "nvfp4-attention", "tp2", "mtp"],
)
def test_deployment_deviations_are_declined(model_config, capability, missed):
    with pytest.raises(NoTarget, match=missed):
        routing.route(model_config(), _llm_args(), capability)


@pytest.mark.parametrize(
    "text_overrides, missed",
    [
        (dict(enable_moe_block=True), "MoE block"),
        (dict(layer_types=["sliding_attention"] * 59 + ["full_attention"]), "layer_types"),
        (dict(final_logit_softcapping=None), "final_logit_softcapping"),
    ],
    ids=["moe-block", "layer-pattern", "no-softcap"],
)
def test_text_tower_deviations_are_declined(text_overrides, missed):
    config = _model_config(_gemma4_config(**text_overrides))
    with pytest.raises(NoTarget, match=missed):
        routing.route(config, _llm_args(), _SM120)


@pytest.mark.parametrize(
    "kv, kwargs, quant_kv, missed",
    [
        (None, dict(attn_backend="TRTLLM"), QuantAlgo.FP8, "attn_backend='TRTLLM'"),
        (None, dict(lora_config=LoraConfig()), QuantAlgo.FP8, "LoRA"),
        (
            None,
            dict(cache_transceiver_config=CacheTransceiverConfig(backend="NIXL")),
            QuantAlgo.FP8,
            "disaggregated serving",
        ),
        (
            dict(dtype="fp8", use_kv_cache_manager_v2=False),
            {},
            QuantAlgo.FP8,
            "use_kv_cache_manager_v2=False",
        ),
        # `auto` keeps the checkpoint's algorithm: none means a bf16 pool.
        (dict(dtype="auto"), {}, None, "a bf16 KV cache"),
        (dict(dtype="nvfp4"), {}, QuantAlgo.FP8, "NVFP4 KV cache"),
    ],
    ids=["trtllm-attention", "lora", "disaggregated", "kv-manager-v1", "bf16-kv", "nvfp4-kv"],
)
def test_engine_deviations_are_declined(kv, kwargs, quant_kv, missed):
    config = _model_config(quant_config=_quant_config(kv_cache_quant_algo=quant_kv))
    with pytest.raises(NoTarget, match=missed):
        routing.route(config, _llm_args(kv, **kwargs), _SM120)


def test_auto_kv_dtype_with_the_checkpoints_fp8_is_claimed():
    """`kv_cache_config.dtype='auto'` keeps the checkpoint's FP8 KV algorithm."""
    assert routing.route(_model_config(), _llm_args(dict(dtype="auto")), _SM120) is _Claimed


def test_a_config_without_text_config_is_declined():
    flat = PretrainedConfig(architectures=["Gemma4ForConditionalGeneration"])
    with pytest.raises(NoTarget, match="carries no text_config"):
        routing.route(_model_config(flat), _llm_args(), _SM120)
