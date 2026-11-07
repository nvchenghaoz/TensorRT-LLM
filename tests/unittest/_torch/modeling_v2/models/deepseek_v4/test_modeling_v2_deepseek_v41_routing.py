# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV41ForCausalLM deployment lands, driven by synthetic configs.

The one target claims DeepSeek-V4.1-Flash on four SM120 GPUs with attention TP4 and the experts
split moe_tp2 x moe_ep2, without speculative decoding. Anything else is declined: ``auto`` falls
back to the built-in implementation and ``require`` names the criterion that missed.
"""

from __future__ import annotations

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_TARGET = (
    "models.deepseek_v4.v41_flash_1017__sm_120__tp4_moetp2ep2.modeling."
    "ModelingV2DeepseekV41Flash1017Sm120Tp4Moetp2ep2"
)
_TP4_MOETP2EP2 = dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2)


@pytest.fixture
def sm(monkeypatch):
    """Route as if on a given GPU, wherever the test runs."""

    def on(capability):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)

    return on


def _config(**overrides):
    """The shape DeepSeek-V4.1-Flash-1017's text config declares."""
    fields = dict(
        architectures=["DeepseekV41ForCausalLM"],
        num_hidden_layers=40,
        hidden_size=5120,
        n_routed_experts=384,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs):
    return ModelConfig(
        pretrained_config=pretrained_config, mapping=Mapping(**mapping_kwargs), spec_config=spec_config
    )


def _llm_args(mode):
    return TorchLlmArgs(model="/tmp/dummy_model", modeling_v2=mode)


def _qualified(target) -> str:
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


def test_flash_on_sm120_tp4_moetp2ep2_matches(sm):
    sm((12, 0))
    target = modeling_v2_resolve(_model_config(_config(), **_TP4_MOETP2EP2), _llm_args("require"))
    assert _qualified(target) == _TARGET


@pytest.mark.parametrize(
    "capability, config_kwargs, mapping_kwargs, spec, missed",
    [
        ((10, 0), {}, _TP4_MOETP2EP2, None, r"compute capability \(10, 0\)"),
        ((12, 0), {"hidden_size": 4096}, _TP4_MOETP2EP2, None, "config shape"),
        ((12, 0), {}, dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4), None, "moe_tp=1"),
        ((12, 0), {}, dict(_TP4_MOETP2EP2, enable_attention_dp=True), None, "attention_dp=True"),
        ((12, 0), {}, _TP4_MOETP2EP2, MTPDecodingConfig(max_draft_len=1), "speculative decoding"),
    ],
    ids=["sm100", "other-shape", "ep4", "attention-dp", "mtp"],
)
def test_near_misses_do_not_match(sm, capability, config_kwargs, mapping_kwargs, spec, missed):
    sm(capability)
    config = _model_config(_config(**config_kwargs), spec_config=spec, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_off_resolves_nothing(sm):
    sm((12, 0))
    assert modeling_v2_resolve(_model_config(_config(), **_TP4_MOETP2EP2), _llm_args("off")) is None
