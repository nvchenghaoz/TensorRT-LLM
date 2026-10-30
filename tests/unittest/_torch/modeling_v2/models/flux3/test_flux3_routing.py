# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3-klein NVFP4 target routing and the VisualGen resolver, driven by synthetic configs
on CPU."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_VISUAL_GEN_ROUTERS,
    modeling_v2_resolve_visual_gen,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CpuOffloadConfig,
    ParallelConfig,
    QuantAttentionConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

pytestmark = pytest.mark.cpu_only

_SM120 = (12, 0)
_PIPELINE = "Flux3Pipeline"
_ROUTING = "tensorrt_llm._torch._experimental.modeling_v2.models.flux3.routing"
_TARGET = "models.flux3.flux3_klein_nvfp4__sm_120__tp1.modeling.ModelingV2Flux3KleinNvfp4Sm120Tp1"
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
_SERVED_ATTENTION = dict(
    backend="FLASHINFER", quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")
)


def _qualified(target) -> str:
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm120(monkeypatch):
    """Route as if this were an RTX PRO 6000 Blackwell, wherever the test runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM120)


def _klein_dit(**overrides) -> SimpleNamespace:
    """The DiT config Flux3Checkpoint.transformer_config() builds for the klein package."""
    dit = dict(
        in_channels={"image": 128, "image_cond": 128},
        context_in_dim=20480,
        hidden_size=3072,
        num_heads=24,
        depth=5,
        depth_single_blocks=28,
        axes_dim=[32, 32, 32, 32],
        theta=2000,
        mlp_ratio=3.0,
        flow_map=True,
        out_modes=["image"],
    )
    dit.update(overrides)
    return SimpleNamespace(**dit)


def _klein_exclusions(bf16_blocks=None) -> list[str]:
    """The package's BF16 projections (``ignore``) plus its BF16 non-block layers."""
    if bf16_blocks is None:
        bf16_blocks = (
            ["content_mode_blocks.image.0"]
            + [f"single_blocks.{i}" for i in range(24, 28)]
            + [f"txt_mode_blocks.{i}" for i in range(5)]
        )
    names = ["txt_in", "emb_in.image", "time_in.in_layer", "final_layer.image.linear"]
    return names + [f"{block}.{p}" for block in bf16_blocks for p in _PROJECTIONS]


def _config(
    mode="auto",
    dit=None,
    variant="nvfp4",
    exclusions=None,
    max_tokens=8192,
    attention=None,
    **fields,
) -> DiffusionPipelineConfig:
    package = SimpleNamespace(
        config={"text_encoder": {"max_tokens": max_tokens, "pad_multiple": 80}}
    )
    config = DiffusionPipelineConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            exclude_modules=_klein_exclusions() if exclusions is None else exclusions,
        ),
        attention=AttentionConfig(**(_SERVED_ATTENTION if attention is None else attention)),
        extra_attrs={"variant": variant, "flux3_checkpoint": package},
        modeling_v2=mode,
        **fields,
    )
    config.model_configs["transformer"] = DiffusionModelConfig(
        pretrained_config=dit if dit is not None else _klein_dit()
    )
    return config


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_klein_nvfp4_sm120_matches(mode):
    target = modeling_v2_resolve_visual_gen(_config(mode), _PIPELINE)
    assert _qualified(target) == _TARGET


@pytest.mark.parametrize("mode", ["off", "auto", "require"])
def test_fp8_attention_needs_modeling_v2(mode):
    """Per-tensor FP8 FLASHINFER attention is the target's recipe; with modeling_v2 off the
    built-in validation rejects it up front."""
    args = dict(attention_config=AttentionConfig(**_SERVED_ATTENTION), modeling_v2=mode)
    if mode == "off":
        with pytest.raises(ValueError, match="served only by a modeling_v2 target"):
            VisualGenArgs(**args)
    else:
        assert VisualGenArgs(**args).modeling_v2 == mode


def test_off_never_imports_or_claims(monkeypatch):
    monkeypatch.delitem(sys.modules, _ROUTING, raising=False)
    assert modeling_v2_resolve_visual_gen(_config("off"), _PIPELINE) is None
    assert _ROUTING not in sys.modules


def test_unrouted_pipeline():
    assert "WanPipeline" not in MODELING_V2_VISUAL_GEN_ROUTERS
    assert modeling_v2_resolve_visual_gen(_config("auto"), "WanPipeline") is None
    with pytest.raises(ValueError, match="no routing module for this pipeline"):
        modeling_v2_resolve_visual_gen(_config("require"), "WanPipeline")


@pytest.mark.parametrize(
    "kwargs, missed",
    [
        (dict(dit=_klein_dit(depth_single_blocks=20)), r"DiT shape \(3072, 24, 5, 20,"),
        (dict(dit=_klein_dit(flow_map=False)), "DiT shape"),
        (dict(variant="fp8r"), "variant 'fp8r'"),
        (dict(exclusions=_klein_exclusions(["content_mode_blocks.image.0"])), "BF16 blocks"),
        (
            dict(exclusions=_klein_exclusions() + ["single_blocks.0.q_proj"]),
            "single_blocks.0 split between BF16 and NVFP4",
        ),
    ],
    ids=["other-shape", "not-flow-map", "fp8r-variant", "other-layout", "split-block"],
)
def test_near_misses_do_not_match(kwargs, missed):
    assert modeling_v2_resolve_visual_gen(_config("auto", **kwargs), _PIPELINE) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve_visual_gen(_config("require", **kwargs), _PIPELINE)


def test_on_sm103_does_not_match(monkeypatch):
    """The target's kernels are built for sm120; a GB300 keeps the built-in pipeline."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    assert modeling_v2_resolve_visual_gen(_config("auto"), _PIPELINE) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve_visual_gen(_config("require"), _PIPELINE)


@pytest.mark.parametrize(
    "kwargs, missed",
    [
        (dict(attention=dict(backend="VANILLA")), "own per-tensor FP8 E4M3 attention"),
        (
            dict(
                attention=dict(
                    backend="FLASHINFER",
                    quant_attention_config=QuantAttentionConfig(qk_dtype="nvfp4", v_dtype="fp8"),
                )
            ),
            "own per-tensor FP8 E4M3 attention",
        ),
        (dict(torch_compile=TorchCompileConfig(enable=False)), "torch_compile"),
        (dict(parallel=ParallelConfig(cfg_size=2)), "parallel sizes"),
        (dict(cpu_offload_config=CpuOffloadConfig(enable=True)), "cpu_offload_config"),
        (dict(cache=TeaCacheConfig()), "cache_config"),
    ],
    ids=["bf16-attention", "nvfp4-qk", "eager", "cfg-parallel", "offload", "teacache"],
)
def test_engine_choices_the_target_refuses(kwargs, missed):
    assert modeling_v2_resolve_visual_gen(_config("auto", **kwargs), _PIPELINE) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve_visual_gen(_config("require", **kwargs), _PIPELINE)
