# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for the VisualGen ``Flux3Pipeline`` (a BFL flux-3-image-klein package).

`route` compares the facts that fix which code runs against the one deployment the
target was written for, and raises `NoTarget` stating them otherwise (`auto` falls back
to the built-in pipeline; `require` raises). It reads only identity: the package's DiT
shape and quantized-block layout, the variant, the GPU, the parallel layout, the features
the target does not implement (runtime LoRA, step caching, CPU offload, an NVFP4 VAE),
and the engine choices whose kernels the target owns. Attention is the target's own
per-tensor FP8 E4M3 kernel, so the attention config must name exactly that recipe;
anything else the target would silently not run, so it declines. Tuning knobs (warmup
shapes, CUDA graphs) are not routing inputs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

# (hidden, heads, mode-block depth, single-block depth, image/image_cond channels, text
# channels, RoPE axes, MLP ratio, flow map, output modes). Shape-sniffing also matches
# fine-tunes; the accuracy gate pins identity.
_DIT_SHAPE = (
    3072,
    24,
    5,
    28,
    (("image", 128), ("image_cond", 128)),
    20480,
    (32, 32, 32, 32),
    3.0,
    True,
    ("image",),
)

# The checkpoint's precision layout per block (the six projections of a block are either
# all NVFP4 or all BF16 in the package): what weights.py loads and converts.
_BF16_BLOCKS = frozenset(
    {"content_mode_blocks.image.0"}
    | {f"single_blocks.{i}" for i in range(24, 28)}
    | {f"txt_mode_blocks.{i}" for i in range(5)}
)

_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
_BLOCK = re.compile(r"^((?:content_mode_blocks\.\w+|txt_mode_blocks|single_blocks)\.\d+)\.(\w+)$")


class EngineChoices(NamedTuple):
    """Engine choices that fix which code the target runs but are not in the package."""

    attention: tuple  # (backend, qk dtype, v dtype, block sizes, smooth_k, vc, sparse)
    torch_compile: tuple  # (enable, fullgraph)
    parallel: tuple  # every parallel size
    runtime_lora: bool
    cache: bool
    cpu_offload: bool
    nvfp4_vae: bool


# Per-tensor E4M3 Q/K/V on FLASHINFER: the recipe the target's attention implements.
_ATTENTION = ("FLASHINFER", "fp8", "fp8", (0, 0, 0), False, False, False)


def engine_choices(config: DiffusionPipelineConfig) -> EngineChoices:
    attention = config.attention
    quant = attention.quant_attention_config
    parallel = config.parallel
    return EngineChoices(
        attention=(
            attention.backend,
            None if quant is None else quant.qk_dtype,
            None if quant is None else quant.v_dtype,
            None if quant is None else (quant.q_block_size, quant.k_block_size, quant.v_block_size),
            None if quant is None else bool(quant.smooth_k),
            None if quant is None else bool(quant.vc_attention),
            attention.sparse_attention_config is not None,
        ),
        torch_compile=(config.torch_compile.enable, config.torch_compile.enable_fullgraph),
        parallel=(
            parallel.cfg_size,
            parallel.ulysses_size,
            parallel.ring_size,
            *parallel.attn2d_size,
            parallel.tp_size,
            parallel.parallel_vae_size,
        ),
        runtime_lora=config.runtime_lora is not None,
        cache=config.cache is not None,
        cpu_offload=bool(config.cpu_offload_config.enable),
        nvfp4_vae=config.vae_conv_quant_config is not None,
    )


def _check_engine(engine: EngineChoices) -> None:
    """Decline what the target does not implement, before anything is built."""
    if engine.attention != _ATTENTION:
        raise NoTarget(
            f"attention {engine.attention}; the target runs its own per-tensor FP8 E4M3 "
            "attention: attention_config.backend='FLASHINFER' with quant_attention_config "
            "qk_dtype='fp8', v_dtype='fp8' and no block sizes, smooth_k, VC or sparse attention"
        )
    if engine.torch_compile != (True, False):
        raise NoTarget(
            f"torch_compile (enable, fullgraph) = {engine.torch_compile}; the target's blocks "
            "are certified compiled per block (enable=True, enable_fullgraph=False)"
        )
    if any(size != 1 for size in engine.parallel):
        raise NoTarget(
            f"parallel sizes {engine.parallel}; the target runs on one GPU (every size 1)"
        )
    for name, on in (
        ("runtime_lora_config", engine.runtime_lora),
        ("cache_config (step caching)", engine.cache),
        ("cpu_offload_config", engine.cpu_offload),
        ("vae_config.quant_conv_config (NVFP4 VAE)", engine.nvfp4_vae),
    ):
        if on:
            raise NoTarget(f"{name} is set; the target does not implement it")


def _dit_shape(pretrained_config) -> tuple:
    p = pretrained_config
    return (
        getattr(p, "hidden_size", None),
        getattr(p, "num_heads", None),
        getattr(p, "depth", None),
        getattr(p, "depth_single_blocks", None),
        tuple(sorted(dict(getattr(p, "in_channels", {}) or {}).items())),
        getattr(p, "context_in_dim", None),
        tuple(getattr(p, "axes_dim", ()) or ()),
        float(getattr(p, "mlp_ratio", 0.0)),
        bool(getattr(p, "flow_map", False)),
        tuple(getattr(p, "out_modes", ()) or ()),
    )


def _bf16_blocks(exclude_modules) -> frozenset | str:
    """The blocks the package keeps BF16 (all six projections excluded), or the name of a
    block whose projections are split between BF16 and NVFP4."""
    excluded: dict[str, set[str]] = {}
    for name in exclude_modules or ():
        match = _BLOCK.match(name)
        if match is not None and match.group(2) in _PROJECTIONS:
            excluded.setdefault(match.group(1), set()).add(match.group(2))
    for block, projections in excluded.items():
        if len(projections) != len(_PROJECTIONS):
            return block
    return frozenset(excluded)


def route(config: DiffusionPipelineConfig, compute_capability: Tuple[int, int]) -> type:
    """Return the target pipeline class for this deployment, or raise ``NoTarget`` saying why.

    The target is imported where it is returned, so declining costs no import of modeling
    code.
    """
    shape = _dit_shape(config.primary_pretrained_config)
    variant = config.extra_attrs.get("variant")
    quant = config.quant_config
    nvfp4 = (
        quant is not None
        and quant.quant_algo == QuantAlgo.NVFP4
        and not config.dynamic_weight_quant
        and not config.force_dynamic_quantization
    )
    layout = _bf16_blocks(quant.exclude_modules if quant is not None else None)

    # FLUX.3-klein NVFP4 (variants/nvfp4) on one RTX PRO 6000 Blackwell (sm120).
    if (
        compute_capability == (12, 0)
        and shape == _DIT_SHAPE
        and variant == "nvfp4"
        and nvfp4
        and layout == _BF16_BLOCKS
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .flux3_klein_nvfp4__sm_120__tp1.modeling import ModelingV2Flux3KleinNvfp4Sm120Tp1
        # isort: on
        # fmt: on

        _check_engine(engine_choices(config))
        return ModelingV2Flux3KleinNvfp4Sm120Tp1

    if isinstance(layout, str):
        layout_words = f"block {layout} split between BF16 and NVFP4"
    elif layout == _BF16_BLOCKS:
        layout_words = "the klein NVFP4 block layout"
    else:
        layout_words = f"BF16 blocks {sorted(layout)}"
    raise NoTarget(
        f"compute capability {compute_capability}, DiT shape {shape}, variant {variant!r}, "
        f"{'static NVFP4' if nvfp4 else 'not static NVFP4'}, {layout_words}; the one flux3 "
        f"target is FLUX.3-klein variants/nvfp4 (image.0, single_blocks.24-27 and the text "
        f"blocks BF16) on sm120"
    )
