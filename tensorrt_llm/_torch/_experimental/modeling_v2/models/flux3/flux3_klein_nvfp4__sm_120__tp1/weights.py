# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loader: FLUX.3-klein ``variants/nvfp4`` / sm_120 / tp1.

`load(model, weights)` fills every buffer `modeling.Flux3KleinTransformer` declares from
the package's ``model.safetensors`` (as ``Flux3Checkpoint`` validated it) and asserts
coverage both ways: every buffer written, every package tensor consumed. Each tensor is
moved to the model's device before its transform, as the built-in loader did.

The package's precision layout (routing.py checks it): the six projections of
content_mode_blocks.image.1-4, image_cond.0-4 and single_blocks.0-23 are NVFP4 (E2M1
weight, per-16 E4M3 scales in linear layout, ``weight_scale_2``, ``input_scale``); every
other tensor is BF16.

Load-time transforms, each reproducing the bytes the built-in ``Flux3Pipeline`` loader
produces:

* **NVFP4 projections.** 128x4-swizzled scales (``block_scale_interleave``). QKV is
  ``[q ; k ; v]`` (weights and linear scales concatenated, then swizzled) with one alpha
  per plane. gate_up is the package's ``mlp_in`` ``[gate | up]`` rows regrouped as
  ``[up, gate]`` pairs, swizzled, then permuted for the SwiGLU epilogue (v2w). Scalars,
  as the built-in loader computes them: the activation global scale ``g = fp32(1 / input_scale)``;
  ``alpha = fp32(double(1 / g) * double(weight_scale_2))`` (Python floats, then one
  rounding).
* **single_blocks.24-27** (BF16 in the package): each projection (gate_up as one
  ``[gate | up]`` weight) quantized to NVFP4 with ModelOpt's recipe (per-tensor
  ``weight_scale_2 = amax / (6 * 448)``, per-16 E4M3 scales) and a unit activation scale,
  then the transforms above. PRECISION: W4A4 where the package has BF16.
* **content_mode_blocks.image.0** (BF16): QKV (``[q ; k ; v]``), attn_out, gate_up
  (``[gate | up]``) and down quantized per output channel to E4M3 with FP32 ``amax / 448``
  scales. PRECISION: W8A8 rowwise where the package has BF16.
* **Text blocks**: the BF16 projections, plus ``cat([wq, wk, wv])`` for the fused QKV GEMM.

Every NVFP4 block's QKV and gate_up must read one activation global scale (they share
the adaLN head's NVFP4 activation); the package's q / k / v / mlp_in input scales are
calibrated on that one activation, and the loader asserts they agree.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .ops.gemm._prepare import (
    Nvfp4Linear,
    prepare_fp8r_block,
    prepare_nvfp4_block,
    quantize_bf16_to_nvfp4,
)

#: Blocks the package keeps BF16 that the target converts to NVFP4 at load.
_CONVERTED_SINGLE = range(24, 28)
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")

u8 = torch.uint8


class _Reader:
    """The package tensors, moved to the model's device on read; records what was read and
    which buffers were written."""

    def __init__(self, weights: dict[str, Tensor], device: torch.device):
        self.weights = weights
        self.device = device
        self.consumed: set[str] = set()
        self.written: set[int] = set()

    def __call__(self, key: str) -> Tensor:
        self.consumed.add(key)
        return self.weights[key].to(self.device)

    def finish(self) -> None:
        unused = sorted(set(self.weights) - self.consumed)
        assert not unused, f"package tensors no buffer reads: {unused[:8]} ({len(unused)})"

    def copy(self, dst: Tensor, src: Tensor, what: str) -> None:
        """Write one buffer (shape and dtype must match exactly)."""
        assert dst.shape == src.shape and dst.dtype == src.dtype, (
            f"{what}: buffer {tuple(dst.shape)} {dst.dtype}, source {tuple(src.shape)} {src.dtype}"
        )
        with torch.no_grad():
            dst.copy_(src)
        self.written.add(id(dst))


def _package_nvfp4(read: _Reader, prefix: str) -> Nvfp4Linear:
    """One package NVFP4 projection as the built-in NVFP4 loader leaves it: scales swizzled,
    ``input_scale = fp32(1 / checkpoint input_scale)``."""
    linear_sf = read(prefix + ".weight_scale")
    swizzled = torch.ops.trtllm.block_scale_interleave(linear_sf.view(u8).contiguous())
    return Nvfp4Linear(
        weight=read(prefix + ".weight").view(u8),
        weight_scale=swizzled.reshape(-1),
        input_scale=(1.0 / read(prefix + ".input_scale").float()).reshape(1),
        weight_scale_2=read(prefix + ".weight_scale_2").float().reshape(1),
    )


def _load_nvfp4_block(block: nn.Module, read: _Reader, prefix: str, converted: bool) -> None:
    if converted:
        q, k, v, out, up, down = (
            quantize_bf16_to_nvfp4(read(f"{prefix}.{n}.weight").to(torch.bfloat16))
            for n in _PROJECTIONS
        )
    else:
        q, k, v, out, up, down = (_package_nvfp4(read, f"{prefix}.{n}") for n in _PROJECTIONS)
    prepared = prepare_nvfp4_block(prefix, q, k, v, out, up, down)
    for buffer, field in (
        ("qkv_weight", "qkv_weight"),
        ("qkv_weight_sf", "qkv_weight_scale"),
        ("qkv_alpha", "qkv_alpha"),
        ("gate_up_weight", "gate_up_weight"),
        ("gate_up_weight_sf", "gate_up_weight_scale"),
        ("gate_up_alpha", "gate_up_alpha"),
        ("gate_up_act_global", "act_global_scale"),
        ("down_weight", "down_weight"),
        ("down_weight_sf", "down_weight_scale"),
        ("down_alpha", "down_alpha"),
        ("down_act_global", "down_act_global_scale"),
        ("out_weight", "attn_out_weight"),
        ("out_weight_sf", "attn_out_weight_scale"),
        ("out_alpha", "attn_out_alpha"),
        ("out_act_global", "attn_out_act_global_scale"),
    ):
        read.copy(getattr(block, buffer), getattr(prepared, field), f"{prefix} {buffer}")
    _load_norms(block, read, prefix)


def _load_fp8r_block(block: nn.Module, read: _Reader, prefix: str) -> None:
    q, k, v, out, up, down = (read(f"{prefix}.{n}.weight").to(torch.bfloat16) for n in _PROJECTIONS)
    prepared = prepare_fp8r_block(q, k, v, out, up, down)
    for name, field in (
        ("qkv", "qkv"),
        ("out", "attn_out"),
        ("gate_up", "gate_up"),
        ("down", "down"),
    ):
        read.copy(
            getattr(block, f"{name}_weight"),
            getattr(prepared, f"{field}_weight"),
            f"{prefix} {name}",
        )
        read.copy(
            getattr(block, f"{name}_weight_scale"),
            getattr(prepared, f"{field}_scale"),
            f"{prefix} {name} scale",
        )
    _load_norms(block, read, prefix)


def _load_text_block(block: nn.Module, read: _Reader, prefix: str) -> None:
    q, k, v = (read(f"{prefix}.{n}.weight") for n in ("q_proj", "k_proj", "v_proj"))
    for name, src in (
        ("wq", q),
        ("wk", k),
        ("wv", v),
        ("wqkv", torch.cat([q, k, v], 0)),
        ("w_out", read(prefix + ".attn_out.weight")),
        ("w_gate_up", read(prefix + ".mlp_in.weight")),
        ("w_down", read(prefix + ".mlp_out.weight")),
    ):
        read.copy(getattr(block, name), src.to(torch.bfloat16), f"{prefix} {name}")
    _load_norms(block, read, prefix)


def _load_norms(block: nn.Module, read: _Reader, prefix: str) -> None:
    read.copy(
        block.norm_q, read(prefix + ".norm.query_norm.scale").to(torch.bfloat16), prefix + " q"
    )
    read.copy(block.norm_k, read(prefix + ".norm.key_norm.scale").to(torch.bfloat16), prefix + " k")


def load(model: nn.Module, weights: dict[str, Tensor]) -> None:
    """Fill ``model`` (a ``Flux3KleinTransformer``) from the package tensors ``weights``."""
    device = model.final_layer.linear.weight.device
    read = _Reader(weights, device)

    def linear(module: nn.Module, key: str) -> None:
        read.copy(module.weight, read(key).to(torch.bfloat16), key)

    for name, module in model.emb_in.items():
        linear(module, f"emb_in.{name}.weight")
    linear(model.txt_in, "txt_in.weight")
    for embedder in ("time_in", "ref_time_in"):
        for layer in ("in_layer", "out_layer"):
            linear(getattr(getattr(model, embedder), layer), f"{embedder}.{layer}.weight")
    for stage in ("early_stream_modulations", "single_stream_modulations"):
        for name, module in getattr(model, stage).items():
            linear(module.lin, f"{stage}.{name}.lin.weight")
    linear(model.final_layer.linear, "final_layer.image.linear.weight")
    linear(model.final_layer.adaLN_modulation[1], "final_layer.image.adaLN_modulation.1.weight")

    blocks = [(f"content_mode_blocks.image.{i}", b) for i, b in enumerate(model.image_blocks)]
    blocks += [(f"content_mode_blocks.image_cond.{i}", b) for i, b in enumerate(model.cond_blocks)]
    for prefix, block in blocks:
        if prefix == "content_mode_blocks.image.0":
            _load_fp8r_block(block, read, prefix)
        else:
            _load_nvfp4_block(block, read, prefix, converted=False)
    for i, block in enumerate(model.single_blocks):
        _load_nvfp4_block(block, read, f"single_blocks.{i}", converted=i in _CONVERTED_SINGLE)
    for i, block in enumerate(model.text_blocks):
        _load_text_block(block, read, f"txt_mode_blocks.{i}")
    read.finish()

    # The LayerNorms' unit weight / zero bias buffers are built, not loaded.
    missing = [
        name
        for name, buffer in model.named_buffers()
        if id(buffer) not in read.written and "norm_final." not in name and "pre_norm." not in name
    ]
    assert not missing, f"buffers no package tensor wrote: {missing[:8]} ({len(missing)})"
    # Frees the converted blocks' BF16 sources and the transforms' temporaries.
    torch.cuda.empty_cache()
