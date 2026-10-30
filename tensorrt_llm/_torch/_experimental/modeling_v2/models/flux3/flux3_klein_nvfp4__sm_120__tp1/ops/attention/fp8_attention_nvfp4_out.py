# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 self-attention whose epilogue writes the output projection's NVFP4 input
(``torch.ops.trtllm.flux3_fp8_attention_nvfp4_out``): the attention of every NVFP4 block
(image 1-4, image_cond 0-4, single 0-26).

The same kernel as ``fp8_attention_bf16_out``; instead of storing BF16 O, each 4-warp group
quantizes its O tile to NVFP4 (FP4 + 128x4-swizzled E4M3 block scales) with the caller's global
scale, so attn_out's activation quantize disappears.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ._fp8_attention import emulated_attention, planes


def _cell(why: str, s: int) -> Cell:
    return Cell(why=why, spec=dict(batch=1, seq_len=s, heads=24))


class Fp8AttentionNvfp4Out(OpWrapper):
    """``(fp4 [B, S, H*64] uint8, sf uint8 flat)``: the NVFP4 quantization (block 16, E4M3 scales
    in the 128x4-swizzled layout over ``round_up(B*S, 128)`` rows, padding rows zero) with
    ``global_scale`` of ``fp8_attention_bf16_out``'s BF16 O.

    ``qkv8``, ``scales``: as ``fp8_attention_bf16_out`` (E4M3 ``[3, B*S, H*128]`` planes; FP32
    scales, ``[2]`` the V descale, ``[3]`` ``q_descale * k_descale * sm_scale``);
    ``global_scale``: FP32 ``[1]``, the output projection's activation global scale (its
    ``inverse_scale``).

    Fusion boundary: attention plus attn_out's input quantize. The output projection is the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        _cell("mode blocks at 2048^2 (image 1-4, image_cond 0-4): S = 16384, no mask", 16384),
        _cell("t2i single blocks at 2048^2: S = 16464 = image + 80 text, masked last step", 16464),
        _cell("i2i single blocks at 2048^2: S = 32848, 172 CTAs per head", 32848),
        _cell("mode blocks at 1024^2: S = 4096, last CTA 2 groups", 4096),
        _cell("single blocks at 1024^2: S = 4176, SF padding rows 4176..4223 zeroed", 4176),
    )

    note = """
    Metadata consumed: none.

    Shapes: as fp8_attention_bf16_out.

    Numerics: bitwise `fp4_quantize(O, global_scale, 16, swizzled)` of
    `fp8_attention_bf16_out`'s O (FlashInfer's cute-dsl nvfp4_quantize arithmetic:
    `block_max * rcp.approx.ftz(6)`, `mul.rn` by the global scale, `cvt.rn.satfinite.e4m3`,
    `rcp.approx.ftz(e4m3 * rcp.approx.ftz(gs))`, `cvt.rn.satfinite.e2m1x2`), scale padding
    included. Against `reference` (FP32 emulation + native-torch NVFP4) only where the BF16 O
    differs by its last bits, and with it a block's scale or a code near a rounding midpoint.
    """

    def __call__(
        self,
        qkv8: torch.Tensor,
        scales: torch.Tensor,
        batch: int,
        heads: int,
        global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q8, k8, v8 = planes(qkv8, batch, heads)
        fp4, sf = torch.ops.trtllm.flux3_fp8_attention_nvfp4_out(
            q8, k8, v8, scales[3:4], scales[2:3], global_scale
        )
        return fp4, sf

    def reference(
        self,
        qkv8: torch.Tensor,
        scales: torch.Tensor,
        batch: int,
        heads: int,
        global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`emulated_attention`'s BF16 O through `fp4_quantize.reference` (swizzled scales)."""
        o = emulated_attention(qkv8, scales, batch, heads)
        b, s, h, d = o.shape
        fp4, sf = fp4_quantize.reference(o.reshape(b * s, h * d), global_scale, 16, False, True)
        return fp4.view(b, s, h * d // 2), sf


fp8_attention_nvfp4_out = Fp8AttentionNvfp4Out()
