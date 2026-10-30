# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VAE decoder's upsample (nearest-2x + 3x3 conv + bias) as one stride-2 cuDNN dgrad.

cuDNN frontend graph (``_subpixel_up_conv``): conv_dgrad of the low-resolution input with
the 4x4 transposed sub-pixel weight, + bias in the fp32 epilogue, one bf16 rounding, NHWC
interleaved output. Replaces the upsampled 3x3 conv (4/9 of its FLOPs), the interleave
pass and the separate bias pass. All three decoder upsamples run here.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._ref import true_fp32_conv
from ._subpixel_up_conv import PLAN_NAME, subpixel_up_conv_op  # noqa: F401 — registers the op


def _cells(out: int) -> tuple[Cell, ...]:
    m = out // 8
    return (
        Cell(why=f"{out}^2 up[3]: 512ch {m}^2 -> {2 * m}^2", spec=dict(c=512, h=m)),
        Cell(why=f"{out}^2 up[2]: 512ch {2 * m}^2 -> {4 * m}^2", spec=dict(c=512, h=2 * m)),
        Cell(why=f"{out}^2 up[1]: 256ch {4 * m}^2 -> {8 * m}^2", spec=dict(c=256, h=4 * m)),
    )


class SubpixelUpConv(OpWrapper):
    """`y = bf16(conv3x3(nearest2x(x), W) + b)` computed as
    `conv_transpose2d(x, wt, stride=2, padding=1)` with an fp32 bias epilogue.

    x: bf16 `[N, C_in, H, W]` channels_last; wt: `[C_in, C_out, 4, 4]` (the decoder holds it
    bf16 channels_last; cast and laid out by the op otherwise), derived from the 3x3 conv
    weight by `subpixel_transpose_weight`; bias: `[C_out]` (rounded to bf16 by the op).
    Returns bf16 `[N, C_out, 2H, 2W]` channels_last.

    Fusion boundary: the upsample conv and its bias. Deriving wt is load-time work.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = _cells(2048) + _cells(1024)

    note = f"""
    Metadata consumed: none.

    Plan: one cuDNN FE graph per (N, C_in, C_out, H, W), built at the first call; it
    runs the plan pinned by exact name for its (C_in, C_out, H) ({len(PLAN_NAME)} pinned
    shapes: every 1024^2 and 2048^2 decoder upsample). A pinned plan this cuDNN does not
    list is an error, not a fallback. An unpinned shape runs the heuristic's first plan.

    Numerics: wt holds the 3x3 taps summed per phase (fp64 sums, rounded once to the
    weight dtype, then to bf16 with the decoder trunk); the dgrad accumulates bf16
    products in fp32 in the engine's order; the bias is added in fp32 and the sum
    rounded once to bf16 (the unfused path rounds the conv, then the bias add).

    Not checked by the op (is_valid covers it): x bf16 channels_last, wt's [C_in, C_out,
    4, 4] orientation.
    """

    def __call__(self, x: torch.Tensor, wt: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.flux3_klein_subpixel_up_conv(x, wt, bias)

    def reference(self, x: torch.Tensor, wt: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """fp32 conv_transpose2d of the same bf16 operands (TF32 off), fp32 bias, one bf16
        rounding."""
        with true_fp32_conv():
            y = torch.nn.functional.conv_transpose2d(
                x.float(), wt.to(torch.bfloat16).float(), None, stride=2, padding=1
            )
        y = y + bias.to(torch.bfloat16).float().view(1, -1, 1, 1)
        return y.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)

    def is_valid(self, x: torch.Tensor, wt: torch.Tensor, bias: torch.Tensor) -> None:
        assert x.dtype == torch.bfloat16, "x must be bf16: the graph reads it as BFLOAT16"
        assert wt.dim() == 4 and wt.shape[0] == x.shape[1] and tuple(wt.shape[2:]) == (4, 4), (
            "wt must be [C_in, C_out, 4, 4] (the transposed sub-pixel weight), not the 3x3 "
            "conv weight"
        )
        assert bias.numel() == wt.shape[1], "bias must be [C_out]"
        assert x.shape[1] % 8 == 0 and wt.shape[1] % 8 == 0, "channels must be multiples of 8"


subpixel_up_conv = SubpixelUpConv()
