# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E4M3 3x3 conv of the VAE decoder's ResnetBlocks, scale / bias / residual fused.

cuDNN frontend graph (``_fp8_conv3x3``): conv_fprop e4m3 x e4m3 with fp32 accumulation,
then ``* s_x * s_w[o] (+ bias[o]) (+ residual)`` in fp32 and one bf16 rounding, NHWC in
and out. Every decoder ResnetBlock conv1 / conv2 (28 convs) runs here, fed by the
GroupNorm + swish + e4m3 quantize prologue the compiled decoder traces in torch.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8_conv3x3 import PLAN_K24, fp8_conv3x3_op  # noqa: F401 — registers the op
from ._ref import true_fp32_conv

#: Graph variants the decoder calls. sb: conv1 (bf16 bias); sbr: conv2 of a same-channel
#: block (bf16 bias + residual); fb_r: conv2 of a 1x1-shortcut block (fp32 bias = conv2 +
#: shortcut bias, residual = the bias-free shortcut); s: the last block's conv2, whose bias
#: and residual the fused conv_out tail adds.
_VARIANTS = {"sb": (1, False), "sbr": (1, True), "fb_r": (2, True), "s": (0, False)}


def _decoder_cells(out: int) -> tuple[Cell, ...]:
    """The decoder's (Cin, Cout, H) keys and variants at an ``out`` x ``out`` image."""
    m = out // 8  # mid-block / up[3] resolution
    calls = (
        ((512, 512, m), ("sb", "sbr"), "mid block_1/2 and up[3]"),
        ((512, 512, 2 * m), ("sb", "sbr"), "up[2]"),
        ((512, 256, 4 * m), ("sb",), "up[1].block[0].conv1"),
        ((256, 256, 4 * m), ("fb_r", "sb", "sbr"), "up[1]: block[0].conv2, blocks 1-2"),
        ((256, 128, 8 * m), ("sb",), "up[0].block[0].conv1"),
        ((128, 128, 8 * m), ("fb_r", "sb", "sbr", "s"), "up[0]: block[0].conv2, 1, 2"),
    )
    cells = []
    for (cin, cout, h), variants, where in calls:
        for v in variants:
            has_bias, has_res = _VARIANTS[v]
            cells.append(
                Cell(
                    why=f"{out}^2 {where}: {cin}->{cout} @{h}^2 {v}",
                    spec=dict(cin=cin, cout=cout, h=h, has_bias=has_bias, has_res=has_res),
                )
            )
    return tuple(cells)


class Fp8Conv3x3(OpWrapper):
    """`y = bf16(conv3x3(xq, wq) * s_x * s_w[o] (+ bias[o]) (+ residual))`, padding 1.

    xq: e4m3 `[N, Cin, H, W]` channels_last; s_x: fp32 `[1]`; wq: e4m3 KRSC
    `[Cout, 3, 3, Cin]` contiguous; s_w: fp32 `[Cout]`; bias: `[Cout]` (fp32 -> added in
    fp32; bf16 -> as is) or None; residual: bf16 `[N, Cout, H, W]` or None. Returns bf16
    `[N, Cout, H, W]` channels_last.

    Fusion boundary: the conv and its epilogue. Quantizing the activation (GroupNorm +
    swish -> e4m3, per-tensor scale) and the weights (per-Cout scale) is the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = _decoder_cells(2048) + _decoder_cells(1024)

    note = f"""
    Metadata consumed: none.

    Plan: one cuDNN FE graph per (N, Cin, Cout, H, W, bias kind, residual), built at the
    first call (warmup); it runs the engine-7 plan whose knob k24 is pinned for its
    (Cin, Cout, H) ({len(PLAN_K24)} pinned shapes, every 1024^2 and 2048^2 decoder key).
    A pinned plan this cuDNN does not list is an error, not a fallback: another plan
    changes the accumulation order. An unpinned shape runs the first engine-7 plan.

    Numerics: e4m3 x e4m3 products are exact in fp32; the fp32 sum over 9 * Cin taps is
    in the engine's order; scale, bias and residual in fp32; one bf16 rounding. Against
    an fp32 reference on the dequantized operands it differs only by that order (within
    one bf16 rounding step).

    Not checked by the op (is_valid covers it): xq channels_last e4m3, wq KRSC e4m3 with
    Cin a multiple of 16, residual shape.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        s_x: torch.Tensor,
        wq: torch.Tensor,
        s_w: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return torch.ops.trtllm.flux3_klein_fp8_conv3x3(xq, s_x, wq, s_w, bias, residual)

    def reference(
        self,
        xq: torch.Tensor,
        s_x: torch.Tensor,
        wq: torch.Tensor,
        s_w: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """fp32 conv of the dequantized operands (TF32 off), fp32 epilogue, one bf16
        rounding."""
        x = xq.float() * s_x.float().view(1, 1, 1, 1)
        w = wq.float().permute(0, 3, 1, 2) * s_w.float().view(-1, 1, 1, 1)
        with true_fp32_conv():
            y = torch.nn.functional.conv2d(x, w, None, padding=1)
        if bias is not None:
            b = bias if bias.dtype == torch.float32 else bias.to(torch.bfloat16)
            y = y + b.float().view(1, -1, 1, 1)
        if residual is not None:
            y = y + residual.to(torch.bfloat16).float()
        return y.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)

    def is_valid(
        self,
        xq: torch.Tensor,
        s_x: torch.Tensor,
        wq: torch.Tensor,
        s_w: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
    ) -> None:
        assert xq.dtype == torch.float8_e4m3fn and wq.dtype == torch.float8_e4m3fn, (
            "xq and wq must be e4m3: the graph reads them as FP8_E4M3 whatever their dtype"
        )
        assert wq.dim() == 4 and tuple(wq.shape[1:3]) == (3, 3) and wq.shape[3] == xq.shape[1], (
            "wq must be KRSC [Cout, 3, 3, Cin]: an OIHW weight is read with the wrong taps"
        )
        assert xq.shape[1] % 16 == 0 and wq.shape[0] % 16 == 0, (
            "Cin and Cout must be multiples of 16 (the e4m3 conv engines' vector width)"
        )
        assert s_w.numel() == wq.shape[0] and s_x.numel() == 1, "s_x is [1], s_w is [Cout]"
        if residual is not None:
            assert tuple(residual.shape) == (xq.shape[0], wq.shape[0], *xq.shape[2:]), (
                "residual must be [N, Cout, H, W]"
            )


fp8_conv3x3 = Fp8Conv3x3()
