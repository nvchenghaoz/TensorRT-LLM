# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VAE decoder's tail in one kernel: last ResnetBlock bias + residual add, norm_out
(GroupNorm affine) + swish, conv_out (3x3, 128 -> 3) + bias.

Triton (``_add_norm_swish_conv_out``): the sum and the normalized activation are formed
in registers as the conv input is loaded and are never stored; the conv accumulates in
fp32 with split-TF32 (tf32x3) dots. It replaces the residual add, the norm apply + swish
pass writing an fp32 copy, cuDNN's convertTensor of that copy, a TF32 implicit GEMM
padded from N = 3 to a full tile, and the bias pass.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._add_norm_swish_conv_out import BC, add_norm_swish_conv_out_3x3_op  # noqa: F401 — registers
from ._ref import true_fp32_conv


class AddNormSwishConvOut3x3(OpWrapper):
    """`y = conv2d(swish(h * scale_c + shift_c), weight, bias, padding=1)` with
    `h = x + (r + r_bias_c)`, each sum rounded to x's dtype; everything after in fp32.

    x: the last ResnetBlock's shortcut, bf16 `[N, C, H, W]`; r: its conv2 output without
    the bias, x's dtype, shape and strides; r_bias: conv2's bias `[C]`, x's dtype;
    scale, shift: fp32 `[N, C]` contiguous, norm_out's per-channel GroupNorm affine
    (`group_norm_channel_affine` on `h`); weight: fp32 conv_out weight `[3, C, 3, 3]`
    (any strides); bias: fp32 `[3]`. Returns fp32 `[N, 3, H, W]`, channels_last when x
    is.

    Fusion boundary: everything from the residual add to conv_out's bias. norm_out's
    statistics (a reduction over the whole of `h`) are the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 output: 128ch @2048^2 channels_last", spec=dict(c=128, h=2048)),
        Cell(why="1024^2 output: 128ch @1024^2 channels_last", spec=dict(c=128, h=1024)),
    )

    note = f"""
    Metadata consumed: none.

    Launch: grid (ceil(W / 62), ceil(H / 16), N), 4 warps, 2 stages; BC = {BC} channels
    per dot step, so C must be a multiple of {BC}.

    Numerics: the two sums rounded to x's dtype exactly as the unfused adds store them;
    the GroupNorm affine and swish in fp32 (the normalized activation is not rounded to
    bf16, unlike the unfused path); 3-pass split-TF32 products (the lo*lo term dropped,
    ~2^-22 relative) accumulated in fp32; fp32 bias; fp32 output.

    Not checked by the op (is_valid covers it): x and r sharing dtype, shape and strides;
    C a multiple of {BC}; scale / shift contiguous [N, C].
    """

    def __call__(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        r_bias: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.flux3_klein_add_norm_swish_conv_out_3x3(
            x, r, r_bias, scale, shift, weight, bias
        )

    def reference(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        r_bias: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """The same roundings of the two sums, then fp32 (TF32 off) to the end."""
        n, c = x.shape[:2]
        rr = (r.float() + r_bias.float().view(1, c, 1, 1)).to(x.dtype)
        h = (x.float() + rr.float()).to(x.dtype).float()
        a = h * scale.float().view(n, c, 1, 1) + shift.float().view(n, c, 1, 1)
        a = a * torch.sigmoid(a)
        with true_fp32_conv():
            y = torch.nn.functional.conv2d(a, weight.float(), bias.float(), padding=1)
        return y

    def is_valid(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        r_bias: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        assert r.dtype == x.dtype and r.shape == x.shape and r.stride() == x.stride(), (
            "r must share x's dtype, shape and strides: the kernel reads both at x's offsets"
        )
        assert x.shape[1] % BC == 0, f"C must be a multiple of {BC} (unmasked channel loads)"
        n, c = x.shape[:2]
        for name, t in (("scale", scale), ("shift", shift)):
            assert t.dtype == torch.float32 and t.shape == (n, c) and t.is_contiguous(), (
                f"{name} must be contiguous fp32 [N, C]: the kernel reads it at n * C + c"
            )
        assert tuple(weight.shape) == (3, c, 3, 3) and bias.numel() == 3, (
            "weight must be [3, C, 3, 3] and bias [3]: the kernel computes exactly 3 outputs"
        )


add_norm_swish_conv_out_3x3 = AddNormSwishConvOut3x3()
