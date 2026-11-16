# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The layer tail in one kernel: norm + residual add + layer scalar (+ the next input norm).

`out = bf16(fp32(residual + rmsnorm(x, w)) * scale)` with `scale` the fp32 layer scalar, and,
when `next_norm_weight` is given, `rmsnorm(out, next_norm_weight)` as a second output (the next
decoder layer's input norm).

Fusion boundary: it replaces the norm, the bf16 add, the fp32 scalar multiply and the next
layer's standalone norm with the built-in model's fused tail kernel (the fp32 `[M, H]`
intermediate is never written). Its norms reduce with Triton's `tl.sum`, so against FlashInfer's
rmsnorm chain the outputs differ by reduction order (about 5e-6 of elements by one bf16 ulp).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rmsnorm_residual_add import rmsnorm_residual_add_scale as _kernel


class RmsnormResidualAddScale(OpWrapper):
    """`(out, normed_next)`; `normed_next` is None without `next_norm_weight`."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(why="decode row, next norm fused", spec=dict(m=1, h=5376, next_norm=True)),
        Cell(why="prefill, next norm fused", spec=dict(m=1000, h=5376, next_norm=True)),
        Cell(why="the last layer: no next norm", spec=dict(m=300, h=5376, next_norm=False)),
    )

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
        next_norm_weight: Optional[torch.Tensor],
        next_norm_eps: float,
    ):
        if next_norm_weight is None:
            return _kernel(x, residual, norm_weight, scale, eps), None
        return _kernel(x, residual, norm_weight, scale, eps, next_norm_weight, next_norm_eps)

    def reference(self, x, residual, norm_weight, scale, eps, next_norm_weight, next_norm_eps):
        """The unfused chain: FlashInfer rmsnorm, the bf16 add, the fp32 scalar multiply (aten
        promotes to fp32 for the fp32 scalar), the bf16 cast, the next FlashInfer rmsnorm."""
        from flashinfer.norm import rmsnorm

        out = ((residual + rmsnorm(x, norm_weight, eps)) * scale).to(torch.bfloat16)
        if next_norm_weight is None:
            return out, None
        return out, rmsnorm(out, next_norm_weight, next_norm_eps)


rmsnorm_residual_add_scale = RmsnormResidualAddScale()
