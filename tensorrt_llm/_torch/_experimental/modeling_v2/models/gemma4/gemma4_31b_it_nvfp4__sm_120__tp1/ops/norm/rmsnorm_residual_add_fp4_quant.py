# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The post-attention junction in one kernel: norm + residual add, then norm + NVFP4 quantize.

`h = residual + rmsnorm(x, w1)` and the NVFP4 payload and 128x4-swizzled scales of
`rmsnorm(h, w2)` quantized with `global_scale` (the gate_up projection's static input scale),
from one read of `x` and `residual`.

Fusion boundary: it replaces the built-in model's `rmsnorm_residual_add` (the same Triton norm
+ add, so `h` is bitwise that kernel's) and FlashInfer's CuTe-DSL `rmsnorm_fp4quant` on `h`, whose
payload and full swizzled scale buffer (padding included) it reproduces bitwise: the second
norm's sum of squares replays the CuTe kernel's per-thread fma chain and lane tree, its mean an
IEEE divide. Against FlashInfer's own rmsnorm, `h` differs by reduction order (about 5e-6 of
elements by one bf16 ulp).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rmsnorm_residual_add import rmsnorm_residual_add_fp4_quant as _kernel
from ._rmsnorm_residual_add import rmsnorm_residual_add_fp4_quant_supported


class RmsnormResidualAddFp4Quant(OpWrapper):
    """`(h, fp4, sf)`: h [M, H] bf16, fp4 [M, H/2] uint8, sf 1-D uint8 (pad128(M) rows)."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(why="one row (decode)", spec=dict(m=1, h=5376)),
        Cell(why="one 300-token prompt", spec=dict(m=300, h=5376)),
        Cell(why="M not a multiple of 128: the scale padding rows", spec=dict(m=1000, h=5376)),
        Cell(why="the largest prefill step", spec=dict(m=8192, h=5376)),
    )
    note = """
    Hard limits: bf16; 3072 < H <= 6144 and H % 16 == 0 (64 threads per row, one CTA per row).
    """

    def __call__(self, x, residual, norm_weight, eps, next_norm_weight, next_norm_eps, global_scale):
        return _kernel(x, residual, norm_weight, eps, next_norm_weight, next_norm_eps, global_scale)

    def reference(self, x, residual, norm_weight, eps, next_norm_weight, next_norm_eps, global_scale):
        """The unfused chain: FlashInfer rmsnorm, the bf16 residual add, then FlashInfer's
        rmsnorm + NVFP4 quantize."""
        from flashinfer.norm import rmsnorm, rmsnorm_fp4quant

        h = residual + rmsnorm(x, norm_weight, eps)
        fp4, sf = rmsnorm_fp4quant(h, next_norm_weight, global_scale, eps=next_norm_eps)
        return h, fp4.view(torch.uint8), sf.view(torch.uint8).reshape(-1)

    def is_valid(self, x, residual, norm_weight, eps, next_norm_weight, next_norm_eps, global_scale):
        assert rmsnorm_residual_add_fp4_quant_supported(x.shape[-1]), x.shape


rmsnorm_residual_add_fp4_quant = RmsnormResidualAddFp4Quant()
