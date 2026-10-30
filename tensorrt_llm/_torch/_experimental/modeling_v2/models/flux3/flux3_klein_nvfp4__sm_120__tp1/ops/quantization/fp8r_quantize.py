# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-token E4M3 quantize of a BF16 activation for the FP8 row-wise GEMM (attn_out's input
in content_mode_blocks.image.0): the one-pass Triton kernel from 16384 rows at K <= 4096,
``torch.ops.trtllm.warp_fp8_per_token_quant`` below (both the same recipe, bitwise)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8r_triton import one_pass_fp8_quant

#: From this many rows (with K <= ONE_PASS_MAX_K) the one-pass kernel (2048^2).
ONE_PASS_MIN_ROWS = 16384
ONE_PASS_MAX_K = 4096


class Fp8rQuantize(OpWrapper):
    """``(data [M, K] float8_e4m3fn, scales [M] fp32)``: ``scales = max(amax(|x|) / 448,
    1e-12)`` per row, ``data = e4m3(clamp(x / scale, +-448))``.

    ``x`` ``[..., K]`` bf16 (M = x.numel() / K rows).
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 image.0 attention output: one-pass kernel", spec=dict(m=16384, k=3072)),
        Cell(
            why="1024^2 image.0 attention output: warp_fp8_per_token_quant",
            spec=dict(m=4096, k=3072),
        ),
    )

    note = """
    Metadata consumed: none. content_mode_blocks.image.0's attn_out input.

    Numerics: exact -- the round-to-nearest division and E4M3 rounding of `reference`.
    """

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        if rows.shape[0] >= ONE_PASS_MIN_ROWS and rows.shape[1] <= ONE_PASS_MAX_K:
            return one_pass_fp8_quant(rows)
        data, scales = torch.ops.trtllm.warp_fp8_per_token_quant(rows, 1e-12)
        return data, scales.reshape(-1)

    def reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-token E4M3 recipe in fp32 with exact division."""
        rows = x.reshape(-1, x.shape[-1]).float()
        amax = rows.abs().amax(dim=1)
        # A tensor divisor: torch turns `/ 448.0` (a CPU scalar) into a multiply by the
        # rounded reciprocal, which is not the round-to-nearest division of the recipe.
        scales = torch.clamp(amax / torch.full_like(amax, 448.0), min=1e-12)
        data = (rows / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return data, scales


fp8r_quantize = Fp8rQuantize()
