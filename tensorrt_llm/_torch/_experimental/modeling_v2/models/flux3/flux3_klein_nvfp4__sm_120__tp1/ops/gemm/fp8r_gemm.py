# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 row-wise GEMM of the BF16-excluded mode block content_mode_blocks.image.0
(``torch.ops.trtllm.flux3_fp8r_gemm``): ``bf16(w_scale * (x_scale * (x @ w.T)))``.

An SM120 CUTLASS E4M3 x E4M3 GEMM with fp32 accumulation, CLC persistent scheduler, PDL and
an EVT epilogue of two fp32 RN multiplies, bitwise ``torch._scaled_mm(x, w.t(), x_scale,
w_scale, out_dtype=bf16)``. The tile by shape (`fp8r_arm`): Pingpong 128x128x64 below
8192 rows, Cooperative 128x256x64 for gate_up (N 18432) and 256x128x64 for the others at
2048^2.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from . import _cpp_fakes  # noqa: F401 — fake for torch.compile tracing

LARGE_M_MIN = 8192


def fp8r_arm(rows: int, columns: int) -> int:
    """Tile of an [rows, columns] output: 1 = Pingpong 128x128x64, 3 = Cooperative
    128x256x64 (wide N), 4 = Cooperative 256x128x64."""
    if rows < LARGE_M_MIN:
        return 1
    return 3 if columns >= 12288 else 4


class Fp8rGemm(OpWrapper):
    """``bf16(w_scale[n] * (x_scale[m] * sum_k x[m, k] * w[n, k]))`` as ``[M, N]`` bf16.

    ``x`` E4M3 ``[M, K]`` / ``x_scale`` fp32 ``[M]``: the per-token activation;
    ``w`` E4M3 ``[N, K]`` / ``w_scale`` fp32 ``[N]``: `_prepare.prepare_fp8r_block`'s weights.
    """

    ARCHS = frozenset({Arch.SM_120})

    # image.0's four projections at 2048^2 (16384 rows) and 1024^2 (4096 rows).
    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 QKV 9216 x 3072, 256x128 tile", spec=dict(m=16384, n=9216, k=3072)),
        Cell(why="2048^2 gate_up 18432 x 3072, 128x256 tile", spec=dict(m=16384, n=18432, k=3072)),
        Cell(why="2048^2 down 3072 x 9216", spec=dict(m=16384, n=3072, k=9216)),
        Cell(why="2048^2 attn_out 3072 x 3072", spec=dict(m=16384, n=3072, k=3072)),
        Cell(why="1024^2 QKV, Pingpong 128x128", spec=dict(m=4096, n=9216, k=3072)),
        Cell(why="1024^2 gate_up", spec=dict(m=4096, n=18432, k=3072)),
        Cell(why="1024^2 down", spec=dict(m=4096, n=3072, k=9216)),
    )

    note = """
    Metadata consumed: none. content_mode_blocks.image.0 only (QKV, gate_up, down,
    attn_out).

    Numerics: a sequential k32 MMA chain into one fp32 accumulator and BF16(w_scale *
    (x_scale * acc)) with two RN multiplies: bitwise torch._scaled_mm's SM120 row-wise
    kernel on these shapes, and every tile gives the same bytes. Against `reference` (fp32
    matmul of the decoded operands) the accumulation order differs, so a BF16 output can
    move by one ulp.

    PDL: launched with the PDL attribute.
    """

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.trtllm.flux3_fp8r_gemm(
            x,
            w,
            x_scale.reshape(-1),
            w_scale.reshape(-1),
            fp8r_arm(x.shape[0], w.shape[0]),
        )

    def reference(
        self, x: torch.Tensor, w: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor
    ) -> torch.Tensor:
        """fp32 matmul of the E4M3 values (TF32 off), then ``w_scale * (x_scale * acc)`` in
        fp32, rounded to BF16."""
        with true_fp32_matmul():
            acc = x.float() @ w.float().t()
        return (w_scale.reshape(1, -1).float() * (x_scale.reshape(-1, 1).float() * acc)).to(
            torch.bfloat16
        )


fp8r_gemm = Fp8rGemm()
