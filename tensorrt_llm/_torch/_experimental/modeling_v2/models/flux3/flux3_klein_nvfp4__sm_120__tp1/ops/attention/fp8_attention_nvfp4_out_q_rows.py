# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``fp8_attention_nvfp4_out`` on a Q-row prefix
(``torch.ops.trtllm.flux3_fp8_attention_nvfp4_out_q_rows``): the attention of the last single
block, whose output only the image rows feed.

The grid covers only the CTAs of Q rows ``[0, q_rows)`` (each computes its rows over every K/V
row exactly as the full launch does), so the text rows' attention is skipped.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ._fp8_attention import emulated_attention, planes


def _cell(why: str, s: int, q_rows: int) -> Cell:
    return Cell(why=why, spec=dict(batch=1, seq_len=s, heads=24, q_rows=q_rows))


class Fp8AttentionNvfp4OutQRows(OpWrapper):
    """``fp8_attention_nvfp4_out`` where only Q rows ``[0, q_rows)`` are defined: fp4 rows
    ``[0, q_rows)`` and the scale bytes of those rows are bitwise the full launch's. Rows up to
    ``min(S, round_up(q_rows, 192))`` are computed as well; later rows and the scale padding
    rows are not written (uninitialized). ``q_rows == 0`` or ``S`` runs every row.

    Arguments as ``fp8_attention_nvfp4_out``, plus ``q_rows`` (``0 <= q_rows <= S``, B = 1 in
    the target).
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        _cell("i2i last single block at 2048^2: 16384 image rows of 32848 (86 CTAs)", 32848, 16384),
        _cell("t2i last single block at 2048^2: the 16384 image rows of 16464", 16464, 16384),
        _cell("last single block at 1024^2: the 4096 image rows of 4176", 4176, 4096),
    )

    note = """
    Metadata consumed: none.

    The caller reads fp4 rows [0, q_rows) and their scale bytes (with the target's keep_rows,
    a multiple of 256, the scale buffer's first `q_rows * round_up(H * 8, 4)` bytes). Anything
    past them is undefined; a consumer GEMM at M = q_rows reads the rest of the last 128-row
    block only as padding.

    Numerics: those rows as fp8_attention_nvfp4_out.
    """

    def __call__(
        self,
        qkv8: torch.Tensor,
        scales: torch.Tensor,
        batch: int,
        heads: int,
        global_scale: torch.Tensor,
        q_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q8, k8, v8 = planes(qkv8, batch, heads)
        fp4, sf = torch.ops.trtllm.flux3_fp8_attention_nvfp4_out_q_rows(
            q8, k8, v8, scales[3:4], scales[2:3], global_scale, q_rows
        )
        return fp4, sf

    def reference(
        self,
        qkv8: torch.Tensor,
        scales: torch.Tensor,
        batch: int,
        heads: int,
        global_scale: torch.Tensor,
        q_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`fp8_attention_nvfp4_out.reference` with only rows [0, q_rows) of O computed (later O
        rows zero, so their codes and scales are zero; only the defined prefix is comparable)."""
        o = emulated_attention(qkv8, scales, batch, heads, q_rows)
        b, s, h, d = o.shape
        fp4, sf = fp4_quantize.reference(o.reshape(b * s, h * d), global_scale, 16, False, True)
        return fp4.view(b, s, h * d // 2), sf


fp8_attention_nvfp4_out_q_rows = Fp8AttentionNvfp4OutQRows()
