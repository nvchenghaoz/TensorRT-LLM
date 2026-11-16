# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding lookup * scale and the first layer's input norm in one launch, plus `rmsnorm`.

`embed_scale_rmsnorm(ids, table, scale, norm_weight, eps)` returns `(h, normed)`:
`h = bf16(table[ids] * scale)` (ids outside the table give zero rows) and `normed =
rmsnorm(h, norm_weight)`, or `normed = None` where the fused norm would not replay the
FlashInfer rmsnorm this GPU runs (`rmsnorm_replay_supported`); the caller then runs `rmsnorm`.

Fusion boundary: the masked lookup, the bf16 scalar multiply (fp32 opmath, one rounding) and
FlashInfer's CuTe-DSL rmsnorm; both outputs are bitwise those of that chain.
"""

from __future__ import annotations

import torch
from flashinfer.norm import rmsnorm as _flashinfer_rmsnorm

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._embed_scale_rmsnorm import embed_scale_rmsnorm as _kernel
from ._embed_scale_rmsnorm import rmsnorm_replay_supported


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FlashInfer's rmsnorm (the stock bf16 RMSNorm's kernel)."""
    return _flashinfer_rmsnorm(x, weight, eps, enable_pdl=get_env_enable_pdl())


class EmbedScaleRmsnorm(OpWrapper):
    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(why="decode batch", spec=dict(m=8, hidden=5376, vocab=262144)),
        Cell(why="prompt with out-of-range ids (zero rows)", spec=dict(m=300, hidden=5376, vocab=262144, oob=True)),
    )

    def __call__(self, input_ids, table, scale, norm_weight, eps):
        use_norm = rmsnorm_replay_supported(table.shape[1], table.dtype, table.device)
        return _kernel(input_ids, table, scale, norm_weight if use_norm else None, eps)

    def reference(self, input_ids, table, scale, norm_weight, eps):
        ids = input_ids.reshape(-1).long()
        valid = (ids >= 0) & (ids < table.shape[0])
        rows = table[ids.clamp(0, table.shape[0] - 1)] * valid[:, None].to(table.dtype)
        h = rows * torch.tensor(scale, dtype=table.dtype, device=table.device)
        return h, rmsnorm(h, norm_weight, eps)


embed_scale_rmsnorm = EmbedScaleRmsnorm()
