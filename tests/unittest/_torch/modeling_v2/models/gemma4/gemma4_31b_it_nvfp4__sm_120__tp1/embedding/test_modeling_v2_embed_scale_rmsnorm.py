# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the embed_scale_rmsnorm entry (embedding lookup * sqrt(hidden) + layer 0's
input norm).

Per cell, against the entry's reference (masked lookup, the bf16 scalar multiply, FlashInfer's
rmsnorm): both outputs equal. The kernel replays the CuTe rmsnorm's summation order and
`div.rn`, so the comparison is exact in value; `torch.equal` (not a bit view) because the
reference zeroes an out-of-range row by multiplying it by 0, which keeps the sign of negative
table entries (-0.0) where the kernel loads +0.0. Where the fused norm would not replay the
rmsnorm FlashInfer runs on this GPU the entry returns `normed = None` (the target then runs
`rmsnorm`), which the test accepts only when `rmsnorm_replay_supported` says so.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.embedding._embed_scale_rmsnorm import (  # noqa: E501
    rmsnorm_replay_supported,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.embedding.embed_scale_rmsnorm import (  # noqa: E501
    embed_scale_rmsnorm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import seed_of  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

EPS = 1e-6
_TABLES: dict = {}


def _table(vocab: int, hidden: int) -> torch.Tensor:
    if (vocab, hidden) not in _TABLES:
        g = torch.Generator(device="cuda").manual_seed(vocab + hidden)
        _TABLES[(vocab, hidden)] = (
            torch.randn(vocab, hidden, generator=g, device="cuda") * 0.03
        ).to(torch.bfloat16)
    return _TABLES[(vocab, hidden)]


def _build(spec, seed: int):
    """`(ids, table, scale, norm_weight, eps)`; with `oob`, every seventh id is outside the
    table (negative or >= vocab). The scale is sqrt(hidden) rounded to bf16, as the target
    holds it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, hidden, vocab = spec["m"], spec["hidden"], spec["vocab"]
    ids = torch.randint(0, vocab, (m,), generator=g, device="cuda", dtype=torch.int32)
    if spec.get("oob"):
        ids[::7] = torch.tensor([-1, vocab, vocab + 5, -100], device="cuda", dtype=torch.int32)[
            torch.arange(len(ids[::7]), device="cuda") % 4
        ]
    scale = float(torch.tensor(math.sqrt(hidden), dtype=torch.bfloat16))
    norm_weight = (1 + 0.3 * torch.randn(hidden, generator=g, device="cuda")).to(torch.bfloat16)
    return ids, _table(vocab, hidden), scale, norm_weight, EPS


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    ids, table, scale, norm_weight, eps = _build(cell.spec, seed_of(cell.why))
    h, normed = op(ids, table, scale, norm_weight, eps)
    ref_h, ref_normed = op.reference(ids, table, scale, norm_weight, eps)

    assert h.shape == (ids.numel(), table.shape[1]) and h.dtype == torch.bfloat16
    assert torch.equal(h, ref_h)
    if cell.spec.get("oob"):
        bad = (ids < 0) | (ids >= table.shape[0])
        assert bool(bad.any()) and not bool(h[bad].any()), "an out-of-range id is not a zero row"
    if normed is None:
        assert not rmsnorm_replay_supported(table.shape[1], table.dtype, table.device)
    else:
        assert torch.equal(normed, ref_normed)
