# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the adaln_nvfp4 entry.

Cells (both kernels, every served table form) against the reference (torch layer_norm, the
block's bf16 rounding points, exact-division NVFP4) in a stated band, SF padding rows
included. Then the two kernels against each other at a row count both take, and the warp
kernel's descending row walk against an independent row subset.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.norm._adaln_nvfp4_triton import (  # noqa: E501
    adaln_nvfp4_triton,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.norm.adaln_nvfp4 import (  # noqa: E501
    adaln_nvfp4 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "../gemm"]
from _flux3_operands import assert_nvfp4_band  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

DEV = "cuda"
D = 3072
EPS = 1e-6
#: The reference's LayerNorm statistics are summed in another order (one fp32 ulp of
#: mean / rstd), FlashInfer's quantize uses rcp.approx where the reference divides
#: exactly: a bf16 modulated value, and with it a block scale or a code, moves only at a
#: rounding boundary.
MAX_SF_FRAC = 2e-3
MAX_CODE_FRAC = 2e-3
#: The two kernels: 1 bf16 ulp on ~1e-6 of the modulated values.
CROSS_FRAC = 1e-4


def _operands(m: int, form: str, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(1, m, D, generator=g, device=DEV) * 3 + 0.5).to(torch.bfloat16)
    rows = {"row": 1, "bool": 2, "index": 3}[form]
    # A wider table than the select uses, sliced: the model's tables are views with a row
    # stride and a storage offset.
    table = (torch.randn(1, 2 * rows, D, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    shift, scale = table[:, 0::2], table[:, 1::2]
    if form == "row":
        select = None
    elif form == "bool":
        select = torch.rand(1, m, 1, generator=g, device=DEV) < 0.5
    else:
        select = torch.randint(0, 3, (1, m, 1), generator=g, device=DEV).to(torch.uint8)
    global_scale = torch.tensor([448.0 * 6.0 / 8.0], device=DEV)
    return x, select, scale, shift, global_scale, EPS


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    args = _operands(s["m"], s["form"], zlib.crc32(cell.why.encode()))
    out = op(*args)
    assert_nvfp4_band(out, op.reference(*args), s["m"], D, MAX_SF_FRAC, MAX_CODE_FRAC)


@pytest.mark.parametrize("form", ["row", "bool"])
def test_warp_and_triton_kernels_agree(form) -> None:
    x, select, scale, shift, gs, eps = _operands(4096, form, seed=17)
    warp = torch.ops.trtllm.flux3_adaln_nvfp4_warp(
        x, None if select is None else select.reshape(-1), scale, shift, gs, eps
    )
    triton_out = adaln_nvfp4_triton(x, select, scale, shift, gs, eps)
    assert_nvfp4_band(warp, triton_out, 4096, D, CROSS_FRAC, CROSS_FRAC)


def test_rows_are_independent() -> None:
    """The first 4224 rows of a 16464-row call equal the 4224-row call on those rows."""
    x, select, scale, shift, gs, eps = _operands(16464, "bool", seed=5)
    full = op(x, select, scale, shift, gs, eps)
    n = 4224
    part = op(x[:, :n].contiguous(), select[:, :n].contiguous(), scale, shift, gs, eps)
    assert torch.equal(part[0], full[0][:n])
    assert torch.equal(part[1], full[1][: n * (D // 16)])


def test_guard_refuses_an_index_past_the_table() -> None:
    x, select, scale, shift, gs, eps = _operands(256, "index", seed=3)
    select = torch.full_like(select, 3)
    with validating(op), pytest.raises(AssertionError, match="past the table"):
        op(x, select, scale, shift, gs, eps)
