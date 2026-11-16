# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rmsnorm_residual_add_scale entry (the layer tail).

Per cell, against the entry's reference (FlashInfer rmsnorm, the bf16 add, the fp32 multiply by
the layer scalar, the bf16 cast, the next FlashInfer rmsnorm). Both norms sum their squares
with Triton's `tl.sum` and divide with Triton's fp32 divide, not in the CuTe kernel's order, so
an output moves by one ulp where its fp32 value sits by a bf16 rounding midpoint (the kernel
module measured ~5e-6 of the elements). The primary output carries the normed term's flip
through two more roundings (add, scale): 4 units of the row maximum's 2^-8 element-wise; the
next norm reads that output, so one flip there plus its own: 4 units too. At most 1e-4 of the
elements differ in either.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.norm.rmsnorm_residual_add_scale import (  # noqa: E501
    rmsnorm_residual_add_scale as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

EPS = 1e-6
MAX_FLIP_FRAC = 1e-4


def _build(spec, seed: int):
    """`(x, residual, w, layer_scalar, eps, next_w | None, eps)`: a down_proj output, a residual
    stream four times larger, norm weights around 1 and a layer scalar below 1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, h = spec["m"], spec["h"]
    x = (torch.randn(m, h, generator=g, device="cuda") * 2).to(torch.bfloat16)
    residual = (torch.randn(m, h, generator=g, device="cuda") * 8).to(torch.bfloat16)
    w = (1 + 0.3 * torch.randn(h, generator=g, device="cuda")).to(torch.bfloat16)
    scale = (0.25 + torch.rand(1, generator=g, device="cuda")).float()
    next_w = (1 + 0.3 * torch.randn(h, generator=g, device="cuda")).to(torch.bfloat16)
    return x, residual, w, scale, EPS, next_w if spec["next_norm"] else None, EPS


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    flips = (out.view(torch.int16) != ref.view(torch.int16)).float().mean().item()
    assert flips <= MAX_FLIP_FRAC, f"{flips:.2e} of the elements differ"
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.25)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed_of(cell.why))
    out, normed = op(*args)
    ref_out, ref_normed = op.reference(*args)
    _compare(out, ref_out)
    if cell.spec["next_norm"]:
        _compare(normed, ref_normed)
    else:
        assert normed is None and ref_normed is None
