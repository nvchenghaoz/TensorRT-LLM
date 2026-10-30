# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8r_adaln_quant entry: cells against the reference (torch layer_norm,
the block's bf16 rounding points, the per-token E4M3 recipe) in a stated band (a fraction
of codes, each within what one bf16 ulp of the normalized value allows)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.norm.fp8r_adaln_quant import (  # noqa: E501
    fp8r_adaln_quant as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

DEV = "cuda"
D = 3072
#: The LayerNorm statistics are summed in another order than torch's: a bf16 normalized
#: value moves by an ulp only at a rounding boundary, and a code with it.
MAX_FRAC = 2e-3


def _bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(v), torch.frexp(v.abs().clamp_min(2.0**-120))[1] - 8)


def _e4m3_step(q: torch.Tensor) -> torch.Tensor:
    """E4M3 spacing at |q| (normals from 2^-6: 3 mantissa bits; subnormals 2^-9)."""
    exponent = torch.frexp(q.abs().clamp_min(2.0**-6))[1] - 1
    return torch.ldexp(torch.ones_like(q), exponent - 3)


def _assert_codes_within_one_norm_ulp(x, shift, scale, scales, data, ref_data) -> None:
    """Each differing code (rows whose scale matches) within what one bf16 ulp of the
    normalized value allows, in value space. Derivation: norm moves by one ulp(norm);
    bf16((1 + scale) * norm) then moves by |1 + scale| * ulp(norm) plus the two sides'
    roundings (half an ulp(prod) each), bf16(prod + shift) by one more ulp(x_mod), and the
    E4M3 rounding of each side by half a step at its value. Where prod + shift cancels,
    ulp(prod) is many E4M3 steps of the small x_mod, so a code count would not bound it."""
    rows = data.shape[0]
    norm = torch.nn.functional.layer_norm(x.reshape(rows, D).float(), (D,), eps=1e-6)
    norm = norm.to(torch.bfloat16).float()
    one_plus = (scale.reshape(1, D).float() + 1.0).to(torch.bfloat16).float()
    prod = (one_plus * norm).to(torch.bfloat16).float()
    x_mod = (prod + shift.reshape(1, D).float()).to(torch.bfloat16).float()
    q, ref_q = data.float(), ref_data.float()
    row_scale = scales.reshape(-1, 1)
    allowed = (
        one_plus.abs() * _bf16_ulp(norm)
        + _bf16_ulp(prod)
        + _bf16_ulp(x_mod)
        + 0.5 * (_e4m3_step(q) + _e4m3_step(ref_q)) * row_scale
    )
    diff = data.view(torch.uint8) != ref_data.view(torch.uint8)
    err = (q - ref_q).abs() * row_scale
    assert bool((err <= allowed)[diff].all()), "a code moved more than one norm ulp allows"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    m = cell.spec["m"]
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()))
    x = (torch.randn(1, m, D, generator=g, device=DEV) * 3 + 0.5).to(torch.bfloat16)
    table = (torch.randn(1, 2, D, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    shift, scale = table[:, 0:1], table[:, 1:2]
    data, scales = op(x, shift, scale)
    ref_data, ref_scales = op.reference(x, shift, scale)
    assert data.shape == (m, D) and scales.shape == (m,)
    moved = scales != ref_scales
    assert moved.float().mean().item() <= MAX_FRAC
    codes_moved = data[~moved].view(torch.uint8) != ref_data[~moved].view(torch.uint8)
    assert codes_moved.float().mean().item() <= MAX_FRAC
    _assert_codes_within_one_norm_ulp(
        x, shift, scale, scales[~moved], data[~moved], ref_data[~moved]
    )
