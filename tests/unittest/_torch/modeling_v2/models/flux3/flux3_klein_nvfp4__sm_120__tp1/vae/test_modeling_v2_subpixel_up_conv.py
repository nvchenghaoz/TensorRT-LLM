# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 subpixel_up_conv entry.

Each decoder upsample at 2048^2 and 1024^2 output, gated against the fp32
conv_transpose2d of the same bf16 operands. Band: bf16 products are exact in fp32, so
only the order of the fp32 sum (4 * C_in taps per output) differs before one bf16
rounding: one bf16 step at most per element (2 units of 2^-8 against the row maximum),
0.25 in relative RMS. Then: the transposed weight equals nearest-2x + 3x3 conv (the
identity the decoder relies on), and each graph runs its pinned plan.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae import (  # noqa: E501
    _subpixel_up_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae.subpixel_up_conv import (  # noqa: E501
    subpixel_up_conv as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

CL = torch.channels_last


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """(x, wt, bias): a bf16 channels_last activation and the bf16 channels_last transposed
    weight derived (as the decoder's load does) from an fp32 3x3 conv weight."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    c, h = spec["c"], spec["h"]
    x = torch.randn(1, c, h, h, generator=g, device="cuda").to(torch.bfloat16)
    x = x.contiguous(memory_format=CL)
    bias = (torch.randn(c, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    # Derived on the CPU from fp32, like the load.
    w = torch.randn(c, c, 3, 3, generator=torch.Generator().manual_seed(seed)) * 0.02
    wt = _subpixel_up_conv.subpixel_transpose_weight(w).cuda().to(torch.bfloat16)
    return x, wt.contiguous(memory_format=CL), bias


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, wt, bias = _build(cell.spec, _seed(cell.why))
    out = op(x, wt, bias)
    ref = op.reference(x, wt, bias)
    assert out.dtype == torch.bfloat16 and out.is_contiguous(memory_format=CL)
    assert out.shape == (1, cell.spec["c"], 2 * cell.spec["h"], 2 * cell.spec["h"])
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)

    c, h = cell.spec["c"], cell.spec["h"]
    key = (1, c, c, h, h)
    assert _subpixel_up_conv._graphs[key].plan_name == _subpixel_up_conv.PLAN_NAME[(c, c, h)]


def test_transpose_weight_is_nearest_upsample_conv() -> None:
    """conv_transpose2d(x, Wt, stride 2, pad 1) == conv3x3(nearest2x(x)) in fp64."""
    g = torch.Generator().manual_seed(_seed("identity"))
    w = torch.randn(16, 8, 3, 3, generator=g, dtype=torch.float64)
    x = torch.randn(1, 8, 5, 7, generator=g, dtype=torch.float64)
    want = torch.nn.functional.conv2d(
        torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"), w, padding=1
    )
    wt = _subpixel_up_conv.subpixel_transpose_weight(w)
    got = torch.nn.functional.conv_transpose2d(x, wt, stride=2, padding=1)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_guard_refuses_3x3_weight() -> None:
    """The 3x3 conv weight passed where the 4x4 transposed weight belongs."""
    x, wt, bias = _build(op.CELLS[0].spec, _seed("3x3"))
    w3 = torch.zeros(wt.shape[0], wt.shape[1], 3, 3, dtype=wt.dtype, device=wt.device)
    with pytest.raises(AssertionError, match="transposed sub-pixel weight"):
        with validating(op):
            op(x, w3, bias)
