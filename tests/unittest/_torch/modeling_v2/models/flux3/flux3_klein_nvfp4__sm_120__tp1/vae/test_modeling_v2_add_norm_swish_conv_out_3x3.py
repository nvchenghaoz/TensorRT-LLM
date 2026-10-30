# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 add_norm_swish_conv_out_3x3 entry.

The decoder tail at 2048^2 and 1024^2 output (128 channels, channels_last, conv_out
weight channels_last as the decoder holds it), gated against the same two bf16-rounded
sums and fp32 math to the end (TF32 off). Band, in fp32 units (2^-23): the split-TF32
products drop the lo * lo term (~2^-21 relative each); over the 1152-term sum with mixed
signs that is ~2^-21 of the output's scale, plus a few ulps from the fp32 sum order and
Triton's sigmoid vs torch's -> 64 units per element against the row maximum (16x
margin), 8 in relative RMS. The two bf16 sums are bitwise the reference's by
construction, so no bf16-sized term enters the band.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae.add_norm_swish_conv_out_3x3 import (  # noqa: E501
    add_norm_swish_conv_out_3x3 as op,
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
    """(x, r, r_bias, scale, shift, weight, bias) with the decoder's dtypes and layouts;
    scale / shift a GroupNorm affine (|scale| ~ 1 / std, shift ~ -mean * scale)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    c, h = spec["c"], spec["h"]

    def act(mult):
        t = (torch.randn(1, c, h, h, generator=g, device="cuda") * mult).to(torch.bfloat16)
        return t.contiguous(memory_format=CL)

    x, r = act(3.0), act(1.0)
    r_bias = (torch.randn(c, generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    scale = (torch.rand(1, c, generator=g, device="cuda") * 0.5 + 0.1).contiguous()
    shift = (torch.randn(1, c, generator=g, device="cuda") * 0.5).contiguous()
    weight = (torch.randn(3, c, 3, 3, generator=g, device="cuda") * 0.03).contiguous(
        memory_format=CL
    )
    bias = torch.randn(3, generator=g, device="cuda") * 0.1
    return x, r, r_bias, scale, shift, weight, bias


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, _seed(cell.why))
    out = op(*args)
    ref = op.reference(*args)
    assert out.dtype == torch.float32 and out.is_contiguous(memory_format=CL)
    assert_within_ulp(out, ref, element_ulp=64.0, rms_ulp=8.0)


def test_guard_refuses_mismatched_residual_layout() -> None:
    """r in NCHW while x is channels_last: the kernel reads r at x's offsets."""
    x, r, r_bias, scale, shift, weight, bias = _build(dict(c=128, h=256), _seed("layout"))
    with pytest.raises(AssertionError, match="strides"):
        with validating(op):
            op(x, r.contiguous(), r_bias, scale, shift, weight, bias)
