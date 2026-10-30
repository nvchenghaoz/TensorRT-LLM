# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 split_attn_e4m3 entry.

The decoder mid block at 2048^2 (T = 65536) and 1024^2 (T = 16384) output, gated against
the same steps in fp32 torch. The steps match one for one, so the differences are
accumulation order: S (512 bf16 products, exact in fp32) moves by ~1e-6 relative, which
flips an e4m3 rounding of P in roughly one element in 10^5 (one e4m3 step, 1/8 of that
element, averaged over T terms); l and P.V in fp32. The bf16 output can then land one
bf16 step from the reference (2 units of 2^-8 against its row maximum); 4 units per
element and 0.5 in relative RMS leave room for the e4m3 MMA's accumulator.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae.split_attn_e4m3 import (  # noqa: E501
    split_attn_e4m3 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """Decoder-like operands: q / k / v as channels_last [C, T] views (strides (1, C)) of
    projected GroupNorm outputs, V quantized per channel as the decoder does."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    c, t = spec["c"], spec["t"]
    q = torch.randn(t, c, generator=g, device="cuda").to(torch.bfloat16).t()
    k = torch.randn(t, c, generator=g, device="cuda").to(torch.bfloat16).t()
    v = (torch.randn(t, c, generator=g, device="cuda") * 2.0).to(torch.bfloat16).t()
    vf = v.float()
    s_v = (vf.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)
    vq = (vf / s_v).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return q, k, vq, s_v.reshape(c), c**-0.5


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, _seed(cell.why))
    out = op(*args)
    torch.cuda.synchronize()
    ref = op.reference(*args)
    assert out.dtype == torch.bfloat16 and out.shape == (cell.spec["t"], cell.spec["c"])
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.5)


def test_guard_refuses_unsupported_token_count() -> None:
    """T = 9 * 4096: NT = 144 tile maxes, which kernel B's unmasked chunks over-read."""
    q, k, vq, sv, scale = _build(dict(c=512, t=36864), _seed("nt144"))
    with pytest.raises(AssertionError, match="power of two"):
        with validating(op):
            op(q, k, vq, sv, scale)
