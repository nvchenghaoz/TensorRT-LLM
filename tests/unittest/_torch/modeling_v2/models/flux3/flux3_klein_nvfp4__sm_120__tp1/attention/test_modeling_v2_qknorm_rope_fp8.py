# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 qknorm_rope_fp8 entry.

Cells (image.0's rows at 2048^2 and 1024^2, q/k/v the column slices of the [M, 9216] GEMM output
as served) against the FP32 reference. Band: the V plane, its scale and factor are bitwise (the
same amax, div.rn, mul.rn and satfinite cast). For Q and K the kernel's lane-ordered sum of
squares can move inv_rms by a few FP32 ulps; a rotated value then lands on the other BF16
neighbour only within that distance of a BF16 midpoint (~2^-14 of the values) and its E4M3
byte moves (by one step) only if that also crosses an E4M3 midpoint. Gate: at most 1e-3 of the
Q/K bytes differ, each by one E4M3 step; the Q/K scales (BF16-rounded amax / 448) within one
BF16 step.

The reference runs first: the op drops the consumed source lines from L2, so the GEMM output is
undefined after the call.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.qknorm_rope_fp8 import (  # noqa: E501
    qknorm_rope_fp8 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_fp8_attention_checks import seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 120,
    reason="the flux3 klein sm_120 entries are certified on sm_120",
)

DEV = "cuda"
SM_SCALE = 128**-0.5
MAX_QK_BYTE_FRAC = 1e-3


def _operands(rows: int, heads: int, seed: int):
    """(q, k, v column views of a fresh [M, 3W] BF16 GEMM output, q_norm, k_norm, cos_sin,
    positions, sm_scale)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = heads * 128
    gemm = (torch.randn(rows, 3 * width, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    q, k, v = gemm[:, :width], gemm[:, width : 2 * width], gemm[:, 2 * width :]
    q_norm = (1.0 + 0.2 * torch.randn(128, generator=g, device=DEV)).to(torch.bfloat16)
    k_norm = (1.0 + 0.2 * torch.randn(128, generator=g, device=DEV)).to(torch.bfloat16)
    theta = torch.rand(rows, 64, generator=g, device=DEV) * 100.0
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=-1).contiguous()
    positions = torch.arange(rows, device=DEV, dtype=torch.int64)
    return (q, k, v, q_norm, k_norm, cos_sin, positions, SM_SCALE)


def _one_e4m3_step(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Whether two E4M3 values are equal or adjacent codes (same sign)."""
    ia = a.view(torch.uint8).long()
    ib = b.view(torch.uint8).long()
    return ((ia >> 7) == (ib >> 7)) & ((ia & 0x7F) - (ib & 0x7F)).abs().le(1)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    args = _operands(s["rows"], s["heads"], seed_of(cell.why))
    ref_qkv8, ref_scales = op.reference(*args)
    qkv8, scales = op(*args)
    assert qkv8.shape == (3, s["rows"], s["heads"] * 128) and qkv8.dtype == torch.float8_e4m3fn
    # V: bitwise.
    assert torch.equal(qkv8[2].view(torch.uint8), ref_qkv8[2].view(torch.uint8))
    assert torch.equal(scales[[2, 6, 7]], ref_scales[[2, 6, 7]])
    # Q, K: one E4M3 step at most, rarely.
    for p in range(2):
        diff = qkv8[p].view(torch.uint8) != ref_qkv8[p].view(torch.uint8)
        assert diff.float().mean().item() <= MAX_QK_BYTE_FRAC
        assert bool(_one_e4m3_step(qkv8[p], ref_qkv8[p]).all())
    torch.testing.assert_close(scales[[0, 1, 3]], ref_scales[[0, 1, 3]], rtol=2.0**-7, atol=0.0)
    torch.testing.assert_close(scales[[4, 5]], ref_scales[[4, 5]], rtol=2.0**-7, atol=0.0)


def test_scales_describe_the_planes() -> None:
    """scales[p] * scales[4 + p] == 1 to rounding and scales[3] = q_scale * k_scale * sm_scale:
    the contract the attention entries read."""
    args = _operands(4096, 24, seed=3)
    _, scales = op(*args)
    ones = torch.ones(3, device=DEV)
    torch.testing.assert_close(scales[:3] * scales[4:7], ones, rtol=2e-7, atol=0.0)
    want = (scales[0] * scales[1]) * torch.tensor(SM_SCALE, dtype=torch.float32)
    assert torch.equal(scales[3], want)


def test_guard_refuses_int32_positions() -> None:
    args = list(_operands(4096, 24, seed=4))
    args[6] = args[6].to(torch.int32)
    with validating(op), pytest.raises(AssertionError, match="int64"):
        op(*args)
