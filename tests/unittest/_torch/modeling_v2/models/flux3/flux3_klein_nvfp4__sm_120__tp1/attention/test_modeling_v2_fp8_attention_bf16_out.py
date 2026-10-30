# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 fp8_attention_bf16_out entry.

Cells (image.0's S at 2048^2 and 1024^2) against the reference, the kernel's per-64-key
recurrence emulated in FP32 torch with the same E4M3 P. Band: the two differ by ex2.approx vs
torch's exp2, the fused multiply-adds and the FP8 MMA's summation of QK^T -- relative errors of
about 2^-13 in the softmax argument. A P byte next to an E4M3 rounding midpoint can then round
to its other neighbour (one E4M3 step, 2^-3 of that p); at S >= 4096 with diffuse attention a
few hundred such flips per row move O by about 1% of its typical magnitude, a few bf16 units of
the row maximum. Gate: 16 units of 2^-8 per element against its row's maximum, 4 in relative
RMS. A wrong V layout, scale or mask is O(1).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.fp8_attention_bf16_out import (  # noqa: E501
    fp8_attention_bf16_out as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_fp8_attention_checks import make_qkv8, seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 120,
    reason="the flux3 klein sm_120 entries are certified on sm_120",
)

ELEMENT_ULP = 16
RMS_ULP = 4


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    qkv8, scales = make_qkv8(s["batch"], s["seq_len"], s["heads"], seed_of(cell.why))
    out = op(qkv8, scales, s["batch"], s["heads"])
    assert out.shape == (s["batch"], s["seq_len"], s["heads"], 128)
    assert out.dtype == torch.bfloat16
    ref = op.reference(qkv8, scales, s["batch"], s["heads"])
    assert_within_ulp(
        out.reshape(-1, 128), ref.reshape(-1, 128), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP
    )


def test_deterministic() -> None:
    """Two launches give the same bytes (no split reduction, no atomics)."""
    qkv8, scales = make_qkv8(1, 4176, 24, seed=11)
    assert torch.equal(op(qkv8, scales, 1, 24), op(qkv8, scales, 1, 24))


def test_refuses_a_sequence_not_a_multiple_of_16() -> None:
    qkv8, scales = make_qkv8(1, 4104, 24, seed=12)
    with pytest.raises(RuntimeError, match="S % 16"):
        op(qkv8, scales, 1, 24)
