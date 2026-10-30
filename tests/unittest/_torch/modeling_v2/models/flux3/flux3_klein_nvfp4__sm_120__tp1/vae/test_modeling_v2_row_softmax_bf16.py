# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 row_softmax_bf16 entry.

The encoder mid block at 2048^2 (T = 65536) and 1024^2 (T = 16384) reference images,
gated against torch.softmax of the fp32 scaled scores. The op evaluates the same
expressions; the fp32 row sum's order and the scaling order ((s - max s) * scale vs
s * scale - max) move a value by ~1e-7 relative, so a bf16 rounding flips on rare
elements: one bf16 step at most (2 units of 2^-8 against the row maximum), 0.25 in
relative RMS. Then the persistent grid's rows past the SM count (rows are walked last
to first) are checked on a small T.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae.row_softmax_bf16 import (  # noqa: E501
    row_softmax_bf16 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

#: The encoder's mid block: 512 channels, softmax scale 512^-0.5.
SCALE = 512**-0.5


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _scores(t: int, seed: int) -> torch.Tensor:
    """[1, T, T] fp32 scores of bf16 q^T k at d = 512: ~N(0, 512) with per-row offsets."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(1, t, t, generator=g, device="cuda").mul_(512**0.5)
    s += torch.randn(1, t, 1, generator=g, device="cuda").mul_(100.0)
    return s


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    s = _scores(cell.spec["t"], _seed(cell.why))
    out = op(s, SCALE)
    ref = op.reference(s, SCALE)
    assert out.dtype == torch.bfloat16 and out.shape == s.shape
    del s
    # In row blocks: the gate's fp32 temporaries of a whole 65536^2 output would not fit.
    for r0 in range(0, out.shape[1], 4096):
        assert_within_ulp(
            out[0, r0 : r0 + 4096], ref[0, r0 : r0 + 4096], element_ulp=2.0, rms_ulp=0.25
        )


def test_rows_past_the_grid() -> None:
    """More rows than programs: each program walks several rows."""
    s = _scores(8192, _seed("rows"))[:, :2000]
    s = s.contiguous()
    assert_within_ulp(op(s, SCALE), op.reference(s, SCALE), element_ulp=2.0, rms_ulp=0.25)


def test_guard_refuses_ragged_rows() -> None:
    """A row length that is not a multiple of the 8192-column block."""
    s = _scores(8192, _seed("ragged"))[:, :, :4096].contiguous()
    with pytest.raises(AssertionError, match="multiple of 8192"):
        with validating(op):
            op(s, SCALE)
