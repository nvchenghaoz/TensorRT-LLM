# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 fp8_attention_nvfp4_out_q_rows entry.

The exactness claim on its own: fp4 rows [0, q_rows) and their scale bytes are bitwise the full
launch's (each Q tile is computed independently over every K/V row).

Cells (the last single block's image-row prefix at 2048^2 t2i / i2i and 1024^2) against the
reference on the prefix rows only, in fp8_attention_nvfp4_out's band (same derivation).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.fp8_attention_bf16_out import (  # noqa: E501
    fp8_attention_bf16_out,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.fp8_attention_nvfp4_out import (  # noqa: E501
    fp8_attention_nvfp4_out,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.fp8_attention_nvfp4_out_q_rows import (  # noqa: E501
    fp8_attention_nvfp4_out_q_rows as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_fp8_attention_checks import (  # noqa: E402
    assert_nvfp4_close,
    global_scale_for,
    make_qkv8,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 120,
    reason="the flux3 klein sm_120 entries are certified on sm_120",
)

MAX_SF_FRAC = 2e-2
MAX_REL_RMS = 2.0**-3
#: Scale bytes per 128-row block of the H = 24 output (H * 128 / 16 columns).
_SF_COLS = 24 * 8


def _prefix(out, q_rows: int):
    """(fp4 rows [0, q_rows), the scale bytes of those rows) of a B = 1 result; q_rows is a
    multiple of 128, so its rows' scales are the buffer's first q_rows / 128 blocks."""
    return out[0][:, :q_rows], out[1].view(torch.uint8)[: q_rows * _SF_COLS]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    b, n, h, q_rows = s["batch"], s["seq_len"], s["heads"], s["q_rows"]
    qkv8, scales = make_qkv8(b, n, h, seed_of(cell.why))
    gs = global_scale_for(fp8_attention_bf16_out(qkv8, scales, b, h))
    out = op(qkv8, scales, b, h, gs, q_rows)
    assert out[0].shape == (b, n, h * 64)
    ref = op.reference(qkv8, scales, b, h, gs, q_rows)
    assert_nvfp4_close(
        _prefix(out, q_rows), _prefix(ref, q_rows), q_rows, h * 128, gs, MAX_SF_FRAC, MAX_REL_RMS
    )


@pytest.mark.parametrize("seq_len,q_rows", [(4176, 4096), (32848, 16384)], ids=["s4176", "s32848"])
def test_prefix_bitwise_the_full_launch(seq_len, q_rows) -> None:
    qkv8, scales = make_qkv8(1, seq_len, 24, seed=seq_len + q_rows)
    gs = global_scale_for(fp8_attention_bf16_out(qkv8, scales, 1, 24))
    full = fp8_attention_nvfp4_out(qkv8, scales, 1, 24, gs)
    part = op(qkv8, scales, 1, 24, gs, q_rows)
    for got, want in zip(_prefix(part, q_rows), _prefix(full, q_rows)):
        assert torch.equal(got, want)


def test_zero_and_full_q_rows_run_every_row() -> None:
    qkv8, scales = make_qkv8(1, 4176, 24, seed=7)
    gs = global_scale_for(fp8_attention_bf16_out(qkv8, scales, 1, 24))
    full = fp8_attention_nvfp4_out(qkv8, scales, 1, 24, gs)
    for q_rows in (0, 4176):
        out = op(qkv8, scales, 1, 24, gs, q_rows)
        assert torch.equal(out[0], full[0]) and torch.equal(out[1], full[1])
