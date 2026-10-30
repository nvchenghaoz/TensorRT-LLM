# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gate_up_swiglu_nvfp4 entry.

Cells against the reference (fp32 GEMM, the same h formula with torch's exp2, exact-
division NVFP4) in a stated band, SF padding rows included. Exact on their own: the load-
time preparation is a pure byte permutation of the checkpoint weight (undoing it gives the
checkpoint bytes back), the caller-provided buffers give the same bytes as the op's own,
and a prefix of the activation's rows gives the prefix of the output (the last single
block runs gate_up on its first 16384 rows).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _prepare as prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.gate_up_swiglu_nvfp4 import (  # noqa: E501
    allocate_outputs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.gate_up_swiglu_nvfp4 import (  # noqa: E501
    gate_up_swiglu_nvfp4 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_reference as nv,
)
from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_operands import (  # noqa: E402
    DEV,
    HIDDEN,
    MLP,
    assert_nvfp4_band,
    nvfp4_act,
    nvfp4_weight,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

N = 2 * MLP

#: Block scales allowed to differ: the kernel's fp32 GEMM sum and the reference's differ
#: in their last bits, and ex2.approx vs torch's exp2 by ~2 ulp; a block scale moves only
#: when its max h sits within that of an E4M3 rounding boundary (~1e-4 of the blocks).
MAX_SF_FRAC = 2e-3
#: Codes allowed to differ (one step) in blocks whose scale matches, for the same reasons.
MAX_CODE_FRAC = 2e-3

_WEIGHTS: dict = {}


def _weights():
    if not _WEIGHTS:
        fp4, sf, ws2, ins = nvfp4_weight(N, HIDDEN, seed=21)
        lin = prep.Nvfp4Linear(fp4, sf, ins, ws2)
        weight, scale = prep.prepare_gate_up(lin.weight, lin.weight_scale)
        in_gs, w_gs = prep.global_scales(lin)
        _WEIGHTS.update(
            raw=lin, weight=weight, scale=scale, alpha=prep.alpha(in_gs, w_gs, DEV),
            # down_proj's activation global scale; the checkpoint's are O(10-100).
            out_scale=prep.inverse_scale(1.0 / 0.0217, DEV),
        )  # fmt: skip
    return _WEIGHTS


def _operands(m: int, seed: int):
    wt = _weights()
    act, act_sf = nvfp4_act(m, HIDDEN, seed, wt["raw"].input_scale, std=2.0)
    return act, wt["weight"], act_sf, wt["scale"], wt["alpha"], wt["out_scale"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    m = cell.spec["m"]
    args = _operands(m, seed_of(cell.why))
    out = op(*args)
    assert out[0].shape == (m, MLP // 2) and out[0].dtype == torch.uint8
    ref = op.reference(*args)
    assert_nvfp4_band(out, ref, m, MLP, MAX_SF_FRAC, MAX_CODE_FRAC)


def test_preparation_is_a_byte_permutation() -> None:
    """Undo v2w and the [value, gate] pairing: the checkpoint's bytes and scales back."""
    wt = _weights()
    perm = prep.v2w_row_permutation(N, DEV)
    paired = torch.empty_like(wt["weight"])
    paired[perm] = wt["weight"]
    gate, up = paired[1::2], paired[0::2]
    assert torch.equal(torch.cat([gate, up]), wt["raw"].weight)
    kb = HIDDEN // 16
    lin = nv.unswizzle_sf(wt["scale"], N, kb)
    paired_sf = torch.empty_like(lin)
    paired_sf[perm] = lin
    raw_sf = unswizzle_sf(wt["raw"].weight_scale.view(torch.uint8), N, HIDDEN)
    assert torch.equal(torch.cat([paired_sf[1::2], paired_sf[0::2]]), raw_sf)


def test_caller_buffers_and_row_prefix() -> None:
    """Pre-allocated buffers give the op's own bytes; the first 16384 rows of a 16464-row
    activation give the first 16384 rows of the full output and its leading scale rows."""
    args = _operands(16464, seed=3)
    full = op(*args)
    out, out_sf = allocate_outputs(16464, N, torch.device(DEV))
    given = op(*args, out, out_sf)
    assert torch.equal(given[0], full[0]) and torch.equal(given[1], full[1])
    n = 16384
    cols = MLP // 16
    prefix = op(args[0][:n], args[1], args[2][: n * (HIDDEN // 16)], *args[3:])
    assert torch.equal(prefix[0], full[0][:n])
    assert torch.equal(prefix[1], full[1][: n * cols])


def test_guard_refuses_a_per_token_alpha() -> None:
    args = list(_operands(256, seed=2))
    args[4] = args[4].repeat(256)
    with validating(op), pytest.raises(AssertionError, match="exactly one element"):
        op(*args)
