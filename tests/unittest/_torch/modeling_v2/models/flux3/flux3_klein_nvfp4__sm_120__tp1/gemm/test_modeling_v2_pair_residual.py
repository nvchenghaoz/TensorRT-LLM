# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the pair_residual entry.

Cells against the reference (fp32 GEMMs rounded to BF16, then the residual with a BF16
rounding after every op) in a stated band, for every served gate form. Exact on their own:
both tile instances at every raster give the same bytes, repeated launches on one
workspace (the launch epoch) give the same bytes, and gate = 0 returns x.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _prepare as prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.pair_residual import (  # noqa: E501
    allocate_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.pair_residual import (  # noqa: E501
    nvfp4_pair_residual as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_operands import (  # noqa: E402
    DEV,
    HIDDEN,
    MLP,
    assert_bf16_band,
    nvfp4_act,
    nvfp4_weight,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

N = HIDDEN

#: The two BF16 GEMM outputs each differ from the fp32 reference's in ~5e-4 of the elements
#: by one ulp (summation order across a rounding midpoint); the residual's three BF16
#: roundings pass such a difference on as at most one ulp of the output, and x dominates the
#: output's magnitude, so most of them vanish.
MAX_FRAC = 2e-3

_STATE: dict = {}


def _state():
    if not _STATE:
        down = prep.Nvfp4Linear(*_reorder(nvfp4_weight(N, MLP, seed=31)))
        out = prep.Nvfp4Linear(*_reorder(nvfp4_weight(N, HIDDEN, seed=32)))
        d_in, d_w = prep.global_scales(down)
        o_in, o_w = prep.global_scales(out)
        _STATE.update(
            down=down, out=out, down_alpha=prep.alpha(d_in, d_w, DEV),
            out_alpha=prep.alpha(o_in, o_w, DEV), workspace=allocate_workspace(torch.device(DEV)),
        )  # fmt: skip
    return _STATE


def _reorder(w):
    fp4, sf, ws2, ins = w
    return fp4, sf, ins, ws2


def _operands(m: int, gate_form: str, seed: int):
    st = _state()
    g = torch.Generator(device=DEV).manual_seed(seed)
    hidden, hidden_sf = nvfp4_act(m, MLP, seed + 1, st["down"].input_scale)
    attn, attn_sf = nvfp4_act(m, HIDDEN, seed + 2, st["out"].input_scale)
    x = (torch.randn(1, m, N, generator=g, device=DEV) * 4).to(torch.bfloat16)
    rows = {"row": 1, "bool": 2, "index": 3}[gate_form]
    gate = (torch.randn(1, rows, N, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    if gate_form == "row":
        index = None
    elif gate_form == "bool":
        index = torch.rand(1, m, 1, generator=g, device=DEV) < 0.5
    else:
        index = torch.randint(0, 3, (1, m, 1), generator=g, device=DEV).to(torch.uint8)
    return (
        hidden, st["down"].weight, hidden_sf, st["down"].weight_scale, st["down_alpha"],
        attn, st["out"].weight, attn_sf, st["out"].weight_scale, st["out_alpha"],
        x, gate, index, st["workspace"],
    )  # fmt: skip


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    args = _operands(s["m"], s["gate"], seed_of(cell.why))
    out = op(*args)
    assert out.shape == args[10].shape and out.dtype == torch.bfloat16
    assert_bf16_band(out, op.reference(*args), MAX_FRAC, max_ulps=1)


def _raw(args, t256: bool, along_n: bool) -> torch.Tensor:
    (h, dw, hs, dws, da, a, ow, as_, ows, oa, x, gate, index, ws) = args
    rows = a.shape[0]
    idx = None if index is None else index.reshape(-1).view(torch.uint8)
    return torch.ops.trtllm.flux3_nvfp4_pair_residual(
        h, dw, hs, dws, da, a, ow, as_, ows, oa, x.reshape(rows, N), gate.reshape(-1, N), idx, ws,
        t256, along_n,
    )  # fmt: skip


@pytest.mark.parametrize(
    "m,gate_form",
    [(4176, "bool"), (16464, "bool"), (32848, "index")],
    ids=["m4176", "m16464", "m32848"],
)
def test_instances_and_rasters_give_the_same_bytes(m, gate_form) -> None:
    args = _operands(m, gate_form, seed=m)
    outs = [_raw(args, t256, along_n) for t256 in (False, True) for along_n in (False, True)]
    for o in outs[1:]:
        assert torch.equal(o.view(torch.int16), outs[0].view(torch.int16))


def test_repeated_launches_and_zero_gate() -> None:
    """The epoch advances per launch on one workspace; gate = 0 returns x exactly."""
    args = list(_operands(16384, "row", seed=9))
    first = op(*args)
    for _ in range(3):
        assert torch.equal(op(*args).view(torch.int16), first.view(torch.int16))
    args[11] = torch.zeros_like(args[11])
    assert torch.equal(op(*args), args[10])


def test_guard_refuses_an_index_past_the_table() -> None:
    args = list(_operands(256, "index", seed=4))
    args[12] = torch.full_like(args[12], 3)
    with validating(op), pytest.raises(AssertionError, match="past the gate table"):
        op(*args)
