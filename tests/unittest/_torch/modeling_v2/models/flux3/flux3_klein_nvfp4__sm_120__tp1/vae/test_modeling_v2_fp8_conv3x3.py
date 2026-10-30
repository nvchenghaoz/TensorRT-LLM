# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 fp8_conv3x3 entry.

Every decoder conv key and graph variant at 2048^2 and 1024^2 output, gated against the
fp32 conv of the dequantized operands. Band: e4m3 x e4m3 products are exact in fp32, so
the op and the reference differ only by the order (and the tensor cores' accumulator) of
the 9 * Cin-term sum before one bf16 rounding; an element can land one bf16 step (2
units of 2^-8 against its row's maximum) away, and a sum over up to 4608 terms through
the FP8 MMA accumulator can add about as much again -> 4 units per element, 0.5 in
relative RMS (a few elements in a thousand move at all). Then: each graph runs its
pinned plan, and the plan selection matches engine tokens exactly and refuses a missing
pin (no GPU needed).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae import (  # noqa: E501
    _cudnn,
    _fp8_conv3x3,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.vae.fp8_conv3x3 import (  # noqa: E501
    fp8_conv3x3 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 120,
    reason="the flux3 klein sm_120 entries are certified on sm_120",
)

FP8, QMAX = torch.float8_e4m3fn, 448.0
CL = torch.channels_last


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """Decoder-like operands: swish(GN(x))-shaped activations (>= -0.28, a long positive
    tail) quantized per tensor, conv weights ~N(0, 0.02) quantized per Cout."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cin, cout, h = spec["cin"], spec["cout"], spec["h"]
    a = torch.randn(1, cin, h, h, generator=g, device="cuda").mul_(1.5)
    a = (a * torch.sigmoid(a)).contiguous(memory_format=CL)
    s_x = (a.abs().amax() / QMAX).reshape(1)
    xq = (a / s_x).clamp(-QMAX, QMAX).to(FP8)
    del a
    w = torch.randn(cout, cin, 3, 3, generator=g, device="cuda") * 0.02
    s_w = w.abs().amax(dim=(1, 2, 3)) / QMAX
    wq = (w / s_w.view(-1, 1, 1, 1)).clamp(-QMAX, QMAX).to(FP8).permute(0, 2, 3, 1).contiguous()
    bias = None
    if spec["has_bias"]:
        bias = torch.randn(cout, generator=g, device="cuda") * 0.1
        bias = bias if spec["has_bias"] == 2 else bias.to(torch.bfloat16)
    residual = None
    if spec["has_res"]:
        residual = torch.randn(1, cout, h, h, generator=g, device="cuda").to(torch.bfloat16)
        residual = residual.contiguous(memory_format=CL)
    return xq, s_x, wq, s_w, bias, residual


@sm120
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, _seed(cell.why))
    out = op(*args)
    ref = op.reference(*args)
    assert out.dtype == torch.bfloat16 and out.is_contiguous(memory_format=CL)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.5)

    spec = cell.spec
    key = (1, spec["cin"], spec["cout"], spec["h"], spec["h"], spec["has_bias"], spec["has_res"])
    engine, knobs = _cudnn.plan_tokens(_fp8_conv3x3._graphs[key].plan_name)
    want = _fp8_conv3x3.PLAN_K24[(spec["cin"], spec["cout"], spec["h"])]
    assert engine == "eng7" and int(knobs["k24"]) == want, _fp8_conv3x3._graphs[key].plan_name


def test_plan_match_is_exact_engine_token() -> None:
    """eng71 with the pinned knob is listed first; the pin is eng7's."""
    names = ["eng71_k24=13", "eng7_k24=12", "eng7_k24=13", "eng1_k24=13"]
    pick = _cudnn.select_plan(
        "test",
        (0,),
        names,
        pinned="eng7_k24=13",
        match=lambda nm: _fp8_conv3x3._is_eng7(nm) and _fp8_conv3x3._k24(nm) == 13,
        unpinned=lambda nms: 0,
    )
    assert names[pick] == "eng7_k24=13"


def test_missing_pinned_plan_fails_loudly() -> None:
    """A cuDNN that no longer lists the pinned plan must not run another one."""
    with pytest.raises(RuntimeError, match="pinned cuDNN plan"):
        _cudnn.select_plan(
            "test",
            (0,),
            ["eng71_k24=13", "eng7_k24=12"],
            pinned="eng7_k24=13",
            match=lambda nm: _fp8_conv3x3._is_eng7(nm) and _fp8_conv3x3._k24(nm) == 13,
            unpinned=lambda nms: 0,
        )


@sm120
def test_guard_refuses_oihw_weight() -> None:
    """An OIHW [Cout, Cin, 3, 3] weight is read as KRSC: wrong taps, plausible output."""
    xq, s_x, wq, s_w, bias, residual = _build(op.CELLS[0].spec, _seed("oihw"))
    with pytest.raises(AssertionError, match="KRSC"):
        with validating(op):
            op(xq, s_x, wq.permute(0, 3, 1, 2).contiguous(), s_w, bias, residual)
