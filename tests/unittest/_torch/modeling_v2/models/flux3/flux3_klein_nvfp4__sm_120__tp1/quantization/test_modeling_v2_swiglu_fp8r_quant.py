# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the swiglu_fp8r_quant entry. Cells against the reference in a stated band;
the exactness claim on its own: bitwise silu_and_mul followed by warp_fp8_per_token_quant."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.quantization.swiglu_fp8r_quant import (  # noqa: E501
    swiglu_fp8r_quant as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "../gemm"]
from _flux3_operands import assert_e4m3_band  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

DEV = "cuda"
#: Triton's sigmoid vs torch's: a bf16 h moves by an ulp at a rounding boundary.
MAX_FRAC = 2e-3


def _x(m: int, width: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(m, 2 * width, generator=g, device=DEV) * 2).to(torch.bfloat16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    x = _x(s["m"], s["width"], zlib.crc32(cell.why.encode()))
    data, scales = op(x)
    ref_data, ref_scales = op.reference(x)
    moved = scales != ref_scales
    assert moved.float().mean().item() <= MAX_FRAC
    assert_e4m3_band(data[~moved], ref_data[~moved], MAX_FRAC)


@pytest.mark.parametrize("m", [4096, 16384])
def test_bitwise_silu_and_mul_then_warp_quant(m) -> None:
    x = _x(m, 9216, seed=m)
    data, scales = op(x)
    h = torch.ops.trtllm.silu_and_mul(x)
    want_data, want_scales = torch.ops.trtllm.warp_fp8_per_token_quant(h, 1e-12)
    assert torch.equal(data.view(torch.uint8), want_data.view(torch.uint8))
    assert torch.equal(scales, want_scales.reshape(-1))
