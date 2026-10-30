# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8r_quantize entry: both kernels bitwise the reference (the per-token
E4M3 recipe with exact division), and the one-pass kernel bitwise warp_fp8_per_token_quant."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.quantization.fp8r_quantize import (  # noqa: E501
    fp8r_quantize as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

DEV = "cuda"


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(1, m, k, generator=g, device=DEV) * 5).to(torch.bfloat16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    x = _x(s["m"], s["k"], zlib.crc32(cell.why.encode()))
    data, scales = op(x)
    ref_data, ref_scales = op.reference(x)
    assert torch.equal(scales, ref_scales)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))


def test_one_pass_bitwise_warp_quant() -> None:
    x = _x(16384, 3072, seed=1)
    data, scales = op(x)
    want_data, want_scales = torch.ops.trtllm.warp_fp8_per_token_quant(x.reshape(-1, 3072), 1e-12)
    assert torch.equal(data.view(torch.uint8), want_data.view(torch.uint8))
    assert torch.equal(scales, want_scales.reshape(-1))
