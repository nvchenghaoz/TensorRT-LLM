# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8r_gemm entry.

Cells against the reference (fp32 matmul of the E4M3 values, the two scale multiplies in
fp32) in a stated band (one bf16 ulp beyond the fp32 summation-order bound); the
exactness claim on its own: bitwise torch._scaled_mm with row-wise scales on every cell,
and every tile (arms 1, 3, 4) gives the same bytes.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _prepare as prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.fp8r_gemm import (  # noqa: E501
    fp8r_gemm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.quantization.fp8r_quantize import (  # noqa: E501
    fp8r_quantize,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_operands import DEV, assert_bf16_band, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

#: fp32 accumulation order (k32 MMA chain vs the reference matmul) crosses a BF16 rounding
#: midpoint in ~5e-4 of the outputs, one ulp.
MAX_FRAC = 2e-3


def _summation_bound(x8, w8, x_scale, w_scale) -> torch.Tensor:
    """Absolute allowance per output for the two accumulation orders. Every E4M3 x E4M3
    product is exact in fp32, and a sum of K exact terms in any order is within
    (K - 1) * 2^-24 * sum_k |x_k w_k| of the exact sum, so two orders differ by at most twice
    that; the epilogue scales it by x_scale * w_scale. At a cancellation (|out| far below
    sum |x w|) this, not the ulp at the reference's magnitude, bounds the difference."""
    k = x8.shape[1]
    with true_fp32_matmul():
        magnitude = x8.float().abs() @ w8.float().abs().t()
    return (2 * (k - 1) * 2.0**-24) * (
        w_scale.reshape(1, -1).float() * (x_scale.reshape(-1, 1).float() * magnitude)
    )


def _operands(m: int, n: int, k: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=DEV) * k**-0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device=DEV).to(torch.bfloat16)
    w8, w_scale = prep.quantize_fp8r_weight(w)
    x8, x_scale = fp8r_quantize(x)
    return x8, w8, x_scale, w_scale


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    args = _operands(s["m"], s["n"], s["k"], seed_of(cell.why))
    out = op(*args)
    assert out.shape == (s["m"], s["n"]) and out.dtype == torch.bfloat16
    assert_bf16_band(
        out, op.reference(*args), MAX_FRAC, max_ulps=1, abs_tol=_summation_bound(*args)
    )
    x8, w8, x_scale, w_scale = args
    want = torch._scaled_mm(
        x8, w8.t(), scale_a=x_scale.reshape(-1, 1), scale_b=w_scale.reshape(1, -1),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize(
    "n,k", [(9216, 3072), (18432, 3072), (3072, 9216)], ids=["qkv", "gate_up", "down"]
)
def test_every_arm_gives_the_same_bytes(n, k) -> None:
    x8, w8, x_scale, w_scale = _operands(4096, n, k, seed=n + k)
    outs = [torch.ops.trtllm.flux3_fp8r_gemm(x8, w8, x_scale, w_scale, arm) for arm in (1, 3, 4)]
    for o in outs[1:]:
        assert torch.equal(o.view(torch.int16), outs[0].view(torch.int16))
