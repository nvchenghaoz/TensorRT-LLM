# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bf16_proj_gemm entry (the attention projections, routed by row count).

Per cell, with the route the target passes (`BF16_PROJ_ROUTES[(in, out)]`):

* bitwise `F.linear` -- the entry's claim: the SM120 sequential-K GEMM, the pinned
  `trtllm::cublas_mm` algorithms and `F.linear` all accumulate in fp32 over ascending K without
  split-K and round once;
* against the fp32 reference (true fp32 matmul): summation order only, so a one-ulp flip at a
  bf16 midpoint (2 units of the row maximum's 2^-8); 4 units element-wise, 2 of relative RMS.

Beyond the cells: the target's native op `sm120_bf16_seqk_gemm` at every tile config is bitwise
`F.linear` on the two qkv shapes it is routed for, over its routed row ranges (288-320). Only
there is `F.linear` claimed split-K free; elsewhere cuBLASLt's heuristic may pick a split-K
kernel, which the native op is not compared against.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1 import (  # noqa: E501
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm.bf16_proj_gemm import (  # noqa: E501
    BF16_PROJ_ROUTES,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm.bf16_proj_gemm import (  # noqa: E501
    bf16_proj_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import seed_of  # noqa: E402
from _precision import true_fp32_matmul  # noqa: E402
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

#: (in, out): sliding qkv, full-attention [q | k], sliding o_proj, full-attention o_proj.
SHAPES = ((5376, 16384), (5376, 18432), (8192, 5376), (16384, 5376))
#: kernels::sm120_seqk_gemm::Config (kNumConfigs = 4).
SEQK_CONFIGS = (0, 1, 2, 3)
_WEIGHTS: dict = {}


def _weight(k: int, n: int) -> torch.Tensor:
    if (k, n) not in _WEIGHTS:
        g = torch.Generator(device="cuda").manual_seed(k * 7 + n)
        _WEIGHTS[(k, n)] = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(
            torch.bfloat16
        )
    return _WEIGHTS[(k, n)]


def _activation(m: int, k: int, seed: int) -> torch.Tensor:
    """Normed hidden states (qkv) or attention outputs (o_proj): unit scale, a few outlier
    channels."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda")
    x[:, :16] *= 20.0
    return x.to(torch.bfloat16)


def _route_taken(route, m: int) -> str:
    if m in route.seqk_rows:
        return "seqk"
    return "cublas_mm" if m in route.cublas_rows else "F.linear"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    k, n, m = cell.spec["k"], cell.spec["n"], cell.spec["m"]
    x, w = _activation(m, k, seed_of(cell.why)), _weight(k, n)
    route = BF16_PROJ_ROUTES[(k, n)]
    out = op(x, w, route)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16)), _route_taken(
        route, m
    )
    with true_fp32_matmul():
        ref = op.reference(x, w, route)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def test_the_cells_reach_every_route() -> None:
    taken = {
        _route_taken(BF16_PROJ_ROUTES[(c.spec["k"], c.spec["n"])], c.spec["m"]) for c in op.CELLS
    }
    assert taken == {"seqk", "cublas_mm", "F.linear"}, taken


@pytest.mark.parametrize("config", SEQK_CONFIGS)
@pytest.mark.parametrize("shape", [s for s in SHAPES if s[1] in (16384, 18432)], ids=str)
def test_native_seqk_gemm_is_bitwise_f_linear(shape, config) -> None:
    k, n = shape
    w = _weight(k, n)
    for m in (288, 289, 300, 310, 320):
        x = _activation(m, k, seed=m + config)
        out = native_ops.sm120_bf16_seqk_gemm(x, w, config)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        assert torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16)), (m, config)
