# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the nvfp4_gemm entry (the MLP projections on the shared runners with this target's
tuning).

* GPU (sm_120), per cell: bitwise the shared `trtllm::nvfp4_gemm` (the entry's reference). On a
  cold tuner cache, as here, the entry runs its pins and dense-bucket picks (gate_up at M 300:
  the pinned cuBLASLt nvjet 256x128x128; M >= 1024: CUTLASS config 2; down at M 8000: the
  dense bucket) where the shared op runs its fallback, so each cell checks the entry's claim
  that every tactic it may pick computes the shared op's bits.
* CPU: the dense tuning buckets -- `dense_bucket` rounds M up to the dense grid inside
  [min_m, 8192] and keeps the last power of 2 outside it, and `dense_buckets` lists exactly the
  grid points every M up to its argument maps to, so the tuner profiles every bucket the
  runtime looks up.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm._nvfp4_gemm_tuning import (  # noqa: E501
    DENSE_MAX_M,
    dense_bucket,
    dense_buckets,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm.nvfp4_gemm import (  # noqa: E501
    nvfp4_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import nvfp4_operands, nvfp4_weight, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402,F401

_SM120 = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

#: `ModelConfig.nvfp4_gemm_allowed_backends`' default, which the target passes.
BACKENDS = ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
_WEIGHTS: dict = {}


def _operands(n: int, k: int, m: int, seed: int):
    """`[act_fp4, weight, act_sf, weight_sf, alpha]` as the target holds them: the activation
    quantized with swizzled scales, the weight as `weights.py` loads it, alpha = input_scale *
    weight_scale_2."""
    if (n, k) not in _WEIGHTS:
        g = torch.Generator(device="cuda").manual_seed(n + k)
        w = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
        _WEIGHTS[(n, k)] = nvfp4_weight(w)
    weight, weight_sf, weight_scale_2 = _WEIGHTS[(n, k)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    act_fp4, act_sf, act_global = nvfp4_operands(x)
    alpha = (weight_scale_2 / act_global).reshape(1)
    return [act_fp4, weight, act_sf, weight_sf, alpha]


@_SM120
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    inputs = _operands(n, k, m, seed_of(cell.why))
    out = op(*inputs, torch.bfloat16, BACKENDS)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref = op.reference(*inputs, torch.bfloat16, BACKENDS)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "m, min_m, bucket",
    [
        (255, 256, 128),  # below the dense range: last power of 2
        (256, 256, 256),
        (257, 256, 320),  # the served 300-row prompt's bucket
        (300, 256, 320),
        (1000, 256, 1024),
        (1025, 256, 1152),  # step 128 above 1024
        (2048, 256, 2048),
        (2049, 256, 2304),  # step 256 above 2048
        (4000, 4096, 2048),  # below down_proj's dense range
        (4096, 4096, 4096),
        (4097, 4096, 4608),  # step 512 above 4096
        (8000, 4096, 8192),
        (8192, 4096, 8192),
        (8193, 4096, 8192),  # above the grid: last power of 2
        (20000, 256, 16384),
    ],
)
def test_dense_bucket(m, min_m, bucket):
    assert dense_bucket(m, min_m) == bucket


@pytest.mark.cpu_only
@pytest.mark.parametrize("x, min_m", [(2048, 256), (8192, 4096), (DENSE_MAX_M, 256), (300, 256)])
def test_dense_buckets_cover_every_lookup(x, min_m):
    buckets = dense_buckets(x, min_m)
    assert list(buckets) == sorted(set(buckets))
    assert all(min_m <= b <= dense_bucket(x, min_m) for b in buckets)
    # Every grid point is its own bucket, and every M in range looks up one of them.
    assert all(dense_bucket(b, min_m) == b for b in buckets)
    assert {dense_bucket(m, min_m) for m in range(min_m, x + 1)} == set(buckets)


@pytest.mark.cpu_only
def test_dense_buckets_of_the_two_tuned_shapes():
    """gate_up (min 256, tuned up to 2048): 64-steps to 1024, then 128-steps; down_proj
    (min 4096): 4096 and the 512-steps above it."""
    assert dense_buckets(2048, 256) == (*range(256, 1025, 64), *range(1152, 2049, 128))
    assert dense_buckets(8192, 4096) == (4096, *range(4608, 8193, 512))
    assert dense_buckets(100, 256) == ()
