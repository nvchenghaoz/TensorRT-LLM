# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target Triton ops against their PyTorch references, plus the frozen run switches.

``mhc_final_collapse`` replaces ``mHC.collapse`` on the last layer and must be bit-identical to it;
``small_n_gemv`` replaces cuBLAS for small-N decode Linears and differs only in summation order.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v4.v41_flash_1017__sm_120__tp4_moetp2ep2 import (  # noqa: E501
    switches,
)


def test_run_switches_ignore_the_environment(monkeypatch):
    monkeypatch.setenv("TRTLLM_DSV4_FUSED_ROUTER", "0")
    assert switches.env_get("TRTLLM_DSV4_FUSED_ROUTER", "1") == "1"
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    assert switches.env_get("TRTLLM_ENABLE_PDL", "1") == "0"


_needs_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="the target's kernels are tuned for SM120",
)


@_needs_sm120
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_final_collapse_is_bitwise_mhc_collapse(m):
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v4.v41_flash_1017__sm_120__tp4_moetp2ep2.ops.mhc import (  # noqa: E501
        hyper_connection,
        mhc_final_collapse,
    )

    g = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn((m, 4, 5120), generator=g, device="cuda").to(torch.bfloat16)
    pre = torch.rand((m, 4, 1), generator=g, device="cuda", dtype=torch.float32)
    assert mhc_final_collapse.final_collapse_supported(x, pre)
    out = mhc_final_collapse.mhc_final_collapse(x, pre)
    torch.testing.assert_close(out, hyper_connection.mHC.collapse(x, pre), rtol=0, atol=0)


@_needs_sm120
@pytest.mark.parametrize("m, n", [(1, 64), (2, 64), (4, 128), (8, 32)])
def test_small_n_gemv_matches_matmul(m, n):
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v4.v41_flash_1017__sm_120__tp4_moetp2ep2.ops import (  # noqa: E501
        small_n_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(m * 1000 + n)
    x = torch.randn((m, 5120), generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn((n, 5120), generator=g, device="cuda") / 64).to(torch.bfloat16)
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(small_n_gemv.small_n_gemv(x, w), ref, rtol=1e-2, atol=1e-2)
