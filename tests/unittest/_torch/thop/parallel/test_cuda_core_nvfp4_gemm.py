# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops.torch_custom_ops import CudaCoreNVFP4Runner
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(get_sm_version() < 100, reason="NVFP4 requires Blackwell")


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    # Finite E4M3 block scales and all E2M1 payload codes, including signed zero.
    a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    sa = torch.randint(
        32, 97, (((m + 127) // 128) * ((k // 16 + 3) // 4) * 512,), device="cuda", dtype=torch.uint8
    )
    sb = torch.randint(
        32, 97, (((n + 127) // 128) * ((k // 16 + 3) // 4) * 512,), device="cuda", dtype=torch.uint8
    )
    alpha = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    return a, b, sa, sb, alpha


def _linear_activation_scales(a: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, packed_k = a.shape
    padded_rows = (rows + 127) // 128 * 128
    # The legacy row-major API expects compact rows, including when K/16 is
    # not a multiple of four. The producer's swizzled layout pads those columns.
    return torch.ops.trtllm.block_scale_interleave_reverse(scales.view(padded_rows, -1))[
        :rows, : packed_k // 8
    ].contiguous()


@pytest.mark.parametrize(
    "m", [1, 2, 4, 8, 16, 32, 64, 128, 256, 3, 7, 15, 17, 31, 33, 65, 127, 129, 255, 257]
)
@pytest.mark.parametrize("n,k", [(18, 96), (130, 160)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_swizzled_activation_scales(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    torch.manual_seed(55)
    a, b, sa, sb, alpha = _inputs(m, n, k)
    bias = torch.randn(n, device="cuda", dtype=dtype)

    def reference(bias_value: torch.Tensor | None) -> torch.Tensor:
        return torch.ops.trtllm.cuda_core_nvfp4_gemm(
            a, b, _linear_activation_scales(a, sa), sb, alpha, bias_value, dtype
        )

    for bias_value in (None, bias):
        actual = torch.ops.trtllm.cuda_core_nvfp4_gemm(
            a, b, sa, sb, alpha, bias_value, dtype, scale_a_swizzled=True
        )
        torch.testing.assert_close(actual, reference(bias_value), atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops.trtllm.cuda_core_nvfp4_gemm(
            a, b, sa, sb, alpha, bias, dtype, scale_a_swizzled=True
        )
    for value in (0.02, 0.005):
        a.random_(0, 256)
        sa.random_(32, 97)
        alpha.fill_(value)
        graph.replay()
        torch.testing.assert_close(captured, reference(bias), atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128, 256])
@torch.inference_mode()
def test_runner_reads_scales_directly(m: int) -> None:
    inputs = _inputs(m, 128, 96)
    a, b, sa, sb, alpha = inputs
    expected = torch.ops.trtllm.cuda_core_nvfp4_gemm(
        a, b, _linear_activation_scales(a, sa), sb, alpha, None, torch.bfloat16
    )
    runner = CudaCoreNVFP4Runner(0, torch.bfloat16)
    assert runner.get_valid_tactics(list(inputs), profile=None) == [0]
    with patch("torch.ops.trtllm.block_scale_interleave_reverse") as reverse:
        actual = runner.forward(list(inputs))
    reverse.assert_not_called()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 256])
def test_swizzled_fake_tensor(m: int) -> None:
    args = _inputs(m, 128, 64)
    mode = FakeTensorMode()
    fake_args = [mode.from_tensor(tensor) for tensor in args]
    with mode:
        output = torch.ops.trtllm.cuda_core_nvfp4_gemm(
            *fake_args, None, torch.bfloat16, scale_a_swizzled=True
        )
    assert output.shape == (m, 128)
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("which", [2, 3], ids=["activation", "weight"])
@pytest.mark.parametrize("invalid", ["strided", "unaligned", "undersized", "cpu"])
def test_scale_buffer_validation(which: int, invalid: str) -> None:
    args = list(_inputs(129, 128, 64))
    if invalid == "strided":
        args[which] = args[which].repeat_interleave(2)[::2]
        message = "must be contiguous"
    elif invalid == "unaligned":
        args[which] = torch.cat((args[which][:1], args[which]))[1:]
        message = "must be 2-byte aligned"
    elif invalid == "undersized":
        args[which] = args[which][:-2]
        message = "buffer is too small"
    else:
        args[which] = args[which].cpu()
        message = "CUDA|cuda"
    with pytest.raises(RuntimeError, match=message):
        torch.ops.trtllm.cuda_core_nvfp4_gemm(*args, None, torch.bfloat16, scale_a_swizzled=True)


@pytest.mark.parametrize("which", [2, 3], ids=["activation", "weight"])
def test_scale_buffer_device_mismatch(which: int) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Two visible CUDA devices required")
    args = list(_inputs(2, 128, 64))
    other = (args[0].device.index + 1) % torch.cuda.device_count()
    args[which] = args[which].to(f"cuda:{other}")
    with pytest.raises(RuntimeError, match="activation device"):
        torch.ops.trtllm.cuda_core_nvfp4_gemm(*args, None, torch.bfloat16, scale_a_swizzled=True)


@pytest.mark.parametrize("which", [0, 1], ids=["activation", "weight"])
def test_payload_alignment(which: int) -> None:
    args = list(_inputs(2, 128, 64))
    original = args[which]
    args[which] = torch.cat((original.flatten()[:1], original.flatten()))[1:].view_as(original)
    assert args[which].is_contiguous()
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        torch.ops.trtllm.cuda_core_nvfp4_gemm(*args, None, torch.bfloat16, scale_a_swizzled=True)


@pytest.mark.parametrize("swizzled", [False, True])
def test_scale_layout_rejects_incomplete_vectors(swizzled: bool) -> None:
    args = _inputs(2, 128, 48)
    with pytest.raises(RuntimeError, match="K divisible by 32"):
        torch.ops.trtllm.cuda_core_nvfp4_gemm(
            *args, None, torch.bfloat16, scale_a_swizzled=swizzled
        )


@pytest.mark.parametrize("swizzled", [False, True])
def test_empty_batch(swizzled: bool) -> None:
    args = _inputs(0, 128, 64)
    result = torch.ops.trtllm.cuda_core_nvfp4_gemm(
        *args, None, torch.bfloat16, scale_a_swizzled=swizzled
    )
    assert result.shape == (0, 128)
