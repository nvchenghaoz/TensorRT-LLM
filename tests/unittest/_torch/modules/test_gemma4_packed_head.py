# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact storage, batched arithmetic, selection and reload tests for packed heads."""

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.modules.gemma4.packed_head import (
    _PackedHeadLinearMethod,
    configure_packed_head,
    pack_lossless_bf16_weights,
)
from tensorrt_llm._torch.modules.linear import Linear


def _unpack_bits(
    data: torch.Tensor, metadata: torch.Tensor, exceptions: torch.Tensor
) -> torch.Tensor:
    sm = data[:, :128].reshape(-1, 32, 4).transpose(1, 2).reshape(-1, 128).long()
    packed = data[:, 128:].long().reshape(-1, 16, 3)
    words = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    words = torch.stack((words & 4095, words >> 12), dim=-1).reshape(-1, 32)
    codes = torch.stack([(words >> (j * 3)) & 7 for j in range(4)], dim=-1)
    codes = codes.transpose(1, 2).reshape(-1, 128)
    exponent = (metadata[:, None] & 255) - codes
    escaped = codes == 7
    offsets = (metadata >> 8) - 1
    indices = offsets[:, None] + escaped.long().cumsum(1) - 1
    exponent[escaped] = exceptions[indices[escaped]].long()
    return (sm & 127) | ((sm & 128) << 8) | (exponent << 7)


def test_every_encoding_and_chunk_offsets() -> None:
    bits = (
        torch.randperm(65536, generator=torch.Generator().manual_seed(123))
        .repeat(3)
        .reshape(-1, 128)
    )
    weight = bits.to(torch.int16).view(torch.bfloat16)
    torch.testing.assert_close(
        _unpack_bits(*pack_lossless_bf16_weights(weight)), bits, rtol=0, atol=0
    )
    torch.testing.assert_close(weight.view(torch.int16).long() & 65535, bits, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 256, 257, 1024])
@pytest.mark.parametrize("n,k", [(19, 128), (132, 384), (257, 5376)])
@torch.inference_mode()
def test_batched_gemm_and_graph_replay(m: int, n: int, k: int) -> None:
    torch.manual_seed(419)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Force exponent exceptions, including small values in otherwise large blocks.
    weight[:, ::7] *= 0.0001
    packed = pack_lossless_bf16_weights(weight)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.ops.trtllm.lossless_bf16_gemm(x, *packed, n)
    for scale in (0.01, 1.0, 10.0):
        x.normal_().mul_(scale)
        graph.replay()
        expected = torch.nn.functional.linear(x.double(), weight.double())
        # FP32 accumulation with one final BF16 round, including cancellation.
        error = (output.double() - expected).abs()
        allowance = expected.abs() * 0.004 + 2e-5 * x.double().norm(dim=1)[
            :, None
        ] * weight.double().norm(dim=1)
        assert torch.all(error <= allowance)


@pytest.mark.parametrize("m", [0, 1, 2, 129, 256])
def test_fake_output(m: int) -> None:
    with FakeTensorMode():
        x = torch.empty(m, 128, device="cuda", dtype=torch.bfloat16)
        data = torch.empty(19, 176, device="cuda", dtype=torch.uint8)
        metadata = torch.empty(19, device="cuda", dtype=torch.int64)
        exceptions = torch.empty(0, device="cuda", dtype=torch.uint8)
        output = torch.ops.trtllm.lossless_bf16_gemm(x, data, metadata, exceptions, 19)
    assert output.shape == (m, 19)
    assert output.dtype == x.dtype


@torch.inference_mode()
def test_selection_reload_and_fallbacks() -> None:
    linear = Linear(256, 128, bias=False, dtype=torch.bfloat16).cuda()
    linear.weight.normal_()
    configure_packed_head(linear)
    method = linear.quant_method
    assert isinstance(method, _PackedHeadLinearMethod)
    assert not any("packed_bf16" in key for key in linear.state_dict())
    for _ in range(2):
        for m in (1, 2, 7, 64, 256):
            x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
            for runner in method.runners:
                with patch.object(AutoTuner.get(), "choose_one", return_value=(runner, 0)):
                    result = linear(x)
                if runner.packed:
                    expected = torch.ops.trtllm.lossless_bf16_gemm(
                        x,
                        linear._packed_bf16_data,
                        linear._packed_bf16_metadata,
                        linear._packed_bf16_exceptions,
                        128,
                    )
                else:
                    expected = torch.nn.functional.linear(x, linear.weight)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        linear.weight.normal_()
        linear.cache_derived_state()
        torch.testing.assert_close(
            _unpack_bits(
                linear._packed_bf16_data,
                linear._packed_bf16_metadata,
                linear._packed_bf16_exceptions,
            ),
            linear.weight.view(torch.int16).long().reshape(-1, 128) & 65535,
            rtol=0,
            atol=0,
        )
    with (
        torch.inference_mode(False),
        torch.enable_grad(),
        patch.object(method.original, "apply") as original,
    ):
        assert method.apply(linear, x, None) is original.return_value
    with patch.object(method.original, "apply") as original:
        assert (
            method.apply(linear, x, torch.ones(128, device="cuda", dtype=x.dtype))
            is original.return_value
        )
    for flag in ("use_custom_cublas_mm", "use_cute_dsl_bf16_gemm"):
        with patch.object(linear, flag, True), patch.object(method.original, "apply") as original:
            assert method.apply(linear, x, None) is original.return_value
    with (
        patch(
            "tensorrt_llm._torch.modules.gemma4.packed_head.is_torch_compiling", return_value=True
        ),
        patch.object(method.original, "apply") as original,
    ):
        assert method.apply(linear, x, None) is original.return_value


@pytest.mark.parametrize("failure", ["input", "weight", "metadata", "exceptions", "alignment", "k"])
def test_buffer_contract(failure: str) -> None:
    packed = list(
        pack_lossless_bf16_weights(torch.ones(4, 128, device="cuda", dtype=torch.bfloat16))
    )
    x = torch.ones(3, 128, device="cuda", dtype=torch.bfloat16)
    if failure == "input":
        x = x.float()
    elif failure == "weight":
        packed[0] = packed[0][:-1]
    elif failure == "metadata":
        packed[1] = packed[1].int()
    elif failure == "exceptions":
        packed[2] = packed[2].float()
    elif failure == "alignment":
        packed[0] = torch.cat((packed[0].flatten()[:1], packed[0].flatten()))[1:].reshape_as(
            packed[0]
        )
    else:
        x = x[:, :64].contiguous()
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.lossless_bf16_gemm(x, *packed, 4)


@torch.inference_mode()
def test_empty_batch() -> None:
    packed = pack_lossless_bf16_weights(torch.ones(4, 128, device="cuda", dtype=torch.bfloat16))
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    assert torch.ops.trtllm.lossless_bf16_gemm(x, *packed, 4).shape == (0, 4)
