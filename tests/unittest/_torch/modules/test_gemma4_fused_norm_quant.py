# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify batched normalization arithmetic at FP8 boundaries and during graph replay."""

from collections.abc import Callable

import pytest
import torch

from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_residual_add import rmsnorm_residual_add
from tensorrt_llm._utils import get_sm_version


def _get_fusion() -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not torch.cuda.is_available() or get_sm_version() < 100:
        pytest.skip("NVFP4 requires Blackwell")
    return pytest.importorskip(
        "tensorrt_llm._torch.modules.gemma4.fused_norm_quant"
    ).gemma4_fused_norm_add_fp4


def _assert_equal(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    m, h = actual[0].shape
    padded_m = (m + 127) // 128 * 128
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if index == 2:
            # Compare valid rows and columns in the padded 128x4 layout.
            left = torch.ops.trtllm.block_scale_interleave_reverse(left.view(padded_m, -1))[
                :m, : h // 16
            ]
            right = torch.ops.trtllm.block_scale_interleave_reverse(right.view(padded_m, -1))[
                :m, : h // 16
            ]
        assert torch.equal(left, right)


@pytest.mark.parametrize(
    "seed, scale_bits", [(111, 1077503422), (115, 1075962387), (117, 1082533449), (119, 1078999434)]
)
@torch.no_grad()
def test_rounded_division_at_fp8_midpoints(seed: int, scale_bits: int) -> None:
    fuse = _get_fusion()
    # Scales put a block near an E4M3 midpoint. Reciprocal multiplication or
    # fusing epsilon changes several payload/scale bytes on these inputs.
    generator = torch.Generator().manual_seed(seed)
    residual = torch.randn(1, 5376, generator=generator, dtype=torch.bfloat16).cuda()
    x = torch.zeros_like(residual)
    weight = torch.ones(5376, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([scale_bits], device="cuda", dtype=torch.int32).view(torch.float32)
    expected = (residual, *rmsnorm_fp4_quant(residual, weight, 1e-6, scale))
    actual = fuse(x, residual, weight, weight, scale, 1e-6, 1e-6)
    _assert_equal(actual, expected)


@pytest.mark.parametrize(
    "pattern", ["random", "zeros", "sparse", "zero_weights", "small", "large", "zero_residual"]
)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128, 256, 17, 129, 257, 1024])
@torch.no_grad()
def test_changed_graph_inputs_and_zero_scales(pattern: str, m: int) -> None:
    fuse = _get_fusion()
    torch.manual_seed(835)
    x = torch.randn(m, 5376, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(5376, device="cuda", dtype=torch.bfloat16)
    next_weight = torch.randn_like(weight)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    def reference() -> tuple[torch.Tensor, ...]:
        output = rmsnorm_residual_add(x, residual, weight, 1e-6)
        return output, *rmsnorm_fp4_quant(output, next_weight, 1e-5, scale)

    def candidate() -> tuple[torch.Tensor, ...]:
        return fuse(x, residual, weight, next_weight, scale, 1e-6, 1e-5)

    reference()
    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        expected = reference()
        actual = candidate()
    for global_scale in (0.0001, 0.01, 1.0, 100.0, 10000.0):
        for _ in range(2):
            x.normal_()
            residual.normal_()
            weight.normal_()
            next_weight.normal_()
            scale.fill_(global_scale)
            if pattern == "zeros":
                x.zero_()
                residual.zero_()
            elif pattern == "sparse":
                x.zero_()
                residual.zero_()
                x[0, 123] = 1000
            elif pattern == "zero_weights":
                next_weight[:256].zero_()
            elif pattern == "small":
                x.mul_(0.0001)
                residual.mul_(0.0001)
            elif pattern == "large":
                x[:, :2688].mul_(1000)
                x[:, 2688:].mul_(0.0001)
            elif pattern == "zero_residual":
                residual.zero_()
            graph.replay()
            _assert_equal(actual, expected)


@pytest.mark.parametrize(
    "h", [64, 96, 128, 256, 512, 1024, 2048, 3072, 4096, 5376, 6144, 8192, 16384]
)
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_hidden_width_and_row_strides(h: int, strided: bool) -> None:
    fuse = _get_fusion()
    torch.manual_seed(839)
    m = 17
    x = torch.randn(m, h * (2 if strided else 1), device="cuda", dtype=torch.bfloat16)[:, :h]
    residual = torch.randn(m, h * (3 if strided else 1), device="cuda", dtype=x.dtype)[:, :h]
    weight = torch.randn(h, device="cuda", dtype=x.dtype)
    next_weight = torch.randn_like(weight)
    scale = torch.tensor([2.7], device="cuda")
    for _ in range(5):
        x.normal_()
        residual.normal_()
        output = rmsnorm_residual_add(x, residual, weight, 1e-6)
        expected = output, *rmsnorm_fp4_quant(output, next_weight, 1e-5, scale)
        _assert_equal(fuse(x, residual, weight, next_weight, scale, 1e-6, 1e-5), expected)


@torch.no_grad()
def test_empty_batch() -> None:
    fuse = _get_fusion()
    x = torch.empty(0, 5376, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(5376, device="cuda", dtype=x.dtype)
    output, fp4, scales = fuse(x, x, weight, weight, torch.ones(1, device="cuda"), 1e-6, 1e-6)
    assert output.shape == (0, 5376)
    assert fp4.shape == (0, 2688)
    assert scales.numel() == 0
