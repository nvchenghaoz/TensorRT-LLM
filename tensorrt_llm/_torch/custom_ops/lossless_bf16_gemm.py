# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python registration and compilation cache for exact packed BF16 projections."""

import functools
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch

from ..cute_dsl_kernels.lossless_bf16_gemm import LosslessBf16Gemm


@functools.cache
def _compile(
    m: int, n: int, k: int, reassociated: bool, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Packed BF16 GEMM must be warmed up before CUDA graph capture")
    blocks = n * (k // 128)
    specs = [
        (cutlass.BFloat16, (m, k), 16 if reassociated else 2),
        (cutlass.Uint8, (blocks, 176), 8),
        (cutlass.Int64, (blocks,), 8),
        (cutlass.Uint8, (cute.sym_int(),), 1),
        (cutlass.BFloat16, (m, n), 16),
    ]
    args = [
        cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment
        )
        for dtype, shape, alignment in specs
    ]
    return cute.compile(
        LosslessBf16Gemm(reassociated),
        *args,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::lossless_bf16_gemm_cute_dsl", mutates_args=(), device_types="cuda"
)
def lossless_bf16_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    metadata: torch.Tensor,
    exceptions: torch.Tensor,
    out_features: int,
    reassociated: bool = False,
) -> torch.Tensor:
    """Project contiguous BF16 ``[M, K]`` inputs using exact packed weight buffers."""
    if input.ndim != 2 or input.dtype != torch.bfloat16 or not input.is_contiguous():
        raise ValueError("input must be a contiguous BF16 matrix")
    m, k = input.shape
    if not 0 < k < 2**31 or k % 128 or not 0 < out_features < 2**31 or m > 4 * 65535:
        raise ValueError("require positive N and K, K divisible by 128, and M <= 262140")
    blocks = out_features * (k // 128)
    if blocks > 2**31 - 1:
        raise ValueError("packed weight index overflow")
    for tensor, dtype, shape in (
        (weight, torch.uint8, (blocks, 176)),
        (metadata, torch.int64, (blocks,)),
        (exceptions, torch.uint8, (exceptions.numel(),)),
    ):
        if tensor.dtype != dtype or tensor.shape != shape:
            raise ValueError("invalid packed buffer shape or dtype")
        if tensor.device != input.device or not tensor.is_contiguous():
            raise ValueError("packed buffers must be contiguous on the input device")
    if weight.data_ptr() % 8 or metadata.data_ptr() % 8:
        raise ValueError("packed buffers must be aligned to eight bytes")
    if reassociated and input.data_ptr() % 16:
        raise ValueError("adjacent-value input must be aligned to 16 bytes")
    output = input.new_empty((m, out_features))
    if m:
        with torch.cuda.device(input.device):
            compiled = _compile(m, out_features, k, reassociated, input.device)
            compiled(input, weight, metadata, exceptions, output)
    return output


@lossless_bf16_gemm.register_fake
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    metadata: torch.Tensor,
    exceptions: torch.Tensor,
    out_features: int,
    reassociated: bool = False,
) -> torch.Tensor:
    return input.new_empty((input.shape[0], out_features))
