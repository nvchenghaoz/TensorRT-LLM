# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-core BF16 GEMV for small-N / small-M Linears on SM120/121.

At decode M=2-8 on SM120, cuBLAS runs BF16 ``[M,K] x [N,K]^T`` with small N
as ``cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2`` (N/16
one-warp CTAs that walk K serially, ~27-32 us at K=5120 whatever N is), and at
M=1 as ``dot_kernel`` + ``reduce_1Block_kernel`` (~5.5 us). This Triton kernel
streams the weight once with all M rows per program, accumulates in FP32 and
rounds once to the output dtype (same math as cuBLAS, another summation order).

``maybe_small_n_gemv`` returns ``None`` whenever the route does not apply
(prefill M > ``MAX_M``, other GPUs, other dtypes, bias, strided operands), so
callers fall back to their normal path unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

MAX_M = 8
# cuBLAS already streams N >= 512 at K=5120 as fast as this kernel (sweep);
# its slow one-warp wmma pick is at small N.
MAX_N = 128
MAX_K = 16384
_ENABLED = triton is not None
_SM_OK: dict = {}


if triton is not None:

    @triton.jit
    def _small_n_gemv_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        M,
        N,
        stride_xm,
        stride_wn,
        stride_ym,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k in tl.static_range(0, K, BLOCK_K):
            x = tl.load(x_ptrs + k, mask=m_mask[:, None], other=0.0).to(tl.float32)
            w = tl.load(w_ptrs + k, mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc += x[:, None, :] * w[None, :, :]
        out = tl.sum(acc, axis=2)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


def _sm120(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ok = _SM_OK.get(idx)
    if ok is None:
        ok = torch.cuda.get_device_capability(idx)[0] == 12
        _SM_OK[idx] = ok
    return ok


def default_config(m: int, n: int, k: int):
    """(BLOCK_N, BLOCK_K, num_warps) for an [m, k] x [n, k]^T GEMV.

    Picked from a cold-L2 CUDA-graph sweep on RTX PRO 6000 (SM120) at
    N=32/K=5120 and N=128/K=512, M=1-8 (experiments/ascent/csa2_small_n_gemv).
    """
    if k >= 4096:
        if m == 1:
            return 1, 256, 8
        if k % 1024 == 0:
            return 1, 1024, 4
    if k % 512 == 0:
        return 1, 512, 4
    return 1, 256, 4


def small_n_gemv(x: torch.Tensor, weight: torch.Tensor, config=None) -> torch.Tensor:
    """y = x @ weight.T for x [M, K] (unit inner stride), weight [N, K] contiguous."""
    m, k = x.shape
    n = weight.shape[0]
    block_n, block_k, warps = config if config is not None else default_config(m, n, k)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m = triton.next_power_of_2(m)
    grid = (triton.cdiv(n, block_n),)
    _small_n_gemv_kernel[grid](
        x,
        weight,
        y,
        m,
        n,
        x.stride(0),
        weight.stride(0),
        y.stride(0),
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
    )
    return y


def maybe_small_n_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Run the GEMV when it applies; otherwise return None (caller keeps cuBLAS)."""
    if not _ENABLED or bias is not None:
        return None
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or not x.is_cuda:
        return None
    if weight.dim() != 2 or x.shape[-1] != weight.shape[1]:
        return None
    lead = x.shape[:-1]
    x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    n = weight.shape[0]
    if m < 1 or m > MAX_M or n > MAX_N or n % 2 or k > MAX_K or k % 256:
        return None
    if x2.stride(1) != 1 or weight.stride(1) != 1 or weight.stride(0) != k:
        return None
    if x2.data_ptr() % 16 or weight.data_ptr() % 16 or x2.stride(0) % 8:
        return None
    if torch.compiler.is_compiling() or not _sm120(x.device):
        return None
    y = small_n_gemv(x2, weight)
    return y if x.dim() == 2 else y.view(*lead, n)


def small_n_linear(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a BF16, bias-free, unsharded Linear, via the GEMV at decode M."""
    weight = getattr(linear, "weight", None)
    if (
        weight is not None
        and getattr(linear, "bias", None) is None
        and getattr(linear, "tp_size", 1) == 1
        and getattr(linear, "lora", None) is None
    ):
        out = maybe_small_n_gemv(x, weight)
        if out is not None:
            return out
    return linear(x)
