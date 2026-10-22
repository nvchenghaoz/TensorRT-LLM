# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-N, small-M BF16 GEMV for decode-size unquantized Linears on SM120/121.

On SM120/121 every BF16 ``Linear`` goes to cuBLASLt. For tiny output widths
(e.g. the GDN ``in_proj_ba``, N=96, K=5120) and decode batch sizes, cuBLAS
picks ``cutlass_80_wmma_tensorop_bf16_s161616gemm_16x16_128x2`` (M=2..4),
which launches only N/16 CTAs that walk K serially and is bound by DRAM
latency, or ``dot_kernel`` + ``reduce_1Block_kernel`` (M=1).

This module computes ``y[M, N] = x[M, K] @ w[N, K]^T`` with a CUDA-core Triton
GEMV: each program owns ``BLOCK_N`` weight rows and ``BLOCK_M`` activation rows,
issues all of its K loads in one statically unrolled pass, accumulates
elementwise products in FP32 registers and reduces them once with a fixed tree.
The reduction order depends only on the (static) launch config, so repeated
calls, eager execution and CUDA-graph replay are bitwise identical. Output class
vs cuBLAS: reordered (same BF16 operands, FP32 accumulation).

Routing (``maybe_small_n_gemv``) is static and conservative: SM120/121 only,
BF16 activations and weights, no bias, contiguous 16-byte aligned operands, N
and M within the measured route table. Prefill and wide Linears (e.g. the BF16
lm_head) keep their cuBLAS path. Kill switch: ``TRTLLM_SMALL_N_BF16_GEMV=0``.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

SMALL_N_GEMV_ENABLED = os.environ.get("TRTLLM_SMALL_N_BF16_GEMV", "1") != "0"

# Route table: (max N, max M). Measured on SM120 (RTX PRO 6000) at N=96,
# K=5120 with cold L2 and CUDA-graph replay (experiments/ascent/small_n_gemv/):
# per call 2.9/2.9/3.2/3.4 us vs cuBLAS 5.1/31.6/31.6/31.6 us at M=1..4.
# M=5..16: cuBLAS runs nvjet split-K + splitKreduce (two launches) on the aux
# stream beside in_proj_qkvz's Marlin; in the 48-layer GDN decode graph window
# (experiments/ascent/ba_decode/window.py) the one-launch GEMV exposes
# 0.26-0.51 us/layer vs cuBLAS 0.70-1.0 us/layer. At M>16 (W4A4 in_proj_qkvz)
# cuBLAS's fork is fully hidden (exposure <= 0), so those rows keep cuBLAS.
_MAX_M_BY_N = ((128, 16), )
_MAX_N = max(n for n, _ in _MAX_M_BY_N)
_MAX_M = max(m for _, m in _MAX_M_BY_N)
# Upper bound on K so the statically unrolled K loop stays small.
_MAX_K = 16384

_sm_supported: Optional[bool] = None


def _is_supported_sm() -> bool:
    global _sm_supported
    if _sm_supported is None:
        _sm_supported = (torch.cuda.is_available()
                         and get_sm_version() in (120, 121))
    return _sm_supported


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
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_rows = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
    w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
    # K % BLOCK_K == 0 is guaranteed by the caller.
    for k0 in tl.static_range(0, K, BLOCK_K):
        kk = k0 + offs_k
        w = tl.load(w_rows + kk[None, :], mask=mask_n[:, None], other=0.0)
        x = tl.load(x_rows + kk[None, :], mask=mask_m[:, None], other=0.0)
        acc += x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :]
    y = tl.sum(acc, axis=2)
    y_ptrs = (y_ptr + offs_m[:, None].to(tl.int64) * stride_ym +
              offs_n[None, :])
    tl.store(y_ptrs,
             y.to(y_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def _default_config(m: int, n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps) for an [m, k] x [n, k]^T GEMV."""
    # One weight row and one activation row per program, 8 warps, one BF16
    # element per thread per 256-wide K step: the fastest config at M<=16 (the
    # routed range) next to in_proj_qkvz on the other stream (more programs,
    # each short, fill the SMs the partner leaves free). Larger M (only
    # reached by benchmarks) amortizes weight loads over more rows.
    if m <= 16 or k % 1024 != 0:
        return 1, 1, 256, 8
    if m <= 8:
        return 1, 2, 1024, 4
    return 2, 4, 1024, 4


def small_n_gemv(x: torch.Tensor,
                 weight: torch.Tensor,
                 config: Optional[tuple] = None) -> torch.Tensor:
    """y = x @ weight.T for 2D BF16 ``x`` [M, K] and ``weight`` [N, K].

    Returns a fresh contiguous [M, N] tensor. ``config`` overrides the
    (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) launch config (benchmarks only).
    """
    m, k = x.shape
    n = weight.shape[0]
    block_m, block_n, block_k, num_warps = (config if config is not None
                                            else _default_config(m, n, k))
    assert k % block_k == 0
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
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
        num_warps=num_warps,
    )
    return y


def _max_m_for_n(n: int) -> int:
    for max_n, max_m in _MAX_M_BY_N:
        if n <= max_n:
            return max_m
    return 0


def maybe_small_n_gemv(input: torch.Tensor,
                       weight: torch.Tensor) -> Optional[torch.Tensor]:
    """Run ``F.linear(input, weight)`` (no bias) on the small-N GEMV if routed.

    Returns None when the shape, dtype, layout or device is not routed; the
    caller then falls through to its normal (cuBLAS) path.
    """
    if not SMALL_N_GEMV_ENABLED:
        return None
    n = weight.shape[0]
    if n > _MAX_N or input.dtype != torch.bfloat16:
        return None
    k = input.shape[-1]
    # Prefill and other large-M calls keep the cuBLAS path.
    if input.numel() > _MAX_M * k:
        return None
    if (weight.dtype != torch.bfloat16 or weight.dim() != 2
            or weight.shape[1] != k or k % 256 != 0 or k > _MAX_K
            or not input.is_cuda):
        return None
    if torch.compiler.is_compiling() or not _is_supported_sm():
        return None
    x = input.reshape(-1, k) if input.dim() != 2 else input
    m = x.shape[0]
    if m == 0 or m > _max_m_for_n(n):
        return None
    if (x.stride(-1) != 1 or weight.stride(-1) != 1
            or weight.stride(0) != k or x.stride(0) % 8 != 0
            or x.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0):
        return None
    y = small_n_gemv(x, weight)
    if input.dim() != 2:
        y = y.view(*input.shape[:-1], n)
    return y
