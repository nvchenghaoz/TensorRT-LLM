# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exact 3-kernel RMSNorm for the FLUX.3 Qwen3-VL text encoder.
#
# Bitwise equal to the eager body of trtllm::flux3_text_rms_norm:
#     n = x.float(); var = n.pow(2).mean(-1, keepdim=True)
#     n = n * torch.rsqrt(var + eps); return weight * n.to(x.dtype)
# which is 8 ATen kernels. Here:
#   K1 (Triton) sq = f32(x) * f32(x)            (ATen pow(x, 2) is base * base)
#   K2 (ATen)   var = sq.mean(-1, keepdim=True)  (same shape/strides/dtype, same reduce order)
#   K3 (Triton) out = bf16(f32(w) * f32(bf16(f32(x) * rsqrt(var + f32(eps)))))
# No mul+add pair exists, so no FMA can form; casts are default round-to-nearest-even.
# Kill switch: TRTLLM_FLUX3_TEXT_FUSED_NORM=0 (read at call time).

import os

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_DIMS = (128, 2560)
# Set when the Triton rsqrt is not bitwise equal to torch.rsqrt: then
# r = torch.rsqrt(var + eps) is computed by ATen (2 small kernels) and passed to K3.
ATEN_RSQRT = False

# Counters of fused vs eager calls of the custom op (diagnostics only).
CALLS = {"fused": 0, "eager": 0}


def enabled() -> bool:
    return os.environ.get("TRTLLM_FLUX3_TEXT_FUSED_NORM", "1") != "0"


def supports(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.device == x.device
        and x.dim() >= 1
        and x.shape[-1] in _DIMS
        and weight.dim() == 1
        and weight.shape[0] == x.shape[-1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and 0 < x.numel() < 2**31 - _BLOCK
    )


@triton.jit
def _rsqrt(v):
    return tl.math.rsqrt(v)


@triton.jit
def _square_kernel(x_ptr, sq_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(sq_ptr + offs, x * x, mask=m)


@triton.jit
def _finalize_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    out_ptr,
    n,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    R_IS_RSQRT: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    row = offs // D
    col = offs % D
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v = tl.load(r_ptr + row, mask=m, other=1.0)
    w = tl.load(w_ptr + col, mask=m, other=0.0).to(tl.float32)
    if R_IS_RSQRT:
        r = v
    else:
        r = _rsqrt(v + eps)
    y = (x * r).to(tl.bfloat16)
    out = (w * y.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=m)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused path; callers check supports(x, weight) first."""
    n = x.numel()
    d = x.shape[-1]
    grid = (triton.cdiv(n, _BLOCK),)
    sq = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _square_kernel[grid](x, sq, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS, enable_fp_fusion=False)
    var = torch.mean(sq, -1, keepdim=True)
    if ATEN_RSQRT:
        var = torch.rsqrt(var.add_(eps))
    out = torch.empty_like(x)
    _finalize_kernel[grid](
        x,
        var,
        weight,
        out,
        n,
        float(eps),
        D=d,
        BLOCK=_BLOCK,
        R_IS_RSQRT=ATEN_RSQRT,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    return out
