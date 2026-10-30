# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact 3-kernel RMSNorm of the Qwen3-VL text encoder.

Bitwise the eager Qwen3-VL RMSNorm body

    n = x.float(); var = n.pow(2).mean(-1, keepdim=True)
    n = n * torch.rsqrt(var + eps); return weight * n.to(x.dtype)

which is 8 ATen kernels. Here:
  K1 (Triton) sq = f32(x) * f32(x)            (ATen pow(x, 2) is base * base)
  K2 (ATen)   var = sq.mean(-1, keepdim=True)  (same shape, strides and dtype: same reduce order)
  K3 (Triton) out = bf16(f32(w) * f32(bf16(f32(x) * rsqrt(var + f32(eps)))))
No mul + add pair exists, so no FMA can form; casts round to nearest even.

Registered as ``torch.ops.trtllm.flux3_klein_text_rms_norm`` (opaque to torch.compile,
which would otherwise fuse and reorder the reduction).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK = 1024
_NUM_WARPS = 4
#: Last dims of Qwen3-VL-4B's norms: the hidden size and the q / k norm head dim.
DIMS = (128, 2560)


def supported(x: Tensor, weight: Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.device == x.device
        and x.dim() >= 1
        and x.shape[-1] in DIMS
        and weight.dim() == 1
        and weight.shape[0] == x.shape[-1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and 0 < x.numel() < 2**31 - _BLOCK
    )


@triton.jit
def _square_kernel(x_ptr, sq_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(sq_ptr + offs, x * x, mask=m)


@triton.jit
def _finalize_kernel(x_ptr, var_ptr, w_ptr, out_ptr, n, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    row = offs // D
    col = offs % D
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v = tl.load(var_ptr + row, mask=m, other=1.0)
    w = tl.load(w_ptr + col, mask=m, other=0.0).to(tl.float32)
    r = tl.math.rsqrt(v + eps)
    y = (x * r).to(tl.bfloat16)
    out = (w * y.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=m)


@torch.library.custom_op("trtllm::flux3_klein_text_rms_norm", mutates_args=())
def text_rms_norm_op(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Qwen3-VL RMSNorm of bf16 x [..., D] (D in DIMS, contiguous), bitwise the eager body."""
    if not supported(x, weight):
        raise ValueError(
            "flux3_klein_text_rms_norm takes contiguous CUDA bf16 x with last dim in "
            f"{DIMS} and a matching bf16 weight; got x {tuple(x.shape)} {x.dtype} "
            f"contiguous={x.is_contiguous()}, weight {tuple(weight.shape)} {weight.dtype}"
        )
    n = x.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    sq = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _square_kernel[grid](x, sq, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS, enable_fp_fusion=False)
    var = torch.mean(sq, -1, keepdim=True)
    out = torch.empty_like(x)
    _finalize_kernel[grid](
        x,
        var,
        weight,
        out,
        n,
        float(eps),
        D=x.shape[-1],
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    return out


@text_rms_norm_op.register_fake
def _(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return torch.empty_like(x)
