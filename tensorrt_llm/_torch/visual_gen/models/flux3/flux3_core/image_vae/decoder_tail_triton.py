# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused tail of the FLUX.3 image VAE decoder: conv_out (3x3, C -> 3, padding 1) + bias.

One Triton kernel replaces cuDNN's fp32 conv_out (convertTensor of the 128-channel input,
a TF32 implicit GEMM padded from N=3 to a full tile, and a separate bias kernel). The
math is the same convolution with fp32 accumulation; products use 3-pass split TF32
(tf32x3, fp32-level error, ~1000x below the single-pass TF32 of cuDNN's path), so the
output class is "reordered".

Structure: a program owns a BW-pixel column strip of R output rows. It walks the R+2
input rows once; for each input row it forms, per kw shift, an fp32 FMA dot
X_kw[BW, C] @ Wkw[C, 16] (tensor cores) whose 9 live columns are (kh, o) = the row's contribution to
output rows ih+1 (kh=0), ih (kh=1) and ih-1 (kh=2). Output row ih-1 is the sum of the
kh=2 columns of row ih, kh=1 of row ih-1 and kh=0 of row ih-2 (two rolling partials),
plus bias. Each input element is read 3x through L1 (once per kw) and (R+2)/R times
from L2/DRAM. Inputs and weights are addressed by strides (NHWC or NCHW).

With the optional prologue the kernel reads the raw pre-norm activation (any float
dtype) and applies the per-channel GroupNorm affine (scale_c, shift_c) and swish in
fp32 before the convolution; zero padding is applied after the activation.

Both entry points are torch.library triton_ops, so torch.compile traces through them
without a graph break.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

# Tile: BW output columns x R output rows per program; BC input channels per dot step.
# Swept at the served shape (NHWC fp32 1x128x1024x1024, RTX PRO 6000 SM120, job 1317).
_BW = 64
_R = 16
_BC = 64
_NUM_WARPS = 4
_NUM_STAGES = 2
# tl.dot input precision: "ieee" = fp32 FMA; "tf32x3" = 3-pass split TF32 on tensor
# cores (fp32-level accuracy, far below the single-pass TF32 error of cuDNN's path).
_PRECISION = "tf32x3"

# Process-wide switch (A/B experiments, debugging).
ENABLED = True


@triton.jit
def _conv3x3_to3_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    H,
    W,
    sxn,
    sxc,
    sxh,
    sxw,
    swo,
    swc,
    swh,
    sww,
    syn,
    syc,
    syh,
    syw,
    C: tl.constexpr,
    BW: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    PROLOGUE: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    n = tl.program_id(2).to(tl.int64)
    cols = pid_w * BW + tl.arange(0, BW)
    col_ok = cols < W
    j = tl.arange(0, 16)
    kh_j = j // 3
    o_j = j - 3 * kh_j
    jv = j < 9
    xb = x_ptr + n * sxn
    yb = y_ptr + n * syn
    b0 = tl.load(b_ptr)
    b1 = tl.load(b_ptr + 1)
    b2 = tl.load(b_ptr + 2)
    r0 = pid_h * R
    s1 = tl.zeros([BW, 16], dtype=tl.float32)  # dot result of input row ih-1
    s2 = tl.zeros([BW, 16], dtype=tl.float32)  # dot result of input row ih-2
    for i in range(R + 2):
        ih = r0 - 1 + i
        row_ok = (ih >= 0) & (ih < H)
        xrow = xb + ih.to(tl.int64) * sxh
        p = tl.zeros([BW, 16], dtype=tl.float32)
        for kw in tl.static_range(3):
            iw = cols + (kw - 1)
            m = row_ok & (iw >= 0) & (iw < W)
            for c0 in tl.static_range(0, C, BC):
                cc = c0 + tl.arange(0, BC)
                xv = tl.load(
                    xrow + iw[:, None].to(tl.int64) * sxw + cc[None, :] * sxc,
                    mask=m[:, None],
                    other=0.0,
                ).to(tl.float32)
                if PROLOGUE:
                    sc = tl.load(scale_ptr + cc)
                    sh = tl.load(shift_ptr + cc)
                    xv = xv * sc[None, :] + sh[None, :]
                    xv = xv * tl.sigmoid(xv)
                    xv = tl.where(m[:, None], xv, 0.0)
                wt = tl.load(
                    w_ptr + o_j[None, :] * swo + cc[:, None] * swc + kh_j[None, :] * swh + kw * sww,
                    mask=jv[None, :],
                    other=0.0,
                )
                p = tl.dot(xv, wt, p, input_precision=PREC)
        t = (
            tl.where(kh_j[None, :] == 2, p, 0.0)
            + tl.where(kh_j[None, :] == 1, s1, 0.0)
            + tl.where(kh_j[None, :] == 0, s2, 0.0)
        )
        oh = ih - 1
        smask = col_ok & (i >= 2) & (oh < H)
        ybase = yb + oh.to(tl.int64) * syh + cols * syw
        out0 = tl.sum(tl.where(o_j[None, :] == 0, t, 0.0), axis=1) + b0
        out1 = tl.sum(tl.where(o_j[None, :] == 1, t, 0.0), axis=1) + b1
        out2 = tl.sum(tl.where(o_j[None, :] == 2, t, 0.0), axis=1) + b2
        tl.store(ybase, out0, mask=smask)
        tl.store(ybase + syc, out1, mask=smask)
        tl.store(ybase + 2 * syc, out2, mask=smask)
        s2 = s1
        s1 = p


def can_use_conv_out_kernel(x: Tensor, conv: torch.nn.Module) -> bool:
    """True when ``conv(x)`` (after casting x to conv's weight dtype) can run here."""
    if not ENABLED or not isinstance(conv, torch.nn.Conv2d):
        return False
    w = conv.weight
    return (
        x.is_cuda
        and x.dim() == 4
        and w.is_cuda
        and w.dtype == torch.float32
        and conv.bias is not None
        and conv.bias.dtype == torch.float32
        and tuple(w.shape[:1]) == (3,)
        and tuple(w.shape[2:]) == (3, 3)
        and w.shape[1] == x.shape[1]
        and x.shape[1] % _BC == 0
        and tuple(conv.stride) == (1, 1)
        and tuple(conv.padding) == (1, 1)
        and tuple(conv.dilation) == (1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
    )


def _out_like(x: Tensor) -> Tensor:
    n, _, h, w = x.shape
    fmt = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous()
        else torch.contiguous_format
    )
    return torch.empty((n, 3, h, w), device=x.device, dtype=torch.float32, memory_format=fmt)


def _launch(x: Tensor, scale, shift, weight: Tensor, bias: Tensor, prologue: bool) -> Tensor:
    n, c, h, w = x.shape
    y = _out_like(x)
    grid = (triton.cdiv(w, _BW), triton.cdiv(h, _R), n)
    if scale is None:
        scale, shift = bias, bias  # unused (PROLOGUE=False)
    wrap_triton(_conv3x3_to3_kernel)[grid](
        x,
        scale,
        shift,
        weight,
        bias,
        y,
        h,
        w,
        *x.stride(),
        *weight.stride(),
        *y.stride(),
        C=c,
        BW=_BW,
        R=_R,
        BC=_BC,
        PROLOGUE=prologue,
        PREC=_PRECISION,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return y


@triton_op("flux3_vae::conv_out_3x3", mutates_args={})
def conv_out_3x3(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """conv2d(x, weight [3, C, 3, 3], bias, padding=1), fp32 accumulation (x any strides)."""
    return _launch(x, None, None, weight, bias, False)


@triton_op("flux3_vae::norm_swish_conv_out_3x3", mutates_args={})
def norm_swish_conv_out_3x3(
    x: Tensor, scale: Tensor, shift: Tensor, weight: Tensor, bias: Tensor
) -> Tensor:
    """conv2d(swish(x * scale_c + shift_c), weight, bias, padding=1), fp32 math.

    x is the pre-norm activation (any float dtype, any strides); scale/shift are fp32
    [C] contiguous per-channel GroupNorm affine terms.
    """
    return _launch(x, scale, shift, weight, bias, True)
