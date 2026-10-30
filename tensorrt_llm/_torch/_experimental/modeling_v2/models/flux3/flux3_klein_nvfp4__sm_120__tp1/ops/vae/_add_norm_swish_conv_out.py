# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused tail of the VAE decoder: the last ResnetBlock's bias + residual add, norm_out's
GroupNorm affine + swish, and conv_out (3x3, C -> 3, padding 1) + bias, as one Triton
kernel.

A program owns BW-2 output columns x R output rows and walks the R+2 input rows once.
Per input row it loads BW input pixels (a 1-pixel halo) x C channels once, applies the
prologue once and forms one dot P = X[BW, C] @ Wall[C, 32] whose 27 live columns are
(kw, kh, o). tl.gather along the pixel axis aligns each kw tap on its output pixel;
output row ih-1 is the sum of the kh=2 taps of row ih, kh=1 of row ih-1 and kh=0 of row
ih-2 (two rolling partials), plus the bias. Inputs and weights are addressed by strides.

Prologue: x + (r + r_bias_c), each sum rounded to x's dtype as the unfused bias and
residual adds store them, then x * scale_c + shift_c and swish in fp32; zero padding
applies after the activation. The dots run as 3-pass split TF32 (tf32x3, fp32-level
error) with fp32 accumulation.

Registered as the triton_op ``torch.ops.trtllm.flux3_klein_add_norm_swish_conv_out_3x3``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

# Tile: BW input columns (BW-2 output columns) x R output rows per program, BC input
# channels per dot step. Swept on SM120 at NHWC 1x128x1024x1024 (the fused-residual
# prologue path the decoder runs).
BW = 64
R = 16
BC = 64
_NUM_WARPS = 4
_NUM_STAGES = 2


@triton.jit
def _add_norm_swish_conv3x3_to3_kernel(
    x_ptr,
    r_ptr,
    rbias_ptr,
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
):
    QW: tl.constexpr = BW - 2
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    n = tl.program_id(2).to(tl.int64)
    q = tl.arange(0, BW)
    iw = pid_w * QW - 1 + q  # input columns
    in_col_ok = (iw >= 0) & (iw < W)
    ocol = pid_w * QW + q  # output columns (valid for q < QW)
    out_col_ok = (q < QW) & (ocol < W)
    j = tl.arange(0, 32)
    kw_j = j // 9
    rem = j - 9 * kw_j
    kh_j = rem // 3
    o_j = rem - 3 * kh_j
    jv = j < 27
    gidx = tl.minimum(q[:, None] + kw_j[None, :], BW - 1)
    xb = x_ptr + n * sxn
    rb = r_ptr + n * sxn
    yb = y_ptr + n * syn
    b0 = tl.load(b_ptr)
    b1 = tl.load(b_ptr + 1)
    b2 = tl.load(b_ptr + 2)
    r0 = pid_h * R
    s1 = tl.zeros([BW, 32], dtype=tl.float32)  # aligned taps of input row ih-1
    s2 = tl.zeros([BW, 32], dtype=tl.float32)  # aligned taps of input row ih-2
    for i in range(R + 2):
        ih = r0 - 1 + i
        m = in_col_ok & (ih >= 0) & (ih < H)
        p = tl.zeros([BW, 32], dtype=tl.float32)
        for c0 in tl.static_range(0, C, BC):
            cc = c0 + tl.arange(0, BC)
            off = ih.to(tl.int64) * sxh + iw[:, None].to(tl.int64) * sxw + cc[None, :] * sxc
            xv = tl.load(xb + off, mask=m[:, None], other=0.0)
            # x + (r + r_bias_c), each sum rounded to x's dtype as the unfused bias add
            # and residual add store them.
            rv = tl.load(rb + off, mask=m[:, None], other=0.0)
            rbv = tl.load(rbias_ptr + cc)
            rv = (rv.to(tl.float32) + rbv[None, :].to(tl.float32)).to(x_ptr.dtype.element_ty)
            xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(x_ptr.dtype.element_ty)
            xv = xv.to(tl.float32)
            sc = tl.load(scale_ptr + n * C + cc)
            sh = tl.load(shift_ptr + n * C + cc)
            xv = xv * sc[None, :] + sh[None, :]
            xv = xv * tl.sigmoid(xv)
            xv = tl.where(m[:, None], xv, 0.0)
            wt = tl.load(
                w_ptr
                + o_j[None, :] * swo
                + cc[:, None] * swc
                + kh_j[None, :] * swh
                + kw_j[None, :] * sww,
                mask=jv[None, :],
                other=0.0,
            )
            p = tl.dot(xv, wt, p, input_precision="tf32x3")
        g = tl.gather(p, gidx, 0)
        t = (
            tl.where(kh_j[None, :] == 2, g, 0.0)
            + tl.where(kh_j[None, :] == 1, s1, 0.0)
            + tl.where(kh_j[None, :] == 0, s2, 0.0)
        )
        oh = ih - 1
        smask = out_col_ok & (i >= 2) & (oh < H)
        ybase = yb + oh.to(tl.int64) * syh + ocol * syw
        out0 = tl.sum(tl.where((o_j[None, :] == 0) & jv[None, :], t, 0.0), axis=1) + b0
        out1 = tl.sum(tl.where((o_j[None, :] == 1) & jv[None, :], t, 0.0), axis=1) + b1
        out2 = tl.sum(tl.where((o_j[None, :] == 2) & jv[None, :], t, 0.0), axis=1) + b2
        tl.store(ybase, out0, mask=smask)
        tl.store(ybase + syc, out1, mask=smask)
        tl.store(ybase + 2 * syc, out2, mask=smask)
        s2 = s1
        s1 = g


@triton_op("trtllm::flux3_klein_add_norm_swish_conv_out_3x3", mutates_args={})
def add_norm_swish_conv_out_3x3_op(
    x: Tensor,
    r: Tensor,
    r_bias: Tensor,
    scale: Tensor,
    shift: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """conv2d(swish((x + (r + r_bias_c)) * scale_c + shift_c), weight [3, C, 3, 3], bias,
    padding=1) in fp32 -> fp32 [N, 3, H, W] (channels_last when x is)."""
    n, c, h, w = x.shape
    fmt = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous()
        else torch.contiguous_format
    )
    y = torch.empty((n, 3, h, w), device=x.device, dtype=torch.float32, memory_format=fmt)
    grid = (triton.cdiv(w, BW - 2), triton.cdiv(h, R), n)
    wrap_triton(_add_norm_swish_conv3x3_to3_kernel)[grid](
        x,
        r,
        r_bias,
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
        BW=BW,
        R=R,
        BC=BC,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return y
