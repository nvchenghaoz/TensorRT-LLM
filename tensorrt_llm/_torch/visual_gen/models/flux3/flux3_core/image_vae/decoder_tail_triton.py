# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused tail of the FLUX.3 image VAE decoder.

conv_out (3x3, C -> 3, padding 1) + bias as one Triton kernel, optionally with the last
ResnetBlock's conv2 bias add + residual add and norm_out's GroupNorm apply + swish as its
prologue. It replaces, in the compiled bf16 decoder: norm_out's apply+swish kernel that
writes an fp32 copy of the 128-channel activation, cuDNN's convertTensor of that copy, a
TF32 implicit GEMM padded from N=3 to a full tile, and the bias kernel. The math is the
same; the convolution accumulates in fp32 with 3-pass split-TF32 products (tf32x3,
fp32-level error, ~1000x below the single-pass TF32 of cuDNN's path) and the prologue
stays in fp32 (no bf16 rounding of the normalized activation), so the output class is
"reordered".

Main kernel (_conv3x3_to3_gather_kernel): a program owns BW-2 output columns x R output
rows and walks the R+2 input rows once. Per input row it loads BW input pixels (with a
1-pixel halo) x C channels once, applies the prologue once, and forms one dot
P = X[BW, C] @ Wall[C, 32] whose 27 live columns are (kw, kh, o). tl.gather along the
pixel axis aligns each kw tap on its output pixel; output row ih-1 is then the sum of the
kh=2 taps of row ih, kh=1 of row ih-1 and kh=0 of row ih-2 (two rolling partials), plus
bias. Inputs and weights are addressed by strides (NHWC or NCHW). _conv3x3_to3_kernel is
the simpler variant (one dot per kw shift, 3x prologue work), kept for A/B.

Prologue: x (+ (r + r_bias_c), each sum rounded to x's dtype as the unfused adds store
them), then x * scale_c + shift_c (GroupNorm from per-channel statistics) and swish in
fp32; zero padding applies after the activation.

All entry points are torch.library triton_ops, so torch.compile traces through them
without a graph break.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

# Tile: BW input columns (BW-2 output columns for the gather kernel) x R output rows per
# program; BC input channels per dot step. Swept at the served shape (NHWC 1x128x1024x1024,
# RTX PRO 6000 SM120, job 1377): best for the served fused-residual prologue path.
_BW = 64
_R = 16
_BC = 64
_NUM_WARPS = 4
_NUM_STAGES = 2
# tl.dot input precision: "ieee" = fp32 FMA; "tf32x3" = 3-pass split TF32 on tensor
# cores (fp32-level accuracy, far below the single-pass TF32 error of cuDNN's path).
_PRECISION = "tf32x3"
# True: _conv3x3_to3_gather_kernel (each input element loaded/transformed once).
_GATHER = True

# Process-wide switches (A/B experiments, debugging). ENABLED gates both entry points;
# FUSE_NORM_PROLOGUE additionally folds norm_out's apply + swish into the kernel.
ENABLED = True
FUSE_NORM_PROLOGUE = True


@triton.jit
def _conv3x3_to3_kernel(
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
    PROLOGUE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    RBIAS: tl.constexpr,
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
    rb = r_ptr + n * sxn
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
        p = tl.zeros([BW, 16], dtype=tl.float32)
        for kw in tl.static_range(3):
            iw = cols + (kw - 1)
            m = row_ok & (iw >= 0) & (iw < W)
            for c0 in tl.static_range(0, C, BC):
                cc = c0 + tl.arange(0, BC)
                off = ih.to(tl.int64) * sxh + iw[:, None].to(tl.int64) * sxw + cc[None, :] * sxc
                xv = tl.load(xb + off, mask=m[:, None], other=0.0)
                if RESIDUAL:
                    # x + (r + r_bias_c), each sum rounded to x's dtype as the unfused
                    # bias add and residual add store them.
                    rv = tl.load(rb + off, mask=m[:, None], other=0.0)
                    if RBIAS:
                        rbv = tl.load(rbias_ptr + cc)
                        rv = (rv.to(tl.float32) + rbv[None, :].to(tl.float32)).to(x_ptr.dtype.element_ty)
                    xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(x_ptr.dtype.element_ty)
                xv = xv.to(tl.float32)
                if PROLOGUE:
                    sc = tl.load(scale_ptr + n * C + cc)
                    sh = tl.load(shift_ptr + n * C + cc)
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


@triton.jit
def _conv3x3_to3_gather_kernel(
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
    PROLOGUE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    RBIAS: tl.constexpr,
    PREC: tl.constexpr,
):
    """Each input element is loaded and transformed once per input row.

    The program loads BW input columns [q0-1, q0+BW-1) and produces BW-2 output
    columns [q0, q0+BW-2). Per input row: P = X[BW, C] @ Wall[C, 32] with live columns
    j = kw*9 + kh*3 + o (27 of 32); G[q, j] = P[q + kw_j, j] (tl.gather along pixels)
    aligns every kw tap on its output pixel; rows combine as in _conv3x3_to3_kernel.
    """
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
            if RESIDUAL:
                # x + (r + r_bias_c), each sum rounded to x's dtype as the unfused
                # bias add and residual add store them.
                rv = tl.load(rb + off, mask=m[:, None], other=0.0)
                if RBIAS:
                    rbv = tl.load(rbias_ptr + cc)
                    rv = (rv.to(tl.float32) + rbv[None, :].to(tl.float32)).to(x_ptr.dtype.element_ty)
                xv = (xv.to(tl.float32) + rv.to(tl.float32)).to(x_ptr.dtype.element_ty)
            xv = xv.to(tl.float32)
            if PROLOGUE:
                sc = tl.load(scale_ptr + n * C + cc)
                sh = tl.load(shift_ptr + n * C + cc)
                xv = xv * sc[None, :] + sh[None, :]
                xv = xv * tl.sigmoid(xv)
                xv = tl.where(m[:, None], xv, 0.0)
            wt = tl.load(
                w_ptr + o_j[None, :] * swo + cc[:, None] * swc + kh_j[None, :] * swh + kw_j[None, :] * sww,
                mask=jv[None, :],
                other=0.0,
            )
            p = tl.dot(xv, wt, p, input_precision=PREC)
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


def _launch(
    x: Tensor, r, r_bias, scale, shift, weight: Tensor, bias: Tensor, prologue: bool
) -> Tensor:
    n, c, h, w = x.shape
    y = _out_like(x)
    grid = (triton.cdiv(w, _BW), triton.cdiv(h, _R), n)
    if scale is None:
        scale, shift = bias, bias  # unused (PROLOGUE=False)
    residual = r is not None
    if not residual:
        r = x  # unused (RESIDUAL=False)
    rbias = r_bias is not None
    if not rbias:
        r_bias = bias  # unused (RBIAS=False)
    kernel = _conv3x3_to3_kernel
    if _GATHER:
        kernel = _conv3x3_to3_gather_kernel
        grid = (triton.cdiv(w, _BW - 2), triton.cdiv(h, _R), n)
    wrap_triton(kernel)[grid](
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
        BW=_BW,
        R=_R,
        BC=_BC,
        PROLOGUE=prologue,
        RESIDUAL=residual,
        RBIAS=rbias,
        PREC=_PRECISION,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return y


@triton_op("flux3_vae::conv_out_3x3", mutates_args={})
def conv_out_3x3(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """conv2d(x, weight [3, C, 3, 3], bias, padding=1), fp32 accumulation (x any strides)."""
    return _launch(x, None, None, None, None, weight, bias, False)


@triton_op("flux3_vae::norm_swish_conv_out_3x3", mutates_args={})
def norm_swish_conv_out_3x3(
    x: Tensor, scale: Tensor, shift: Tensor, weight: Tensor, bias: Tensor
) -> Tensor:
    """conv2d(swish(x * scale_c + shift_c), weight, bias, padding=1), fp32 math.

    x is the pre-norm activation (any float dtype, any strides); scale/shift are fp32
    [N, C] contiguous per-sample, per-channel GroupNorm affine terms.
    """
    return _launch(x, None, None, scale, shift, weight, bias, True)


@triton_op("flux3_vae::add_norm_swish_conv_out_3x3", mutates_args={})
def add_norm_swish_conv_out_3x3(
    x: Tensor,
    r: Tensor,
    r_bias: Optional[Tensor],
    scale: Tensor,
    shift: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """norm_swish_conv_out_3x3(x + (r + r_bias_c), ...) without materializing the sum.

    r is the residual branch before its conv bias r_bias ([C] or None). Each sum is
    rounded to x's dtype in-kernel (as the unfused bias and residual adds store them).
    x and r must have the same dtype, shape and strides.
    """
    return _launch(x, r, r_bias, scale, shift, weight, bias, True)


def group_norm_channel_affine(
    x: Tensor, num_groups: int, weight: Tensor, bias: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """fp32 [N, C] (scale, shift) with GroupNorm(x) == x * scale + shift per channel.

    Statistics as in image_vae.group_norm_channel_stats: per-channel var_mean over
    (H, W) in fp32 (fp64 for fp32 x), then the exact equal-count group combine.
    scale_c = weight_c * rstd_g(c), shift_c = bias_c - mean_g(c) * scale_c.
    """
    n, c = x.shape[:2]
    g = num_groups
    acc = torch.float64 if x.dtype == torch.float32 else torch.float32
    red = tuple(range(2, x.dim()))
    var_c, mean_c = torch.var_mean(x.to(acc), dim=red, correction=0)
    mean_c = mean_c.view(n, g, c // g)
    var_c = var_c.view(n, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    rstd_g = torch.rsqrt(var_g + eps)
    mean = mean_g.expand(n, g, c // g).reshape(n, c).float()
    rstd = rstd_g.expand(n, g, c // g).reshape(n, c).float()
    scale = rstd * weight.float().view(1, c)
    shift = bias.float().view(1, c) - mean * scale
    return scale.contiguous(), shift.contiguous()


def can_use_fused_tail(x: Tensor, norm: torch.nn.Module, conv: torch.nn.Module) -> bool:
    """True when conv(swish(norm(x))) can run as norm_swish_conv_out_3x3."""
    return (
        can_use_conv_out_kernel(x, conv)
        and isinstance(norm, torch.nn.GroupNorm)
        and norm.affine
        and norm.num_channels == x.shape[1]
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )
