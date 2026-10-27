# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-group fused GroupNorm (+SiLU) for the FLUX.3 image VAE.

aten's GroupNorm launches one CTA per (sample, group) for its statistics (32 CTAs at
batch 1) and then runs separate elementwise passes for the affine apply, sigmoid and
mul of ``x * sigmoid(x)``. This module computes the statistics with G x S CTAs
(per-lane Welford, combined with Chan's formula over equal counts) and applies the
normalization, the affine transform and the optional SiLU in a single pass.

The elementwise math mirrors aten exactly: a = rstd * gamma, b = -a * mean + beta,
y = a * x + b (FMA), sigmoid = 1 / (1 + expf(-y)) with an IEEE divide, rounding to
the storage dtype where eager materializes a tensor. Only the summation order of the
statistics differs from aten.
"""

import os

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover - older Triton
    from triton.language.extra import libdevice

_STATS_BLOCK = 4096
_STATS_WARPS = 8
_APPLY_BLOCK = 2048
_APPLY_WARPS = 4
_MAX_SPLITS = 64

_ENABLED = os.environ.get("TRTLLM_FLUX3_VAE_FUSED_GN", "1") != "0"


def set_enabled(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def is_enabled() -> bool:
    return _ENABLED


@triton.jit
def _gn_stats_kernel(
    x_ptr,
    mean_ptr,
    m2_ptr,
    group_numel,
    chunk,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    gid = tl.program_id(0)
    sid = tl.program_id(1)
    base = gid.to(tl.int64) * group_numel + sid.to(tl.int64) * chunk
    offs = tl.arange(0, BLOCK)
    mean = tl.zeros([BLOCK], dtype=tl.float32)
    m2 = tl.zeros([BLOCK], dtype=tl.float32)
    n_iter = chunk // BLOCK
    for i in range(n_iter):
        x = tl.load(x_ptr + base + i * BLOCK + offs).to(tl.float32)
        inv_n = 1.0 / (i + 1).to(tl.float32)
        delta = x - mean
        mean = mean + delta * inv_n
        m2 = m2 + delta * (x - mean)
    # Every lane holds n_iter samples: combine equal-count partials.
    cmean = tl.sum(mean, axis=0) / BLOCK
    d = mean - cmean
    cm2 = tl.sum(m2, axis=0) + n_iter.to(tl.float32) * tl.sum(d * d, axis=0)
    tl.store(mean_ptr + gid * S + sid, cmean)
    tl.store(m2_ptr + gid * S + sid, cm2)


@triton.jit
def _gn_apply_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    m2_ptr,
    C,
    HW,
    cpg,
    chunk_f,
    group_numel_f,
    eps,
    S: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)  # n * C + c
    tile = tl.program_id(1)
    c = row % C
    gid = row // cpg  # n * G + c // cpg, since C == G * cpg

    so = tl.arange(0, S)
    pm = tl.load(mean_ptr + gid * S + so)
    pm2 = tl.load(m2_ptr + gid * S + so)
    mean = tl.sum(pm, axis=0) / S
    d = pm - mean
    m2 = tl.sum(pm2, axis=0) + chunk_f * tl.sum(d * d, axis=0)
    var = tl.maximum(m2 / group_numel_f, 0.0)
    rstd = libdevice.rsqrt(var + eps)

    gamma = tl.load(w_ptr + c).to(tl.float32)
    beta = tl.load(b_ptr + c).to(tl.float32)
    a = rstd * gamma
    b = tl.fma(-a, mean, beta)

    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    ptr_off = row.to(tl.int64) * HW + offs
    x = tl.load(x_ptr + ptr_off, mask=mask, other=0.0)
    out_ty = x.dtype
    y = tl.fma(a, x.to(tl.float32), b)
    if APPLY_SILU:
        y = y.to(out_ty).to(tl.float32)
        e = libdevice.exp(-y)
        s = libdevice.div_rn(1.0, 1.0 + e)
        s = s.to(out_ty).to(tl.float32)
        y = y * s
    tl.store(y_ptr + ptr_off, y.to(out_ty), mask=mask)


def _num_splits(n_groups: int, group_numel: int) -> int:
    """Largest power-of-two split count S <= _MAX_SPLITS dividing the group into
    whole _STATS_BLOCK tiles, aiming at roughly 2048 CTAs in total."""
    n_tiles = group_numel // _STATS_BLOCK
    target = max(1, min(_MAX_SPLITS, 2048 // max(1, n_groups)))
    s = 1
    while s * 2 <= target and n_tiles % (s * 2) == 0:
        s *= 2
    return s


def can_use_fused(x: torch.Tensor, gn: torch.nn.GroupNorm) -> bool:
    if not _ENABLED:
        return False
    if not x.is_cuda or x.dim() < 2:
        return False
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if not gn.affine or gn.weight.dtype != x.dtype or gn.bias.dtype != x.dtype:
        return False
    if torch.is_grad_enabled() and (x.requires_grad or gn.weight.requires_grad):
        return False
    if not x.is_contiguous():
        return False
    C = x.shape[1]
    if C % gn.num_groups != 0:
        return False
    group_numel = x.numel() // (x.shape[0] * gn.num_groups)
    if group_numel == 0 or group_numel % _STATS_BLOCK != 0:
        return False
    return True


def fused_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    N, C = x.shape[0], x.shape[1]
    HW = x.numel() // (N * C)
    G = num_groups
    cpg = C // G
    group_numel = cpg * HW
    n_groups = N * G
    S = _num_splits(n_groups, group_numel)
    chunk = group_numel // S

    part = torch.empty((2, n_groups, S), device=x.device, dtype=torch.float32)
    part_mean, part_m2 = part[0], part[1]
    _gn_stats_kernel[(n_groups, S)](
        x,
        part_mean,
        part_m2,
        group_numel,
        chunk,
        S=S,
        BLOCK=_STATS_BLOCK,
        num_warps=_STATS_WARPS,
    )
    y = torch.empty_like(x)
    _gn_apply_kernel[(N * C, triton.cdiv(HW, _APPLY_BLOCK))](
        x,
        y,
        weight,
        bias,
        part_mean,
        part_m2,
        C,
        HW,
        cpg,
        float(chunk),
        float(group_numel),
        float(eps),
        S=S,
        APPLY_SILU=apply_silu,
        BLOCK=_APPLY_BLOCK,
        num_warps=_APPLY_WARPS,
    )
    return y
