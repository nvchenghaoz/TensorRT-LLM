# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent row softmax over fp32 scores with a bf16 output, for the VAE encoder's
mid-block attention.

At a 2048^2 reference image the encoder's mid block attends over T = 65536 tokens and
S [T, T] fp32 is 17.2 GB. Inductor's fused softmax reads S from DRAM once per pass (3x);
here one program per SM walks the rows (last row first) and makes all three passes over
one 256 KB row back to back, so with ~188 rows in flight (48 MB, under the 128 MB L2)
the passes after the first hit L2.

Element for element the expressions of Inductor's softmax decomposition (same libdevice
exp, same non-finite branch):
  finite rows: t = (s - max s) * scale, otherwise t = s * scale - max(s * scale)
  p = (exp(t) / sum exp(t)).to(bf16)
Only the order of the fp32 row sum differs.

Registered as the triton_op ``torch.ops.trtllm.flux3_klein_row_softmax_bf16``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice

# Swept at T = 65536 on SM120 (cold L2): BLOCK 8192 / 8 warps / one program per SM
# 18.41 ms (Inductor's softmax 38.59 ms). 2-3 programs per SM put 96-144 MB of rows in
# flight and lose the L2 hits.
BLOCK = 8192
_NUM_WARPS = 8
_NUM_STAGES = 1

_SM_COUNT = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SM_COUNT.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT[idx] = n
    return n


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, tl.PropagateNan.ALL)


@triton.jit
def _row_stats(src, scale, offs, N: tl.constexpr, BLOCK: tl.constexpr):
    """max(s), max(s * scale) and the any-non-finite flag of one row."""
    m1 = tl.full([BLOCK], float("-inf"), tl.float32)
    m2 = tl.full([BLOCK], float("-inf"), tl.float32)
    bad = tl.zeros([BLOCK], dtype=tl.int1)
    for j in tl.range(0, N, BLOCK):
        x = tl.load(src + j + offs, eviction_policy="evict_last")
        z = x * scale
        fin = (z == z) & (tl.abs(z) != float("inf"))
        bad = bad | (fin == 0)
        m1 = tl.maximum(m1, x * 1.0, tl.PropagateNan.ALL)
        m2 = tl.maximum(m2, z, tl.PropagateNan.ALL)
    nonfinite = tl.max(bad.to(tl.int32), 0) != 0
    return tl.reduce(m1, 0, _max_nan), tl.reduce(m2, 0, _max_nan), nonfinite


@triton.jit
def _exp_arg(x, mx1, mx2, scale, nonfinite):
    return tl.where(nonfinite == 0, (x * 1.0 - mx1) * scale, x * scale - mx2)


@triton.jit
def _softmax_rows_bf16_kernel(
    S,
    P,
    n_rows,
    scale,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Inductor passes Python floats as fp64, a direct launch as fp32: the multiplier is
    # the fp32 rounding of scale either way.
    scale = scale.to(tl.float32)
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for r in range(pid, n_rows, nprog):
        row = n_rows - 1 - r
        src = S + row.to(tl.int64) * N
        dst = P + row.to(tl.int64) * N
        mx1, mx2, nonfinite = _row_stats(src, scale, offs, N, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.range(0, N, BLOCK):
            x = tl.load(src + j + offs, eviction_policy="evict_last")
            acc = acc + libdevice.exp(_exp_arg(x, mx1, mx2, scale, nonfinite))
        total = tl.sum(acc, 0)
        for j in tl.range(0, N, BLOCK):
            x = tl.load(src + j + offs, eviction_policy="evict_first")
            e = libdevice.exp(_exp_arg(x, mx1, mx2, scale, nonfinite))
            p = (e / total).to(tl.bfloat16)
            tl.store(dst + j + offs, p, cache_modifier=".cs")


def supported(s: Tensor) -> bool:
    """fp32 CUDA scores [..., N], contiguous, N a multiple of BLOCK (loads are unmasked)."""
    return (
        s.is_cuda
        and s.dtype == torch.float32
        and s.dim() >= 2
        and s.is_contiguous()
        and s.shape[-1] % BLOCK == 0
        and s.numel() > 0
    )


@triton_op("trtllm::flux3_klein_row_softmax_bf16", mutates_args={})
def row_softmax_bf16_op(s: Tensor, scale: float) -> Tensor:
    """softmax(s * scale, dim=-1).to(bf16) for fp32 s [..., N]."""
    n = s.shape[-1]
    n_rows = s.numel() // n
    p = torch.empty(s.shape, device=s.device, dtype=torch.bfloat16)
    grid = (max(1, min(n_rows, _sm_count(s.device))),)
    wrap_triton(_softmax_rows_bf16_kernel)[grid](
        s,
        p,
        n_rows,
        scale,
        N=n,
        BLOCK=BLOCK,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return p
