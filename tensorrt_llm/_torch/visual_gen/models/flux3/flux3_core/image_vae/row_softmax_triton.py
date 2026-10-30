# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent single-DRAM-pass row softmax for the FLUX.3 VAE mid-block attention.

At 2048^2 the mid block attends over T = 65536 tokens with d = 512, and _attention_tc
materializes the fp32 score matrix S [T, T] (17.2 GB). Inductor's fused softmax reads S
from DRAM 2x (decoder, e4m3 P) or 3x (encoder, bf16 P), once per looped-reduction pass.
Here a persistent grid (a few programs per SM) walks the rows; each program makes all of
its passes over one 256 KB row back to back, so with ~188-376 rows in flight (48-96 MB,
under the 128 MB L2) every pass after the first hits L2.

Math is element-for-element the same as main's Inductor kernels (same expressions, same
libdevice exp, same NaN/inf branch of the scaled-softmax decomposition):
  finite rows: t = (s - max s) * scale, otherwise t = s * scale - max(s * scale)
  e = exp(t)
  e4m3 path: pq = (e * 448).to(e4m3), scale_a = 1 / (sum(e) * 448)        [decoder]
  bf16 path: p = (e / sum(e)).to(bf16)                                      [encoder]
Only the order of the fp32 row sum differs (output class "reordered"): pq does not
depend on it; scale_a and p move by the summation-order rounding (~1e-6 relative).

Both entry points are torch.library triton_ops, so torch.compile traces through them
without a graph break.
"""

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice

# Process-wide switch (A/B experiments); TLLM_FLUX3_VAE_ROW_SOFTMAX=0 keeps main's path.
ENABLED = os.environ.get("TLLM_FLUX3_VAE_ROW_SOFTMAX", "1") != "0"

# Launch configuration, swept at T = 65536 on RTX PRO 6000 SM120 (branch 46 job 564, cold
# L2): e4m3 BLOCK 4096 / 8 warps 14.49 ms (main's Inductor kernel 23.18 ms; DRAM floor of
# 17.2 GB read + 4.3 GB write ~14.3 ms); bf16 BLOCK 8192 / 8 warps 18.41 ms (main 38.59 ms).
# One program per SM: 2-3 per SM put 96-144 MB of rows in flight and lose L2 hits.
CFG = {
    "BLOCK": 4096,
    "num_warps": 8,
    "ctas_per_sm": 1,
    "reverse": True,
    "num_stages": 1,
    "store_cs": True,
}
CFG_BF16 = dict(CFG, BLOCK=8192)

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
    """max(s), max(s * scale) and the any-non-finite flag of one row (main's pass 1)."""
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
def _softmax_rows_e4m3_kernel(
    S,
    PQ,
    SA,
    n_rows,
    scale,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    REVERSE: tl.constexpr,
    STORE_CS: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for r in range(pid, n_rows, nprog):
        if REVERSE:
            row = n_rows - 1 - r
        else:
            row = r
        src = S + row.to(tl.int64) * N
        dst = PQ + row.to(tl.int64) * N
        mx1, mx2, nonfinite = _row_stats(src, scale, offs, N, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.range(0, N, BLOCK):
            x = tl.load(src + j + offs, eviction_policy="evict_first")
            e = libdevice.exp(_exp_arg(x, mx1, mx2, scale, nonfinite))
            acc = acc + e
            q = (e * 448.0).to(tl.float8e4nv)
            if STORE_CS:
                tl.store(dst + j + offs, q, cache_modifier=".cs")
            else:
                tl.store(dst + j + offs, q)
        total = tl.sum(acc, 0)
        tl.store(SA + row, 1.0 / (total * 448.0))


@triton.jit
def _softmax_rows_bf16_kernel(
    S,
    P,
    n_rows,
    scale,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    REVERSE: tl.constexpr,
    STORE_CS: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for r in range(pid, n_rows, nprog):
        if REVERSE:
            row = n_rows - 1 - r
        else:
            row = r
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
            if STORE_CS:
                tl.store(dst + j + offs, p, cache_modifier=".cs")
            else:
                tl.store(dst + j + offs, p)


def can_use(s: Tensor, e4m3: bool) -> bool:
    """s: fp32 CUDA scores [..., T, N], contiguous, N a multiple of the path's block."""
    cfg = CFG if e4m3 else CFG_BF16
    return (
        ENABLED
        and s.is_cuda
        and s.dtype == torch.float32
        and s.dim() >= 2
        and s.is_contiguous()
        and s.shape[-1] % cfg["BLOCK"] == 0
        and s.numel() > 0
    )


def _grid(s: Tensor, n_rows: int, cfg: dict) -> tuple:
    return (max(1, min(n_rows, _sm_count(s.device) * cfg["ctas_per_sm"])),)


@triton_op("flux3_vae::row_softmax_e4m3", mutates_args={})
def row_softmax_e4m3(s: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    """s [T, N] fp32 -> (pq [T, N] e4m3 = (exp(z - max z) * 448), scale_a [T, 1] fp32).

    z = s * scale; scale_a = 1 / (448 * sum_row exp(z - max z)) (see module docstring).
    """
    n_rows, n = s.shape
    pq = torch.empty((n_rows, n), device=s.device, dtype=torch.float8_e4m3fn)
    sa = torch.empty((n_rows, 1), device=s.device, dtype=torch.float32)
    wrap_triton(_softmax_rows_e4m3_kernel)[_grid(s, n_rows, CFG)](
        s,
        pq,
        sa,
        n_rows,
        scale,
        N=n,
        BLOCK=CFG["BLOCK"],
        REVERSE=CFG["reverse"],
        STORE_CS=CFG["store_cs"],
        num_warps=CFG["num_warps"],
        num_stages=CFG["num_stages"],
        # main's decoder graph compiles with emulate_precision_casts (no FMA contraction)
        enable_fp_fusion=False,
    )
    return pq, sa


@triton_op("flux3_vae::row_softmax_bf16", mutates_args={})
def row_softmax_bf16(s: Tensor, scale: float) -> Tensor:
    """softmax(s * scale, dim=-1).to(bf16) for fp32 s [..., N] (Inductor's decomposition)."""
    n = s.shape[-1]
    n_rows = s.numel() // n
    p = torch.empty(s.shape, device=s.device, dtype=torch.bfloat16)
    cfg = CFG_BF16
    wrap_triton(_softmax_rows_bf16_kernel)[_grid(s, n_rows, cfg)](
        s,
        p,
        n_rows,
        scale,
        N=n,
        BLOCK=cfg["BLOCK"],
        REVERSE=cfg["reverse"],
        STORE_CS=cfg["store_cs"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return p
