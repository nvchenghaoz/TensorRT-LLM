# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bf16 attention projections (qkv, o_proj), routed by the step's row count.

`bf16_proj_gemm(x, weight, route)` computes `x @ weight^T` (bf16 out, fp32 accumulation) on
one of three kernels, chosen by `x`'s rows from the per-shape `route` (`BF16_PROJ_ROUTES`,
keyed `(in_features, out_features)`):

1. the target's SM120 sequential-K GEMM (`csrc/sm120Bf16SeqKGemm*`) with the route's tile
   config, at the row counts where it was faster (about 300 rows: non-power-of-2 tiles that fill
   one wave);
2. `trtllm::cublas_mm`, whose SM120 table (`sm120_bf16_range_algo_list` in
   `cpp/tensorrt_llm/thop/cublasScaledMMLut.h`) pins a measured cuBLASLt algorithm for these
   shapes at the listed row ranges;
3. `F.linear` (cuBLASLt's heuristic) everywhere else.

All three accumulate each output in fp32 over ascending K with no split-K and round once, so
they are bitwise equal; the route changes speed only. The ranges were generated together with
the cuBLASLt table by a same-GPU sweep.
"""

from __future__ import annotations

import bisect
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ... import native_ops


class _Ranges:
    """The union of sorted, disjoint inclusive row ranges (`rows in ranges`)."""

    __slots__ = ("_starts", "_ends")

    def __init__(self, ranges):
        ranges = sorted(ranges)
        self._starts = tuple(lo for lo, _ in ranges)
        self._ends = tuple(hi for _, hi in ranges)

    def __contains__(self, rows: int) -> bool:
        i = bisect.bisect_right(self._starts, rows) - 1
        return i >= 0 and rows <= self._ends[i]


class Bf16ProjRoute(NamedTuple):
    seqk_rows: _Ranges
    seqk_config: int
    cublas_rows: _Ranges


# Rows where `trtllm::cublas_mm`'s SM120 table pins an algorithm, per (in, out) projection
# shape: sliding qkv, full-attention [q | k] (K = V), sliding and full-attention o_proj.
_CUBLAS_LUT_RANGES = {
    # sliding qkv
    (5376, 16384): (
        (300, 300), (600, 600), (900, 900), (1000, 1000), (1200, 1200), (1345, 1600), (1729, 1920), (1985, 2000),
        (2241, 2400), (2433, 2496), (2625, 2700), (2753, 2880), (3000, 3072), (3201, 3264), (3393, 3456),
        (3777, 3840), (4161, 4224), (4481, 4608), (4929, 4992), (5313, 5376), (5697, 5760), (6081, 6144),
        (6300, 6528), (6721, 6900), (6913, 7000), (7041, 7808), (7873, 8000), (8065, 8192),
    ),
    # global qkv, V duplicated from K
    (5376, 20480): (
        (900, 900), (1000, 1000), (1200, 1200), (1280, 1536), (1800, 1920), (2100, 2304), (2369, 2496),
        (2625, 2688), (3000, 3136), (3265, 3328), (3600, 3648), (3900, 4032), (4161, 4224), (4501, 4608),
        (4801, 4992), (5000, 5376), (5401, 5568), (5697, 5952), (6145, 6208), (6273, 6400), (6465, 6528),
        (6593, 6720), (6849, 6912), (7001, 7168), (7201, 7424), (7489, 7808), (7873, 8064), (8101, 8192),
    ),
    # global qkv, k_eq_v V-dedup ([q | k])
    (5376, 18432): (
        (300, 300), (600, 600), (900, 900), (1000, 1000), (1200, 1200), (1280, 1408), (1473, 1600), (1793, 1856),
        (1985, 2112),
        (2369, 2496), (2625, 2700), (3000, 3072), (3600, 3712), (4000, 4032), (4289, 4416), (4673, 4800),
        (5001, 5184), (5313, 5568), (5697, 5952), (6081, 6400), (6465, 6912), (6977, 7104), (7201, 7296),
        (7361, 7500), (7617, 7680), (7745, 7936), (8001, 8192),
    ),
    # sliding o_proj
    (8192, 5376): (
        (900, 900), (1200, 1200), (1500, 1500), (1793, 1856), (2000, 2000), (2100, 2496), (2625, 3968),
        (4001, 4032), (4225, 4992), (5001, 5056),
        (5121, 5184), (5313, 5376), (5441, 5568), (5697, 6336), (6849, 7500), (7553, 7680), (8000, 8192),
    ),
    # global o_proj
    (16384, 5376): (
        (900, 900), (1000, 1000), (1200, 1200), (1280, 1280), (1500, 1500), (1793, 1856), (2000, 2000),
        (2100, 2496), (2625, 3840), (3900, 3904),
        (4200, 5120), (5441, 5568), (5697, 6400), (6657, 6784), (6849, 7680), (8000, 8192),
    ),
}


# The sequential-K GEMM: (row ranges, tile config). M=300 stays on its cuBLASLt pin, which ties
# the kernel alone and is faster in a 60-layer graph chain.
_SEQK_RANGES = {
    # sliding qkv: 160 tokens x 176 weights, 94 x 2 = 188 CTAs (one wave); not M=288.
    (5376, 16384): (((289, 299), (301, 320)), 0),
    # full-attention [q | k]: 160 tokens x 208 weights, 89 x 2 = 178 CTAs.
    (5376, 18432): (((288, 299), (301, 320)), 1),
}

BF16_PROJ_ROUTES = {
    shape: Bf16ProjRoute(
        _Ranges(_SEQK_RANGES[shape][0]) if shape in _SEQK_RANGES else _Ranges(()),
        _SEQK_RANGES[shape][1] if shape in _SEQK_RANGES else -1,
        _Ranges(ranges),
    )
    for shape, ranges in _CUBLAS_LUT_RANGES.items()
}


class Bf16ProjGemm(OpWrapper):
    """`x [M, K] @ weight[N, K]^T -> [M, N]` bf16."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = tuple(
        Cell(why=f"{shape} at {m} rows", spec=dict(k=shape[0], n=shape[1], m=m))
        for shape in ((5376, 16384), (5376, 18432), (8192, 5376), (16384, 5376))
        for m in (1, 300, 310, 1000, 2000, 4000)
    )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, route: Optional[Bf16ProjRoute]):
        m = x.shape[0]
        if route is not None:
            if m in route.seqk_rows:
                return native_ops.sm120_bf16_seqk_gemm(x, weight, route.seqk_config)
            if m in route.cublas_rows:
                return torch.ops.trtllm.cublas_mm(x, weight.t(), None, out_dtype=None)
        return F.linear(x, weight)

    def reference(self, x, weight, route):
        """fp32 accumulation, one rounding (`route` is ignored)."""
        return (x.float() @ weight.float().t()).to(x.dtype)


bf16_proj_gemm = Bf16ProjGemm()
