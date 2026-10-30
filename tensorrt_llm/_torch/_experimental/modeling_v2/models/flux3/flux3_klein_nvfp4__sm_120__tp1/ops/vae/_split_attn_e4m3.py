# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-softmax e4m3 single-head attention of the VAE decoder's mid block (b == 1).

Two Triton kernels:
  A  _qk_tilemax_kernel: S = Q^T K (bf16 in, fp32 accumulate, fp32 out) and, per (row,
     256-column tile), the tile max of S (a small [T, T / 256] fp32 buffer).
  B  _pv_prologue_kernel: m = max over the row's tile maxes (== the row max, bitwise),
     then per K block: e = exp((s - m) * scale), l += e, pq = e4m3(e * 448) in registers,
     acc += pq @ vq (e4m3 tensor cores, fp32 accumulate). Epilogue:
     O^T = bf16(acc * (1 / (448 l)) * s_v). pq never reaches DRAM.
No FMA contraction in B (enable_fp_fusion=False), libdevice exp. Rows with non-finite
scores are not handled; S of finite bf16 q / k is finite.

Registered as the triton_op ``torch.ops.trtllm.flux3_klein_split_attn_e4m3`` so the
compiled decoder traces through to the two kernels.
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

# Kernel A at T = 65536, cold L2, on channels_last q / k ([C, T] strides (1, C)), SM120:
# 128x256x32, 8 warps, 4 stages: 16.63 ms (3 stages 19.39 ms). S is bitwise the fp32-out
# bf16 bmm's.
CFG_A = {"BM": 128, "BN": 256, "BK": 32, "GROUP_M": 8, "num_warps": 8, "num_stages": 4}
# Kernel B at T = 65536: 64x512x64, 8 warps, 2 stages: 13.01 ms (an e4m3 row softmax
# 14.49 ms + an e4m3 _scaled_mm 5.90 ms). BN = C = 512, so each S tile is read once.
CFG_B = {"BM": 64, "BN": 512, "BK": 64, "num_warps": 8, "num_stages": 2}


@triton.jit
def _qk_tilemax_kernel(
    Q,  # [C, T] bf16 (strides sq_c, sq_t; channels_last decoder: (1, C))
    K,  # [C, T] bf16 (strides sk_c, sk_t)
    S,  # [T, T] fp32 out
    MP,  # [T, T // BN] fp32 out: per-(row, n-tile) max of S
    T,
    sq_c,
    sq_t,
    sk_c,
    sk_t,
    C: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = T // BM
    num_pid_n = T // BN
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsz = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsz)
    pid_n = (pid % num_in_group) // gsz
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = Q + offs_k[None, :] * sq_c + offs_m[:, None] * sq_t  # [BM, BK] = Q^T tile
    b_ptrs = K + offs_k[:, None] * sk_c + offs_n[None, :] * sk_t  # [BK, BN]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for _ in range(0, C, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sq_c
        b_ptrs += BK * sk_c
    s_ptrs = S + offs_m.to(tl.int64)[:, None] * T + offs_n[None, :]
    tl.store(s_ptrs, acc)
    tm = tl.max(acc, axis=1)
    tl.store(MP + offs_m * num_pid_n + pid_n, tm)


@triton.jit
def _pv_prologue_kernel(
    S,  # [T, T] fp32
    MP,  # [T, NT] fp32 tile maxes
    VQ,  # [C, T] e4m3 (per-channel quantized V)
    SV,  # [C] fp32 per-channel V scale
    OUT,  # [T, C] bf16 out
    T,
    sv_c,
    sv_t,
    scale,
    C: tl.constexpr,
    NT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    scale = scale.to(tl.float32)  # Inductor passes Python floats as fp64
    pid = tl.program_id(0)
    num_pid_n = C // BN
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m = tl.full([BM], float("-inf"), tl.float32)
    MPC: tl.constexpr = 64 if NT >= 64 else NT
    for j in range(0, NT, MPC):
        mp = tl.load(MP + offs_m[:, None] * NT + (j + tl.arange(0, MPC))[None, :])
        m = tl.maximum(m, tl.max(mp, axis=1))  # [BM] exact row max of S
    s_ptrs = S + offs_m.to(tl.int64)[:, None] * T + offs_k[None, :]
    v_ptrs = VQ + offs_n[None, :] * sv_c + offs_k[:, None] * sv_t  # [BK, BN] = vq^T tile
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    lacc = tl.zeros([BM, BK], dtype=tl.float32)
    for k in range(0, T, BK):
        x = tl.load(s_ptrs, eviction_policy="evict_first")
        e = libdevice.exp((x * 1.0 - m[:, None]) * scale)
        lacc = lacc + e
        p = (e * 448.0).to(tl.float8e4nv)
        v = tl.load(v_ptrs)
        acc = tl.dot(p, v, acc)
        s_ptrs += BK
        v_ptrs += BK * sv_t
    row_sum = tl.sum(lacc, axis=1)
    sa = 1.0 / (row_sum * 448.0)
    sv = tl.load(SV + offs_n)
    out = (acc * sa[:, None]) * sv[None, :]
    o_ptrs = OUT + offs_m[:, None] * C + offs_n[None, :]
    tl.store(o_ptrs, out.to(tl.bfloat16))


def supported(c: int, t: int) -> bool:
    """The kernels' loads and stores are unmasked: c must be 512 (kernel B's BN) and T a
    multiple of every tile; kernel B reduces the [T, T / 256] tile maxes in chunks of
    min(64, NT) columns, so NT = T / 256 must be a power of two <= 64 or a multiple of 64.
    S offsets are int64; T * c must stay below 2^31."""
    nt = t // CFG_A["BN"]
    return (
        c == CFG_B["BN"]
        and t % max(CFG_A["BM"], CFG_A["BN"], CFG_B["BM"], CFG_B["BK"]) == 0
        and c % CFG_A["BK"] == 0
        and t >= CFG_A["BN"]
        and t * c < 2**31
        and nt >= 1
        and ((nt <= 64 and nt & (nt - 1) == 0) or nt % 64 == 0)
    )


@triton_op("trtllm::flux3_klein_split_attn_e4m3", mutates_args={})
def split_attn_e4m3_op(q: Tensor, k: Tensor, vq: Tensor, sv: Tensor, scale: float) -> Tensor:
    """q, k [C, T] bf16 (any strides), vq [C, T] e4m3 row-major, sv [C] fp32 ->
    O^T [T, C] bf16."""
    c, t = q.shape
    s = torch.empty((t, t), device=q.device, dtype=torch.float32)
    mp = torch.empty((t, t // CFG_A["BN"]), device=q.device, dtype=torch.float32)
    o = torch.empty((t, c), device=q.device, dtype=torch.bfloat16)
    cfg = CFG_A
    wrap_triton(_qk_tilemax_kernel)[((t // cfg["BM"]) * (t // cfg["BN"]),)](
        q,
        k,
        s,
        mp,
        t,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        C=c,
        BM=cfg["BM"],
        BN=cfg["BN"],
        BK=cfg["BK"],
        GROUP_M=cfg["GROUP_M"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    cfg = CFG_B
    wrap_triton(_pv_prologue_kernel)[((t // cfg["BM"]) * (c // cfg["BN"]),)](
        s,
        mp,
        vq,
        sv,
        o,
        t,
        vq.stride(0),
        vq.stride(1),
        scale,
        C=c,
        NT=mp.shape[1],
        BM=cfg["BM"],
        BN=cfg["BN"],
        BK=cfg["BK"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        enable_fp_fusion=False,
    )
    return o
