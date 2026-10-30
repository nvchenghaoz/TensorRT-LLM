# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-softmax e4m3 mid-block attention for the FLUX.3 VAE decoder (b == 1, one head).

main (_attention_tc, fp8_pv=True) runs three passes over the fp32 score matrix S [T, T]:
bmm -> S (write), row_softmax_e4m3 (read S, write pq e4m3 + scale_a), _scaled_mm (read pq).
Here two kernels:
  A  _qk_tilemax_kernel: S = Q^T K (bf16 in, fp32 accumulate, fp32 out) and, per (row,
     N-tile), the tile max of S (written to a small [T, T / BN_A] fp32 buffer).
  B  _pv_prologue_kernel: m = max over the row's tile maxes (== max of the row, bitwise),
     then for each K block: e = exp((s - m) * scale), l += e, pq = e4m3(e * 448) in
     registers, acc += pq @ vq (e4m3 tensor cores, fp32 accumulate). Epilogue:
     O^T = bf16(acc * (1 / (448 l)) * s_v). pq never reaches DRAM.
Math is main's expression element for element (enable_fp_fusion=False, libdevice exp,
same rowmax); only the accumulation orders of S (K order), l and P.V differ (reordered).
Non-finite rows (main's NaN/inf branch) are not reproduced; S of finite bf16 q/k is finite.
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

# TLLM_FLUX3_VAE_SPLIT_ATTN=0 keeps landing 25's path (bmm + row_softmax_e4m3 + _scaled_mm).
ENABLED = os.environ.get("TLLM_FLUX3_VAE_SPLIT_ATTN", "1") != "0"

# Kernel A sweep at T=65536, cold L2, on the served channels_last q/k ([C, T] strides (1, C)),
# job 765: 128x256x32 / 8 warps / 4 stages 16.63 ms (3 stages 19.39; main's nvjet bmm 15.27 ms).
# S is bitwise equal to the bmm's (jobs 757, 765).
CFG_A = {"BM": 128, "BN": 256, "BK": 32, "GROUP_M": 8, "num_warps": 8, "num_stages": 4}
# Kernel B sweep (job 759): 64x512x64 / 8 warps / 2 stages 13.01 ms (main's row_softmax_e4m3
# 14.49 + e4m3 _scaled_mm 5.90 ms); BN = C = 512, so each S tile is read once from DRAM.
CFG_B = {"BM": 64, "BN": 512, "BK": 64, "num_warps": 8, "num_stages": 2, "N_FAST": True}


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
    VQ,  # [C, T] e4m3 row-major (per-channel quantized V)
    SV,  # [C] fp32 per-channel V scale
    O,  # [T, C] bf16 out
    SA,  # [T] fp32 out: 1 / (448 l)
    PQD,  # [T, T] e4m3 debug out (only if DEBUG)
    T,
    sv_c,
    sv_t,
    scale,
    C: tl.constexpr,
    NT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    N_FAST: tl.constexpr,
    DEBUG: tl.constexpr,
):
    scale = scale.to(tl.float32)  # Inductor passes Python floats as fp64 (pitfall 887)
    pid = tl.program_id(0)
    num_pid_n = C // BN
    if N_FAST:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        pid_m = pid % (T // BM)
        pid_n = pid // (T // BM)
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
        if DEBUG:
            if pid_n == 0:
                d_ptrs = PQD + offs_m.to(tl.int64)[:, None] * T + (k + offs_k)[None, :]
                tl.store(d_ptrs, p)
        v = tl.load(v_ptrs)
        acc = tl.dot(p, v, acc)
        s_ptrs += BK
        v_ptrs += BK * sv_t
    l = tl.sum(lacc, axis=1)
    sa = 1.0 / (l * 448.0)
    sv = tl.load(SV + offs_n)
    out = (acc * sa[:, None]) * sv[None, :]
    o_ptrs = O + offs_m[:, None] * C + offs_n[None, :]
    tl.store(o_ptrs, out.to(tl.bfloat16))
    if pid_n == 0:
        tl.store(SA + offs_m, sa)


def can_use(q: Tensor) -> bool:
    """q: [b, c, T] bf16 CUDA; b == 1, T a multiple of every tile."""
    if not (ENABLED and q.is_cuda and q.dim() == 3 and q.shape[0] == 1):
        return False
    c, t = q.shape[1], q.shape[2]
    return (
        t % max(CFG_A["BM"], CFG_A["BN"], CFG_B["BM"], CFG_B["BK"]) == 0
        and c % CFG_A["BK"] == 0
        and c % CFG_B["BN"] == 0
        and t >= CFG_A["BN"]
    )


def qk_tilemax(q: Tensor, k: Tensor, cfg: dict = None) -> tuple[Tensor, Tensor]:
    """q, k [C, T] bf16 contiguous -> (S [T, T] fp32, MP [T, T // BN] fp32). Direct launch."""
    cfg = cfg or CFG_A
    c, t = q.shape
    s = torch.empty((t, t), device=q.device, dtype=torch.float32)
    nt = t // cfg["BN"]
    mp = torch.empty((t, nt), device=q.device, dtype=torch.float32)
    _launch_a(q, k, s, mp, cfg, triton_wrap=False)
    return s, mp


def _launch_a(q, k, s, mp, cfg, triton_wrap):
    c, t = q.shape
    grid = ((t // cfg["BM"]) * (t // cfg["BN"]),)
    kern = wrap_triton(_qk_tilemax_kernel) if triton_wrap else _qk_tilemax_kernel
    kern[grid](
        q, k, s, mp, t, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        C=c, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP_M=cfg["GROUP_M"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )


def _launch_b(s, mp, vq, sv, o, sa, pqd, scale, cfg, triton_wrap, debug=False):
    c, t = vq.shape
    grid = ((t // cfg["BM"]) * (c // cfg["BN"]),)
    kern = wrap_triton(_pv_prologue_kernel) if triton_wrap else _pv_prologue_kernel
    kern[grid](
        s, mp, vq, sv, o, sa, pqd, t, vq.stride(0), vq.stride(1), scale,
        C=c, NT=mp.shape[1], BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"],
        N_FAST=cfg["N_FAST"], DEBUG=debug,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        enable_fp_fusion=False,
    )


def pv_prologue(s, mp, vq, sv, scale, cfg=None, debug=False):
    """Direct launch of kernel B. Returns (o_t [T, C] bf16, scale_a [T] fp32, pq debug|None)."""
    cfg = cfg or CFG_B
    c, t = vq.shape
    o = torch.empty((t, c), device=s.device, dtype=torch.bfloat16)
    sa = torch.empty((t,), device=s.device, dtype=torch.float32)
    pqd = torch.empty((t, t) if debug else (1,), device=s.device, dtype=torch.float8_e4m3fn)
    _launch_b(s, mp, vq, sv, o, sa, pqd, scale, cfg, triton_wrap=False, debug=debug)
    return o, sa, (pqd if debug else None)


@triton_op("flux3_vae::split_attn_e4m3", mutates_args={})
def split_attn_e4m3(q: Tensor, k: Tensor, vq: Tensor, sv: Tensor, scale: float) -> Tensor:
    """q, k [C, T] bf16, vq [C, T] e4m3, sv [C] fp32 -> O^T [T, C] bf16.

    O^T = (pq @ vq^T) * (1 / (448 l)) * sv with pq = e4m3(exp((S - rowmax S) * scale) * 448),
    l = row sum of the unrounded exp, S = q^T k in fp32.
    """
    c, t = q.shape
    s = torch.empty((t, t), device=q.device, dtype=torch.float32)
    mp = torch.empty((t, t // CFG_A["BN"]), device=q.device, dtype=torch.float32)
    o = torch.empty((t, c), device=q.device, dtype=torch.bfloat16)
    sa = torch.empty((t,), device=q.device, dtype=torch.float32)
    pqd = torch.empty((1,), device=q.device, dtype=torch.float8_e4m3fn)
    _launch_a(q, k, s, mp, CFG_A, triton_wrap=True)
    _launch_b(s, mp, vq, sv, o, sa, pqd, scale, CFG_B, triton_wrap=True)
    return o


def split_attn_e4m3_direct(q, k, vq, sv, scale):
    """Same as split_attn_e4m3 without the torch.library wrapper (direct launches)."""
    s, mp = qk_tilemax(q, k)
    o, _, _ = pv_prologue(s, mp, vq, sv, scale)
    return o
