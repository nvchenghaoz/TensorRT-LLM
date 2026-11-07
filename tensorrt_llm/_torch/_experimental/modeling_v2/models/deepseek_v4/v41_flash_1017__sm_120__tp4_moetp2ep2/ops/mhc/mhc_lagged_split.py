# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 lagged mHC boundary split (decode, small M).

In V4.1's lagged wiring, a boundary's ``layer_input`` depends only on the post-updated
residual and the PREVIOUS sublayer's ``pre`` (``pre_mix_ext``), not on the boundary's own
coefficient GEMM or Sinkhorn. So the critical path needs only

    new_r[j]    = bf16(fma_k(comb[k][j], r_prev[k], post[j] * x))      (post update)
    li          = bf16(fma_j(pre_ext[j], new_r[j], 0))                  (lagged collapse)
    layer_input = bf16((li * rsqrt(sum(li^2) / hidden + eps)) * w)      (RMSNorm)

which :func:`mhc_lagged_layer_input` (K1, one Triton program per token) computes. The full
``mhc_fused_hc`` op (K2: post update + GEMM + Sinkhorn, unchanged) can then run on a side
stream and join before the boundary's other outputs are read.

K1 reproduces the CUDA kernels' arithmetic bit for bit (mhcKernels are built with
--use_fast_math), including the RMSNorm sum-of-squares reduction order, which depends on the
tactic's thread count ``P``:
  * ``fused_half_*`` (mhcBigFuseKernel Phase 2): ``P = bigfuse_bs - 32`` threads,
  * ``fused_all_fma`` (all-in-one Phase 4):       ``P = 256`` threads.
Thread ``t`` sums the 8-element chunks ``t, t + P, ...`` sequentially, each pair as
``fma(x, x, y * y)``; then lane 0 of a warp xor-butterfly (16, 8, 4, 2, 1), a sequential sum
over warps from 0, and ``rsqrt.approx(fma(total, rcp.approx(hidden), eps))`` (the SASS of
mhcBigFuseKernel: FFMA.FTZ total * MUFU.RCP(hidden) + eps, then MUFU.RSQ).

Every order-sensitive step is written as explicit elementwise fp32 ops (no tl.sum / reshape
reductions, whose association Triton may change with the layout): per-thread partials are
sequential adds over reloaded pairs, the butterfly is an explicit 31-add tree over the
partials staged in a scratch row, and the warp sum is sequential scalar adds.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber
from ...switches import env_get  # run switches frozen at served values

# Fastest for every P in graph replays at M=1..8 (job 793: 3.9-4.2 us; 5.1 us at P=96).
_DEFAULT_NUM_WARPS = 4


@triton.jit
def _lane(scr, w, k: tl.constexpr):
    # Partial of lane k of every warp w: scr[w * 32 + k].
    return tl.load(scr + w * 32 + k)


@triton.jit
def _lane0_butterfly(scr, w):
    """Lane 0 after `s += shfl_xor(s, off)` for off = 16, 8, 4, 2, 1, for every warp w."""
    a0 = _lane(scr, w, 0) + _lane(scr, w, 16)
    a1 = _lane(scr, w, 1) + _lane(scr, w, 17)
    a2 = _lane(scr, w, 2) + _lane(scr, w, 18)
    a3 = _lane(scr, w, 3) + _lane(scr, w, 19)
    a4 = _lane(scr, w, 4) + _lane(scr, w, 20)
    a5 = _lane(scr, w, 5) + _lane(scr, w, 21)
    a6 = _lane(scr, w, 6) + _lane(scr, w, 22)
    a7 = _lane(scr, w, 7) + _lane(scr, w, 23)
    a8 = _lane(scr, w, 8) + _lane(scr, w, 24)
    a9 = _lane(scr, w, 9) + _lane(scr, w, 25)
    a10 = _lane(scr, w, 10) + _lane(scr, w, 26)
    a11 = _lane(scr, w, 11) + _lane(scr, w, 27)
    a12 = _lane(scr, w, 12) + _lane(scr, w, 28)
    a13 = _lane(scr, w, 13) + _lane(scr, w, 29)
    a14 = _lane(scr, w, 14) + _lane(scr, w, 30)
    a15 = _lane(scr, w, 15) + _lane(scr, w, 31)
    b0 = a0 + a8
    b1 = a1 + a9
    b2 = a2 + a10
    b3 = a3 + a11
    b4 = a4 + a12
    b5 = a5 + a13
    b6 = a6 + a14
    b7 = a7 + a15
    c0 = b0 + b4
    c1 = b1 + b5
    c2 = b2 + b6
    c3 = b3 + b7
    d0 = c0 + c2
    d1 = c1 + c3
    return d0 + d1


@triton.jit
def _k1_chunk(i: tl.constexpr, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P: tl.constexpr,
              NCHUNK: tl.constexpr, HIDDEN: tl.constexpr, P_PAD: tl.constexpr):
    """Chunk i of every thread t (elements 8 * (i * P + t) + [0, 8)): bf16 li [P_PAD, 4, 2]."""
    chunk = i * P + t
    cm = (t < P) & (chunk < NCHUNK)
    offs = chunk[:, None, None] * 8 + v4[None, :, None] * 2 + h2[None, None, :]  # [P_PAD, 4, 2]
    m3 = cm[:, None, None] & (v4[None, :, None] < 4) & (h2[None, None, :] < 2)
    x = tl.load(xbase + offs, mask=m3, other=0.0).to(tl.float32)
    r0 = tl.load(rbase + offs, mask=m3, other=0.0).to(tl.float32)
    r1 = tl.load(rbase + HIDDEN + offs, mask=m3, other=0.0).to(tl.float32)
    r2 = tl.load(rbase + 2 * HIDDEN + offs, mask=m3, other=0.0).to(tl.float32)
    r3 = tl.load(rbase + 3 * HIDDEN + offs, mask=m3, other=0.0).to(tl.float32)
    li = tl.zeros([P_PAD, 4, 2], dtype=tl.float32)
    for j in tl.static_range(4):
        # new_r[j] = post[j] * x, then fma(comb[k][j], r[k], .) for k = 0..3 (pmap order).
        nr = tl.load(post_ptr + tok * 4 + j) * x
        nr = tl.fma(tl.load(comb_ptr + tok * 16 + 0 * 4 + j), r0, nr)
        nr = tl.fma(tl.load(comb_ptr + tok * 16 + 1 * 4 + j), r1, nr)
        nr = tl.fma(tl.load(comb_ptr + tok * 16 + 2 * 4 + j), r2, nr)
        nr = tl.fma(tl.load(comb_ptr + tok * 16 + 3 * 4 + j), r3, nr)
        nr = nr.to(tl.bfloat16).to(tl.float32)
        # Lagged collapse: fma from 0 over j = 0..3 with the previous sublayer's pre.
        li = tl.fma(tl.load(pre_ptr + tok * 4 + j), nr, li)
    return li.to(tl.bfloat16), offs, m3


@triton.jit
def _k1_sumsq(li_b, s, P_PAD: tl.constexpr):
    """s += fma(x, x, y * y) for pairs v = 0, 1, 2, 3 of the chunk, in that order."""
    lx, ly = tl.split(li_b.to(tl.float32))  # [P_PAD, 4]
    term = tl.fma(lx, lx, ly * ly)
    t_even, t_odd = tl.split(tl.reshape(term, [P_PAD, 2, 2]))  # (v0, v2), (v1, v3)
    tv0, tv2 = tl.split(t_even)
    tv1, tv3 = tl.split(t_odd)
    s = s + tv0
    s = s + tv1
    s = s + tv2
    s = s + tv3
    return s


@triton.jit
def _k1_norm(li_b, nw_b, rstd3, DEBUG: tl.constexpr):
    if DEBUG == 2:
        res = li_b  # test-only: the unnormalized collapse
    else:
        res = ((li_b.to(tl.float32) * rstd3) * nw_b.to(tl.float32)).to(tl.bfloat16)
    return res


@triton.jit
def _mhc_lagged_layer_input_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    nw_ptr,
    out_ptr,
    scr_ptr,
    dbg_ptr,
    norm_eps,
    HIDDEN: tl.constexpr,
    P: tl.constexpr,
    P_PAD: tl.constexpr,
    NI: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    SCR: tl.constexpr,
    DEBUG: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    TRIGGER_FIRST: tl.constexpr = 0,
):
    if LAUNCH_WITH_PDL:
        if TRIGGER_FIRST == 1:
            # Trigger first (idea 208): the dependent (wq_a||wkv, which stages only constant
            # weights before its own wait) may launch while this kernel's producer still runs.
            # Still no global access above the wait (memory clobber).
            tl.extra.cuda.gdc_launch_dependents()
            gdc_wait_with_memory_clobber()
        elif TRIGGER_FIRST == 2:
            # Test-only negative control: trigger and SKIP the wait (reads a live producer's output).
            tl.extra.cuda.gdc_launch_dependents()
        else:
            # First statement: no global access may be hoisted above the wait (memory clobber).
            gdc_wait_with_memory_clobber()
            tl.extra.cuda.gdc_launch_dependents()
    tok = tl.program_id(0).to(tl.int64)
    NCHUNK: tl.constexpr = HIDDEN // 8

    t = tl.arange(0, P_PAD)
    v4 = tl.arange(0, 4)
    h2 = tl.arange(0, 2)
    obase = out_ptr + tok * HIDDEN
    xbase = x_ptr + tok * HIDDEN
    rbase = res_ptr + tok * (4 * HIDDEN)
    # ---- pass 1: post update + lagged collapse -> bf16 li, and the per-thread partials
    # (chunks t, t + P, ...; pairs v = 0..3; s += fma(x, x, y * y)) with only elementwise ops
    # and tl.split (pure data movement). Chunks 0..2 stay in registers; any further chunk
    # (P = 96) is staged in `out` and reloaded in pass 2. ----
    s = tl.zeros([P_PAD], dtype=tl.float32)
    li0, o0, m0 = _k1_chunk(0, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK, HIDDEN, P_PAD)
    s = _k1_sumsq(li0, s, P_PAD)
    nw0 = tl.load(nw_ptr + o0, mask=m0, other=0.0)
    if NI > 1:
        li1, o1, m1 = _k1_chunk(1, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK,
                                HIDDEN, P_PAD)
        s = _k1_sumsq(li1, s, P_PAD)
        nw1 = tl.load(nw_ptr + o1, mask=m1, other=0.0)
    if NI > 2:
        li2, o2, m2 = _k1_chunk(2, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK,
                                HIDDEN, P_PAD)
        s = _k1_sumsq(li2, s, P_PAD)
        nw2 = tl.load(nw_ptr + o2, mask=m2, other=0.0)
    if NI > 3:
        for i in tl.static_range(3, NI):
            lik, ok, mk = _k1_chunk(i, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK,
                                    HIDDEN, P_PAD)
            s = _k1_sumsq(lik, s, P_PAD)
            tl.store(obase + ok, lik, mask=mk)
    sbase = scr_ptr + tok * SCR
    tl.store(sbase + t, s)
    tl.debug_barrier()

    # ---- warp butterfly (lane 0), then the sequential sum over warps ----
    w = tl.arange(0, W_PAD)
    tl.store(sbase + P_PAD + w, _lane0_butterfly(sbase, w))
    tl.debug_barrier()
    # CUDA: total = 0; total += s_sumsq[w] for w = 0..W-1 (0 + w0 == w0 exactly).
    total = tl.load(sbase + P_PAD)
    for k in tl.static_range(1, W):
        total = total + tl.load(sbase + P_PAD + k)
    # mhcKernels (--use_fast_math): FFMA.FTZ total * MUFU.RCP(hidden) + eps, MUFU.RSQ.
    hidden_f = tl.full([1], HIDDEN, tl.float32)
    rcp = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [hidden_f],
                                    dtype=tl.float32, is_pure=True, pack=1)
    total1 = tl.full([1], 0.0, tl.float32) + total
    arg = tl.fma(total1, rcp, tl.full([1], 0.0, tl.float32) + norm_eps)
    rstd = tl.math.rsqrt(arg)  # [1]
    if DEBUG:
        z = tl.arange(0, 1)
        tl.store(dbg_ptr + tok * 2 + z, total1)
        tl.store(dbg_ptr + tok * 2 + 1 + z, rstd)

    # ---- pass 2: out = bf16((li * rstd) * w) ----
    rstd3 = tl.reshape(rstd, [1, 1, 1])
    tl.store(obase + o0, _k1_norm(li0, nw0, rstd3, DEBUG), mask=m0)
    if NI > 1:
        tl.store(obase + o1, _k1_norm(li1, nw1, rstd3, DEBUG), mask=m1)
    if NI > 2:
        tl.store(obase + o2, _k1_norm(li2, nw2, rstd3, DEBUG), mask=m2)
    if NI > 3:
        for i in tl.static_range(3, NI):
            chunk = i * P + t
            cm = (t < P) & (chunk < NCHUNK)
            offs = chunk[:, None, None] * 8 + v4[None, :, None] * 2 + h2[None, None, :]
            m3 = cm[:, None, None] & (v4[None, :, None] < 4) & (h2[None, None, :] < 2)
            lik = tl.load(obase + offs, mask=m3, other=0.0)
            nwk = tl.load(nw_ptr + offs, mask=m3, other=0.0)
            tl.store(obase + offs, _k1_norm(lik, nwk, rstd3, DEBUG), mask=m3)


# ---------------------------------------------------------------------------------------------
# Fast K1 (TRTLLM_MHC_K1_FAST, default on): the same values, the same adds in the same order as
# _mhc_lagged_layer_input_kernel, but the lane-0 butterfly and the warp sum run in registers
# (reshape / permute / tl.split are pure data movement, every add is an explicit elementwise
# fp32 add), so there is no global scratch round trip and no debug_barrier; and the constant
# norm weight is loaded before the PDL wait.
# ---------------------------------------------------------------------------------------------


@triton.jit
def _halves(v, R: tl.constexpr, H: tl.constexpr):
    """v [R, 2H] -> (v[:, :H], v[:, H:]) by data movement only."""
    return tl.split(tl.permute(tl.reshape(v, [R, 2, H]), (0, 2, 1)))


@triton.jit
def _evod(v, N: tl.constexpr):
    """v [N] -> (v[0::2], v[1::2]) by data movement only."""
    return tl.split(tl.reshape(v, [N // 2, 2]))


@triton.jit
def _reg_lane0_butterfly(s, W_PAD: tl.constexpr):
    """[P_PAD] partials -> [W_PAD] lane-0 values, in _lane0_butterfly's exact pairing."""
    s2 = tl.reshape(s, [W_PAD, 32])  # [w, k] = lane k of warp w
    lo, hi = _halves(s2, W_PAD, 16)
    a = lo + hi  # a_k = lane k + lane k+16
    lo, hi = _halves(a, W_PAD, 8)
    b = lo + hi  # b_k = a_k + a_{k+8}
    lo, hi = _halves(b, W_PAD, 4)
    c = lo + hi  # c_k = b_k + b_{k+4}
    lo, hi = _halves(c, W_PAD, 2)
    d = lo + hi  # d_k = c_k + c_{k+2}
    d0, d1 = tl.split(d)
    return d0 + d1


@triton.jit
def _seq_add(t, wk, k: tl.constexpr, W: tl.constexpr):
    if k < W:
        t = t + wk
    return t


@triton.jit
def _warp_sum(wv, W: tl.constexpr, W_PAD: tl.constexpr):
    """total = w_0 + w_1 + ... + w_{W-1}, sequential scalar adds (shape [1])."""
    if W_PAD == 4:
        e, o = _evod(wv, 4)
        w0, w2 = _evod(e, 2)
        w1, w3 = _evod(o, 2)
        t = w0
        t = _seq_add(t, w1, 1, W)
        t = _seq_add(t, w2, 2, W)
        t = _seq_add(t, w3, 3, W)
    elif W_PAD == 8:
        e, o = _evod(wv, 8)  # [0,2,4,6], [1,3,5,7]
        ee, eo = _evod(e, 4)  # [0,4], [2,6]
        oe, oo = _evod(o, 4)  # [1,5], [3,7]
        w0, w4 = _evod(ee, 2)
        w2, w6 = _evod(eo, 2)
        w1, w5 = _evod(oe, 2)
        w3, w7 = _evod(oo, 2)
        t = w0
        t = _seq_add(t, w1, 1, W)
        t = _seq_add(t, w2, 2, W)
        t = _seq_add(t, w3, 3, W)
        t = _seq_add(t, w4, 4, W)
        t = _seq_add(t, w5, 5, W)
        t = _seq_add(t, w6, 6, W)
        t = _seq_add(t, w7, 7, W)
    else:
        tl.static_assert(W_PAD == 16)
        e, o = _evod(wv, 16)  # [0,2,..,14], [1,3,..,15]
        ee, eo = _evod(e, 8)  # [0,4,8,12], [2,6,10,14]
        oe, oo = _evod(o, 8)  # [1,5,9,13], [3,7,11,15]
        eee, eeo = _evod(ee, 4)  # [0,8], [4,12]
        eoe, eoo = _evod(eo, 4)  # [2,10], [6,14]
        oee, oeo = _evod(oe, 4)  # [1,9], [5,13]
        ooe, ooo = _evod(oo, 4)  # [3,11], [7,15]
        w0, w8 = _evod(eee, 2)
        w4, w12 = _evod(eeo, 2)
        w2, w10 = _evod(eoe, 2)
        w6, w14 = _evod(eoo, 2)
        w1, w9 = _evod(oee, 2)
        w5, w13 = _evod(oeo, 2)
        w3, w11 = _evod(ooe, 2)
        w7, w15 = _evod(ooo, 2)
        t = w0
        t = _seq_add(t, w1, 1, W)
        t = _seq_add(t, w2, 2, W)
        t = _seq_add(t, w3, 3, W)
        t = _seq_add(t, w4, 4, W)
        t = _seq_add(t, w5, 5, W)
        t = _seq_add(t, w6, 6, W)
        t = _seq_add(t, w7, 7, W)
        t = _seq_add(t, w8, 8, W)
        t = _seq_add(t, w9, 9, W)
        t = _seq_add(t, w10, 10, W)
        t = _seq_add(t, w11, 11, W)
        t = _seq_add(t, w12, 12, W)
        t = _seq_add(t, w13, 13, W)
        t = _seq_add(t, w14, 14, W)
        t = _seq_add(t, w15, 15, W)
    return t


@triton.jit
def _nw_offs(i: tl.constexpr, t, v4, h2, P: tl.constexpr, NCHUNK: tl.constexpr):
    chunk = i * P + t
    cm = (t < P) & (chunk < NCHUNK)
    offs = chunk[:, None, None] * 8 + v4[None, :, None] * 2 + h2[None, None, :]
    m3 = cm[:, None, None] & (v4[None, :, None] < 4) & (h2[None, None, :] < 2)
    return offs, m3


@triton.jit
def _mhc_lagged_layer_input_fast_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    nw_ptr,
    out_ptr,
    dbg_ptr,
    norm_eps,
    HIDDEN: tl.constexpr,
    P: tl.constexpr,
    P_PAD: tl.constexpr,
    NI: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    DEBUG: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    TRIGGER_FIRST: tl.constexpr,
):
    NCHUNK: tl.constexpr = HIDDEN // 8
    t = tl.arange(0, P_PAD)
    v4 = tl.arange(0, 4)
    h2 = tl.arange(0, 2)
    if LAUNCH_WITH_PDL:
        if TRIGGER_FIRST != 0:
            # Trigger first (idea 208; 2 = test-only negative control that also skips the wait).
            tl.extra.cuda.gdc_launch_dependents()
    # The norm weight is a constant module weight that no preceding kernel writes: it is the
    # only load issued before the PDL wait.
    n0o, n0m = _nw_offs(0, t, v4, h2, P, NCHUNK)
    nw0 = tl.load(nw_ptr + n0o, mask=n0m, other=0.0)
    if NI > 1:
        n1o, n1m = _nw_offs(1, t, v4, h2, P, NCHUNK)
        nw1 = tl.load(nw_ptr + n1o, mask=n1m, other=0.0)
    if NI > 2:
        n2o, n2m = _nw_offs(2, t, v4, h2, P, NCHUNK)
        nw2 = tl.load(nw_ptr + n2o, mask=n2m, other=0.0)
    if LAUNCH_WITH_PDL:
        if TRIGGER_FIRST == 0:
            # Every input other than the norm weight is read after the wait (memory clobber).
            gdc_wait_with_memory_clobber()
            tl.extra.cuda.gdc_launch_dependents()
        elif TRIGGER_FIRST == 1:
            gdc_wait_with_memory_clobber()
    tok = tl.program_id(0).to(tl.int64)

    obase = out_ptr + tok * HIDDEN
    xbase = x_ptr + tok * HIDDEN
    rbase = res_ptr + tok * (4 * HIDDEN)
    s = tl.zeros([P_PAD], dtype=tl.float32)
    li0, o0, m0 = _k1_chunk(0, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK, HIDDEN, P_PAD)
    s = _k1_sumsq(li0, s, P_PAD)
    if NI > 1:
        li1, o1, m1 = _k1_chunk(1, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK,
                                HIDDEN, P_PAD)
        s = _k1_sumsq(li1, s, P_PAD)
    if NI > 2:
        li2, o2, m2 = _k1_chunk(2, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK,
                                HIDDEN, P_PAD)
        s = _k1_sumsq(li2, s, P_PAD)
    if NI > 3:
        for i in tl.static_range(3, NI):
            lik, ok, mk = _k1_chunk(i, t, v4, h2, xbase, rbase, post_ptr, comb_ptr, pre_ptr, tok, P, NCHUNK,
                                    HIDDEN, P_PAD)
            s = _k1_sumsq(lik, s, P_PAD)
            tl.store(obase + ok, lik, mask=mk)

    if DEBUG == 3:
        # Test-only floor probe (wrong numerics): no reduction at all.
        total = tl.full([1], 1.0, tl.float32)
    else:
        # Lane-0 butterfly (16, 8, 4, 2, 1), then total = w_0 + ... + w_{W-1}, in registers.
        total = _warp_sum(_reg_lane0_butterfly(s, W_PAD), W, W_PAD)
    hidden_f = tl.full([1], HIDDEN, tl.float32)
    rcp = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [hidden_f],
                                    dtype=tl.float32, is_pure=True, pack=1)
    total1 = tl.full([1], 0.0, tl.float32) + total
    arg = tl.fma(total1, rcp, tl.full([1], 0.0, tl.float32) + norm_eps)
    rstd = tl.math.rsqrt(arg)  # [1]
    if DEBUG == 1 or DEBUG == 2:
        z = tl.arange(0, 1)
        tl.store(dbg_ptr + tok * 2 + z, total1)
        tl.store(dbg_ptr + tok * 2 + 1 + z, rstd)

    rstd3 = tl.reshape(rstd, [1, 1, 1])
    tl.store(obase + o0, _k1_norm(li0, nw0, rstd3, DEBUG), mask=m0)
    if NI > 1:
        tl.store(obase + o1, _k1_norm(li1, nw1, rstd3, DEBUG), mask=m1)
    if NI > 2:
        tl.store(obase + o2, _k1_norm(li2, nw2, rstd3, DEBUG), mask=m2)
    if NI > 3:
        for i in tl.static_range(3, NI):
            offs, m3 = _nw_offs(i, t, v4, h2, P, NCHUNK)
            lik = tl.load(obase + offs, mask=m3, other=0.0)
            nwk = tl.load(nw_ptr + offs, mask=m3, other=0.0)
            tl.store(obase + offs, _k1_norm(lik, nwk, rstd3, DEBUG), mask=m3)


# TRTLLM_MHC_K1_FAST=0 selects _mhc_lagged_layer_input_kernel (main's K1, unchanged compile key).
K1_FAST = env_get("TRTLLM_MHC_K1_FAST", "1") != "0"
# Fast-K1 num_warps per P (graph replays at M=1..8); P not listed -> _DEFAULT_NUM_WARPS.
_FAST_NUM_WARPS = {}


_SPLIT_ENV_ON = env_get("TRTLLM_MHC_SPLIT_LAGGED", "1") != "0"
_SPLIT_MAX_TOKENS = 8
# K1 trigger-first (idea 208), used only where the caller asks for it (the hc_attn boundary, whose
# PDL dependent is the attention-input wq_a||wkv low-M GEMM). 0 keeps the wait-then-trigger K1
# (the old compile key: TRIGGER_FIRST is then not passed at all).
K1_TRIGGER_FIRST_ENV = env_get("TRTLLM_MHC_K1_TRIGGER_FIRST", "1") != "0"
# Per-M rule (idea 208 gate, 1-GPU chain at the last arriver's AR window W_crit): trigger-first
# pays only from this row count up. Below it, A's early launch costs more than its staging hides
# inside a ~3 us all-reduce, so K1 keeps wait-then-trigger there. Read at call (capture) time.
K1_TRIGGER_FIRST_MIN_TOKENS = int(env_get("TRTLLM_MHC_K1_TRIGGER_FIRST_MIN_TOKENS", "4"))
# Test-only race probe (default 0, never set in timed or serving runs): cycles of
# torch.cuda._sleep on the aux stream before every K2. Read at capture time.
TEST_AUX_SLEEP_CYCLES = int(env_get("TRTLLM_MHC_SPLIT_TEST_SLEEP_CYCLES", "0"))
# Test-only: keep K2's (discarded) layer_input on the join object for comparison.
TEST_KEEP_K2_LAYER_INPUT = env_get("TRTLLM_MHC_SPLIT_TEST_KEEP_K2", "0") == "1"


# K1 fold (decode M <= 4, SM120): at the hc_ffn boundary K1 is not launched; the MoE's fused router launch
# (native_ops.dsv4_fused_router[_route]_k1fold) computes layer_input itself after its PDL wait and writes it
# into the boundary's layer_input tensor, and the shared expert (aux stream) runs its own K1 into its own buffer.
# TRTLLM_DSV4_ROUTER_K1_FOLD=0 turns it off (K1 on the main stream as before).
K1_FOLD_ENV_ON = env_get("TRTLLM_DSV4_ROUTER_K1_FOLD", "1") != "0"
# The kernel takes 1..4 tokens. Default 2: in the 40-layer cold-L2 chain (job 1601) the fold saves 4.1 / 3.1
# us/layer at M=1 / 2 but costs +3.1 us/layer at M=4 (every CTA reads M x 50 KB of K1 inputs).
K1_FOLD_MAX_TOKENS = min(4, int(env_get("TRTLLM_DSV4_ROUTER_K1_FOLD_MAX_M", "2")))


class K1FoldPending:
    """The K1 of one split boundary, not launched yet: ``out`` (the boundary's layer_input) is written by the
    consumer's fused router launch (:meth:`fold_args`), or by :meth:`materialize` (the plain K1 into ``out`` on
    the current stream) when the consumer cannot fold it. :meth:`run_copy` runs K1 into a fresh buffer on the
    current stream (the shared expert's input on its aux stream)."""

    __slots__ = ("out", "args", "reducer_threads", "norm_eps", "done", "folded")

    def __init__(self, out, args, norm_eps, reducer_threads):
        self.out = out
        self.args = args  # (x_prev, residual_prev, post, comb, pre, norm_weight)
        self.norm_eps = float(norm_eps)
        self.reducer_threads = int(reducer_threads)
        self.done = False
        self.folded = False  # True when the router launch produced ``out`` (not materialize)

    def fold_args(self):
        """(x_prev, residual, post, comb, pre, norm_weight, norm_eps, reducer_threads) for the fold op; marks
        ``out`` as produced by that launch."""
        self.done = True
        self.folded = True
        return (*self.args, self.norm_eps, self.reducer_threads)

    def materialize(self):
        if not self.done:
            self.done = True
            mhc_lagged_layer_input(*self.args, self.norm_eps, self.reducer_threads, out=self.out)
        return self.out

    def run_copy(self):
        if TEST_AUX_SLEEP_CYCLES > 0:
            torch.cuda._sleep(TEST_AUX_SLEEP_CYCLES)
        return mhc_lagged_layer_input(*self.args, self.norm_eps, self.reducer_threads)


class MhcSplitJoin:
    """Pending K2 of one split boundary: ``join()`` makes the calling stream wait for it.

    Holds a reference to every K2 input until the join: the caller rebinds/drops them
    (``residual, ... = boundary(...)``), and the caching allocator would otherwise hand
    their blocks to later main-stream allocations while K2 still reads them.
    ``k1_fold``: the boundary's K1 when it was deferred to the consumer (K1 fold), else None.
    """

    __slots__ = ("event", "keep", "k2_layer_input", "k1_fold")

    def __init__(self, event, keep, k2_layer_input=None, k1_fold=None):
        self.event = event
        self.keep = keep
        self.k2_layer_input = k2_layer_input
        self.k1_fold = k1_fold

    def join(self):
        if self.event is not None:
            self.event.wait()
        self.event = None
        self.keep = None


def split_supported_here(num_tokens: int) -> bool:
    """Host-side gate: SM120, multi-stream (graph capture/warmup), 1 <= M <= 8, env on."""
    if not _SPLIT_ENV_ON or not (1 <= num_tokens <= _SPLIT_MAX_TOKENS):
        return False
    if env_get("FORCE_DETERMINISTIC", "0") == "1":
        return False
    from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream

    if not do_multi_stream() or torch.compiler.is_compiling():
        return False
    from tensorrt_llm._torch.autotuner import AutoTuner

    if AutoTuner.get().is_tuning_mode:
        return False
    return _is_sm120()


_SM120 = None


def _is_sm120() -> bool:
    global _SM120
    if _SM120 is None:
        _SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
    return _SM120


def _pdl_enabled() -> bool:
    if env_get("TRTLLM_ENABLE_PDL", "1") == "0":
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def reducer_threads_for_tactic(tactic) -> Optional[int]:
    """Phase-2 thread count ``P`` of the mhc_fused_hc tactic, or None if K1 can't mirror it."""
    if tactic is None or tactic == -1:
        return None
    backend, _tile_n, _ks, bigfuse_bs, _tile_m = tactic
    if backend == "fused_half_fma":
        if bigfuse_bs >= 512:
            return 480
        if bigfuse_bs >= 256:
            return 224
        if bigfuse_bs > 0:
            return 96
        return None  # C++ heuristic block size: not mirrored
    if backend == "fused_all_fma":
        return 256
    return None  # MMA backends (SM100 only) are not mirrored


def mhc_lagged_layer_input(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    pre_mix_ext: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    reducer_threads: int,
    enable_pdl: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
    debug: Optional[torch.Tensor] = None,
    debug_li: bool = False,
    num_warps: Optional[int] = None,
    trigger_first: bool = False,
    _test_skip_wait: bool = False,
    fast: Optional[bool] = None,
    test_floor: bool = False,
) -> torch.Tensor:
    """K1: post update + lagged collapse + RMSNorm -> layer_input [M, hidden] bf16.

    Bitwise equal to the ``layer_input`` of ``mhc_fused_hc(..., pre_mix_ext=...)`` run with a
    tactic whose phase-2 thread count is ``reducer_threads`` (see module docstring).
    ``debug`` (test-only, fp32 [M, 2]) receives (sum of squares, rstd); with ``debug_li`` the
    output is the unnormalized collapse.
    ``trigger_first`` (honoured when TRTLLM_MHC_K1_TRIGGER_FIRST != 0, PDL is on and
    M >= K1_TRIGGER_FIRST_MIN_TOKENS) launches
    the PDL dependents before this kernel's own wait; its successor must read nothing produced
    upstream before its own griddepcontrol.wait. ``_test_skip_wait`` is a test-only negative
    control (trigger, no wait).
    ``fast`` (default ``K1_FAST``) picks the register K1 (same bits, no scratch round trip);
    ``test_floor`` (test-only, fast K1, wrong numerics) skips the reduction.
    """
    M, hidden = x_prev.shape
    assert residual_prev.shape == (M, 4, hidden) and residual_prev.is_contiguous()
    assert x_prev.is_contiguous() and x_prev.dtype == torch.bfloat16
    assert residual_prev.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    assert hidden % 8 == 0
    assert post_mix_prev.dtype == torch.float32 and post_mix_prev.is_contiguous()
    assert comb_mix_prev.dtype == torch.float32 and comb_mix_prev.is_contiguous()
    assert pre_mix_ext.dtype == torch.float32 and pre_mix_ext.is_contiguous()
    P = int(reducer_threads)
    assert P % 32 == 0
    if enable_pdl is None:
        enable_pdl = _pdl_enabled()
    nchunk = hidden // 8
    NI = (nchunk + P - 1) // P
    P_PAD = triton.next_power_of_2(P)
    W = P // 32
    W_PAD = max(triton.next_power_of_2(W), 1)
    SCR = P_PAD + W_PAD
    if out is None:
        out = torch.empty_like(x_prev)
    if M == 0:
        return out
    if fast is None:
        fast = K1_FAST
    if fast and W_PAD in (4, 8, 16):
        if enable_pdl and _test_skip_wait:
            tf = 2
        elif enable_pdl and trigger_first and K1_TRIGGER_FIRST_ENV and M >= K1_TRIGGER_FIRST_MIN_TOKENS:
            tf = 1
        else:
            tf = 0
        _mhc_lagged_layer_input_fast_kernel[(M,)](
            x_prev,
            residual_prev,
            post_mix_prev,
            comb_mix_prev,
            pre_mix_ext,
            norm_weight,
            out,
            debug if debug is not None else out,
            float(norm_eps),
            HIDDEN=hidden,
            P=P,
            P_PAD=P_PAD,
            NI=NI,
            W=W,
            W_PAD=W_PAD,
            DEBUG=3 if test_floor else (0 if debug is None else (2 if debug_li else 1)),
            LAUNCH_WITH_PDL=bool(enable_pdl),
            TRIGGER_FIRST=tf,
            num_warps=num_warps if num_warps is not None else _FAST_NUM_WARPS.get(P, _DEFAULT_NUM_WARPS),
            enable_fp_fusion=False,
            launch_pdl=bool(enable_pdl),
        )
        return out
    scratch = torch.empty((M, SCR), dtype=torch.float32, device=x_prev.device)
    extra = {}
    if enable_pdl and _test_skip_wait:
        extra["TRIGGER_FIRST"] = 2
    elif enable_pdl and trigger_first and K1_TRIGGER_FIRST_ENV and M >= K1_TRIGGER_FIRST_MIN_TOKENS:
        extra["TRIGGER_FIRST"] = 1
    _mhc_lagged_layer_input_kernel[(M,)](
        x_prev,
        residual_prev,
        post_mix_prev,
        comb_mix_prev,
        pre_mix_ext,
        norm_weight,
        out,
        scratch,
        debug if debug is not None else scratch,
        float(norm_eps),
        HIDDEN=hidden,
        P=P,
        P_PAD=P_PAD,
        NI=NI,
        W=W,
        W_PAD=W_PAD,
        SCR=SCR,
        DEBUG=0 if debug is None else (2 if debug_li else 1),
        LAUNCH_WITH_PDL=bool(enable_pdl),
        num_warps=num_warps if num_warps is not None else _DEFAULT_NUM_WARPS,
        enable_fp_fusion=False,
        launch_pdl=bool(enable_pdl),
        **extra,
    )
    return out
