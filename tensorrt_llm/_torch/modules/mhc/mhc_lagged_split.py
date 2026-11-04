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

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber

_DEFAULT_NUM_WARPS = 8


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
):
    if LAUNCH_WITH_PDL:
        # First statement: no global access may be hoisted above the wait (memory clobber).
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    tok = tl.program_id(0).to(tl.int64)
    NCHUNK: tl.constexpr = HIDDEN // 8

    t = tl.arange(0, P_PAD)
    e = tl.arange(0, 8)
    obase = out_ptr + tok * HIDDEN
    xbase = x_ptr + tok * HIDDEN
    rbase = res_ptr + tok * (4 * HIDDEN)

    # ---- pass 1: post update + lagged collapse -> bf16 li (staged in out) ----
    for i in tl.static_range(NI):
        chunk = i * P + t
        cm = (t < P) & (chunk < NCHUNK)
        offs = chunk[:, None] * 8 + e[None, :]  # [P_PAD, 8]
        m2 = cm[:, None] & (e[None, :] < 8)
        x = tl.load(xbase + offs, mask=m2, other=0.0).to(tl.float32)
        r0 = tl.load(rbase + offs, mask=m2, other=0.0).to(tl.float32)
        r1 = tl.load(rbase + HIDDEN + offs, mask=m2, other=0.0).to(tl.float32)
        r2 = tl.load(rbase + 2 * HIDDEN + offs, mask=m2, other=0.0).to(tl.float32)
        r3 = tl.load(rbase + 3 * HIDDEN + offs, mask=m2, other=0.0).to(tl.float32)
        li = tl.zeros([P_PAD, 8], dtype=tl.float32)
        for j in tl.static_range(4):
            pj = tl.load(post_ptr + tok * 4 + j)
            # new_r[j] = post[j] * x, then fma(comb[k][j], r[k], .) for k = 0..3 (pmap order).
            nr = pj * x
            nr = tl.fma(tl.load(comb_ptr + tok * 16 + 0 * 4 + j), r0, nr)
            nr = tl.fma(tl.load(comb_ptr + tok * 16 + 1 * 4 + j), r1, nr)
            nr = tl.fma(tl.load(comb_ptr + tok * 16 + 2 * 4 + j), r2, nr)
            nr = tl.fma(tl.load(comb_ptr + tok * 16 + 3 * 4 + j), r3, nr)
            nr = nr.to(tl.bfloat16).to(tl.float32)
            # Lagged collapse: fma from 0 over j = 0..3 with the previous sublayer's pre.
            li = tl.fma(tl.load(pre_ptr + tok * 4 + j), nr, li)
        tl.store(obase + offs, li.to(tl.bfloat16), mask=m2)
    tl.debug_barrier()

    # ---- per-thread partials: chunks t, t + P, ...; pairs v = 0..3; fma(x, x, y * y) ----
    s = tl.zeros([P_PAD], dtype=tl.float32)
    for i in tl.static_range(NI):
        chunk = i * P + t
        cm = (t < P) & (chunk < NCHUNK)
        cbase = obase + chunk * 8
        for v in tl.static_range(4):
            lx = tl.load(cbase + 2 * v, mask=cm, other=0.0).to(tl.float32)
            ly = tl.load(cbase + 2 * v + 1, mask=cm, other=0.0).to(tl.float32)
            s = s + tl.fma(lx, lx, ly * ly)
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

    # ---- pass 2: normalize the staged li in place ----
    if DEBUG != 2:
        rstd2 = tl.reshape(rstd, [1, 1])
        for i in tl.static_range(NI):
            chunk = i * P + t
            cm = (t < P) & (chunk < NCHUNK)
            offs = chunk[:, None] * 8 + e[None, :]
            m2 = cm[:, None] & (e[None, :] < 8)
            li = tl.load(obase + offs, mask=m2, other=0.0).to(tl.float32)
            nw = tl.load(nw_ptr + offs, mask=m2, other=0.0).to(tl.float32)
            tl.store(obase + offs, ((li * rstd2) * nw).to(tl.bfloat16), mask=m2)


_SPLIT_ENV_ON = os.environ.get("TRTLLM_MHC_SPLIT_LAGGED", "1") != "0"
_SPLIT_MAX_TOKENS = 8
# Test-only race probe (default 0, never set in timed or serving runs): cycles of
# torch.cuda._sleep on the aux stream before every K2. Read at capture time.
TEST_AUX_SLEEP_CYCLES = int(os.environ.get("TRTLLM_MHC_SPLIT_TEST_SLEEP_CYCLES", "0"))
# Test-only: keep K2's (discarded) layer_input on the join object for comparison.
TEST_KEEP_K2_LAYER_INPUT = os.environ.get("TRTLLM_MHC_SPLIT_TEST_KEEP_K2", "0") == "1"


class MhcSplitJoin:
    """Pending K2 of one split boundary: ``join()`` makes the calling stream wait for it.

    Holds a reference to every K2 input until the join: the caller rebinds/drops them
    (``residual, ... = boundary(...)``), and the caching allocator would otherwise hand
    their blocks to later main-stream allocations while K2 still reads them.
    """

    __slots__ = ("event", "keep", "k2_layer_input")

    def __init__(self, event, keep, k2_layer_input=None):
        self.event = event
        self.keep = keep
        self.k2_layer_input = k2_layer_input

    def join(self):
        if self.event is not None:
            self.event.wait()
        self.event = None
        self.keep = None


def split_supported_here(num_tokens: int) -> bool:
    """Host-side gate: SM120, multi-stream (graph capture/warmup), 1 <= M <= 8, env on."""
    if not _SPLIT_ENV_ON or not (1 <= num_tokens <= _SPLIT_MAX_TOKENS):
        return False
    if os.environ.get("FORCE_DETERMINISTIC", "0") == "1":
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
    if os.environ.get("TRTLLM_ENABLE_PDL", "1") == "0":
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
) -> torch.Tensor:
    """K1: post update + lagged collapse + RMSNorm -> layer_input [M, hidden] bf16.

    Bitwise equal to the ``layer_input`` of ``mhc_fused_hc(..., pre_mix_ext=...)`` run with a
    tactic whose phase-2 thread count is ``reducer_threads`` (see module docstring).
    ``debug`` (test-only, fp32 [M, 2]) receives (sum of squares, rstd); with ``debug_li`` the
    output is the unnormalized collapse.
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
    scratch = torch.empty((M, SCR), dtype=torch.float32, device=x_prev.device)
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
    )
    return out
