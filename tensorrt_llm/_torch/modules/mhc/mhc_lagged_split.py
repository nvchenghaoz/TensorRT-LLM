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

K1 reproduces the CUDA kernels' arithmetic bit for bit, including the RMSNorm sum-of-squares
reduction order, which depends on the tactic's thread count ``P``:
  * ``fused_half_*`` (mhcBigFuseKernel Phase 2): ``P = bigfuse_bs - 32`` threads,
  * ``fused_all_fma`` (all-in-one Phase 4):       ``P = 256`` threads.
Thread ``t`` sums the 8-element chunks ``t, t + P, ...`` sequentially, each pair as
``fma(x, x, y * y)``; then a warp xor-butterfly (16, 8, 4, 2, 1) and a sequential sum over
warps from 0.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber

_VEC = 8  # bf16 elements per thread-chunk (uint4) in the CUDA kernels


@triton.jit
def _extract_col(t, idx, q):
    # Exactly t[:, q]: one nonzero term plus +0.0s, so the sum is exact in any order.
    return tl.sum(tl.where(idx[None, :] == q, t, 0.0), axis=1)


@triton.jit
def _mhc_lagged_layer_input_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    nw_ptr,
    out_ptr,
    norm_eps,
    HIDDEN: tl.constexpr,
    P: tl.constexpr,
    P_PAD: tl.constexpr,
    NI: tl.constexpr,
    NI_PAD: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    SQ_MODE: tl.constexpr,
    DIV_RN: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    if LAUNCH_WITH_PDL:
        # First statement: no global access may be hoisted above the wait (memory clobber).
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    tok = tl.program_id(0).to(tl.int64)
    NCHUNK: tl.constexpr = HIDDEN // 8
    Q: tl.constexpr = NI_PAD * 4

    t = tl.arange(0, P_PAD)
    it = tl.arange(0, NI_PAD)
    e = tl.arange(0, 8)
    chunk = it[None, :] * P + t[:, None]  # [P_PAD, NI_PAD]
    cmask = (t[:, None] < P) & (it[None, :] < NI) & (chunk < NCHUNK)
    offs = chunk[:, :, None] * 8 + e[None, None, :]  # [P_PAD, NI_PAD, 8]
    mask = cmask[:, :, None] & (e[None, None, :] < 8)

    x = tl.load(x_ptr + tok * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
    rbase = res_ptr + tok * (4 * HIDDEN)
    r0 = tl.load(rbase + offs, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(rbase + HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(rbase + 2 * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(rbase + 3 * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)

    li = tl.zeros([P_PAD, NI_PAD, 8], dtype=tl.float32)
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
    li = li.to(tl.bfloat16).to(tl.float32)

    # Sum of squares in the CUDA kernels' order.
    li4 = tl.reshape(li, [P_PAD, Q, 2])
    bx, by = tl.split(li4)  # [P_PAD, Q]; q = it * 4 + v
    if SQ_MODE == 0:
        term = tl.fma(bx, bx, by * by)
    elif SQ_MODE == 1:
        term = tl.fma(by, by, bx * bx)
    else:
        term = bx * bx + by * by
    qidx = tl.arange(0, Q)
    s = tl.zeros([P_PAD], dtype=tl.float32)
    for q in tl.static_range(Q):
        s = s + _extract_col(term, qidx, q)
    # Warp xor-butterfly (16, 8, 4, 2, 1): lane 0's value. A size-2 sum is one add.
    s = tl.reshape(s, [W_PAD, 2, 16])
    s = tl.sum(s, axis=1)
    s = tl.reshape(s, [W_PAD, 2, 8])
    s = tl.sum(s, axis=1)
    s = tl.reshape(s, [W_PAD, 2, 4])
    s = tl.sum(s, axis=1)
    s = tl.reshape(s, [W_PAD, 2, 2])
    s = tl.sum(s, axis=1)
    s = tl.reshape(s, [W_PAD, 2, 1])
    s = tl.sum(s, axis=1)  # [W_PAD, 1]
    s = tl.reshape(s, [1, W_PAD])
    widx = tl.arange(0, W_PAD)
    total = tl.zeros([1], dtype=tl.float32)
    for w in tl.static_range(W):
        total = total + _extract_col(s, widx, w)
    # mhcKernels is built with --use_fast_math: `total / hidden` is div.full.f32 (approximate,
    # not IEEE div.rn), which is also what Triton's fp32 `/` emits; rsqrtf is rsqrt.approx.
    hidden_f = tl.full([1], HIDDEN, tl.float32)
    if DIV_RN:
        mean = tl.math.div_rn(total, hidden_f)
    else:
        mean = total / hidden_f
    rstd = tl.math.rsqrt(mean + norm_eps)  # [1]

    nw = tl.load(nw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd3 = tl.reshape(rstd, [1, 1, 1])
    out = (li * rstd3) * nw
    tl.store(out_ptr + tok * HIDDEN + offs, out.to(tl.bfloat16), mask=mask)


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
    if backend in ("fused_half_fma", "fused_half_mma"):
        if bigfuse_bs >= 512:
            return 480
        if bigfuse_bs >= 256:
            return 224
        if bigfuse_bs > 0:
            return 96
        return None  # C++ heuristic block size: not mirrored
    if backend == "fused_all_fma":
        return 256
    return None


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
    sq_mode: int = 0,
    div_rn: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """K1: post update + lagged collapse + RMSNorm -> layer_input [M, hidden] bf16.

    Bitwise equal to the ``layer_input`` of ``mhc_fused_hc(..., pre_mix_ext=...)`` run with a
    tactic whose phase-2 thread count is ``reducer_threads`` (see module docstring).
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
    if out is None:
        out = torch.empty_like(x_prev)
    if M == 0:
        return out
    _mhc_lagged_layer_input_kernel[(M,)](
        x_prev,
        residual_prev,
        post_mix_prev,
        comb_mix_prev,
        pre_mix_ext,
        norm_weight,
        out,
        float(norm_eps),
        HIDDEN=hidden,
        P=P,
        P_PAD=P_PAD,
        NI=NI,
        NI_PAD=triton.next_power_of_2(NI),
        W=W,
        W_PAD=P_PAD // 32,
        SQ_MODE=sq_mode,
        DIV_RN=bool(div_rn),
        LAUNCH_WITH_PDL=bool(enable_pdl),
        num_warps=16 if P_PAD >= 512 else 8,
        enable_fp_fusion=False,
        launch_pdl=bool(enable_pdl),
    )
    return out
