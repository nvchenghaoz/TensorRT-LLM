# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-token (T == 1 per sequence) GDN decode kernel.

Specialization of ``fused_sigmoid_gating_delta_rule_update_kernel`` for pure
decode steps where every sequence contributes exactly one token. It computes
the same math in the same order (sigmoid/softplus gating, in-kernel q/k L2
norm, fp32 delta-rule update, state loaded from and stored to the pool exactly
once), but without the per-token time loop or the grid-stride loop, and with a
flat grid of ``N * HV * (V // BV)`` programs whose state tiles are laid out like
the pool (``[V, K]``, K innermost) so bf16 state I/O uses full-width vectors.
The launch shape is chosen per ``N * HV`` to keep enough programs in flight to
fill the GPU at small decode batches.
"""

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.gdn_state_q8 import (
    Q8_FORMAT,
    is_q8_state_pool,
    q8_load_rows,
    q8_state_rows,
    q8_store_rows,
)
from tensorrt_llm._torch.modules.fla.utils import custom_device_ctx


def _pool_row_stride(initial_state_source: torch.Tensor, K: int, V: int) -> int:
    """Asserts the [slots, HV, rows, K] pool layout; returns rows * K."""
    rows = q8_state_rows(V, K) if is_q8_state_pool(initial_state_source) else V
    s_h0_0, s_h0_1, s_h0_2, s_h0_3 = initial_state_source.stride()
    assert s_h0_3 == 1, f"s_h0_3: {s_h0_3} is not 1"
    assert s_h0_2 == K, f"s_h0_2: {s_h0_2} is not {K}"
    assert s_h0_1 == rows * K, f"s_h0_1: {s_h0_1} is not {rows * K}"
    if is_q8_state_pool(initial_state_source):
        assert initial_state_source.shape[2] == rows
        # fp32 row scales are read through the int8 base pointer.
        assert s_h0_0 % 16 == 0 and initial_state_source.data_ptr() % 16 == 0
    return s_h0_1

# Escape hatch for A/B comparison against the generic Triton kernel.
GDN_T1_DECODE_ENABLED = os.environ.get("TRTLLM_FLA_DISABLE_GDN_T1_DECODE",
                                       "0") != "1"


@triton.jit
def fused_sigmoid_gating_delta_rule_update_t1_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    scale,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    s_h0_0,
    h0_dim0,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    Q8: tl.constexpr = False,
    Q8_FMT: tl.constexpr = 0,
):
    NV: tl.constexpr = V // BV
    pid = tl.program_id(0)
    i_nh = pid // NV
    i_v = pid % NV
    # Pure decode: sequence i_n owns exactly token i_n.
    i_n = (i_nh // HV).to(tl.int64)
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    # The state load depends on the slot index: issue it first.
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    has_state = idx >= 0
    tl.device_assert(idx < h0_dim0, "idx out of bounds in h0_source access")
    if Q8:
        # 8-bit pool [slots, HV, V + 4, K]: V int8 rows + V fp32 row scales.
        p_head = h0_source + idx * s_h0_0 + i_hv * (V + (V * 4) // K) * K
        b_h = q8_load_rows(p_head, o_v, o_k, has_state, K, V, Q8_FMT)
    else:
        # Pool layout [slots, HV, V, K] with K innermost (stride 1); the tile
        # is [BV, K] in the same order, so element [v, k] lives at v * K + k.
        p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_v[:, None] * K +
                o_k[None, :])
        b_h = tl.load(p_h0, mask=has_state, other=0).to(tl.float32)

    b_q = tl.load(q + i_n * stride_q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_n * stride_k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_n * stride_v + i_hv * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_n * stride_b + i_hv).to(tl.float32)
    b_a = tl.load(a + i_n * stride_a + i_hv).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    # g = -exp(A_log) * softplus(a + dt_bias)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x

    # beta = sigmoid(b)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
    b_q = b_q * scale

    # h *= exp(g)
    b_h *= tl.exp(b_g)
    # v -= sum_k(h * k)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    # v *= beta
    b_v *= b_beta
    # h += k (x) v
    b_h += b_k[None, :] * b_v[:, None]
    # o = sum_k(h * q)
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # o is allocated/kept contiguous by the wrapper: [1, N, HV, V].
    tl.store(o + (i_n * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty))
    if Q8:
        q8_store_rows(p_head, o_v, o_k, b_h, has_state, K, V, Q8_FMT)
    else:
        tl.store(p_h0, b_h.to(h0_source.dtype.element_ty), mask=has_state)


@triton.jit
def _warp32x4_sum_sq(x):
    """sum(x * x) over a 1-D [128] fp32 vector in exactly the order Triton
    3.8 uses for tl.sum over a single-warp blocked layout with 4 contiguous
    elements per lane (sizePerThread [4], threadsPerWarp [32]): per lane the
    packed tree (x0^2 + x2^2) + (x1^2 + x3^2) (mul.f32 x4, add.f32x2,
    add.f32), then an xor butterfly over the 32 lanes (offsets 16, 8, 4, 2,
    1). This is the order of the q/k L2 norms in the T == 1 kernel and of the
    sum of squares in _rms_norm_gated_fwd_multirow_kernel ([1, 4] x [1, 32]).
    Element extraction is a masked sum with zeros (exact), every butterfly
    level is one add per pair, and squares of bf16-valued x are exact in
    fp32 (so FMA contraction cannot change them): the result is bitwise
    independent of the layout Triton picks for this multi-warp program."""
    xr = tl.reshape(x, (32, 4))
    cols = tl.arange(0, 4)[None, :]
    e0 = tl.sum(tl.where(cols == 0, xr, 0.0), axis=1)
    e1 = tl.sum(tl.where(cols == 1, xr, 0.0), axis=1)
    e2 = tl.sum(tl.where(cols == 2, xr, 0.0), axis=1)
    e3 = tl.sum(tl.where(cols == 3, xr, 0.0), axis=1)
    s = (e0 * e0 + e2 * e2) + (e1 * e1 + e3 * e3)
    s = tl.sum(tl.reshape(s, (2, 16)), axis=0)
    s = tl.sum(tl.reshape(s, (2, 8)), axis=0)
    s = tl.sum(tl.reshape(s, (2, 4)), axis=0)
    s = tl.sum(tl.reshape(s, (2, 2)), axis=0)
    return tl.sum(s, axis=0)


@triton.jit
def fused_sigmoid_gating_delta_rule_update_t1_norm_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    z,
    norm_w,
    y,
    o_dbg,
    h0_source,
    h0_indices,
    scale,
    eps,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_z,
    s_h0_0,
    h0_dim0,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STORE_DEBUG: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    r_dbg=None,
    Q8: tl.constexpr = False,
    Q8_FMT: tl.constexpr = 0,
):
    """T == 1 decode with the GDN gated RMSNorm (silu gate, norm before gate)
    folded in: one program (4 or 8 warps) per (sequence, value head) owns the whole
    [V, K] state, so the norm over V needs no cross-program step. Bitwise
    equal to fused_sigmoid_gating_delta_rule_update_t1_kernel followed by
    _rms_norm_gated_fwd_multirow_kernel: the state tile keeps the T == 1
    kernel's per-warp [1, 8] x [2, 16] grouping over K (warps split V only),
    so the K contractions are unchanged; the q/k L2 norms and the norm's sum
    of squares reproduce those kernels' single-warp reduce order with
    _warp32x4_sum_sq; o is rounded to the output dtype before the norm, as
    the stored o would be, and the norm's elementwise ops are the multirow
    kernel's in the same order. STORE_DEBUG also stores o and rstd."""
    tl.static_assert(K == 128 and V == 128)
    pid = tl.program_id(0)
    i_n = (pid // HV).to(tl.int64)
    i_hv = pid % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    idx = tl.load(h0_indices + i_n).to(tl.int64)
    has_state = idx >= 0
    tl.device_assert(idx < h0_dim0, "idx out of bounds in h0_source access")
    if Q8:
        p_head = h0_source + idx * s_h0_0 + i_hv * (V + (V * 4) // K) * K
        b_h = q8_load_rows(p_head, o_v, o_k, has_state, K, V, Q8_FMT)
    else:
        p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_v[:, None] * K +
                o_k[None, :])
        b_h = tl.load(p_h0, mask=has_state, other=0).to(tl.float32)

    b_q = tl.load(q + i_n * stride_q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_n * stride_k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_n * stride_v + i_hv * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_n * stride_b + i_hv).to(tl.float32)
    b_a = tl.load(a + i_n * stride_a + i_hv).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    b_z = tl.load(z + i_n * stride_z + i_hv * V + o_v).to(tl.float32)
    b_w = tl.load(norm_w + o_v).to(tl.float32)

    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / (tl.sqrt(_warp32x4_sum_sq(b_q)) + 1e-6)
        b_k = b_k / (tl.sqrt(_warp32x4_sum_sq(b_k)) + 1e-6)
    b_q = b_q * scale

    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_k[None, :] * b_v[:, None]
    b_o = tl.sum(b_h * b_q[None, :], 1)

    if Q8:
        q8_store_rows(p_head, o_v, o_k, b_h, has_state, K, V, Q8_FMT)
    else:
        tl.store(p_h0, b_h.to(h0_source.dtype.element_ty), mask=has_state)
    b_o = b_o.to(y.dtype.element_ty)
    if STORE_DEBUG:
        tl.store(o_dbg + (i_n * HV + i_hv) * V + o_v, b_o)
    # Gated RMSNorm (norm before gate, silu gate), as the multirow kernel.
    xo = b_o.to(tl.float32)
    var = _warp32x4_sum_sq(xo) / V
    rstd = 1.0 / tl.sqrt(var + eps)
    if STORE_DEBUG:
        tl.store(r_dbg + i_n * HV + i_hv, rstd)
    b_y = xo * rstd * b_w
    gate = tl.sigmoid(b_z)
    gate *= b_z
    b_y *= gate
    tl.store(y + (i_n * HV + i_hv) * V + o_v, b_y.to(y.dtype.element_ty))


def fused_sigmoid_gating_delta_rule_update_t1_norm(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor] = None,
    o_debug: Optional[torch.Tensor] = None,
    rstd_debug: Optional[torch.Tensor] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    """T == 1 decode + GDN gated RMSNorm (silu gate); returns y [N * HV, V].

    z is the [N, HV, V] gate view (V contiguous, heads packed per token, any
    token stride). Bitwise equal to fused_sigmoid_gating_delta_rule_update_t1
    followed by rms_norm_gated_token_major(o, z, norm_weight, eps) with a
    bf16 output (see the kernel docstring). Requires K == V == 128."""
    _, N, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    s_h0_0 = initial_state_source.stride(0)
    _pool_row_stride(initial_state_source, K, V)
    q8 = is_q8_state_pool(initial_state_source)
    assert z.stride(-1) == 1 and z.stride(-2) == V
    norm_weight = norm_weight.contiguous()
    y = output if output is not None else q.new_empty(N * HV, V)
    debug = o_debug is not None
    if num_warps is None:
        # SM120, HV=48, 48-layer cold-state chain (us/layer vs T1 + norm):
        # 8 warps is best at N <= 6, 4 warps at N = 8/16/32.
        num_warps = 4 if N >= 8 else 8
    grid = (N * HV, )
    with custom_device_ctx(q.device.index):
        fused_sigmoid_gating_delta_rule_update_t1_norm_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            z=z,
            norm_w=norm_weight,
            y=y,
            o_dbg=o_debug if o_debug is not None else y,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            scale=scale,
            eps=eps,
            stride_q=q.stride(1),
            stride_k=k.stride(1),
            stride_v=v.stride(1),
            stride_a=a.stride(-2),
            stride_b=b.stride(-2),
            stride_z=z.stride(0),
            s_h0_0=s_h0_0,
            h0_dim0=initial_state_source.shape[0],
            H=H,
            HV=HV,
            K=K,
            V=V,
            STORE_DEBUG=debug,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            r_dbg=rstd_debug if debug else None,
            Q8=q8,
            Q8_FMT=Q8_FORMAT,
            num_warps=num_warps,
            num_stages=1,
        )
    return y


def gdn_t1_decode_config(num_heads_total: int, V: int) -> Tuple[int, int]:
    """(BV, num_warps) for N * HV (sequence, value-head) pairs.

    Tuned on SM120 (RTX PRO 6000, 188 SMs) at HV=32 and HV=48, K=V=128:
    BV=8 keeps enough programs in flight up to N * HV = 96 (HV=48: N=1
    3.28 us, N=2 5.17 us vs 5.30 us at BV=16); above that the kernel is
    state-bandwidth bound and BV=16 is marginally best (HV=48, N=4: 9.26 us).
    One warp per program keeps every reduction (q/k L2 norm, k- and
    q-contractions) in the same per-thread grouping and order as the generic
    single-warp kernel, so the results are bitwise identical to it; 2-8 warps
    were within ~2-5%.
    """
    bv = 8 if num_heads_total <= 96 else 16
    return min(bv, V), 1


# (BV, num_warps) of the T == 1 kernel on the 8-bit state pool, overridable
# for sweeps as "BV,WARPS" (unset: same as the bf16 pool).
_Q8_T1_CONFIG = os.environ.get("TRTLLM_GDN_T1_Q8_CONFIG", "")


def gdn_t1_q8_decode_config(num_heads_total: int, V: int) -> Tuple[int, int]:
    """(BV, num_warps) for the 8-bit state pool."""
    if _Q8_T1_CONFIG:
        bv, warps = (int(x) for x in _Q8_T1_CONFIG.split(","))
        return min(bv, V), warps
    return gdn_t1_decode_config(num_heads_total, V)


def can_use_gdn_t1_decode(
    initial_state_source: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    B: int,
    T: int,
    N: int,
    K: int,
    V: int,
) -> bool:
    """Pure decode with a state pool: one token per sequence, packed varlen."""
    if not GDN_T1_DECODE_ENABLED:
        return False
    if cu_seqlens is None or initial_state_source is None:
        return False
    # T is the flattened token total of the packed [1, T, ...] varlen layout,
    # so T == N means exactly one token per sequence (decode never schedules
    # empty sequences), i.e. cu_seqlens == arange(N + 1).
    if B != 1 or N == 0 or T != N:
        return False
    if K != triton.next_power_of_2(K) or K > 256:
        return False
    if V != triton.next_power_of_2(V) or V < 8:
        return False
    return True


def fused_sigmoid_gating_delta_rule_update_t1(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor] = None,
    BV: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    """T == 1 decode launch; returns ``[1, N, HV, V]`` like the generic path."""
    _, N, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    q8 = is_q8_state_pool(initial_state_source)
    if q8:
        cfg_bv, cfg_warps = gdn_t1_q8_decode_config(N * HV, V)
    else:
        cfg_bv, cfg_warps = gdn_t1_decode_config(N * HV, V)
    BV = cfg_bv if BV is None else BV
    num_warps = cfg_warps if num_warps is None else num_warps
    assert V % BV == 0, f"V={V} must be a multiple of BV={BV}"

    s_h0_0 = initial_state_source.stride(0)
    # Pool layout is [slots, HV, V, K] (8-bit: [slots, HV, V + 4, K]) with K
    # innermost (stride 1).
    _pool_row_stride(initial_state_source, K, V)

    o = output.unsqueeze(0) if output is not None else q.new_empty(1, *v.shape)
    grid = (N * HV * (V // BV), )
    with custom_device_ctx(q.device.index):
        fused_sigmoid_gating_delta_rule_update_t1_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            scale=scale,
            stride_q=q.stride(1),
            stride_k=k.stride(1),
            stride_v=v.stride(1),
            stride_a=a.stride(-2),
            stride_b=b.stride(-2),
            s_h0_0=s_h0_0,
            h0_dim0=initial_state_source.shape[0],
            H=H,
            HV=HV,
            K=K,
            V=V,
            BV=BV,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            Q8=q8,
            Q8_FMT=Q8_FORMAT,
            num_warps=num_warps,
            num_stages=1,
        )
    return o.squeeze(0)
