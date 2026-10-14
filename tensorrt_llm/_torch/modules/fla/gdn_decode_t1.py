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

from tensorrt_llm._torch.modules.fla.utils import custom_device_ctx

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
    # Pool layout [slots, HV, V, K] with K innermost (stride 1); the tile is
    # [BV, K] in the same order, so element [v, k] lives at v * K + k.
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
    tl.store(p_h0, b_h.to(h0_source.dtype.element_ty), mask=has_state)


def gdn_t1_decode_config(num_heads_total: int, V: int) -> Tuple[int, int]:
    """(BV, num_warps) for N * HV (sequence, value-head) pairs.

    Tuned on SM120 (RTX PRO 6000, 188 SMs) at HV=32, K=V=128: BV=8 keeps
    enough programs in flight at N <= 2; above that the kernel is
    state-bandwidth bound and BV=16 is marginally best. One warp per program
    keeps every reduction (q/k L2 norm, k- and q-contractions) in the same
    per-thread grouping and order as the generic single-warp kernel, so the
    results are bitwise identical to it; 2-4 warps were within ~2%.
    """
    bv = 8 if num_heads_total <= 64 else 16
    return min(bv, V), 1


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
    cfg_bv, cfg_warps = gdn_t1_decode_config(N * HV, V)
    BV = cfg_bv if BV is None else BV
    num_warps = cfg_warps if num_warps is None else num_warps
    assert V % BV == 0, f"V={V} must be a multiple of BV={BV}"

    s_h0_0, s_h0_1, s_h0_2, s_h0_3 = initial_state_source.stride()
    # Pool layout is [slots, HV, V, K] with K innermost (stride 1).
    assert s_h0_3 == 1, f"s_h0_3: {s_h0_3} is not 1"
    assert s_h0_2 == K, f"s_h0_2: {s_h0_2} is not {K}"
    assert s_h0_1 == V * K, f"s_h0_1: {s_h0_1} is not {V * K}"

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
            num_warps=num_warps,
            num_stages=1,
        )
    return o.squeeze(0)
