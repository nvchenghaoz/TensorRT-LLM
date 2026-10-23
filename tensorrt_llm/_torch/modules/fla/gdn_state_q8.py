# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""8-bit GDN recurrent-state pool (``kv_cache_config.mamba_ssm_cache_dtype="int8"``).

Pool layout per slot: int8 ``[HV, V + Q8_SCALE_ROWS, K]`` (K innermost). For
each value head, rows ``0..V-1`` hold the 8-bit state (one row = the K
elements of one value row ``h[v, :]``) and the ``Q8_SCALE_ROWS * K`` bytes
after them hold the ``V`` fp32 row scales (``V * 4 == Q8_SCALE_ROWS * K`` for
``K == V == 128``). ``state[v, k] ~= q[v, k] * scale[v]``.

Formats (one switch, ``Q8_FORMAT``):
  * ``0`` int8-row: ``scale = amax / 127``, ``q = clamp(rint(x * 127 / amax), +-127)``.
  * ``1`` e4m3-row: ``scale = amax / 448``, ``q = e4m3_rn_satfinite(x * 448 / amax)``
    (bit pattern stored in the int8 pool).
A zero row stores ``scale = 0`` and ``q = 0``, so the zeroed pool (fresh slot
reset) dequantizes to an exact zero state.

The quantize/dequantize helpers are ``@triton.jit`` functions shared by the
T == 1 decode kernels (gdn_decode_t1.py) and the dequant-gather /
quant-scatter that bridges the pool and FlashInfer's fp32 prefill state.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Pool storage dtype selecting the 8-bit state.
Q8_POOL_DTYPE = torch.int8
# Extra K-wide rows per value head holding the V fp32 row scales.
Q8_SCALE_ROWS = 4
# 0: int8-row (default), 1: e4m3-row.
Q8_FORMAT = 0


def is_q8_state_pool(state: Optional[torch.Tensor]) -> bool:
    return state is not None and state.dtype == Q8_POOL_DTYPE


def q8_state_rows(v: int, k: int) -> int:
    """Rows of K bytes per value head: V state rows + the fp32 row scales."""
    assert (v * 4) % k == 0, f"V={v} fp32 scales must fill whole K={k} rows"
    return v + (v * 4) // k


@triton.jit
def q8_load_rows(p_head, o_v, o_k, mask, K: tl.constexpr, V: tl.constexpr,
                 FMT: tl.constexpr):
    """Dequantize rows ``o_v`` of one head's 8-bit state to fp32 [len(o_v), K].

    ``p_head`` points at the head's ``[V + scale rows, K]`` int8 block."""
    q = tl.load(p_head + o_v[:, None] * K + o_k[None, :], mask=mask, other=0)
    p_s = (p_head + V * K).to(tl.pointer_type(tl.float32))
    s = tl.load(p_s + o_v, mask=mask, other=0.0)
    if FMT == 0:
        x = q.to(tl.float32)
    else:
        x = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return x * s[:, None]


@triton.jit
def q8_store_rows(p_head, o_v, o_k, h, mask, K: tl.constexpr, V: tl.constexpr,
                  FMT: tl.constexpr):
    """Quantize fp32 rows ``h`` [len(o_v), K] (one scale per row) and store
    the 8-bit rows and their fp32 scales into one head's block."""
    amax = tl.max(tl.abs(h), axis=1)
    if FMT == 0:
        qmax = 127.0
    else:
        qmax = 448.0
    nz = amax > 0.0
    inv = tl.where(nz, qmax / tl.where(nz, amax, 1.0), 0.0)
    s = amax / qmax
    y = h * inv[:, None]
    if FMT == 0:
        y = libdevice.rint(y)
        y = tl.minimum(tl.maximum(y, -127.0), 127.0)
        q = y.to(tl.int8)
    else:
        q = y.to(tl.float8e4nv).to(tl.int8, bitcast=True)
    tl.store(p_head + o_v[:, None] * K + o_k[None, :], q, mask=mask)
    p_s = (p_head + V * K).to(tl.pointer_type(tl.float32))
    tl.store(p_s + o_v, s, mask=mask)


@triton.jit
def _q8_gather_dequant_kernel(
    pool,
    out,
    indices,
    s_pool_n,
    s_out_n,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    FMT: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_vb = tl.program_id(2)
    ROWS: tl.constexpr = V + (V * 4) // K
    slot = tl.load(indices + i_n).to(tl.int64)
    valid = slot >= 0
    o_v = i_vb * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    p_head = pool + slot * s_pool_n + i_h * ROWS * K
    x = q8_load_rows(p_head, o_v, o_k, valid, K, V, FMT)
    p_out = (out + i_n.to(tl.int64) * s_out_n + i_h * V * K +
             o_v[:, None] * K + o_k[None, :])
    tl.store(p_out, x)


@triton.jit
def _q8_quant_scatter_kernel(
    src,
    pool,
    indices,
    s_src_n,
    s_pool_n,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    FMT: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_vb = tl.program_id(2)
    ROWS: tl.constexpr = V + (V * 4) // K
    slot = tl.load(indices + i_n).to(tl.int64)
    valid = slot >= 0
    o_v = i_vb * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    x = tl.load(src + i_n.to(tl.int64) * s_src_n + i_h * V * K +
                o_v[:, None] * K + o_k[None, :])
    p_head = pool + slot * s_pool_n + i_h * ROWS * K
    q8_store_rows(p_head, o_v, o_k, x, valid, K, V, FMT)


def _check_pool(pool: torch.Tensor, v: int, k: int) -> None:
    assert is_q8_state_pool(pool), pool.dtype
    _, _, rows, kk = pool.shape
    assert kk == k and rows == q8_state_rows(v, k), (tuple(pool.shape), v, k)
    assert pool.stride()[1:] == (rows * k, k, 1), pool.stride()
    assert pool.stride(0) % 4 == 0 and pool.data_ptr() % 16 == 0


def q8_gather_dequant(pool: torch.Tensor,
                      indices: torch.Tensor,
                      v: int,
                      out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fp32 ``[len(indices), HV, V, K]`` = dequantized ``pool[indices]``
    (a negative index yields a zero state)."""
    hv, k = pool.shape[1], pool.shape[3]
    _check_pool(pool, v, k)
    n = indices.shape[0]
    if out is None:
        out = torch.empty(n, hv, v, k, dtype=torch.float32, device=pool.device)
    assert out.shape == (n, hv, v, k) and out.dtype == torch.float32
    assert out.stride()[1:] == (v * k, k, 1)
    if n == 0:
        return out
    bv = 32
    _q8_gather_dequant_kernel[(n, hv, triton.cdiv(v, bv))](
        pool, out, indices, pool.stride(0), out.stride(0),
        HV=hv, V=v, K=k, BV=bv, FMT=Q8_FORMAT, num_warps=4)
    return out


def q8_quant_scatter(src: torch.Tensor, pool: torch.Tensor,
                     indices: torch.Tensor) -> None:
    """``pool[indices] = quantize(src)`` for fp32 ``src`` ``[n, HV, V, K]``
    (negative indices are skipped)."""
    n, hv, v, k = src.shape
    _check_pool(pool, v, k)
    assert src.dtype == torch.float32 and src.stride()[1:] == (v * k, k, 1)
    assert indices.shape == (n, )
    if n == 0:
        return
    bv = 32
    _q8_quant_scatter_kernel[(n, hv, triton.cdiv(v, bv))](
        src, pool, indices, src.stride(0), pool.stride(0),
        HV=hv, V=v, K=k, BV=bv, FMT=Q8_FORMAT, num_warps=4)
