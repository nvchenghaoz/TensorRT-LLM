# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120/SM121 per-tensor FP8 (E4M3) attention for VisualGen as one in-graph custom op.

Q/K/V arrive as strided ``[B, S, H, D]`` views (rows of a packed QKV buffer, planes of a
planar buffer, or plain contiguous tensors). Two Triton launches replace the three
``.contiguous()`` copies plus three ``trtllm::quantize_e4m3_per_tensor`` calls (each a
memset, an amax kernel and a scale kernel) and the descale multiply:

* K1 reads Q, K and V straight from their strided views and writes one abs-max partial
  per program and tensor (``[3, n_prog]`` FP32, no memset, no atomics).
* K2 reduces the partials (max is order-free, so the amax is exact), computes the scale
  exactly as ``computeFP8QuantizeScalePerTensorVec`` does
  (``bf16(max(amax / 448, 1 / (448 * 512)))``) and the factor as
  ``scaleMatrixPerTensorVec`` does (``1 / float(scale)``, both IEEE round-to-nearest),
  writes contiguous E4M3 Q/K/V with ``cvt.rn.satfinite`` and, from program 0, the FP32
  device scalars ``[q_descale, k_descale, v_descale, (q_descale * k_descale) * sm_scale]``.

The result is bit-identical to ``FlashInferAttention._quantize_e4m3`` on contiguous copies.
The attention kernel itself is FlashInfer's fmha_v2 through the backend's
``run_fmha_v2_fp8_sm12x`` helper (looked up on the module at call time, so whatever that
helper routes is what launches).
"""

import numpy as np
import torch
import triton
import triton.language as tl

_E4M3_MAX = 448.0
# FP32 constant as the C++ kernel computes it: 1.0f / (448.f * 512.f).
_MIN_SCALING_FACTOR = float(np.float32(1.0) / (np.float32(448.0) * np.float32(512.0)))
_MAX_PARTIALS = 1024
_K1_WARPS = 4
_K2_ROWS = 4
_K2_WARPS = 4


@triton.jit
def _qkv_amax_partials_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    part_ptr,
    num_rows,
    seq_len,
    rows_per_prog,
    n_prog,
    q_bs,
    q_rs,
    k_bs,
    k_rs,
    v_bs,
    v_rs,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    acc_q = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_k = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_N], dtype=tl.float32)
    row0 = pid * rows_per_prog
    for i in range(rows_per_prog):
        r = row0 + i
        valid = r < num_rows
        b = (r // seq_len).to(tl.int64)
        s = (r % seq_len).to(tl.int64)
        q_off = b * q_bs + s * q_rs
        k_off = b * k_bs + s * k_rs
        v_off = b * v_bs + s * v_rs
        for c in tl.static_range(0, N, BLOCK_N):
            m = valid & ((cols + c) < N)
            xq = tl.load(q_ptr + q_off + c + cols, mask=m, other=0.0)
            xk = tl.load(k_ptr + k_off + c + cols, mask=m, other=0.0)
            xv = tl.load(v_ptr + v_off + c + cols, mask=m, other=0.0)
            acc_q = tl.maximum(acc_q, tl.abs(xq.to(tl.float32)))
            acc_k = tl.maximum(acc_k, tl.abs(xk.to(tl.float32)))
            acc_v = tl.maximum(acc_v, tl.abs(xv.to(tl.float32)))
    tl.store(part_ptr + pid, tl.max(acc_q, axis=0))
    tl.store(part_ptr + n_prog + pid, tl.max(acc_k, axis=0))
    tl.store(part_ptr + 2 * n_prog + pid, tl.max(acc_v, axis=0))


@triton.jit
def _scale_of(part_ptr, base, n_part, min_scale, BLOCK_P: tl.constexpr):
    p = tl.arange(0, BLOCK_P)
    amax = tl.max(tl.load(part_ptr + base + p, mask=p < n_part, other=0.0), axis=0)
    scale = tl.maximum(tl.math.div_rn(amax, tl.full([], 448.0, tl.float32)), min_scale)
    scale = scale.to(tl.bfloat16).to(tl.float32)
    factor = tl.math.div_rn(tl.full([], 1.0, tl.float32), scale)
    return scale, factor


@triton.jit
def _qkv_quantize_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q8_ptr,
    k8_ptr,
    v8_ptr,
    part_ptr,
    scales_ptr,
    n_part,
    num_rows,
    seq_len,
    q_bs,
    q_rs,
    k_bs,
    k_rs,
    v_bs,
    v_rs,
    sm_scale,
    min_scale,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    sq, fq = _scale_of(part_ptr, 0, n_part, min_scale, BLOCK_P)
    sk, fk = _scale_of(part_ptr, n_part, n_part, min_scale, BLOCK_P)
    sv, fv = _scale_of(part_ptr, 2 * n_part, n_part, min_scale, BLOCK_P)
    if pid == 0:
        tl.store(scales_ptr + 0, sq)
        tl.store(scales_ptr + 1, sk)
        tl.store(scales_ptr + 2, sv)
        tl.store(scales_ptr + 3, (sq * sk) * sm_scale)
    cols = tl.arange(0, BLOCK_N)
    for i in tl.static_range(ROWS):
        r = pid * ROWS + i
        valid = r < num_rows
        b = (r // seq_len).to(tl.int64)
        s = (r % seq_len).to(tl.int64)
        q_off = b * q_bs + s * q_rs
        k_off = b * k_bs + s * k_rs
        v_off = b * v_bs + s * v_rs
        o_off = r.to(tl.int64) * N
        for c in tl.static_range(0, N, BLOCK_N):
            m = valid & ((cols + c) < N)
            xq = tl.load(q_ptr + q_off + c + cols, mask=m, other=0.0)
            xk = tl.load(k_ptr + k_off + c + cols, mask=m, other=0.0)
            xv = tl.load(v_ptr + v_off + c + cols, mask=m, other=0.0)
            yq = (xq.to(tl.float32) * fq).to(tl.float8e4nv)
            yk = (xk.to(tl.float32) * fk).to(tl.float8e4nv)
            yv = (xv.to(tl.float32) * fv).to(tl.float8e4nv)
            tl.store(q8_ptr + o_off + c + cols, yq, mask=m)
            tl.store(k8_ptr + o_off + c + cols, yk, mask=m)
            tl.store(v8_ptr + o_off + c + cols, yv, mask=m)


def _row_strides(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Return (tensor, batch stride, row stride) for a [B, S, H, D] view with packed H*D rows."""
    _, _, h, d = x.shape
    if x.stride(3) != 1 or x.stride(2) != d:
        x = x.contiguous()
    return x, x.stride(0), x.stride(1)


def quantize_qkv_e4m3(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 Q/K/V from strided BF16 [B, S, H, D] views.

    Returns contiguous E4M3 q8/k8/v8 and FP32 scales
    ``[q_descale, k_descale, v_descale, q_descale * k_descale * sm_scale]``.
    """
    batch, seq_len, heads, head_dim = q.shape
    num_rows = batch * seq_len
    n = heads * head_dim
    q, q_bs, q_rs = _row_strides(q)
    k, k_bs, k_rs = _row_strides(k)
    v, v_bs, v_rs = _row_strides(v)
    out8 = torch.empty((3, batch, seq_len, heads, head_dim), dtype=torch.float8_e4m3fn,
                       device=q.device)
    scales = torch.empty(4, dtype=torch.float32, device=q.device)
    block_n = min(1024, triton.next_power_of_2(n))
    rows_per_prog = triton.cdiv(num_rows, _MAX_PARTIALS)
    n_prog = triton.cdiv(num_rows, rows_per_prog)
    partials = torch.empty(3 * n_prog, dtype=torch.float32, device=q.device)
    _qkv_amax_partials_kernel[(n_prog,)](
        q, k, v, partials, num_rows, seq_len, rows_per_prog, n_prog,
        q_bs, q_rs, k_bs, k_rs, v_bs, v_rs,
        N=n, BLOCK_N=block_n, num_warps=_K1_WARPS,
    )
    q8, k8, v8 = out8.unbind(0)
    _qkv_quantize_kernel[(triton.cdiv(num_rows, _K2_ROWS),)](
        q, k, v, q8, k8, v8, partials, scales, n_prog, num_rows, seq_len,
        q_bs, q_rs, k_bs, k_rs, v_bs, v_rs, sm_scale, _MIN_SCALING_FACTOR,
        N=n, BLOCK_N=block_n, BLOCK_P=triton.next_power_of_2(n_prog), ROWS=_K2_ROWS,
        num_warps=_K2_WARPS,
    )
    return q8, k8, v8, scales


@torch.library.custom_op("trtllm::visual_gen_fp8_attention_sm120", mutates_args=())
def visual_gen_fp8_attention_sm120(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, is_causal: bool
) -> torch.Tensor:
    """Per-tensor E4M3 Q/K/V quantize + SM120 fmha_v2; BF16 contiguous [B, S, H, D] out."""
    from . import flashinfer as _fi

    _, seq_len, heads, head_dim = q.shape
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, sm_scale)
    output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    _fi.run_fmha_v2_fp8_sm12x(
        q8,
        k8,
        v8,
        output,
        heads,
        head_dim,
        seq_len,
        scale_bmm1_d=scales[3:4],
        scale_bmm2_d=scales[2:3],
        causal=is_causal,
        return_lse=False,
    )
    return output


@visual_gen_fp8_attention_sm120.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, is_causal: bool):
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
