# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the FP8 attention entries: the fakes of the C++ operators
(``thop/flux3/flux3FmhaOp.cpp``), the Q/K/V plane views and the plain-torch emulation of the
kernel's flash recurrence that the entries' ``reference`` use.

Imported by the entries, so the fakes exist before any of the operators is traced.
"""

from __future__ import annotations

import math

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul

HEAD_DIM = 128
#: Keys per KV step of the kernel (its running max and rescale happen per step).
KV_STEP = 64
#: Q rows per CTA: a Q-row prefix launch writes rows [0, round_up(q_rows, CTA_ROWS)).
CTA_ROWS = 192
#: The kernel quantizes P as e4m3(256 * exp(s * (x - max))).
_P_QUANT_SCALE = 256.0
_LOG2E = 1.4426950408889634


def nvfp4_scale_bytes(rows: int, width: int) -> int:
    """Bytes of the flat 128x4-swizzled E4M3 scale buffer of an [rows, width] NVFP4 tensor."""
    return (rows + 127) // 128 * 128 * ((width // 16 + 3) // 4 * 4)


@torch.library.register_fake("trtllm::flux3_fp8_attention_bf16_out")
def _bf16_out_fake(q, k, v, scale_bmm1, scale_bmm2):
    return q.new_empty(q.shape, dtype=torch.bfloat16)


def _nvfp4_out_shapes(q):
    b, s, h, d = q.shape
    return [
        q.new_empty((b, s, h * d // 2), dtype=torch.uint8),
        q.new_empty((nvfp4_scale_bytes(b * s, h * d),), dtype=torch.uint8),
    ]


@torch.library.register_fake("trtllm::flux3_fp8_attention_nvfp4_out")
def _nvfp4_out_fake(q, k, v, scale_bmm1, scale_bmm2, global_scale):
    return _nvfp4_out_shapes(q)


@torch.library.register_fake("trtllm::flux3_fp8_attention_nvfp4_out_q_rows")
def _nvfp4_out_q_rows_fake(q, k, v, scale_bmm1, scale_bmm2, global_scale, q_rows):
    return _nvfp4_out_shapes(q)


def planes(qkv8: torch.Tensor, batch: int, heads: int):
    """(q8, k8, v8) ``[B, S, H, 128]`` views of the contiguous ``[3, B*S, H*128]`` planes."""
    _, rows, width = qkv8.shape
    seq_len = rows // batch
    head_dim = width // heads
    return tuple(plane.view(batch, seq_len, heads, head_dim) for plane in qkv8.unbind(0))


def emulated_attention(
    qkv8: torch.Tensor, scales: torch.Tensor, batch: int, heads: int, q_rows: int = 0
) -> torch.Tensor:
    """BF16 O ``[B, S, H, 128]`` by the kernel's recurrence in plain torch, FP32 throughout.

    Per 64-key step: ``x = q8 @ k8^T`` (raw, FP32), ``m' = max(m, rowmax x)``,
    ``p = exp2(x * c + (8 - m' * c))`` with ``c = scales[3] * log2(e)`` (= 256 exp(s (x - m'))),
    ``alpha = exp2((m - m') * c)``, ``l = l * alpha + sum p``,
    ``acc = acc * alpha + e4m3(p) @ v8``; then ``O = bf16(acc * (1 / l) * scales[2])``.

    It differs from the kernel only in rounding: ``ex2.approx`` vs torch's exp2, the kernel's
    fused multiply-adds, and the MMA's summation order -- and through them, rarely, one P byte
    rounding to the other E4M3 neighbour. q_rows > 0 computes only rows [0, q_rows) (the rest are
    zero).
    """
    q8, k8, v8 = planes(qkv8, batch, heads)
    seq_len = q8.shape[1]
    n = seq_len if q_rows <= 0 else min(q_rows, seq_len)
    c = scales[3].float() * _LOG2E
    log2_quant = math.log2(_P_QUANT_SCALE)
    out = torch.zeros(q8.shape, dtype=torch.bfloat16, device=q8.device)
    with true_fp32_matmul():
        for b in range(batch):
            q = q8[b, :n].float().transpose(0, 1)  # [H, n, D]
            k = k8[b].float().transpose(0, 1)  # [H, S, D]
            v = v8[b].float().transpose(0, 1)
            m = torch.full((heads, n, 1), -math.inf, device=q.device)
            row_sum = torch.zeros((heads, n, 1), device=q.device)
            acc = torch.zeros((heads, n, HEAD_DIM), device=q.device)
            for j0 in range(0, seq_len, KV_STEP):
                x = q @ k[:, j0 : j0 + KV_STEP].transpose(1, 2)
                m_new = torch.maximum(m, x.amax(-1, keepdim=True))
                p = torch.exp2(x * c + (log2_quant - m_new * c))
                alpha = torch.exp2((m - m_new) * c)
                row_sum = row_sum * alpha + p.sum(-1, keepdim=True)
                p8 = p.to(torch.float8_e4m3fn).float()
                acc = acc * alpha + p8 @ v[:, j0 : j0 + KV_STEP]
                m = m_new
                del x, p, p8
            o = acc * (1.0 / row_sum) * scales[2].float()
            out[b, :n] = o.transpose(0, 1).to(torch.bfloat16)
    return out
