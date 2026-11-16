# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""q/k/v RMSNorm + RoPE + FP8 K/V in one kernel over the packed projection output.

Reads the bf16 `[q | k | v]` (sliding-window layers) or `[q | k]` (full-attention layers, K = V:
V is the raw K block) projection output and emits BF16 Q and FP8-e4m3 K/V: q_norm and k_norm
(weighted) and v_norm (unweighted, on the raw values) RMSNorms, NeoX RoPE on Q and K from the
`[max_positions, 2, head_dim / 2]` table, FP8 casts. On an eager step it also stores K/V
straight into this layer's pages of the FlashInfer FP8 pool, at the slots and with the bytes
FlashInfer's append kernel writes, and returns `k = v = None` (the attention then appends
nothing). Under CUDA-graph capture, and on CUDA-graph metadata, it returns K/V for the append.

Fusion boundary: it replaces the three norms (FlashInfer's CuTe-DSL rmsnorm), FlashInfer's RoPE,
the FP8 casts and the KV append; Q/K/V are bitwise those of that chain (the sums of squares
replay the CuTe kernel's order, RoPE uses FlashInfer's fma; head_dim 256 or 512). `reference`
rotates with two rounded products, so Q and K are within one bf16 ulp of it.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttentionMetadata

from ._qkv_norm_rope_quant import gemma4_fused_qkv_norm_rope_quant as _kernel


def paged_store_target(md: FlashInferAttentionMetadata, layer_idx: int, num_rows: int, head_dim: int):
    """The paged-pool arguments of a direct K/V store for this layer, or None to return K/V.

    Eligible only on an eager step: never while capturing a CUDA graph and never for CUDA-graph
    metadata, so a captured body (and its replay) appends through the attention."""
    if (
        md is None
        or torch.cuda.is_current_stream_capturing()
        or md.is_cuda_graph
        or md.kv_cache_manager is None
        or num_rows != md.num_tokens
        or num_rows == 0
        or md.get_mixed_precision_kv_view(layer_idx) is not None
    ):
        return None
    kv_cache = md.kv_cache_manager.get_buffers(layer_idx, kv_layout=md.kv_layout)
    if (
        not isinstance(kv_cache, torch.Tensor)
        or kv_cache.dtype != torch.float8_e4m3fn
        or kv_cache.dim() != 5
        or kv_cache.shape[1] != 2
        or kv_cache.stride(-1) != 1
        or kv_cache.shape[-1] != head_dim
    ):
        return None
    target = dict(
        paged_kv_cache=kv_cache,
        kv_batch_indices=md.batch_indices,
        kv_positions=md.positions,
        kv_indptr=md.paged_kv_indptr,
        # This layer's pool (VSWA): the indices the attention swaps in before its append.
        kv_indices=md.get_paged_kv_indices_for_layer(layer_idx),
        kv_layout=md.kv_layout,
    )
    for name in ("kv_batch_indices", "kv_positions", "kv_indptr", "kv_indices"):
        t = target[name]
        if t.dtype != torch.int32 or not t.is_contiguous():
            return None
    return target


class QkvNormRopeQuant(OpWrapper):
    """`(q, k, v)`: q [T, Hq * D] bf16; k, v [T, Hkv * D] fp8, or None when stored in the pool."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(why="sliding-window layer, head 256, q32/kv16", spec=dict(t=300, hd=256, hq=32, hkv=16, k_eq_v=False)),
        Cell(why="full-attention layer, head 512, K = V", spec=dict(t=300, hd=512, hq=32, hkv=4, k_eq_v=True)),
        Cell(why="decode rows", spec=dict(t=4, hd=512, hq=32, hkv=4, k_eq_v=True)),
    )
    note = """
    Hard limits: head_dim 256 or 512 (the CuTe-DSL norm order it replays); bf16 projection
    output with unit inner stride; an fp32 contiguous RoPE table.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        eps: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        k_eq_v: bool,
        md: Optional[FlashInferAttentionMetadata],
        layer_idx: int,
    ):
        paged = paged_store_target(md, layer_idx, qkv.shape[0], head_dim)
        common = dict(
            out_fp8=False,
            packed_output=False,
            kv_out_fp8=True,
            cute_norm_order=True,
            k_eq_v=k_eq_v,
        )
        if paged is not None:
            q, _, _ = _kernel(
                qkv, positions, cos_sin, q_norm, k_norm, eps, num_heads, num_kv_heads, head_dim,
                **common, **paged,
            )
            return q, None, None
        return _kernel(
            qkv, positions, cos_sin, q_norm, k_norm, eps, num_heads, num_kv_heads, head_dim, **common
        )

    def reference(
        self, qkv, positions, cos_sin, q_norm, k_norm, eps, num_heads, num_kv_heads, head_dim,
        k_eq_v, md, layer_idx,
    ):
        """The unfused chain on the returned-K/V path (`md` and `layer_idx` are ignored): split,
        v_norm on the raw V, FlashInfer rmsnorm on q and k, NeoX RoPE, FP8 casts."""
        from flashinfer.norm import rmsnorm

        qw, kw = num_heads * head_dim, num_kv_heads * head_dim
        q = qkv[:, :qw].reshape(-1, head_dim)
        k = qkv[:, qw : qw + kw].reshape(-1, head_dim)
        v = k if k_eq_v else qkv[:, qw + kw : qw + 2 * kw].reshape(-1, head_dim)
        ones = torch.ones(head_dim, dtype=qkv.dtype, device=qkv.device)
        v = rmsnorm(v.contiguous(), ones, eps)
        q = rmsnorm(q.contiguous(), q_norm, eps).view(-1, num_heads, head_dim)
        k = rmsnorm(k.contiguous(), k_norm, eps).view(-1, num_kv_heads, head_dim)
        cs = cos_sin[positions.long()]
        cos = cs[:, 0, :].unsqueeze(1)
        sin = cs[:, 1, :].unsqueeze(1)

        def rope(x):
            x1, x2 = x.float().chunk(2, dim=-1)
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)

        q = rope(q).reshape(-1, qw)
        k = rope(k).reshape(-1, kw).to(torch.float8_e4m3fn)
        return q, k, v.reshape(-1, kw).to(torch.float8_e4m3fn)


qkv_norm_rope_quant = QkvNormRopeQuant()
