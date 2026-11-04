# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 fused MXFP8 activation quantization + low-M MXFP8 GEMM for the CSA2 decode input projections.

``torch.ops.trtllm.csa2_mxfp8_low_m_gemm`` is the SM120 port of the MiniMax-M3 ``mxfp8LowMGemm`` kernel
(cpp/tensorrt_llm/kernels/minimaxM3/mxfp8LowMGemm.cu). Each CTA bulk-copies its weight slice into shared memory
before ``griddepcontrol.wait``, quantizes its activation K slice in-kernel (byte-equal to
``trtllm::mxfp8_quantize(x, True)``) and runs m16n8k32 e4m3 MMAs with fixed-order warp + cluster (DSMEM) split-K
reduction. It replaces ``mxfp8_quantize`` + ``flashinfer_mm_mxfp8`` (b12x) at decode M; the product is the same
math in another summation order (reordered output class).

Only the (N, K) projections where the port measured faster than quantize + b12x are routed here; everything else
returns ``None`` and the caller keeps the existing path. ``TRTLLM_LOW_M_MXFP8=0`` disables it.
Every configuration of a projection's table is launched once eagerly on first use outside stream capture, so each
kernel instance has its shared-memory attributes set before any CUDA graph is captured.
"""

from __future__ import annotations

import os

import torch

MAX_M = 8

# (N, K) -> configuration per row count M: (warps, cluster_k, row_tiles). Measured on RTX PRO 6000 (SM120,
# 188 SMs), cold-L2 PDL CUDA-graph chains (experiments/ascent/csa2_lowm/lowm_bench.py, jobs 835/925).
# The K=1280 projections (wq_b 8192, wq_b||index_wq_b 12288, index_wq_b 4096) stay on quantize + b12x: there the
# port measured equal or slower (e.g. 8192 x 1280: 10.9 vs 10.3 us), because its stage-everything-then-compute
# CTA exposes ~3 us of fixed latency that b12x's pipelined mainloop hides.
_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    # wq_a || wkv: 13.2 -> 10.5 us at M=1, 13.9 -> 10.6 at M=2, 14.0 -> 12.3 at M=4, 14.0 -> 13.4 at M=8.
    (1792, 5120): {1: (8, 2, 2), 2: (8, 2, 2), 3: (8, 4, 1), 4: (8, 4, 1), 5: (8, 4, 1), 6: (8, 4, 1),
                   7: (8, 4, 1), 8: (8, 4, 1)},
}

_state: dict = {}


def _available(device: torch.device) -> bool:
    ok = _state.get(device.index)
    if ok is None:
        ok = (
            os.environ.get("TRTLLM_LOW_M_MXFP8", "1") != "0"
            and torch.cuda.get_device_capability(device) == (12, 0)
            and hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm")
        )
        _state[device.index] = ok
    return ok


def config_for(n: int, k: int, m: int):
    table = _CONFIGS.get((n, k))
    return None if table is None else table.get(m)


def _warmup(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    """Launch every configuration of this (N, K) table once, eagerly, before any graph capture."""
    n, k = weight.shape
    key = (x.device.index, n, k)
    if key in _warmed or torch.cuda.is_current_stream_capturing():
        return
    for m, cfg in sorted(_CONFIGS[(n, k)].items()):
        xw = torch.zeros((m, k), dtype=torch.bfloat16, device=x.device)
        torch.ops.trtllm.csa2_mxfp8_low_m_gemm(xw, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True)
    _warmed.add(key)


_warmed: set = set()


def low_m_mm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
    """``mxfp8(x) @ weight^T`` in BF16 through the fused kernel, or ``None`` when it does not apply."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        return None
    m, k = x.shape
    if m == 0 or m > MAX_M or weight.dim() != 2 or weight.shape[1] != k or weight_scale.dim() != 1:
        return None
    cfg = config_for(weight.shape[0], k, m)
    if cfg is None or not _available(x.device):
        return None
    _warmup(x, weight, weight_scale)
    if x.stride(1) != 1 or x.stride(0) % 8 != 0 or x.data_ptr() % 16 != 0:
        x = x.contiguous()
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True)[0]


if hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm"):

    @torch.library.register_fake("trtllm::csa2_mxfp8_low_m_gemm")
    def _(x, w, wsf, warps, cluster_k, row_tiles, n_split, pdl, xq=None, xsf=None, xq_out=None, xsf_out=None,
          dbg_ts=None, stages=1, debug_flags=0):
        n = w.shape[0]
        if n_split == 0:
            return [x.new_empty((x.shape[0], n))]
        return [x.new_empty((x.shape[0], n_split)), x.new_empty((x.shape[0], n - n_split))]
