# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 fused MXFP8 activation quantization + low-M MXFP8 GEMM for the CSA2 decode input projections.

``native_ops.csa2_mxfp8_low_m_gemm`` is the SM120 port of the MiniMax-M3 ``mxfp8LowMGemm`` kernel
(cpp/tensorrt_llm/kernels/minimaxM3/mxfp8LowMGemm.cu). Each CTA bulk-copies its weight slice into shared memory
before ``griddepcontrol.wait``, quantizes its activation K slice in-kernel (byte-equal to
``trtllm::mxfp8_quantize(x, True)``) and runs m16n8k32 e4m3 MMAs with fixed-order warp + cluster (DSMEM) split-K
reduction. It replaces ``mxfp8_quantize`` + ``flashinfer_mm_mxfp8`` (b12x) at decode M; the product is the same
math in another summation order (reordered output class).

Only the (N, K) projections where the port measured faster than quantize + b12x are routed here; everything else
returns ``None`` and the caller keeps the existing path.
Every configuration of a projection's table is launched once eagerly on first use outside stream capture, so each
kernel instance has its shared-memory attributes set before any CUDA graph is captured.
"""

from __future__ import annotations


import torch
from ..native import native_ops  # target-owned C++ ops

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

# Attention-input chain (idea 232): wq_a||wkv is launched with its dependents triggered at
# start so the fused q_norm + MXFP8 quant + wq_b(||index_wq_b) kernel (``native_ops.csa2_qnorm_low_m_gemm``)
# stages its weight slice while wq_a||wkv streams. Measured in a 40-layer cold-L2 PDL CUDA-graph chain on RTX PRO 6000
# (experiments/ascent/qb_fused/bench.py, jobs 1087/1097), us/layer main -> chain:
#   wq_b [8192, 1280]: 24.2 -> 19.5 (M=1), 22.7 -> 21.1 (2), 24.5 -> 21.5 (3), 24.6 -> 21.6 (4), 24.7 -> 21.9 (5),
#                      24.8 -> 24.1 (6), 25.8 -> 24.8 (7), 25.8 -> 24.9 (8);
#   wq_b||index_wq_b [12288, 1280] (n_split 8192): 26.4 -> 24.0 (1), 28.2 -> 23.9 (2), 29.7 -> 27.0 (3), 30.0 -> 27.0 (4);
#   at M >= 5 it is within noise or slower, so those row counts keep the unfused route.
# wq_a||wkv configuration when the fused kernel follows it (the co-resident pair differs from the standalone best).
_CHAIN_A_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    (1792, 5120): {1: (4, 4, 2), 2: (8, 2, 2), 3: (8, 4, 1), 4: (8, 4, 1), 5: (8, 4, 1), 6: (8, 4, 1),
                   7: (8, 4, 1), 8: (8, 4, 1)},
}
# (N, K) -> fused q_norm + quant + GEMM configuration per M: (warps, cluster_k, row_tiles).
_QNORM_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    (8192, 1280): {1: (4, 1, 2), 2: (4, 1, 4), 3: (8, 1, 4), 4: (8, 1, 4), 5: (8, 1, 4), 6: (8, 1, 2),
                   7: (8, 1, 2), 8: (8, 1, 2)},
    (12288, 1280): {1: (4, 1, 2), 2: (8, 1, 2), 3: (4, 1, 1), 4: (4, 1, 1)},
}
_TRIGGER_AT_START = 2  # csa2_mxfp8_low_m_gemm debug_flags bit 1: trigger dependents at start
_L2PF = 4  # debug_flags bit 2: L2-prefetch pf_w / pf_sf (bits 3-4 position, bit 5 evict_last)

# Attention-input chain L2 prefetch (idea 272): wq_a||wkv (A) prefetches the tail of the fused
# q_norm + wq_b(||index_wq_b) kernel's (B) weight + scale bytes into L2, so B's CTAs that only become resident once
# A's CTAs exit stage from L2 instead of DRAM. Per M: (position, evict_last, frac/256); position 0 = after A's own
# bulk copies (before its trigger), 1 = after A's weights landed, 2 = after A's PDL wait. A prefetch is a hint, so
# outputs are bitwise unchanged.
# Measured in the 40-layer cold-L2 chain with the real hc_attn K1 (experiments/ascent/qb_l2pf/bench.py arms, job
# 1592), us/layer vs off: M=1 every arm slower (+0.5..+2.1: all of B's CTAs are co-resident there, so the prefetch
# only lengthens A) -> off; M=2 pos 1 frac 154 -0.73 (25.48 vs 26.21); M=4 pos 2 frac 102 -1.58 (25.91 vs 27.49).
# M=3 shares M=4's A/B configurations; M>=5 was not measured and stays off.
_L2PF_DEFAULT: dict[int, tuple[int, int, int]] = {2: (1, 0, 154), 3: (2, 0, 102), 4: (2, 0, 102)}


def l2pf_setting(m: int):
    """``(position, evict_last, frac)`` of the attention-input chain's L2 prefetch at ``m`` rows, or ``None``."""
    return _L2PF_DEFAULT.get(m)


_state: dict = {}


def _available(device: torch.device) -> bool:
    ok = _state.get(device.index)
    if ok is None:
        ok = (
            torch.cuda.get_device_capability(device) == (12, 0)
            and hasattr(native_ops, "csa2_mxfp8_low_m_gemm")
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
        native_ops.csa2_mxfp8_low_m_gemm(xw, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True)
    _warmed.add(key)


_warmed: set = set()


def low_m_mm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, trigger_at_start: bool = False,
             l2_prefetch=None):
    """``mxfp8(x) @ weight^T`` in BF16 through the fused kernel, or ``None`` when it does not apply.

    ``trigger_at_start`` launches the attention-input chain's configuration, which triggers its PDL dependents at
    start (its consumer must be ``qnorm_mm``, whose kernel reads nothing of this output before its PDL wait).
    ``l2_prefetch`` = (weight, weight_scale) of that consumer: with ``trigger_at_start`` and
    ``l2pf_setting(m)`` on, the kernel L2-prefetches them (a hint; the output is unchanged).
    """
    if x.dim() != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        return None
    m, k = x.shape
    if m == 0 or m > MAX_M or weight.dim() != 2 or weight.shape[1] != k or weight_scale.dim() != 1:
        return None
    cfg = chain_a_config_for(weight.shape[0], k, m) if trigger_at_start else config_for(weight.shape[0], k, m)
    if cfg is None or not _available(x.device):
        return None
    _warmup(x, weight, weight_scale)
    if x.stride(1) != 1 or x.stride(0) % 8 != 0 or x.data_ptr() % 16 != 0:
        x = x.contiguous()
    flags = _TRIGGER_AT_START if trigger_at_start else 0
    pf = l2pf_setting(m) if (trigger_at_start and l2_prefetch is not None) else None
    if pf is not None:
        pos, last, frac = pf
        flags |= _L2PF | (pos << 3) | ((1 if last else 0) << 5)
        return native_ops.csa2_mxfp8_low_m_gemm(x, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True,
                                                     debug_flags=flags, pf_w=l2_prefetch[0],
                                                     pf_sf=l2_prefetch[1], pf_frac=frac)[0]
    return native_ops.csa2_mxfp8_low_m_gemm(x, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True,
                                                 debug_flags=flags)[0]


def chain_a_config_for(n: int, k: int, m: int):
    table = _CHAIN_A_CONFIGS.get((n, k))
    return None if table is None else table.get(m)


def qnorm_config_for(n: int, k: int, m: int):
    table = _QNORM_CONFIGS.get((n, k))
    return None if table is None else table.get(m)


def _qnorm_available(device: torch.device) -> bool:
    key = ("qnorm", device.index)
    ok = _state.get(key)
    if ok is None:
        ok = (
            _available(device)
            and hasattr(native_ops, "csa2_qnorm_low_m_gemm")
        )
        _state[key] = ok
    return ok


def qnorm_chain_applies(m: int, a_weight: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                        norm_weight: torch.Tensor) -> bool:
    """Whether the attention-input chain (A trigger-at-start -> fused q_norm + quant + GEMM) serves ``m`` rows."""
    if not 1 <= m <= MAX_M or weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or weight_scale.dim() != 1:
        return False
    if norm_weight.dtype != torch.bfloat16 or norm_weight.numel() != weight.shape[1] or not norm_weight.is_contiguous():
        return False
    if a_weight.dim() != 2 or chain_a_config_for(a_weight.shape[0], a_weight.shape[1], m) is None:
        return False
    return qnorm_config_for(weight.shape[0], weight.shape[1], m) is not None and _qnorm_available(weight.device)


def _qnorm_warmup(weight: torch.Tensor, weight_scale: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                  n_split: int) -> None:
    """Launch every fused q_norm configuration of this (N, K) table once, eagerly, before any graph capture."""
    n, k = weight.shape
    key = ("qnorm", weight.device.index, n, k, n_split)
    if key in _warmed or torch.cuda.is_current_stream_capturing():
        return
    for m, cfg in sorted(_QNORM_CONFIGS[(n, k)].items()):
        xw = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
        native_ops.csa2_qnorm_low_m_gemm(xw, weight, weight_scale, norm_weight, eps, cfg[0], cfg[1], cfg[2],
                                               n_split, True)
    _warmed.add(key)


def _chain_a_warmup(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    n, k = weight.shape
    key = ("chain_a", weight.device.index, n, k)
    if key in _warmed or torch.cuda.is_current_stream_capturing():
        return
    for m, cfg in sorted(_CHAIN_A_CONFIGS[(n, k)].items()):
        xw = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
        native_ops.csa2_mxfp8_low_m_gemm(xw, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True,
                                             debug_flags=_TRIGGER_AT_START)
    _warmed.add(key)


def qnorm_chain_warmup(a_weight, a_weight_scale, weight, weight_scale, norm_weight, eps, n_split=0) -> None:
    """Eager warmup of both chain kernels (run before capture; a no-op once done or while capturing)."""
    _chain_a_warmup(a_weight, a_weight_scale)
    _qnorm_warmup(weight, weight_scale, norm_weight, eps, n_split)


def qnorm_mm(rows: torch.Tensor, norm_weight: torch.Tensor, eps: float, weight: torch.Tensor,
             weight_scale: torch.Tensor, n_split: int = 0):
    """``qr = rmsnorm(rows)`` (bit for bit as FlashInfer rmsnorm_cute) and ``mxfp8(qr) @ weight^T`` in one launch.

    ``rows`` is the raw [M, K] row view (unit column stride, 16-byte aligned rows). Returns ``[qr, out]`` or, with
    ``n_split``, ``[qr, out[:, :n_split], out[:, n_split:]]``. Only call it when ``qnorm_chain_applies``.
    """
    m, k = rows.shape
    cfg = qnorm_config_for(weight.shape[0], k, m)
    if rows.stride(1) != 1 or rows.stride(0) % 8 != 0 or rows.data_ptr() % 16 != 0:
        rows = rows.contiguous()
    return native_ops.csa2_qnorm_low_m_gemm(rows, weight, weight_scale, norm_weight, eps, cfg[0], cfg[1],
                                                  cfg[2], n_split, True)
