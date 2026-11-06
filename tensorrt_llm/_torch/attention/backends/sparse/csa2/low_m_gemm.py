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

# Attention-input chain (TRTLLM_CSA2_QB_FUSED, idea 232): wq_a||wkv is launched with its dependents triggered at
# start so the fused q_norm + MXFP8 quant + wq_b(||index_wq_b) kernel (``torch.ops.trtllm.csa2_qnorm_low_m_gemm``)
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

# Attention-input chain L2 prefetch (TRTLLM_CSA2_QB_L2PF, idea 272): wq_a||wkv (A) prefetches the tail of the fused
# q_norm + wq_b(||index_wq_b) kernel's (B) weight + scale bytes into L2, so B's CTAs that only become resident once
# A's CTAs exit stage from L2 instead of DRAM. Per M: (position, evict_last, frac/256); position 0 = after A's own
# bulk copies (before its trigger), 1 = after A's weights landed, 2 = after A's PDL wait. A prefetch is a hint, so
# outputs are bitwise unchanged. TRTLLM_CSA2_QB_L2PF=0 disables it; "pos,evict_last,frac" overrides every M.
# Measured in the 40-layer cold-L2 chain with the real hc_attn K1 (experiments/ascent/qb_l2pf/bench.py arms, job
# 1592), us/layer vs off: M=1 every arm slower (+0.5..+2.1: all of B's CTAs are co-resident there, so the prefetch
# only lengthens A) -> off; M=2 pos 1 frac 154 -0.73 (25.48 vs 26.21); M=4 pos 2 frac 102 -1.58 (25.91 vs 27.49).
# M=3 shares M=4's A/B configurations; M>=5 was not measured and stays off.
_L2PF_DEFAULT: dict[int, tuple[int, int, int]] = {2: (1, 0, 154), 3: (2, 0, 102), 4: (2, 0, 102)}
_l2pf_cache: dict = {}


def l2pf_setting(m: int):
    """``(position, evict_last, frac)`` of the attention-input chain's L2 prefetch at ``m`` rows, or ``None``."""
    env = os.environ.get("TRTLLM_CSA2_QB_L2PF", "1")
    key = (env, m)
    if key not in _l2pf_cache:
        if env == "0":
            val = None
        elif env == "1":
            val = _L2PF_DEFAULT.get(m)
        else:
            pos, last, frac = (int(v) for v in env.split(","))
            val = (pos, last, frac)
        _l2pf_cache[key] = val
    return _l2pf_cache[key]


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
        return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True,
                                                     debug_flags=flags, pf_w=l2_prefetch[0],
                                                     pf_sf=l2_prefetch[1], pf_frac=frac)[0]
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True,
                                                 debug_flags=flags)[0]


def chain_a_config_for(n: int, k: int, m: int):
    table = _CHAIN_A_CONFIGS.get((n, k))
    return None if table is None else table.get(m)


# hc_attn K1 fold into the attention-input chain's A (idea 386, SM120 decode M <= 2): the boundary's mHC lagged K1
# (mhc_lagged_split.mhc_lagged_layer_input) is not launched; A = torch.ops.trtllm.csa2_mxfp8_low_m_gemm_k1fold computes
# layer_input itself after its PDL wait, bit for bit (cluster split of K1's reduction, DSMEM exchange), quantizes it in
# registers and writes it to the boundary's layer_input tensor; the CSA2 aux stream runs its own K1 copy.
# TRTLLM_CSA2_A_K1_FOLD=0 turns it off; TRTLLM_CSA2_A_K1_FOLD_MAX_M (default 2, at most 2) bounds the row count.
A_K1_FOLD_ENV_ON = os.environ.get("TRTLLM_CSA2_A_K1_FOLD", "1") != "0"
A_K1_FOLD_MAX_M = max(0, min(2, int(os.environ.get("TRTLLM_CSA2_A_K1_FOLD_MAX_M", "2"))))
_K1_HIDDEN = 5120
_K1FOLD_CONFIGS = ((4, 4, 2), (8, 2, 2))  # chain-A configurations with a K1FOLD kernel instance


def _k1_first_block(w: int, p: int, c_lo: int) -> int:
    s = 32 * w
    return 0 if s >= c_lo else (c_lo - s + p - 1) // p


def _k1_blocks_in(w: int, p: int, c_lo: int, cpr: int) -> int:
    n = 0
    s = 32 * w + _k1_first_block(w, p, c_lo) * p
    while s < c_lo + cpr:
        n += 1
        s += p
    return n


def k1fold_plan_ok(warps: int, cluster_k: int, m: int, p: int) -> bool:
    """Mirror of csa2K1FoldPlan / isCsa2K1FoldSupported (mxfp8LowMGemm.cu) for one chain-A configuration: units
    (token, K1 warp with blocks in the rank), token-major, dealt round-robin to the hardware warps; each warp holds at
    most 2 (4-warp instance) or 4 (8-warp instance) 32-chunk blocks."""
    chunks = _K1_HIDDEN // 8
    if not 1 <= m <= 2 or p <= 0 or p % 32 or p // 32 > 16 or p > chunks or not 2 <= cluster_k <= 4:
        return False
    if chunks % cluster_k or warps > 8:
        return False
    cpr = chunks // cluster_k
    if cpr % 32:
        return False
    slots_max = 2 if warps == 4 else 4
    for r in range(cluster_k):
        ws = [w for w in range(p // 32) if _k1_blocks_in(w, p, r * cpr, cpr) > 0]
        nslots = [0] * warps
        for u in range(m * len(ws)):
            nslots[u % warps] += _k1_blocks_in(ws[u % len(ws)], p, r * cpr, cpr)
        if max(nslots) > slots_max:
            return False
    return True


def k1fold_applies(m: int, reducer_threads: int, a_weight: torch.Tensor) -> bool:
    """Whether A (wq_a||wkv chain configuration) can take the hc_attn K1 at ``m`` rows (needs k1fold_warmup first)."""
    if not A_K1_FOLD_ENV_ON or not 1 <= m <= A_K1_FOLD_MAX_M or a_weight.dim() != 2:
        return False
    if not hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm_k1fold") or not _available(a_weight.device):
        return False
    n, k = a_weight.shape
    cfg = chain_a_config_for(n, k, m)
    if k != _K1_HIDDEN or cfg is None or tuple(cfg) not in _K1FOLD_CONFIGS:
        return False
    if ("k1fold", a_weight.device.index, n, k) not in _warmed:
        return False
    return k1fold_plan_ok(cfg[0], cfg[1], m, int(reducer_threads))


def k1fold_warmup(a_weight: torch.Tensor, a_weight_scale: torch.Tensor) -> None:
    """Eager launch of every K1FOLD instance (with and without the L2 prefetch) before any graph capture."""
    n, k = a_weight.shape
    key = ("k1fold", a_weight.device.index, n, k)
    if key in _warmed or torch.cuda.is_current_stream_capturing() or not A_K1_FOLD_ENV_ON:
        return
    if k != _K1_HIDDEN or not hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm_k1fold"):
        return
    if not _available(a_weight.device):
        return
    dev = a_weight.device
    for m in range(1, A_K1_FOLD_MAX_M + 1):
        cfg = chain_a_config_for(n, k, m)
        if cfg is None or tuple(cfg) not in _K1FOLD_CONFIGS or not k1fold_plan_ok(cfg[0], cfg[1], m, 480):
            continue
        x = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
        res = torch.zeros((m, 4, k), dtype=torch.bfloat16, device=dev)
        post = torch.zeros((m, 4), dtype=torch.float32, device=dev)
        comb = torch.zeros((m, 16), dtype=torch.float32, device=dev)
        nw = torch.ones((k,), dtype=torch.bfloat16, device=dev)
        li = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
        for pf in (False, True):
            extra = {"debug_flags": _TRIGGER_AT_START}
            if pf:
                extra = {"debug_flags": _TRIGGER_AT_START | _L2PF | (1 << 3), "pf_w": a_weight, "pf_sf": a_weight_scale,
                         "pf_frac": 1}
            torch.ops.trtllm.csa2_mxfp8_low_m_gemm_k1fold(x, res, post, comb, post, nw, 1e-6, 480, li, a_weight,
                                                          a_weight_scale, cfg[0], cfg[1], cfg[2], True, **extra)
    _warmed.add(key)


def low_m_mm_k1fold(pending, weight: torch.Tensor, weight_scale: torch.Tensor, l2_prefetch=None) -> torch.Tensor:
    """A of the attention-input chain (``low_m_mm(layer_input, ..., trigger_at_start=True, l2_prefetch=...)``) with the
    deferred hc_attn K1 ``pending`` (mhc_lagged_split.K1FoldPending) folded in: writes ``pending.out`` and returns the
    [M, N] rows. Only call it when ``k1fold_applies``."""
    x_prev, residual, post, comb, pre, nw, eps, p = pending.fold_args()
    m, k = x_prev.shape
    cfg = chain_a_config_for(weight.shape[0], k, m)
    flags = _TRIGGER_AT_START
    pf = l2pf_setting(m) if l2_prefetch is not None else None
    extra = {}
    if pf is not None:
        pos, last, frac = pf
        flags |= _L2PF | (pos << 3) | ((1 if last else 0) << 5)
        extra = {"pf_w": l2_prefetch[0], "pf_sf": l2_prefetch[1], "pf_frac": frac}
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm_k1fold(x_prev, residual, post, comb, pre, nw, eps, p, pending.out,
                                                         weight, weight_scale, cfg[0], cfg[1], cfg[2], True,
                                                         debug_flags=flags, **extra)


def qnorm_config_for(n: int, k: int, m: int):
    table = _QNORM_CONFIGS.get((n, k))
    return None if table is None else table.get(m)


def _qnorm_available(device: torch.device) -> bool:
    key = ("qnorm", device.index)
    ok = _state.get(key)
    if ok is None:
        ok = (
            os.environ.get("TRTLLM_CSA2_QB_FUSED", "1") != "0"
            and _available(device)
            and hasattr(torch.ops.trtllm, "csa2_qnorm_low_m_gemm")
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
        torch.ops.trtllm.csa2_qnorm_low_m_gemm(xw, weight, weight_scale, norm_weight, eps, cfg[0], cfg[1], cfg[2],
                                               n_split, True)
    _warmed.add(key)


def _chain_a_warmup(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    n, k = weight.shape
    key = ("chain_a", weight.device.index, n, k)
    if key in _warmed or torch.cuda.is_current_stream_capturing():
        return
    for m, cfg in sorted(_CHAIN_A_CONFIGS[(n, k)].items()):
        xw = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
        torch.ops.trtllm.csa2_mxfp8_low_m_gemm(xw, weight, weight_scale, cfg[0], cfg[1], cfg[2], 0, True,
                                             debug_flags=_TRIGGER_AT_START)
    _warmed.add(key)


def qnorm_chain_warmup(a_weight, a_weight_scale, weight, weight_scale, norm_weight, eps, n_split=0) -> None:
    """Eager warmup of both chain kernels (run before capture; a no-op once done or while capturing)."""
    _chain_a_warmup(a_weight, a_weight_scale)
    _qnorm_warmup(weight, weight_scale, norm_weight, eps, n_split)
    k1fold_warmup(a_weight, a_weight_scale)


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
    return torch.ops.trtllm.csa2_qnorm_low_m_gemm(rows, weight, weight_scale, norm_weight, eps, cfg[0], cfg[1],
                                                  cfg[2], n_split, True)


if hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm"):

    @torch.library.register_fake("trtllm::csa2_mxfp8_low_m_gemm")
    def _(x, w, wsf, warps, cluster_k, row_tiles, n_split, pdl, xq=None, xsf=None, xq_out=None, xsf_out=None,
          dbg_ts=None, stages=1, debug_flags=0, pf_w=None, pf_sf=None, pf_frac=256):
        n = w.shape[0]
        if n_split == 0:
            return [x.new_empty((x.shape[0], n))]
        return [x.new_empty((x.shape[0], n_split)), x.new_empty((x.shape[0], n - n_split))]


if hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm_k1fold"):

    @torch.library.register_fake("trtllm::csa2_mxfp8_low_m_gemm_k1fold")
    def _(x_prev, residual, post, comb, pre, norm_w, eps, reducer_threads, layer_input, w, wsf, warps, cluster_k,
          row_tiles, pdl, debug_flags=0, pf_w=None, pf_sf=None, pf_frac=256, xq_out=None, xsf_out=None):
        return x_prev.new_empty((x_prev.shape[0], w.shape[0]))


if hasattr(torch.ops.trtllm, "csa2_qnorm_low_m_gemm"):

    @torch.library.register_fake("trtllm::csa2_qnorm_low_m_gemm")
    def _(x, w, wsf, norm_w, eps, warps, cluster_k, row_tiles, n_split, pdl, debug_flags=0, xq_out=None,
          xsf_out=None):
        n = w.shape[0]
        qr = x.new_empty((x.shape[0], x.shape[1]))
        if n_split == 0:
            return [qr, x.new_empty((x.shape[0], n))]
        return [qr, x.new_empty((x.shape[0], n_split)), x.new_empty((x.shape[0], n - n_split))]
