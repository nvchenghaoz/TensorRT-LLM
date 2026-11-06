# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill AR -> lagged mHC boundary row-chunk pipeline (DeepSeek-V4.1, design D).

Each eager prefill boundary in layers below the decoder replay split is ``AR(x)`` on ``[T, hidden]`` followed by the
row-wise ``mHC.fused_hc_lagged``. The rows are split into ``k`` chunks: chunk ``c``'s all-reduce runs on ONE
high-priority comm stream (strictly in order: the LOWPRECISION two-shot uses one barrier workspace, never two at
once), and the main stream waits on chunk ``c``'s event and runs ``fused_hc`` on chunk ``c`` while ``AR(c+1)`` flies.
Every chunk writes its rows of preallocated full-size outputs with the FULL-M tactic pinned (the FMA tactics are
row-local, so the chunked mHC is bitwise equal per row to the full-M call on the same inputs). The main stream waits
on the last chunk's event before returning, so no comm work outlives the call.

Env: ``TRTLLM_DSV41_PREFILL_AR_PIPELINE`` = number of chunks k (``0``/``1`` = off, bitwise main);
``TRTLLM_DSV41_PREFILL_AR_PIPELINE_MIN_ROWS`` = smallest boundary T that is pipelined.
"""

import os

import torch

from .mhc_cuda import (
    _FUSED_HC_BACKEND_CODE,
    _alloc_fused_hc_scratch,
    _fused_hc_call,
    mhc_fused_hc_tactic,
)

# A chunk below 2 MiB (205 rows x 5120 bf16) would leave LOWPRECISION for AUTO/NCCL inside one boundary.
MIN_CHUNK_ROWS = 256
# Chunk starts on multiples of the mHC tuning bucket step, so each chunk M is an exact tuning bucket.
ROW_ALIGN = 128


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


PIPELINE_K = _env_int("TRTLLM_DSV41_PREFILL_AR_PIPELINE", 2)
PIPELINE_MIN_ROWS = _env_int("TRTLLM_DSV41_PREFILL_AR_PIPELINE_MIN_ROWS", 4096)


def chunk_bounds(T: int, k: int, min_chunk_rows: int = MIN_CHUNK_ROWS, align: int = ROW_ALIGN):
    """Row bounds [(a, b), ...] of k near-equal chunks (starts aligned), or None when fewer than 2 chunks fit."""
    if k <= 1 or T < 2 * min_chunk_rows:
        return None
    k = min(k, T // min_chunk_rows)
    if k <= 1:
        return None
    step = -(-T // k)
    step = -(-step // align) * align
    bounds = []
    a = 0
    while a < T:
        b = min(T, a + step)
        bounds.append((a, b))
        a = b
    if len(bounds) >= 2 and bounds[-1][1] - bounds[-1][0] < min_chunk_rows:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds if len(bounds) >= 2 else None


class ArFusedHcPipeline:
    """One per model: owns the comm stream that every pipelined boundary's chunk all-reduces run on."""

    def __init__(self, priority: int = -1):
        self._priority = priority
        self._stream = None

    @property
    def stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream(priority=self._priority)
        return self._stream

    def run(self, allreduce_fn, mhc, x_partial, residual_prev, post_mix_prev, comb_mix_prev, pre_mix, norm_weight,
            norm_eps, bounds, tactic=None):
        """``mhc.fused_hc_lagged(allreduce_fn(x_partial), residual_prev, ...)`` with k-chunk AR/mHC overlap.

        ``allreduce_fn(rows)`` must return the all-reduced rows; it is called on the comm stream once per chunk,
        in chunk order. Returns ``fused_hc_lagged``'s 5-tuple ``(residual, pre, post, comb, layer_input)``.
        """
        n = mhc.mult
        hidden = mhc.hidden_size
        outer_shape = residual_prev.shape[:-2]
        res_flat = residual_prev.reshape(-1, n, hidden).contiguous()
        B = res_flat.shape[0]
        x2 = x_partial.reshape(B, hidden).contiguous()
        post_f = post_mix_prev.reshape(B, n).to(torch.float32).contiguous()
        comb_f = comb_mix_prev.reshape(B, n, n).to(torch.float32).contiguous()
        pre_f = pre_mix.reshape(B, n).to(torch.float32).contiguous()
        fn = mhc.fn.contiguous()
        scale = mhc.scale.to(torch.float32).contiguous()
        base = mhc.base.to(torch.float32).contiguous()
        nw = norm_weight
        if nw is not None:
            if nw.dtype != torch.bfloat16:
                nw = nw.to(torch.bfloat16)
            nw = nw.contiguous()

        main = torch.cuda.current_stream()
        comm = self.stream
        comm.wait_stream(main)
        ys, events = [], []
        with torch.cuda.stream(comm):
            for a, b in bounds:
                y = allreduce_fn(x2[a:b])
                ev = torch.cuda.Event()
                ev.record(comm)
                ys.append(y)
                events.append(ev)
        x2.record_stream(comm)

        if tactic is None:
            # The full-M tactic, pinned for every chunk (the autotuner's choice at this boundary's M).
            tactic = mhc_fused_hc_tactic(
                x2, res_flat, post_f, comb_f, fn, scale, base, n, hidden, mhc.norm_eps, mhc.eps,
                mhc.sinkhorn_eps, mhc.post_mult_value, mhc.sinkhorn_iters, norm_weight=nw, norm_eps=norm_eps,
                pre_mix_ext=pre_f)
        backend, tile_n, num_k_splits, bigfuse_bs, tile_m = tactic
        code = _FUSED_HC_BACKEND_CODE[backend]

        residual_cur = torch.empty_like(res_flat)
        post_cur = torch.empty((B, n), dtype=torch.float32, device=x2.device)
        comb_cur = torch.empty((B, n * n), dtype=torch.float32, device=x2.device)
        layer_input = torch.empty((B, hidden), dtype=x2.dtype, device=x2.device)
        pre_cur = torch.empty((B, n), dtype=torch.float32, device=x2.device)
        for (a, b), y, ev in zip(bounds, ys, events):
            main.wait_event(ev)
            y.record_stream(main)
            m = b - a
            y_acc_ws, r_acc_ws, done_ws = _alloc_fused_hc_scratch(backend, m, n, num_k_splits, tile_m, x2.device)
            _fused_hc_call(code, tile_n, num_k_splits, bigfuse_bs, tile_m, y, res_flat[a:b], post_f[a:b],
                           comb_f[a:b], fn, scale, base, residual_cur[a:b], post_cur[a:b], comb_cur[a:b],
                           layer_input[a:b], y_acc_ws, r_acc_ws, done_ws, m, hidden, n, mhc.norm_eps, mhc.eps,
                           mhc.sinkhorn_eps, mhc.post_mult_value, mhc.sinkhorn_iters, norm_weight=nw,
                           norm_eps=norm_eps, pre_mix_ext=pre_f[a:b], pre_mix_out=pre_cur[a:b])
        return (
            residual_cur.view(*outer_shape, n, hidden),
            pre_cur.view(*outer_shape, n, 1),
            post_cur.view(*outer_shape, n, 1),
            comb_cur.view(*outer_shape, n, n),
            layer_input.view(*outer_shape, hidden),
        )
