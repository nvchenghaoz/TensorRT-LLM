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

Chunk size: ~1024 rows. On SM120 (188 SMs) the M=1024 fused_hc tactic (mtile_v3, 8 tokens per CTA) launches 128
CTAs, which leaves SMs for the 16-CTA two-shot all-reduce; 2048-row chunks fill every SM and the AR and mHC chunks
then serialize (branch 199 Stage 0, job 1840).

Chunking:
  k(T) = max(2, round(T / PIPELINE_CHUNK_ROWS)) chunks, PIPELINE_CHUNK_ROWS = 1024.
  ``PIPELINE_MIN_ROWS``: smallest boundary T that is pipelined (4096: each pipelined boundary adds ~0.1-0.3 ms of
      host launch work, which the host-paced c1 prefill (T=2048) would pay 1:1).
  ``PIPELINE_MAX_ROWS``: largest boundary T that is pipelined (8192 = max_num_tokens of the served engine).

Stream roles by T (constant ``SWAP_ABOVE_ROWS`` = 6144):
  T <= 6144: chunk all-reduces on the priority -1 comm stream, chunk mHCs on the caller (main) stream.
  T >  6144: ROLE SWAP (design B, idea 374): chunk all-reduces on the caller (main) stream, chunk mHCs on a priority 0
      side stream. Every allocation of the boundary (the chunk all-reduce outputs included) then lands in the main
      stream's allocator pool, where the unpipelined boundary allocates it, and the side stream allocates nothing. With
      the comm-stream roles the chunk outputs lived in the comm stream's pool, which pushed the max_num_tokens=8192
      warm-up peak out of memory (branch 199, job 1898). Same kernels, bounds and tactic in both roles: bitwise equal.
"""


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


PIPELINE_CHUNK_ROWS = 1024
PIPELINE_MIN_ROWS = 4096
# Largest boundary T that is pipelined: the served engine's max_num_tokens (8192). Above SWAP_ABOVE_ROWS the boundary
# runs with swapped stream roles (_run_swap), which allocate only on the main stream; the comm-stream roles' chunk
# outputs (~T x hidden x 2 bytes in the comm stream's pool) pushed the T=8192 activation peak out of memory (branch
# 199, job 1898).
PIPELINE_MAX_ROWS = 8192
# Boundaries with more rows than this run the swapped roles (chunk ARs on main, chunk mHCs on a side stream).
SWAP_ABOVE_ROWS = 6144


def k_for(T: int) -> int:
    return max(2, int(round(T / PIPELINE_CHUNK_ROWS)))


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


def plan(T: int):
    """The shape part of the pipelining decision: chunk bounds for a T-row boundary, or None (unpipelined)."""
    if T < max(PIPELINE_MIN_ROWS, 2 * MIN_CHUNK_ROWS) or T > PIPELINE_MAX_ROWS:
        return None
    return chunk_bounds(T, k_for(T))


class ArFusedHcPipeline:
    """One per model: owns the comm stream that every pipelined boundary's chunk all-reduces run on."""

    def __init__(self, priority: int = -1):
        self._priority = priority
        self._stream = None
        self._events = []
        self._side_stream = None  # T > SWAP_ABOVE_ROWS: the chunk mHCs' stream (priority 0)
        self._swap_events = []

    @property
    def stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream(priority=self._priority)
        return self._stream

    @property
    def side_stream(self) -> torch.cuda.Stream:
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(priority=0)
        return self._side_stream

    def _event(self, i: int) -> torch.cuda.Event:
        # Reused across calls: each is recorded and then waited on within one call, in order.
        while len(self._events) <= i:
            self._events.append(torch.cuda.Event())
        return self._events[i]

    def _swap_event(self, i: int) -> torch.cuda.Event:
        while len(self._swap_events) <= i:
            self._swap_events.append(torch.cuda.Event())
        return self._swap_events[i]

    def run(self, allreduce_fn, mhc, x_partial, residual_prev, post_mix_prev, comb_mix_prev, pre_mix, norm_weight,
            norm_eps, bounds, tactic=None):
        """``mhc.fused_hc_lagged(allreduce_fn(x_partial), residual_prev, ...)`` with k-chunk AR/mHC overlap.

        ``allreduce_fn(rows)`` must return the all-reduced rows; it is called on the comm stream once per chunk,
        in chunk order. Returns ``fused_hc_lagged``'s 5-tuple ``(residual, pre, post, comb, layer_input)``.
        Boundaries with more than SWAP_ABOVE_ROWS rows run _run_swap (chunk ARs on the current stream instead).
        """
        if x_partial.numel() // mhc.hidden_size > SWAP_ABOVE_ROWS:
            return self._run_swap(allreduce_fn, mhc, x_partial, residual_prev, post_mix_prev, comb_mix_prev,
                                  pre_mix, norm_weight, norm_eps, bounds, tactic)
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
        ys = []
        with torch.cuda.stream(comm):
            for i, (a, b) in enumerate(bounds):
                ys.append(allreduce_fn(x2[a:b]))
                self._event(i).record(comm)
        x2.record_stream(comm)

        if tactic is None:
            # The full-M tactic, pinned for every chunk: the autotuner's choice at this boundary's M, looked up on
            # every call exactly as main's mhc.fused_hc_lagged does. Never cache it here: a forward that runs before
            # warm-up tuning (the memory-profiling pass) gets the untuned fallback, which a cache would keep.
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
        # Half-fused scratch is per row (y_acc [ks?, B, 24], r_acc [ks?, B]) and its done counter unused: one
        # full-size allocation, sliced per chunk. The all-in-one backends keep per-tile counters: per-chunk scratch.
        shared_ws = backend in ("fused_half_fma", "fused_half_mma") and not (
            backend == "fused_half_fma" and num_k_splits > 1)
        if shared_ws:
            y_ws, r_ws, d_ws = _alloc_fused_hc_scratch(backend, B, n, num_k_splits, tile_m, x2.device)
        for i, ((a, b), y) in enumerate(zip(bounds, ys)):
            main.wait_event(self._event(i))
            y.record_stream(main)
            m = b - a
            if shared_ws:
                y_acc_ws, r_acc_ws, done_ws = y_ws[a:b], r_ws[a:b], d_ws
            else:
                y_acc_ws, r_acc_ws, done_ws = _alloc_fused_hc_scratch(backend, m, n, num_k_splits, tile_m,
                                                                      x2.device)
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

    def _run_swap(self, allreduce_fn, mhc, x_partial, residual_prev, post_mix_prev, comb_mix_prev, pre_mix,
                  norm_weight, norm_eps, bounds, tactic):
        """T > SWAP_ABOVE_ROWS: chunk ARs on the current (main) stream, chunk mHCs on the priority 0 side stream.

        Every tensor of the boundary is allocated on main (the chunk AR outputs by the AR op itself, on main; the mHC
        outputs and scratch here, before the loop), so the side stream allocates nothing and its kernels only read
        and write main-allocated memory. Every such tensor stays referenced until main has waited on the side
        stream's last chunk, so main's allocator cannot hand its blocks out while a side kernel still uses them.
        Same kernels, same chunk bounds and the same pinned full-M tactic as the comm-stream roles: outputs are
        bitwise equal to them.
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

        if tactic is None:
            # Pinned full-M tactic, looked up on every call (never cached; see run()).
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
        shared_ws = backend in ("fused_half_fma", "fused_half_mma") and not (
            backend == "fused_half_fma" and num_k_splits > 1)
        if shared_ws:
            y_ws, r_ws, d_ws = _alloc_fused_hc_scratch(backend, B, n, num_k_splits, tile_m, x2.device)
            scratch = [(y_ws[a:b], r_ws[a:b], d_ws) for a, b in bounds]
        else:
            scratch = [_alloc_fused_hc_scratch(backend, b - a, n, num_k_splits, tile_m, x2.device) for a, b in bounds]

        main = torch.cuda.current_stream()
        side = self.side_stream
        ys = []
        for i, (a, b) in enumerate(bounds):
            y = allreduce_fn(x2[a:b])  # main stream: its output comes from main's pool
            ys.append(y)
            ev = self._swap_event(i)
            ev.record(main)
            side.wait_event(ev)  # also orders every main-stream producer of this call's inputs before the side
            m = b - a
            y_acc_ws, r_acc_ws, done_ws = scratch[i]
            with torch.cuda.stream(side):
                _fused_hc_call(code, tile_n, num_k_splits, bigfuse_bs, tile_m, y, res_flat[a:b], post_f[a:b],
                               comb_f[a:b], fn, scale, base, residual_cur[a:b], post_cur[a:b], comb_cur[a:b],
                               layer_input[a:b], y_acc_ws, r_acc_ws, done_ws, m, hidden, n, mhc.norm_eps,
                               mhc.eps, mhc.sinkhorn_eps, mhc.post_mult_value, mhc.sinkhorn_iters,
                               norm_weight=nw, norm_eps=norm_eps, pre_mix_ext=pre_f[a:b],
                               pre_mix_out=pre_cur[a:b])
        done = self._swap_event(len(bounds))
        done.record(side)
        main.wait_event(done)
        # ys, scratch and the inputs above are released after this wait in main's stream order.
        del ys, scratch
        return (
            residual_cur.view(*outer_shape, n, hidden),
            pre_cur.view(*outer_shape, n, 1),
            post_cur.view(*outer_shape, n, 1),
            comb_cur.view(*outer_shape, n, n),
            layer_input.view(*outer_shape, hidden),
        )
