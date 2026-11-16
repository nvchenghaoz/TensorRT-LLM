# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer FA2 paged attention over the engine's FP8 KV cache, driven from the target.

`paged_attention` runs one layer's attention for a step: the KV append (unless the fused QKV
prep already stored this layer's K/V in the pool), then the context rows on the target's own FA2
prefill wrapper (`Fa2PrefillWrapper`, the target's FlashInfer kernels), planned once per step
and attention kind from the pinned host indptrs the engine's metadata built (no stream drain),
and the generation rows on the engine metadata's decode wrapper. Steps whose metadata has no
host plan inputs (CUDA-graph metadata) run entirely on the engine metadata's wrappers.

`read_rows_prefill` is the last decoder layer's context attention on the rows anything reads
(each request's last row; only the final-norm gather reads that layer's output): each
request's trailing n rows run through FA2 with the full-row plan's template (its CTA_TILE_Q,
unsplit, same variant, same in-tile row offsets), so the kept row is bitwise the full-row
result; every other output row is zero.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import flashinfer
import numpy as np
import torch

from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttentionMetadata
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.functional import AttentionMaskType

from .._flashinfer_jit import Fa2PrefillWrapper

_MAX_CUDA_THREADS_PER_BLOCK = 1024

#: Query tiles (packed rows) of the long-q head_dim-512 FA2 plans: 32 (the FA2 template) and 64
#: (the d512 64-row kernel the planner promotes unsplit, full-attention FP8-KV plans to).
READ_ROWS_CTA_TILES_Q = (32, 64)

#: Context steps below this many rows keep the full-row call: there the sub-plan, gather,
#: zero-fill and scatter cost about what the dead rows do (1x300 on SM120: 0.047 ms full-row
#: vs 0.055-0.066 ms read-rows), plus host time.
READ_ROWS_MIN_TOKENS = 512

#: Read-rows dispatch counters (tests and A/B harnesses read them).
READ_ROWS_STATS = {"ran": 0, "fallback": 0, "below_min_tokens": 0}


def append_paged_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    md: FlashInferAttentionMetadata,
    kv_cache: torch.Tensor,
) -> None:
    """Append [T, H_kv, D] FP8 K/V into the paged pool with one FlashInfer launch."""
    head_dim = k.shape[-1]
    vec_size = max(16 // k.element_size(), head_dim // 32)
    # One CTA per token covers every head: (D / vec) threads per head. The FlashInfer backend
    # splits the heads over launches beyond this; this target's layers never get there.
    assert k.shape[1] * (head_dim // vec_size) <= _MAX_CUDA_THREADS_PER_BLOCK, k.shape
    flashinfer.page.append_paged_kv_cache(
        append_key=k,
        append_value=v,
        batch_indices=md.batch_indices,
        positions=md.positions,
        paged_kv_cache=kv_cache,
        kv_indices=md.paged_kv_indices,
        kv_indptr=md.paged_kv_indptr,
        kv_last_page_len=md.paged_kv_last_page_len,
        kv_layout=md.kv_layout,
    )


def paged_attention(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    md: FlashInferAttentionMetadata,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_scaling: float,
    window_left: Optional[int],
    output: torch.Tensor,
    read_rows: bool,
) -> None:
    """One layer's causal attention for a step, written into ``output`` [T, H_q * D].

    ``k``/``v`` are FP8 [T, H_kv * D] to append, or None when the prep stored them in the pool.
    ``window_left`` is the inclusive FlashInfer window (TRTLLM window - 1) or None.
    ``read_rows`` lets a context-only step run `read_rows_prefill` (last layer only).
    """
    q = q.view(-1, num_heads, head_dim)
    kv_cache = md.kv_cache_manager.get_buffers(layer_idx, kv_layout=md.kv_layout)
    # VSWA: this layer's pool's page indices, before both the KV append and the plan/run.
    md.swap_paged_kv_indices_for_layer(layer_idx)
    if k is not None:
        assert v is not None and kv_cache.dtype == torch.float8_e4m3fn
        append_paged_kv(
            k.view(-1, num_kv_heads, head_dim), v.view(-1, num_kv_heads, head_dim), md, kv_cache
        )
    num_ctx_tokens = md.num_ctx_tokens
    out3 = output.view(-1, num_heads, head_dim)
    host_inputs = md.host_prefill_plan_inputs
    own_prefill = (
        md.num_contexts > 0 and host_inputs is not None and host_inputs[0] == md.num_contexts
    )
    if md.num_generations > 0 or not own_prefill:
        plan_params = md.plan(
            num_heads,
            num_kv_heads,
            head_dim,
            q_dtype=q.dtype,
            kv_dtype=kv_cache.dtype,
            q_scaling=q_scaling,
            attention_window_size=window_left,
            attention_mask_type=int(AttentionMaskType.causal),
            attention_mask_data=None,
            flashinfer_backend="fa2",
        )
    if md.num_contexts == 0:
        md.get_decode_wrapper(plan_params).run(q, kv_cache, out=out3)
        return
    if own_prefill:
        plan = _plan_prefill(md, num_heads, num_kv_heads, head_dim, q_scaling, window_left)
        wrapper = plan.wrapper
    else:
        plan = None
        wrapper = md.get_prefill_wrapper(plan_params)
    if md.num_generations == 0:
        if read_rows and plan is not None and read_rows_prefill(md, plan, q, kv_cache, output):
            return
        wrapper.run(q[:num_ctx_tokens], kv_cache, out=out3)
        return
    wrapper.run(q[:num_ctx_tokens], kv_cache, out=out3[:num_ctx_tokens])
    md.get_decode_wrapper(plan_params).run(
        q[num_ctx_tokens:], kv_cache, out=out3[num_ctx_tokens:]
    )


class PrefillPlan(NamedTuple):
    """One step's context-row plan for one attention kind."""

    wrapper: Fa2PrefillWrapper
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sm_scale: float
    window_left: int


def _plan_prefill(md, num_heads, num_kv_heads, head_dim, q_scaling, window_left) -> PrefillPlan:
    """Plan (once per step and attention kind) the target's FA2 prefill wrapper for this step's
    context rows, with the arguments the engine metadata's prefill plan uses.

    The wrapper shares the metadata's index buffers, as the metadata's own prefill wrapper does,
    so a run reads the page indices `swap_paged_kv_indices_for_layer` put there for its layer.
    The plan reads host data only; the one cross-step hazard is the wrapper's pinned staging
    buffer, which its previous plan may still be copying from, so it waits for that copy alone.
    """
    window = -1 if window_left is None else int(window_left)
    sm_scale = 1.0 / (math.sqrt(head_dim) * q_scaling)
    key = (num_heads, num_kv_heads, head_dim, window, sm_scale)
    states = md.__dict__.setdefault("_gemma4_prefill_plans", {})
    state = states.get(key)
    if state is None:
        state = states[key] = {
            "wrapper": Fa2PrefillWrapper(
                md.workspace_buffer,
                md.kv_layout,
                backend="fa2",
                qo_indptr_buf=md.qo_indptr,
                paged_kv_indptr_buf=md.paged_kv_indptr_prefill,
                paged_kv_indices_buf=md._paged_kv_indices,
                paged_kv_last_page_len_buf=md._paged_kv_last_page_len,
                use_cuda_graph=False,
            ),
            "event": None,
            "step": None,
        }
    host_inputs = md.host_prefill_plan_inputs
    plan = PrefillPlan(state["wrapper"], num_heads, num_kv_heads, head_dim, sm_scale, window)
    if state["step"] is host_inputs:
        return plan
    _, qo_indptr, paged_kv_indptr, paged_kv_last_page_len = host_inputs
    if state["event"] is not None:
        state["event"].synchronize()
    state["wrapper"].plan(
        qo_indptr,
        paged_kv_indptr,
        md._paged_kv_indices[: md.num_context_blocks],
        paged_kv_last_page_len,
        num_heads,
        num_kv_heads,
        head_dim,
        md.page_size,
        causal=True,
        sm_scale=sm_scale,
        window_left=window,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=None,
        custom_mask=None,
    )
    if state["event"] is None:
        state["event"] = torch.cuda.Event()
    state["event"].record()
    state["step"] = host_inputs
    return plan


def read_rows_subset_lengths(seq_lens: np.ndarray, group_size: int, tile: int):
    """Trailing query rows per request for a bitwise last-row attention pass, or None.

    ``tile`` is the full-row plan's CTA_TILE_Q. n is the smallest count with
    n * group_size > tile (so the planner picks the same tile again; the caller checks the
    sub-plan's tile) and with (seq_len - n) * group_size a multiple of the tile, which puts
    every kept row at the same packed offset inside its query tile as in the full-row plan:
    each CTA's KV range, iteration count and mask iteration are those of the full-row CTA that
    holds the row. None when a request is too short for that.
    """
    n_min = tile // group_size + 1
    period = tile // math.gcd(tile, group_size)
    seq_lens = seq_lens.astype(np.int64, copy=False)
    n = n_min + (seq_lens - n_min) % period
    if (n > seq_lens).any():
        return None
    return n


def build_read_rows_indices(host_qo_indptr: np.ndarray, n: np.ndarray):
    """Sub-plan qo_indptr (int32) and the int64 index arrays of the read-rows pass: query rows
    to gather, the kept row of each request inside the sub-batch, and its destination row
    (qo_indptr[i + 1] - 1) in the output."""
    ends = host_qo_indptr[1:].astype(np.int64)
    sub_indptr = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    total = int(sub_indptr[-1])
    q_rows = (
        np.repeat(ends - n, n)
        + np.arange(total, dtype=np.int64)
        - np.repeat(sub_indptr[:-1], n)
    )
    keep = sub_indptr[1:] - 1
    dst = ends - 1
    return sub_indptr.astype(np.int32), q_rows, keep, dst


def read_rows_prefill(
    md: FlashInferAttentionMetadata,
    plan: PrefillPlan,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Context attention on each request's last row only; False (before any work) when a guard
    fails, and the caller runs the full-row call. K/V of every row is already appended."""
    if md.num_ctx_tokens < READ_ROWS_MIN_TOKENS:
        READ_ROWS_STATS["below_min_tokens"] += 1
        return False
    host_inputs = md.host_prefill_plan_inputs
    num_contexts = md.num_contexts
    if not (
        num_contexts > 0
        and md.num_generations == 0
        and not md.is_cuda_graph
        and not torch.cuda.is_current_stream_capturing()
        and plan.window_left < 0
        and q.dtype == torch.bfloat16
        and host_inputs is not None
        and host_inputs[0] == num_contexts
        and q.shape[0] == md.num_ctx_tokens
        and output.shape[0] == md.num_ctx_tokens
    ):
        READ_ROWS_STATS["fallback"] += 1
        return False
    main_wrapper = plan.wrapper
    main_info = getattr(main_wrapper, "_plan_info", None)
    if (
        main_info is None
        or getattr(main_wrapper, "_backend", None) != "fa2"
        or int(main_info[3]) not in READ_ROWS_CTA_TILES_Q
        or int(main_info[14]) != 0
    ):
        READ_ROWS_STATS["fallback"] += 1
        return False
    _, host_qo_indptr, paged_kv_indptr, paged_kv_last_page_len = host_inputs

    state = getattr(md, "_gemma4_read_rows_state", None)
    if state is None:
        state = {
            "wrapper": Fa2PrefillWrapper(md.workspace_buffer, md.kv_layout, backend="fa2"),
            "event": None,
            "key": None,
        }
        md._gemma4_read_rows_state = state
    sub_wrapper = state["wrapper"]
    page_indices = md._paged_kv_indices[: md.num_context_blocks]
    # The sub-plan depends only on these host inputs (the page ids are read at run time from the
    # same device buffer), so a step with the same ones reuses the wrapper's plan and indices.
    main_tile = int(main_info[3])
    key = (
        host_qo_indptr.numpy().tobytes(),
        paged_kv_indptr.numpy().tobytes(),
        paged_kv_last_page_len.numpy().tobytes(),
        page_indices.data_ptr(),
        page_indices.shape[0],
        md.page_size,
        plan[1:],
        main_tile,
    )
    if state["key"] != key:
        state["key"] = None
        qo_np = host_qo_indptr.numpy()
        group_size = plan.num_heads // plan.num_kv_heads
        n = read_rows_subset_lengths(qo_np[1:] - qo_np[:-1], group_size, main_tile)
        if n is None:
            READ_ROWS_STATS["fallback"] += 1
            return False
        sub_indptr, q_rows, keep, dst = build_read_rows_indices(qo_np, n)
        # The wrapper's pinned staging buffer may still feed its previous plan's H2D copy.
        if state["event"] is not None:
            state["event"].synchronize()
        sub_wrapper.plan(
            maybe_pin_memory(torch.from_numpy(sub_indptr)),
            paged_kv_indptr,
            page_indices,
            paged_kv_last_page_len,
            plan.num_heads,
            plan.num_kv_heads,
            plan.head_dim,
            md.page_size,
            causal=True,
            sm_scale=plan.sm_scale,
            window_left=plan.window_left,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=None,
            custom_mask=None,
            disable_split_kv=True,
        )
        if state["event"] is None:
            state["event"] = torch.cuda.Event()
        state["event"].record()
        sub_info = sub_wrapper._plan_info
        if int(sub_info[3]) != main_tile or int(sub_info[14]) != 0:
            READ_ROWS_STATS["fallback"] += 1
            return False
        idx_host = maybe_pin_memory(torch.from_numpy(np.concatenate([q_rows, keep, dst])))
        state["idx"] = idx_host.to(q.device, non_blocking=True)
        state["total"] = q_rows.shape[0]
        state["key"] = key
    idx = state["idx"]
    total = state["total"]
    q_idx = idx[:total]
    keep_idx = idx[total : total + num_contexts]
    dst_idx = idx[total + num_contexts :]

    q_sub = q.index_select(0, q_idx)
    sub_out = torch.empty_like(q_sub)
    sub_wrapper.run(q_sub, kv_cache, out=sub_out)
    out2d = output.view(output.shape[0], -1)
    out2d.zero_()
    out2d.index_copy_(0, dst_idx, sub_out.view(total, -1).index_select(0, keep_idx))
    READ_ROWS_STATS["ran"] += 1
    return True
