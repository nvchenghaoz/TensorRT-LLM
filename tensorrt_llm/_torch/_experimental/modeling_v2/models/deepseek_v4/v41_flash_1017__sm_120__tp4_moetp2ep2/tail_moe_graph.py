# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer CUDA graph of the DeepSeek-V4.1 decoder-replay MoE window (128 rows only).

Scope (idea 156): on an EAGER, context-only forward with decoder bounded replay engaged,
layers L20-39 run on the replayed tail. When that tail has exactly ``TAIL_ROWS`` (128)
rows, each layer's MoE window -- the FFN mHC boundary (``hc_ffn`` fused lagged entry
with the post-attention RMSNorm), ``forward_MoE`` (router, MXFP8 quant, CUTLASS
grouped GEMM, shared experts, TP all-reduce) and, when not deferred, ``hc_ffn``'s
post_mapping -- is captured once into a CUDA graph and replayed on later steps.

The math is the eager math: the same Python code runs inside the capture, with
``do_multi_stream()`` still False (single stream, as eager), so the graph records
the same kernels, with the same tactics (autotuner cache / FlashInfer memo lookups
resolve on the host at capture), and the same all-reduce.

Engagement (rank-symmetric under TP without attention DP; every rank sees the
same batch): not ``attn_metadata.is_cuda_graph``, not capturing, no
multi-stream, ``csa2_replay_query_rows`` set (replay engaged with compacted
rows), ``num_generations == 0``, ``x_attn.shape[0] == TAIL_ROWS``, no
spec_metadata, no image_mask, a non-hashed gate, no attention DP, and neither
TRT-LLM's nor FlashInfer's AutoTuner is tuning. Each layer runs its first
``WARM_CALLS`` eligible windows eagerly (warms the FlashInfer MXFP8 memo and
CUTLASS tactic lookups at 128 rows), captures on the next one, then replays.

Graphs share one private memory pool and replay in capture order (L20..L39)
within a step; each graph's static outputs stay referenced, so no later capture
reuses them. Captures use ``capture_error_mode="thread_local"`` so executor
helper threads are not affected, and any implicit sync inside the window fails
the capture (then the feature disables itself on that rank and runs eager).
"""

import time
from typing import Any, Optional

import torch

from tensorrt_llm.logger import logger

TAIL_ROWS = 128
WARM_CALLS = 1

_pool = None
_capture_stream: Optional[torch.cuda.Stream] = None
_disabled = False  # set on any capture failure: every later window runs eager


def _tuner_busy() -> bool:
    from tensorrt_llm._torch.autotuner import AutoTuner

    if AutoTuner.get().is_tuning_mode:
        return True
    try:
        from flashinfer.autotuner import AutoTuner as FiAutoTuner

        if FiAutoTuner.get().is_tuning_mode:
            return True
    except Exception:  # pragma: no cover - flashinfer missing
        pass
    return False


def eligible(layer, x_attn, attn_metadata, spec_metadata, image_mask) -> bool:
    """Cheap host checks; True only for the 128-row eager replay tail."""
    if _disabled or x_attn.dim() != 2 or x_attn.shape[0] != TAIL_ROWS:
        return False
    if spec_metadata is not None or image_mask is not None:
        return False
    if getattr(attn_metadata, "csa2_replay_query_rows", None) is None:
        return False
    if getattr(attn_metadata, "is_cuda_graph", False) or attn_metadata.num_generations != 0:
        return False
    if layer.mapping.enable_attention_dp or getattr(layer.mlp.gate, "is_hashed", False):
        return False
    from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream

    if do_multi_stream() or torch.cuda.is_current_stream_capturing():
        return False
    return not _tuner_busy()


def moe_window(layer, x_attn, residual, post_mix, comb_mix, attn_pre, attn_metadata, input_ids):
    """The decoder layer's MoE window, as in DeepseekV41DecoderLayer.forward (no image_mask)."""
    (residual, ffn_pre, post_mix, comb_mix, layer_input), ffn_join = layer._mhc_boundary(
        layer.hc_ffn,
        1,
        x_attn,
        residual,
        post_mix,
        comb_mix,
        attn_pre,
        layer.post_attention_layernorm,
    )
    x_ffn = layer.forward_MoE(
        hidden_states=layer_input,
        attn_metadata=attn_metadata,
        spec_metadata=None,
        input_ids=input_ids,
        hidden_states_normalized=True,
    )
    if ffn_join is not None:
        ffn_join.join()
    if layer._v41_defer_post_mapping:
        return (residual, post_mix, comb_mix, x_ffn, ffn_pre)
    residual = layer.hc_ffn.post_mapping(
        x=x_ffn,
        residual=residual,
        post_layer_mix=post_mix,
        comb_res_mix=comb_mix,
    )
    return (residual, ffn_pre)


def _signature(tensors, deferred) -> tuple:
    return (deferred,) + tuple(
        None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
        for t in tensors
    )


def _static_copy(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    s = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    s.copy_(t)
    return s


class _LayerGraph:
    __slots__ = ("warm", "graph", "static_in", "static_out", "key")

    def __init__(self):
        self.warm = 0
        self.graph = None
        self.static_in = None
        self.static_out = None
        self.key = None


def _capture(layer, state: _LayerGraph, inputs, attn_metadata, key) -> bool:
    global _pool, _capture_stream, _disabled
    if _pool is None:
        _pool = torch.cuda.graph_pool_handle()
    if _capture_stream is None:
        _capture_stream = torch.cuda.Stream()
    static_in = tuple(_static_copy(t) for t in inputs)
    graph = torch.cuda.CUDAGraph()
    cur = torch.cuda.current_stream()
    _capture_stream.wait_stream(cur)
    t0 = time.perf_counter()
    try:
        with torch.cuda.stream(_capture_stream):
            graph.capture_begin(pool=_pool, capture_error_mode="thread_local")
            try:
                outs = moe_window(layer, *static_in[:5], attn_metadata, static_in[5])
            finally:
                graph.capture_end()
    except Exception as error:
        _disabled = True
        cur.wait_stream(_capture_stream)
        logger.error(
            f"[TAIL_MOE_GRAPH] rank {layer.mapping.rank} layer {layer.layer_idx}: capture "
            f"failed ({error!r}); tail MoE graphs disabled, running eager"
        )
        return False
    cur.wait_stream(_capture_stream)
    state.graph = graph
    state.static_in = static_in
    state.static_out = outs
    state.key = key
    logger.info(
        f"[TAIL_MOE_GRAPH] rank {layer.mapping.rank} layer {layer.layer_idx}: captured "
        f"{TAIL_ROWS}-row MoE window (deferred={layer._v41_defer_post_mapping}, "
        f"{len(outs)} outputs) in {(time.perf_counter() - t0) * 1e3:.1f} ms host"
    )
    return True


def run(
    layer, x_attn, residual, post_mix, comb_mix, attn_pre, attn_metadata, input_ids
) -> Optional[Any]:
    """Graphed MoE window outputs (the tuple ``moe_window`` returns), or None to run eager.

    Call only when ``eligible`` holds. None means the caller runs main's inline eager
    window (warm-up calls, signature changes, or a disabled feature).
    """
    state = getattr(layer, "_tail_moe_graph_state", None)
    if state is None:
        state = _LayerGraph()
        layer._tail_moe_graph_state = state
    inputs = (x_attn, residual, post_mix, comb_mix, attn_pre, input_ids)
    key = _signature(inputs, layer._v41_defer_post_mapping)
    if state.graph is None:
        if state.warm < WARM_CALLS:
            state.warm += 1
            return None
        if not _capture(layer, state, inputs, attn_metadata, key):
            return None
    elif key != state.key:
        return None
    else:
        for s, t in zip(state.static_in, inputs):
            if s is not None:
                s.copy_(t)
    state.graph.replay()
    return state.static_out
