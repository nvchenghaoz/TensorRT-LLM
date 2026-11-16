# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One layer's causal attention over the FlashInfer FP8 paged pool (`_flashinfer_paged`).

The KV append (unless the QKV prep stored K/V), the engine metadata's plan, and FlashInfer FA2
prefill and/or decode runs; with `read_rows` on a context-only step of 512+ rows, the last
layer's attention on each request's last row only (the rows the logits read), bitwise the
full-row result on those rows and zero elsewhere.
"""

from __future__ import annotations

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ._flashinfer_paged import paged_attention as _impl


class PagedAttention(OpWrapper):
    """Writes `output` [T, Hq * D] bf16 in place and returns None."""

    ARCHS = frozenset({Arch.SM_120})
    note = """
    No cells: the op needs the engine's KV cache manager and FlashInfer metadata for a step,
    which the unit tests do not build. The target's FA2 prefill module is certified bitwise
    against FlashInfer's own (`test_modeling_v2_flashinfer_jit.py`), the read-rows index math
    on the CPU (`test_modeling_v2_read_rows.py`), and the op as a whole by the accuracy gate.
    """

    def __call__(self, q, k, v, md, layer_idx, num_heads, num_kv_heads, head_dim, q_scaling,
                 window_left, output, read_rows):
        return _impl(q, k, v, md, layer_idx, num_heads, num_kv_heads, head_dim, q_scaling,
                     window_left, output, read_rows)

    def reference(self, q, k, v, md, layer_idx, num_heads, num_kv_heads, head_dim, q_scaling,
                  window_left, output, read_rows):
        """Not stated in plain torch: see `note`."""
        raise NotImplementedError("paged_attention is certified by the accuracy gate")


paged_attention = PagedAttention()
