# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-attention output projection with this target's SM120 prefill path.

The shared ``project_sparse_attn_output`` (``attention/backends/sparse/deepseek_v4/module.py``)
runs inverse RoPE in place, then quantizes and transforms the latent before the O-LoRA BMM. On
SM120 the target fuses inverse RoPE, the 1x128 UE8M0 quantization and the permute into one Triton
launch (``prefill_wo_quant``), bitwise equal to the shared route, and writes the BMM straight into
the strided output when the TMA store allows it. Every other case delegates to the shared function.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
    project_sparse_attn_output as _shared_project_sparse_attn_output,
)
from tensorrt_llm._torch.attention.mla import _is_cute_dsl_fp8_bmm_available, _sm120_bmm_out_direct_ok
from tensorrt_llm._utils import is_sm_120f

from . import prefill_wo_quant as pwq


def _sm120_project(self, attn_output_tensor, position_ids):
    num_tokens = attn_output_tensor.shape[0]
    attn_output_tensor = attn_output_tensor.view(num_tokens, self.num_heads_tp, -1)
    cos_sin = self.inverse_rotary_emb.rotary_cos_sin
    if not (
        not self.inverse_rotary_emb.is_neox
        and pwq.supports(
            attn_output_tensor,
            cos_sin,
            self.n_local_groups,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
        )
    ):
        return None
    mat1_fp8, mat1_scale = pwq.inv_rope_quant_permute102(
        attn_output_tensor,
        position_ids.view(-1),
        cos_sin,
        self.n_local_groups,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
    )
    o_lora = torch.empty(
        [num_tokens, self.n_local_groups, self.o_lora_rank],
        device=attn_output_tensor.device,
        dtype=attn_output_tensor.dtype,
    )
    out = o_lora.transpose(0, 1)
    if _sm120_bmm_out_direct_ok(out):
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            mat1_fp8, self.o_a_proj, mat1_scale, self.o_a_proj_scale, out
        )
    else:
        tmp = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            mat1_fp8, self.o_a_proj, mat1_scale, self.o_a_proj_scale, tmp
        )
        out.copy_(tmp)
    return self.o_b_proj(o_lora.flatten(1))


def project_sparse_attn_output(
    self,
    attn_output: list[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
    attn_metadata: Optional[AttentionMetadata] = None,
    all_reduce_params=None,
) -> torch.Tensor:
    attn_output_tensor = attn_output[0]
    # Same order as the run: the fused-epilogue and CuTe-DSL fused inverse-RoPE paths come first.
    if (
        attn_output_tensor.ndim != 3
        and position_ids is not None
        and self.o_a_proj.dtype == torch.float8_e4m3fn
        and not _is_cute_dsl_fp8_bmm_available()
        and is_sm_120f()
    ):
        projected = _sm120_project(self, attn_output_tensor, position_ids)
        if projected is not None:
            return projected
    return _shared_project_sparse_attn_output(
        self, attn_output, position_ids, attn_metadata, all_reduce_params
    )
