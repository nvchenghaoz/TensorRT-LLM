# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for this target's C++ GEMM-family operators (``thop/flux3/flux3GemmOps.cpp``).

Imported by the entries that call them, so the fakes exist before any of those ops is
traced (the denoiser blocks run under torch.compile).
"""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

_SF_VEC = 16
#: FLUX.3 hidden size, the only row width the adaLN head serves.
_ADALN_D = 3072


def _pad128(rows: int) -> int:
    return (rows + 127) // 128 * 128


@torch.library.register_fake("trtllm::flux3_qkv_epi_fp8")
def _flux3_qkv_epi_fp8(
    act, weight, act_sf, weight_sf, qkv_alpha, q_norm, k_norm, rope_tab, rope_idx, sm_scale, keep_v
):
    # (qkv8 [3, M, W] e4m3, scales [8] fp32, gemm [M, 3W] bf16).
    rows, columns = act.shape[0], weight.shape[0]
    return (
        act.new_empty((3, rows, columns // 3), dtype=torch.float8_e4m3fn),
        act.new_empty((8,), dtype=torch.float32),
        act.new_empty((rows, columns), dtype=torch.bfloat16),
    )


@torch.library.register_fake("trtllm::flux3_gate_up_swiglu_nvfp4")
def _flux3_gate_up_swiglu_nvfp4(act, weight, act_sf, weight_sf, alpha, out_scale, out, out_sf):
    # Writes the caller's out / out_sf in place.
    return None


@torch.library.register_fake("trtllm::flux3_nvfp4_pair_residual")
def _flux3_nvfp4_pair_residual(
    down_act,
    down_weight,
    down_act_sf,
    down_weight_sf,
    down_alpha,
    attn_act,
    attn_weight,
    attn_act_sf,
    attn_weight_sf,
    attn_alpha,
    x,
    gate,
    gate_index,
    workspace,
    t256,
    along_n,
):
    # out [M, N] bf16.
    return x.new_empty((down_act.shape[0], down_weight.shape[0]), dtype=torch.bfloat16)


@torch.library.register_fake("trtllm::flux3_fp8r_gemm")
def _flux3_fp8r_gemm(a, b, sx, sw, arm):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


@torch.library.register_fake("trtllm::flux3_adaln_nvfp4_warp")
def _flux3_adaln_nvfp4_warp(x, select, scale, shift, global_scale, eps):
    # (fp4 [M, D/2] uint8, swizzled sf [ceil128(M) * D/16] uint8).
    rows = x.numel() // _ADALN_D
    return (
        x.new_empty((rows, _ADALN_D // 2), dtype=torch.uint8),
        x.new_empty((_pad128(rows) * (_ADALN_D // _SF_VEC),), dtype=torch.uint8),
    )
