# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for this target's C++ GEMM operators (``thop/qwen35``).

Only the operators that have no fake in ``custom_ops/cpp_custom_ops.py``: the
cuda-core W4A16 GEMV (plain and with the fused conv1d update) and the SM120
gate_up GEMM with the SwiGLU + NVFP4 epilogue. Imported by the entries that call
them, so the fakes exist before any of those ops is traced.
"""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

_SF_VEC = 16


@torch.library.register_fake("trtllm::nvfp4_w4a16_gemv")
def _nvfp4_w4a16_gemv(mat_a, weight, weight_sf, global_scale):
    # weight [N, K/2] packed E2M1 -> output [M, N] BF16.
    return mat_a.new_empty((mat_a.shape[0], weight.shape[0]))


@torch.library.register_fake("trtllm::nvfp4_w4a16_gemv_conv")
def _nvfp4_w4a16_gemv_conv(
    mat_a,
    weight,
    weight_sf,
    global_scale,
    conv_state,
    conv_weight,
    conv_state_indices,
    conv_dim,
    pad_slot_id,
):
    # Mutates conv_state in place; output [M, N] BF16.
    return mat_a.new_empty((mat_a.shape[0], weight.shape[0]))


@torch.library.register_fake("trtllm::nvfp4_gate_up_swiglu_nvfp4")
def _nvfp4_gate_up_swiglu_nvfp4(mat1, mat2, mat1Scale, mat2Scale, alpha, outScale, tile=0):
    # (fp4 [m, n/4] uint8, swizzled SF [ceil128(m) * ceil4(n/32)] uint8).
    m, n = mat1.shape[0], mat2.shape[0]
    d = n // 2
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (d // _SF_VEC + 3) // 4
    return [
        mat1.new_empty((m, d // 2), dtype=torch.uint8),
        mat1.new_empty((m_pad * num_k_tiles * 4,), dtype=torch.uint8),
    ]
