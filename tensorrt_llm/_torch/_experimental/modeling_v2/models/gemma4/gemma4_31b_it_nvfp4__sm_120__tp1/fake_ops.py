# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The SM120 sequential-K BF16 GEMM of the attention projections. The loader imports this module
right after the target is found in the family library, so a fake for an op the library does
not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


@torch.library.register_fake(f"{_NS}::sm120_bf16_seqk_gemm")
def _sm120_bf16_seqk_gemm(mat_a, weight, config):
    return mat_a.new_empty((mat_a.shape[0], weight.shape[0]))
