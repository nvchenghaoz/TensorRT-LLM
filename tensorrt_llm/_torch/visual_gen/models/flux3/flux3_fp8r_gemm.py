# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 FP8-rowwise GEMM for FLUX.3's FP8R (BF16 checkpoint-excluded) blocks (idea 122).

FlashInfer overlay entry ``csrc/flux3_fp8r_gemm_sm120.cu`` (module ``flux3_fp8r_gemm_sm120``):
a CUTLASS SM120 E4M3 x E4M3 GEMM with FP32 accumulation, a CLC persistent tile scheduler, >= 4
smem stages, PDL, and an EVT epilogue BF16(s_outer * (s_inner * acc)) with separate FP32 RN
multiplies. It replaces torch._scaled_mm (PyTorch's SM120 rowwise CUTLASS kernel, 2 smem stages)
at image.0's 4 GEMM sites when TRTLLM_FLUX3_FP8R_GEMM=1 (default); 0 restores torch._scaled_mm.

The module is JIT-built and its workspace allocated at model load (``preload``); a call never
builds or allocates anything but its output.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

# Arm: 0 = Cooperative 128x128x64 (A), 1 = Pingpong 128x128x64 (C), 2 = Cooperative 128x128x128.
ARM = int(os.environ.get("TRTLLM_FLUX3_FP8R_GEMM_ARM", "0"))
# Epilogue association per weight shape (N, K) -> order (1: sw*(sx*acc), 0: sx*(sw*acc)), chosen
# to match torch._scaled_mm bitwise (it runs N=3072 transposed, which swaps the two scales).
ORDER_DEFAULT = 1
ORDER_BY_N = {3072: 0}

_WORKSPACE_BYTES = 1024 * 1024
_WORKSPACES: dict[int, torch.Tensor] = {}
ROUTES = {"count": 0}


@lru_cache(maxsize=1)
def enabled() -> bool:
    """TRTLLM_FLUX3_FP8R_GEMM=0 restores torch._scaled_mm at all 4 FP8R sites (read once)."""
    return os.environ.get("TRTLLM_FLUX3_FP8R_GEMM", "1").strip() != "0"


@lru_cache(maxsize=1)
def module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_fp8r_gemm_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_fp8r_gemm_sm120.cu"],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def workspace(device: torch.device, slot: int = 0) -> torch.Tensor:
    # One buffer per concurrent slot (the tail pair runs down and attn_out on two streams), keyed
    # by slot, not stream, so graph capture reuses the load-time buffers.
    key = (_device_index(device), int(slot))
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACES[key] = ws
    return ws


def preload(device: torch.device) -> None:
    """Build the module and allocate the workspace at model load."""
    if not enabled():
        return
    mod = module()
    need = int(mod.fp8r_workspace_size(1 << 16, 18432, 9216, ARM))
    if need > _WORKSPACE_BYTES:
        raise RuntimeError(f"FP8R GEMM workspace {need} B exceeds {_WORKSPACE_BYTES} B")
    workspace(device, 0)
    workspace(device, 1)


def order_for(n: int) -> int:
    return ORDER_BY_N.get(int(n), ORDER_DEFAULT)


def gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    arm: int | None = None,
    order: int | None = None,
    slot: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 [M, N] = rowwise-scaled FP8 GEMM, same contract as torch._scaled_mm(a, w.t(), ...).

    activation E4M3 [M, K] contiguous, weight E4M3 [N, K] contiguous, activation_scale FP32 [M]
    (any shape with M elements), weight_scale FP32 [N].
    """
    m, k = activation.shape
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=activation.device)
    module().fp8r_gemm(
        activation,
        weight,
        activation_scale.reshape(-1),
        weight_scale.reshape(-1),
        out,
        workspace(activation.device, slot),
        int(ARM if arm is None else arm),
        int(order_for(n) if order is None else order),
    )
    ROUTES["count"] += 1
    return out
