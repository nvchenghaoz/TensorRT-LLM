# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 FP8-rowwise GEMM for FLUX.3's FP8R (BF16 checkpoint-excluded) blocks (idea 122).

FlashInfer overlay entry ``csrc/flux3_fp8r_gemm_sm120.cu`` (module ``flux3_fp8r_gemm_sm120``):
a CUTLASS SM120 E4M3 x E4M3 GEMM with FP32 accumulation, CLC persistent tile scheduler, PDL, and
an EVT epilogue BF16(w_scale * (x_scale * acc)) with two separate FP32 RN multiplies. The served
arm is Pingpong 128x128x64 with 5 smem stages; torch._scaled_mm (PyTorch's SM120 rowwise CUTLASS
kernel) is Cooperative 128x128x128 with 2 stages. Same per-element k32 MMA chain and the same
epilogue association, so the output is bitwise equal to torch._scaled_mm (idea 122 STEP 0(c)).

TRTLLM_FLUX3_FP8R_GEMM=1 (default) routes image.0's 4 FP8R GEMMs here:
- gate_up: primsts.PreparedFP8Sm120.gemm (through the custom op trtllm::flux3_sm120_fp8r_gemm);
- QKV: flux3_fp8r_qkv.fp8r_qkv_qknorm_rope is rebound at prepare time (``install``) to
  trtllm::flux3_sm120_fp8r_qkv_qknorm_rope_cg, the same op with this GEMM (no text change there);
- down / attn_out: flux3_primsts.fp8r_out_pair_q's launch_down and _fp8r_linear.
TRTLLM_FLUX3_FP8R_GEMM=0 keeps torch._scaled_mm at all 4 sites and rebinds nothing.
The module is JIT-built at prepare time (``install``); a call never builds.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

import torch

# Arm: 0 = Cooperative 128x128x64, 1 = Pingpong 128x128x64 (served), 2 = Cooperative 128x128x128,
# 3 = Cooperative 128x256x64, 4 = Cooperative 256x128x64 (3/4: order 1 only).
ARM = int(os.environ.get("TRTLLM_FLUX3_FP8R_GEMM_ARM", "1"))
# Epilogue association: 1 -> BF16(w_scale * (x_scale * acc)), bitwise equal to torch._scaled_mm on
# every image.0 shape (0 -> BF16(x_scale * (w_scale * acc)) differs in ~5e-6 of elements by 1 ulp).
ORDER = 1

_WORKSPACE_BYTES = 1024 * 1024
_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
# Routed calls by weight N (counted in the op bodies, i.e. in the worker process).
ROUTES: Counter = Counter()


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
    """Build the module and allocate the workspaces at model load."""
    if not enabled():
        return
    mod = module()
    need = int(mod.fp8r_workspace_size(1 << 16, 18432, 9216, ARM))
    if need > _WORKSPACE_BYTES:
        raise RuntimeError(f"FP8R GEMM workspace {need} B exceeds {_WORKSPACE_BYTES} B")
    workspace(device, 0)
    workspace(device, 1)


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

    activation E4M3 [M, K] contiguous, weight E4M3 [N, K] contiguous, activation_scale FP32 with
    M elements, weight_scale FP32 with N elements. Current stream; allocates only the output.
    """
    m = activation.shape[0]
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
        int(ORDER if order is None else order),
    )
    ROUTES[int(n)] += 1
    return out


@torch.library.custom_op("trtllm::flux3_sm120_fp8r_gemm", mutates_args=(), device_types="cuda")
def fp8r_gemm_op(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """torch._scaled_mm(activation, weight.t(), rowwise scales, bf16) through the overlay GEMM."""
    return gemm(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.float().reshape(-1).contiguous(),
        weight_scale.float().reshape(-1).contiguous(),
    )


@fp8r_gemm_op.register_fake
def _fp8r_gemm_op_fake(activation, weight, activation_scale, weight_scale):
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8r_qkv_qknorm_rope_cg", mutates_args=(), device_types="cuda"
)
def fp8r_qkv_qknorm_rope_cg(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    planar: bool = False,
) -> torch.Tensor:
    """flux3_fp8r_qkv.fp8r_qkv_qknorm_rope with its torch._scaled_mm replaced by ``gemm``
    (bitwise equal); the QK-norm/RoPE pass and its arguments are exactly the original's."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
        qknorm_rope,
    )

    from .flux3_primsts import _token_positions

    out = gemm(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.float().reshape(-1).contiguous(),
        weight_scale.float().reshape(-1).contiguous(),
    )
    width = out.shape[1] // 3
    positions = _token_positions(out.shape[0], out.device)
    dst = torch.empty_like(out) if planar else out
    return qknorm_rope(
        out[:, :width],
        out[:, width:2 * width],
        out[:, 2 * width:],
        dst,
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions,
        planar=planar,
    )


@fp8r_qkv_qknorm_rope_cg.register_fake
def _fp8r_qkv_qknorm_rope_cg_fake(
    activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin, planar=False
):
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


# The original flux3_fp8r_qkv.fp8r_qkv_qknorm_rope, saved exactly once by install().
_ORIGINAL_QKV_OP = None


def install(device: torch.device) -> bool:
    """Build at load and rebind the FP8R QKV op (idempotent). Returns True when routed."""
    global _ORIGINAL_QKV_OP
    if not enabled():
        return False
    preload(device)
    from . import flux3_fp8r_qkv

    if _ORIGINAL_QKV_OP is None:
        _ORIGINAL_QKV_OP = flux3_fp8r_qkv.fp8r_qkv_qknorm_rope
    flux3_fp8r_qkv.fp8r_qkv_qknorm_rope = fp8r_qkv_qknorm_rope_cg
    return True
