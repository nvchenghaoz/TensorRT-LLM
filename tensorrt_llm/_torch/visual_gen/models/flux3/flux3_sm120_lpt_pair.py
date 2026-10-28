# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120 NVFP4 block-tail pair (down K=9216 + attn_out K=3072) as ONE persistent CUTLASS launch.

FlashInfer overlay entry ``csrc/flux3_fp4_lpt_pair_sm120.cu`` (module ``flux3_fp4_lpt_pair_sm120``,
exports ``lpt_pair`` and ``lpt_pair_info``): a 2-group SM120 block-scaled ptr-array GEMM
(128x128x128 FP4 tiles, Cooperative mainloop, BF16(alpha * acc) epilogue, per-group alpha) whose
tile scheduler hands out down's tiles first and then attn_out's from a global atomic counter
(longest-first dynamic order). Each tile runs the same MMA sequence as the stock DP kernel, so each
output is bitwise equal to FlashInfer tactic 17 / TRT-LLM FP4GemmRunner tactic 0.

TRTLLM_FLUX3_SM120_PAIR_KERNEL=0 restores the two-stream pair (down on the current stream,
attn_out on a side stream). TRTLLM_FLUX3_SM120_PAIR_SCHED=static selects the stock static
round-robin order of the same kernel (diagnostics).
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

# Args region (8 KB) + CUTLASS workspace (per-CTA tensormaps; ~0.2 MB at 188 CTAs). Sized for
# diagnostic grids up to a few thousand CTAs. Allocated once per device at model load.
_WORKSPACE_BYTES = 16 * 1024 * 1024

_WORKSPACES: dict[int, torch.Tensor] = {}


@lru_cache(maxsize=1)
def enabled() -> bool:
    """TRTLLM_FLUX3_SM120_PAIR_KERNEL=0 restores the two-stream tail pair."""
    return os.environ.get("TRTLLM_FLUX3_SM120_PAIR_KERNEL", "1").strip() != "0"


_SCHED_MODES = {"static": 0, "dynamic": 1, "ondemand": 2}


@lru_cache(maxsize=1)
def sched_mode() -> int:
    """Tile order of the one-launch pair: 'dynamic' (default) claims tiles from an atomic counter
    one tile ahead, 'ondemand' when the DMA warp asks, 'static' is CUTLASS's round-robin."""
    sched = os.environ.get("TRTLLM_FLUX3_SM120_PAIR_SCHED", "dynamic").strip().lower()
    if sched not in _SCHED_MODES:
        raise ValueError(
            f"TRTLLM_FLUX3_SM120_PAIR_SCHED must be one of {sorted(_SCHED_MODES)}, got {sched!r}"
        )
    return _SCHED_MODES[sched]


@lru_cache(maxsize=1)
def module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_fp4_lpt_pair_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_fp4_lpt_pair_sm120.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def workspace(device: torch.device) -> torch.Tensor:
    """The one workspace per device (the pair runs on one stream; calls are stream ordered)."""
    key = _device_index(device)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACES[key] = ws
    return ws


def preload(device: torch.device) -> None:
    """Build the module and allocate the workspace at model load."""
    if not enabled():
        return
    module()
    workspace(device)


def run(
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
    down_out: torch.Tensor,
    attn_activation: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_activation_scale: torch.Tensor,
    attn_weight_scale: torch.Tensor,
    attn_alpha: torch.Tensor,
    attn_out: torch.Tensor,
    mode: int | None = None,
    grid_ctas: int = 0,
) -> None:
    """down_out = BF16(down_alpha * down_activation @ down_weight^T), attn_out likewise, in one
    launch on the current stream (down's tiles first)."""
    module().lpt_pair(
        down_activation,
        down_weight,
        down_activation_scale,
        down_weight_scale,
        down_alpha,
        down_out,
        attn_activation,
        attn_weight,
        attn_activation_scale,
        attn_weight_scale,
        attn_alpha,
        attn_out,
        workspace(down_activation.device),
        sched_mode() if mode is None else int(mode),
        int(grid_ctas),
    )
