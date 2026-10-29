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

Idea 32 (L2-aware tile order): at rows >= RASTER_MIN_ROWS (the ~32K bucket, i2i single blocks at
M=32848) the pair runs on this 128x128 module with its tiles rasterized AlongN (raster_for), not
on the t256 AlongM module: AlongM streams down's whole A operand (M x 9216 FP4, ~150 MB, over the
128 MB L2) from DRAM once per N column. Below that, routes are main's (t256 at 8192 <= rows,
this module at AlongM below). TRTLLM_FLUX3_LPT_RASTER=0 or TRTLLM_FLUX3_SM120_LARGE_M_TILES
without 'pair' restores main's route and order at every M. Outputs are bitwise identical.
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
def raster_enabled() -> bool:
    """TRTLLM_FLUX3_LPT_RASTER=0 restores main's route and tile order at every M."""
    return os.environ.get("TRTLLM_FLUX3_LPT_RASTER", "1").strip() != "0"


# (raster, max_swizzle) of this module: raster 0 = AlongM (CUTLASS's grouped default, M fastest
# within an N column; main's order), 1 = AlongN (N fastest within an M panel).
_MAIN_ORDER = (0, 1)
_ALONG_N = (1, 1)
# Rows from which the pair takes this module at AlongN instead of the t256 AlongM module: the
# ~32K bucket (M range, never M equality). Idea 32 gate 1 (experiments/ascent/idea32_raster,
# CUDA-event medians, us/call): M=32848 128-AlongN 1667 vs t256-AlongM 2481 (A/A 0.4%);
# 16464 895 vs 861 and 16384 895 vs 837 (t256 stays).
RASTER_MIN_ROWS = 24576


def along_n(rows: int) -> bool:
    """True when a call with `rows` rows runs this module at AlongN (static per shape)."""
    from . import flux3_sm120_large_m

    return (
        rows >= RASTER_MIN_ROWS and raster_enabled() and flux3_sm120_large_m.enabled("pair")
    )


def raster_for(rows: int) -> tuple[int, int]:
    return _ALONG_N if along_n(rows) else _MAIN_ORDER


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
    # Idea 17: the 256x128x128 large-M pair module (TRTLLM_FLUX3_SM120_LARGE_M_TILES).
    from . import flux3_sm120_large_m

    flux3_sm120_large_m.preload_pair()
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
    raster: tuple[int, int] | None = None,
) -> None:
    """down_out = BF16(down_alpha * down_activation @ down_weight^T), attn_out likewise, in one
    launch on the current stream (down's tiles first).

    At M >= flux3_sm120_large_m.LARGE_M_MIN the same kernel at TileShape 256x128x128 runs
    (bitwise equal; TRTLLM_FLUX3_SM120_LARGE_M_TILES=0 keeps main's module at every M), except at
    rows >= RASTER_MIN_ROWS, where this module runs at AlongN (along_n). raster=(order,
    max_swizzle) forces this module at that order (diagnostics)."""
    from . import flux3_sm120_large_m

    rows = down_activation.shape[0]
    if raster is None and not along_n(rows) and flux3_sm120_large_m.use("pair", rows):
        flux3_sm120_large_m.pair_module().lpt_pair(
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
        return
    order, swizzle = raster_for(rows) if raster is None else raster
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
        int(order),
        int(swizzle),
    )
