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

"""SM120 NVFP4 block-tail pair with the block's gated residual in the attn_out tiles' epilogue.

    out = x + gate * (attn_out(attn) + down(packed))      (BF16 round after every op)

as ONE persistent CUTLASS launch (FlashInfer overlay ``csrc/flux3_fp4_lpt_pair_residual_sm120.cu``,
module ``flux3_fp4_lpt_pair_residual_sm120``): down's 128x128 tiles (K=9216) first, then
attn_out's (K=3072), every tile claimed from one atomic counter. A down tile stores BF16(alpha *
acc) to a scratch mlp_y and publishes a per-warp flag (the launch epoch); the attn_out tile (m, n)
waits on it, reads its mlp_y fragment, x and the gate row and writes out. Each tile's MMA sequence
and BF16(alpha * acc) rounding are the stock LPT pair's (flux3_sm120_lpt_pair.py), and the residual
math is Inductor's emulate_precision_casts kernel's (f32 .rn ops, BF16 round after the add, the mul
and the final add), so ``out`` is bitwise main's ``x + mod_gate * (attn_y + mlp_y)``.

The gate is a [1, R, D] row table plus an optional per-row index ([1, S, 1] bool or uint8; row 0
without one): the single blocks' [1, 2, D] table + select and the mode blocks' [1, 1, D] row.

TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=0 restores main's pair + Inductor residual.

Raster (idea 59): the grouped scheduler's default walks AlongM. raster_for() picks the launch's
raster statically by M range (M < 8192 and M >= 24576: the default, i.e. main's launch;
8192 <= M < 24576: AlongN). Only the tile -> CTA map changes, so outputs are bitwise equal.
TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_RASTER=0 restores main's launch at every M.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

# [0, 8 KB) args (tile counter, launch epoch, grouped-GEMM arrays), then the CUTLASS workspace
# (per-CTA tensormaps, ~0.2 MB), flags from 4 MB: 8 uint32 per 128x128 tile (1M flags = 131072
# tiles, M up to ~700k rows at N=3072). Zeroed once at allocation: the epoch starts at 0 and every
# launch increments it before use, so a zero flag never matches.
_WORKSPACE_BYTES = 8 * 1024 * 1024

_WORKSPACES: dict[int, torch.Tensor] = {}


@lru_cache(maxsize=1)
def enabled() -> bool:
    """TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=0 keeps main's pair + Inductor gated residual."""
    return os.environ.get("TRTLLM_FLUX3_SM120_PAIR_RESIDUAL", "1").strip() != "0"


@lru_cache(maxsize=1)
def raster_enabled() -> bool:
    """TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_RASTER=0 keeps main's launch (scheduler default raster,
    AlongM for this grouped launch) at every M."""
    return os.environ.get("TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_RASTER", "1").strip() != "0"


# Raster by M range (CUTLASS RasterOrderOptions: 0 scheduler default = main, 1 AlongM, 2 AlongN).
# Range buckets, never M equality (the served M is image tokens + the padded text bucket).
_RASTER_MID_MIN = 8192  # 8192 <= M < 24576: t2i single 16464, mode / image_cond 16384
_RASTER_LARGE_MIN = 24576  # M >= 24576: i2i single 32848
_RASTER_MID = 2  # AlongN: 16464 1033.6 -> 1003.2 us, 16384 1012.6 -> 985.0 us (jobs 332/344)
# M >= 24576 keeps main's launch: the module is not reached there on this base, and the route
# decision belongs to idea 62 (job 344 prior at 32848 u8 K=3: AlongM 3356 us, AlongN 1853 us).
_RASTER_LARGE = 0


def raster_for(rows: int) -> int:
    """The raster argument of the launch at M=rows (static by M range; M < 8192 is main's)."""
    if not raster_enabled() or rows < _RASTER_MID_MIN:
        return 0
    return _RASTER_LARGE if rows >= _RASTER_LARGE_MIN else _RASTER_MID


@lru_cache(maxsize=1)
def module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_fp4_lpt_pair_residual_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_fp4_lpt_pair_residual_sm120.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def workspace(device: torch.device) -> torch.Tensor:
    """One zero-initialized workspace per device (the op runs on one stream; stream ordered)."""
    key = _device_index(device)
    ws = _WORKSPACES.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            # A zero fill captured into a graph would reset the epoch on every replay, so stale
            # flags of the previous replay would match. Allocate at load (preload) or warmup.
            raise RuntimeError("flux3_sm120_lpt_pair_residual: workspace first used in capture")
        ws = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACES[key] = ws
    return ws


def preload(device: torch.device) -> None:
    """Build the module and allocate the workspace at model load."""
    if not enabled():
        return
    module()
    workspace(device)


def available(device: torch.device) -> bool:
    """True when the fused op may run: switch on, SM120 CUTLASS LINEAR backend, LPT kernel on."""
    if not enabled() or device.type != "cuda":
        return False
    from . import flux3_sm120_lpt_pair as lpt
    from .flux3_primsts import _sm120_linear_backend

    return lpt.enabled() and _sm120_linear_backend() == "cutlass"


# Row range the fused op serves. Fused-128 at Heuristic raster (AlongM) loses to main's t256 pair
# + Inductor gated residual at large M: i2i single blocks M=32848 ([1,3,D] table + u8 index)
# 3.360 vs 3.025 ms per tail, outputs bitwise (branch 15 job 300; A/A 0.7 us). At M=16464 it wins
# (1.009 vs 1.114 ms). So M >= _DECLINE_MIN_ROWS takes main's route (range check, never an M
# equality). Before branch 15 no caller reached this op at M >= 24576 (the i2i singles carried a
# per-token gate), so every route that existed is unchanged.
_DECLINE_MIN_ROWS = 24576


def serves(rows: int) -> bool:
    """Whether the fused op takes a call of `rows` output rows (see _DECLINE_MIN_ROWS)."""
    return 0 < rows < _DECLINE_MIN_ROWS


def gate_form(
    gate: torch.Tensor, index: torch.Tensor | None, rows: int, width: int
) -> bool:
    """Whether (gate, index) is a row table this op takes: [1, R, D] bf16 with a [1, rows, 1]
    bool/uint8 index, or [1, 1, D] without one, at a row count it serves."""
    if not serves(rows):
        return False
    if gate.dim() != 3 or gate.shape[0] != 1 or gate.shape[-1] != width:
        return False
    if gate.dtype != torch.bfloat16:
        return False
    if index is None:
        return gate.shape[1] == 1
    return (
        index.dtype in (torch.bool, torch.uint8)
        and index.numel() == rows
        and 1 <= gate.shape[1] <= 255
    )


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """The kernel reads/writes 16-B row segments: copy a (contiguous) view that is not 16-B
    aligned (a sliced view; fresh allocations always are)."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


@torch.library.custom_op(
    "trtllm::flux3_sm120_nvfp4_out_pair_residual", mutates_args=(), device_types="cuda"
)
def nvfp4_out_pair_residual(
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    attn: torch.Tensor,
    attn_scales: torch.Tensor | None,
    attn_weight: torch.Tensor,
    attn_weight_scale: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor | None,
) -> torch.Tensor:
    """x + gate[gate_index] * (attn_out(attn) + down(down_activation)) as [M, N] bf16.

    down_activation/down_activation_scale: the packed-SwiGLU FP4 hidden and its 128x4 scales.
    attn: attn_out's activation, already NVFP4 ([M, K/2] uint8 with attn_scales, as the FMHA
    epilogue writes it) or BF16 [M, K] (attn_scales None: quantized here with attn_out's static
    scale, as nvfp4_out_pair does). x: [..., N] bf16 with M rows. gate: [1, R, N] bf16 row table;
    gate_index: [1, M, 1] bool/uint8 row per output row, or None (row 0).
    """
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar

    from .flux3_primsts import nvfp4_owner, nvfp4_quantize_prepared

    down_owner = nvfp4_owner(down_weight)
    attn_owner = nvfp4_owner(attn_weight)
    down_weight, down_weight_scale = down_weight.detach(), down_weight_scale.detach()
    attn_weight, attn_weight_scale = attn_weight.detach(), attn_weight_scale.detach()
    if attn_scales is None:
        data, scales = nvfp4_quantize_prepared(attn.detach(), attn_weight)
    else:
        data, scales = attn.detach(), attn_scales.detach()
    device = data.device
    rows = data.shape[0]
    width = down_weight.shape[0]
    x2 = _aligned(x.detach().reshape(rows, width).contiguous())
    table = _aligned(gate.detach().reshape(-1, width).contiguous())
    if gate_index is None:
        index = table  # not read
        has_index = 0
    else:
        index = gate_index.detach().reshape(-1).contiguous().view(torch.uint8)
        has_index = 1
    mlp_y = torch.empty((rows, width), device=device, dtype=torch.bfloat16)
    out = torch.empty((rows, width), device=device, dtype=torch.bfloat16)
    module().lpt_pair_residual(
        down_activation.detach(),
        down_weight,
        down_activation_scale.detach(),
        down_weight_scale,
        _scalar(device, down_owner.input_global_scale * down_owner.weight_global_scale),
        mlp_y,
        data,
        attn_weight,
        scales,
        attn_weight_scale,
        _scalar(device, attn_owner.input_global_scale * attn_owner.weight_global_scale),
        x2,
        table,
        index,
        has_index,
        out,
        workspace(device),
        raster_for(rows),
    )
    return out


@nvfp4_out_pair_residual.register_fake
def _nvfp4_out_pair_residual_fake(
    down_activation,
    down_weight,
    down_activation_scale,
    down_weight_scale,
    attn,
    attn_scales,
    attn_weight,
    attn_weight_scale,
    x,
    gate,
    gate_index,
):
    return x.new_empty((attn.shape[0], down_weight.shape[0]), dtype=torch.bfloat16)


def run_raw(
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
    attn_data: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_scales: torch.Tensor,
    attn_weight_scale: torch.Tensor,
    attn_alpha: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor | None,
    mlp_y: torch.Tensor | None = None,
    raster: int | None = None,
) -> torch.Tensor:
    """The kernel with explicit device alphas (microbenchmarks/tests). Returns out [M, N].
    raster None routes by M (raster_for); 0/1/2 forces default/AlongM/AlongN."""
    rows, width = attn_data.shape[0], down_weight.shape[0]
    table = _aligned(gate.reshape(-1, width).contiguous())
    index = table if gate_index is None else gate_index.reshape(-1).contiguous().view(torch.uint8)
    if mlp_y is None:
        mlp_y = torch.empty((rows, width), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((rows, width), device=x.device, dtype=torch.bfloat16)
    module().lpt_pair_residual(
        down_activation, down_weight, down_activation_scale, down_weight_scale, down_alpha, mlp_y,
        attn_data, attn_weight, attn_scales, attn_weight_scale, attn_alpha,
        _aligned(x.reshape(rows, width).contiguous()), table, index,
        0 if gate_index is None else 1, out,
        workspace(x.device),
        raster_for(rows) if raster is None else int(raster),
    )
    return out
