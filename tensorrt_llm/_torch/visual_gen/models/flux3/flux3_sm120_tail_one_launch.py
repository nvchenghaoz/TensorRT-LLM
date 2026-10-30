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

"""SM120 NVFP4 block tail in one launch: x + gate * (attn_out(attn) + down(hidden)).

FlashInfer overlay entry ``csrc/flux3_fp4_pair_residual_sm120.cu`` (exports ``pair_residual``):
per 128x128 output tile, attn_out's K loop (K=3072) then down's (K=9216) through one smem
pipeline, the BF16 attn_out tile kept in registers, and an epilogue that applies main's residual
pointwise with its BF16 rounding points (bf16(attn_y + mlp_y), bf16(gate * out), bf16(x + prod)).
Each GEMM tile runs the stock SM120 DP kernel's MMA sequence, so x_new is bitwise equal to main's
pair + Inductor residual. Static persistent tile order along N (row-major tiles).

TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH=0 restores main's tail path exactly. The route is taken for
tail calls with M >= TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH_MIN_M rows (a static M-range bucket fixed
at weight preparation); smaller M keeps main's path.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

_ENV = "TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH"
# Default M-range bucket (rows of the block tail): M >= 24576, i.e. the i2i single blocks
# (served M 32848, per-token gate), where this kernel beats main's pair + Inductor residual. The
# 8192 <= M < 24576 bucket (t2i single blocks 16464, mode blocks 16384) stays on main's route
# (fusion's flag-wait residual pair), and M < 8192 always keeps main's path.
_DEFAULT_MIN_M = 24576


def _env(suffix: str, default: str) -> str:
    return os.environ.get(_ENV + suffix, default).strip()


@lru_cache(maxsize=1)
def enabled() -> bool:
    """TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH=0 keeps main's block-tail path."""
    return _env("", "1") != "0"


@lru_cache(maxsize=1)
def min_rows() -> int:
    """Smallest tail M routed to the one-launch kernel (TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH_MIN_M)."""
    return max(int(_env("_MIN_M", str(_DEFAULT_MIN_M))), 1)


@lru_cache(maxsize=1)
def prefetch_x() -> int:
    """Producer-warp L2 prefetch of the residual tile (..._PREFETCH, default on)."""
    return int(_env("_PREFETCH", "1"))


@lru_cache(maxsize=1)
def early_loads() -> int:
    """Issue the residual x / gate loads before down's K loop (..._EARLY, default on)."""
    return int(_env("_EARLY", "1"))


@lru_cache(maxsize=1)
def module():
    return jit_spec().build_and_load()


def jit_spec():
    """Default: direct register stores of x_new (5 mainloop stages). ..._STAGED=1 (experimental,
    NOT bitwise-safe: job 257 saw intermittent mismatches at M=32848): x_new staged in smem and
    written by asynchronous bulk copies (3 stages). ..._STAGES overrides the stage count."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    extra = []
    stages = _env("_STAGES", "")
    name = "flux3_fp4_tail_one_launch_sm120"
    if _env("_STAGED", "0") == "1":
        extra.append("-DFLUX3_TRES_STAGED")
        name += "_staged"
        stages = stages or "3"
    if stages:
        extra.append(f"-DFLUX3_TRES_STAGES={int(stages)}")
        name += f"_s{int(stages)}"
    return gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_fp4_pair_residual_sm120.cu"],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP4"] + extra,
        extra_cflags=["-DFAST_BUILD"],
    )


def preload(device: torch.device) -> None:
    """Build the module at model load (never inside a request)."""
    if enabled():
        module()


def run(
    attn_activation: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_activation_scale: torch.Tensor,
    attn_weight_scale: torch.Tensor,
    attn_alpha: torch.Tensor,
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    select: torch.Tensor | None,
    out: torch.Tensor,
    gate_mode: int | None = None,
    mode: int = 0,
    dbg_attn: torch.Tensor | None = None,
    dbg_mlp: torch.Tensor | None = None,
    grid_ctas: int = 0,
    prefetch: int | None = None,
    early: int | None = None,
) -> None:
    """out = x + gate[row] * (BF16(attn_alpha * attn_act @ attn_w^T) + BF16(down ...)) with
    main's BF16 rounding points, one launch on the current stream.

    gate_mode 0: gate is one [N] row; 1: gate is a [K, N] table and select ([M] bool/uint8)
    holds each row's index; 2: gate is per row ([M, N]). Default: 1 with select, else 0.
    mode 1 (tests): write attn_y / mlp_y to dbg_attn / dbg_mlp instead of out; mode 2
    (diagnostics): mainloops only."""
    if gate_mode is None:
        gate_mode = 1 if select is not None else 0
    module().pair_residual(
        attn_activation,
        attn_weight,
        attn_activation_scale,
        attn_weight_scale,
        attn_alpha,
        down_activation,
        down_weight,
        down_activation_scale,
        down_weight_scale,
        down_alpha,
        x,
        gate,
        gate if select is None else select,
        int(gate_mode),
        out,
        out if dbg_attn is None else dbg_attn,
        out if dbg_mlp is None else dbg_mlp,
        int(mode),
        int(grid_ctas),
        prefetch_x() if prefetch is None else int(prefetch),
        early_loads() if early is None else int(early),
    )
