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

"""Large-M (2048^2: M ~ 16.4K-33.3K) tile routes of FLUX.3's SM120 denoiser GEMMs (idea 17).

Every main tile was chosen at the 1024^2 workload (M ~ 4.2K), where wave quantization favours
128x128 tiles. At M >= LARGE_M_MIN the per-tile operand reuse decides, and these add-only
variants are faster (STEP 0, experiments/ascent/idea17_tiles):

- 'qkv3': batched L=3 QKV GEMM at TileShape 256x128x128, Cooperative, CLC dynamic scheduler
  (overlay csrc/flux3_fp4_qkv3_t256_sm120.cu, module flux3_fp4_qkv3_t256_sm120) instead of main's
  Pingpong 128x128x128 (flux3_sm120_strided_gemm.gemm_qkv3).
- 'pair': the one-launch NVFP4 tail pair (down || attn_out) at TileShape 256x128x128 (overlay
  csrc/flux3_fp4_lpt_pair_t256_sm120.cu, module flux3_fp4_lpt_pair_t256_sm120): main's kernel,
  scheduler, tile order and raster with only the tile shape changed.
- 'fp8r': image.0's FP8-rowwise GEMMs on the existing Cooperative 128x256x64 / 256x128x64 arms
  of flux3_fp8r_gemm_sm120 (arms 3/4) instead of the Pingpong 128x128x64 arm 1.

Each variant runs the same per-element MMA chain over K (k64 FP4 / k32 FP8) and the same
epilogue as main's kernel, so outputs are bitwise equal; only the tile -> CTA mapping changes.
Routing is by the static row count of the call (a CUDA graph captures one route; no per-call
query or autotune). Modules are JIT-built at model load (``preload``).

TRTLLM_FLUX3_SM120_LARGE_M_TILES: '1' (default) = all routes, '0' = none (main's modules and
kernels for every call), or a comma list of route names (e.g. 'qkv3,pair').
"""

from __future__ import annotations

import os
from functools import lru_cache

# Calls with at least this many rows take the large-M route (served Ms: 16384, ~16.9K, ~33.3K;
# the 1024^2 Ms ~4.2K keep main's kernels, where these tiles are slower).
LARGE_M_MIN = 8192

ROUTES = ("qkv3", "pair", "fp8r")

# Raster of the 256x128 QKV3 kernel (1 = AlongM; measured best at M 16896 and 33280).
QKV3_T256_RASTER = 1
QKV3_T256_SWIZZLE = 1
QKV3_T256_DYNAMIC = 1


@lru_cache(maxsize=1)
def _enabled_routes() -> frozenset:
    value = os.environ.get("TRTLLM_FLUX3_SM120_LARGE_M_TILES", "1").strip().lower()
    if value in ("1", "all", ""):
        return frozenset(ROUTES)
    if value in ("0", "none"):
        return frozenset()
    names = frozenset(v.strip() for v in value.split(",") if v.strip())
    bad = names - set(ROUTES)
    if bad:
        raise ValueError(
            f"TRTLLM_FLUX3_SM120_LARGE_M_TILES: unknown routes {sorted(bad)}; "
            f"use 1, 0 or a comma list of {ROUTES}"
        )
    return names


def enabled(route: str) -> bool:
    return route in _enabled_routes()


def use(route: str, rows: int) -> bool:
    """True when a call with `rows` rows takes the large-M route `route`."""
    return rows >= LARGE_M_MIN and enabled(route)


def _build(name: str, source: str):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / source],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


@lru_cache(maxsize=1)
def qkv3_module():
    return _build("flux3_fp4_qkv3_t256_sm120", "flux3_fp4_qkv3_t256_sm120.cu")


@lru_cache(maxsize=1)
def pair_module():
    return _build("flux3_fp4_lpt_pair_t256_sm120", "flux3_fp4_lpt_pair_t256_sm120.cu")


def fp8r_arm(n: int) -> int:
    """FP8R arm of a large-M call: Cooperative 128x256x64 for gate_up (N 18432), 256x128x64
    otherwise (both bitwise to torch._scaled_mm, idea 17 STEP 0)."""
    return 3 if n >= 12288 else 4


def preload_qkv3(workspace_bytes: int) -> None:
    if not enabled("qkv3"):
        return
    mod = qkv3_module()
    need = int(mod.qkv3_t256_workspace_size(1 << 16, 3072, 3072, QKV3_T256_DYNAMIC))
    if need > workspace_bytes:
        raise RuntimeError(f"large-M QKV3 workspace {need} B exceeds {workspace_bytes} B")


def preload_pair() -> None:
    if enabled("pair"):
        pair_module()


def gemm_qkv3(activation, weight, activation_scale, weight_scale, alpha, out, workspace) -> None:
    qkv3_module().gemm_qkv3_t256(
        activation,
        weight,
        activation_scale,
        weight_scale,
        alpha,
        out,
        workspace,
        int(QKV3_T256_RASTER),
        int(QKV3_T256_SWIZZLE),
        int(QKV3_T256_DYNAMIC),
    )
