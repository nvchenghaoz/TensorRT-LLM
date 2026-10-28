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

"""SM120 NVFP4 GEMM into a strided slice of a caller-given BF16 output (FLUX.3 QKV in place).

FlashInfer overlay entry ``csrc/flux3_fp4_gemm_sm120_strided.cu`` (module
``flux3_fp4_gemm_sm120_strided``, export ``fp4_gemm_strided``): FlashInfer's stock SM120 CUTLASS
instantiation (bf16, TileShape 128x128x128 FP4 elements, StaticPersistent DP, LinearCombination
epilogue), bitwise equal to TRT-LLM FP4GemmRunner tactic 0, with D at ``out + offset`` and row
stride ``ldd``. The module is JIT-built and its workspaces allocated at model load
(``preload``); a call never builds or allocates.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

# DP StaticPersistent needs (almost) no workspace; one small buffer per stream, never per call.
_WORKSPACE_BYTES = 1024 * 1024

_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}


@lru_cache(maxsize=1)
def inplace_enabled() -> bool:
    """TRTLLM_FLUX3_SM120_QKV_INPLACE=0 restores the separate-buffer QKV path."""
    return os.environ.get("TRTLLM_FLUX3_SM120_QKV_INPLACE", "1").strip() != "0"


@lru_cache(maxsize=1)
def module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_fp4_gemm_sm120_strided",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_fp4_gemm_sm120_strided.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


@lru_cache(maxsize=1)
def qkv_schedule() -> str:
    """Mainloop schedule of the QKV plane GEMMs.

    'pingpong' (default): overlay entry ``csrc/flux3_fp4_gemm_sm120_pingpong.cu`` (module
    ``flux3_fp4_gemm_sm120_pingpong_route``, export ``gemm_into_pingpong``), the same CUTLASS SM120
    NVFP4 128x128x128 StaticPersistent kernel with KernelTmaWarpSpecializedPingpong: the two
    consumer warpgroups alternate tiles, so one tile's epilogue overlaps the next tile's
    mainloop. Each output element sees the same MMA sequence and epilogue, so it is bitwise
    equal to the Cooperative entry (and to TRT-LLM t0).
    TRTLLM_FLUX3_SM120_QKV_SCHED=cooperative restores the stock Cooperative strided entry.
    """
    sched = os.environ.get("TRTLLM_FLUX3_SM120_QKV_SCHED", "pingpong").strip().lower()
    if sched not in ("pingpong", "cooperative"):
        raise ValueError(
            f"TRTLLM_FLUX3_SM120_QKV_SCHED must be 'pingpong' or 'cooperative', got {sched!r}"
        )
    return sched


@lru_cache(maxsize=1)
def pingpong_module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_fp4_gemm_sm120_pingpong_route",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_fp4_gemm_sm120_pingpong.cu"],
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "-DFLUX3_PP_ONLY",
        ],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


_SLOTS = 3  # QKV planes: q on the current stream, k and v on the two QKV side streams.


def workspace(device: torch.device, slot: int) -> torch.Tensor:
    """Workspace of one concurrent GEMM slot (concurrent GEMMs must not share one).

    Keyed by slot, not by stream id, so a CUDA-graph capture stream reuses the load-time buffers
    instead of allocating inside the capture.
    """
    key = (_device_index(device), slot)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACES[key] = ws
    return ws


def preload(device: torch.device) -> None:
    """Build the module and allocate every slot's workspace at model load."""
    if not inplace_enabled():
        return
    if qkv_schedule() == "pingpong":
        pingpong_module()
    else:
        module()
    for slot in range(_SLOTS):
        workspace(device, slot)


def gemm_into(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    offset: int,
    ldd: int,
    slot: int,
) -> None:
    """out.flat[offset + r*ldd + c] = BF16(alpha * (A @ W^T))[r, c], on the current stream."""
    if qkv_schedule() == "pingpong":
        fn = pingpong_module().gemm_into_pingpong
    else:
        fn = module().fp4_gemm_strided
    fn(
        activation,
        weight,
        activation_scale,
        weight_scale,
        alpha,
        out,
        int(offset),
        int(ldd),
        workspace(activation.device, slot),
    )
