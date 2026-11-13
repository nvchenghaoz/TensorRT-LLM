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
"""Bitwise torch-SDPA-FA2 attention for SM120 at higher occupancy.

torch's FlashAttention-2 dispatch treats only compute capability 8.x as "sm8x", so on SM120
(RTX PRO 6000) a BF16 d=128 non-causal call runs ``Flash_fwd_kernel_traits<128, 128, 64, 4>``:
4 warps and 64 KB smem per CTA, one CTA per SM. This module JIT-compiles torch's own FA2
forward sources (the ones torch was built from) with torch's compile definitions and runs the
same per-row math with a 256-row, 8-warp CTA (and a 64-row tile for the rows left after whole
waves). The output is bitwise equal to ``F.scaled_dot_product_attention`` on the flash backend.

Disable with ``TLLM_FA2_SM120=0``. The FA2 and CUTLASS headers default to torch's source tree in
the image (``/opt/pytorch/pytorch``); override with ``TLLM_FA2_SM120_FLASH_SRC`` and
``TLLM_FA2_SM120_CUTLASS_INC``. When anything is missing the caller keeps using SDPA.
"""

import os
import threading
import time
from typing import Optional

import torch

from tensorrt_llm.logger import logger

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TORCH_SRC = "/opt/pytorch/pytorch"
_DEFINES = [
    # torch's flash_attention target (aten/src/ATen/CMakeLists.txt) compile definitions.
    "-DFLASH_NAMESPACE=tllm_fa2_sm120",
    "-DFLASHATTENTION_DISABLE_ALIBI",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DUNFUSE_FMA",
]
# Leftover rows run as a second 64-row-tile launch only when they fill at most this
# fraction of a 256-row wave; otherwise one 256-row launch covers all rows.
_MAX_REM_FRAC = 0.5

_LOG_EVERY = 500
_calls = 0

_lock = threading.Lock()
_ext = None
_ext_state: Optional[bool] = None  # None: not tried, True: loaded, False: failed


def _paths():
    flash_src = os.environ.get(
        "TLLM_FA2_SM120_FLASH_SRC",
        os.path.join(_DEFAULT_TORCH_SRC, "third_party/flash-attention/csrc/flash_attn/src"),
    )
    cutlass_inc = os.environ.get(
        "TLLM_FA2_SM120_CUTLASS_INC", os.path.join(_DEFAULT_TORCH_SRC, "third_party/cutlass/include")
    )
    return flash_src, cutlass_inc


def _load_ext():
    global _ext, _ext_state
    if _ext_state is not None:
        return _ext
    with _lock:
        if _ext_state is not None:
            return _ext
        try:
            flash_src, cutlass_inc = _paths()
            for p in (
                os.path.join(flash_src, "flash_fwd_kernel.h"),
                os.path.join(flash_src, "flash_fwd_launch_template.h"),
                os.path.join(cutlass_inc, "cute/tensor.hpp"),
            ):
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
            from torch.utils.cpp_extension import load

            arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            t0 = time.time()
            try:
                _ext = load(
                    name="tllm_vg_fa2_sm120",
                    sources=[
                        os.path.join(_HERE, "csrc", "fa2_sm120_bind.cpp"),
                        os.path.join(_HERE, "csrc", "fa2_sm120.cu"),
                    ],
                    extra_include_paths=[flash_src, cutlass_inc],
                    extra_cflags=["-O2"] + _DEFINES,
                    # No --use_fast_math: torch builds FA2 without it (exp2f, 1/sum).
                    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
                    + _DEFINES,
                    verbose=False,
                )
            finally:
                if arch_env is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_env
            _ext_state = True
            logger.info(f"[fa2_sm120] extension ready in {time.time() - t0:.1f} s")
        except Exception as e:  # noqa: BLE001 - any failure keeps SDPA
            _ext = None
            _ext_state = False
            logger.warning(f"[fa2_sm120] disabled, falling back to SDPA: {e!r}")
    return _ext


def is_available() -> bool:
    """True when the SM120 FA2 path can run in this process (builds the extension once)."""
    if os.environ.get("TLLM_FA2_SM120", "1") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    return _load_ext() is not None


_enabled: Optional[bool] = None
_num_sms: Optional[int] = None


def enabled() -> bool:
    """is_available(), memoized for hot paths (the extension builds on the first call)."""
    global _enabled
    if _enabled is None:
        _enabled = is_available()
    return _enabled


def _min_q_row_blocks() -> int:
    # torch's mha_fwd may split KV (flash_fwd_splitkv_kernel, different numerics) when
    # B*H*ceil(S_q/128) < 0.8 * 2 * num_SMs; at or above that it always runs one split,
    # the kernel this op reproduces bitwise. Smaller calls stay on SDPA.
    global _num_sms
    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return int(0.8 * 2 * _num_sms) + 1


def supports(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Shape/dtype/layout gate on SDPA-layout [B, H, S, D] inputs (metadata only).

    k and v share one shape; q has the same B, H and D and may have its own S_q (q-sliced
    calls). Non-causal only (the caller checks)."""
    return (
        q.dim() == 4
        and k.dim() == 4
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and q.is_cuda
        and k.shape == v.shape
        and q.shape[0] == k.shape[0]
        and q.shape[1] == k.shape[1]
        and q.shape[-1] == 128
        and k.shape[-1] == 128
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and q.shape[0] * q.shape[1] * ((q.shape[2] + 127) // 128) >= _min_q_row_blocks()
        and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    )


_TAGS = tuple(
    getattr(torch.Tag, t) for t in ("needs_exact_strides",) if hasattr(torch.Tag, t)
) or tuple(getattr(torch.Tag, t) for t in ("needs_fixed_stride_order",) if hasattr(torch.Tag, t))


@torch.library.custom_op("trtllm_vg::fa2_sm120_fwd", mutates_args=(), tags=_TAGS)
def fa2_sm120_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """q, k, v: [B, H, S, D] (SDPA layout). Returns the output as [B, S, H, D] contiguous."""
    global _calls
    ext = _load_ext()
    b, h, s, d = q.shape
    out = torch.empty((b, s, h, d), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    launches = ext.fwd(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), out, lse, scale, _MAX_REM_FRAC
    )
    _calls += 1
    if _calls == 1 or _calls % _LOG_EVERY == 0:
        logger.info(
            f"[fa2_sm120] calls={_calls} device={q.device} B={b} H={h} S={s} launches={launches}"
        )
    return out


@fa2_sm120_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    b, h, s, d = q.shape
    return q.new_empty((b, s, h, d))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """SDPA-layout in and out: returns [B, H, S, D] with SDPA FA2's strides (physical [B, S, H, D])."""
    return fa2_sm120_fwd(q, k, v, scale).transpose(1, 2)


def try_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> Optional[torch.Tensor]:
    """The one dispatch gate for every backend: the op's output (SDPA layout and strides) when
    it applies, else None and the caller runs torch SDPA. TLLM_FA2_SM120=0 disables it."""
    if is_causal or enable_gqa or not enabled():
        return None
    if not supports(q, k, v):
        return None
    return attention(q, k, v, scale)
