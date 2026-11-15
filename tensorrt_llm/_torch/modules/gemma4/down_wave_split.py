# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave-aligned M split for Gemma4's down_proj NVFP4 GEMM on SM120.

At prefill M in [4096, 8192] the down_proj GEMM (N=5376, K=21504) is wave
quantized: a whole-M data-parallel kernel spends its last round on a few
tiles.  This path writes one bf16 output with up to two FlashInfer SM120
CUTLASS *data-parallel* launches (``fp4_gemm`` with an explicit tactic and a
caller-given ``out`` row slice):

* rows [0, M1) on the 256x128 DP tile (tactic 25, or its swap-AB twin 24),
  with M1 the largest 128-aligned row count that takes one round fewer;
* rows [M1, M) on a small DP tile when that tail is <= 512 rows;

otherwise one whole-M 256x128 DP launch (24 or 25, fewest rounds).

Exactness: each output row depends only on its A row and all of W, and every
DP tactic accumulates the full K in the same order in FP32 with one bf16
rounding, so outputs are bitwise equal to main's ``trtllm::nvfp4_gemm``
(checked int16-equal on synthetic and captured activations).  Stream-K
tactics (variant 2/3) are never used.

Kill switch: ``TRTLLM_GEMMA4_DOWN_WAVE_SPLIT=0`` gives main's single
``trtllm::nvfp4_gemm`` call.
"""
import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

ENABLED = os.environ.get("TRTLLM_GEMMA4_DOWN_WAVE_SPLIT", "1") != "0"
M_MIN = 4096
M_MAX = 8192
# Only the shape whose tactics were timed and checked bitwise.
_SHAPES = {(5376, 21504)}
_MAX_TAIL = 512

_state = {"mod": None, "failed": False, "ws": {}, "sms": {}}


def _load() -> Optional[object]:
    if _state["mod"] is not None or _state["failed"]:
        return _state["mod"]
    try:
        from flashinfer.jit.gemm.core import gen_gemm_sm120_module_cutlass_fp4
        mod = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
        # Needs the workspace-size export (experiments/ascent-flashinfer).
        mod.fp4_gemm_workspace_size(M_MIN, 5376, 21504)
        _state["mod"] = mod
        logger.info("Gemma4 down_proj NVFP4 wave split: FlashInfer SM120 "
                    "CUTLASS FP4 module loaded")
    except Exception as e:  # noqa: BLE001
        _state["failed"] = True
        logger.warning(f"Gemma4 down_proj wave split disabled: {e}")
    return _state["mod"]


def prepare() -> bool:
    """Load the FlashInfer module eagerly (first down_proj call, i.e. at
    warmup), never inside a CUDA graph capture."""
    if not ENABLED or _state["failed"]:
        return False
    if _state["mod"] is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        major, minor = torch.cuda.get_device_capability()
        if major != 12:
            _state["failed"] = True
            return False
        _load()
    return _state["mod"] is not None


def _num_sms(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _state["sms"].get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _state["sms"][idx] = n
    return n


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def plan(m: int, n: int, sms: int) -> List[Tuple[int, int, int]]:
    """Return [(row0, row1, fi_tactic), ...] covering rows [0, m)."""
    nt128 = _cdiv(n, 128)  # tactic 25: 256(M) x 128(N) tiles
    nt256 = _cdiv(n, 256)  # tactic 24: swap-AB, 128(M) x 256(N) tiles
    r25 = _cdiv(_cdiv(m, 256) * nt128, sms)
    r24 = _cdiv(_cdiv(m, 128) * nt256, sms)
    rounds = min(r25, r24)
    whole = 25 if r25 <= r24 else 24
    if rounds > 1:
        lim = (rounds - 1) * sms
        m1_24 = (lim // nt256) * 128
        m1_25 = (lim // nt128) * 256
        m1, t1 = (m1_24, 24) if m1_24 >= m1_25 else (m1_25, 25)
        tail = m - m1
        if 0 < tail <= _MAX_TAIL and m1 > 0:
            # 128x32x256 / 128x64x256 / 128x128x128 swap-AB DP tails.
            t2 = 4 if tail <= 64 else (12 if tail <= 256 else 16)
            return [(0, m1, t1), (m1, m, t2)]
    return [(0, m, whole)]


def _workspace(mod, pieces, n: int, k: int, device) -> torch.Tensor:
    sizes = _state.setdefault("ws_size", {})
    need = 0
    for r0, r1, _ in pieces:
        s = sizes.get((r1 - r0, n, k))
        if s is None:
            s = int(mod.fp4_gemm_workspace_size(r1 - r0, n, k))
            sizes[(r1 - r0, n, k)] = s
        need = max(need, s)
    key = device.index
    buf = _state["ws"].get(key)
    if buf is None or buf.numel() < need:
        buf = torch.empty(max(need, 1 << 20), dtype=torch.uint8, device=device)
        _state["ws"][key] = buf
    return buf


def applies(act_fp4: torch.Tensor, weight: torch.Tensor, act_sf: torch.Tensor,
            alpha: torch.Tensor, out_dtype: torch.dtype) -> bool:
    if not ENABLED or _state["mod"] is None:
        return False
    if act_fp4.dim() != 2 or out_dtype != torch.bfloat16:
        return False
    m = act_fp4.shape[0]
    if not (M_MIN <= m <= M_MAX):
        return False
    n = weight.shape[0]
    k = act_fp4.shape[1] * 2
    if (n, k) not in _SHAPES:
        return False
    if not (act_fp4.dtype == torch.uint8 and weight.dtype == torch.uint8
            and act_sf.dtype == torch.uint8 and act_fp4.is_contiguous()
            and act_sf.is_contiguous() and weight.is_contiguous()
            and alpha.dtype == torch.float32 and alpha.numel() == 1):
        return False
    sfk = _cdiv(k // 16, 4) * 4
    if act_sf.numel() < _cdiv(m, 128) * 128 * sfk:
        return False
    return True


def down_wave_split_gemm(act_fp4: torch.Tensor,
                         weight: torch.Tensor,
                         act_sf: torch.Tensor,
                         weight_scale: torch.Tensor,
                         alpha: torch.Tensor,
                         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    mod = _state["mod"]
    m = act_fp4.shape[0]
    n = weight.shape[0]
    k = act_fp4.shape[1] * 2
    sfk = _cdiv(k // 16, 4) * 4
    dev = act_fp4.device
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    sf = act_sf.view(-1)
    w_sf = weight_scale.view(-1)
    pieces = plan(m, n, _num_sms(dev))
    ws = _workspace(mod, pieces, n, k, dev)
    for r0, r1, t in pieces:
        # Swizzled SF is stored in contiguous 128-row blocks; r0 % 128 == 0.
        mod.fp4_gemm(act_fp4[r0:r1], weight, sf[r0 * sfk:], w_sf, alpha,
                     out[r0:r1], ws, t)
    return out
