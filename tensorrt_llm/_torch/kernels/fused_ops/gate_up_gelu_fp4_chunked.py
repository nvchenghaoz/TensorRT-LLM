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
"""M-chunked NVFP4 gate_up GEMM -> gelu_tanh_mul_fp4_quant chain.

At large M the packed bf16 [M, 2I] gate_up output is far larger than L2, so
the activation+quant kernel re-reads it from DRAM. Running the GEMM and the
activation per block of C rows keeps each block's output (C x 2I x 2 bytes,
110 MB at C=1280, I=21504) mostly L2-resident when the activation reads it.

Exact by construction:
- every chunk runs the GEMM tactic main's autotuner picks for the FULL M
  (looked up in the tuner cache with main's runner and full-M shapes), and
  only CUTLASS tactics are chunked (their config is M-independent and each
  output row depends only on its own input row);
- the activation writes into one full-M FP4 payload + swizzled SF buffer;
  chunk boundaries are multiples of 128 rows, so a chunk's SF is the
  contiguous byte range of its 128-row tiles and the last chunk's padding
  lines up with the full-M padding.
Callers fall back to the unchunked path when this returns None.
"""

from typing import Callable, List, Optional, Tuple

import torch

from ...autotuner import AutoTuner
from ...custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
from .gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant

_GEMM_OP = "trtllm::nvfp4_gemm::gemm"
_BUFFER_KIND_DEFAULT = 0  # BufferKind.DEFAULT
# A last chunk shorter than this is merged into the previous one: each extra
# GEMM launch costs ~25-40 us of ramp/tail (SM120, N=43008, K=5376), more
# than a short chunk's L2 saving (measured: M=1536 as 1280+256 is 4% slower
# than unchunked, M=1792 as 1280+512 is 5% faster).
_MIN_LAST_CHUNK_ROWS = 384


def chunk_bounds(m: int, chunk_rows: int) -> List[Tuple[int, int]]:
    """Row ranges [r0, r1): r0 multiples of chunk_rows (so of 128); a short
    tail (< _MIN_LAST_CHUNK_ROWS) joins the previous chunk."""
    bounds = [(r0, min(m, r0 + chunk_rows)) for r0 in range(0, m, chunk_rows)]
    if len(bounds) >= 2 and bounds[-1][1] - bounds[-1][0] < _MIN_LAST_CHUNK_ROWS:
        bounds[-2:] = [(bounds[-2][0], m)]
    return bounds


def full_m_gemm_tactic(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    allowed_backends: List[str],
):
    """(runner, tactic) that trtllm::nvfp4_gemm uses for these exact inputs
    in inference mode, or None on a cache miss / tuning mode."""
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        return None
    runner = NVFP4GemmUnifiedRunner(_BUFFER_KIND_DEFAULT, output_dtype, list(allowed_backends))
    shapes = (act_fp4.size(), weight.size(), act_sf.size(), weight_scale.size(), alpha.size())
    hit, _, tactic, _ = tuner.profiling_cache.search_cache(
        _GEMM_OP, [runner], shapes, NVFP4GemmUnifiedRunner.tuning_config
    )
    if not hit:
        return None
    return runner, tactic


def gate_up_gelu_fp4_chunked(
    act_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    allowed_backends: List[str],
    down_input_scale: torch.Tensor,
    chunk_rows: int,
    act_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """gelu_tanh_mul_fp4_quant(nvfp4_gemm(act)) computed in row chunks.

    Args:
        act_fp4: [M, K/2] uint8 gate_up input payload (contiguous).
        act_sf: its 128x4-swizzled E4M3 scales (contiguous, pad128(M) rows).
        weight, weight_scale, alpha: gate_up NVFP4 weight, swizzled weight
            scales and alpha, as Linear passes them to nvfp4_gemm.
        output_dtype / allowed_backends: as Linear passes them.
        down_input_scale: [1] fp32 static input scale of down_proj.
        chunk_rows: rows per chunk, a positive multiple of 128.
        act_fn: the activation+quant kernel (default gelu_tanh_mul_fp4_quant);
            it must take out= / sf_out= like gelu_tanh_mul_fp4_quant. Permuted
            gate_up rows pass gelu_tanh_mul_fp4_quant_permuted.

    Returns:
        (fp4, sf) byte-identical to gelu_tanh_mul_fp4_quant(nvfp4_gemm(...)),
        or None when the chunked path does not apply (the caller then runs
        the unchunked chain).
    """
    if act_fn is None:
        act_fn = gelu_tanh_mul_fp4_quant
    if chunk_rows <= 0 or chunk_rows % 128 != 0:
        return None
    if act_fp4.dim() != 2 or not act_fp4.is_contiguous() or not act_sf.is_contiguous():
        return None
    m, k2 = act_fp4.shape
    bounds = chunk_bounds(m, chunk_rows)
    if len(bounds) < 2:
        return None
    nvec = (k2 * 2) // 16
    sf_rows_bytes = ((nvec + 3) // 4) * 4 * 128  # SF bytes per 128 rows
    m_pad = (m + 127) // 128 * 128
    if act_sf.numel() != m_pad // 128 * sf_rows_bytes:
        return None
    pick = full_m_gemm_tactic(
        act_fp4, weight, act_sf, weight_scale, alpha, output_dtype, allowed_backends
    )
    if pick is None:
        return None
    runner, tactic = pick
    # Only CUTLASS configs are M-independent (no heuristic re-query per shape).
    if not (isinstance(tactic, tuple) and len(tactic) == 2 and tactic[0] == "cutlass"):
        return None

    i2 = weight.shape[0]
    i = i2 // 2
    nkt_act = (i // 16 + 3) // 4
    act_sf_rows_bytes = nkt_act * 4 * 128
    out = torch.empty((m, i // 2), dtype=torch.uint8, device=act_fp4.device)
    sf_out = torch.empty((m_pad * nkt_act * 4,), dtype=torch.uint8, device=act_fp4.device)
    sf_flat = act_sf.view(-1)
    for r0, r1 in bounds:
        t0, t1 = r0 // 128, (r1 + 127) // 128
        h1 = runner.forward(
            [
                act_fp4[r0:r1],
                weight,
                sf_flat[t0 * sf_rows_bytes : t1 * sf_rows_bytes],
                weight_scale,
                alpha,
            ],
            tactic=tactic,
        )
        act_fn(
            h1,
            down_input_scale,
            out=out[r0:r1],
            sf_out=sf_out[t0 * act_sf_rows_bytes : t1 * act_sf_rows_bytes],
        )
        # Free before the next chunk's GEMM so the caching allocator hands the
        # same block back and its rewrite hits L2.
        del h1
    return out, sf_out
