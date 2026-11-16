# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLP's first half: gate_up NVFP4 GEMM -> gelu_tanh(gate) * up -> NVFP4 quantize.

`gate_up_gelu_nvfp4(fp4, sf, w, w_sf, alpha, down_input_scale, backends, chunk_rows,
permuted)` returns down_proj's NVFP4 input `(fp4, sf)` from gate_up's. Three routes, in this
order, every one byte-identical to the unfused chain (the tuned gate_up GEMM, the bf16 [M, 2I]
output, the fused activation + quantize kernel):

* **fused epilogue** (`permuted`, the tuned gate_up tactic is CUTLASS config 2): one GEMM whose
  epilogue applies gelu_tanh * up and quantizes, from FlashInfer's `nvfp4_gelu_gemm_sm120` JIT
  module (256x128x128 SM120 block-scaled mainloop). It needs gate_up's rows interleaved as the
  epilogue reads them (`prepare_gate_up_weights` at load);
* **M-chunked** (`chunk_rows`, 1664+ rows): the GEMM and the activation per block of 1280 rows,
  so each block's bf16 output is read back from L2;
* **unchunked**: the tuned GEMM, then the activation kernel.

With the FlashInfer module unavailable the rows stay in checkpoint order (`permuted=False`) and
the last two routes run with the plain activation kernel.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

from ..activation._gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
from ._gate_up_gelu_fp4_chunked import gate_up_gelu_fp4_chunked
from ._nvfp4_gelu_gemm_sm120 import (
    ensure_gelu_gemm_workspace,
    gate_up_gelu_fp4_permuted,
    gelu_gemm_epi_supported,
    get_gelu_gemm_module,
    permute_gate_up_rows,
)
from ._nvfp4_gemm_tuning import nvfp4_gemm

#: Rows per block of the M-chunked route, and the fewest rows it runs at.
MLP_CHUNK_ROWS = 1280
MLP_CHUNK_MIN_M = 1664


def prepare_gate_up_weights(weight: torch.Tensor, weight_sf: torch.Tensor, in_features: int) -> bool:
    """Put gate_up's FP4 rows and swizzled scale rows in the fused-epilogue order, in place, and
    allocate the fused GEMM's workspace (before any CUDA-graph capture). False, leaving the
    weights as loaded, where the fused GEMM is unavailable."""
    if not gelu_gemm_epi_supported() or get_gelu_gemm_module() is None:
        return False
    w, ws = permute_gate_up_rows(weight.data, weight_sf.data, in_features)
    weight.data.copy_(w)
    weight_sf.data.copy_(ws)
    ensure_gelu_gemm_workspace(weight.device)
    logger.info_once(
        "Gemma4 fused gate_up NVFP4 GEMM + gelu_tanh*up + NVFP4-quant epilogue: enabled",
        key="gemma4_mv2_gateup_gelu_epi",
    )
    return True


class GateUpGeluNvfp4(OpWrapper):
    """`(fp4 [M, I/2] uint8, sf 1-D uint8)`: down_proj's NVFP4 input."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = tuple(
        Cell(why=f"{m} rows, {'permuted' if p else 'checkpoint'} row order", spec=dict(m=m, permuted=p))
        for m in (1, 300, 1000, 2000, 4000)
        for p in (True, False)
    )
    note = """
    Shapes: gate_up [2 * 21504, 5376 / 2] FP4 with 128x4-swizzled scales; the fused epilogue
    needs N % 128 == 0 and K % 64 == 0. `permuted` must match how `prepare_gate_up_weights`
    left the weights.
    """

    def __call__(
        self,
        fp4: torch.Tensor,
        sf: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        alpha: torch.Tensor,
        down_input_scale: torch.Tensor,
        backends: List[str],
        chunk_rows: Optional[int],
        permuted: bool,
    ):
        def gemm() -> torch.Tensor:
            return nvfp4_gemm(fp4, weight, sf, weight_sf, alpha, torch.bfloat16, backends)

        if permuted:
            out_fp4, out_sf, _ = gate_up_gelu_fp4_permuted(
                fp4, sf, weight, weight_sf, alpha, torch.bfloat16, backends, down_input_scale,
                chunk_rows, gemm,
            )
            return out_fp4, out_sf
        if chunk_rows is not None:
            res = gate_up_gelu_fp4_chunked(
                fp4, sf, weight, weight_sf, alpha, torch.bfloat16, backends, down_input_scale,
                chunk_rows,
            )
            if res is not None:
                return res
        return gelu_tanh_mul_fp4_quant(gemm(), down_input_scale)

    def reference(self, fp4, sf, weight, weight_sf, alpha, down_input_scale, backends, chunk_rows,
                  permuted):
        """The unfused chain on checkpoint-order rows (`chunk_rows` is ignored): the shared
        NVFP4 GEMM, then exact gelu_tanh(gate) * up in fp32 rounded to bf16, then
        `trtllm::fp4_quantize`. The routes use FlashInfer's `tanh.approx` (bitwise the built-in
        fused activation kernel) and zero the scale padding, so they match this within an
        NVFP4 band, not bitwise."""
        assert not permuted, "the reference reads checkpoint-order gate_up rows"
        h = torch.ops.trtllm.nvfp4_gemm(
            fp4, weight, sf, weight_sf, alpha, torch.bfloat16, allowed_backends=",".join(backends)
        )
        gate, up = h.float().chunk(2, dim=-1)
        act = (torch.nn.functional.gelu(gate, approximate="tanh") * up).to(torch.bfloat16)
        return torch.ops.trtllm.fp4_quantize(act, down_input_scale, 16, False)


gate_up_gelu_nvfp4 = GateUpGeluNvfp4()
