# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU + per-token E4M3 quantize of the FP8 row-wise block's gate_up output, one Triton
kernel (`_fp8r_triton.swiglu_fp8r_quant`): down_proj's input straight from gate_up's."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8r_triton import swiglu_fp8r_quant as _swiglu_fp8r_quant_op


class SwigluFp8rQuant(OpWrapper):
    """``(data [M, I] float8_e4m3fn, scales [M] fp32)`` of ``h = bf16(silu(gate) * up)`` with
    ``scales = max(amax(|h|) / 448, 1e-12)`` and ``data = e4m3(clamp(h / scale, +-448))``.

    ``x`` ``[M, 2I]`` bf16: the FP8 row-wise gate_up output in the checkpoint's
    ``[gate | up]`` column order.
    """

    ARCHS = frozenset({Arch.SM_120})

    # image.0's gate_up output, 2 x 9216 wide.
    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 image.0: 16384 rows", spec=dict(m=16384, width=9216)),
        Cell(why="1024^2 image.0: 4096 rows", spec=dict(m=4096, width=9216)),
    )

    note = """
    Metadata consumed: none. content_mode_blocks.image.0 only.

    Numerics: bitwise silu_and_mul (its Triton expression a * sigmoid(a) * b, BF16 out)
    followed by warp_fp8_per_token_quant. Against `reference` (torch's sigmoid) a BF16
    value can differ by one ulp and move a code.
    """

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _swiglu_fp8r_quant_op(x)

    def reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``bf16(gate * sigmoid(gate) * up)`` in fp32, then the per-token E4M3 recipe."""
        rows = x.reshape(-1, x.shape[-1]).float()
        width = rows.shape[1] // 2
        gate, up = rows[:, :width], rows[:, width:]
        h = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).float()
        amax = h.abs().amax(dim=1)
        # A tensor divisor: torch turns `/ 448.0` (a CPU scalar) into a multiply by the
        # rounded reciprocal, which is not the round-to-nearest division of the recipe.
        scales = torch.clamp(amax / torch.full_like(amax, 448.0), min=1e-12)
        data = (h / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return data, scales


swiglu_fp8r_quant = SwigluFp8rQuant()
