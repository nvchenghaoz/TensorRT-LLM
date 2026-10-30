# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 row-wise block head: LayerNorm + adaLN modulate + per-token E4M3 quantize in one
Triton kernel (`_fp8r_adaln_triton`), for content_mode_blocks.image.0. The BF16 modulated
activation is never stored: its QKV and gate_up GEMMs read only the E4M3 data and scales."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8r_adaln_triton import D, fp8r_adaln_quant_op

#: FLUX.3 pre_norm's epsilon.
_EPS = 1e-6


class Fp8rAdalnQuant(OpWrapper):
    """``(data [M, D] float8_e4m3fn, scales [M] fp32)`` of ``x_mod = (1 + scale) *
    LayerNorm(x) + shift`` (bf16), ``scales = max(amax(|x_mod|) / 448, 1e-12)`` per row and
    ``data = e4m3(clamp(x_mod / scale, +-448))``.

    ``x`` ``[1, S, D]`` contiguous bf16, D = 3072; ``shift`` / ``scale``: the mode block's
    ``[1, 1, D]`` bf16 rows (unit last stride); ``eps``: pre_norm's epsilon. No LayerNorm
    affine.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 image.0: 16384 image rows", spec=dict(m=16384)),
        Cell(why="1024^2 image.0: 4096 image rows", spec=dict(m=4096)),
    )

    note = """
    Metadata consumed: none. content_mode_blocks.image.0 only.

    Numerics: bitwise Inductor's LN + modulate kernel for the block followed by the
    per-token E4M3 quantize (warp_fp8_per_token_quant's arithmetic). Against `reference`
    (torch layer_norm) the LayerNorm statistics are summed in another order, so a bf16
    value, a row scale and with them E4M3 codes can move by one step.
    """

    def __call__(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = _EPS
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fp8r_adaln_quant_op(x, shift, scale, eps)

    def reference(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = _EPS
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 LayerNorm (torch's, no affine) rounded to bf16, ``bf16(1 + scale)``,
        ``bf16(that * norm)``, ``bf16(that + shift)``; then the per-token E4M3 recipe."""
        rows = x.numel() // D
        norm = torch.nn.functional.layer_norm(x.reshape(rows, D).float(), (D,), eps=eps)
        norm = norm.to(torch.bfloat16).float()
        one_plus = (scale.reshape(1, D).float() + 1.0).to(torch.bfloat16).float()
        prod = (one_plus * norm).to(torch.bfloat16).float()
        x_mod = (prod + shift.reshape(1, D).float()).to(torch.bfloat16).float()
        amax = x_mod.abs().amax(dim=1)
        # A tensor divisor: torch turns `/ 448.0` (a CPU scalar) into a multiply by the
        # rounded reciprocal, which is not the round-to-nearest division of the recipe.
        scales = torch.clamp(amax / torch.full_like(amax, 448.0), min=1e-12)
        data = (x_mod / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return data, scales


fp8r_adaln_quant = Fp8rAdalnQuant()
