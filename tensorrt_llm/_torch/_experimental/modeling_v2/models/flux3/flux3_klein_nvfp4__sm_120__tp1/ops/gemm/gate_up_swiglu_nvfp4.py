# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gate_up NVFP4 GEMM with SwiGLU + NVFP4 quantize in its epilogue
(``torch.ops.trtllm.flux3_gate_up_swiglu_nvfp4``).

An SM120 CUTLASS block-scaled GEMM (Pingpong 128x128x128, CLC persistent scheduler) whose
register epilogue writes down_proj's NVFP4 input (packed E2M1 and 128x4-swizzled E4M3
scales) instead of the ``[M, 2I]`` gate_up output. The weight is prepared at load
(`_prepare.prepare_gate_up`): rows paired as [value (up), gate] and permuted within every
64-row group so each lane of a quad owns 8 consecutive outputs of a scale block.

The block runs it on a side stream concurrently with the attention FMHA: allocate the
outputs on the current stream before the fork (`allocate_outputs`) and pass them in.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ..quantization._nvfp4_reference import (
    SF_VEC,
    dequantize_swizzled,
    quantize,
    swizzle_sf,
    swizzled_sf_numel,
)
from . import _cpp_fakes  # noqa: F401 — fake for torch.compile tracing
from ._prepare import v2w_row_permutation

#: -log2(e) as the epilogue carries it (fp32 0xBFB8AA3B).
_NEG_LOG2E = -1.4426950216293335


def allocate_outputs(
    rows: int, columns: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed FP4 [ceil128(rows), columns / 4] uint8, swizzled scales uint8) for a gate_up
    weight of ``columns`` rows (2 x the MLP width)."""
    padded = (rows + 127) // 128 * 128
    out = torch.empty((padded, columns // 4), device=device, dtype=torch.uint8)
    out_sf = torch.empty((padded * (columns // 32),), device=device, dtype=torch.uint8)
    return out, out_sf


class GateUpSwigluNvfp4(OpWrapper):
    """``nvfp4(up * silu(gate), out_scale)`` of ``[value | gate] = alpha * (act @ weight.T)``
    as ``(fp4 [M, N/4] uint8, swizzled sf uint8)``; the SF buffer covers ``ceil128(M)`` rows.

    ``act_fp4`` ``[M, K/2]`` / ``act_sf`` swizzled: the block's shared NVFP4 activation (or
    its first M rows); ``weight`` ``[N, K/2]`` / ``weight_scale`` swizzled: gate_up from
    `_prepare.prepare_gate_up`; ``alpha`` fp32 ``[1]``: ``input_gs * weight_gs``;
    ``out_scale`` fp32 ``[1]``: down_proj's activation global scale (its inverse input
    scale); ``out`` / ``out_sf``: optional buffers from `allocate_outputs` (written in place,
    padding rows included). Returns ``out[:M]`` and ``out_sf``.
    """

    ARCHS = frozenset({Arch.SM_120})

    # gate_up 18432 x 3072 at the served M (single block 27 runs it on the first 16384 rows).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="2048^2 mode blocks and the last single block's 16384 kept rows", spec=dict(m=16384)
        ),
        Cell(why="2048^2 t2i single blocks", spec=dict(m=16464)),
        Cell(why="2048^2 i2i single blocks", spec=dict(m=32848)),
        Cell(why="1024^2 mode blocks", spec=dict(m=4096)),
        Cell(why="1024^2 t2i single blocks, a partial last 128-row tile", spec=dict(m=4176)),
    )

    note = """
    Metadata consumed: none. Every NVFP4 block; on its own stream beside the FMHA.

    Preconditions: weight and scales from `_prepare.prepare_gate_up` (an unprepared weight
    is silently a different SwiGLU pairing); N a multiple of 128 with N / 32 a multiple of
    4, K a multiple of 128; the activation from the adaLN head with gate_up's global scale.

    Numerics: value = acc_even * alpha and gate = acc_odd * alpha (mul.rn),
    h = value * gate / (ex2.approx.ftz(-log2e * gate) + 1) (div.rn), the block's E4M3
    scale e4m3(amax * out_scale / 6) and codes e2m1(h * out_scale / scale) (div.rn).
    Against `reference` (fp32 GEMM, the same h formula with torch's exp2, exact-division
    quantize) a code or a block scale can move by one step where h sits within fp32
    rounding of an E2M1 / E4M3 boundary.

    Launch: plain (no PDL attribute), CLC persistent grid.
    """

    def __call__(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        out_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        out_sf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = act_fp4.shape[0]
        if out is None or out_sf is None:
            out, out_sf = allocate_outputs(rows, weight.shape[0], act_fp4.device)
        torch.ops.trtllm.flux3_gate_up_swiglu_nvfp4(
            act_fp4, weight, act_sf, weight_scale, alpha, out_scale, out, out_sf
        )
        return out[:rows], out_sf

    def reference(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        out_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        out_sf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 GEMM of the dequantized operands (TF32 off) times alpha, with the prepared
        row order undone; ``h = up * gate / (2^(-log2e * gate) + 1)`` in fp32; the exact-
        division NVFP4 recipe with ``out_scale``, scales swizzled over ``ceil128(M)`` rows
        (padding zero). ``out`` / ``out_sf`` are ignored."""
        rows, half = act_fp4.shape
        n = weight.shape[0]
        a = dequantize_swizzled(act_fp4, act_sf)
        w = dequantize_swizzled(weight, weight_scale)
        with true_fp32_matmul():
            acc = alpha.float().reshape(()) * (a @ w.t())
        del a, w
        # Prepared row r holds packed row v2w_row_permutation(n)[r]; packed row 2o is up_o
        # (value), 2o + 1 is gate_o.
        packed = torch.empty_like(acc)
        packed[:, v2w_row_permutation(n, acc.device)] = acc
        value, gate = packed[:, 0::2], packed[:, 1::2]
        h = (value * gate) / (torch.exp2(gate * _NEG_LOG2E) + 1.0)
        codes, sf = quantize(h, float(out_scale.float().reshape(()).item()))
        return codes, swizzle_sf(sf)

    def is_valid(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        out_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        out_sf: torch.Tensor | None = None,
    ) -> None:
        # The kernel reads alpha[0] / out_scale[0] and the scale buffers over their padded
        # rectangles without checking their lengths.
        assert alpha.numel() == 1 and out_scale.numel() == 1, (
            "alpha and out_scale must hold exactly one element"
        )
        k = act_fp4.shape[1] * 2
        assert act_sf.numel() >= swizzled_sf_numel(act_fp4.shape[0], k // SF_VEC), (
            "act_sf is shorter than the padded swizzled rectangle of the activation"
        )
        assert weight_scale.numel() >= swizzled_sf_numel(weight.shape[0], k // SF_VEC), (
            "weight_scale is shorter than the padded swizzled rectangle of the weight"
        )


gate_up_swiglu_nvfp4 = GateUpSwigluNvfp4()
