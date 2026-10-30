# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row softmax of the VAE encoder's mid-block attention scores: fp32 in, bf16 out.

One persistent Triton kernel (``_row_softmax_bf16``) that makes the max, sum and write
passes over each row while it is still in L2, so the T x T fp32 scores cross DRAM once
instead of once per pass.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._row_softmax_bf16 import BLOCK, row_softmax_bf16_op, supported  # noqa: F401 — registers

#: Rows per chunk of the reference (bounds its fp32 temporaries).
_REF_ROWS = 2048


class RowSoftmaxBf16(OpWrapper):
    """`softmax(s * scale, dim=-1).to(bf16)` for fp32 `s [..., N]`.

    s: fp32 contiguous, N a multiple of 8192 (the encoder passes `[1, T, T]`, the fp32
    output of the bf16 `q^T k` bmm); scale: the softmax scale (`C^-0.5`). Returns bf16
    of s's shape.

    Fusion boundary: the scale, the softmax and the bf16 cast. The two bmms around it
    are the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(why="2048^2 reference image: encoder mid block T = 65536", spec=dict(t=65536)),
        Cell(why="1024^2 reference image: encoder mid block T = 16384", spec=dict(t=16384)),
    )

    note = f"""
    Metadata consumed: none.

    Launch: one program per SM (capped at the row count), rows walked last to first,
    BLOCK {BLOCK} columns per step, 8 warps. Loads are unmasked: the row length must be a
    multiple of {BLOCK} (`supported(s)`); the encoder takes torch's softmax otherwise.

    Numerics: the expressions of Inductor's softmax decomposition element for element
    (fp32, libdevice exp, finite rows `(s - max s) * scale`, non-finite rows
    `s * scale - max(s * scale)`); only the order of the fp32 row sum differs. One bf16
    rounding of `e / sum`.

    Not checked by the op (is_valid covers it): contiguity and the row-length multiple.
    """

    def __call__(self, s: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.ops.trtllm.flux3_klein_row_softmax_bf16(s, scale)

    def reference(self, s: torch.Tensor, scale: float) -> torch.Tensor:
        """torch.softmax of the fp32 scaled scores, rounded once to bf16, in row chunks."""
        flat = s.reshape(-1, s.shape[-1])
        out = torch.empty(flat.shape, dtype=torch.bfloat16, device=s.device)
        for r0 in range(0, flat.shape[0], _REF_ROWS):
            chunk = flat[r0 : r0 + _REF_ROWS].float()
            out[r0 : r0 + _REF_ROWS] = torch.softmax(chunk * scale, dim=-1).to(torch.bfloat16)
        return out.view(s.shape)

    def is_valid(self, s: torch.Tensor, scale: float) -> None:
        assert supported(s), (
            f"s must be fp32, contiguous, with a row length that is a multiple of {BLOCK}: "
            "the kernel's loads are unmasked and it addresses rows as row * N"
        )


row_softmax_bf16 = RowSoftmaxBf16()
