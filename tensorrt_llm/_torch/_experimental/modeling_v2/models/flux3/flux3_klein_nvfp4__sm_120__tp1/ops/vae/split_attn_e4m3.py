# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-head attention of the VAE decoder's mid block with an e4m3 P.V.

Two Triton kernels (``_split_attn_e4m3``): S = Q^T K into fp32 plus per-tile row maxes,
then a softmax prologue that forms the unnormalized e4m3 P in registers and feeds it to
an e4m3 tensor-core P.V; the 1 / (448 l) row scale and the per-channel V scale are the
epilogue. The T x T scores are written once and read once.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ._split_attn_e4m3 import split_attn_e4m3_op, supported  # noqa: F401 — registers the op

#: Rows per chunk of the reference (bounds its [rows, T] fp32 temporaries).
_REF_ROWS = 2048


class SplitAttnE4m3(OpWrapper):
    """`O^T = (pq @ vq^T) * (1 / (448 l)) * sv`, `[T, C]` bf16, where `S = q^T k` (fp32),
    `e = exp((S - rowmax S) * scale)`, `l = rowsum e`, `pq = e4m3(e * 448)`.

    q, k: bf16 `[C, T]`, any strides (the decoder passes channels_last views, strides
    `(1, C)`); vq: e4m3 `[C, T]` row-major, V quantized per channel; sv: fp32 `[C]`, the
    per-channel V scale; scale: the softmax scale (`C^-0.5`).

    Fusion boundary: QK^T, the softmax and P.V. The q / k / v projections, quantizing V
    (`sv = amax_t |v| / 448`) and the output projection are the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="2048^2 output: mid block 256^2, T = 65536 (NT = 256 tile maxes)",
            spec=dict(c=512, t=65536),
        ),
        Cell(
            why="1024^2 output: mid block 128^2, T = 16384 (NT = 64)",
            spec=dict(c=512, t=16384),
        ),
    )

    note = """
    Metadata consumed: none.

    Shapes: C == 512 and T with NT = T / 256 a power of two <= 64 or a multiple of 64
    (`supported(c, t)`); every load and store is unmasked. The decoder checks this
    before its first call and refuses other output sizes.

    Memory: S is a [T, T] fp32 scratch, 17.2 GB at T = 65536.

    Numerics: S is bitwise the fp32-output bf16 GEMM; the row max is exact; e, l and the
    epilogue are fp32 without FMA contraction; P is rounded once to e4m3 (unnormalized,
    <= 448) and V once to e4m3 per channel; P.V accumulates in fp32; one bf16 rounding.
    Against the reference below (same steps, torch's accumulation orders) it differs by
    the order of S, l and P.V accumulation only.

    Not checked by the op (is_valid covers it): the shape constraints above, b == 1
    (single sample, single head).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        vq: torch.Tensor,
        sv: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return torch.ops.trtllm.flux3_klein_split_attn_e4m3(q, k, vq, sv, scale)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        vq: torch.Tensor,
        sv: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """The same steps in fp32 torch (true fp32 matmuls), in row chunks."""
        c, t = q.shape
        kf = k.float()
        vf = vq.float()
        out = torch.empty((t, c), dtype=torch.bfloat16, device=q.device)
        with true_fp32_matmul():
            for r0 in range(0, t, _REF_ROWS):
                s = q[:, r0 : r0 + _REF_ROWS].float().t() @ kf  # [rows, T]
                z = s - s.amax(dim=-1, keepdim=True)
                e = torch.exp(z * torch.tensor(scale, dtype=torch.float32))
                pq = (e * 448.0).to(torch.float8_e4m3fn).float()
                o = (pq @ vf.t()) / (448.0 * e.sum(dim=-1, keepdim=True))
                out[r0 : r0 + _REF_ROWS] = (o * sv.float().view(1, c)).to(torch.bfloat16)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        vq: torch.Tensor,
        sv: torch.Tensor,
        scale: float,
    ) -> None:
        assert q.dim() == 2 and k.shape == q.shape and vq.shape == q.shape, (
            "q, k, vq must all be [C, T] (one sample, one head)"
        )
        assert supported(*q.shape), (
            "C must be 512 and T / 256 a power of two <= 64 or a multiple of 64: the "
            "kernels' loads and stores are unmasked"
        )
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, "q, k must be bf16"
        assert vq.dtype == torch.float8_e4m3fn, "vq must be e4m3: kernel B reads it as e4m3"
        assert sv.dtype == torch.float32 and sv.numel() == q.shape[0], "sv must be fp32 [C]"


split_attn_e4m3 = SplitAttnE4m3()
