# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm of the Qwen3-VL text encoder (hidden and q / k norms), bitwise the eager body.

Two Triton kernels around one ATen mean (``_text_rms_norm``) instead of eager's eight
ATen kernels, registered as an opaque custom op so the compiled encoder cannot fuse and
reorder the reduction.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._text_rms_norm import DIMS, supported, text_rms_norm_op  # noqa: F401 — registers the op

#: Qwen3-VL-4B's `rms_norm_eps`.
_EPS = 1e-6


class TextRmsNorm(OpWrapper):
    """`weight * bf16(f32(x) * rsqrt(mean(f32(x)^2, -1) + eps))`, rounded to bf16.

    x: bf16 `[..., D]` contiguous, D the hidden size (2560) or the head dim (128);
    weight: bf16 `[D]`; eps: the checkpoint's `rms_norm_eps`. Returns bf16 of x's shape.

    Fusion boundary: the norm only.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="hidden norms at the captured 80-token bucket",
            spec=dict(shape=(1, 80, 2560), eps=_EPS),
        ),
        Cell(why="q_norm at T = 80: 32 heads of 128", spec=dict(shape=(1, 80, 32, 128), eps=_EPS)),
        Cell(why="k_norm at T = 80: 8 heads of 128", spec=dict(shape=(1, 80, 8, 128), eps=_EPS)),
        Cell(
            why="hidden norms at the uncaptured 160-token bucket (a longer prompt)",
            spec=dict(shape=(1, 160, 2560), eps=_EPS),
        ),
    )

    note = f"""
    Metadata consumed: none.

    Kernels: K1 squares x into an fp32 buffer of x's shape; ATen's mean reduces it (the
    same reduction eager runs on the same shape, strides and dtype); K2 applies
    `rsqrt(var + eps)`, the bf16 rounding, the weight and the second bf16 rounding.
    1024 elements per program, 4 warps, no FMA contraction.

    Numerics: bitwise the eager body (no mul + add pair to contract; the mean is ATen's).

    Raises (not is_valid): x not contiguous CUDA bf16 with last dim in {DIMS}, or a
    weight that does not match.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.ops.trtllm.flux3_klein_text_rms_norm(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """The eager Qwen3-VL RMSNorm body, which the op reproduces bit for bit."""
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        return weight * normalized.to(x.dtype)


text_rms_norm = TextRmsNorm()
