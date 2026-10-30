# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 self-attention of the joint image stream with a BF16 output
(``torch.ops.trtllm.flux3_fp8_attention_bf16_out``), the attention of ``image.0``, the one
FP8-rowwise mode block.

The fmha_v2 tiled flash kernel of the e4m3 / FP32-accumulator 64x64 tile at head 128 with a
base-2 softmax, run as 3 groups of 64 Q rows per CTA over one TMA K/V ring
(``cpp/tensorrt_llm/kernels/flux3/fmha``).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8_attention import emulated_attention, planes


def _cell(why: str, s: int) -> Cell:
    return Cell(why=why, spec=dict(batch=1, seq_len=s, heads=24))


class Fp8AttentionBf16Out(OpWrapper):
    """``O`` ``[B, S, H, 128]`` BF16 ``= softmax(scales[3] * Q K^T) V * scales[2]``, non-causal,
    with P quantized to E4M3 as ``e4m3(256 exp(.))`` against each 64-key step's running max.

    ``qkv8``: E4M3 ``[3, B*S, H*128]``, the contiguous q, k, v planes (``qknorm_rope_fp8``'s
    output); ``scales``: FP32 ``[>= 4]`` in ``qknorm_rope_fp8``'s layout -- ``scales[2]`` the V
    descale, ``scales[3]`` ``q_descale * k_descale * sm_scale``. Both are read on device.

    Fusion boundary: QK^T, the softmax and P V. Q/K norm, RoPE and the E4M3 quantization are the
    caller's (``qknorm_rope_fp8``), the output projection too.
    """

    ARCHS = frozenset({Arch.SM_120})

    CELLS: tuple[Cell, ...] = (
        _cell("image.0 at 2048^2 (t2i and i2i): S = 16384, every KV step full", 16384),
        _cell("image.0 at 1024^2: S = 4096 (22 CTAs per head, the last one 2 groups)", 4096),
    )

    note = """
    Metadata consumed: none.

    Shapes: head_dim 128, S % 16 == 0 (S % 64 == 0 when B > 1), B * S >= 64; the op refuses
    others. A partial last KV step is masked; earlier steps are not.

    Numerics: base-2 softmax on the raw QK accumulators,
    `p = ex2.approx(fma(x, c, 8 - max * c))` with `c = scales[3] * log2(e)` and the O rescale
    `ex2.approx((prev_max - max) * c)` (skipped by a warp whose rows all keep their max), the
    row max as a balanced fmaxf tree, the row sum in the stock order, P to E4M3 unscaled
    (scale_softmax == 1), `O = bf16((acc * rcp(l)) * scales[2])` -- compiled with
    --use_fast_math. Reordered against the stock fmha_v2 kernel (expf of pre-scaled scores), not
    bitwise it. `reference` runs the same recurrence in FP32 torch.
    """

    def __call__(
        self, qkv8: torch.Tensor, scales: torch.Tensor, batch: int, heads: int
    ) -> torch.Tensor:
        q8, k8, v8 = planes(qkv8, batch, heads)
        return torch.ops.trtllm.flux3_fp8_attention_bf16_out(q8, k8, v8, scales[3:4], scales[2:3])

    def reference(
        self, qkv8: torch.Tensor, scales: torch.Tensor, batch: int, heads: int
    ) -> torch.Tensor:
        """`_fp8_attention.emulated_attention`: the kernel's per-64-key recurrence in FP32."""
        return emulated_attention(qkv8, scales, batch, heads)


fp8_attention_bf16_out = Fp8AttentionBf16Out()
