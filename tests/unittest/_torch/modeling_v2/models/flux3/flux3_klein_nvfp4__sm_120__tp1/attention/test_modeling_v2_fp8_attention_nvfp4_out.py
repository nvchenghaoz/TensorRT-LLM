# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 fp8_attention_nvfp4_out entry.

The exactness claim on its own: bitwise ``fp4_quantize(O, global_scale, 16, swizzled)`` of the
BF16-out op's O on the same operands, scale padding rows included.

Cells (every NVFP4 block's S at 2048^2 and 1024^2) against the reference (FP32 emulation of the
kernel's recurrence, native-torch NVFP4). Band: the BF16 O's differ as in the bf16_out test (a
few bf16 units of the row maximum in a small fraction of elements); an NVFP4 code or block
scale moves only where such a difference crosses its rounding midpoint, one E2M1 / E4M3 step.
Gate: at most 2% of the block scales differ, and the dequantized tensors' relative RMS
difference is at most 2^-3 (NVFP4's own quantization error against the BF16 O is ~2^-3.5; a
wrong row, head or block mapping is O(1)).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.fp8_attention_bf16_out import (  # noqa: E501
    fp8_attention_bf16_out,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.attention.fp8_attention_nvfp4_out import (  # noqa: E501
    fp8_attention_nvfp4_out as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_fp8_attention_checks import (  # noqa: E402
    assert_nvfp4_close,
    global_scale_for,
    make_qkv8,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 120,
    reason="the flux3 klein sm_120 entries are certified on sm_120",
)

MAX_SF_FRAC = 2e-2
MAX_REL_RMS = 2.0**-3


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    b, n, h = s["batch"], s["seq_len"], s["heads"]
    qkv8, scales = make_qkv8(b, n, h, seed_of(cell.why))
    gs = global_scale_for(fp8_attention_bf16_out(qkv8, scales, b, h))
    out = op(qkv8, scales, b, h, gs)
    assert out[0].shape == (b, n, h * 64) and out[0].dtype == torch.uint8
    assert out[1].numel() == (b * n + 127) // 128 * 128 * (h * 8)
    ref = op.reference(qkv8, scales, b, h, gs)
    assert_nvfp4_close(out, ref, b * n, h * 128, gs, MAX_SF_FRAC, MAX_REL_RMS)


@pytest.mark.parametrize("seq_len", [4176, 16464], ids=["s4176", "s16464"])
def test_bitwise_fp4_quantize_of_the_bf16_out_op(seq_len) -> None:
    """The epilogue's whole contract: the bytes `fp4_quantize` writes for the BF16-out op's O
    (FlashInfer's cute-dsl nvfp4_quantize arithmetic), including the zeroed scale rows past
    B * S (4176 -> 4224)."""
    qkv8, scales = make_qkv8(1, seq_len, 24, seed=seq_len)
    o = fp8_attention_bf16_out(qkv8, scales, 1, 24)
    gs = global_scale_for(o)
    fp4, sf = op(qkv8, scales, 1, 24, gs)
    want_fp4, want_sf = torch.ops.trtllm.fp4_quantize(
        o.reshape(seq_len, 24 * 128), gs, 16, False, True
    )
    assert torch.equal(fp4.reshape(seq_len, -1), want_fp4.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8), want_sf.view(torch.uint8).reshape(-1))
