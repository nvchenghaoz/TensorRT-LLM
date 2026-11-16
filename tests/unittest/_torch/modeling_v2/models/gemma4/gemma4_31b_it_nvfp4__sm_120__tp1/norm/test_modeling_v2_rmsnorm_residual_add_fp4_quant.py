# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rmsnorm_residual_add_fp4_quant entry (the post-attention junction).

Per cell, against the entry's reference (FlashInfer rmsnorm, the bf16 add, FlashInfer's CuTe-DSL
`rmsnorm_fp4quant`):

* `h`: the first norm's sum of squares is a Triton `tl.sum` (not the CuTe order) and its mean
  Triton's fp32 divide, so where the fp32 result sits by a bf16 rounding midpoint `h` moves by
  one ulp (the kernel module measured ~5e-6 of the elements at Gemma4 shapes). A flip of the
  normed term is at most 2^-7 of it and the add rounds once more, so 4 units of the row
  maximum's 2^-8 element-wise; at most 1e-4 of the elements differ.
* `(fp4, sf)`: `_nvfp4_compare.assert_nvfp4_close`, at most 1e-3 of the scales and codes moved
  (only through a moved `h`).

The exactness claim of the second stage on its own: given the kernel's `h`, FlashInfer's
`rmsnorm_fp4quant` returns the same payload and the same whole scale buffer, padding rows
included (zero-filled).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.norm.rmsnorm_residual_add_fp4_quant import (  # noqa: E501
    rmsnorm_residual_add_fp4_quant as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import assert_nvfp4_close, assert_scale_padding_zero, seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

EPS = 1e-6
MAX_H_FLIP_FRAC = 1e-4
MAX_NVFP4_FRAC = 1e-3


def _build(m: int, h: int, seed: int):
    """`(x, residual, w1, eps, w2, eps, global_scale)`: an o_proj output, a residual stream
    four times larger, norm weights around 1, and the gate_up input scale 448 * 6 / amax of the
    normed rows (unit RMS, so amax ~ 5 x max |w2|)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, h, generator=g, device="cuda") * 2).to(torch.bfloat16)
    residual = (torch.randn(m, h, generator=g, device="cuda") * 8).to(torch.bfloat16)
    w1 = (1 + 0.3 * torch.randn(h, generator=g, device="cuda")).to(torch.bfloat16)
    w2 = (1 + 0.3 * torch.randn(h, generator=g, device="cuda")).to(torch.bfloat16)
    global_scale = (2688.0 / (5 * w2.float().abs().amax())).reshape(1)
    return x, residual, w1, EPS, w2, EPS, global_scale


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    m, h = cell.spec["m"], cell.spec["h"]
    args = _build(m, h, seed_of(cell.why))
    out_h, fp4, sf = op(*args)
    ref_h, ref_fp4, ref_sf = op.reference(*args)

    assert out_h.shape == (m, h) and out_h.dtype == torch.bfloat16
    flips = (out_h.view(torch.int16) != ref_h.view(torch.int16)).float().mean().item()
    assert flips <= MAX_H_FLIP_FRAC, f"{flips:.2e} of h differs"
    assert_within_ulp(out_h, ref_h, element_ulp=4.0, rms_ulp=0.25)
    assert sf.numel() == ref_sf.numel()
    assert_nvfp4_close(
        (fp4, sf), (ref_fp4, ref_sf), m, h, max_sf_frac=MAX_NVFP4_FRAC, max_code_frac=MAX_NVFP4_FRAC
    )
    assert_scale_padding_zero(sf, m, h)


@pytest.mark.parametrize("m", [1, 300, 1000, 8192])
def test_second_stage_is_flashinfers_rmsnorm_fp4quant(m) -> None:
    """Given the kernel's own `h`: the CuTe kernel's payload and whole scale buffer."""
    from flashinfer.norm import rmsnorm_fp4quant

    x, residual, w1, eps, w2, eps2, global_scale = _build(m, 5376, seed_of(f"second-{m}"))
    out_h, fp4, sf = op(x, residual, w1, eps, w2, eps2, global_scale)
    want_fp4, want_sf = rmsnorm_fp4quant(out_h, w2, global_scale, eps=eps2)
    assert torch.equal(fp4, want_fp4.view(torch.uint8))
    want_sf = want_sf.view(torch.uint8).reshape(-1)
    assert sf.numel() == want_sf.numel()
    assert torch.equal(sf, want_sf)


def test_guard_refuses_a_hidden_size_outside_the_replayed_layout() -> None:
    """H = 2048 runs 32 threads per row in the CuTe kernel, which the replay does not cover:
    the guard names the shape (the kernel's own assert carries no message)."""
    args = _build(4, 2048, seed=1)
    with validating(op), pytest.raises(AssertionError, match="2048"):
        op(*args)
