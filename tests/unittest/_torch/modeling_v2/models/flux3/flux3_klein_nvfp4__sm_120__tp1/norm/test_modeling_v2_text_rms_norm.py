# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flux3 klein sm_120 tp1 text_rms_norm entry.

The op reproduces the eager Qwen3-VL RMSNorm body bit for bit (no FMA can form, the mean
is ATen's on the same tensor), so every cell -- the hidden and q / k norms at the
captured 80-token bucket, and the hidden norm at a longer prompt's 160-token bucket --
is gated bitwise against the reference. Then the op inside torch.compile (where the
encoder calls it) and the loud refusal of an unsupported input.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.norm.text_rms_norm import (  # noqa: E501
    text_rms_norm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """Hidden states with per-token magnitudes over e^[-3, 4] and a Qwen-like weight."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = spec["shape"]
    mag = torch.empty(*shape[:-1], 1, device="cuda").uniform_(-3.0, 4.0, generator=g).exp()
    x = (torch.randn(*shape, generator=g, device="cuda") * mag).to(torch.bfloat16)
    weight = (1.0 + 0.5 * torch.randn(shape[-1], generator=g, device="cuda")).to(torch.bfloat16)
    return x, weight, spec["eps"]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, weight, eps = _build(cell.spec, _seed(cell.why))
    out = op(x, weight, eps)
    ref = op.reference(x, weight, eps)
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    assert torch.equal(_bits(out), _bits(ref)), "not bitwise the eager RMSNorm body"


def test_under_torch_compile() -> None:
    """Compiled with the encoder's options, the op stays opaque and stays bitwise."""
    x, weight, eps = _build(op.CELLS[0].spec, _seed("compile"))

    def f(x, weight):
        return op(x * 1.0, weight, eps)

    compiled = torch.compile(f, dynamic=False, options={"emulate_precision_casts": True})
    assert torch.equal(_bits(compiled(x, weight)), _bits(op.reference(x * 1.0, weight, eps)))


def test_refuses_unsupported_input() -> None:
    """A non-contiguous view raises instead of computing a wrong layout."""
    x, weight, eps = _build(dict(shape=(1, 80, 2560), eps=1e-6), _seed("strided"))
    with pytest.raises(ValueError, match="contiguous"):
        op(torch.stack([x[0], x[0]]).transpose(0, 1), weight, eps)
