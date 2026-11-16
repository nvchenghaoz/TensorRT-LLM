# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gate_up_gelu_nvfp4 entry (gate_up NVFP4 GEMM -> gelu_tanh(gate) * up ->
NVFP4 quantize, down_proj's input).

Every cell runs with the arguments the target passes (`chunk_rows = MLP_CHUNK_ROWS` from
`MLP_CHUNK_MIN_M` rows), on the gate_up weight as loaded (checkpoint order) or as
`prepare_gate_up_weights` left it (permuted).

* Cells against the entry's reference, `_nvfp4_compare.assert_nvfp4_close`: the reference
  takes torch's exact `gelu(approximate="tanh")`, the kernels FlashInfer's `tanh.approx.f32`
  (relative error ~2^-11), so a few percent of the bf16 activations sit one ulp apart and a
  fraction of those cross an E2M1 or E4M3 rounding boundary; at most 1e-2 of the codes and of
  the scales move, each to a neighbour.
* The exactness claim, checkpoint order: bitwise the unfused in-tree chain -- the shared
  `trtllm::nvfp4_gemm`, `trtllm::flashinfer_gelu_tanh_and_mul`, `trtllm::fp4_quantize` -- on
  the real rows (`fp4_quantize` leaves its scale padding unwritten; the kernels zero it).
* Permuted rows: every route (unchunked, M-chunked, and the fused epilogue forced on, as a
  tuned `('cutlass', 2)` pick selects it) bitwise the checkpoint-order route, scale padding
  included.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _nvfp4_gelu_gemm_sm120 as epi,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm.gate_up_gelu_nvfp4 import (  # noqa: E501
    MLP_CHUNK_MIN_M,
    MLP_CHUNK_ROWS,
    prepare_gate_up_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.gemm.gate_up_gelu_nvfp4 import (  # noqa: E501
    gate_up_gelu_nvfp4 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import (  # noqa: E402
    assert_nvfp4_close,
    assert_nvfp4_equal,
    assert_scale_padding_zero,
    nvfp4_operands,
    nvfp4_weight,
    seed_of,
)
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

HIDDEN, INTER = 5376, 21504
#: `ModelConfig.nvfp4_gemm_allowed_backends`' default, which the target passes.
BACKENDS = ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
MAX_NVFP4_FRAC = 1e-2
_WEIGHTS: dict = {}


def _weights(permuted: bool):
    """`(weight, weight_sf, weight_scale_2)` of gate_up as loaded, or a permuted copy (None
    where the fused GEMM is unavailable and `prepare_gate_up_weights` declines)."""
    if not _WEIGHTS:
        g = torch.Generator(device="cuda").manual_seed(11)
        w = (torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(
            torch.bfloat16
        )
        _WEIGHTS[False] = nvfp4_weight(w)
        weight, weight_sf, ws2 = _WEIGHTS[False]
        w_perm, sf_perm = weight.clone(), weight_sf.clone()
        prepared = prepare_gate_up_weights(w_perm, sf_perm, HIDDEN)
        _WEIGHTS[True] = (w_perm, sf_perm, ws2) if prepared else None
    return _WEIGHTS[permuted]


def _activation(m: int, seed: int):
    """`(fp4, sf, alpha, down_input_scale)`: gate_up's NVFP4 input as the junction kernel
    emits it, alpha = input_scale * weight_scale_2, and a down_proj input scale (1 / ckpt
    input_scale): 448 * 6 / 32, as gelu(gate) * up of unit-variance gate and up stays below 32
    here, so no block scale saturates E4M3."""
    _, _, ws2 = _weights(False)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    fp4, sf, act_global = nvfp4_operands(x)
    alpha = (ws2 / act_global).reshape(1)
    down_input_scale = torch.tensor([2688.0 / 32.0], dtype=torch.float32, device="cuda")
    return fp4, sf, alpha, down_input_scale


def _chunk_rows(m: int):
    return MLP_CHUNK_ROWS if m >= MLP_CHUNK_MIN_M else None


def _call(m: int, seed: int, permuted: bool):
    fp4, sf, alpha, down_is = _activation(m, seed)
    weight, weight_sf, _ = _weights(permuted)
    return op(fp4, sf, weight, weight_sf, alpha, down_is, BACKENDS, _chunk_rows(m), permuted)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    m, permuted = cell.spec["m"], cell.spec["permuted"]
    if _weights(permuted) is None:
        pytest.skip("the fused gate_up epilogue module is unavailable: no permuted weights")
    certifying(op)  # armed after the skip: an armed entry must run its reference
    seed = seed_of(cell.why)
    out = _call(m, seed, permuted)
    assert out[0].shape == (m, INTER // 2) and out[0].dtype == torch.uint8
    fp4, sf, alpha, down_is = _activation(m, seed)
    weight, weight_sf, _ = _weights(False)
    ref = op.reference(fp4, sf, weight, weight_sf, alpha, down_is, BACKENDS, _chunk_rows(m), False)
    assert out[1].numel() == ref[1].numel()
    assert_nvfp4_close(
        out, ref, m, INTER, max_sf_frac=MAX_NVFP4_FRAC, max_code_frac=MAX_NVFP4_FRAC
    )
    assert_scale_padding_zero(out[1], m, INTER)


@pytest.mark.parametrize("m", [1, 300, 1000, 2000])
def test_checkpoint_order_is_the_unfused_chain(m) -> None:
    """The unchunked (m < 1664) and the M-chunked (2000) routes against the in-tree chain."""
    fp4, sf, alpha, down_is = _activation(m, seed=m)
    weight, weight_sf, _ = _weights(False)
    out = op(fp4, sf, weight, weight_sf, alpha, down_is, BACKENDS, _chunk_rows(m), False)
    h = torch.ops.trtllm.nvfp4_gemm(
        fp4, weight, sf, weight_sf, alpha, torch.bfloat16, allowed_backends=",".join(BACKENDS)
    )
    act = torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(h)
    want = torch.ops.trtllm.fp4_quantize(act, down_is, 16, False, True)
    assert_nvfp4_equal(out, want, m, INTER)
    assert_scale_padding_zero(out[1], m, INTER)


@pytest.mark.parametrize("m", [1, 300, 2000, 4000])
def test_permuted_rows_are_bitwise_checkpoint_order(m) -> None:
    """Without a tuned `('cutlass', 2)` pick the permuted route runs the GEMM on the permuted
    rows and the permuted-gather activation (chunked from 1664 rows)."""
    if _weights(True) is None:
        pytest.skip("the fused gate_up epilogue module is unavailable: no permuted weights")
    perm = _call(m, seed=100 + m, permuted=True)
    ckpt = _call(m, seed=100 + m, permuted=False)
    assert torch.equal(perm[0], ckpt[0]) and torch.equal(perm[1], ckpt[1])


@pytest.mark.parametrize("m", [1, 300, 1000, 2000])
def test_fused_epilogue_is_bitwise_checkpoint_order(m, monkeypatch) -> None:
    """The fused GEMM + GELU-tanh * up + NVFP4 epilogue, as a tuned `('cutlass', 2)` pick
    selects it (forced here: the tuner cache is cold)."""
    if _weights(True) is None:
        pytest.skip("the fused gate_up epilogue module is unavailable: no permuted weights")
    calls = dict(epi.path_calls)
    monkeypatch.setattr(epi, "epi_pick_is_fused", lambda *args, **kwargs: True)
    fused = _call(m, seed=200 + m, permuted=True)
    assert epi.path_calls["fused"] == calls["fused"] + 1, "the fused epilogue did not run"
    ckpt = _call(m, seed=200 + m, permuted=False)
    assert torch.equal(fused[0], ckpt[0]) and torch.equal(fused[1], ckpt[1])
