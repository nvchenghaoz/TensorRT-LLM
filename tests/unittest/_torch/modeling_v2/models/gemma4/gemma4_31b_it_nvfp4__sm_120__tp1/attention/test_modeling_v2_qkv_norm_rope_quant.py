# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qkv_norm_rope_quant entry, on its returned-K/V path (`md=None`).

Per cell (head 256 with q32 / kv16 and `[q | k | v]`; head 512 with q32 / kv4 and K = V,
`[q | k]` with V the raw K block), with the RoPE tables the target builds:

1. Bitwise the unfused chain the kernel replays, from in-tree ops: FlashInfer rmsnorm on q, k
   and (weightless) on the raw v, FlashInfer's RoPE (`flashinfer_apply_rope_with_cos_sin_cache
   _inplace`, the stock RotaryEmbedding's path) on q and k, FP8 casts of k and v.
2. Against the entry's `reference`: V bitwise (the same norm). Q and K within one rounding:
   the reference rotates in plain torch, `x1 * cos - x2 * sin` with both products rounded,
   where FlashInfer's RoPE (and the kernel) contracts one into an fma, so the fp32 result can
   differ in its last bit and, by a bf16 rounding midpoint, the output by one bf16 ulp
   (<= 2^-7 |v|: 2 units of the row maximum's 2^-8) -- for K, after the e4m3 cast, by one e4m3
   step (<= 2^-3 |v|: 1 unit of 2^-3). Such flips are rare; the relative RMS gate (0.25 units)
   is far below a wrong head, position or table.

The paged store (eager steps with metadata) is covered by the accuracy test.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.modeling import (  # noqa: E501
    _full_cos_sin,
    _sliding_cos_sin,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.attention.qkv_norm_rope_quant import (  # noqa: E501
    qkv_norm_rope_quant as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_compare import seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Gemma-4-31B sm_120 tp1 entries are certified on sm_120"
)

EPS = 1e-6
#: RoPE table length (the checkpoint's is 262144; positions below this exercise the same code).
MAX_POSITIONS = 32768
fp8 = torch.float8_e4m3fn
_TABLES: dict = {}


def _cos_sin(head_dim: int) -> torch.Tensor:
    """The target's tables: default RoPE theta 10k over the whole 256-wide sliding head;
    proportional RoPE theta 1M over a quarter of the 512-wide full-attention head."""
    if head_dim not in _TABLES:
        if head_dim == 256:
            table = _sliding_cos_sin(10_000.0, 256, MAX_POSITIONS)
        else:
            table = _full_cos_sin(1_000_000.0, 512, 0.25, MAX_POSITIONS)
        _TABLES[head_dim] = table.to("cuda").contiguous()
    return _TABLES[head_dim]


def _build(spec, seed: int):
    """The op's arguments (`md=None`, layer 0): a projection output at two magnitudes per
    row, norm weights around 1, positions over the table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, hd, hq, hkv, k_eq_v = (spec[k] for k in ("t", "hd", "hq", "hkv", "k_eq_v"))
    width = (hq + (1 if k_eq_v else 2) * hkv) * hd
    row_scale = torch.empty(t, 1, device="cuda").uniform_(-2.0, 3.0, generator=g).exp()
    qkv = (torch.randn(t, width, generator=g, device="cuda") * row_scale).to(torch.bfloat16)
    q_norm = (1 + 0.3 * torch.randn(hd, generator=g, device="cuda")).to(torch.bfloat16)
    k_norm = (1 + 0.3 * torch.randn(hd, generator=g, device="cuda")).to(torch.bfloat16)
    positions = torch.randint(
        0, MAX_POSITIONS, (t,), generator=g, device="cuda", dtype=torch.int32
    )
    return (qkv, positions, _cos_sin(hd), q_norm, k_norm, EPS, hq, hkv, hd, k_eq_v, None, 0)


def _in_tree_chain(qkv, positions, cos_sin, q_norm, k_norm, eps, hq, hkv, hd, k_eq_v):
    """split -> FlashInfer rmsnorm (v weightless, on the raw v; K = V reads raw K) ->
    FlashInfer RoPE on q and k in place -> FP8 k, v."""
    from flashinfer.norm import rmsnorm

    qw, kw = hq * hd, hkv * hd
    k_raw = qkv[:, qw : qw + kw]
    v_raw = k_raw if k_eq_v else qkv[:, qw + kw : qw + 2 * kw]
    ones = torch.ones(hd, dtype=qkv.dtype, device=qkv.device)
    q = rmsnorm(qkv[:, :qw].reshape(-1, hd).contiguous(), q_norm, eps).view(-1, qw)
    k = rmsnorm(k_raw.reshape(-1, hd).contiguous(), k_norm, eps).view(-1, kw)
    v = rmsnorm(v_raw.reshape(-1, hd).contiguous(), ones, eps).view(-1, kw)
    torch.ops.trtllm.flashinfer_apply_rope_with_cos_sin_cache_inplace(
        positions, q, k, hd, cos_sin.view(cos_sin.shape[0], -1), True
    )
    return q, k.to(fp8), v.to(fp8)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.uint8)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed_of(cell.why))
    t, hd, hq, hkv = (cell.spec[k] for k in ("t", "hd", "hq", "hkv"))
    q, k, v = op(*args)
    assert q.shape == (t, hq * hd) and q.dtype == torch.bfloat16
    assert k.shape == v.shape == (t, hkv * hd) and k.dtype == v.dtype == fp8

    want_q, want_k, want_v = _in_tree_chain(*args[:10])
    assert torch.equal(_bits(q), _bits(want_q)), "q differs from the in-tree chain"
    assert torch.equal(_bits(k), _bits(want_k)), "k differs from the in-tree chain"
    assert torch.equal(_bits(v), _bits(want_v)), "v differs from the in-tree chain"

    ref_q, ref_k, ref_v = op.reference(*args)
    assert torch.equal(_bits(v), _bits(ref_v))
    assert_within_ulp(q, ref_q, element_ulp=2.0, rms_ulp=0.25)
    assert_within_ulp(k, ref_k, element_ulp=1.0, rms_ulp=0.25, ulp=2.0**-3)


def test_k_eq_v_reads_v_from_the_raw_k_block() -> None:
    """K = V: V is the weightless norm of the raw K block, never of the normed, rotated K. A
    `[q | k | v]` call whose V block repeats K returns the same V."""
    spec = dict(t=64, hd=512, hq=32, hkv=4, k_eq_v=True)
    args = _build(spec, seed=7)
    qkv = args[0]
    qkv_dup = torch.cat([qkv, qkv[:, 32 * 512 :]], dim=1)
    _, _, v = op(*args)
    _, _, v_dup = op(qkv_dup, *args[1:9], False, None, 0)
    assert torch.equal(_bits(v), _bits(v_dup))
