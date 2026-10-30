# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qkv_epi_fp8 entry.

Cells against the reference (fp32 GEMM per plane, torch RMSNorm / RoPE, exact scales) in
a stated band. Exact on their own: scales[0..7] and V's E4M3 bytes are functions of the
q/k norm weights and the op's own BF16 v plane, recomputed here with the kernel's fp32 ops
in order; the compact RoPE table reads back every row's cos|sin bit for bit; and the QKV
preparation refuses a QKV activation scale that differs from gate_up's.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _prepare as prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.qkv_epi_fp8 import (  # noqa: E501
    _MIN_SCALE_BITS,
    _f32,
    _static_scale,
    build_rope_compact,
    rope_from_compact,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.gemm.qkv_epi_fp8 import (  # noqa: E501
    qkv_epi_fp8 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
from _flux3_operands import (  # noqa: E402
    DEV,
    HIDDEN,
    assert_bf16_band,
    assert_e4m3_band,
    nvfp4_act,
    nvfp4_weight,
    seed_of,
)
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the flux3 klein sm_120 entries are certified on sm_120"
)

W = HIDDEN
SM_SCALE = 1.0 / math.sqrt(128)

#: v plane: the kernel's BF16(alpha * acc) and the fp32 reference's differ where the fp32
#: summation order crosses a BF16 rounding midpoint, ~5e-4 of the elements, one ulp.
V_FRAC = 2e-3
#: q/k codes: the BF16 GEMM value (above), torch's rsqrt vs rsqrt.approx (2^-22 relative)
#: and the RoPE fma each move a pre-rounding value by far less than an E4M3 step; a code
#: moves only when that value sits at a rounding boundary: measured well under 1e-3.
QK_FRAC = 5e-3

_WEIGHTS: dict = {}


def _weights():
    """The block's prepared QKV (q, k, v planes of 3072) and its norm weights."""
    if not _WEIGHTS:
        planes = [nvfp4_weight(W, HIDDEN, seed=11 + p) for p in range(3)]
        linears = [prep.Nvfp4Linear(fp4, sf, ins, ws2) for fp4, sf, ws2, ins in planes]
        weight, scale = prep.prepare_qkv(
            tuple(lin.weight for lin in linears), tuple(lin.weight_scale for lin in linears)
        )
        gs = [prep.global_scales(lin) for lin in linears]
        alpha = torch.tensor([a * b for a, b in gs], device=DEV, dtype=torch.float32)
        g = torch.Generator(device=DEV).manual_seed(5)
        q_norm = (1.0 + 0.3 * torch.randn(128, generator=g, device=DEV)).to(torch.bfloat16)
        k_norm = (1.0 + 0.3 * torch.randn(128, generator=g, device=DEV)).to(torch.bfloat16)
        _WEIGHTS.update(
            weight=weight, scale=scale, alpha=alpha, q_norm=q_norm, k_norm=k_norm,
            input_scale=planes[0][3],
        )  # fmt: skip
    return _WEIGHTS


def _cos_sin(rows: int) -> torch.Tensor:
    """A FLUX-like [rows, 128] fp32 cos|sin table: 4 axes of 16 pairs, positions on a 2-D
    image grid (axes 1, 2) with a shared time / layer axis, so many slices repeat."""
    side = max(1, int(math.sqrt(rows)))
    r = torch.arange(rows, device=DEV)
    pos = torch.stack([r // (side * side), (r // side) % side, r % side, torch.zeros_like(r)], 1)
    freq = 10000.0 ** (-torch.arange(16, device=DEV, dtype=torch.float64) / 16)
    ang = pos.double().repeat_interleave(16, dim=1) * freq.repeat(4)  # [rows, 64]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1).float().contiguous()


def _operands(m: int, seed: int):
    wt = _weights()
    act, act_sf = nvfp4_act(m, HIDDEN, seed, wt["input_scale"])
    tab, idx = build_rope_compact(_cos_sin(m))
    return (
        act,
        wt["weight"],
        act_sf,
        wt["scale"],
        wt["alpha"],
        wt["q_norm"],
        wt["k_norm"],
        tab,
        idx,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    m = cell.spec["m"]
    args = _operands(m, seed_of(cell.why))
    qkv8, scales, gemm = op(*args, SM_SCALE, True)
    assert qkv8.shape == (3, m, W) and qkv8.dtype == torch.float8_e4m3fn
    ref8, ref_scales, ref_gemm = op.reference(*args, SM_SCALE, True)
    v, ref_v = gemm.view(3, m, W)[2], ref_gemm.view(3, m, W)[2]
    assert_bf16_band(v, ref_v, V_FRAC, max_ulps=1)
    for p in range(2):
        assert_e4m3_band(qkv8[p], ref8[p], QK_FRAC)
    # The static q/k scales are exact; the v scale follows max|v|, equal unless the max
    # element moved by an ulp.
    assert torch.equal(scales[[0, 1, 4, 5, 7]], ref_scales[[0, 1, 4, 5, 7]])


@pytest.mark.parametrize("m", [4176, 16464], ids=["m4176", "m16464"])
def test_scales_and_v_bytes_exact_from_the_v_plane(m) -> None:
    """scales[0..7] and V's E4M3 bytes recomputed from the op's own BF16 v plane and the
    norm weights with the kernel's fp32 ops, in order: bitwise."""
    args = _operands(m, seed=m)
    qkv8, scales, gemm = op(*args, SM_SCALE, True)
    v = gemm.view(3, m, W)[2].float()
    s_q, f_q = _static_scale(args[5])
    s_k, f_k = _static_scale(args[6])
    min_scale = torch.tensor(_f32(_MIN_SCALE_BITS), device=DEV)
    s_v = torch.maximum(v.abs().amax() / 448.0, min_scale).to(torch.bfloat16).float()
    f_v = 1.0 / s_v
    sm = torch.tensor(SM_SCALE, dtype=torch.float32, device=DEV)
    want = torch.stack([s_q, s_k, s_v, (s_q * s_k) * sm, f_q, f_k, f_v, torch.zeros_like(s_v)])
    assert torch.equal(scales, want)
    v8 = (v * f_v).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(qkv8[2].view(torch.uint8), v8.view(torch.uint8))


def test_keep_v_false_gives_the_same_e4m3_planes() -> None:
    """keep_v=False only drops the v plane's L2 lines after reading it."""
    args = _operands(4176, seed=7)
    keep = op(*args, SM_SCALE, True)
    drop = op(*args, SM_SCALE, False)
    assert torch.equal(keep[0].view(torch.uint8), drop[0].view(torch.uint8))
    assert torch.equal(keep[1], drop[1])


def test_compact_rope_reads_back_bit_for_bit() -> None:
    cos_sin = _cos_sin(16464)
    tab, idx = build_rope_compact(cos_sin)
    cos, sin = rope_from_compact(tab, idx, cos_sin.shape[0])
    assert torch.equal(cos.view(torch.int32), cos_sin[:, :64].view(torch.int32))
    assert torch.equal(sin.view(torch.int32), cos_sin[:, 64:].view(torch.int32))
    # The table dedups: far fewer distinct entries than rows x axes.
    assert torch.unique(idx).numel() < idx.numel() // 4


def test_prepare_refuses_a_qkv_scale_unlike_gate_up() -> None:
    fp4, sf, ws2, ins = nvfp4_weight(256, 256, seed=3)
    lin = prep.Nvfp4Linear(fp4, sf, ins, ws2)
    other = prep.Nvfp4Linear(fp4, sf, ins * 2, ws2)
    up_fp4, up_sf, up_ws2, _ = nvfp4_weight(512, 256, seed=4)
    gate_up = prep.Nvfp4Linear(up_fp4, up_sf, ins * 2, up_ws2)
    with pytest.raises(ValueError, match="share one NVFP4 activation global scale"):
        prep.prepare_nvfp4_block("single_blocks.0", lin, lin, lin, lin, gate_up, other)
