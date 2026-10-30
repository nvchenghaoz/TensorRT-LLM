# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operands and comparison bands shared by the FLUX.3 klein sm_120 GEMM-family entry tests."""

from __future__ import annotations

import zlib

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.flux3.flux3_klein_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_reference as nv,
)

DEV = "cuda"
HIDDEN = 3072
MLP = 9216


def seed_of(text: str) -> int:
    return zlib.crc32(text.encode()) & 0x7FFFFFFF


def nvfp4(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed [R, K/2] uint8, swizzled scales uint8) of a bf16 matrix."""
    fp4, sf = torch.ops.trtllm.fp4_quantize(x.contiguous(), global_scale, 16, False, True)
    return fp4.view(torch.uint8), sf.view(torch.uint8).reshape(-1)


def nvfp4_weight(n: int, k: int, seed: int):
    """A random [n, k] weight as the checkpoint stores it: (fp4, swizzled sf, weight_scale_2
    fp32 [1], input_scale fp32 [1])."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=DEV) * k**-0.5).to(torch.bfloat16)
    w_global = (2688.0 / w.abs().amax().float()).reshape(1)
    fp4, sf = nvfp4(w, w_global)
    # Activation input scales of the checkpoint are O(1e-2); any positive value works here.
    input_scale = torch.tensor([0.0123], device=DEV)
    return fp4, sf, (1.0 / w_global).reshape(1), input_scale


def nvfp4_act(m: int, k: int, seed: int, input_scale: torch.Tensor, std: float = 1.0):
    """(fp4, swizzled sf) of a random [m, k] activation quantized with 1 / input_scale."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device=DEV) * std).to(torch.bfloat16)
    return nvfp4(x, (1.0 / input_scale).reshape(1))


def assert_bf16_band(
    out: torch.Tensor,
    ref: torch.Tensor,
    max_frac: float,
    max_ulps: int = 2,
    abs_tol: torch.Tensor | None = None,
) -> None:
    """bf16 tensors equal except a fraction <= max_frac of elements, each within max_ulps
    bf16 ulps of the reference (measured at the reference's magnitude) beyond ``abs_tol``
    (an absolute allowance per element, e.g. an fp32 summation error bound; default 0)."""
    assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16
    o, r = out.float(), ref.float()
    diff = o != r
    frac = diff.float().mean().item()
    assert frac <= max_frac, f"{frac:.3e} of the elements differ (band {max_frac:.1e})"
    if diff.any():
        ulp = torch.ldexp(torch.ones_like(r), (torch.frexp(r.abs().clamp_min(2.0**-120))[1] - 8))
        err = (o - r).abs()
        if abs_tol is not None:
            err = (err - abs_tol).clamp_min(0.0)
        worst = (err / ulp)[diff].max().item()
        assert worst <= max_ulps, f"an element is {worst:.1f} bf16 ulps off (band {max_ulps})"


def assert_e4m3_band(out: torch.Tensor, ref: torch.Tensor, max_frac: float) -> None:
    """E4M3 tensors equal except a fraction <= max_frac of codes, each one step away
    (same sign; E4M3 bytes order like their magnitudes)."""
    o = out.view(torch.uint8).to(torch.int16)
    r = ref.view(torch.uint8).to(torch.int16)
    diff = o != r
    frac = diff.float().mean().item()
    assert frac <= max_frac, f"{frac:.3e} of the E4M3 codes differ (band {max_frac:.1e})"
    if diff.any():
        same_sign = (o & 0x80) == (r & 0x80)
        step = ((o & 0x7F) - (r & 0x7F)).abs()
        zero_pair = ((o & 0x7F) == 0) & ((r & 0x7F) == 0)
        ok = (same_sign & (step <= 1)) | (zero_pair) | (((o & 0x7F) + (r & 0x7F)) <= 1)
        assert bool(ok[diff].all()), "an E4M3 code is more than one step from the reference"


def assert_nvfp4_band(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
    rows: int,
    k: int,
    max_sf_frac: float,
    max_code_frac: float,
) -> None:
    """(packed, swizzled sf) pairs: the whole SF buffer the same length, its padding rows
    zero, the real rows' scale bytes equal except a fraction <= max_sf_frac (one E4M3 step
    each); in the blocks whose scale matches, codes equal except a fraction <= max_code_frac
    (one E2M1 step each). A block whose scale byte is 0 on both sides decodes to zero
    whatever its codes (the kernels floor that divisor at 1e-12 and saturate, the reference
    writes 0), so its codes are not compared."""
    fp4, sf = out[0].view(torch.uint8), out[1].reshape(-1).view(torch.uint8)
    ref_fp4, ref_sf = ref[0].view(torch.uint8), ref[1].reshape(-1).view(torch.uint8)
    assert fp4.shape == ref_fp4.shape == (rows, k // 2)
    assert sf.numel() == ref_sf.numel() == nv.swizzled_sf_numel(rows, k // nv.SF_VEC)
    cols = k // nv.SF_VEC
    index = nv.swizzled_sf_index(rows, cols, sf.device).reshape(-1)
    pad = torch.ones(sf.numel(), dtype=torch.bool, device=sf.device)
    pad[index] = False
    assert not bool(sf[pad].any()), "SF padding rows / columns must be zero"
    lin, ref_lin = nv.unswizzle_sf(sf, rows, cols), nv.unswizzle_sf(ref_sf, rows, cols)
    sf_diff = lin != ref_lin
    assert sf_diff.float().mean().item() <= max_sf_frac, (
        f"{sf_diff.float().mean().item():.3e} of the block scales differ (band {max_sf_frac:.1e})"
    )
    if sf_diff.any():
        step = (lin.to(torch.int16) - ref_lin.to(torch.int16)).abs()
        assert bool((step[sf_diff] <= 1).all()), "a block scale is more than one E4M3 step off"
    codes, ref_codes = nv.unpack_codes(fp4), nv.unpack_codes(ref_fp4)
    same_block = (~sf_diff & (ref_lin != 0)).repeat_interleave(nv.SF_VEC, dim=1)
    code_diff = (codes != ref_codes) & same_block
    frac = code_diff.float().mean().item()
    assert frac <= max_code_frac, f"{frac:.3e} of the codes differ (band {max_code_frac:.1e})"
    if code_diff.any():
        mag = (codes & 7).to(torch.int16) - (ref_codes & 7).to(torch.int16)
        assert bool((mag.abs()[code_diff] <= 1).all()), "a code is more than one E2M1 step off"
