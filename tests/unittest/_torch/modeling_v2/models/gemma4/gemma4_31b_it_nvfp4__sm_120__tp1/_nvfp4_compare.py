# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 output checks shared by this target's norm and gemm entry tests, and their inputs.

An NVFP4 output is `(packed [rows, cols/2] uint8, swizzled 128x4 E4M3 scales, 1-D)`.
`assert_nvfp4_close` gates a fused kernel against a reference whose BF16 intermediate differs
in a few elements (a reduction order, an approximate tanh); `assert_nvfp4_equal` is the
bitwise claim on the real rows (the reference op may leave its scale padding unwritten).
"""

from __future__ import annotations

import zlib

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.activation._gelu_tanh_mul_fp4_quant import (  # noqa: E501
    sf_swizzled_offsets,
)

SF_VEC = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def seed_of(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def nvfp4_operands(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An activation as the target feeds an NVFP4 GEMM: `fp4_quantize` with swizzled scales
    at global scale 448 * 6 / amax. Returns `(fp4, sf, global_scale)`."""
    global_scale = (2688.0 / x.abs().amax().float()).reshape(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(x, global_scale, SF_VEC, False, True)
    return fp4, sf, global_scale


def nvfp4_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A weight as `weights.py` stores it: the packed payload, the linear `[rows, K/16]` E4M3
    scales 128x4-swizzled by `block_scale_interleave`, and the checkpoint's weight_scale_2
    (amax / (448 * 6))."""
    global_scale = (2688.0 / w.abs().amax().float()).reshape(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(w, global_scale, SF_VEC, False, False)
    sf = torch.ops.trtllm.block_scale_interleave(sf.view(torch.uint8).reshape(w.shape[0], -1))
    return fp4, sf, 1.0 / global_scale


def unpack_codes(fp4: torch.Tensor) -> torch.Tensor:
    """`[rows, cols/2]` packed E2M1 -> `[rows, cols]` 4-bit codes (element 2j in the low nibble)."""
    fp4 = fp4.view(torch.uint8)
    return torch.stack((fp4 & 0xF, fp4 >> 4), dim=-1).reshape(fp4.shape[0], -1)


def linear_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The `[rows, cols/16]` E4M3 scale bytes of the real rows of a swizzled buffer."""
    off = sf_swizzled_offsets(rows, cols // SF_VEC, sf.device)
    return sf.view(torch.uint8).reshape(-1)[off].reshape(rows, cols // SF_VEC)


def assert_nvfp4_equal(out, ref, rows: int, cols: int) -> None:
    """Same payload and the same scale bytes of every real row and block."""
    assert out[0].shape == (rows, cols // 2), out[0].shape
    assert torch.equal(out[0].view(torch.uint8), ref[0].view(torch.uint8)), "payload differs"
    assert torch.equal(linear_sf(out[1], rows, cols), linear_sf(ref[1], rows, cols)), "scales differ"


def assert_scale_padding_zero(sf: torch.Tensor, rows: int, cols: int) -> None:
    """Every byte of the swizzled buffer outside the real rectangle is zero."""
    flat = sf.view(torch.uint8).reshape(-1)
    padding = torch.ones(flat.numel(), dtype=torch.bool, device=flat.device)
    padding[sf_swizzled_offsets(rows, cols // SF_VEC, flat.device)] = False
    assert not bool(flat[padding].any()), "scale padding is not zero-filled"


def assert_nvfp4_close(out, ref, rows: int, cols: int, *, max_sf_frac: float, max_code_frac: float):
    """The fused output against a reference whose BF16 intermediate moved by an ulp in a few
    elements. A moved element changes its block's scale only if it is the block maximum and
    crosses an E4M3 rounding boundary (then to the neighbouring code, at most `max_sf_frac` of
    the blocks), and its own code only if it crosses an E2M1 boundary (then to the neighbouring
    magnitude, same sign, at most `max_code_frac` of the elements). Where the scales differ each
    side is within half an E2M1 step (<= one scale unit) of the same value at its own scale, so
    the dequantized values differ by at most the sum of the two scales. A wrong column, row
    order, scale or pairing moves most codes by more than a step and fails these."""
    fp4, sf = out
    ref_fp4, ref_sf = ref
    assert fp4.shape == ref_fp4.shape == (rows, cols // 2), (fp4.shape, ref_fp4.shape)
    lin, ref_lin = linear_sf(sf, rows, cols), linear_sf(ref_sf, rows, cols)
    sf_diff = lin != ref_lin
    sf_frac = sf_diff.float().mean().item()
    assert sf_frac <= max_sf_frac, f"{sf_frac:.2e} of the block scales differ (> {max_sf_frac})"
    gap = (lin[sf_diff].int() - ref_lin[sf_diff].int()).abs()
    assert bool((gap <= 1).all()), "a block scale differs by more than one E4M3 code"

    codes, ref_codes = unpack_codes(fp4), unpack_codes(ref_fp4)
    same_scale = ~sf_diff.repeat_interleave(SF_VEC, dim=1)
    diff = (codes != ref_codes) & same_scale
    code_frac = diff.float().mean().item()
    assert code_frac <= max_code_frac, f"{code_frac:.2e} of the codes differ (> {max_code_frac})"
    mag, ref_mag = (codes & 7).int(), (ref_codes & 7).int()
    same_sign = ((codes ^ ref_codes) & 8) == 0
    neighbour = ((mag - ref_mag).abs() <= 1) & (same_sign | ((mag == 0) & (ref_mag == 0)))
    assert bool(neighbour[diff].all()), "a code differs by more than one E2M1 step"

    if bool(sf_diff.any()):
        table = torch.tensor(_E2M1, device=fp4.device)

        def dequant(c, s):
            sign = 1.0 - 2.0 * ((c >> 3) & 1).float()
            scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(SF_VEC, dim=1)
            return sign * table[(c & 7).long()] * scale, scale

        value, scale = dequant(codes, lin)
        ref_value, ref_scale = dequant(ref_codes, ref_lin)
        block = sf_diff.repeat_interleave(SF_VEC, dim=1)
        err = (value - ref_value).abs()[block]
        bound = (scale + ref_scale)[block] * (1 + 2**-10)
        assert bool((err <= bound).all()), "a re-scaled block dequantizes away from the reference"
