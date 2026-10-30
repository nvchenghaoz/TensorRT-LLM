# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch readers and writers of the NVFP4 layouts this target's kernels take.

The entries' ``reference`` methods take the packed operands their kernels take, not the
values those were built from, so the reference covers the layout as well as the
arithmetic. Two layouts:

* packed E2M1 rows ``[R, K/2]`` uint8, element ``2i`` in the low nibble of byte ``i``;
* E4M3 block scales, one per 16 elements of K, linear ``[R, K/16]`` or in the
  128x4-swizzled order (rows padded to 128, scale columns to 4, padding zero).

Readers are exact (integer and power-of-two arithmetic). ``quantize`` is the
round-to-nearest-even NVFP4 recipe in fp32 with exact division, the reference side of the
kernels' fast-math quantizers. Private to the target.
"""

from __future__ import annotations

import torch

SF_VEC = 16

#: e2m1 code (bit 3 = sign) -> value.
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: Midpoints between consecutive e2m1 magnitudes; at a tie round-to-nearest-even goes up
#: where ``_TIE_UP`` is True.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)


def pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """``[R, K/2]`` packed bytes -> ``[R, K]`` 4-bit codes (element ``2i`` in the low nibble)."""
    packed = packed.view(torch.uint8)
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def e2m1_to_float(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.long()]


def e4m3_to_float(sf: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes (uint8 or float8_e4m3fn) -> fp32, exact."""
    return sf.view(torch.uint8).view(torch.float8_e4m3fn).float()


def swizzled_sf_index(rows: int, cols: int, device) -> torch.Tensor:
    """``[rows, cols]`` scale coordinates -> flat offsets in the 128x4-swizzled buffer."""
    padded_cols = pad_up(cols, 4)
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(cols, device=device).view(1, -1)
    return (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )


def swizzled_sf_numel(rows: int, cols: int) -> int:
    return pad_up(rows, 128) * pad_up(cols, 4)


def unswizzle_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Swizzled scale buffer -> linear ``[rows, cols]`` uint8."""
    index = swizzled_sf_index(rows, cols, sf.device).reshape(-1)
    return sf.reshape(-1).view(torch.uint8)[index].reshape(rows, cols)


def swizzle_sf(sf: torch.Tensor, padded_rows: int | None = None) -> torch.Tensor:
    """Linear ``[rows, cols]`` scale bytes -> the swizzled 1-D buffer, padding zero.
    ``padded_rows`` (a multiple of 128 >= rows) sizes the buffer when it is larger."""
    rows, cols = sf.shape
    numel = swizzled_sf_numel(max(rows, padded_rows or 0), cols)
    out = torch.zeros(numel, dtype=torch.uint8, device=sf.device)
    out[swizzled_sf_index(rows, cols, sf.device).reshape(-1)] = sf.reshape(-1).view(torch.uint8)
    return out


def dequantize(packed: torch.Tensor, sf_linear: torch.Tensor) -> torch.Tensor:
    """``[R, K/2]`` codes and linear ``[R, K/16]`` scales -> ``[R, K]`` fp32 values (exact)."""
    values = e2m1_to_float(unpack_codes(packed))
    rows, k = values.shape
    scale = e4m3_to_float(sf_linear.reshape(rows, k // SF_VEC))
    return (values.view(rows, k // SF_VEC, SF_VEC) * scale.unsqueeze(-1)).view(rows, k)


def dequantize_swizzled(packed: torch.Tensor, sf_swizzled: torch.Tensor) -> torch.Tensor:
    """As `dequantize`, the scales read from the swizzled buffer."""
    rows, half = packed.shape
    return dequantize(packed, unswizzle_sf(sf_swizzled, rows, half * 2 // SF_VEC))


def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> 4-bit e2m1 codes (bit 3 = sign), round-to-nearest-even, saturating."""
    mid = torch.tensor(_E2M1_MIDPOINTS, device=v.device, dtype=torch.float32)
    tie_up = torch.tensor(_TIE_UP, device=v.device, dtype=torch.bool)
    a = v.abs().unsqueeze(-1)
    code = (a > mid).sum(-1)
    tie = a == mid
    code = code + (tie & tie_up).sum(-1)
    sign = (v < 0) | ((v == 0) & torch.signbit(v))
    return (code | (sign.to(code.dtype) << 3)).to(torch.uint8)


def quantize(values: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[R, K]`` -> (packed codes ``[R, K/2]`` uint8, linear E4M3 scales ``[R, K/16]``
    uint8): per block ``scale = e4m3((amax * global_scale) / 6)``, data ``values *
    (global_scale / scale)`` rounded to e2m1 (zero for an all-zero block)."""
    rows, k = values.shape
    blocks = values.float().reshape(rows, k // SF_VEC, SF_VEC)
    amax = blocks.abs().amax(-1)
    sf = ((amax * global_scale) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    out_scale = torch.where(sf_f != 0, global_scale / sf_f, torch.zeros_like(sf_f))
    codes = _e2m1_codes((blocks * out_scale.unsqueeze(-1)).reshape(rows, k))
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, sf.view(torch.uint8)
