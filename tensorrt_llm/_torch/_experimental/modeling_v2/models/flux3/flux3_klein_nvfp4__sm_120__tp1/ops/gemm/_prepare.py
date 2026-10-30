# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time weight transforms the denoiser GEMM entries take (pure functions of the
checkpoint tensors; ``weights.py`` calls them once per block).

* NVFP4 blocks (image.1-4, image_cond.0-4, single.0-27): `prepare_nvfp4_block` -- QKV
  concatenated with per-plane alphas, gate_up as [value, gate] row pairs permuted for the
  lane-local SwiGLU epilogue, and every device scalar the kernels read (alphas, the
  activation global scales).
* single_blocks.24-27 (BF16 in the checkpoint): `quantize_bf16_to_nvfp4` first, with a unit
  activation global scale.
* content_mode_blocks.image.0 (BF16 in the checkpoint): `prepare_fp8r_block`, per-output-
  channel E4M3 weights with fp32 amax / 448 scales.

The scalars are formed exactly as the built-in ``Flux3Pipeline`` loader forms them: global scales as Python
floats (``1 / input_scale``, ``weight_scale_2``), products in double, one fp32 rounding at
the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4

from ..quantization._nvfp4_reference import swizzled_sf_index


@dataclass(frozen=True)
class Nvfp4Linear:
    """One NVFP4 projection as loaded: ``weight`` [N, K/2] uint8 (packed E2M1),
    ``weight_scale`` 128x4-swizzled E4M3 bytes, ``input_scale`` fp32 [1] (None: unit
    activation global scale), ``weight_scale_2`` fp32 [1]."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor


def global_scales(linear: Nvfp4Linear) -> tuple[float, float]:
    """(input global scale, weight global scale) as Python floats: ``1 / input_scale``
    (1.0 without one) and ``weight_scale_2``."""
    inverse = 1.0 if linear.input_scale is None else float(linear.input_scale.item())
    weight_gs = float(linear.weight_scale_2.item())
    if not all(math.isfinite(s) and s > 0 for s in (inverse, weight_gs)):
        raise ValueError("FLUX.3 NVFP4 needs finite positive checkpoint global scales")
    return 1.0 / inverse, weight_gs


def inverse_scale(input_global_scale: float, device) -> torch.Tensor:
    """fp32 [1] activation quantization scale (the NVFP4 quantizers' global scale)."""
    return torch.tensor([1.0 / input_global_scale], device=device, dtype=torch.float32)


def alpha(input_global_scale: float, weight_global_scale: float, device) -> torch.Tensor:
    """fp32 [1] GEMM dequantization scale ``input_gs * weight_gs``."""
    return torch.full(
        (1,), input_global_scale * weight_global_scale, device=device, dtype=torch.float32
    )


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def pack_swiglu_pairs(
    weight: torch.Tensor, linear_scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint ``[gate | up]`` rows -> adjacent ``[value (up), gate]`` pairs (row 2o =
    up_o, 2o + 1 = gate_o), the same permutation on the linear ``[N, K/16]`` scales."""
    n = weight.shape[0]
    if n % 128 or linear_scales.shape[0] != n:
        raise ValueError("NVFP4 SwiGLU requires aligned weight and block-scale rows")

    def pack(t: torch.Tensor) -> torch.Tensor:
        half = n // 2
        return torch.stack((t[half:], t[:half]), dim=1).reshape_as(t).contiguous()

    return pack(weight), pack(linear_scales)


def v2w_row_permutation(columns: int, device) -> torch.Tensor:
    """Source row of every prepared gate_up row: within each 64-row group, row 2s + e (pair
    slot s = q + 4r + 16nb of the epilogue, e = value / gate) takes the paired row
    2o + e of output o = 8q + 4nb + r."""
    if columns % 64:
        raise ValueError("gate_up rows must be a multiple of 64")
    p = torch.arange(64, device=device)
    e, s = p & 1, p >> 1
    q, r, nb = s & 3, (s >> 2) & 3, s >> 4
    local = 2 * (8 * q + 4 * nb + r) + e
    rows = torch.arange(columns, device=device)
    return (rows // 64) * 64 + local[rows % 64]


def permute_rows_v2w(
    weight: torch.Tensor, swizzled_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(weight, 128x4-swizzled scale) with rows reordered by `v2w_row_permutation`; a pure
    byte permutation."""
    w = _u8(weight)
    columns, half_k = w.shape
    kb = half_k * 2 // 16
    if columns % 128 or kb % 4:
        raise ValueError("gate_up must have N % 128 == 0 and K / 16 % 4 == 0")
    perm = v2w_row_permutation(columns, w.device)
    sf = _u8(swizzled_scale).reshape(-1)
    if sf.numel() != columns * kb:
        raise ValueError("gate_up scale is not the swizzled [N, K/16] buffer")
    idx = swizzled_sf_index(columns, kb, w.device)
    out_sf = torch.empty_like(sf)
    out_sf[idx] = sf[idx][perm]
    return w[perm].contiguous(), out_sf


def prepare_gate_up(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """gate_up as the SwiGLU-epilogue GEMM takes it: ``[value, gate]`` row pairs, scales
    re-swizzled, then the v2w row permutation. Returns (weight [N, K/2] uint8, swizzled
    scale 1-D uint8)."""
    w = _u8(weight.detach())
    n, half_k = w.shape
    linear = unswizzle_sf(_u8(weight_scale.detach()), n, half_k * 2)
    w, linear = pack_swiglu_pairs(w, linear)
    swizzled = torch.ops.trtllm.block_scale_interleave(_u8(linear).contiguous())
    return permute_rows_v2w(w, swizzled)


def prepare_qkv(
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k, v concatenated by rows with their scales unswizzled, concatenated and
    re-swizzled. Returns (weight [3W, K/2] uint8, swizzled scale 1-D uint8)."""
    ws = [_u8(w.detach()) for w in weights]
    linear = [
        unswizzle_sf(_u8(s.detach()), w.shape[0], w.shape[1] * 2) for w, s in zip(ws, weight_scales)
    ]
    weight = torch.cat(ws, dim=0).contiguous()
    scale = torch.ops.trtllm.block_scale_interleave(_u8(torch.cat(linear, dim=0)).contiguous())
    return weight, scale.reshape(-1)


@dataclass(frozen=True)
class PreparedNvfp4Block:
    """What the NVFP4 block's kernels read. Weights uint8 [N, K/2] with 1-D swizzled scales;
    device scalars fp32."""

    qkv_weight: torch.Tensor
    qkv_weight_scale: torch.Tensor
    qkv_alpha: torch.Tensor  # [3] per-plane input_gs * weight_gs
    act_global_scale: torch.Tensor  # [1] the adaLN head's NVFP4 scale (gate_up's)
    gate_up_weight: torch.Tensor
    gate_up_weight_scale: torch.Tensor
    gate_up_alpha: torch.Tensor  # [1]
    down_weight: torch.Tensor
    down_weight_scale: torch.Tensor
    down_alpha: torch.Tensor  # [1]
    down_act_global_scale: torch.Tensor  # [1] the SwiGLU epilogue's out_scale
    attn_out_weight: torch.Tensor
    attn_out_weight_scale: torch.Tensor
    attn_out_alpha: torch.Tensor  # [1]
    attn_out_act_global_scale: torch.Tensor  # [1] the FMHA's NVFP4-out scale


def prepare_nvfp4_block(
    name: str,
    to_q: Nvfp4Linear,
    to_k: Nvfp4Linear,
    to_v: Nvfp4Linear,
    attn_out: Nvfp4Linear,
    gate_up: Nvfp4Linear,
    down: Nvfp4Linear,
) -> PreparedNvfp4Block:
    """Every load-time transform of one NVFP4 block. Copies: the inputs stay untouched.

    The adaLN head quantizes the block input once, with gate_up's activation global
    scale, and QKV and gate_up both read that activation; so the three QKV input scales
    must equal each other and gate_up's. Raises otherwise (the built-in pipeline would re-quantize
    the BF16 input for QKV, a path this target does not carry).
    """
    device = gate_up.weight.device
    q_gs, k_gs, v_gs = (global_scales(p) for p in (to_q, to_k, to_v))
    up_in, up_w = global_scales(gate_up)
    if not (q_gs[0] == k_gs[0] == v_gs[0] == up_in):
        raise ValueError(
            f"{name}: QKV and gate_up must share one NVFP4 activation global scale (the adaLN "
            f"head quantizes the block input once); got q {q_gs[0]!r}, k {k_gs[0]!r}, "
            f"v {v_gs[0]!r}, gate_up {up_in!r}"
        )
    qkv_weight, qkv_scale = prepare_qkv(
        (to_q.weight, to_k.weight, to_v.weight),
        (to_q.weight_scale, to_k.weight_scale, to_v.weight_scale),
    )
    qkv_alpha = torch.tensor(
        [gs[0] * gs[1] for gs in (q_gs, k_gs, v_gs)], device=device, dtype=torch.float32
    )
    gate_up_weight, gate_up_scale = prepare_gate_up(gate_up.weight, gate_up.weight_scale)
    down_in, down_w = global_scales(down)
    out_in, out_w = global_scales(attn_out)
    return PreparedNvfp4Block(
        qkv_weight=qkv_weight,
        qkv_weight_scale=qkv_scale,
        qkv_alpha=qkv_alpha,
        act_global_scale=inverse_scale(up_in, device),
        gate_up_weight=gate_up_weight,
        gate_up_weight_scale=gate_up_scale,
        gate_up_alpha=alpha(up_in, up_w, device),
        down_weight=_u8(down.weight.detach()).contiguous(),
        down_weight_scale=_u8(down.weight_scale.detach()).reshape(-1).contiguous(),
        down_alpha=alpha(down_in, down_w, device),
        down_act_global_scale=inverse_scale(down_in, device),
        attn_out_weight=_u8(attn_out.weight.detach()).contiguous(),
        attn_out_weight_scale=_u8(attn_out.weight_scale.detach()).reshape(-1).contiguous(),
        attn_out_alpha=alpha(out_in, out_w, device),
        attn_out_act_global_scale=inverse_scale(out_in, device),
    )


def quantize_bf16_to_nvfp4(weight: torch.Tensor) -> Nvfp4Linear:
    """A BF16 linear weight [N, K] as an NVFP4 W4A4 projection (single_blocks.24-27): the
    loader's recipe (per-tensor ``weight_scale_2 = amax / (6 * 448)``, per-16 E4M3 scales,
    128x4-swizzled) and a unit activation global scale (``input_scale`` None)."""
    with torch.no_grad():
        qweight, scale, weight_scale_2 = quantize_nvfp4(weight.detach())
        swizzled = torch.ops.trtllm.block_scale_interleave(_u8(scale).contiguous())
    return Nvfp4Linear(
        weight=_u8(qweight).contiguous(),
        weight_scale=swizzled.reshape(-1).contiguous(),
        input_scale=None,
        weight_scale_2=weight_scale_2.reshape(1).to(device=weight.device, dtype=torch.float32),
    )


def quantize_fp8r_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel E4M3 weights [N, K] with fp32 ``amax / 448`` scales [N] (floored at
    1e-12), bitwise the fp8r checkpoint variant's recipe for these BF16 layers."""
    w = weight.detach().float()
    scale = torch.clamp(w.abs().amax(dim=1) / 448.0, min=1e-12)
    data = (w / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.contiguous(), scale.contiguous()


@dataclass(frozen=True)
class PreparedFp8rBlock:
    """content_mode_blocks.image.0 as FP8 row-wise GEMM operands: E4M3 [N, K] weights with
    fp32 [N] scales. QKV is q | k | v by rows; gate_up keeps the checkpoint's [gate | up]."""

    qkv_weight: torch.Tensor
    qkv_scale: torch.Tensor
    attn_out_weight: torch.Tensor
    attn_out_scale: torch.Tensor
    gate_up_weight: torch.Tensor
    gate_up_scale: torch.Tensor
    down_weight: torch.Tensor
    down_scale: torch.Tensor


def prepare_fp8r_block(
    to_q: torch.Tensor,
    to_k: torch.Tensor,
    to_v: torch.Tensor,
    attn_out: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> PreparedFp8rBlock:
    """The BF16 weights of the excluded mode block -> FP8 row-wise operands."""
    with torch.no_grad():
        qkv_w, qkv_s = quantize_fp8r_weight(torch.cat([to_q, to_k, to_v], dim=0))
        out_w, out_s = quantize_fp8r_weight(attn_out)
        up_w, up_s = quantize_fp8r_weight(gate_up)
        down_w, down_s = quantize_fp8r_weight(down)
    return PreparedFp8rBlock(qkv_w, qkv_s, out_w, out_s, up_w, up_s, down_w, down_s)
