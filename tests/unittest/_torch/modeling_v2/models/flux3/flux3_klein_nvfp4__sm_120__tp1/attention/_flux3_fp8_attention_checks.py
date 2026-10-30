# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operands and NVFP4 checks shared by the FLUX.3 FP8 attention entry tests."""

import zlib

import numpy as np
import torch

DEV = "cuda"
HEAD_DIM = 128
E4M3_MAX = 448.0
MIN_SCALE = float(np.float32(1.0) / (np.float32(448.0) * np.float32(512.0)))
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def seed_of(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _per_tensor_e4m3(x: torch.Tensor):
    """qknorm_rope_fp8's per-tensor rule: scale = bf16(max(amax / 448, min)), x * (1 / scale)."""
    scale = torch.clamp_min(x.abs().amax().float() / E4M3_MAX, MIN_SCALE)
    scale = scale.to(torch.bfloat16).float()
    factor = 1.0 / scale
    return (x.float() * factor).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale, factor


def make_qkv8(batch: int, seq_len: int, heads: int, seed: int):
    """``(qkv8 [3, B*S, H*128] e4m3, scales [8] fp32)`` as the target's Q/K/V prologue makes them:
    Q/K RMS-normed rows (unit RMS per head, weights ~1) and a BF16 V, each quantized per
    tensor, ``scales[3] = q_scale * k_scale * 128^-0.5``."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, width = batch * seq_len, heads * HEAD_DIM
    planes, scales, factors = [], [], []
    for p in range(3):
        x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
        if p < 2:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            x = x * (1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g, device=DEV))
        x8, s, f = _per_tensor_e4m3(x.reshape(rows, width).to(torch.bfloat16))
        planes.append(x8)
        scales.append(s)
        factors.append(f)
    qkv8 = torch.stack(planes).contiguous()
    sm_scale = torch.tensor(HEAD_DIM**-0.5, dtype=torch.float32, device=DEV)
    out = torch.stack(
        [
            scales[0],
            scales[1],
            scales[2],
            (scales[0] * scales[1]) * sm_scale,
            *factors,
            torch.zeros((), device=DEV),
        ]
    ).float()
    return qkv8, out


def global_scale_for(o: torch.Tensor) -> torch.Tensor:
    """An activation global scale as the checkpoint's: 448 * 6 / amax."""
    return (2688.0 / o.float().abs().amax()).reshape(1).to(torch.float32)


def sf_index(rows: int, cols: int, device) -> torch.Tensor:
    """[rows, cols] -> flat 128x4-swizzled byte index."""
    padded_cols = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(cols, device=device).view(1, -1)
    return (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )


def dequantize_nvfp4(
    fp4: torch.Tensor, sf: torch.Tensor, rows: int, width: int, gs: torch.Tensor
) -> torch.Tensor:
    """FP32 ``[rows, width]`` of packed E2M1 ``fp4`` and swizzled E4M3 ``sf`` (global scale gs)."""
    packed = fp4.reshape(rows, width // 2).view(torch.uint8)
    codes = torch.stack((packed & 0xF, packed >> 4), -1).reshape(rows, width).long()
    table = torch.tensor(_E2M1, device=fp4.device)
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    sf_bytes = sf.view(torch.uint8).reshape(-1)[sf_index(rows, width // 16, fp4.device)]
    block = sf_bytes.view(torch.float8_e4m3fn).float() / gs.float().reshape(())
    return values * block.repeat_interleave(16, dim=1)


def assert_nvfp4_close(
    out, ref, rows: int, width: int, gs, max_sf_frac: float, max_rel_rms: float
) -> None:
    """Gate an NVFP4 result against a reference NVFP4 of a nearly equal BF16 tensor: at most
    max_sf_frac of the block scales differ, and the dequantized values' relative RMS difference
    is at most max_rel_rms."""
    idx = sf_index(rows, width // 16, out[1].device).reshape(-1)
    sf_out = out[1].view(torch.uint8).reshape(-1)[idx]
    sf_ref = ref[1].view(torch.uint8).reshape(-1)[idx]
    sf_frac = (sf_out != sf_ref).float().mean().item()
    assert sf_frac <= max_sf_frac, f"{sf_frac:.2e} of the block scales differ (> {max_sf_frac})"
    a = dequantize_nvfp4(out[0], out[1], rows, width, gs)
    b = dequantize_nvfp4(ref[0], ref[1], rows, width, gs)
    rel = ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()
    assert rel <= max_rel_rms, f"dequantized relative RMS {rel:.3e} (> {max_rel_rms})"
