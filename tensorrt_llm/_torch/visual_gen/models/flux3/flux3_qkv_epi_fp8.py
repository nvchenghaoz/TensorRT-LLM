# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 SM120 QKV GEMM with an E4M3 Q/K epilogue (idea 14).

FlashInfer overlay entry ``csrc/flux3_qkv_epi_fp8_sm120.cu`` (module ``flux3_qkv_epi_fp8_sm120``):
the batched (L=3) NVFP4 QKV GEMM of main's FP8-direct route whose epilogue does, for the q/k
planes, BF16(alpha*acc) -> per-(row, head) RMSNorm + RoPE -> x static factor -> E4M3 straight into
qkv8 planes 0/1 (the same arithmetic as flux3_sm120_qknorm_rope_fp8's VEC mode 2, replayed
operation for operation), and for the v plane stores BF16(alpha*acc) plus one max|v| partial per
tile. A second launch (``v_quant``) reduces the partials, writes scales[0..7] exactly as the VEC
mode 2 does and quantizes V. The BF16 q/k planes are never written, and mode 1 disappears.

Same outputs as main's nvfp4_qkv_rope_fp8_impl (qkv8 bytes, scales[8], the gemm buffer's v plane)
with the STATIC_QK rule; the gemm buffer's q/k planes are undefined (main drops them from L2 too).
TRTLLM_FLUX3_QKV_EPI_FP8=0 restores main's GEMM + two-pass path byte for byte.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

# One wave of 1 CTA/SM x 2 for v_quant (512 threads each).
_VQ_CTAS_PER_SM = 2
_SM_COUNT: dict[int, int] = {}
_PARTS: dict[tuple[int, int], torch.Tensor] = {}


# TRTLLM_FLUX3_QKV_EPI_FP8=0 restores main's route byte for byte (read at import; module-level so
# one process can A/B it: the route is decided per call inside the custom op, and CUDA graphs
# captured under the other setting must be cleared before replaying).
ENABLED = os.environ.get("TRTLLM_FLUX3_QKV_EPI_FP8", "1").strip() != "0"

# Idea 51: v_quant walks the BF16 v plane from its last 16-B chunk down. The GEMM's L=3 tile
# order writes the v plane last, its last rows last, so the lines still in L2 are read first
# (32848: 202 MB of v does not fit the 128 MB L2). Elementwise with the same scale: v8 bitwise.
# TRTLLM_FLUX3_QKV_EPI_VQ_REVERSE=0 keeps the forward walk (read at import; module-level so one
# process can A/B it; graphs captured under the other setting must be re-captured).
VQ_REVERSE = os.environ.get("TRTLLM_FLUX3_QKV_EPI_VQ_REVERSE", "1").strip() != "0"

# Idea 84: the Q/K epilogue reads RoPE through a compact per-axis index instead of each row's
# [128] FP32 cos_sin (re-read by all 48 Q/K tiles of every 128-row band). Axis a = pairs
# 16a..16a+15. rope_tab [4M, 32] is cos_sin permuted into the epilogue's lane order (entry
# 4r + a = row r's axis-a slice); rope_idx [M, 4] int32 points each row's axis at the entry of the
# FIRST row with the same slice bits, so a tile reads a few hundred hot entries (L1/L2) instead of
# 128 rows x 512 B. Fixed shapes, no host sync: every index is checked on device bit for bit
# (all 32 words) and a row whose slice differs points at its own entry. Same values, so outputs
# are bitwise. Built once per request with the cos_sin table (_packed_positions).
# TRTLLM_FLUX3_QKV_EPI_ROPE_COMPACT=0: no table, landing 22's kernel and inputs (read at import;
# module-level so one process can A/B it; graphs captured under the other setting must be
# re-captured).
ROPE_COMPACT = os.environ.get("TRTLLM_FLUX3_QKV_EPI_ROPE_COMPACT", "1").strip() != "0"
# Test hook: make the compact table's verification fail (the full table is used).
_ROPE_COMPACT_FORCE_MISMATCH_ENV = "TRTLLM_FLUX3_QKV_EPI_ROPE_COMPACT_FORCE_MISMATCH"
_ROPE_AXES = 4
_ROPE_SLOTS = 8  # lane slot s = q + 4 wn of the epilogue (its pair p0)


def enabled() -> bool:
    return ENABLED


@lru_cache(maxsize=None)
def _rope_columns(device: torch.device) -> torch.Tensor:
    """[4, 32] cos_sin columns of each axis entry: slot s, k -> cos(16a+s), cos(16a+s+8),
    sin(16a+s), sin(16a+s+8) (k = 0..3)."""
    cols = [
        [(k >> 1) * 64 + 16 * a + s + 8 * (k & 1) for s in range(_ROPE_SLOTS) for k in range(4)]
        for a in range(_ROPE_AXES)
    ]
    return torch.tensor(cols, dtype=torch.long, device=device)


def build_rope_compact(cos_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(rope_tab f32 [4M, 32], rope_idx i32 [M, 4]) for the FP32 [M, 128] cos_sin table, or None
    (route off, unsupported table, or the forced-mismatch test hook: the full table is used).

    Device ops of fixed shape only (no host sync). rope_tab[4r + a] is row r's axis-a slice in
    the epilogue's lane order; rope_idx[r][a] = 4 rep + a for the first row rep whose slice key
    (its first cos and sin words) matches, kept only where all 32 words of the two slices are
    bit-equal, else 4r + a (the row's own entry). So rope_tab[rope_idx[r][a]] == row r's slice
    bit for bit for every row and axis."""
    if not (ENABLED and ROPE_COMPACT):
        return None
    if (cos_sin.dtype != torch.float32 or cos_sin.dim() != 2 or cos_sin.shape[1] != 128
            or not cos_sin.is_contiguous() or not cos_sin.is_cuda or cos_sin.shape[0] == 0
            or 4 * cos_sin.shape[0] >= 2**31):
        return None
    if os.environ.get(_ROPE_COMPACT_FORCE_MISMATCH_ENV, "0").strip() == "1":
        return None
    rows = cos_sin.shape[0]
    dev = cos_sin.device
    perm = _rope_columns(dev).reshape(-1)
    tab = cos_sin.view(torch.int32)[:, perm].contiguous()  # [M, 4 * 32]
    slab = tab.view(rows, _ROPE_AXES, 32)
    key = (slab[:, :, 0].long() << 32) | (slab[:, :, 2].long() & 0xFFFFFFFF)  # [M, 4]
    sorted_key, order = torch.sort(key.t().contiguous(), dim=1, stable=True)  # [4, M]
    start = torch.ones_like(sorted_key, dtype=torch.bool)
    start[:, 1:] = sorted_key[:, 1:] != sorted_key[:, :-1]
    pos = torch.arange(rows, device=dev).expand(_ROPE_AXES, rows)
    seg = torch.where(start, pos, torch.zeros_like(pos)).cummax(dim=1).values
    rep = torch.empty_like(order).scatter_(1, order, order.gather(1, seg)).t()  # [M, 4]
    axis = torch.arange(_ROPE_AXES, device=dev)
    ent = rep * _ROPE_AXES + axis
    own = torch.arange(rows, device=dev).unsqueeze(1) * _ROPE_AXES + axis
    entries = tab.view(rows * _ROPE_AXES, 32)
    match = (entries[ent.reshape(-1)].view(rows, _ROPE_AXES, 32) == slab).all(dim=-1)
    idx = torch.where(match, ent, own).to(torch.int32).contiguous()
    return entries.view(torch.float32), idx


def _rope_compact_ok(rows: int, cos_sin, rope_tab, rope_idx) -> bool:
    if not ROPE_COMPACT or rope_tab is None or rope_idx is None:
        return False
    return (rope_tab.dtype == torch.float32 and rope_tab.dim() == 2 and rope_tab.shape[1] == 32
            and rope_idx.dtype == torch.int32 and rope_idx.dim() == 2 and rope_idx.shape[1] == 4
            and rope_idx.shape[0] == cos_sin.shape[0] and rope_idx.shape[0] >= rows
            and rope_tab.shape[0] == _ROPE_AXES * rope_idx.shape[0]
            and rope_tab.is_contiguous() and rope_idx.is_contiguous()
            and rope_tab.data_ptr() % 16 == 0 and rope_idx.data_ptr() % 16 == 0)


@lru_cache(maxsize=1)
def module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_qkv_epi_fp8_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_qkv_epi_fp8_sm120.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _sm_count(device: torch.device) -> int:
    index = _device_index(device)
    count = _SM_COUNT.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = count
    return count


def preload(device: torch.device) -> None:
    """Build the module at model load and check its workspace fits main's QKV slot buffer."""
    if not enabled():
        return
    from . import flux3_sm120_strided_gemm as strided

    mod = module()
    need = int(mod.qkv3_epi_workspace_size(1 << 16, 3072, 3072))
    if need > strided._WORKSPACE_BYTES:
        raise RuntimeError(f"QKV epilogue workspace {need} B exceeds {strided._WORKSPACE_BYTES} B")
    strided.workspace(device, 0)
    _sm_count(device)
    _parts(device, 16384)  # M <= 87k at W = 3072: no allocation on the served path


def covers(rows: int, columns: int, heads_dim: int) -> bool:
    """Shapes the epilogue handles: head_dim 128, plane width a multiple of 128 (8-head pairs)."""
    return heads_dim == 128 and columns % 3 == 0 and (columns // 3) % 128 == 0 and rows > 0


def _parts(device: torch.device, n_part: int) -> torch.Tensor:
    """Per-device V partial buffer, grown (outside capture) to the largest M seen; never per call
    once warm. Every slot the call reads is rewritten by its GEMM before v_quant runs."""
    key = (_device_index(device), 0)
    buf = _PARTS.get(key)
    if buf is None or buf.numel() < n_part:
        size = max(n_part, 8192)
        buf = torch.empty(size, dtype=torch.float32, device=device)
        _PARTS[key] = buf
    return buf


def qkv_rope_fp8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    qkv_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    sm_scale: float,
    keep_v: bool = True,
    rope_tab: torch.Tensor | None = None,
    rope_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(qkv8 [3, M, W] E4M3, scales [8] FP32, gemm [M, 3W] BF16 with only its v plane defined).

    cos_sin: FP32 [M, 128] RoPE table indexed by token row (identity positions).
    keep_v=False: the caller never reads gemm's v plane; v_quant drops its L2 lines.
    rope_tab / rope_idx: build_rope_compact(cos_sin) (idea 84), or None for the full table.
    """
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR

    from . import flux3_sm120_strided_gemm as strided

    mod = module()
    device = activation.device
    rows = activation.shape[0]
    width = weight.shape[0] // 3
    qkv8 = torch.empty((3, rows, width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(8, dtype=torch.float32, device=device)
    out = torch.empty((rows, 3 * width), dtype=torch.bfloat16, device=device)
    v = out.view(3, rows, width)[2]
    n_part = ((rows + 127) // 128) * (width // 128)
    parts = _parts(device, n_part)
    qkv8_u8 = qkv8.view(torch.uint8)
    gemm_args = (
        activation,
        weight,
        activation_scale,
        weight_scale,
        qkv_scale,
        q_norm,
        k_norm,
        cos_sin,
        qkv8_u8,
        v,
        parts,
        strided.workspace(device, 0),
        float(_MIN_SCALING_FACTOR),
    )
    if _rope_compact_ok(rows, cos_sin, rope_tab, rope_idx):
        mod.qkv3_epi_fp8_compact(*gemm_args, rope_tab, rope_idx)
    else:
        mod.qkv3_epi_fp8(*gemm_args)
    grid = max(1, min(_VQ_CTAS_PER_SM * _sm_count(device), (rows * width // 8 + 511) // 512))
    mod.v_quant(
        v,
        qkv8_u8[2],
        parts,
        int(n_part),
        q_norm,
        k_norm,
        scales,
        float(sm_scale),
        float(_MIN_SCALING_FACTOR),
        int(grid),
        0 if keep_v else 1,
        1 if VQ_REVERSE else 0,
    )
    return qkv8, scales, out
