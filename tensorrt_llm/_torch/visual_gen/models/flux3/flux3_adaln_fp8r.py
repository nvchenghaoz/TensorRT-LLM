# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 FP8R block head: LayerNorm + adaLN-modulate + per-token E4M3 quantize in one kernel.

The SM120 FP8R blocks (BF16 checkpoint-excluded image/joint blocks running the fp8r recipe)
start with two kernels on the critical path: Inductor's LN+modulate (per-block compile with
emulate_precision_casts) and quantize_fp8r (warpPerTokenFp8QuantKernel) re-reading x_mod.
_adaln_fp8r writes x_mod and its E4M3 data + fp32 row scales from one program per row.

Bitwise contract:
- x_mod: the Inductor kernel's arithmetic, copied verbatim (looped Welford with the same
  [XBLOCK=1, R0_BLOCK] layout and num_warps, the bf16 round points of emulate_precision_casts,
  libdevice.rsqrt, the where-select of the [1, 2, D] table row, enable_fp_fusion=False).
- data/scales: warpPerTokenFp8QuantKernel's math on the bf16-rounded x_mod
  (scale = max(amax / 448, 1e-12) round-to-nearest; q = clamp(x / scale, +-448) RN satfinite).
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import (
    PreparedFP8Sm120,
    shared_nvfp4_input,
)

# Module-level so an in-process ON/OFF A/B can flip it (Dynamo guards on it).
FP8R_ADALN_QUANT = os.environ.get("TRTLLM_FLUX3_FP8R_ADALN_QUANT", "1") != "0"
# Force the bf16 x_mod store even where no consumer reads it (diagnostics / A/B).
FP8R_ADALN_STORE_X_MOD = os.environ.get("TRTLLM_FLUX3_FP8R_ADALN_STORE_X_MOD", "0") == "1"

# Inductor's reduction config for main's LN+modulate kernels at these shapes
# (ReductionHint.INNER -> one config: XBLOCK 1, R0_BLOCK, num_warps), keyed by select/rows.
_R0_BLOCK = {True: 1024, False: 1024}
_NUM_WARPS = {True: 8, False: 8}


@triton.jit
def _modulated(
    x_ptr,
    shift_ptr,
    scale_ptr,
    w_ptr,
    b_ptr,
    sel,
    x0,
    r0_index,
    r0_mask,
    mean,
    m2,
    mod_row_stride,
    r0_numel: tl.constexpr,
    EPS: tl.constexpr,
    HAS_SELECT: tl.constexpr,
):
    # Same op order and round points as triton_red_fused__to_copy_add_mul_native_layer_norm_*.
    if HAS_SELECT:
        sc1 = tl.load(scale_ptr + (mod_row_stride + r0_index), r0_mask, other=0.0).to(tl.float32)
        sc0 = tl.load(scale_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
        sh1 = tl.load(shift_ptr + (mod_row_stride + r0_index), r0_mask, other=0.0).to(tl.float32)
        sh0 = tl.load(shift_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
        sc = tl.where(sel, sc1.to(tl.bfloat16).to(tl.float32), sc0.to(tl.bfloat16).to(tl.float32))
        sh = tl.where(sel, sh1.to(tl.bfloat16).to(tl.float32), sh0.to(tl.bfloat16).to(tl.float32))
        sc = sc.to(tl.bfloat16).to(tl.float32)
        sh = sh.to(tl.bfloat16).to(tl.float32)
    else:
        sc = tl.load(scale_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
    one = tl.full([1, 1], 1.0, tl.float32)
    one_plus = (sc + one).to(tl.bfloat16).to(tl.float32)
    xv = tl.load(x_ptr + (r0_index + r0_numel * x0), r0_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + (r0_index), r0_mask, other=0.0).to(tl.float32)
    centered = xv - mean
    var = m2 / tl.full([1, 1], r0_numel, tl.float32)
    rstd = libdevice.rsqrt(var + tl.full([1, 1], EPS, tl.float32))
    normed = ((centered * rstd) * w + b).to(tl.bfloat16).to(tl.float32)
    prod = (one_plus * normed).to(tl.bfloat16).to(tl.float32)
    return (prod + sh).to(tl.bfloat16)


@triton.jit
def _adaln_fp8r(
    x_ptr,
    shift_ptr,
    scale_ptr,
    sel_ptr,
    w_ptr,
    b_ptr,
    xmod_ptr,
    q_ptr,
    s_ptr,
    mod_row_stride,
    r0_numel: tl.constexpr,
    EPS: tl.constexpr,
    HAS_SELECT: tl.constexpr,
    STORE_XMOD: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """One row per program: x_mod (bf16, stored when STORE_XMOD), its per-row E4M3 data and
    fp32 scale."""
    xoffset = tl.program_id(0) * XBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    # Pass 1: Welford statistics, verbatim from Inductor's looped reduction.
    t_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    t_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    t_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        v = tl.load(
            x_ptr + (r0_index + r0_numel * x0),
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        v = tl.broadcast_to(v.to(tl.bfloat16).to(tl.float32), [XBLOCK, R0_BLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            v, t_mean, t_m2, t_weight, r0_offset == 0
        )
        t_mean = tl.where(r0_mask, mean_next, t_mean)
        t_m2 = tl.where(r0_mask, m2_next, t_m2)
        t_weight = tl.where(r0_mask, weight_next, t_weight)
    mean_r, m2_r, _ = triton_helpers.welford(t_mean, t_m2, t_weight, 1)
    mean = mean_r[:, None]
    m2 = m2_r[:, None]
    sel = tl.full([XBLOCK, 1], 0, tl.int1)
    if HAS_SELECT:
        sel = tl.load(sel_ptr + x0).to(tl.int1)
    # Pass 2: x_mod (bf16) and the row's amax.
    amax = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        h = _modulated(
            x_ptr, shift_ptr, scale_ptr, w_ptr, b_ptr, sel, x0, r0_index, r0_mask, mean, m2,
            mod_row_stride, r0_numel, EPS, HAS_SELECT,
        )
        if STORE_XMOD:
            tl.store(xmod_ptr + (r0_index + r0_numel * x0), h, r0_mask)
        amax = tl.maximum(amax, tl.where(r0_mask, tl.abs(h.to(tl.float32)), 0.0))
    row_amax = tl.max(amax, 1)[:, None]
    scale = tl.maximum(tl.div_rn(row_amax, 448.0), 1.0e-12)
    # Pass 3: recompute the same bf16 x_mod (deterministic) and quantize it.
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        h = _modulated(
            x_ptr, shift_ptr, scale_ptr, w_ptr, b_ptr, sel, x0, r0_index, r0_mask, mean, m2,
            mod_row_stride, r0_numel, EPS, HAS_SELECT,
        ).to(tl.float32)
        q = tl.minimum(tl.maximum(tl.div_rn(h, scale), -448.0), 448.0)
        tl.store(q_ptr + (r0_index + r0_numel * x0), q.to(tl.float8e4nv), r0_mask)
    tl.store(s_ptr + x0, scale)


@torch.library.custom_op("trtllm::flux3_fp8r_adaln_quant", mutates_args=())
def fp8r_adaln_quant(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    select: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    store_x_mod: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(1 + scale) * LayerNorm(x) + shift (bf16) and its fp8r per-token E4M3 data + fp32 scales.

    x [1, S, D] contiguous bf16. Without select, shift/scale are [1, 1, D] rows (any storage
    offset, unit last stride). With select ([1, S, 1] bool, True picks row 1), shift/scale are
    [1, 2, D] tables with unit last stride and a shared row stride.
    With store_x_mod=False the returned x_mod is allocated but NOT written (uninitialized): only
    for callers whose consumers read the quantized data alone (x_mod as shape metadata).
    """
    width = x.shape[-1]
    rows = x.numel() // width
    x_mod = torch.empty_like(x, memory_format=torch.contiguous_format)
    data = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows,), device=x.device, dtype=torch.float32)
    if rows:
        has_select = select is not None
        row_stride = shift.stride(1) if has_select else 0
        _adaln_fp8r[(rows,)](
            x,
            shift,
            scale,
            select if has_select else x,
            weight,
            bias,
            x_mod,
            data,
            scales,
            row_stride,
            width,
            eps,
            has_select,
            store_x_mod,
            1,
            _R0_BLOCK[has_select],
            num_warps=_NUM_WARPS[has_select],
            enable_fp_fusion=False,
        )
    return x_mod, data, scales


@fp8r_adaln_quant.register_fake
def _fp8r_adaln_quant_fake(x, shift, scale, select, weight, bias, eps, store_x_mod=True):
    width = x.shape[-1]
    rows = x.numel() // width
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        x.new_empty((rows, width), dtype=torch.float8_e4m3fn),
        x.new_empty((rows,), dtype=torch.float32),
    )


def _row_ok(value: torch.Tensor, x: torch.Tensor, rows: int) -> bool:
    return (
        value.dtype == x.dtype
        and value.device == x.device
        and value.dim() == 3
        and tuple(value.shape) == (1, rows, x.shape[-1])
        and value.stride(-1) == 1
    )


def _x_mod_unread(block, x: torch.Tensor, freqs) -> bool:
    """True when the block's FP8R paths read only the quantized x_mod (bf16 x_mod = metadata):
    QKV through the fused qknorm_rope op (flux3_fp8r_qkv._sm120_fp8r_qkv_forward, which takes
    the shared data/scales and reads only the original's shape when its cos/sin checks pass),
    and the MLP through the two-stream tail pair (sm120_tail_pair_forward: up(mlp_input) reads
    .quantized/.scales; its fallbacks sm120_fp8_mlp_forward / PreparedFP8 do too)."""
    if not (getattr(block.attn, "_flux3_sm120_fp8r_qknorm_rope", False) and block._sm120_tail_pair):
        return False
    if len(freqs) < 2:
        return False
    cos, sin = freqs[0], freqs[1]
    return cos.numel() == x.numel() // x.shape[-1] * 128 and sin.shape == cos.shape


def fp8r_adaln_quant_head(block, x: torch.Tensor, mod: tuple, freqs=()):
    """(x_mod, Fp8RowwiseQuantizedTensor) for an FP8R block's head, or None (use the base path).

    Eligible only where main's head would run sm120_fp8_shared_input on the Inductor
    LN+modulate output: SM120 FP8R QKV + MLP-up prepared, broadcast [1, 1, D] rows (mode
    block) or [1, 2, D] tables + [1, S, 1] select (single block), contiguous [1, S, D] x.
    The bf16 x_mod is written only when some consumer may read its values (_x_mod_unread).
    """
    if not FP8R_ADALN_QUANT or block._fuse_proj_out:
        return None
    attn, up = block.attn, block.ff.gate_up_proj
    qkv = getattr(attn, "_flux3_sm120_fp8_qkv", None)
    prepared_up = getattr(up, "_flux3_primsts_fp8", None)
    if not (isinstance(qkv, PreparedFP8Sm120) and isinstance(prepared_up, PreparedFP8Sm120)):
        return None
    if not (qkv.can_run(x) and prepared_up.can_run(x)):
        return None
    if not (x.dim() == 3 and x.shape[0] == 1 and x.is_contiguous()):
        return None
    if shared_nvfp4_input(attn, up, x) is not None:
        return None
    pre_norm = block.pre_norm
    if len(mod) == 4:
        shift, scale, _, select = mod
        if not (
            _row_ok(shift, x, 2)
            and _row_ok(scale, x, 2)
            and shift.stride(1) == scale.stride(1)
            and select.dtype == torch.bool
            and tuple(select.shape) == (1, x.shape[1], 1)
            and select.is_contiguous()
        ):
            return None
    elif len(mod) == 3:
        shift, scale, _ = mod
        select = None
        if not (_row_ok(shift, x, 1) and _row_ok(scale, x, 1)):
            return None
    else:
        return None
    x_mod, data, scales = torch.ops.trtllm.flux3_fp8r_adaln_quant(
        x,
        shift,
        scale,
        select,
        pre_norm.weight,
        pre_norm.bias,
        pre_norm.variance_epsilon,
        FP8R_ADALN_STORE_X_MOD or not _x_mod_unread(block, x, freqs),
    )
    return x_mod, Fp8RowwiseQuantizedTensor(x_mod, data, scales)


def prepare_fp8r_adaln_quant(blocks) -> list:
    """Attach the fused head to the FP8R blocks (static, read in their forwards)."""
    names = []
    for name, block in blocks:
        target = getattr(block, "_orig_mod", block)
        if type(target).__name__ in ("_Flux3FP8RModeBlock", "_Flux3FP8RSingleStreamBlock"):
            target._flux3_fp8r_adaln_head = fp8r_adaln_quant_head
            names.append(name)
    return names
