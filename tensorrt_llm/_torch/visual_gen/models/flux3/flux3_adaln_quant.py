# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
# rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""FLUX.3 fused LayerNorm + adaLN modulate + NVFP4 activation quantize (one Triton kernel).

Replaces, for the NVFP4 image/joint blocks, the Inductor LN+modulate reduction kernel followed
by trtllm::flux3_nvfp4_quantize_prepared (FlashInfer cute-dsl nvfp4_quantize, 128x4 scales,
fast-math path) on the modulated activation.

Exactness contract (bitwise to main):
- The LayerNorm + modulate math is Inductor's own kernel for this block (per-block compile with
  emulate_precision_casts=True): the same looped Welford reduction (triton_helpers.welford_reduce
  + welford), the same reduction block (XBLOCK=1, R0_BLOCK) and num_warps, the same bf16 rounding
  points and enable_fp_fusion=False.
- The quantize matches FlashInfer's default fast-math path on the bf16-rounded x_mod:
  scale = gs * (amax * rcp.approx.ftz(6)), cvt.rn.satfinite.e4m3, output scale
  rcp.approx.ftz(f32(e4m3) * rcp.approx.ftz(gs)), values * output scale with
  cvt.rn.satfinite.e2m1x2 (even element in the low nibble), 128x4 swizzled scale layout with
  zero-filled padding rows.
The global scale is read on device from PreparedNVFP4.inverse_scale.
"""

import os

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from .primsts import NVFP4Activation, PreparedNVFP4

_ENV = "TRTLLM_FLUX3_ADALN_QUANT"
# Runtime switch read in the block forwards (tests flip it to compare with main's path).
FLUX3_ADALN_NVFP4 = True

# Inductor's reduction config for these LN+modulate kernels at D=3072 (dumped from output_code:
# per-block compile, emulate_precision_casts=True, torch 2.14.0a0 nv26.09, triton 3.8).
# key: has_select (single-block [1,2,D] table + [1,S,1] select) -> (R0_BLOCK, num_warps)
_INDUCTOR_LN_CONFIG = {
    True: (1024, 8),
    False: (1024, 8),
}
# Width of the pointwise modulate+quantize pass. 1024 (= Inductor's) is bitwise to main; 2048/4096
# were ~1 us/call faster but left ~1 bf16 ulp on ~1e-6 of x_mod elements (a different Welford
# combine layout; fp4/SF stayed bitwise), so they are not used.
_Q_BLOCK = 1024


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _cvt_e4m3(x):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b16 fp8_pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
            cvt.u32.u16 $0, fp8_pair;
        }
        """,
        "=r,f",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _output_scale(sf_bits, gs):
    # FlashInfer nvfp4_compute_output_scale: rcp(f32(e4m3) * rcp(gs)), 0 when the scale is 0.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred p_zero;
            .reg .b16 fp8_pair;
            .reg .b32 h2_32;
            .reg .b16 h_lo, h_hi;
            .reg .f32 scale_f32, rcp_gs, product, result;
            cvt.u16.u32 fp8_pair, $1;
            cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
            mov.b32 {h_lo, h_hi}, h2_32;
            cvt.f32.f16 scale_f32, h_lo;
            rcp.approx.ftz.f32 rcp_gs, $2;
            mul.f32 product, scale_f32, rcp_gs;
            rcp.approx.ftz.f32 result, product;
            setp.eq.f32 p_zero, scale_f32, 0f00000000;
            selp.f32 $0, 0f00000000, result, p_zero;
        }
        """,
        "=f,r,f",
        [sf_bits, gs],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e2m1x2(lo, hi, out_scale):
    # One packed byte (lo nibble = lo) replicated in all four bytes of the u32; caller masks 0xFF.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 a, b;
            .reg .b8 byte0;
            mul.f32 a, $1, $3;
            mul.f32 b, $2, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, b, a;
            mov.b32 $0, {byte0, byte0, byte0, byte0};
        }
        """,
        "=r,f,f,f",
        [lo, hi, out_scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _adaln_nvfp4_kernel(
    in_ptr0,  # x [M, D] bf16
    in_ptr1,  # select [M] bool (HAS_SELECT)
    in_ptr2,  # scale: table row 0 (HAS_SELECT) or the [D] row
    in_ptr3,  # LayerNorm weight [D] bf16
    in_ptr4,  # LayerNorm bias [D] bf16
    in_ptr5,  # shift: table row 0 (HAS_SELECT) or the [D] row
    gs_ptr,  # [1] f32 NVFP4 global scale (PreparedNVFP4.inverse_scale)
    out_ptr0,  # x_mod [M, D] bf16
    fp4_ptr,  # [M, D // 2] uint8
    sf_ptr,  # [padded_M * SF_COLS] uint8, 128x4 swizzled
    xnumel,
    padded_xnumel,
    scale_row_stride,
    shift_row_stride,
    r0_numel: tl.constexpr,
    EPS: tl.constexpr,
    HAS_SELECT: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
    Q_BLOCK: tl.constexpr,
):
    tl.static_assert(XBLOCK == 1)
    SF_COLS: tl.constexpr = ((r0_numel // 16 + 3) // 4) * 4
    GROUPS: tl.constexpr = Q_BLOCK // 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex.to(tl.int64)
    gs = tl.load(gs_ptr)
    fp4_max_rcp = _rcp_approx(tl.full([1, 1], 6.0, tl.float32))
    if xoffset < xnumel:
        # ---- Inductor's LN reduction (verbatim) ----
        tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            r0_1 = r0_index
            tmp0 = tl.load(
                in_ptr0 + (r0_1 + r0_numel * x0),
                r0_mask & xmask,
                eviction_policy="evict_last",
                other=0.0,
            ).to(tl.float32)
            tmp1 = tmp0.to(tl.bfloat16)
            tmp2 = tmp1.to(tl.float32)
            tmp3 = tmp2.to(tl.float32)
            tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
            tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
                tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
            )
            tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
            tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
            tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
        tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
        # XBLOCK == 1: take the row's mean/M2 as scalars (a max over one element is an exact
        # copy). This decouples the reduction's register layout (and so its combine tree) from
        # the wider pointwise pass below, keeping Inductor's 1024-wide Welford order.
        tmp5 = tl.max(tmp6, axis=0)
        tmp9 = tl.max(tmp7, axis=0)
        if HAS_SELECT:
            tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy="evict_last").to(tl.int1)
        # Pointwise pass (no reduction order): Q_BLOCK-wide chunks so each thread owns 8
        # contiguous elements and the NVFP4 lane splits stay in registers.
        q_base = tl.arange(0, Q_BLOCK)[None, :]
        for r0_offset in tl.range(0, r0_numel, Q_BLOCK):
            r0_index = r0_offset + q_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            if HAS_SELECT:
                tmp12 = tl.load(
                    in_ptr2 + (scale_row_stride + r0_1),
                    r0_mask,
                    eviction_policy="evict_last",
                    other=0.0,
                ).to(tl.float32)
                tmp15 = tl.load(
                    in_ptr2 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0
                ).to(tl.float32)
                tmp13 = tmp12.to(tl.bfloat16)
                tmp14 = tmp13.to(tl.float32)
                tmp16 = tmp15.to(tl.bfloat16)
                tmp17 = tmp16.to(tl.float32)
                tmp18 = tl.where(tmp11, tmp14, tmp17)
            else:
                tmp18 = tl.load(
                    in_ptr2 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0
                ).to(tl.float32)
            tmp19 = tmp18.to(tl.bfloat16)
            tmp20 = tmp19.to(tl.float32)
            tmp21 = tmp20.to(tl.bfloat16)
            tmp22 = tmp21.to(tl.float32)
            tmp23 = tl.full([1, 1], 1.0, tl.float32)
            tmp24 = tmp22 + tmp23
            tmp25 = tmp24.to(tl.bfloat16)
            tmp26 = tmp25.to(tl.float32)
            tmp27 = tmp26.to(tl.bfloat16)
            tmp28 = tmp27.to(tl.float32)
            tmp29 = tl.load(
                in_ptr0 + (r0_1 + r0_numel * x0),
                r0_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            ).to(tl.float32)
            tmp30 = tmp29.to(tl.bfloat16)
            tmp31 = tmp30.to(tl.float32)
            tmp32 = tmp31.to(tl.float32)
            tmp33 = tmp32 - tmp5
            tmp34 = tl.full([1, 1], r0_numel, tl.float32)
            tmp35 = tmp9 / tmp34
            tmp36 = tl.full([1, 1], EPS, tl.float32)
            tmp37 = tmp35 + tmp36
            tmp38 = libdevice.rsqrt(tmp37)
            tmp39 = tmp33 * tmp38
            tmp40 = tl.load(
                in_ptr3 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0
            ).to(tl.float32)
            tmp41 = tmp40.to(tl.bfloat16)
            tmp42 = tmp41.to(tl.float32)
            tmp43 = tmp42.to(tl.float32)
            tmp44 = tmp39 * tmp43
            tmp45 = tl.load(
                in_ptr4 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0
            ).to(tl.float32)
            tmp46 = tmp45.to(tl.bfloat16)
            tmp47 = tmp46.to(tl.float32)
            tmp48 = tmp47.to(tl.float32)
            tmp49 = tmp44 + tmp48
            tmp50 = tmp49.to(tl.bfloat16)
            tmp51 = tmp50.to(tl.float32)
            tmp52 = tmp51.to(tl.bfloat16)
            tmp53 = tmp52.to(tl.float32)
            tmp54 = tmp53.to(tl.bfloat16)
            tmp55 = tmp54.to(tl.float32)
            tmp56 = tmp28 * tmp55
            tmp57 = tmp56.to(tl.bfloat16)
            tmp58 = tmp57.to(tl.float32)
            tmp59 = tmp58.to(tl.bfloat16)
            tmp60 = tmp59.to(tl.float32)
            if HAS_SELECT:
                tmp61 = tl.load(
                    in_ptr5 + (shift_row_stride + r0_1),
                    r0_mask,
                    eviction_policy="evict_last",
                    other=0.0,
                ).to(tl.float32)
                tmp64 = tl.load(
                    in_ptr5 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0
                ).to(tl.float32)
                tmp62 = tmp61.to(tl.bfloat16)
                tmp63 = tmp62.to(tl.float32)
                tmp65 = tmp64.to(tl.bfloat16)
                tmp66 = tmp65.to(tl.float32)
                tmp67 = tl.where(tmp11, tmp63, tmp66)
            else:
                tmp67 = tl.load(
                    in_ptr5 + (r0_1), r0_mask, eviction_policy="evict_last", other=0.0
                ).to(tl.float32)
            tmp68 = tmp67.to(tl.bfloat16)
            tmp69 = tmp68.to(tl.float32)
            tmp70 = tmp69.to(tl.bfloat16)
            tmp71 = tmp70.to(tl.float32)
            tmp72 = tmp60 + tmp71
            tmp73 = tmp72.to(tl.bfloat16)
            tmp74 = tmp73.to(tl.float32)
            tl.store(out_ptr0 + (r0_1 + r0_numel * x0), tmp73, r0_mask & xmask)

            # ---- NVFP4 quantize of the bf16-rounded x_mod (FlashInfer fast-math path) ----
            # Pairs stay in one thread (4 elements/thread): one packed byte per pair, and the
            # 16-element amax is an in-thread max plus a 4-lane max (exact: bf16 magnitudes).
            pairs = tl.reshape(tmp74, [XBLOCK, Q_BLOCK // 2, 2])
            lo, hi = tl.split(pairs)
            m2 = tl.maximum(tl.abs(lo), tl.abs(hi))
            amax = tl.max(tl.reshape(m2, [XBLOCK, GROUPS, 8]), axis=2)  # [XBLOCK, GROUPS]
            scale_float = gs * (amax * fp4_max_rcp)
            sf_bits = _cvt_e4m3(scale_float)
            out_scale = _output_scale(sf_bits, gs)
            os_p = tl.reshape(
                tl.broadcast_to(out_scale[:, :, None], [XBLOCK, GROUPS, 8]),
                [XBLOCK, Q_BLOCK // 2],
            )
            pbyte = (_e2m1x2(lo, hi, os_p) & 0xFF).to(tl.uint8)
            byte_idx = (r0_offset // 2) + tl.arange(0, Q_BLOCK // 2)[None, :]
            tl.store(
                fp4_ptr + (byte_idx + (r0_numel // 2) * x0),
                pbyte,
                (byte_idx < r0_numel // 2) & xmask,
            )
            col = (r0_offset // 16) + tl.arange(0, GROUPS)[None, :]
            sf_off = (
                (col % 4)
                + (col // 4) * 512
                + (x0 % 32) * 16
                + ((x0 % 128) // 32) * 4
                + (x0 // 128) * (128 * SF_COLS)
            )
            sf_val = tl.where(xmask, sf_bits, 0).to(tl.uint8)
            tl.store(sf_ptr + sf_off, sf_val, (col < r0_numel // 16) & (xindex < padded_xnumel))
        # Column padding of the 128x4 layout (none at D=3072, SF_COLS == D/16).
        if SF_COLS != r0_numel // 16:
            pcol = r0_numel // 16 + tl.arange(0, 4)[None, :]
            poff = (
                (pcol % 4)
                + (pcol // 4) * 512
                + (x0 % 32) * 16
                + ((x0 % 128) // 32) * 4
                + (x0 // 128) * (128 * SF_COLS)
            )
            tl.store(
                sf_ptr + poff,
                tl.zeros([XBLOCK, 4], tl.uint8),
                (pcol < SF_COLS) & (xindex < padded_xnumel),
            )
    else:
        # Padding rows of the 128x4 layout: zero scale factors.
        pcol = tl.arange(0, 256)[None, :]
        for c0 in tl.range(0, SF_COLS, 256):
            c = c0 + pcol
            poff = (
                (c % 4)
                + (c // 4) * 512
                + (x0 % 32) * 16
                + ((x0 % 128) // 32) * 4
                + (x0 // 128) * (128 * SF_COLS)
            )
            tl.store(
                sf_ptr + poff, tl.zeros([XBLOCK, 256], tl.uint8), (c < SF_COLS) & (xindex < padded_xnumel)
            )


def _scale_size(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * ((k // 16 + 3) // 4) * 4


def _launch(
    x, select, scale, shift, weight, bias, global_scale, eps, config=None, outs=None, q_block=None
):
    d = x.shape[-1]
    m = x.numel() // d
    has_select = select is not None
    r0_block, num_warps = config or _INDUCTOR_LN_CONFIG[has_select]
    if outs is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        fp4 = torch.empty((m, d // 2), dtype=torch.uint8, device=x.device)
        sf = torch.empty((_scale_size(m, d),), dtype=torch.uint8, device=x.device)
    else:
        out, fp4, sf = outs
    padded = ((m + 127) // 128) * 128
    _adaln_nvfp4_kernel[(padded,)](
        x,
        select if has_select else x,
        scale,
        weight,
        bias,
        shift,
        global_scale,
        out,
        fp4,
        sf,
        m,
        padded,
        scale.stride(1) if has_select else 0,
        shift.stride(1) if has_select else 0,
        r0_numel=d,
        EPS=eps,
        HAS_SELECT=has_select,
        XBLOCK=1,
        R0_BLOCK=r0_block,
        Q_BLOCK=_Q_BLOCK if q_block is None else q_block,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out, fp4, sf


@torch.library.custom_op("trtllm::flux3_adaln_nvfp4_quant", mutates_args=(), device_types="cuda")
def adaln_nvfp4_quant(
    x: torch.Tensor,
    select: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(1 + scale) * LayerNorm(x) + shift in bf16, plus its NVFP4 data and 128x4 scales.

    x: contiguous [..., D] bf16. Single-block form: scale/shift are [1, 2, D] tables (last dim
    contiguous, any row stride / storage offset) and select is the [..., 1] bool row picker.
    Row form (select None): scale/shift are [1, 1, D] rows.
    """
    return _launch(x, select, scale, shift, weight, bias, global_scale, eps)


@adaln_nvfp4_quant.register_fake
def _adaln_nvfp4_quant_fake(x, select, scale, shift, weight, bias, global_scale, eps):
    d = x.shape[-1]
    m = x.numel() // d
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        x.new_empty((m, d // 2), dtype=torch.uint8),
        x.new_empty((_scale_size(m, d),), dtype=torch.uint8),
    )


def enabled_forms() -> tuple[bool, bool]:
    """(single blocks, mode blocks) from TRTLLM_FLUX3_ADALN_QUANT (default on for both)."""
    value = os.environ.get(_ENV, "1").strip().lower()
    if value in ("0", "off", "false", ""):
        return False, False
    if value == "single":
        return True, False
    return True, True


def _block_ok(block) -> bool:
    if getattr(block, "_fuse_proj_out", True):
        return False
    adapter = getattr(block.ff.gate_up_proj, "_flux3_primsts_nvfp4", None)
    qkv = getattr(block.attn, "_flux3_primsts_nvfp4_qkv", None)
    norm = block.pre_norm
    return (
        isinstance(adapter, PreparedNVFP4)
        and qkv is not None
        and adapter.weight.is_cuda
        # Inline PTX (e2m1x2 cvt) and the bitwise contract are validated on SM120 only; SM100
        # runs the CuTe per-token adaLN instead.
        and torch.cuda.get_device_capability(adapter.weight.device) == (12, 0)
        and block.hidden_size == 3072
        and adapter.weight.shape[1] * 2 == block.hidden_size
        and norm.weight.dtype == torch.bfloat16
        and norm.bias.dtype == torch.bfloat16
        and norm.weight.numel() == block.hidden_size
        and norm.bias.numel() == block.hidden_size
        and not getattr(block, "_pertoken_adaln")._enabled
    )


def prepare_adaln_nvfp4(blocks, single_cls, mode_cls) -> list[str]:
    """Set block._flux3_adaln_nvfp4 (a static bool read in forward) on eligible NVFP4 blocks."""
    single_on, mode_on = enabled_forms()
    names = []
    for name, block in blocks:
        target = getattr(block, "_orig_mod", block)
        want = (type(target) is single_cls and single_on) or (type(target) is mode_cls and mode_on)
        enabled = bool(want and _block_ok(target))
        target._flux3_adaln_nvfp4 = enabled
        if enabled:
            names.append(name)
    return names


def _row_ok(t: torch.Tensor, rows: int, d: int) -> bool:
    return (
        t.dim() == 3
        and t.shape[0] == 1
        and t.shape[1] == rows
        and t.shape[2] == d
        and t.stride(2) == 1
        and t.dtype == torch.bfloat16
    )


def adaln_nvfp4_input(block, x: torch.Tensor, mod: tuple, eps: float):
    """Fused path for a prepared block: returns (x_mod, NVFP4Activation) or None to fall back.

    block._flux3_adaln_nvfp4 is a static bool (set by prepare_adaln_nvfp4); FLUX3_ADALN_NVFP4 is a
    module-level switch (Dynamo guards on both), so one process can compare ON and OFF.
    """
    if not (FLUX3_ADALN_NVFP4 and getattr(block, "_flux3_adaln_nvfp4", False)):
        return None
    d = x.shape[-1]
    if not (
        x.dim() == 3
        and x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.numel() > 0
        and not torch.is_grad_enabled()
    ):
        return None
    if len(mod) == 4:
        shift, scale, _, select = mod
        if not (
            _row_ok(shift, 2, d)
            and _row_ok(scale, 2, d)
            and select.dtype == torch.bool
            and select.shape == (1, x.shape[1], 1)
            and select.is_contiguous()
        ):
            return None
    else:
        shift, scale, _ = mod
        select = None
        if not (_row_ok(shift, 1, d) and _row_ok(scale, 1, d)):
            return None
    adapter = block.ff.gate_up_proj._flux3_primsts_nvfp4
    x_mod, data, scales = torch.ops.trtllm.flux3_adaln_nvfp4_quant(
        x,
        select,
        scale,
        shift,
        block.pre_norm.weight,
        block.pre_norm.bias,
        adapter.inverse_scale,
        eps,
    )
    return x_mod, NVFP4Activation(x_mod, data, scales, adapter.weight)
