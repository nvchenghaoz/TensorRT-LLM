# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Triton kernel for SwiGLU + NVFP4 (group-16, E4M3 SF) quantization.

Replaces ``torch.ops.trtllm.silu_and_mul`` (Triton ``silu_and_mul_kernel``,
BF16 out) followed by ``torch.ops.trtllm.fp4_quantize(h, global_scale, 16,
False)`` (CUDA ``quantize_with_block_size`` / ``cvt_warp_fp16_to_fp4``) with
bitwise-identical FP4 bytes and swizzled (128x4) SF bytes, including the
zeroed SF padding rows up to a multiple of 128.

Bitwise contract:
  * SwiGLU uses exactly ``silu_and_mul_kernel``'s expression (same constexpr
    alpha=1.0 / beta=0.0) and is rounded to BF16, as the stored intermediate.
  * The quantizer reproduces ``cvt_warp_fp16_to_fp4`` as built (FAST_MATH
    off): ``rcp.approx.ftz.f32``, non-ftz ``mul.rn.f32`` (subnormals kept),
    ``cvt.rn.satfinite.e4m3x2.f32`` for the SF, ``vecMax != 0``, and
    ``cvt.rn.satfinite.e2m1x2.f32`` with the same nibble order (even element
    in the low nibble).
"""

import os

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

_SF_VEC = 16


@triton.jit
def _gdc_wait_with_memory_clobber():
    # griddepcontrol.wait with a memory clobber, so no load or store of this
    # kernel can be scheduled above it.
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",
                                     "=f,f", [x],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _mul_rn(a, b):
    # quantization.cuh is built without --use_fast_math (FAST_MATH off), so
    # its fp32 multiplies keep subnormals; only rcp.approx is .ftz.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;",
                                     "=f,f,f", [a, b],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _cvt_e4m3_bits(x):
    # Low byte = e4m3(x), round-to-nearest-even, satfinite.
    return tl.inline_asm_elementwise(
        "{\n"
        ".reg .b16 t;\n"
        "cvt.rn.satfinite.e4m3x2.f32 t, $2, $1;\n"
        "cvt.u32.u16 $0, t;\n"
        "}", "=r,f,f", [x, tl.zeros_like(x)],
        dtype=tl.int32,
        is_pure=True,
        pack=1)


@triton.jit
def _cvt_e2m1x8(e0, e1, e2, e3, e4, e5, e6, e7):
    # Same instruction sequence as fp32_vec_to_e2m1 in quantization.cuh.
    return tl.inline_asm_elementwise(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, $2, $1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, $4, $3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, $6, $5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, $8, $7;\n"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n"
        "}",
        "=r,f,f,f,f,f,f,f,f", [e0, e1, e2, e3, e4, e5, e6, e7],
        dtype=tl.int32,
        is_pure=True,
        pack=1)


@triton.jit
def _sf_offset(m, c, NUM_K_TILES: tl.constexpr):
    # get_sf_out_offset_128x4 (quantization.cuh), batch 0.
    return (((m // 128) * NUM_K_TILES + c // 4) * 512 + (m % 32) * 16 +
            ((m % 128) // 32) * 4 + c % 4)


@triton.jit
def fused_silu_mul_nvfp4_kernel(fp4_ptr, sf_ptr, x_ptr, x_stride, scale_ptr, M,
                                d, swiglu_alpha: tl.constexpr,
                                swiglu_beta: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr,
                                NUM_K_TILES: tl.constexpr,
                                USE_PDL: tl.constexpr = False,
                                PERMUTED: tl.constexpr = False) -> None:
    if USE_PDL:
        # Let the consumer GEMM launch (and prefetch its weights) now; every
        # access of this kernel stays behind the wait.
        tl.extra.cuda.gdc_launch_dependents()
        _gdc_wait_with_memory_clobber()
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    NSF: tl.constexpr = BLOCK_SIZE // 16
    sf_cols = j * NSF + tl.arange(0, NSF)
    num_sf_cols = d // 16

    if i < M:
        x_row_ptr = x_ptr + x_stride * i
        offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < d

        if PERMUTED:
            # gate_up rows permuted for the SwiGLU GEMM epilogue
            # (gate_up_swiglu_row_permutation): output j's gate/up sit at
            # adjacent columns of its 128-column tile. Only the load
            # addresses change; the arithmetic below is unchanged.
            jl = offsets % 64
            col = ((offsets // 64) * 128 + 32 * (2 * ((jl % 32) // 16) +
                                                 (jl % 4) // 2) +
                   16 * (jl // 32) + 8 * (jl % 2) + 2 * ((jl % 16) // 4))
            a = tl.load(x_row_ptr + col, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(x_row_ptr + col + 1, mask=mask,
                        other=0.0).to(tl.float32)
        else:
            a = tl.load(x_row_ptr + offsets, mask=mask,
                        other=0.0).to(tl.float32)
            b = tl.load(x_row_ptr + offsets + d, mask=mask,
                        other=0.0).to(tl.float32)
        # Exactly silu_and_mul_kernel's expression (plain SwiGLU).
        result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)
        # The unfused path stores BF16 and the quantizer reloads it.
        h = result.to(tl.bfloat16).to(tl.float32)

        h2 = tl.reshape(h, [NSF, 16])
        vmax = tl.max(tl.abs(h2), axis=1)  # [NSF], exact (max of BF16 values)

        s = tl.load(scale_ptr)
        s_v = tl.full([NSF], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([NSF], 6.0, tl.float32))
        sf_val = _mul_rn(s_v, _mul_rn(vmax, r6))
        sf_bits = _cvt_e4m3_bits(sf_val)
        sf_u8 = sf_bits.to(tl.uint8)
        sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        out_scale = tl.where(
            vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, _rcp_approx_ftz(s_v))),
            0.0)

        y = _mul_rn(h2, tl.broadcast_to(out_scale[:, None], [NSF, 16]))
        y8 = tl.reshape(y, [BLOCK_SIZE // 8, 2, 2, 2])
        p, q = tl.split(y8)  # last index 0 / 1
        p0, p1 = tl.split(p)
        q0, q1 = tl.split(q)
        p00, p01 = tl.split(p0)
        p10, p11 = tl.split(p1)
        q00, q01 = tl.split(q0)
        q10, q11 = tl.split(q1)
        # element index = 4*i + 2*j + k for [.., i, j, k]
        packed = _cvt_e2m1x8(p00, q00, p10, q10, p01, q01, p11, q11)

        w_offsets = j * (BLOCK_SIZE // 8) + tl.arange(0, BLOCK_SIZE // 8)
        fp4_row_ptr = fp4_ptr + i * (d // 8)
        tl.store(fp4_row_ptr + w_offsets, packed, mask=w_offsets < d // 8)

        sf_off = _sf_offset(i, sf_cols, NUM_K_TILES)
        tl.store(sf_ptr + sf_off, sf_u8, mask=sf_cols < num_sf_cols)
    else:
        # One extra program row zeroes the SF padding rows [M, ceil128(M))
        # for this column block (same as quantize_with_block_size).
        m0 = (M // 128) * 128
        rows = m0 + tl.arange(0, 128).to(tl.int64)
        rmask = rows >= M
        off = _sf_offset(rows[:, None], sf_cols[None, :], NUM_K_TILES)
        pmask = rmask[:, None] & (sf_cols < num_sf_cols)[None, :]
        tl.store(sf_ptr + off, tl.zeros([128, NSF], tl.uint8), mask=pmask)


def _fused_silu_mul_nvfp4_impl(x: torch.Tensor,
                               global_scale: torch.Tensor,
                               permuted: bool = False):
    assert x.dim() == 2 and x.stride(1) == 1
    m, n = x.shape
    assert n % 2 == 0
    d = n // 2
    assert d % 64 == 0, "SF columns must be a multiple of 4"
    fp4 = torch.empty((m, d // 2), dtype=torch.uint8, device=x.device)
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (d // _SF_VEC + 3) // 4
    sf = torch.empty((m_pad * num_k_tiles * 4, ),
                     dtype=torch.uint8,
                     device=x.device)
    _launch(x, global_scale, fp4, sf, permuted=permuted)
    return fp4, sf


def gate_up_swiglu_row_permutation(n: int) -> torch.Tensor:
    """Row order of the gate_up weight for the SM120 SwiGLU GEMM epilogue.

    Permuted row r holds source row perm[r] of the [gate | up] weight. Inside
    every 128-row tile t, row c = 32a + 16nw + 8b + 2lq + p is gate (p = 0) or
    up (p = 1) of output j = 64t + 32nw + 16(a // 2) + 4lq + 2(a % 2) + b
    (see nvfp4_swiglu_gemm_sm120.h). The PERMUTED mode of
    fused_silu_mul_nvfp4_kernel inverts this map.
    """
    assert n % 128 == 0
    d = n // 2
    c = torch.arange(128)
    a, nw = c // 32, (c % 32) // 16
    b, lq, p = (c % 16) // 8, (c % 8) // 2, c % 2
    jl = 32 * nw + 4 * lq + 16 * (a // 2) + 2 * (a % 2) + b
    j = 64 * torch.arange(n // 128)[:, None] + jl[None, :]
    return torch.where(p[None, :] == 0, j, d + j).reshape(-1)


def _launch(x: torch.Tensor,
            global_scale: torch.Tensor,
            fp4: torch.Tensor,
            sf: torch.Tensor,
            permuted: bool = False) -> None:
    """Write FP4 [m, d/2] and swizzled SF [ceil128(m) * ceil4(d/16)]."""
    m, n = x.shape
    d = n // 2
    num_k_tiles = (d // _SF_VEC + 3) // 4
    block = 512
    grid_rows = m + (1 if m % 128 != 0 else 0)
    if grid_rows > 0:
        grid = (grid_rows, triton.cdiv(d, block))
        # PDL only while the whole grid is resident at once (one wave), so
        # early-launched consumer CTAs never wait behind unscheduled ones.
        use_pdl = (_PDL_ENABLED and grid[0] * grid[1] < _num_sms(x.device))
        fused_silu_mul_nvfp4_kernel[grid](
            fp4.view(torch.int32),
            sf,
            x,
            x.stride(0),
            global_scale,
            m,
            d,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            BLOCK_SIZE=block,
            NUM_K_TILES=num_k_tiles,
            USE_PDL=use_pdl,
            PERMUTED=permuted,
            num_warps=4,
            launch_pdl=use_pdl,
        )


# TRTLLM_FUSED_SILU_NVFP4_PDL=0 restores the plain launch.
_PDL_ENABLED = (os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
                and os.environ.get("TRTLLM_FUSED_SILU_NVFP4_PDL", "1") != "0")
_NUM_SMS = None


def _num_sms(device: torch.device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            device).multi_processor_count
    return _NUM_SMS


@torch.library.custom_op("trtllm::fused_silu_mul_nvfp4", mutates_args=())
def fused_silu_mul_nvfp4(
        x: torch.Tensor,
        global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_silu_mul_nvfp4_impl(x, global_scale)


@fused_silu_mul_nvfp4.register_fake
def _(x: torch.Tensor, global_scale: torch.Tensor):
    m, n = x.shape
    d = n // 2
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (d // _SF_VEC + 3) // 4
    return (x.new_empty((m, d // 2), dtype=torch.uint8),
            x.new_empty((m_pad * num_k_tiles * 4, ), dtype=torch.uint8))


@torch.library.custom_op("trtllm::fused_silu_mul_nvfp4_permuted",
                         mutates_args=())
def fused_silu_mul_nvfp4_permuted(
        x: torch.Tensor,
        global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fused_silu_mul_nvfp4 for a gate_up output whose columns are in
    gate_up_swiglu_row_permutation order; same bytes as the unpermuted op."""
    return _fused_silu_mul_nvfp4_impl(x, global_scale, permuted=True)


@fused_silu_mul_nvfp4_permuted.register_fake
def _(x: torch.Tensor, global_scale: torch.Tensor):
    m, n = x.shape
    d = n // 2
    m_pad = (m + 127) // 128 * 128
    num_k_tiles = (d // _SF_VEC + 3) // 4
    return (x.new_empty((m, d // 2), dtype=torch.uint8),
            x.new_empty((m_pad * num_k_tiles * 4, ), dtype=torch.uint8))


try:
    if hasattr(torch.ops.trtllm, "nvfp4_gate_up_swiglu_nvfp4"):

        @torch.library.register_fake("trtllm::nvfp4_gate_up_swiglu_nvfp4")
        def _(mat1, mat2, mat1Scale, mat2Scale, alpha, outScale, tile=0):
            m, n = mat1.shape[0], mat2.shape[0]
            d = n // 2
            m_pad = (m + 127) // 128 * 128
            num_k_tiles = (d // _SF_VEC + 3) // 4
            return [
                mat1.new_empty((m, d // 2), dtype=torch.uint8),
                mat1.new_empty((m_pad * num_k_tiles * 4, ), dtype=torch.uint8)
            ]
except Exception:  # noqa: BLE001 - fake registration is optional
    pass
