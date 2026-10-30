/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// NVFP4 output epilogue of the FLUX.3 FP8 attention (flux3FmhaG3Kernel.cuh). Instead of the BF16 O store it writes
// FP4 (E2M1, [B*S, H*D/2] bytes, two per byte, element 0 in the low nibble) plus E4M3 block scales in the 128x4
// swizzled layout (flat, round_up(B*S, 128) rows x round_up(H*D/16, 4) columns), applying the exact math of
// FlashInfer's cute-dsl nvfp4_quantize (fast-math default) to the BF16 values the BF16 store would have written:
//   block_max = max |bf16| over 16 elements (bf16 max, then max.f32 of the two halves)
//   scale     = gs * (block_max * rcp.approx.ftz(6))       -> cvt.rn.satfinite.e4m3
//   out_scale = rcp.approx.ftz(f32(e4m3) * rcp.approx.ftz(gs)), 0 when the scale is 0
//   q[i]      = cvt.rn.satfinite.e2m1(f32(bf16[i]) * out_scale)
// Padded SF rows (B*S .. round_up(B*S, 128)) are written as 0 by the CTAs of the last Q tile.

#include <fused_multihead_attention.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

#include <cuda_bf16.h>
#include <stdint.h>

namespace fused_multihead_attention
{
namespace flux3
{

// The NVFP4 output of one launch.
struct Nvfp4Out
{
    // [B*S, H*D/2] packed E2M1.
    uint8_t* fp4;
    // The 128x4-swizzled E4M3 block scales.
    uint8_t* sf;
    // The global scale (one float32, device).
    float const* global_scale;
};

namespace nvfp4_out
{

static inline __device__ uint32_t bf2_abs(uint32_t x)
{
    uint32_t r;
    asm("and.b32 %0, %1, 0x7FFF7FFF;" : "=r"(r) : "r"(x));
    return r;
}

static inline __device__ uint32_t bf2_max(uint32_t a, uint32_t b)
{
    uint32_t r;
    asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

static inline __device__ float bf2_hmax_to_f32(uint32_t x)
{
    float r;
    asm("{\n"
        "  .reg .b32 lo, hi;\n"
        "  .reg .f32 f0, f1;\n"
        "  and.b32 lo, %1, 0xFFFF;\n"
        "  shr.b32 hi, %1, 16;\n"
        "  shl.b32 lo, lo, 16;\n"
        "  shl.b32 hi, hi, 16;\n"
        "  mov.b32 f0, lo;\n"
        "  mov.b32 f1, hi;\n"
        "  max.f32 %0, f0, f1;\n"
        "}\n"
        : "=f"(r)
        : "r"(x));
    return r;
}

static inline __device__ float rcp_approx_ftz(float a)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

static inline __device__ float mul_rn(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static inline __device__ uint32_t cvt_f32_to_e4m3(float a)
{
    uint32_t r;
    asm("{\n"
        "  .reg .b16 fp8_pair;\n"
        "  .reg .f32 zero;\n"
        "  mov.f32 zero, 0f00000000;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, %1;\n"
        "  cvt.u32.u16 %0, fp8_pair;\n"
        "}\n"
        : "=r"(r)
        : "f"(a));
    return r;
}

// FlashInfer's nvfp4_compute_output_scale with its rcp.approx.ftz(global_scale) hoisted (rcp_gs): the same
// instructions on the same values, so the same bits.
static inline __device__ float output_scale(uint32_t fp8_val, float rcp_gs)
{
    float r;
    asm("{\n"
        "  .reg .pred p_zero;\n"
        "  .reg .b16 fp8_pair;\n"
        "  .reg .b32 h2_32;\n"
        "  .reg .b16 h_lo, h_hi;\n"
        "  .reg .f32 scale_f32, product, result;\n"
        "  cvt.u16.u32 fp8_pair, %1;\n"
        "  cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n"
        "  mov.b32 {h_lo, h_hi}, h2_32;\n"
        "  cvt.f32.f16 scale_f32, h_lo;\n"
        "  mul.f32 product, scale_f32, %2;\n"
        "  rcp.approx.ftz.f32 result, product;\n"
        "  setp.eq.f32 p_zero, scale_f32, 0f00000000;\n"
        "  selp.f32 %0, 0f00000000, result, p_zero;\n"
        "}\n"
        : "=f"(r)
        : "r"(fp8_val), "f"(rcp_gs));
    return r;
}

// 4 bfloat2 (8 BF16) scaled to f32 and converted to 8 E2M1 in one u32 (element 0 in the low nibble): FlashInfer's
// bfloat2_to_float2_scaled + cvt_e2m1x8_f32, the same instructions.
static inline __device__ uint32_t bf2x4_to_e2m1x8(uint32_t a, uint32_t b, uint32_t c, uint32_t d, float scale)
{
    float f[8];
    f[0] = __uint_as_float(a << 16);
    f[1] = __uint_as_float(a & 0xFFFF0000u);
    f[2] = __uint_as_float(b << 16);
    f[3] = __uint_as_float(b & 0xFFFF0000u);
    f[4] = __uint_as_float(c << 16);
    f[5] = __uint_as_float(c & 0xFFFF0000u);
    f[6] = __uint_as_float(d << 16);
    f[7] = __uint_as_float(d & 0xFFFF0000u);
    uint32_t r;
    asm("{\n"
        "  .reg .f32 s0, s1, s2, s3, s4, s5, s6, s7;\n"
        "  .reg .b8 byte0, byte1, byte2, byte3;\n"
        "  mul.f32 s0, %1, %9;\n"
        "  mul.f32 s1, %2, %9;\n"
        "  mul.f32 s2, %3, %9;\n"
        "  mul.f32 s3, %4, %9;\n"
        "  mul.f32 s4, %5, %9;\n"
        "  mul.f32 s5, %6, %9;\n"
        "  mul.f32 s6, %7, %9;\n"
        "  mul.f32 s7, %8, %9;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 byte0, s1, s0;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 byte1, s3, s2;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 byte2, s5, s4;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 byte3, s7, s6;\n"
        "  mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(r)
        : "f"(f[0]), "f"(f[1]), "f"(f[2]), "f"(f[3]), "f"(f[4]), "f"(f[5]), "f"(f[6]), "f"(f[7]), "f"(scale));
    return r;
}

static inline __device__ int sf_offset_128x4(int row, int col, int padded_cols)
{
    return (col & 3) + (col >> 2) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (row >> 7) * (128 * padded_cols);
}

// Store a 4-warp group's O tile (already reduced into smem by smem_o.store) as NVFP4, reading smem with a
// block-per-lane mapping instead of smem_o.load's 4-columns-per-lane one: lane l of warp w owns the 16-column block
// b = l % 8 of rows w + 16k + 4 * (l / 8), k = 0..3, so each lane holds a whole NVFP4 block (no shuffles; block max,
// E4M3 scale and 1/scale once per block) and writes it as one 8-byte store. The values are smem_o.load's (same fp32
// accumulators, same Acc_packer BF16 conversion with scale_bmm2).
// Layout of Smem_tile_o_base_8bit_mma (Ada e4m3 fp32, 4x1x1 warps, N=128): row r at r * 512 B, logical 16-byte chunk
// c at chunk c ^ (4 * (r & 1)). In step j a lane loads chunk (j + b / 2) & 3 of its block: per LDS the 8 lanes of a
// row hit 8 distinct 16-byte bank slots.
template <typename Gmem_tile_o, typename Smem_tile_o, typename Params>
inline __device__ void store_from_smem(Params const& params, Nvfp4Out const& out, Gmem_tile_o const& gmem_o,
    Smem_tile_o const& smem_o, int mi, int bidb, int bidh, int tidx)
{
    static_assert(Smem_tile_o::BYTES_PER_ROW == 512 && Smem_tile_o::THREADS_PER_ROW == 32
            && Smem_tile_o::ROW_PACKING == 1 && Smem_tile_o::XOR_FACTOR == 4 && Smem_tile_o::WARPS_K == 1
            && Smem_tile_o::LOOPS == 1 && Smem_tile_o::ROWS_PER_LOOP == 64,
        "the NVFP4 epilogue expects the d128 E4M3 fp32 64x128 O tile");
    static_assert(Gmem_tile_o::ROWS == 64 && Gmem_tile_o::LOOPS == 1, "");
    using Packer = fmha::v2::Acc_packer<float, nv_bfloat16>;
    int const warp = tidx >> 5;
    int const lane = tidx & 31;
    int const blk = lane & 7;
    int const rs = lane >> 3;
    int const rot = blk >> 1;
    // smem_o.smem_read_ = base + warp * 512 + ((lane ^ (4 * (warp & 1))) * 16).
    uint32_t const base = smem_o.smem_read_ - uint32_t(warp * 512 + ((lane ^ ((warp & 1) * 4)) * 16));
    uint32_t const blk_base = base + uint32_t((4 * (blk ^ (warp & 1))) * 16);
    int const fp4_row_bytes = params.h * params.dv / 2;
    int const padded_cols = ((params.h * params.dv / 16) + 3) & ~3;
    float const gs = *out.global_scale;
    float const rcp_gs = rcp_approx_ftz(gs);
    float const fp4_max_rcp = rcp_approx_ftz(6.0f);
    int const sf_col = bidh * (params.dv / 16) + blk;
    // gmem_o.row_ = q_loop * 64 + warp (the group's first row plus this warp's row in the tile).
    int const tile_row0 = gmem_o.row_ - warp + mi * 64;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int const r = warp + 16 * k + 4 * rs;
        uint2 bf[4];
        int jr[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            jr[j] = (j + rot) & 3;
            uint4 v;
            fmha::lds(v, blk_base + uint32_t(r * 512 + jr[j] * 16));
            bf[j] = Packer::run(&gmem_o, v);
        }
        uint32_t m = bf2_max(bf2_max(bf2_abs(bf[0].x), bf2_abs(bf[0].y)), bf2_max(bf2_abs(bf[1].x), bf2_abs(bf[1].y)));
        m = bf2_max(
            m, bf2_max(bf2_max(bf2_abs(bf[2].x), bf2_abs(bf[2].y)), bf2_max(bf2_abs(bf[3].x), bf2_abs(bf[3].y))));
        float const block_max = bf2_hmax_to_f32(m);
        float const scale_float = mul_rn(gs, mul_rn(block_max, fp4_max_rcp));
        uint32_t const scale_fp8 = cvt_f32_to_e4m3(scale_float);
        float const out_scale = output_scale(scale_fp8, rcp_gs);
        // Load order j holds chunk (j + rot) & 3: pack in load order, then rotate the 64-bit word left by 16 * rot so
        // chunk c lands at bits [16c, 16c + 16).
        uint32_t lo = bf2x4_to_e2m1x8(bf[0].x, bf[0].y, bf[1].x, bf[1].y, out_scale);
        uint32_t hi = bf2x4_to_e2m1x8(bf[2].x, bf[2].y, bf[3].x, bf[3].y, out_scale);
        if (rot & 2)
        {
            uint32_t const t = lo;
            lo = hi;
            hi = t;
        }
        uint32_t const sh = (rot & 1) * 16;
        uint32_t const nlo = __funnelshift_l(hi, lo, sh);
        uint32_t const nhi = __funnelshift_l(lo, hi, sh);
        uint64_t const packed = (uint64_t(nhi) << 32) | nlo;
        int const row = tile_row0 + r;
        if (row < gmem_o.actual_seqlen_)
        {
            int64_t const grow = int64_t(bidb) * params.s + row;
            *reinterpret_cast<uint64_t*>(out.fp4 + grow * fp4_row_bytes + bidh * (params.dv / 2) + blk * 8) = packed;
            out.sf[sf_offset_128x4(int(grow), sf_col, padded_cols)] = uint8_t(scale_fp8 & 0xFF);
        }
    }
}

// Zero this head's SF columns of the padded rows [B*S, round_up(B*S, 128)). Called by every 4-warp group; only the
// groups of the last Q tile of the last batch write. tidx is group-local (0..127) and nthreads its index space.
template <typename Params>
inline __device__ void zero_padding(
    Params const& params, Nvfp4Out const& out, int q_tile_rows, int q_loop, int bidb, int bidh, int tidx, int nthreads)
{
    int const total = params.b * params.s;
    int const last_q_loop = (params.s + q_tile_rows - 1) / q_tile_rows - 1;
    if (bidb != params.b - 1 || q_loop != last_q_loop)
    {
        return;
    }
    int const padded_rows = (total + 127) & ~127;
    int const cols_per_head = params.dv / 16;
    int const padded_cols = ((params.h * params.dv / 16) + 3) & ~3;
    int const n = (padded_rows - total) * cols_per_head;
    for (int i = tidx; i < n; i += nthreads)
    {
        int const row = total + i / cols_per_head;
        int const col = bidh * cols_per_head + i % cols_per_head;
        out.sf[sf_offset_128x4(row, col, padded_cols)] = 0;
    }
    // Padded SF columns (H*D/16 not a multiple of 4) do not occur: H*D/16 = 8H.
}

} // namespace nvfp4_out
} // namespace flux3
} // namespace fused_multihead_attention
