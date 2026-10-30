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

// FLUX.3 SM120 warp-per-row LayerNorm + adaLN modulate + NVFP4 quantize (D = 3072): one warp owns one
// row, the row stays in registers (one x read), all reductions are shuffles. The LayerNorm affine is
// the identity (FLUX.3's pre_norm has no weight and no bias): fmul(n, 1.0f) == n for every n, so only
// the +0.0f add stays (it still maps -0 to +0).
//
// Numerics (class: reordered vs the Triton kernel of the adaln_nvfp4 entry). The LN statistics replay
// that kernel's Welford tree: Inductor's looped Welford over a 1024-wide block held by 8 warps x 32
// lanes x 4 contiguous slots, followed by tl.reduce: in-thread combine of the 4 slots, a 32-lane xor
// butterfly (each lane combines (own, partner)), then the 8 warp partials (lane 0 of each warp) through
// an 8-lane xor butterfly. Real lane L of the row's warp holds the virtual threads (w, l) with w = 2g +
// (L >> 4), l = 2 (L & 15) + l0 for g = 0..3, l0 = 0..1, i.e. the 8 contiguous columns 256 g + 8 L
// (+1024 c for chunk c), so every x load is a coalesced 16-byte load. Virtual butterfly offsets 16..2
// are real-lane xor 8..1, offset 1 is in-register. The best-matching tree leaves 1-bf16-ulp differences
// on ~1e-6 of the modulated elements (a residual reduction-order difference); fp4 data and scales were
// bitwise the Triton kernel's in every tested case. fp32 division is div.full.f32 (what Triton emits
// for '/'), and every fp32 op is PTX-pinned (add/sub/mul .rn, no contraction, no ftz), so the build
// flags do not matter.
// Pointwise: the bf16 roundings of (1 + scale), (1 + scale) * norm and + shift take bf16 inputs, so
// fp32-op-then-round equals the single-rounding bf16x2 op (double rounding is innocuous for + and x when
// the wide format has >= 2p + 2 bits: 24 >= 18); the NVFP4 quantize is FlashInfer's fast-math PTX.
//
// Launch: one warp (row) per CTA, __launch_bounds__(32, 12) -> 167 registers, no local memory (12
// warps/SM). 4 warps with 4 blocks/SM (128 registers + a 24 B spill frame) ran 15.1-15.2 us/call vs
// 12.4-12.6 here at M = 4176. Bool / no-select calls walk the rows in descending CTA order: after the
// AlongN pair-residual writer the last-written x rows are still in L2, so reading them first takes the
// 16464-row head from 87.7 to 65.3 us/call. The u8-index form (the i2i single blocks' [1, 3, D] table)
// keeps the ascending order it was measured with.

#include "flux3AdalnNvfp4Sm120.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::flux3
{

#ifndef PLACEHOLDER_KERNELS

namespace adalnNvfp4
{

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef long long i64;

constexpr int kDim = kAdalnHidden;
constexpr int kSfCols = kDim / 16;
constexpr int kWarps = 1;
constexpr int kMinBlocks = 12;

__device__ __forceinline__ float add_rn(float a, float b)
{
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float sub_rn(float a, float b)
{
    float d;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float mul_rn(float a, float b)
{
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float rsqrt_approx(float a)
{
    float d;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float bf_lo(u32 w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf_hi(u32 w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__device__ __forceinline__ u32 word(uint4 const& v, int k)
{
    return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}

__device__ __forceinline__ float elem(uint4 const& v, int e)
{
    u32 w = word(v, e >> 1);
    return (e & 1) ? bf_hi(w) : bf_lo(w);
}

__device__ __forceinline__ u32 cvt_bf16x2(float hi, float lo)
{
    u32 r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ u32 add_bf16x2(u32 a, u32 b)
{
    u32 r;
    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ u32 mul_bf16x2(u32 a, u32 b)
{
    u32 r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ u32 umax(u32 a, u32 b)
{
    return a > b ? a : b;
}

__device__ __forceinline__ u32 max_u16x2(u32 a, u32 b)
{
    u32 r;
    asm("max.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Triton lowers an fp32 '/' to div.full.f32 (not IEEE div.rn).
__device__ __forceinline__ float fdiv(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float rcp_approx3()
{
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(3.0f));
    return r;
}

// Welford-reduce divisions by the constant weights 2 and 3. On SM120, div.full.f32(a, 2) == a * 0.5
// and div.full.f32(a, 3) == a * rcp.approx(3) bit for bit for EVERY finite fp32 a (exhaustive 2^32
// check), so the multiply forms are exact replacements.
__device__ __forceinline__ float div2(float a)
{
    return mul_rn(a, 0.5f);
}

__device__ __forceinline__ float div3(float a)
{
    return mul_rn(a, rcp_approx3());
}

// triton_helpers.welford_combine, no contraction. m2 - m1 is +0 when m1 == m2 (finite), so the
// helper's equality guard only differs for inf - inf.
__device__ __forceinline__ void wcomb(float& m1, float& q1, float& w1, float m2, float q2, float w2)
{
    float delta = sub_rn(m2, m1);
    float nw = add_rn(w1, w2);
    float r = (nw == 0.0f) ? 0.0f : fdiv(w2, nw);
    float nm = add_rn(m1, mul_rn(delta, r));
    float nq = add_rn(add_rn(q1, q2), mul_rn(mul_rn(mul_rn(delta, delta), w1), r));
    m1 = nm;
    q1 = nq;
    w1 = nw;
}

__device__ __forceinline__ float rcp_approx(float x)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ u32 cvt_e4m3(float x)
{
    u32 out;
    asm("{\n .reg .b16 fp8_pair;\n .reg .f32 zero;\n mov.f32 zero, 0f00000000;\n"
        " cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, %1;\n cvt.u32.u16 %0, fp8_pair;\n}"
        : "=r"(out)
        : "f"(x));
    return out;
}

__device__ __forceinline__ float out_scale(u32 sf_bits, float gs)
{
    float out;
    asm("{\n .reg .pred p_zero;\n .reg .b16 fp8_pair;\n .reg .b32 h2_32;\n .reg .b16 h_lo, h_hi;\n"
        " .reg .f32 scale_f32, rcp_gs, product, result;\n cvt.u16.u32 fp8_pair, %1;\n"
        " cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n mov.b32 {h_lo, h_hi}, h2_32;\n"
        " cvt.f32.f16 scale_f32, h_lo;\n rcp.approx.ftz.f32 rcp_gs, %2;\n"
        " mul.f32 product, scale_f32, rcp_gs;\n rcp.approx.ftz.f32 result, product;\n"
        " setp.eq.f32 p_zero, scale_f32, 0f00000000;\n selp.f32 %0, 0f00000000, result, p_zero;\n}"
        : "=f"(out)
        : "r"(sf_bits), "f"(gs));
    return out;
}

// 8 bf16 values (4 bf16x2 words) * out_scale -> 8 e2m1 nibbles (even element in the low nibble).
__device__ __forceinline__ u32 e2m1x8(u32 w0, u32 w1, u32 w2, u32 w3, float os)
{
    u32 out;
    asm("{\n .reg .b8 b0, b1, b2, b3;\n .reg .f32 p0, p1, p2, p3, p4, p5, p6, p7;\n"
        " mul.f32 p0, %1, %9;\n mul.f32 p1, %2, %9;\n mul.f32 p2, %3, %9;\n mul.f32 p3, %4, %9;\n"
        " mul.f32 p4, %5, %9;\n mul.f32 p5, %6, %9;\n mul.f32 p6, %7, %9;\n mul.f32 p7, %8, %9;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b0, p1, p0;\n cvt.rn.satfinite.e2m1x2.f32 b1, p3, p2;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b2, p5, p4;\n cvt.rn.satfinite.e2m1x2.f32 b3, p7, p6;\n"
        " mov.b32 %0, {b0, b1, b2, b3};\n}"
        : "=r"(out)
        : "f"(bf_lo(w0)), "f"(bf_hi(w0)), "f"(bf_lo(w1)), "f"(bf_hi(w1)), "f"(bf_lo(w2)), "f"(bf_hi(w2)),
        "f"(bf_lo(w3)), "f"(bf_hi(w3)), "f"(os));
    return out;
}

// kSelIndex: `sel` is a u8 row index into a K-row table; else a bool picking row 1 (rows walked in
// descending CTA order, see the file comment).
template <bool kSelIndex>
__global__ void __launch_bounds__(kWarps * 32, kMinBlocks) adalnNvfp4WarpKernel(uint4 const* __restrict__ x,
    u8 const* __restrict__ sel, u16 const* __restrict__ scale, u16 const* __restrict__ shift,
    float const* __restrict__ gs_ptr, u32* __restrict__ fp4, u8* __restrict__ sf, int M, int padded, i64 scale_rs,
    i64 shift_rs, float eps, int has_select)
{
    int const lane = threadIdx.x & 31;
    int const row = kSelIndex ? static_cast<int>(blockIdx.x) * kWarps + (threadIdx.x >> 5)
                              : static_cast<int>(gridDim.x - 1 - blockIdx.x) * kWarps + (threadIdx.x >> 5);
    if (row >= padded)
    {
        return;
    }
    i64 const rowpart = (i64) (row & 31) * 16 + (i64) ((row & 127) >> 5) * 4 + (i64) (row >> 7) * (128 * kSfCols);
    if (row >= M)
    {
        // Padding rows of the 128x4 layout: zero scale factors (48 u32 per row).
        for (int k = lane; k < kSfCols / 4; k += 32)
        {
            *(u32*) (sf + (i64) k * 512 + rowpart) = 0u;
        }
        return;
    }
    uint4 const* xr = x + (i64) row * (kDim / 8);
    uint4 v[12];
#pragma unroll
    for (int i = 0; i < 12; ++i)
    {
        v[i] = xr[i * 32 + lane];
    }

    // ---- Inductor's looped Welford per slot (chunks c = 0, 1, 2), then the in-thread combine of the 4
    // slots of each virtual thread k = 2 g + l0 (slot j = column 4 l0 + j of v[.]) ----
    float vm[8], vq[8];
    float wt = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        int const g = k >> 1;
        float mean[4], m2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const e = (k & 1) * 4 + j;
            float mu = elem(v[g], e);
            float q = 0.0f;
            float v1 = elem(v[4 + g], e);
            float d1 = sub_rn(v1, mu);
            mu = add_rn(mu, div2(d1));
            q = add_rn(q, mul_rn(d1, sub_rn(v1, mu)));
            float v2 = elem(v[8 + g], e);
            float d2 = sub_rn(v2, mu);
            mu = add_rn(mu, div3(d2));
            q = add_rn(q, mul_rn(d2, sub_rn(v2, mu)));
            mean[j] = mu;
            m2[j] = q;
        }
        float am = mean[0], aq = m2[0], aw = 3.0f;
        wcomb(am, aq, aw, mean[1], m2[1], 3.0f);
        float bm = mean[2], bq = m2[2], bw = 3.0f;
        wcomb(bm, bq, bw, mean[3], m2[3], 3.0f);
        wcomb(am, aq, aw, bm, bq, bw);
        vm[k] = am;
        vq[k] = aq;
        wt = aw;
    }
    // ---- 32-lane butterfly per virtual warp ----
#pragma unroll
    for (int off = 8; off >= 1; off >>= 1)
    {
        float nw = wt;
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float om = __shfl_xor_sync(0xffffffffu, vm[k], off);
            float oq = __shfl_xor_sync(0xffffffffu, vq[k], off);
            float w1 = wt;
            wcomb(vm[k], vq[k], w1, om, oq, wt);
            nw = w1;
        }
        wt = nw;
    }
    float pm[4], pq[4];
    float pw = wt;
#pragma unroll
    for (int g = 0; g < 4; ++g)
    {
        float m = vm[2 * g], q = vq[2 * g], w = wt;
        // virtual xor 1 last: only virtual lane 0 (l0 = 0) is read afterwards.
        wcomb(m, q, w, vm[2 * g + 1], vq[2 * g + 1], wt);
        pm[g] = m;
        pq[g] = q;
        pw = w;
    }
    // ---- 8 warp partials P_w, w = 2 g + (lane >> 4): P_2g at lane 0, P_2g+1 at lane 16 ----
    // xor 4, 2, 1 at lane 0: comb(comb(comb(P0,P4), comb(P2,P6)), comb(comb(P1,P5), comb(P3,P7)))
    float am = pm[0], aq = pq[0], aw = pw;
    wcomb(am, aq, aw, pm[2], pq[2], pw);
    float bm = pm[1], bq = pq[1], bw = pw;
    wcomb(bm, bq, bw, pm[3], pq[3], pw);
    wcomb(am, aq, aw, bm, bq, bw);
    float fm = __shfl_sync(0xffffffffu, am, 0);
    float fq = __shfl_sync(0xffffffffu, aq, 0);
    float cm = __shfl_sync(0xffffffffu, am, 16);
    float cq = __shfl_sync(0xffffffffu, aq, 16);
    float fw = aw;
    wcomb(fm, fq, fw, cm, cq, aw);

    float const mu = fm;
    float const rstd = rsqrt_approx(add_rn(fdiv(fq, static_cast<float>(kDim)), eps));

    // ---- pointwise modulate + NVFP4 quantize ----
    float const gs = *gs_ptr;
    float const fp4_max_rcp = rcp_approx(6.0f);
    uint4 const* scr;
    uint4 const* shr;
    if constexpr (kSelIndex)
    {
        int const r = has_select ? (int) sel[row] : 0;
        scr = (uint4 const*) (scale + (i64) r * scale_rs);
        shr = (uint4 const*) (shift + (i64) r * shift_rs);
    }
    else
    {
        bool s = false;
        if (has_select)
        {
            s = sel[row] != 0;
        }
        scr = (uint4 const*) (scale + (s ? scale_rs : 0));
        shr = (uint4 const*) (shift + (s ? shift_rs : 0));
    }
    u32* fo = fp4 + (i64) row * (kDim / 8);
#pragma unroll
    for (int i = 0; i < 12; ++i)
    {
        int const vi = i * 32 + lane; // 16-byte column index: columns 256 i + 8 lane .. +7
        uint4 const sc = scr[vi];
        uint4 const sh = shr[vi];
        u32 xmw[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            u32 const xw = word(v[i], k);
            float const t0 = add_rn(mul_rn(sub_rn(bf_lo(xw), mu), rstd), 0.0f);
            float const t1 = add_rn(mul_rn(sub_rn(bf_hi(xw), mu), rstd), 0.0f);
            u32 const t50 = cvt_bf16x2(t1, t0);
            u32 const s1p = add_bf16x2(word(sc, k), 0x3F803F80u);
            xmw[k] = add_bf16x2(mul_bf16x2(s1p, t50), word(sh, k));
        }
        // amax of |bf16| as unsigned 16-bit compare (non-negative floats order like their bits)
        u32 a2 = max_u16x2(
            max_u16x2(xmw[0] & 0x7FFF7FFFu, xmw[1] & 0x7FFF7FFFu), max_u16x2(xmw[2] & 0x7FFF7FFFu, xmw[3] & 0x7FFF7FFFu));
        u32 ab = umax(a2 & 0xFFFFu, a2 >> 16);
        ab = umax(ab, __shfl_xor_sync(0xffffffffu, ab, 1));
        float const amax = __uint_as_float(ab << 16);
        float const scale_float = mul_rn(gs, mul_rn(amax, fp4_max_rcp));
        u32 const sfb = cvt_e4m3(scale_float);
        float const os = out_scale(sfb, gs);
        fo[vi] = e2m1x8(xmw[0], xmw[1], xmw[2], xmw[3], os);
        // SF column 64 c + 16 g + lane/2 = 16 i + lane/2; 4 columns (lanes 8m..8m+7) -> one u32.
        u32 const s1 = __shfl_down_sync(0xffffffffu, sfb, 2);
        u32 const s2 = __shfl_down_sync(0xffffffffu, sfb, 4);
        u32 const s3 = __shfl_down_sync(0xffffffffu, sfb, 6);
        if ((lane & 7) == 0)
        {
            u32 const wd = (sfb & 0xFFu) | ((s1 & 0xFFu) << 8) | ((s2 & 0xFFu) << 16) | ((s3 & 0xFFu) << 24);
            *(u32*) (sf + (i64) (4 * i + (lane >> 3)) * 512 + rowpart) = wd;
        }
    }
}

} // namespace adalnNvfp4

#endif // PLACEHOLDER_KERNELS

void adalnNvfp4WarpSm120(void const* x, uint8_t const* select, bool selectIsIndex, void const* scale,
    void const* shift, int64_t scaleRowStride, int64_t shiftRowStride, float const* globalScale, uint8_t* fp4,
    uint8_t* sf, int m, float eps, cudaStream_t stream)
{
#ifdef PLACEHOLDER_KERNELS
    TLLM_THROW("[FLUX.3 adaLN NVFP4] not compiled for this architecture.");
#else
    using namespace adalnNvfp4;
    int const padded = (m + 127) / 128 * 128;
    int const grid = (padded + kWarps - 1) / kWarps;
    int const hasSelect = select != nullptr ? 1 : 0;
    auto const* xv = static_cast<uint4 const*>(x);
    auto const* sc = static_cast<u16 const*>(scale);
    auto const* sh = static_cast<u16 const*>(shift);
    auto* fo = reinterpret_cast<u32*>(fp4);
    if (selectIsIndex)
    {
        adalnNvfp4WarpKernel<true><<<grid, kWarps * 32, 0, stream>>>(xv, select, sc, sh, globalScale, fo, sf, m,
            padded, hasSelect ? scaleRowStride : 0, hasSelect ? shiftRowStride : 0, eps, hasSelect);
    }
    else
    {
        adalnNvfp4WarpKernel<false><<<grid, kWarps * 32, 0, stream>>>(xv, select, sc, sh, globalScale, fo, sf, m,
            padded, hasSelect ? scaleRowStride : 0, hasSelect ? shiftRowStride : 0, eps, hasSelect);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
#endif
}

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
