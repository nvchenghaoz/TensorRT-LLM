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

// SM120 decode wo_a for the CSA2 output projection (DeepSeek-V4.1 TP rank: groups x [rank, groupWidth] E4M3 with
// 128x128 UE8M0 block scales), in the structure of csa2_mxfp8_low_m_gemm_kernel (mxfp8LowMGemm.cu):
//  - warp 0 bulk-copies the CTA's 16*RT weight rows x groupWidth/CLUSTER_K columns into shared memory BEFORE
//    griddepcontrol.wait (static data), and every thread triggers the dependents at start (kTriggerAtStart);
//  - after the wait, the CTA applies the BF16 inverse RoPE and the 1x128 UE8M0 quantization to its K slice for every
//    token, with the exact instruction sequence of b12x_wo._inv_rope_quant128_kernel (Triton): fma.rn rotary form
//    r_even = fma(c, x, -(ns * y)), r_odd = fma(c, y, ns * x) with ns = -sin, rounded to BF16; amax = max(|v|, 1e-10);
//    sc = ex2.approx.ftz(cvt.rpi.ftz(__nv_log2f(|div.full(amax, 448)|))); q = e4m3(clamp(div.full(v, sc), +-448));
//  - m16n8k32 e4m3 MMAs, C zeroed per 32-k block, acc += d * (ws * xs) in FP32, fixed-order warp + cluster (DSMEM)
//    split-K reduction, BF16 out[m, g * rank + r].

#include "deepseek_v4/v41_flash_1017__sm_120__tp4_moetp2ep2/csrc/csa2LowMWo.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
#define CSA2_LOW_M_WO_ENABLED 1
#else
#define CSA2_LOW_M_WO_ENABLED 0
#endif

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace cg = cooperative_groups;

namespace
{

constexpr int kWoRowPad = 32;

struct WoaParams
{
    __nv_bfloat16 const* o;
    int ldo;
    void const* pos;
    int pos64;
    float const* cs;
    int cs_stride;
    uint8_t const* w;
    uint8_t const* wsc;
    __nv_bfloat16* out;
    int ld_out;
    int M, G, R, GW, head_dim, nope, half_rope;
    uint8_t* xq_dbg;
    uint8_t* xs_dbg;
    int flags;
    float fp8_max;
};

__device__ __forceinline__ uint32_t wo_smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t wo_mapa(uint32_t addr, uint32_t target_rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(target_rank));
    return r;
}

__device__ __forceinline__ float wo_pow2(uint32_t e)
{
    return __uint_as_float(e == 0 ? 0x00400000u : (e << 23));
}

__device__ __forceinline__ void wo_mma_e4m3(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

__device__ __forceinline__ void wo_bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     wo_smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(wo_smem_u32(bar))
                 : "memory");
}

__device__ __forceinline__ void wo_mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra "
        "WAIT_%=;\n}\n" ::"r"(wo_smem_u32(bar)),
        "r"(parity)
        : "memory");
}

// sc = exp2(ceil(log2(|amax / fp8_max|))) with the instructions of the Triton K1 (libdevice __nv_log2f inlined).
__device__ __forceinline__ float wo_k1_scale(float amax, float fp8_max)
{
    float r;
    asm("{\n"
        ".reg .pred p2, p3, p5;\n"
        ".reg .b32 q, a, s, xs, e0, i100, i101, i102, f103, f105, f106, k, t, u, v, w, z;\n"
        "div.full.f32 q, %1, %2;\n"
        "abs.f32 a, q;\n"
        "setp.lt.f32 p2, a, 0f00800000;\n"
        "mul.f32 s, a, 0f4B000000;\n"
        "selp.f32 xs, s, a, p2;\n"
        "selp.f32 e0, 0fC1B80000, 0f00000000, p2;\n"
        "add.s32 i100, xs, -1060439283;\n"
        "and.b32 i101, i100, -8388608;\n"
        "sub.s32 i102, xs, i101;\n"
        "cvt.rn.f32.s32 f103, i101;\n"
        "mov.b32 k, 0f34000000;\n"
        "fma.rn.ftz.f32 f105, f103, k, e0;\n"
        "add.f32 f106, i102, 0fBF800000;\n"
        "mov.b32 k, 0fBE2C7F30;\n"
        "mov.b32 t, 0f3DC6B27F;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0f3E2FCF2A;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0fBE374E43;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0f3E520BF4;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0fBE763C8B;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0f3E93BF99;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0fBEB8AA49;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0f3EF6384A;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mov.b32 k, 0fBF38AA3B;\n"
        "fma.rn.ftz.f32 t, t, f106, k;\n"
        "mul.f32 u, f106, t;\n"
        "mul.f32 v, f106, u;\n"
        "mov.b32 k, 0f3FB8AA3B;\n"
        "fma.rn.ftz.f32 w, f106, k, v;\n"
        "add.f32 z, f105, w;\n"
        "setp.lt.u32 p3, xs, 2139095040;\n"
        "@!p3 mov.b32 k, 0f7F800000;\n"
        "@!p3 fma.rn.ftz.f32 z, xs, k, k;\n"
        "setp.eq.f32 p5, xs, 0f00000000;\n"
        "selp.f32 z, 0fFF800000, z, p5;\n"
        "cvt.rpi.ftz.f32.f32 z, z;\n"
        "ex2.approx.ftz.f32 %0, z;\n"
        "}\n"
        : "=f"(r)
        : "f"(amax), "f"(fp8_max));
    return r;
}

__device__ __forceinline__ float wo_div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

template <int WARPS, int CLUSTER_K, int RT>
__global__ void __launch_bounds__(WARPS * 32) csa2_lowm_wo_a_kernel(WoaParams const p, int red_aliased)
{
#if CSA2_LOW_M_WO_ENABLED
    // Trigger the dependents right after the pre-wait weight staging: the wo_b kernel that follows only stages its
    // own weights before its griddepcontrol.wait, so its staging overlaps this kernel.
    constexpr bool kTriggerAtStart = true;
    constexpr int kThreads = WARPS * 32;
    constexpr int kRows = 16 * RT;
    constexpr int kTile = kRows * 8;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bar;
    __shared__ __align__(8) uint64_t red_bar;

    int rank = 0;
    if constexpr (CLUSTER_K > 1)
    {
        rank = static_cast<int>(cg::this_cluster().block_rank());
    }
    int const M = p.M;
    int const tile = blockIdx.x / CLUSTER_K;
    int const row0 = tile * kRows;
    int const g = row0 / p.R;
    int const lrow0 = row0 - g * p.R;
    int const kc = p.GW / CLUSTER_K;
    int const kc_blocks = kc / 32;
    int const kb128 = kc / 128;
    int const k_begin = rank * kc;
    int const row_stride = kc + kWoRowPad;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const live = kRows * M;

    uint8_t* w_s = smem;                                 // [kRows][row_stride]
    uint8_t* xq_s = w_s + kRows * row_stride;            // [M][kc]
    uint8_t* xs_s = xq_s + M * kc;                       // [M][kb128]
    uint8_t* ws_s = xs_s + ((M * kb128 + 15) / 16) * 16; // [kb128]
    float* cluster_red = reinterpret_cast<float*>(ws_s + ((kb128 + 15) / 16) * 16); // [CLUSTER_K-1][live]
    float* red = red_aliased ? reinterpret_cast<float*>(w_s) : cluster_red + (CLUSTER_K - 1) * live;

    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(wo_smem_u32(&bar)));
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(wo_smem_u32(&red_bar)));
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(wo_smem_u32(&bar)),
                     "r"(static_cast<uint32_t>(kRows * kc))
                     : "memory");
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(wo_smem_u32(&red_bar)),
                         "r"(static_cast<uint32_t>((CLUSTER_K - 1) * live * sizeof(float)))
                         : "memory");
        }
    }
    __syncthreads();
    if constexpr (CLUSTER_K > 1)
    {
        asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    }
    // Weight rows and block scales do not depend on the producer kernel: stage them before the PDL wait.
    if (warp == 0)
    {
        for (int r = lane; r < kRows; r += 32)
        {
            wo_bulk_g2s(w_s + r * row_stride, p.w + static_cast<int64_t>(row0 + r) * p.GW + k_begin, kc, &bar);
        }
    }
    {
        int const rb = lrow0 / 128;
        int const kt = p.GW / 128;
        for (int t = threadIdx.x; t < kb128; t += kThreads)
        {
            ws_s[t] = p.wsc[(g * (p.R / 128) + rb) * kt + k_begin / 128 + t];
        }
    }
    if constexpr (kTriggerAtStart)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if ((p.flags & 1) == 0)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    // Inverse RoPE + 1x128 quantization of this CTA's K slice for every token: 4 lanes x 32 values per 128-block.
    {
        int const items = M * kb128;
        int const sub = threadIdx.x & 3;
        bool const dump = p.xq_dbg != nullptr && lrow0 == 0;
        constexpr int kPer = kThreads / 4;
        int const rounds = (items + kPer - 1) / kPer;
        for (int it = 0; it < rounds; ++it)
        {
            int const q = it * kPer + (threadIdx.x >> 2);
            bool const act = q < items;
            int const m = act ? q / kb128 : 0;
            int const b = act ? q - m * kb128 : 0;
            int const kq = k_begin + b * 128 + sub * 32; // column within the group
            float v[32] = {};
            if (act)
            {
                uint4 const* src = reinterpret_cast<uint4 const*>(
                    p.o + static_cast<int64_t>(m) * p.ldo + static_cast<int64_t>(g) * p.GW + kq);
                uint4 raw[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    raw[u] = src[u];
                }
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw[u]);
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        float2 const f = __bfloat1622float2(h[t]);
                        v[u * 8 + t * 2] = f.x;
                        v[u * 8 + t * 2 + 1] = f.y;
                    }
                }
                int const hd = kq % p.head_dim;
                if (hd >= p.nope)
                {
                    int const ci0 = (hd - p.nope) >> 1;
                    int64_t const pos = p.pos64 ? static_cast<int64_t>(static_cast<int64_t const*>(p.pos)[m])
                                                : static_cast<int64_t>(static_cast<int const*>(p.pos)[m]);
                    float const* cb = p.cs + pos * p.cs_stride;
                    float c[16], s[16];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                    {
                        float4 const cv = *reinterpret_cast<float4 const*>(cb + ci0 + 4 * u);
                        float4 const sv = *reinterpret_cast<float4 const*>(cb + p.half_rope + ci0 + 4 * u);
                        c[4 * u] = cv.x;
                        c[4 * u + 1] = cv.y;
                        c[4 * u + 2] = cv.z;
                        c[4 * u + 3] = cv.w;
                        s[4 * u] = sv.x;
                        s[4 * u + 1] = sv.y;
                        s[4 * u + 2] = sv.z;
                        s[4 * u + 3] = sv.w;
                    }
#pragma unroll
                    for (int t = 0; t < 16; ++t)
                    {
                        float const xe = v[2 * t];
                        float const xo = v[2 * t + 1];
                        float const ns = -s[t];
                        float const re = __fmaf_rn(c[t], xe, -__fmul_rn(ns, xo));
                        float const ro = __fmaf_rn(c[t], xo, __fmul_rn(ns, xe));
                        v[2 * t] = __bfloat162float(__float2bfloat16_rn(re));
                        v[2 * t + 1] = __bfloat162float(__float2bfloat16_rn(ro));
                    }
                }
            }
            float amax = 0.f;
#pragma unroll
            for (int t = 0; t < 32; ++t)
            {
                amax = fmaxf(amax, fabsf(v[t]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            if (!act)
            {
                continue;
            }
            amax = fmaxf(amax, 1e-10f);
            float const sc = wo_k1_scale(amax, p.fp8_max);
            uint32_t packed[8];
#pragma unroll
            for (int t = 0; t < 8; ++t)
            {
                float f[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    f[u] = fminf(fmaxf(wo_div_full(v[4 * t + u], sc), -p.fp8_max), p.fp8_max);
                }
                __nv_fp8x2_e4m3 const lo(make_float2(f[0], f[1]));
                __nv_fp8x2_e4m3 const hi(make_float2(f[2], f[3]));
                packed[t] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
            }
            uint8_t const byte = static_cast<uint8_t>((__float_as_uint(sc) >> 23) & 0xFF);
            uint4* dst = reinterpret_cast<uint4*>(xq_s + m * kc + b * 128 + sub * 32);
            dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
            if (sub == 0)
            {
                xs_s[m * kb128 + b] = byte;
            }
            if (dump)
            {
                int64_t const row = static_cast<int64_t>(m) * p.G + g;
                uint4* dd = reinterpret_cast<uint4*>(p.xq_dbg + row * p.GW + kq);
                dd[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
                dd[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
                if (sub == 0)
                {
                    p.xs_dbg[row * (p.GW / 128) + kq / 128] = byte;
                }
            }
        }
    }
    __syncthreads();

    int const gq = lane >> 2;
    int const j = lane & 3;
    int const col0 = 2 * j;
    int const bps = kc_blocks / WARPS;
    float acc[RT][4] = {};
    wo_mbar_wait(&bar, 0);
#pragma unroll 2
    for (int i = 0; i < bps; ++i)
    {
        int const b = warp * bps + i;
        int const b4 = b >> 2;
        uint2 const xb
            = gq < M ? *reinterpret_cast<uint2 const*>(xq_s + gq * kc + b * 32 + 8 * j) : make_uint2(0, 0);
        float const ws = wo_pow2(ws_s[b4]);
        float const xs0 = col0 < M ? wo_pow2(xs_s[col0 * kb128 + b4]) : 0.f;
        float const xs1 = col0 + 1 < M ? wo_pow2(xs_s[(col0 + 1) * kb128 + b4]) : 0.f;
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            uint8_t const* wa_s = w_s + (16 * rt + gq) * row_stride + 8 * j + b * 32;
            uint2 const wa = *reinterpret_cast<uint2 const*>(wa_s);
            uint2 const wb = *reinterpret_cast<uint2 const*>(wa_s + 8 * row_stride);
            float d[4];
            wo_mma_e4m3(d, wa.x, wb.x, wa.y, wb.y, xb.x, xb.y);
            acc[rt][0] += d[0] * (ws * xs0);
            acc[rt][1] += d[1] * (ws * xs1);
            acc[rt][2] += d[2] * (ws * xs0);
            acc[rt][3] += d[3] * (ws * xs1);
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
    if (red_aliased)
    {
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
    {
        int const ra = 16 * rt + gq;
        red[warp * kTile + ra * 8 + col0] = acc[rt][0];
        red[warp * kTile + ra * 8 + col0 + 1] = acc[rt][1];
        red[warp * kTile + (ra + 8) * 8 + col0] = acc[rt][2];
        red[warp * kTile + (ra + 8) * 8 + col0 + 1] = acc[rt][3];
    }
    __syncthreads();
    if constexpr (CLUSTER_K > 1)
    {
        asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    }
    if (CLUSTER_K == 1 || rank == 0)
    {
        if constexpr (CLUSTER_K > 1)
        {
            uint32_t ready = 0;
            while (!ready)
            {
                asm volatile(
                    "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                    " selp.u32 %0, 1, 0, p;\n}\n"
                    : "=r"(ready)
                    : "r"(wo_smem_u32(&red_bar))
                    : "memory");
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < live; idx += kThreads)
        {
            int const r = idx / M;
            int const m = idx - r * M;
            float total = 0.f;
#pragma unroll
            for (int ww = 0; ww < WARPS; ++ww)
            {
                total += red[ww * kTile + r * 8 + m];
            }
#pragma unroll
            for (int c = 1; c < CLUSTER_K; ++c)
            {
                total += cluster_red[(c - 1) * live + idx];
            }
            p.out[static_cast<int64_t>(m) * p.ld_out + row0 + r] = __float2bfloat16_rn(total);
        }
    }
    else
    {
        uint32_t const remote_bar = wo_mapa(wo_smem_u32(&red_bar), 0);
        uint32_t const remote_dst = wo_mapa(wo_smem_u32(cluster_red + (rank - 1) * live), 0);
        for (int idx = threadIdx.x; idx < live; idx += kThreads)
        {
            int const r = idx / M;
            int const m = idx - r * M;
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < WARPS; ++ww)
            {
                s += red[ww * kTile + r * 8 + m];
            }
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(
                             remote_dst + idx * 4),
                         "r"(__float_as_uint(s)), "r"(remote_bar)
                         : "memory");
        }
    }
    __syncthreads();
#endif
}

size_t woaSmemBytes(int warps, int ck, int rt, int m, int gw)
{
    int const kc = gw / ck;
    int const kb128 = kc / 128;
    size_t const rows = 16 * rt;
    size_t const w_bytes = rows * (kc + kWoRowPad);
    size_t const red_bytes = static_cast<size_t>(warps) * rows * 8 * sizeof(float);
    bool const aliased = red_bytes <= w_bytes;
    return w_bytes + static_cast<size_t>(m) * kc + ((m * kb128 + 15) / 16) * 16 + ((kb128 + 15) / 16) * 16
        + static_cast<size_t>(ck - 1) * rows * m * sizeof(float) + (aliased ? 0 : red_bytes);
}

template <int WARPS, int CLUSTER_K, int RT>
void launchWoA(WoaParams const& p, bool pdl, cudaStream_t stream)
{
    auto kernel = csa2_lowm_wo_a_kernel<WARPS, CLUSTER_K, RT>;
    static bool const configured = [&]()
    {
        int device = 0, max_optin = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cudaFuncAttributes attrs = {};
        if (cudaFuncGetAttributes(&attrs, kernel) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        bool ok = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                      max_optin - static_cast<int>(attrs.sharedSizeBytes))
            == cudaSuccess;
        ok = ok
            && cudaFuncSetAttribute(
                   kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)
                == cudaSuccess;
        if (!ok)
        {
            cudaGetLastError();
        }
        return ok;
    }();
    TLLM_CHECK_WITH_INFO(configured, "csa2_lowm_wo_a: failed to configure the kernel");
    size_t const smem = woaSmemBytes(WARPS, CLUSTER_K, RT, p.M, p.GW);
    cudaFuncAttributes fattrs = {};
    cudaFuncGetAttributes(&fattrs, kernel);
    TLLM_CHECK_WITH_INFO(static_cast<int>(smem) <= fattrs.maxDynamicSharedSizeBytes,
        "csa2_lowm_wo_a: shared memory request %d exceeds the limit %d", static_cast<int>(smem),
        fattrs.maxDynamicSharedSizeBytes);
    int const kc = p.GW / CLUSTER_K;
    size_t const w_bytes = static_cast<size_t>(16 * RT) * (kc + kWoRowPad);
    int const red_aliased = static_cast<size_t>(WARPS) * 16 * RT * 8 * sizeof(float) <= w_bytes ? 1 : 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.G * p.R / (16 * RT) * CLUSTER_K);
    cfg.blockDim = dim3(WARPS * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CLUSTER_K;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, p, red_aliased);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

} // namespace

// (warps, cluster_k, row_tiles)
#define CSA2_LOW_M_WO_CONFIGS(X) X(4, 2, 2) X(8, 4, 2) X(4, 4, 2) X(4, 2, 1) X(8, 4, 1)

bool isCsa2LowMWoASupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t groups,
    int32_t rank, int32_t groupWidth, int32_t headDim, int32_t nopeDim)
{
    bool built = false;
#define CSA2_WO_IS_BUILT(W, C, R) built = built || (warps == W && clusterK == C && rowTiles == R);
    CSA2_LOW_M_WO_CONFIGS(CSA2_WO_IS_BUILT)
#undef CSA2_WO_IS_BUILT
    if (!built || m < 1 || m > 8 || groups < 1 || rank % 128 != 0 || rank % (16 * rowTiles) != 0)
    {
        return false;
    }
    if (groupWidth % (clusterK * 128) != 0 || (groupWidth / clusterK / 32) % warps != 0)
    {
        return false;
    }
    int const rope = headDim - nopeDim;
    if (headDim <= 0 || headDim % 32 != 0 || nopeDim % 32 != 0 || rope <= 0 || rope % 32 != 0
        || groupWidth % headDim != 0)
    {
        return false;
    }
    return csa2LowMWoASmemBytes(warps, clusterK, rowTiles, m, groupWidth) + 64 <= 101376;
}

size_t csa2LowMWoASmemBytes(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t groupWidth)
{
    return woaSmemBytes(warps, clusterK, rowTiles, m, groupWidth);
}

void invokeCsa2LowMWoA(__nv_bfloat16 const* o, int32_t ldo, void const* positions, bool positions64,
    float const* cosSin, int32_t csStride, uint8_t const* w, uint8_t const* wsc, __nv_bfloat16* out, int32_t ldOut,
    int32_t m, int32_t groups, int32_t rank, int32_t groupWidth, int32_t headDim, int32_t nopeDim, int32_t warps,
    int32_t clusterK, int32_t rowTiles, bool pdl, cudaStream_t stream, uint8_t* xqDbg, uint8_t* xsDbg,
    int32_t debugFlags)
{
    TLLM_CHECK_WITH_INFO(
        isCsa2LowMWoASupported(warps, clusterK, rowTiles, m, groups, rank, groupWidth, headDim, nopeDim),
        "csa2_lowm_wo_a: unsupported configuration (warps %d, cluster_k %d, row_tiles %d) at M=%d", warps, clusterK,
        rowTiles, m);
    WoaParams p{};
    p.o = o;
    p.ldo = ldo;
    p.pos = positions;
    p.pos64 = positions64 ? 1 : 0;
    p.cs = cosSin;
    p.cs_stride = csStride;
    p.w = w;
    p.wsc = wsc;
    p.out = out;
    p.ld_out = ldOut;
    p.M = m;
    p.G = groups;
    p.R = rank;
    p.GW = groupWidth;
    p.head_dim = headDim;
    p.nope = nopeDim;
    p.half_rope = (headDim - nopeDim) / 2;
    p.xq_dbg = xqDbg;
    p.xs_dbg = xsDbg;
    p.flags = debugFlags;
    p.fp8_max = 448.0f;
#define CSA2_WO_DISPATCH(W, C, R)                                                                                      \
    if (warps == W && clusterK == C && rowTiles == R)                                                                  \
    {                                                                                                                  \
        launchWoA<W, C, R>(p, pdl, stream);                                                                            \
        return;                                                                                                        \
    }
    CSA2_LOW_M_WO_CONFIGS(CSA2_WO_DISPATCH)
#undef CSA2_WO_DISPATCH
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
