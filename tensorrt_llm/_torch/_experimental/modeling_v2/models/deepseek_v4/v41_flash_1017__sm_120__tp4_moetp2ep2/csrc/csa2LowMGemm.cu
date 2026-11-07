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

#include "deepseek_v4/v41_flash_1017__sm_120__tp4_moetp2ep2/csrc/csa2LowMGemm.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// The SM120 port (csa2_mxfp8_low_m_gemm_kernel below) only uses sm_90-level cluster / bulk-copy / st.async features
// and the sm_89 m16n8k32 e4m3 mma.sync, so any sm_120 target (120, 120a, 120f) builds it.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
#define CSA2_SM120_LOW_M_GEMM_ENABLED 1
#else
#define CSA2_SM120_LOW_M_GEMM_ENABLED 0
#endif

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// Device helpers shared with the MiniMax-M3 SM100 kernel this code was ported from (minimaxM3/mxfp8LowMGemm.cu).
namespace cg = cooperative_groups;

namespace
{

constexpr int kRowPad = 32;

// Already-quantized activation (E4M3 rows [M, K] + UE8M0 scales in the 128x4-swizzled layout of
// trtllm::mxfp8_quantize(x, True)); all-null to quantize the BF16 activation in-kernel.
struct PreqArgs
{
    uint8_t const* xq;
    uint8_t const* xsf;
    int sf_k_tiles; // ceil(K / 128)
};

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t target_rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(target_rank));
    return r;
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float pow2_from_ue8m0(uint32_t e)
{
    return __uint_as_float(e == 0 ? 0x00400000u : (e << 23));
}

__device__ __forceinline__ void mma_e4m3(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra "
        "WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// SM120 port for the CSA2 decode input projections (DeepSeek-V4.1 wq_a+wkv, wq_b, index_wq_b).
//
// Same structure and math as mxfp8_low_m_gemm_tma_kernel (AR=false, NT=1): warp 0 bulk-copies the CTA's
// 16*RT weight rows x K/CLUSTER_K columns plus their UE8M0 scale chunks into shared memory BEFORE
// griddepcontrol.wait, the CTA then quantizes its activation K slice (or stages a pre-quantized one), the warps
// run m16n8k32 e4m3 MMAs with C zeroed per 32-k block and pow2 scales accumulated in FP32, and the warp + cluster
// (DSMEM st.async) reduction runs in fixed order. Differences for SM120 (99 KB per CTA, 100 KB per SM):
//  - the cluster partial tiles hold only the M live activation columns ((CLUSTER_K-1) * 16RT * M floats instead
//    of (CLUSTER_K-1) * 16RT * 8), so several CTAs fit per SM and the whole weight streams in one wave;
//  - the activation quantization uses the output scale of quantization.cuh cvt_warp_fp16_to_mxfp8 (exp2f_rcp of
//    the UE8M0 code), so the quantized activation is byte-equal to trtllm::mxfp8_quantize(x, True);
//  - optional second output (rows >= n_split go to out2), optional pre-quantized input (PreqArgs) and an optional
//    dump of the quantized activation in mxfp8_quantize's swizzled layout (QoutArgs, written by tile 0).
// ---------------------------------------------------------------------------------------------------------------
namespace
{

struct QoutArgs
{
    uint8_t* xq;    // [M, K] e4m3 bytes, or null
    uint8_t* xsf;   // 128x4-swizzled UE8M0 scales, or null
    int sf_k_tiles; // ceil(K / 128)
    unsigned long long* ts; // debug: per-CTA %globaltimer at 5 phase points, or null
    int flags;              // debug: bit 0 skips griddepcontrol.wait (negative control of the PDL checks only)
                            // bit 1: trigger dependents at start
                            // bit 2: L2-prefetch the L2pfArgs ranges (bits 3-4 position, bit 5 evict_last)
};

// L2 prefetch of the NEXT kernel's weight + scale bytes (debug flag bit 2). Each CTA's warp 0 prefetches its
// 1/gridDim share of the last frac/256 of each range; the successor's late (high-row) CTAs then stage from L2.
struct L2pfArgs
{
    uint8_t const* p0;
    long long n0;
    uint8_t const* p1;
    long long n1;
    int frac; // out of 256
};

__device__ __forceinline__ void csa2_l2_prefetch(L2pfArgs const& pf, int lane, bool evict_last)
{
    constexpr long long kChunk = 16384;
    long long const t0 = (pf.n0 * pf.frac / 256) & ~15ll;
    long long const t1 = (pf.n1 * pf.frac / 256) & ~15ll;
    long long const s0 = pf.n0 - t0; // first byte of the prefetched tail
    long long const s1 = pf.n1 - t1;
    long long const c0 = (t0 + kChunk - 1) / kChunk;
    long long const c1 = (t1 + kChunk - 1) / kChunk;
    uint64_t pol = 0;
    if (evict_last)
    {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    }
    long long const grid = gridDim.x;
    for (long long c = static_cast<long long>(lane) * grid + blockIdx.x; c < c0 + c1; c += 32 * grid)
    {
        bool const first = c < c0;
        long long const off = first ? s0 + c * kChunk : s1 + (c - c0) * kChunk;
        long long const end = first ? pf.n0 : pf.n1;
        long long const rem = end - off;
        unsigned int const n = static_cast<unsigned int>(rem < kChunk ? rem : kChunk) & ~15u;
        uint8_t const* a = (first ? pf.p0 : pf.p1) + off;
        if (n == 0 || (reinterpret_cast<uintptr_t>(a) & 15) != 0)
        {
            continue;
        }
        if (evict_last)
        {
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n" ::"l"(a), "r"(n), "l"(pol)
                         : "memory");
        }
        else
        {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a), "r"(n) : "memory");
        }
    }
}

// Fused q_norm prologue (NORM instances): the activation is the raw [M, K] row view, normalized in-kernel exactly
// as FlashInfer rmsnorm_cute (H = K = 1280) before the MXFP8 quantization; tile 0 writes the normalized rows.
struct NormArgs
{
    __nv_bfloat16 const* norm_w; // [K] RMSNorm weight
    float eps;
    __nv_bfloat16* qr_out; // [M, K] normalized rows (row stride ld_qr), or null
    int ld_qr;
};

constexpr int kQnormK = 1280;
constexpr int kQnormVecBlocks = kQnormK / 256;

// One warp normalizes one K = 1280 row bit for bit as FlashInfer's CuTe-DSL RMSNormKernel at H = 1280
// (flashinfer/norm/kernels/rmsnorm.py: 32 threads per row, vec 8, lane t holds columns 8t + v + 256b): a sequential
// per-thread FMA sum of squares in (b, v) order, an xor butterfly over the warp, mean = sum / 1280 as a true
// division, rstd = rsqrt.approx(mean + eps), y = (x * rstd) * (w + 0), BF16 RN. The BF16-rounded y is then quantized
// exactly as trtllm::mxfp8_quantize(qr, True) (4 adjacent lanes form one 1x32 group). Quantized columns
// [k_begin, k_begin + kc) go to xq_row / xc_row relative to k_begin; qr_row (or null) receives the same columns of y.
__device__ __forceinline__ void csa2_qnorm_quant_row(__nv_bfloat16 const* __restrict__ xrow,
    uint4 const (&g)[kQnormVecBlocks], float eps, int lane, __nv_bfloat16* qr_row, uint8_t* xq_row, uint8_t* xc_row,
    int k_begin, int kc)
{
#if CSA2_SM120_LOW_M_GEMM_ENABLED
    float v[kQnormVecBlocks * 8];
#pragma unroll
    for (int b = 0; b < kQnormVecBlocks; ++b)
    {
        uint4 const raw = __ldcg(reinterpret_cast<uint4 const*>(xrow + 8 * lane + 256 * b));
        __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
        for (int t = 0; t < 4; ++t)
        {
            float2 const f = __bfloat1622float2(h[t]);
            v[b * 8 + 2 * t] = f.x;
            v[b * 8 + 2 * t + 1] = f.y;
        }
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < kQnormVecBlocks * 8; ++i)
    {
        s = __fmaf_rn(v[i], v[i], s);
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    }
    float const mean = __fdiv_rn(s, static_cast<float>(kQnormK));
    float rstd;
    asm("rsqrt.approx.f32 %0, %1;\n" : "=f"(rstd) : "f"(__fadd_rn(mean, eps)));
#pragma unroll
    for (int b = 0; b < kQnormVecBlocks; ++b)
    {
        __nv_bfloat162 const* gh = reinterpret_cast<__nv_bfloat162 const*>(&g[b]);
        float y[8];
        uint32_t yb[4];
#pragma unroll
        for (int e2 = 0; e2 < 4; ++e2)
        {
            float2 const gf = __bfloat1622float2(gh[e2]);
            __nv_bfloat16 const lo
                = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[b * 8 + 2 * e2], rstd), __fadd_rn(gf.x, 0.f)));
            __nv_bfloat16 const hi
                = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[b * 8 + 2 * e2 + 1], rstd), __fadd_rn(gf.y, 0.f)));
            y[2 * e2] = __bfloat162float(lo);
            y[2 * e2 + 1] = __bfloat162float(hi);
            yb[e2] = static_cast<uint32_t>(__bfloat16_as_ushort(lo))
                | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
        }
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            amax = fmaxf(amax, fabsf(y[e]));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        float const sf = amax * rcp_approx_ftz(448.0f);
        __nv_fp8_e8m0 code;
        code.__x = __nv_cvt_float_to_e8m0(sf, __NV_SATFINITE, cudaRoundPosInf);
        // quantization.cuh exp2f_rcp: exact reciprocal of the UE8M0 scale, no subnormal flush.
        float const out_scale
            = amax != 0.f ? (code.__x == 0 ? 1.0f : exp2f(127.0f - static_cast<float>(code.__x))) : 0.0f;
        uint32_t packed[2];
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2)
        {
            __nv_fp8x2_e4m3 const lo(make_float2(y[4 * t2] * out_scale, y[4 * t2 + 1] * out_scale));
            __nv_fp8x2_e4m3 const hi(make_float2(y[4 * t2 + 2] * out_scale, y[4 * t2 + 3] * out_scale));
            packed[t2] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
        }
        int const col = 8 * lane + 256 * b;
        if (col >= k_begin && col < k_begin + kc)
        {
            if (qr_row != nullptr)
            {
                *reinterpret_cast<uint4*>(qr_row + col) = make_uint4(yb[0], yb[1], yb[2], yb[3]);
            }
            *reinterpret_cast<uint2*>(xq_row + (col - k_begin)) = make_uint2(packed[0], packed[1]);
            if ((lane & 3) == 0)
            {
                xc_row[(col - k_begin) / 32] = code.__x;
            }
        }
    }
#endif
}

__device__ __forceinline__ void csa2_stamp(QoutArgs const& q, int point)
{
    if (q.ts != nullptr && threadIdx.x == 0)
    {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        q.ts[blockIdx.x * 5 + point] = t;
    }
}

__device__ __forceinline__ int csa2_sf_offset(int m, int kb, int sf_k_tiles)
{
    return (m / 128) * sf_k_tiles * 512 + (kb / 4) * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + kb % 4;
}

// STAGES: the CTA's K slice is split into STAGES consecutive sub-slices, each bulk-copied (in order) onto its own
// mbarrier; all warps consume stage s while later stages are still arriving.
// NORM: x is the raw (un-normalized) K = 1280 row view; the CTA applies q_norm in-kernel (csa2_qnorm_quant_row).
// PF: the instance carries the L2 prefetch (flag bit 2); PF = false instances compile it out entirely, so the
// gate-off path runs the same code as before the prefetch existed.
template <int WARPS, int CLUSTER_K, int RT, int STAGES, bool NORM = false, bool PF = false>
__global__ void __launch_bounds__(WARPS * 32) csa2_mxfp8_low_m_gemm_kernel(__nv_bfloat16 const* __restrict__ x,
    int ldx, int M, int N, int K, uint8_t const* __restrict__ w, uint8_t const* __restrict__ wsf, int num_k_tiles,
    __nv_bfloat16* __restrict__ out, int ldo, __nv_bfloat16* __restrict__ out2, int ldo2, int n_split,
    int red_aliased, PreqArgs preq, QoutArgs qout, NormArgs nrm, L2pfArgs pf)
{
#if CSA2_SM120_LOW_M_GEMM_ENABLED
    constexpr int kThreads = WARPS * 32;
    constexpr int kRows = 16 * RT;
    constexpr int kTile = kRows * 8;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bar[STAGES];
    __shared__ __align__(8) uint64_t red_bar;

    int rank = 0;
    if constexpr (CLUSTER_K > 1)
    {
        rank = static_cast<int>(cg::this_cluster().block_rank());
    }
    int const tile = blockIdx.x / CLUSTER_K;
    int const row0 = tile * kRows;
    int const kc = K / CLUSTER_K;
    int const kc_blocks = kc / 32;
    int const k_begin = rank * kc;
    int const kb_begin = k_begin / 32;
    int const row_stride = kc + kRowPad;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const live = kRows * M; // live outputs of this CTA
    csa2_stamp(qout, 0);

    uint8_t* w_s = smem;                               // [kRows][row_stride]
    uint8_t* sf_s = w_s + kRows * row_stride;          // [RT][kc_blocks / 4][256]
    uint8_t* xq_s = sf_s + RT * (kc_blocks / 4) * 256; // [M][kc]
    uint8_t* xc_s = xq_s + M * kc;                     // [M][kc_blocks]
    float* cluster_red = reinterpret_cast<float*>(xc_s + ((M * kc_blocks + 15) / 16) * 16); // [CK-1][kRows*M]
    float* red = red_aliased ? reinterpret_cast<float*>(w_s) : cluster_red + (CLUSTER_K - 1) * live;

    int const kcs = kc / STAGES;         // columns per stage
    int const kcbs = kc_blocks / STAGES; // 32-blocks per stage
    int const chunks_s = kcbs / 4;       // scale chunks per stage and row tile
    uint32_t const stage_bytes = kRows * kcs + RT * chunks_s * 256u;
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bar[s])));
        }
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&red_bar)));
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&bar[s])),
                         "r"(stage_bytes)
                         : "memory");
        }
        if (CLUSTER_K > 1 && rank == 0)
        {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&red_bar)),
                         "r"(static_cast<uint32_t>((CLUSTER_K - 1) * live * sizeof(float)))
                         : "memory");
        }
    }
    __syncthreads();
    if constexpr (CLUSTER_K > 1)
    {
        asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    }
    // Weight rows and scale chunks do not depend on the producer kernel: stage them before the PDL wait.
    if (warp == 0)
    {
        int const chunks = kc_blocks / 4;
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            for (int r = lane; r < kRows; r += 32)
            {
                bulk_g2s(w_s + r * row_stride + s * kcs, w + static_cast<int64_t>(row0 + r) * K + k_begin + s * kcs,
                    kcs, &bar[s]);
            }
            for (int t = lane; t < RT * chunks_s; t += 32)
            {
                int const rt = t / chunks_s;
                int const c = s * chunks_s + t % chunks_s;
                int const rr = row0 + 16 * rt;
                int64_t const sf_base = static_cast<int64_t>(rr / 128) * num_k_tiles * 512 + (rr % 32) * 16;
                bulk_g2s(sf_s + (rt * chunks + c) * 256, wsf + sf_base + static_cast<int64_t>(kb_begin / 4 + c) * 512,
                    256, &bar[s]);
            }
        }
    }
    bool const pf_on = PF && (qout.flags & 4) != 0;
    int const pf_pos = (qout.flags >> 3) & 3;
    bool const pf_last = (qout.flags & 32) != 0;
    if (pf_on && pf_pos == 0 && warp == 0)
    {
        csa2_l2_prefetch(pf, lane, pf_last);
    }
    if (qout.flags & 2)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    [[maybe_unused]] uint4 norm_w[kQnormVecBlocks];
    if constexpr (NORM)
    {
        // The RMSNorm weight is a constant: load it before the PDL wait.
#pragma unroll
        for (int b = 0; b < kQnormVecBlocks; ++b)
        {
            norm_w[b] = __ldg(reinterpret_cast<uint4 const*>(nrm.norm_w + 8 * lane + 256 * b));
        }
    }
    if ((qout.flags & 1) == 0)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (pf_on && pf_pos == 2 && warp == 0)
    {
        csa2_l2_prefetch(pf, lane, pf_last);
    }
    if constexpr (NORM)
    {
        for (int m = warp; m < M; m += WARPS)
        {
            __nv_bfloat16* qr_row
                = (tile == 0 && nrm.qr_out != nullptr) ? nrm.qr_out + static_cast<int64_t>(m) * nrm.ld_qr : nullptr;
            csa2_qnorm_quant_row(x + static_cast<int64_t>(m) * ldx, norm_w, nrm.eps, lane, qr_row, xq_s + m * kc,
                xc_s + m * kc_blocks, k_begin, kc);
        }
    }
    else if (preq.xq != nullptr)
    {
        int const vec_per_row = kc / 16;
        int const vecs = M * vec_per_row;
#pragma unroll 4
        for (int p = threadIdx.x; p < vecs; p += kThreads)
        {
            int const m = p / vec_per_row;
            int const v = p - m * vec_per_row;
            reinterpret_cast<uint4*>(xq_s + m * kc)[v]
                = __ldcg(reinterpret_cast<uint4 const*>(preq.xq + static_cast<int64_t>(m) * K + k_begin) + v);
        }
        for (int p = threadIdx.x; p < M * kc_blocks; p += kThreads)
        {
            int const m = p / kc_blocks;
            int const b = p - m * kc_blocks;
            xc_s[m * kc_blocks + b] = __ldcg(preq.xsf + csa2_sf_offset(m, kb_begin + b, preq.sf_k_tiles));
        }
    }
    else
    {
        int const items = M * kc_blocks;
        for (int p0 = threadIdx.x; p0 < items; p0 += 2 * kThreads)
        {
            uint4 raw[2][4];
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                int const p = p0 + u * kThreads;
                if (p < items)
                {
                    int const m = p / kc_blocks;
                    int const b = p % kc_blocks;
                    uint4 const* src
                        = reinterpret_cast<uint4 const*>(x + static_cast<int64_t>(m) * ldx + k_begin + b * 32);
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        raw[u][q] = src[q];
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                int const p = p0 + u * kThreads;
                if (p >= items)
                {
                    continue;
                }
                int const m = p / kc_blocks;
                int const b = p % kc_blocks;
                float v[32];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw[u][q]);
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        float2 const f = __bfloat1622float2(h[t]);
                        v[q * 8 + t * 2] = f.x;
                        v[q * 8 + t * 2 + 1] = f.y;
                    }
                }
                float amax = 0.f;
#pragma unroll
                for (int t = 0; t < 32; ++t)
                {
                    amax = fmaxf(amax, fabsf(v[t]));
                }
                float const sf = amax * rcp_approx_ftz(448.0f);
                __nv_fp8_e8m0 code;
                code.__x = __nv_cvt_float_to_e8m0(sf, __NV_SATFINITE, cudaRoundPosInf);
                // quantization.cuh exp2f_rcp: exact reciprocal of the UE8M0 scale, no subnormal flush.
                float const out_scale = amax != 0.f
                    ? (code.__x == 0 ? 1.0f : exp2f(127.0f - static_cast<float>(code.__x)))
                    : 0.0f;
                uint32_t packed[8];
#pragma unroll
                for (int t = 0; t < 8; ++t)
                {
                    __nv_fp8x2_e4m3 const lo(make_float2(v[4 * t] * out_scale, v[4 * t + 1] * out_scale));
                    __nv_fp8x2_e4m3 const hi(make_float2(v[4 * t + 2] * out_scale, v[4 * t + 3] * out_scale));
                    packed[t] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
                }
                uint4* dst = reinterpret_cast<uint4*>(xq_s + m * kc + b * 32);
                dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
                dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
                xc_s[m * kc_blocks + b] = code.__x;
            }
        }
    }
    __syncthreads();
    if (qout.xq != nullptr && tile == 0)
    {
        // Publish this K slice of the quantized activation (debug / checks; mxfp8_quantize layout).
        for (int p = threadIdx.x; p < M * kc_blocks; p += kThreads)
        {
            int const m = p / kc_blocks;
            int const b = p % kc_blocks;
            uint4 const* src = reinterpret_cast<uint4 const*>(xq_s + m * kc + b * 32);
            uint4* dst = reinterpret_cast<uint4*>(qout.xq + static_cast<int64_t>(m) * K + k_begin + b * 32);
            dst[0] = src[0];
            dst[1] = src[1];
            qout.xsf[csa2_sf_offset(m, kb_begin + b, qout.sf_k_tiles)] = xc_s[m * kc_blocks + b];
        }
    }
    csa2_stamp(qout, 1);

    int const g = lane >> 2;
    int const j = lane & 3;
    int const col0 = 2 * j;
    int const bps = kcbs / WARPS; // blocks per warp and stage
    int const chunks = kc_blocks / 4;
    float acc[RT][4] = {};
    for (int s = 0; s < STAGES; ++s)
    {
    mbar_wait(&bar[s], 0);
    if (s == 0)
    {
        csa2_stamp(qout, 2);
    }
    if (s == STAGES - 1 && pf_on && pf_pos == 1 && warp == 0)
    {
        csa2_l2_prefetch(pf, lane, pf_last);
    }
#pragma unroll 2
    for (int i = 0; i < bps; ++i)
    {
        int const b = s * kcbs + warp * bps + i;
        int const kb = kb_begin + b;
        uint2 const xb = g < M ? *reinterpret_cast<uint2 const*>(xq_s + g * kc + b * 32 + 8 * j) : make_uint2(0, 0);
        float const xs0 = col0 < M ? pow2_from_ue8m0(xc_s[col0 * kc_blocks + b]) : 0.f;
        float const xs1 = col0 + 1 < M ? pow2_from_ue8m0(xc_s[(col0 + 1) * kc_blocks + b]) : 0.f;
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            uint8_t const* wa_s = w_s + (16 * rt + g) * row_stride + 8 * j + b * 32;
            uint2 const wa = *reinterpret_cast<uint2 const*>(wa_s);
            uint2 const wb = *reinterpret_cast<uint2 const*>(wa_s + 8 * row_stride);
            int const inner_m = ((row0 + 16 * rt) % 128) / 32;
            uint8_t const* sft = sf_s + (rt * chunks + b / 4) * 256 + inner_m * 4 + kb % 4;
            float const wsa = pow2_from_ue8m0(sft[g * 16]);
            float const wsb = pow2_from_ue8m0(sft[(g + 8) * 16]);
            float d[4];
            mma_e4m3(d, wa.x, wb.x, wa.y, wb.y, xb.x, xb.y);
            acc[rt][0] += d[0] * (wsa * xs0);
            acc[rt][1] += d[1] * (wsa * xs1);
            acc[rt][2] += d[2] * (wsb * xs0);
            acc[rt][3] += d[3] * (wsb * xs1);
        }
    }
    }
    csa2_stamp(qout, 3);
    asm volatile("griddepcontrol.launch_dependents;");
    if (red_aliased)
    {
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
    {
        int const ra = 16 * rt + g;
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
                    : "r"(smem_u32(&red_bar))
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
            int const row = row0 + r;
            if (row < n_split)
            {
                out[static_cast<int64_t>(m) * ldo + row] = __float2bfloat16_rn(total);
            }
            else
            {
                out2[static_cast<int64_t>(m) * ldo2 + (row - n_split)] = __float2bfloat16_rn(total);
            }
        }
    }
    else
    {
        uint32_t const remote_bar = mapa_shared(smem_u32(&red_bar), 0);
        uint32_t const remote_dst = mapa_shared(smem_u32(cluster_red + (rank - 1) * live), 0);
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
    csa2_stamp(qout, 4);
#endif
}

size_t csa2SmemBytes(int warps, int clusterK, int rowTiles, int m, int k)
{
    int const kc = k / clusterK;
    int const kc_blocks = kc / 32;
    size_t const rows = 16 * rowTiles;
    size_t const w_bytes = rows * (kc + kRowPad);
    size_t const red_bytes = static_cast<size_t>(warps) * rows * 8 * sizeof(float);
    bool const aliased = red_bytes <= w_bytes;
    return w_bytes + rowTiles * (kc_blocks / 4) * 256 + static_cast<size_t>(m) * kc
        + ((m * kc_blocks + 15) / 16) * 16 + static_cast<size_t>(clusterK - 1) * rows * m * sizeof(float)
        + (aliased ? 0 : red_bytes);
}

template <int WARPS, int CLUSTER_K, int RT, int STAGES, bool NORM = false, bool PF = false>
void launchCsa2(__nv_bfloat16 const* x, int ldx, int M, int N, int K, uint8_t const* w, uint8_t const* wsf,
    __nv_bfloat16* out, int ldo, __nv_bfloat16* out2, int ldo2, int nSplit, PreqArgs preq, QoutArgs qout, bool pdl,
    cudaStream_t stream, NormArgs nrm = NormArgs{}, L2pfArgs pf = L2pfArgs{})
{
    int const kc = K / CLUSTER_K;
    int const kc_blocks = kc / 32;
    TLLM_CHECK_WITH_INFO(N % (16 * RT) == 0 && K % (CLUSTER_K * 128) == 0 && kc_blocks % (WARPS * STAGES) == 0
            && (kc_blocks / STAGES) % 4 == 0 && nSplit % (16 * RT) == 0 && (!NORM || K == kQnormK),
        "csa2_mxfp8_low_m_gemm: unsupported shape N=%d K=%d n_split=%d for (%d, %d, %d, %d)", N, K, nSplit, WARPS,
        CLUSTER_K, RT, STAGES);
    size_t const w_bytes = 16 * RT * (kc + kRowPad);
    size_t const red_bytes = static_cast<size_t>(WARPS) * 16 * RT * 8 * sizeof(float);
    int const red_aliased = red_bytes <= w_bytes ? 1 : 0;
    size_t const smem = csa2SmemBytes(WARPS, CLUSTER_K, RT, M, K);
    auto kernel = csa2_mxfp8_low_m_gemm_kernel<WARPS, CLUSTER_K, RT, STAGES, NORM, PF>;
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
        // Prefer the full shared-memory carveout so several CTAs share an SM (one-wave weight streaming).
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
    TLLM_CHECK_WITH_INFO(configured, "csa2_mxfp8_low_m_gemm: failed to configure the kernel");
    cudaFuncAttributes fattrs = {};
    cudaFuncGetAttributes(&fattrs, kernel);
    TLLM_CHECK_WITH_INFO(static_cast<int>(smem) <= fattrs.maxDynamicSharedSizeBytes,
        "csa2_mxfp8_low_m_gemm: shared memory request %d exceeds the limit %d", static_cast<int>(smem),
        fattrs.maxDynamicSharedSizeBytes);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N / (16 * RT) * CLUSTER_K);
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
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, x, ldx, M, N, K, w, wsf, (K / 32 + 3) / 4, out, ldo,
        out2, ldo2, nSplit, red_aliased, preq, qout, nrm, pf);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

} // namespace

// (warps, cluster_k, row_tiles, stages)
#define CSA2_LOW_M_CONFIGS(X)                                                                                          \
    X(4, 8, 1, 1) X(4, 8, 2, 1) X(4, 4, 1, 1) X(4, 4, 2, 1) X(4, 2, 1, 1) X(4, 2, 2, 1) X(4, 2, 4, 1) X(4, 1, 1, 1)    \
        X(4, 1, 2, 1) X(4, 1, 4, 1) X(8, 4, 1, 1) X(8, 4, 2, 1) X(8, 2, 2, 1) X(8, 2, 4, 1) X(8, 1, 2, 1)              \
            X(8, 1, 4, 1) X(8, 2, 2, 2) X(4, 2, 2, 2) X(4, 2, 2, 4) X(4, 4, 2, 2) X(4, 4, 1, 2) X(4, 1, 2, 2)          \
                X(4, 1, 1, 2) X(4, 1, 4, 2)

bool isCsa2Mxfp8LowMGemmSupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t stages)
{
    bool built = false;
#define CSA2_IS_BUILT(W, C, R, S) built = built || (warps == W && clusterK == C && rowTiles == R && stages == S);
    CSA2_LOW_M_CONFIGS(CSA2_IS_BUILT)
#undef CSA2_IS_BUILT
    return built && m >= 1 && m <= 8;
}

size_t csa2Mxfp8LowMGemmSmemBytes(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t k)
{
    return csa2SmemBytes(warps, clusterK, rowTiles, m, k);
}

void invokeCsa2Mxfp8LowMGemm(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t n, int32_t k, uint8_t const* w,
    uint8_t const* wsf, __nv_bfloat16* out, int32_t ldo, __nv_bfloat16* out2, int32_t ldo2, int32_t nSplit,
    uint8_t const* xq, uint8_t const* xsf, uint8_t* xqOut, uint8_t* xsfOut, int32_t warps, int32_t clusterK,
    int32_t rowTiles, bool pdl, cudaStream_t stream, unsigned long long* dbgTs, int32_t stages, int32_t debugFlags,
    uint8_t const* pfP0, int64_t pfN0, uint8_t const* pfP1, int64_t pfN1, int32_t pfFrac)
{
    TLLM_CHECK_WITH_INFO(isCsa2Mxfp8LowMGemmSupported(warps, clusterK, rowTiles, m, stages),
        "csa2_mxfp8_low_m_gemm: unsupported configuration (warps %d, cluster_k %d, row_tiles %d, stages %d) at M=%d",
        warps, clusterK, rowTiles, stages, m);
    int const sfKTiles = (k / 32 + 3) / 4;
    PreqArgs const preq{xq, xsf, sfKTiles};
    if (pfP0 == nullptr && pfP1 == nullptr)
    {
        debugFlags &= ~4;
    }
    L2pfArgs const pf{pfP0, pfP0 == nullptr ? 0 : pfN0, pfP1, pfP1 == nullptr ? 0 : pfN1,
        pfFrac < 0 ? 0 : (pfFrac > 256 ? 256 : pfFrac)};
    if (debugFlags & 4)
    {
        // Only the attention-input chain's wq_a||wkv configurations carry a prefetch instance (PF = true).
        QoutArgs const qoutPf{xqOut, xsfOut, sfKTiles, dbgTs, debugFlags};
#define CSA2_PF_DISPATCH(W, C, R, S)                                                                                   \
    if (warps == W && clusterK == C && rowTiles == R && stages == S)                                                   \
    {                                                                                                                  \
        launchCsa2<W, C, R, S, false, true>(                                                                           \
            x, ldx, m, n, k, w, wsf, out, ldo, out2, ldo2, nSplit, preq, qoutPf, pdl, stream, NormArgs{}, pf);         \
        return;                                                                                                        \
    }
        CSA2_PF_DISPATCH(4, 4, 2, 1)
        CSA2_PF_DISPATCH(8, 2, 2, 1)
        CSA2_PF_DISPATCH(8, 4, 1, 1)
#undef CSA2_PF_DISPATCH
        debugFlags &= ~4; // no prefetch instance for this configuration: the prefetch is a hint, drop it
    }
    QoutArgs const qout{xqOut, xsfOut, sfKTiles, dbgTs, debugFlags};
#define CSA2_DISPATCH(W, C, R, S)                                                                                      \
    if (warps == W && clusterK == C && rowTiles == R && stages == S)                                                   \
    {                                                                                                                  \
        launchCsa2<W, C, R, S>(                                                                                        \
            x, ldx, m, n, k, w, wsf, out, ldo, out2, ldo2, nSplit, preq, qout, pdl, stream, NormArgs{}, pf);           \
        return;                                                                                                        \
    }
    CSA2_LOW_M_CONFIGS(CSA2_DISPATCH)
#undef CSA2_DISPATCH
}

// (warps, cluster_k, row_tiles) of the fused q_norm instances (K = 1280, one stage).
#define CSA2_QNORM_CONFIGS(X) X(4, 1, 1) X(4, 1, 2) X(4, 1, 4) X(8, 1, 2) X(8, 1, 4)

bool isCsa2QnormLowMGemmSupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t k)
{
    bool built = false;
#define CSA2_QNORM_IS_BUILT(W, C, R) built = built || (warps == W && clusterK == C && rowTiles == R);
    CSA2_QNORM_CONFIGS(CSA2_QNORM_IS_BUILT)
#undef CSA2_QNORM_IS_BUILT
    return built && m >= 1 && m <= 8 && k == kQnormK;
}

void invokeCsa2QnormLowMGemm(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t n, int32_t k, uint8_t const* w,
    uint8_t const* wsf, __nv_bfloat16 const* normW, float eps, __nv_bfloat16* qrOut, int32_t ldqr, __nv_bfloat16* out,
    int32_t ldo, __nv_bfloat16* out2, int32_t ldo2, int32_t nSplit, int32_t warps, int32_t clusterK, int32_t rowTiles,
    bool pdl, cudaStream_t stream, int32_t debugFlags, uint8_t* xqOut, uint8_t* xsfOut)
{
    TLLM_CHECK_WITH_INFO(isCsa2QnormLowMGemmSupported(warps, clusterK, rowTiles, m, k),
        "csa2_qnorm_low_m_gemm: unsupported configuration (warps %d, cluster_k %d, row_tiles %d) at M=%d K=%d", warps,
        clusterK, rowTiles, m, k);
    int const sfKTiles = (k / 32 + 3) / 4;
    PreqArgs const preq{nullptr, nullptr, sfKTiles};
    QoutArgs const qout{xqOut, xsfOut, sfKTiles, nullptr, debugFlags};
    NormArgs const nrm{normW, eps, qrOut, ldqr};
#define CSA2_QNORM_DISPATCH(W, C, R)                                                                                   \
    if (warps == W && clusterK == C && rowTiles == R)                                                                  \
    {                                                                                                                  \
        launchCsa2<W, C, R, 1, true>(                                                                                  \
            x, ldx, m, n, k, w, wsf, out, ldo, out2, ldo2, nSplit, preq, qout, pdl, stream, nrm);                      \
        return;                                                                                                        \
    }
    CSA2_QNORM_CONFIGS(CSA2_QNORM_DISPATCH)
#undef CSA2_QNORM_DISPATCH
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
