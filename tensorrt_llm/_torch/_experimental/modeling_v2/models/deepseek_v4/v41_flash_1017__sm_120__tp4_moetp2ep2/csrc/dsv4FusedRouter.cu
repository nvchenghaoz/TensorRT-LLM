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

// DeepSeek-V4.1-Flash decode router (1..8 tokens) in one launch: router GEMV + top-6 sqrt-softplus gate.
//
// GEMV. 48 CTAs x 640 threads, 8 experts per CTA: exactly router_gemm_mma_kernel<M,384,5120,8,640>
// (dsv3RouterGemm.cu routerGemmMma): warp w owns k = 256 w .. 256 w + 255 and walks its 8 32-wide chunks in order,
// two mma.sync m16n8k16 (afrag0 then afrag1) per chunk into fp32, the 20 warp partials go through shared memory and
// one thread per output sums warps 0..19 in order from 0.0f. So every logit is bitwise the router kernel's.
// The router weight (and the gate bias) is read-only for the whole forward, so all of it is loaded before
// griddepcontrol.wait; the activations (written by the PDL predecessor) are read only after it.
//
// Gate. The last CTA to arrive (atomic counter after a __threadfence of the logit stores; no co-residency
// assumption, no polling) reads the [M,384] logits back through L2 and runs gate_forward_kernel<384,6,false,false>'s
// body (same expressions and selection order: softplus with threshold 20, sqrtf, + bias, packed-key top-6, the
// unbiased scores gathered, weight = v / (sum + 1e-20f) * routeScale), one warp per token. It then resets the
// counter to 0 for the next launch (launches sharing a counter run in stream order: the next one's atomics come
// after its own griddepcontrol.wait). Every CTA triggers its dependents after its last global store.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "deepseek_v4/v41_flash_1017__sm_120__tp4_moetp2ep2/csrc/dsv4FusedRouter.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <type_traits>

namespace cg = cooperative_groups;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::dsv4_fused_router
{

namespace reduce_topk = ::tensorrt_llm::kernels::reduce_topk;
namespace
{

constexpr int kExpertsPerCta = 8;
constexpr int kBlockSize = 640;
constexpr int kNumWarps = kBlockSize / 32;                           // 20
constexpr int kChunk = 32;                                           // k per lane load group = two m16n8k16 MMAs
constexpr int kChunksPerWarp = kHiddenDim / (kNumWarps * kChunk);    // 8
constexpr int kNumCtas = kNumExperts / kExpertsPerCta;               // 48
constexpr int kExpertsPerLane = kNumExperts / 32;                    // 12
static_assert(kChunksPerWarp == 8, "8 chunks per warp, all in flight (router_gemm_mma_kernel's kBatch)");

// Same instruction as dsv3RouterGemm.cu mma_bf16_m16n8k16.
__device__ __forceinline__ void mmaBf16M16n8k16(float (&c)[4], uint32_t const (&a)[4], uint32_t const (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Read-only loads issued before the dependency wait: asm volatile keeps them ahead of the (asm volatile) wait.
__device__ __forceinline__ uint4 loadReadOnly16(void const* ptr)
{
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ void gridDependencyWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void triggerDependents()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint32_t warpMaxU32(uint32_t value)
{
    uint32_t result;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n" : "=r"(result) : "r"(value));
    return result;
}

// The fp32 division of the Triton q0 kernel (amax / 448.0 and 1.0 / sf): its PTX uses div.full.f32 for both
// (experiments/ascent/fused_router/ptx_probe.py).
__device__ __forceinline__ float kQuantDiv(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// fp32 -> e4m3 with round-to-nearest-even and saturation (cvt.rn.satfinite.e4m3x2.f32), 4 values -> 4 bytes.
__device__ __forceinline__ uint32_t packE4m3x4(float v0, float v1, float v2, float v3)
{
    uint16_t lo;
    uint16_t hi;
    // cvt.rn.satfinite.e4m3x2.f32 d, a, b: a -> upper byte, b -> lower byte.
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(v1), "f"(v0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(v3), "f"(v2));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Stage 2 constants: FlashInfer moe_mxfp8_q0_route_triton (TILE_K 128 per ue8m0 exponent, 4 exponents per int32).
constexpr int kQTileK = 128;
constexpr int kQTiles = kHiddenDim / kQTileK;   // 40 exponent tiles per token
constexpr int kSfPack = 4;                      // exponents per int32 scale word
constexpr int kSfBlocks = kQTiles / kSfPack;     // 10 scale words per routed row
constexpr int kMaxPairs = kMaxTokens * kTopK;   // 48
static_assert(kNumWarps * 2 == kQTiles && kChunksPerWarp * kChunk == 2 * kQTileK,
    "each warp's GEMV k-range is exactly two q0 tiles");

// K1 fold (kFoldK1): dynamic shared memory layout. Raw (pre-norm) bf16 layer_input of every token, the norm weight,
// the per-warp sums of squares and the rstd of every token.
constexpr int kFoldChunks = kHiddenDim / 8; // 640 8-element chunks per token: one per thread
static_assert(kFoldChunks == kBlockSize, "pass A maps chunk c of every token to thread c");

template <int kNumTokens>
struct FoldSmem
{
    static constexpr int kLiBytes = kNumTokens * kHiddenDim * 2;
    static constexpr int kNwOffset = kLiBytes;
    static constexpr int kWarpSumOffset = kNwOffset + kHiddenDim * 2;
    static constexpr int kRstdOffset = kWarpSumOffset + kNumTokens * 32 * 4;
    static constexpr int kBytes = kRstdOffset + 16 * 4;
};

__device__ __forceinline__ float foldRcpApproxFtz(float x)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float foldRsqrtApprox(float x)
{
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float foldBf16Round(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

template <int kNumTokens, bool kRoute, bool kFoldK1>
__global__ __launch_bounds__(kBlockSize, 1) void dsv4FusedRouterKernel(
    std::conditional_t<kFoldK1, FusedRouterFoldParams, FusedRouterParams> const __grid_constant__ p)
{
    static_assert(kNumTokens >= 1 && kNumTokens <= kMaxTokens, "1..8 tokens");
    static_assert(!kFoldK1 || kNumTokens <= kMaxFoldTokens, "K1 fold: 1..4 tokens (shared memory)");
    constexpr int kPairs = kNumTokens * kTopK;
    int const warpId = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const g = lane / 4; // token row (A) / expert column (B) of this lane in the MMA fragments
    int const q = lane % 4; // 8-wide k sub-chunk of this lane
    int const nBase = blockIdx.x * kExpertsPerCta;
    int const kWarp = warpId * (kChunksPerWarp * kChunk);

    __shared__ float smRed[kNumWarps][32][4];
    __shared__ float smScores[kNumTokens][kNumExperts];
    __shared__ __align__(16) float smBias[kNumExperts];
    __shared__ int smIsLast;
    // Stage 2: the routing of every pair and its rows, and the exponents of every 128-wide tile.
    __shared__ int smIds[kRoute ? kPairs : 1];
    __shared__ float smWts[kRoute ? kPairs : 1];
    __shared__ int smDst[kRoute ? kPairs : 1];
    __shared__ int smScaleDst[kRoute ? kPairs : 1];
    __shared__ uint32_t smExp[kRoute ? kNumTokens : 1][kQTiles];

    // ---- Before the wait: the router weight slice of this lane (all 8 chunks) and, for the routing warps, the
    // gate bias. Neither is written by any kernel of the forward.
    uint4 b[kChunksPerWarp];
    __nv_bfloat16 const* wRow = p.weight + static_cast<int64_t>(nBase + g) * kHiddenDim + kWarp + q * 8;
#pragma unroll
    for (int c = 0; c < kChunksPerWarp; c++)
    {
        b[c] = loadReadOnly16(wRow + c * kChunk);
    }
    // The bias goes to shared memory asynchronously (no registers held, nothing waits on it until the gate).
    if (threadIdx.x < kNumExperts / 4)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\ncp.async.commit_group;\n" ::"r"(
                         static_cast<uint32_t>(__cvta_generic_to_shared(&smBias[threadIdx.x * 4]))),
                     "l"(p.bias + threadIdx.x * 4)
                     : "memory");
    }
    [[maybe_unused]] uint8_t* foldSmem = nullptr;
    if constexpr (kFoldK1)
    {
        // The norm weight (read-only for the forward) goes to shared memory like the bias: 640 x 16 B.
        extern __shared__ __align__(16) uint8_t dsv4FoldDynSmem[];
        foldSmem = dsv4FoldDynSmem;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\ncp.async.commit_group;\n" ::"r"(
                         static_cast<uint32_t>(__cvta_generic_to_shared(
                             foldSmem + FoldSmem<kNumTokens>::kNwOffset + threadIdx.x * 16))),
                     "l"(p.k1NormWeight + threadIdx.x * 8)
                     : "memory");
    }

    if (!p.testSkipWait)
    {
        gridDependencyWait();
    }

    if constexpr (kFoldK1)
    {
        // ---- K1 fold, pass A (every thread, chunk = threadIdx.x of every token): post update + lagged collapse
        // -> raw bf16 li in shared memory. _k1_chunk's expressions: nr = post[j] * x, nr = fma(comb[k][j], r[k], nr)
        // for k = 0..3, nr = bf16(nr), li = fma(pre[j], nr, li) from 0 for j = 0..3, li = bf16(li).
        __nv_bfloat16* smLi = reinterpret_cast<__nv_bfloat16*>(foldSmem);
        int const chunk = threadIdx.x;
#pragma unroll
        for (int m = 0; m < kNumTokens; ++m)
        {
            int64_t const off = static_cast<int64_t>(m) * kHiddenDim + chunk * 8;
            uint4 const xv = *reinterpret_cast<uint4 const*>(p.k1X + off);
            __nv_bfloat16 const* rBase = p.k1Residual + static_cast<int64_t>(m) * 4 * kHiddenDim + chunk * 8;
            uint4 rv[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                rv[k] = *reinterpret_cast<uint4 const*>(rBase + k * kHiddenDim);
            }
            float post[4];
            float pre[4];
            float comb[16];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                post[j] = p.k1Post[m * 4 + j];
                pre[j] = p.k1Pre[m * 4 + j];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
            {
                comb[j] = p.k1Comb[m * 16 + j];
            }
            __nv_bfloat16 const* xe = reinterpret_cast<__nv_bfloat16 const*>(&xv);
            uint4 outv;
            __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&outv);
#pragma unroll
            for (int e = 0; e < 8; ++e)
            {
                float const xf = __bfloat162float(xe[e]);
                float rf[4];
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    rf[k] = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&rv[k])[e]);
                }
                float li = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float nr = __fmul_rn(post[j], xf);
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                    {
                        nr = __fmaf_rn(comb[k * 4 + j], rf[k], nr);
                    }
                    nr = foldBf16Round(nr);
                    li = __fmaf_rn(pre[j], nr, li);
                }
                oe[e] = __float2bfloat16_rn(li);
            }
            *reinterpret_cast<uint4*>(smLi + m * kHiddenDim + chunk * 8) = outv;
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        // ---- pass B: the sum of squares in K1's order for its P threads. Thread t of a P-thread group sums chunks
        // t, t + P, ... of its token, each pair v = 0..3 as s += fma(x, x, y * y); lane 0 of every warp then holds
        // the xor butterfly (16, 8, 4, 2, 1). P % 32 == 0, so every group is whole warps.
        int const P = p.k1ReducerThreads;
        int const groups = kBlockSize / P;
        int const grp = threadIdx.x / P;
        int const t = threadIdx.x - grp * P;
        float* smWarpSum = reinterpret_cast<float*>(foldSmem + FoldSmem<kNumTokens>::kWarpSumOffset);
        float* smRstd = reinterpret_cast<float*>(foldSmem + FoldSmem<kNumTokens>::kRstdOffset);
        if (grp < groups)
        {
            for (int m = grp; m < kNumTokens; m += groups)
            {
                float s = 0.0f;
                for (int c = t; c < kFoldChunks; c += P)
                {
                    uint4 const v = *reinterpret_cast<uint4 const*>(smLi + m * kHiddenDim + c * 8);
                    __nv_bfloat162 const* hv = reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        float const lx = __low2float(hv[j]);
                        float const ly = __high2float(hv[j]);
                        s = __fadd_rn(s, __fmaf_rn(lx, lx, __fmul_rn(ly, ly)));
                    }
                }
#pragma unroll
                for (int off = 16; off >= 1; off /= 2)
                {
                    s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
                }
                if (lane == 0)
                {
                    smWarpSum[m * 32 + t / 32] = s;
                }
            }
        }
        __syncthreads();
        // ---- pass C: total = sequential sum over warps from warp 0; rstd = rsqrt(fma(total, rcp(hidden), eps)).
        if (threadIdx.x < kNumTokens)
        {
            int const m = threadIdx.x;
            float total = smWarpSum[m * 32];
            for (int w = 1; w < P / 32; ++w)
            {
                total = __fadd_rn(total, smWarpSum[m * 32 + w]);
            }
            float const rcp = foldRcpApproxFtz(static_cast<float>(kHiddenDim));
            smRstd[m] = foldRsqrtApprox(__fmaf_rn(total, rcp, p.k1NormEps));
        }
        __syncthreads();
    }

    // ---- GEMV: router_gemm_mma_kernel's loads, MMA order and reduction.
    bool const row0Valid = g < kNumTokens;
    uint4 const zero4 = make_uint4(0u, 0u, 0u, 0u);
    __nv_bfloat16 const* aRow0 = p.x + g * kHiddenDim;
    uint4 a0[kChunksPerWarp];
    if constexpr (kFoldK1)
    {
        // K1 fold: the operand is layer_input = bf16((li * rstd) * w) (_k1_norm), from shared memory. This CTA also
        // stores its share (8-element units u = k / 8 with u % kNumCtas == blockIdx.x) of layer_input to out.
        __nv_bfloat16 const* smLi = reinterpret_cast<__nv_bfloat16 const*>(foldSmem);
        __nv_bfloat16 const* smNw
            = reinterpret_cast<__nv_bfloat16 const*>(foldSmem + FoldSmem<kNumTokens>::kNwOffset);
        float const rstd
            = row0Valid ? reinterpret_cast<float const*>(foldSmem + FoldSmem<kNumTokens>::kRstdOffset)[g] : 0.0f;
#pragma unroll
        for (int c = 0; c < kChunksPerWarp; c++)
        {
            int const k = kWarp + c * kChunk + q * 8;
            if (row0Valid)
            {
                uint4 const lv = *reinterpret_cast<uint4 const*>(smLi + g * kHiddenDim + k);
                uint4 const wv = *reinterpret_cast<uint4 const*>(smNw + k);
                __nv_bfloat16 const* le = reinterpret_cast<__nv_bfloat16 const*>(&lv);
                __nv_bfloat16 const* we = reinterpret_cast<__nv_bfloat16 const*>(&wv);
                uint4 ov;
                __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                {
                    oe[e] = __float2bfloat16_rn(
                        __fmul_rn(__fmul_rn(__bfloat162float(le[e]), rstd), __bfloat162float(we[e])));
                }
                a0[c] = ov;
                if ((k / 8) % kNumCtas == static_cast<int>(blockIdx.x))
                {
                    *reinterpret_cast<uint4*>(p.k1Out + g * kHiddenDim + k) = ov;
                }
            }
            else
            {
                a0[c] = zero4;
            }
        }
    }
    else
    {
#pragma unroll
        for (int c = 0; c < kChunksPerWarp; c++)
        {
            int const k = kWarp + c * kChunk + q * 8;
            a0[c] = row0Valid ? *reinterpret_cast<uint4 const*>(aRow0 + k) : zero4;
        }
    }
    if constexpr (kRoute)
    {
        // route_assign_decode's fills, grid-stride over every CTA, after the wait (the previous layer's FC1/FC2 read
        // these buffers): the caller's q1/sf1/out buffer and every scale_out word. The last CTA writes the routed
        // scale words only after every CTA has arrived (fence + counter below).
        int const gtid = blockIdx.x * kBlockSize + threadIdx.x;
        constexpr int kGridThreads = kNumCtas * kBlockSize;
        for (int64_t i = gtid; i < p.zeroNumel; i += kGridThreads)
        {
            p.zeroBuf[i] = 0;
        }
        int const totalScale = kSfBlocks * p.paddedRows;
        for (int i = gtid; i < totalScale; i += kGridThreads)
        {
            p.scaleOut[i] = 0;
        }
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < kChunksPerWarp; c++)
    {
        // Rows g + 8 of the m16 tile are tokens >= 8: always zero (router_gemm_mma_kernel's a1 for M <= 8).
        uint32_t const afrag0[4] = {a0[c].x, 0u, a0[c].y, 0u};
        uint32_t const afrag1[4] = {a0[c].z, 0u, a0[c].w, 0u};
        uint32_t const bfrag0[2] = {b[c].x, b[c].y};
        uint32_t const bfrag1[2] = {b[c].z, b[c].w};
        mmaBf16M16n8k16(acc, afrag0, bfrag0);
        mmaBf16M16n8k16(acc, afrag1, bfrag1);
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        smRed[warpId][lane][i] = acc[i];
    }
    __syncthreads();
    if (threadIdx.x < 128)
    {
        int const l = threadIdx.x / 4;
        int const i = threadIdx.x % 4;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kNumWarps; w++)
        {
            sum += smRed[w][l][i];
        }
        int const row = (l / 4) + (i >= 2 ? 8 : 0);
        int const col = nBase + (l % 4) * 2 + (i & 1);
        if (row < kNumTokens)
        {
            p.logits[row * kNumExperts + col] = sum;
        }
        if constexpr (!kRoute)
        {
            __threadfence(); // the logit stores are visible device-wide before this CTA counts itself in
        }
    }
    if constexpr (kRoute)
    {
        __threadfence(); // logits and fills
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int const prev = atomicAdd(p.counter, 1);
        smIsLast = prev == kNumCtas - 1;
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    if (!smIsLast)
    {
        triggerDependents();
        return;
    }

    // ---- Last CTA: every CTA's logits are stored. gate_forward_kernel<384,6,false,false>, one warp per token.
    __threadfence();
    if (warpId < kNumTokens)
    {
        using RedType = reduce_topk::TopKRedType<float>;
        constexpr float kSoftplusThreshold = 20.0F;
        constexpr float kWeightEpsilon = 1e-20F;
        constexpr float minValue = -1e30f;
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        int const row = warpId;
        float* myScores = smScores[row];
        float const* logitsRow = p.logits + row * kNumExperts;

        float raw[kExpertsPerLane];
#pragma unroll
        for (int e = 0; e < kExpertsPerLane; ++e)
        {
            raw[e] = __ldcg(logitsRow + lane + e * 32);
        }
        RedType keys[kExpertsPerLane];
#pragma unroll
        for (int e = 0; e < kExpertsPerLane; ++e)
        {
            int const expertId = lane + e * 32;
            float const s = raw[e];
            float const sp = s > kSoftplusThreshold ? s : log1pf(expf(s));
            float const score = sqrtf(sp);
            myScores[expertId] = score;
            keys[e] = RedType{score + smBias[expertId], expertId};
        }

        reduce_topk::Sort<kExpertsPerLane, RedType>::run(keys);

        int myTopkIndex = 0;
        uint64_t packedMax = 0;
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
        {
            if (k > 0 && packedMax == keys[0].compValIdx)
            {
#pragma unroll
                for (int i = 0; i + 1 < kExpertsPerLane; ++i)
                {
                    keys[i] = keys[i + 1];
                }
                keys[kExpertsPerLane - 1] = RedType{minValue, lane + (kExpertsPerLane - 1) * 32};
            }
            uint32_t const scoreBits = static_cast<uint32_t>(keys[0].compValIdx >> 32);
            uint32_t const maxScoreBits = warpMaxU32(scoreBits);
            uint32_t const indexBits
                = warpMaxU32(scoreBits == maxScoreBits ? static_cast<uint32_t>(keys[0].compValIdx) : 0U);
            packedMax = (static_cast<uint64_t>(maxScoreBits) << 32) | indexBits;
            if (lane == k)
            {
                myTopkIndex = RedType::kMaxIdx - static_cast<int32_t>(indexBits & 0xFFFF);
            }
        }
        __syncwarp();

        float myTopkValue = 0.0f;
        if (lane < kTopK)
        {
            myTopkValue = myScores[myTopkIndex];
        }
        float const weightSum = cg::reduce(warp, myTopkValue, cg::plus<float>{});
        if (lane < kTopK)
        {
            float const weight = (myTopkValue / (weightSum + kWeightEpsilon)) * p.routeScale;
            p.outWeights[row * kTopK + lane] = weight;
            p.outIndices[row * kTopK + lane] = myTopkIndex;
            if constexpr (kRoute)
            {
                smIds[row * kTopK + lane] = myTopkIndex;
                smWts[row * kTopK + lane] = weight;
            }
        }
    }
    if constexpr (kRoute)
    {
        __syncthreads();
        // route_assign_decode_kernel (remap_ids, expert_offset = slotStart): local expert = id - slotStart when in
        // [0, numLocal), else unrouted. Rows sorted by (expert, pair index).
        int const numLocal = p.numLocalExperts;
        auto localOf = [&](int pair)
        {
            int const e = smIds[pair] - p.slotStart;
            return (e >= 0 && e < numLocal) ? e : -1;
        };
        if (threadIdx.x < kPairs)
        {
            int const pid = threadIdx.x;
            int const pe = localOf(pid);
            int expertBegin = 0;
            int rank = 0;
#pragma unroll
            for (int j = 0; j < kPairs; ++j)
            {
                int const ej = localOf(j);
                bool const rj = ej >= 0;
                expertBegin += (rj && ej < pe) ? 1 : 0;
                rank += (rj && ej == pe && j < pid) ? 1 : 0;
            }
            int const routedRow = expertBegin + rank;
            constexpr int kAlign = 4; // SF_M_ALIGN
            int const scaleBegin = (expertBegin + pe * (kAlign - 1)) & -kAlign;
            bool const routed = pe >= 0;
            if (routed)
            {
                p.tokenMap[routedRow] = pid / kTopK;
                p.tokenWeights[routedRow] = smWts[pid];
            }
            int const dst = routed ? routedRow : -1;
            int const sdst = routed ? scaleBegin + rank : -1;
            p.dstRows[pid] = dst;
            p.scaleDstRows[pid] = sdst;
            smDst[pid] = dst;
            smScaleDst[pid] = sdst;
        }
        else if (threadIdx.x >= 64)
        {
            // offsets[e] = routed pairs with expert < e; offsets[numLocal] = all. Strided, so any numLocal works
            // (EP4: 96, EP2xTP2: 192, EP1: 384).
            for (int e = threadIdx.x - 64; e <= numLocal; e += kBlockSize - 64)
            {
                int count = 0;
#pragma unroll
                for (int j = 0; j < kPairs; ++j)
                {
                    int const ej = localOf(j);
                    count += (ej >= 0 && (e == numLocal || ej < e)) ? 1 : 0;
                }
                p.offsets[e] = count;
            }
        }
        __syncthreads();

        // _mxfp8_q0_route_direct_kernel: per token and 128-wide tile, amax (floor 1e-4) -> ue8m0 exponent
        // (amax / 448, rounded up, clamped to 1..254) -> e4m3 of x / 2^(exp-127), scattered to the token's routed rows.
        // The x values come from this CTA's GEMV operand registers (a0, loaded after the wait): warp w holds tiles
        // 2w (chunks 0..3) and 2w+1 (chunks 4..7) of token g, lanes 4g..4g+3 (q) splitting each tile's 128 values.
        // amax is a max (order-free) and the rest is elementwise, so this is bitwise the per-tile kernel.
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int const kt = 2 * warpId + h;
            float amax = 0.0f;
#pragma unroll
            for (int cc = 0; cc < 4; ++cc)
            {
                __nv_bfloat162 const* hv = reinterpret_cast<__nv_bfloat162 const*>(&a0[h * 4 + cc]);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    amax = fmaxf(amax, fmaxf(fabsf(__low2float(hv[j])), fabsf(__high2float(hv[j]))));
                }
            }
            // lanes of one token differ only in q = lane % 4 (rows g >= kNumTokens hold zeros and are not stored)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
            amax = fmaxf(amax, 1.0e-4f);
            float const sf = kQuantDiv(amax, 448.0f);
            uint32_t const bitsSf = __float_as_uint(sf);
            int exp = static_cast<int>((bitsSf >> 23) & 0xFFu) + ((bitsSf & 0x7FFFFFu) != 0u ? 1 : 0);
            exp = min(max(exp, 1), 254);
            float const sfE8 = __uint_as_float(static_cast<uint32_t>(exp) << 23);
            float const inv = kQuantDiv(1.0f, sfE8);
            if (row0Valid)
            {
                uint2 packed[4];
#pragma unroll
                for (int cc = 0; cc < 4; ++cc)
                {
                    __nv_bfloat162 const* hv = reinterpret_cast<__nv_bfloat162 const*>(&a0[h * 4 + cc]);
                    packed[cc].x = packE4m3x4(__low2float(hv[0]) * inv, __high2float(hv[0]) * inv,
                        __low2float(hv[1]) * inv, __high2float(hv[1]) * inv);
                    packed[cc].y = packE4m3x4(__low2float(hv[2]) * inv, __high2float(hv[2]) * inv,
                        __low2float(hv[3]) * inv, __high2float(hv[3]) * inv);
                }
#pragma unroll
                for (int s = 0; s < kTopK; ++s)
                {
                    int const r = smDst[g * kTopK + s];
                    if (r >= 0)
                    {
                        uint8_t* dstRow = p.qOut + static_cast<int64_t>(r) * kHiddenDim + kWarp + h * 4 * kChunk + q * 8;
#pragma unroll
                        for (int cc = 0; cc < 4; ++cc)
                        {
                            *reinterpret_cast<uint2*>(dstRow + cc * kChunk) = packed[cc];
                        }
                    }
                }
                if (q == 0)
                {
                    smExp[g][kt] = static_cast<uint32_t>(exp);
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kPairs * kSfBlocks; i += kBlockSize)
        {
            int const pair = i / kSfBlocks;
            int const blk = i % kSfBlocks;
            int const sr = smScaleDst[pair];
            if (sr >= 0)
            {
                int const t = pair / kTopK;
                uint32_t const word = smExp[t][blk * 4] | (smExp[t][blk * 4 + 1] << 8) | (smExp[t][blk * 4 + 2] << 16)
                    | (smExp[t][blk * 4 + 3] << 24);
                p.scaleOut[blk * p.paddedRows + sr] = static_cast<int>(word);
            }
        }
    }
    if (threadIdx.x == 0)
    {
        *p.counter = 0; // all kNumCtas arrivals are in: ready for the next launch
    }
    __syncthreads();
    triggerDependents();
}

template <int kNumTokens>
void launchFold(FusedRouterFoldParams const& params, cudaStream_t stream)
{
    auto kernel = params.qOut != nullptr ? dsv4FusedRouterKernel<kNumTokens, true, true>
                                         : dsv4FusedRouterKernel<kNumTokens, false, true>;
    constexpr int kSmem = FoldSmem<kNumTokens>::kBytes;
    static bool const kAttrSet = []
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(dsv4FusedRouterKernel<kNumTokens, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(dsv4FusedRouterKernel<kNumTokens, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        return true;
    }();
    (void) kAttrSet;
    cudaLaunchConfig_t config{};
    config.gridDim = kNumCtas;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = kSmem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

template <int kNumTokens>
void launch(FusedRouterParams const& params, cudaStream_t stream)
{
    auto kernel = params.qOut != nullptr ? dsv4FusedRouterKernel<kNumTokens, true, false>
                                         : dsv4FusedRouterKernel<kNumTokens, false, false>;
    cudaLaunchConfig_t config{};
    config.gridDim = kNumCtas;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

} // namespace

void invokeDsv4FusedRouter(FusedRouterParams const& paramsIn, cudaStream_t stream)
{
    FusedRouterParams params = paramsIn;
    // Test only: read once per process (set it before the first launch, e.g. in a dedicated subprocess).
    static bool const kTestSkipWait = []
    {
        char const* skipWait = std::getenv("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT");
        return skipWait != nullptr && skipWait[0] == '1';
    }();
    params.testSkipWait = kTestSkipWait ? 1 : 0;
    switch (params.numTokens)
    {
    case 1: launch<1>(params, stream); break;
    case 2: launch<2>(params, stream); break;
    case 3: launch<3>(params, stream); break;
    case 4: launch<4>(params, stream); break;
    case 5: launch<5>(params, stream); break;
    case 6: launch<6>(params, stream); break;
    case 7: launch<7>(params, stream); break;
    case 8: launch<8>(params, stream); break;
    default: TLLM_CHECK_WITH_INFO(false, "dsv4 fused router supports 1..8 tokens");
    }
}

void invokeDsv4FusedRouterK1Fold(FusedRouterFoldParams const& paramsIn, cudaStream_t stream)
{
    FusedRouterFoldParams params = paramsIn;
    static bool const kTestSkipWait = []
    {
        char const* skipWait = std::getenv("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT");
        return skipWait != nullptr && skipWait[0] == '1';
    }();
    params.testSkipWait = kTestSkipWait ? 1 : 0;
    int const P = params.k1ReducerThreads;
    TLLM_CHECK_WITH_INFO(P == 96 || P == 224 || P == 256 || P == 480, "K1 fold: reducer threads must be 96/224/256/480");
    TLLM_CHECK_WITH_INFO(params.k1Out == params.x, "K1 fold: out must be the router input x");
    switch (params.numTokens)
    {
    case 1: launchFold<1>(params, stream); break;
    case 2: launchFold<2>(params, stream); break;
    case 3: launchFold<3>(params, stream); break;
    case 4: launchFold<4>(params, stream); break;
    default: TLLM_CHECK_WITH_INFO(false, "dsv4 fused router K1 fold supports 1..4 tokens");
    }
}

} // namespace kernels::dsv4_fused_router

TRTLLM_MV2_NAMESPACE_END
