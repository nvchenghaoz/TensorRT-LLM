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
#include "tensorrt_llm/kernels/moe/routing/dsv4FusedRouter.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cstdlib>
#include <math.h>

namespace cg = cooperative_groups;

TRTLLM_NAMESPACE_BEGIN

namespace kernels::dsv4_fused_router
{
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

template <int kNumTokens>
__global__ __launch_bounds__(kBlockSize, 1) void dsv4FusedRouterKernel(FusedRouterParams const __grid_constant__ p)
{
    static_assert(kNumTokens >= 1 && kNumTokens <= kMaxTokens, "1..8 tokens");
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

    if (!p.testSkipWait)
    {
        gridDependencyWait();
    }

    // ---- GEMV: router_gemm_mma_kernel's loads, MMA order and reduction.
    bool const row0Valid = g < kNumTokens;
    uint4 const zero4 = make_uint4(0u, 0u, 0u, 0u);
    __nv_bfloat16 const* aRow0 = p.x + g * kHiddenDim;
    uint4 a0[kChunksPerWarp];
#pragma unroll
    for (int c = 0; c < kChunksPerWarp; c++)
    {
        int const k = kWarp + c * kChunk + q * 8;
        a0[c] = row0Valid ? *reinterpret_cast<uint4 const*>(aRow0 + k) : zero4;
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
        __threadfence(); // the logit stores are visible device-wide before this CTA counts itself in
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
            p.outWeights[row * kTopK + lane] = (myTopkValue / (weightSum + kWeightEpsilon)) * p.routeScale;
            p.outIndices[row * kTopK + lane] = myTopkIndex;
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
void launch(FusedRouterParams const& params, cudaStream_t stream)
{
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
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, dsv4FusedRouterKernel<kNumTokens>, params));
}

} // namespace

void invokeDsv4FusedRouter(FusedRouterParams const& paramsIn, cudaStream_t stream)
{
    FusedRouterParams params = paramsIn;
    char const* skipWait = std::getenv("TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT");
    params.testSkipWait = skipWait != nullptr && skipWait[0] == '1';
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

} // namespace kernels::dsv4_fused_router

TRTLLM_NAMESPACE_END
