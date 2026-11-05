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

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::dsv4_fused_router
{

// DeepSeek-V4.1-Flash decode router in one launch: router GEMV (384 experts x hidden 5120, bf16 -> fp32 logits,
// the arithmetic of dsv3RouterGemm.cu's router_gemm_mma_kernel<M,384,5120,8,640>) followed by the top-6
// sqrt-softplus gate (gate_forward_kernel<384,6,false,false>'s arithmetic), for 1 <= M <= 8 tokens.
constexpr int kNumExperts = 384;
constexpr int kHiddenDim = 5120;
constexpr int kTopK = 6;
constexpr int kMaxTokens = 8;

struct FusedRouterParams
{
    __nv_bfloat16 const* x;      // [M, 5120] row-major
    __nv_bfloat16 const* weight; // [384, 5120] row-major (the gate's nn.Parameter)
    float const* bias;           // [384] e_score_correction_bias
    float* logits;               // [M, 384] scratch / debug output
    int* outIndices;             // [M, 6] global expert ids
    float* outWeights;           // [M, 6]
    int* counter;                // [1] arrival counter, 0 between launches (reset in-kernel)
    float routeScale;
    int numTokens;
    // Test only (TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT=1, read once per process): skip griddepcontrol.wait, the negative
    // control of the PDL poison check.
    int testSkipWait;

    // Stage 2 (qOut != nullptr): also write what FlashInfer's mxfp8_q0_route_triton decode branch writes
    // (route_assign_decode_kernel with remap_ids/expert_offset = slotStart and zero_buf, then
    // _mxfp8_q0_route_direct_kernel) into its Mxfp8Q0RouteWorkspace.
    int slotStart;
    int numLocalExperts;
    int* offsets;          // [numLocal + 1]
    int* tokenMap;         // [M * 6]
    float* tokenWeights;   // [M * 6]
    int* dstRows;          // [M, 6]
    int* scaleDstRows;     // [M, 6]
    uint8_t* qOut;         // [M * 6, 5120] e4m3
    int* scaleOut;         // [5120 / 512, paddedRows] contiguous
    int paddedRows;
    int64_t* zeroBuf;      // caller buffer zero-filled (int64 words)
    int64_t zeroNumel;
};

void invokeDsv4FusedRouter(FusedRouterParams const& params, cudaStream_t stream);

} // namespace kernels::dsv4_fused_router

TRTLLM_NAMESPACE_END
