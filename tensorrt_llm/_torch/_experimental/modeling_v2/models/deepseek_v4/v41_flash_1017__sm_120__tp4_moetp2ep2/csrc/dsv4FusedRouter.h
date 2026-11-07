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
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

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

// K1 fold (decode M <= kMaxFoldTokens; TRTLLM_DSV4_ROUTER_K1_FOLD): the router launch also computes the hc_ffn mHC
// lagged layer_input (_mhc_lagged_layer_input_kernel's arithmetic, mhc_lagged_split.py, bit for bit for its reducer
// thread count P) from the boundary's inputs, after griddepcontrol.wait, redundantly in every CTA (shared memory),
// and runs the GEMV (and the Stage 2 q0) from it. It also writes layer_input to `out` (== x, for the MoE kernels
// that read it after this launch).
constexpr int kMaxFoldTokens = 4;

struct FusedRouterFoldParams : FusedRouterParams
{
    __nv_bfloat16 const* k1X;          // [M, 5120] x_prev (the attention output, all-reduced)
    __nv_bfloat16 const* k1Residual;   // [M, 4, 5120]
    float const* k1Post;               // [M, 4]
    float const* k1Comb;               // [M, 4, 4]
    float const* k1Pre;                // [M, 4] (the previous sublayer's pre)
    __nv_bfloat16 const* k1NormWeight; // [5120] (read-only for the forward: staged before the wait)
    __nv_bfloat16* k1Out;              // [M, 5120] layer_input (== x)
    float k1NormEps;
    int k1ReducerThreads;              // P in {96, 224, 256, 480}
};

void invokeDsv4FusedRouterK1Fold(FusedRouterFoldParams const& params, cudaStream_t stream);

} // namespace kernels::dsv4_fused_router

TRTLLM_MV2_NAMESPACE_END
