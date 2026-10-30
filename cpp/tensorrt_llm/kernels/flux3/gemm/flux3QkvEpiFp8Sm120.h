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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::flux3
{

// The minimum per-tensor E4M3 attention scale, fp32(1 / (448 * 512)) = 0x36924925.
constexpr float kFp8AttentionMinScale = 1.0f / (448.0f * 512.0f);

// SM120 NVFP4 QKV GEMM (batched L = 3: q, k, v planes of width n) whose epilogue writes E4M3 Q/K
// directly (see flux3QkvEpiFp8Sm120.cu). a: [m, k / 2] packed E2M1 with sfa (128x4-swizzled, shared
// by the planes); b: [3n, k / 2] packed E2M1 (q, k, v rows) with sfb (the three planes' swizzled
// scales back to back); alpha: fp32 [3] per-plane alphas; qNorm / kNorm: bf16 [128]; ropeTab: fp32
// [entries, 32] / ropeIdx: int32 [>= m, 4] compact RoPE table. Writes qk8 planes 0 and 1 ([2][m][n]
// E4M3 bytes), the bf16 v plane [m][n] and vParts (fp32 [ceil(m / 128) * n / 128] per-tile max|v|).
// Returns the workspace size when a, b and qk8 are all null.
size_t qkvEpiFp8GemmSm120(void const* a, void const* b, void const* sfa, void const* sfb, float const* alpha,
    void const* qNorm, void const* kNorm, float const* ropeTab, int const* ropeIdx, uint8_t* qk8, void* v,
    float* vParts, int m, int n, int k, char* workspace, size_t workspaceBytes, cudaStream_t stream);

// Reduces vParts[0, nPart) to max|v|, writes scales[0..7] (q, k, v scales; q * k * smScale; the three
// reciprocal factors; 0) and quantizes the bf16 v plane (elems elements, a multiple of 64) to E4M3
// v8. discard: drop every consumed v line from L2 (the caller never reads v again). grid CTAs of 512
// threads walk the plane from its last 16-B chunk down.
void vQuantSm120(void const* v, uint8_t* v8, float const* vParts, int nPart, void const* qNorm, void const* kNorm,
    float* scales, int64_t elems, float smScale, int grid, bool discard, cudaStream_t stream);

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
