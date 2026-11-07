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

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! Whether `invokeCsa2LowMWoA` is built for (warps, clusterK, rowTiles) at M rows and supports the shape.
bool isCsa2LowMWoASupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t groups,
    int32_t rank, int32_t groupWidth, int32_t headDim, int32_t nopeDim);

//! Dynamic shared memory one CTA of `invokeCsa2LowMWoA` requests.
size_t csa2LowMWoASmemBytes(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t groupWidth);

//! SM120 decode wo_a of the CSA2 (DeepSeek-V4.1) output projection in one launch:
//!   out[m, g * rank + r] = sum_k q(m, g, k) * w[g, r, k], q = 1x128 UE8M0 quant of the BF16 inverse-RoPE'd
//! attention output `o` [M, groups * groupWidth] (heads of `headDim`, the last headDim - nopeDim of each interleaved
//! rotary dims; cos/sin rows `cosSin` [pos][cos(half) | sin(half)] FP32). The quantized activation is byte-equal to
//! b12x_wo._inv_rope_quant128 (mla_rope_inplace inverse form + fp8_utils 1x128 UE8M0 arithmetic). `w` is E4M3
//! [groups * rank, groupWidth] with UE8M0 128x128 block scales `wsc` [groups, rank / 128, groupWidth / 128].
//! Weights are staged into shared memory before griddepcontrol.wait; the kernel triggers its dependents at start.
//! `xqDbg` [M, groups, groupWidth] / `xsDbg` [M, groups, groupWidth / 128] (optional) receive the quantized
//! activation. `debugFlags` bit 0 skips griddepcontrol.wait (negative control of the PDL checks only).
void invokeCsa2LowMWoA(__nv_bfloat16 const* o, int32_t ldo, void const* positions, bool positions64,
    float const* cosSin, int32_t csStride, uint8_t const* w, uint8_t const* wsc, __nv_bfloat16* out, int32_t ldOut,
    int32_t m, int32_t groups, int32_t rank, int32_t groupWidth, int32_t headDim, int32_t nopeDim, int32_t warps,
    int32_t clusterK, int32_t rowTiles, bool pdl, cudaStream_t stream, uint8_t* xqDbg = nullptr,
    uint8_t* xsDbg = nullptr, int32_t debugFlags = 0);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
