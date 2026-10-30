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

// SM120 FP8 row-wise GEMM: d [m, n] bf16 = BF16(sw[n] * (sx[m] * sum_k a[m, k] * b[n, k])), a E4M3
// [m, k] row-major, b E4M3 [n, k] K-major, sx fp32 [m], sw fp32 [n], fp32 accumulation. arm: 1 =
// Pingpong 128x128x64, 3 = Cooperative 128x256x64, 4 = Cooperative 256x128x64 (all bitwise equal).
// Returns the workspace size when a, b and d are all null.
size_t fp8RowwiseGemmSm120(void const* a, void const* b, float const* sx, float const* sw, void* d, int m, int n,
    int k, int arm, char* workspace, size_t workspaceBytes, cudaStream_t stream);

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
