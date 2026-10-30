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

// SM120 NVFP4 W4A4 gate_up GEMM + SwiGLU + NVFP4 quantize epilogue (see flux3GateUpSwigluNvfp4Sm120.cu
// for the gate_up weight-row layout it expects). a: [m, k / 2] packed E2M1 activations with sfa
// (128x4-swizzled); b: [n, k / 2] packed E2M1 gate_up weight in [value, gate] pairs, rows permuted
// within every 64-row group, with sfb; alpha: device fp32 scalar (input * weight global scale);
// outScale: device fp32 scalar, down_proj's activation global scale. Writes out [rowsPadded, n / 4]
// FP4 bytes and outSf (rowsPadded x n / 32, 128x4-swizzled); every row below rowsPadded (a multiple
// of 128 >= m) is written, the padding rows from the GEMM's zero-filled rows.
void gateUpSwigluNvfp4Sm120(void const* a, void const* b, void const* sfa, void const* sfb, float const* alpha,
    float const* outScale, uint8_t* out, uint8_t* outSf, int m, int n, int k, int rowsPadded, cudaStream_t stream);

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
