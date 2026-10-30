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

// Row width the adaLN head serves (FLUX.3 hidden size).
constexpr int kAdalnHidden = 3072;

// SM120 warp-per-row LayerNorm (identity affine) + adaLN modulate + NVFP4 quantize, D = 3072 (see
// flux3AdalnNvfp4Sm120.cu). x: bf16 [m, D]; scale / shift: bf16 row tables, row r at r * rowStride
// elements (unit last stride, 16-B aligned rows); select: per-row uint8 (null: row 0 of the tables);
// selectIsIndex: select is a row index into the tables (else a bool picking row 1 when nonzero);
// globalScale: device fp32 scalar. Writes fp4 [m, D / 2] bytes and sf (ceil128(m) x D / 16,
// 128x4-swizzled, padding rows zero).
void adalnNvfp4WarpSm120(void const* x, uint8_t const* select, bool selectIsIndex, void const* scale,
    void const* shift, int64_t scaleRowStride, int64_t shiftRowStride, float const* globalScale, uint8_t* fp4,
    uint8_t* sf, int m, float eps, cudaStream_t stream);

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
