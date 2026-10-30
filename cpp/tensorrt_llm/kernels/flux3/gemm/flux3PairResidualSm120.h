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

// [0, 8 KB) args (tile counter, launch epoch, grouped-GEMM arrays), then the CUTLASS workspace
// (per-CTA tensormaps, ~0.2 MB), flags from 4 MB: 8 uint32 per 128x128 tile (1M flags = 131072
// tiles, M up to ~700k rows at N = 3072). Zero-initialized once, before first use and outside any
// CUDA graph capture: the epoch starts at 0 and every launch increments it before use, so a zero flag
// never matches; a zero fill captured into a graph would reset the epoch on every replay.
constexpr size_t kPairResidualWorkspaceBytes = size_t(8) << 20;

// The NVFP4 block tail as ONE persistent SM120 launch (see flux3PairResidualSm120.cu):
//   mlpY = BF16(alpha0 * (a0 @ b0^T))                        (group 0, down_proj, scratch)
//   out  = x + gate[gateIndex] * (BF16(alpha1 * (a1 @ b1^T)) + mlpY)
// a0: [m, k0 / 2] / a1: [m, k1 / 2] packed E2M1 with sfa0 / sfa1 (128x4-swizzled), b0 / b1: [n, k / 2]
// with sfb0 / sfb1; alpha0 / alpha1: device fp32 scalars; mlpY, x, out: bf16 [m, n] (out must not
// alias x); gate: bf16 [R, n] row table; gateIndex: uint8 [m] row per output row, or null (row 0).
// t256: the 256x128x128 tile instance (else 128x128x128); alongN: raster the tiles along N (else the
// scheduler's default, AlongM). Every instance and raster gives the same bytes.
void nvfp4PairResidualSm120(void const* a0, void const* b0, void const* sfa0, void const* sfb0, float const* alpha0,
    void* mlpY, void const* a1, void const* b1, void const* sfa1, void const* sfb1, float const* alpha1, void const* x,
    void const* gate, uint8_t const* gateIndex, void* out, int m, int n, int k0, int k1, void* workspace,
    size_t workspaceBytes, bool t256, bool alongN, cudaStream_t stream);

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
