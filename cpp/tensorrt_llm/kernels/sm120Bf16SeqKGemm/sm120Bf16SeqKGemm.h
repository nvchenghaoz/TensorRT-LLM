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

#include <cuda_runtime.h>
#include <stdint.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::sm120_seqk_gemm
{

// Tile configurations (tokens x weights per CTA, warps). All walk K in ascending k16 steps with FP32
// accumulation and one RN cast, so their outputs are bitwise identical to each other and to
// cuBLASLt's non-split-K BF16 kernels; the choice only affects speed.
enum Config : int
{
    kT160W176 = 0,   // 12 warps; sliding qkv (N=16384) at M=300: 94 x 2 = 188 CTAs, one wave
    kT160W208W8 = 1, // 8 warps; global [q|k] (N=18432) at M=300: 89 x 2 = 178 CTAs
    kT160W208 = 2,   // 12 warps
    kT128W176 = 3,   // 12 warps; 128-token tiles (e.g. M=1000: 8 x 94 = 752 CTAs, 4 waves)
    kNumConfigs = 4
};

enum Status : int
{
    kOk = 0,
    kErrShape = 1,  // K, N, strides or alignment unsupported
    kErrEncode = 2, // cuTensorMapEncodeTiled unavailable or failed
    kErrAttr = 3,   // cudaFuncSetAttribute failed
    kErrLaunch = 4, // launch error
    kErrConfig = 5  // unknown config
};

// Y[M, N] = X[M, K] @ W[N, K]^T for BF16 X, W, Y with unit inner stride. Requires SM90+ device
// (TMA); written for SM120. Requirements: K % 32 == 0, N % 8 == 0, ldx/ldw/ldy multiples of 8
// elements, 16-byte aligned pointers. Capturable in CUDA graphs (no allocation, no host sync).
int run(void const* x, void const* w, void* y, int M, int N, int K, int64_t ldx, int64_t ldw, int64_t ldy, int config,
    cudaStream_t stream);

// Tile rows (tokens) and columns (weights) of a config, or {0, 0} for an unknown one.
void tileShape(int config, int* tokens, int* weights);

} // namespace kernels::sm120_seqk_gemm

TRTLLM_NAMESPACE_END
