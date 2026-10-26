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

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace qwen35
{

// Context FMHA of the Qwen3.5-family staircase target on SM120/121: the e4m3-input, BF16-output, head-256, causal,
// Q_PAGED_KV tiled flash kernel (64-row q-tiles, 128-key KV tiles), compiled with the base-2 softmax
// (Kernel_traits::USE_EXP2_SOFTMAX). When dropping the last q-tile from its grid saves a whole wave (1 to
// kContextFmhaSm120MaxBatch sequences), the last q-tile runs split along KV in pieces plus a fixed-order merge. The
// launcher is reached through Launch_params::sm120_target_fmha (see fused_multihead_attention_v2.cpp) and lives in
// qwen35ContextFmhaSm120.cu, which is only built with the SM120 FMHA kernels (TLLM_ENABLE_SKIP_SOFTMAX_SM120).

// The largest batch the last q-tile split runs for.
constexpr int kContextFmhaSm120MaxBatch = 3;
// The most pieces the split uses in total; each piece needs 128 threads x (32 + 1) float4 of FP32 accumulators and
// row max / sum.
constexpr int kContextFmhaSm120MaxPiecesTotal = 384;
// The workspace of the split (Launch_params::sm120_target_fmha_workspace): 25 952 256 bytes.
constexpr size_t kContextFmhaSm120WorkspaceBytes = size_t{kContextFmhaSm120MaxPiecesTotal} * 128 * (32 + 1) * 16;

} // namespace qwen35
} // namespace kernels

TRTLLM_NAMESPACE_END
