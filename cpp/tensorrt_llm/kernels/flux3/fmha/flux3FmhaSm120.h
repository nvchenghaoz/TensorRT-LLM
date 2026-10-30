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
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace flux3
{

// Self-attention of the FLUX.3 modeling_v2 target (flux3_klein_nvfp4__sm_120__tp1) on SM120/121: per-tensor E4M3
// Q/K/V [B, S, H, 128] (contiguous BSHD), non-causal, FP32 accumulation, O = softmax(scale_bmm1 * Q K^T) V *
// scale_bmm2, where P is quantized to E4M3 with the x256 softmax quant scale. The kernel is the fmha_v2 tiled flash
// kernel of the e4m3 / fp32 64x64 tile, head 128 (Kernel_traits_v2_q_k_v<Ada_qmma_e4m3_fp32_traits, 64, 128, 128, 64,
// 4, 1, 1, 0x1207>) with a base-2 softmax, run as 3 groups of 64 Q rows per CTA over a TMA K/V ring (see
// flux3FmhaG3Kernel.cuh). The output is either BF16 O [B, S, H, 128] or O quantized to NVFP4 in the epilogue.
// The launcher (flux3FmhaSm120.cu) is built with the generated SM120 context FMHA kernels
// (contextFusedMultiHeadAttention/CMakeLists.txt), with --use_fast_math.
struct Fp8AttentionParams
{
    // E4M3 [B, S, H, 128], contiguous, 16-B aligned.
    void const* q;
    void const* k;
    void const* v;
    // The QK^T dequant scale including the softmax scale (q_descale * k_descale * sm_scale), one float32 on device.
    float const* scale_bmm1;
    // The V dequant scale, one float32 on device.
    float const* scale_bmm2;
    int b;
    int s;
    int h;

    // BF16 O [B, S, H, 128] (BF16-out launch; null for the NVFP4-out launch).
    void* o;

    // NVFP4-out launch (o == nullptr): packed E2M1 [B*S, H*64] uint8, the 128x4-swizzled E4M3 block scales
    // (round_up(B*S, 128) * H*8 bytes; rows B*S .. round_up(B*S, 128) are zeroed, unless q_rows trims the launch) and
    // the global scale (one float32 on device). Bitwise FlashInfer's cute-dsl nvfp4_quantize of the BF16 O.
    uint8_t* fp4;
    uint8_t* sf;
    float const* fp4_global_scale;

    // Q-row prefix: 0 or >= s runs every row. Otherwise only Q rows [0, round_up(q_rows, 192)) are computed, each
    // bitwise the full launch's, over every K/V row; later rows (and the padded SF rows) are not written.
    int q_rows;
};

// Q rows per CTA (3 groups x 64); a q_rows prefix launch writes rows [0, round_up(q_rows, kFp8AttentionCtaRows)).
constexpr int kFp8AttentionCtaRows = 192;

// Launch on stream. Throws on unsupported shapes (S % 16 != 0, S % 64 != 0 with B > 1, B * S < 64).
void launchFp8AttentionSm120(Fp8AttentionParams const& params, cudaStream_t stream);

} // namespace flux3
} // namespace kernels

TRTLLM_NAMESPACE_END
