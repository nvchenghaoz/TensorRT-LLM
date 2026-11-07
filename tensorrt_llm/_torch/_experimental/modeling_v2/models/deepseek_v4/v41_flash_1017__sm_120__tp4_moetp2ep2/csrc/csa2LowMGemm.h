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

//! SM120 port of the MiniMax-M3 low-M MXFP8 GEMM (minimaxM3/mxfp8LowMGemm.h) for the CSA2 decode input
//! projections. Built configurations (warps, clusterK, rowTiles): see CSA2_LOW_M_CONFIGS in csa2LowMGemm.cu, each for
//! 1 <= M <= 8.
bool isCsa2Mxfp8LowMGemmSupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t stages = 1);

//! Dynamic shared memory one CTA of `invokeCsa2Mxfp8LowMGemm` requests.
size_t csa2Mxfp8LowMGemmSmemBytes(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t k);

//! `out[M, :nSplit] / out2[M, :N - nSplit] = mxfp8(x)[M, K] @ w[N, K]^T` in BF16 (SM120).
//! `xq`/`xsf` (optional): an already-quantized activation (E4M3 [M, K] + 128x4-swizzled UE8M0 scales of
//! `mxfp8_quantize(x, True)`); then `x` is not read. `xqOut`/`xsfOut` (optional): receive the in-kernel quantized
//! activation in that same layout.
void invokeCsa2Mxfp8LowMGemm(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t n, int32_t k, uint8_t const* w,
    uint8_t const* wsf, __nv_bfloat16* out, int32_t ldo, __nv_bfloat16* out2, int32_t ldo2, int32_t nSplit,
    uint8_t const* xq, uint8_t const* xsf, uint8_t* xqOut, uint8_t* xsfOut, int32_t warps, int32_t clusterK,
    int32_t rowTiles, bool pdl, cudaStream_t stream, unsigned long long* dbgTs = nullptr, int32_t stages = 1,
    int32_t debugFlags = 0, uint8_t const* pfP0 = nullptr, int64_t pfN0 = 0, uint8_t const* pfP1 = nullptr,
    int64_t pfN1 = 0, int32_t pfFrac = 256);
//! debugFlags bit 2 (with pfP0/pfP1): warp 0 of every CTA issues cp.async.bulk.prefetch.L2 over its 1/gridDim share
//! of the LAST pfFrac/256 of each byte range [pfP0, pfP0 + pfN0) and [pfP1, pfP1 + pfN1) (the next kernel's weight and
//! scale bytes). Bits 3-4 place it: 0 after the CTA's own bulk copies (before the trigger), 1 after its own weight
//! mbarrier completes, 2 after griddepcontrol.wait. Bit 5 uses an L2 evict_last policy. A hint: outputs unchanged.

//! Whether `invokeCsa2QnormLowMGemm` is built for (warps, clusterK, rowTiles) at M rows and K columns (K = 1280).
bool isCsa2QnormLowMGemmSupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m, int32_t k);

//! Fused CSA2 q_norm + MXFP8 quantization + low-M GEMM (SM120 decode attention-input chain):
//! `qr = rmsnorm(x, normW, eps)` bit for bit as FlashInfer rmsnorm_cute (written to `qrOut`, row stride `ldqr`, when
//! non-null), then `out[M, :nSplit] / out2[M, :N - nSplit] = mxfp8(qr) @ w[N, K]^T` in BF16, with mxfp8(qr) byte-equal
//! to `mxfp8_quantize(qr, True)`. `x` is the raw [M, K] row view (row stride `ldx`). `debugFlags` as in
//! `invokeCsa2Mxfp8LowMGemm` (bit 0 skips the PDL wait: negative control only; bit 1 triggers dependents at start).
void invokeCsa2QnormLowMGemm(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t n, int32_t k, uint8_t const* w,
    uint8_t const* wsf, __nv_bfloat16 const* normW, float eps, __nv_bfloat16* qrOut, int32_t ldqr, __nv_bfloat16* out,
    int32_t ldo, __nv_bfloat16* out2, int32_t ldo2, int32_t nSplit, int32_t warps, int32_t clusterK, int32_t rowTiles,
    bool pdl, cudaStream_t stream, int32_t debugFlags = 0, uint8_t* xqOut = nullptr, uint8_t* xsfOut = nullptr);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
