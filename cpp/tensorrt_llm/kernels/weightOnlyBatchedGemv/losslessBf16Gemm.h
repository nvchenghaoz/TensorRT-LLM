// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
//! Multiply exact three-bit BF16 storage by a batch, accumulating in FP32.
void invokeLosslessBf16Gemm(__nv_bfloat16 const* input, uint8_t const* weight, uint64_t const* metadata,
    uint8_t const* exceptions, __nv_bfloat16* output, int rows, int outFeatures, int inFeatures, cudaStream_t stream);
} // namespace kernels

TRTLLM_NAMESPACE_END
