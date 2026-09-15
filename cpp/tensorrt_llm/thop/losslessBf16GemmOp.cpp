// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/losslessBf16Gemm.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <torch/library.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
at::Tensor losslessBf16Gemm(at::Tensor const& input, at::Tensor const& weight, at::Tensor const& metadata,
    at::Tensor const& exceptions, int64_t outFeatures)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == at::kBFloat16,
        "input must be contiguous CUDA BF16");
    TORCH_CHECK(input.dim() == 2 && input.size(0) <= 4 * 65535 && input.size(1) > 0 && input.size(1) % 128 == 0,
        "input must have shape [M, K] (M <= 262140), with positive K divisible by 128");
    TORCH_CHECK(outFeatures > 0 && outFeatures <= std::numeric_limits<int>::max(), "invalid output size");
    int64_t const inFeatures = input.size(1);
    TORCH_CHECK(inFeatures <= std::numeric_limits<int>::max()
            && outFeatures <= std::numeric_limits<int>::max() / (inFeatures / 128),
        "weight index overflow");
    int64_t const blocks = outFeatures * (inFeatures / 128);
    for (auto const* tensor : {&weight, &metadata, &exceptions})
    {
        TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous() && tensor->device() == input.device(),
            "packed tensors must be contiguous on the input CUDA device");
    }
    TORCH_CHECK(
        weight.scalar_type() == at::kByte && weight.dim() == 2 && weight.size(0) == blocks && weight.size(1) == 176,
        "invalid packed weight shape or dtype");
    TORCH_CHECK(metadata.dim() == 1 && metadata.numel() == blocks && metadata.scalar_type() == at::kLong,
        "invalid metadata shape or dtype");
    TORCH_CHECK(exceptions.scalar_type() == at::kByte && exceptions.dim() == 1, "invalid exception shape or dtype");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 4 == 0
            && reinterpret_cast<uintptr_t>(metadata.data_ptr()) % 8 == 0,
        "packed buffers are not aligned");
    c10::cuda::CUDAGuard const guard(input.device());
    auto output = at::empty({input.size(0), outFeatures}, input.options());
    kernels::invokeLosslessBf16Gemm(reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
        weight.data_ptr<uint8_t>(), reinterpret_cast<uint64_t const*>(metadata.data_ptr()),
        exceptions.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<int>(input.size(0)), static_cast<int>(outFeatures), static_cast<int>(inFeatures),
        at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "lossless_bf16_gemm(Tensor input, Tensor weight, Tensor metadata, Tensor exceptions, int out_features) -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lossless_bf16_gemm", &losslessBf16Gemm);
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END
