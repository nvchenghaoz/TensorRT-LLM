/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmNVFP4.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/outputTensor.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <cstdint>
#include <torch/extension.h>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;

void cuda_core_nvfp4_gemm_caller(Tensor& out, Tensor const& a, Tensor const& b, Tensor const& scale_a,
    Tensor const& scale_b, Tensor const& alpha, std::optional<at::Tensor> const& bias, bool scale_a_swizzled)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k = a.sizes()[1] * 2;
    TORCH_CHECK(a.sizes()[1] == b.sizes()[1]);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    void* a_scale = static_cast<void*>(scale_a.data_ptr());
    void* b_scale = static_cast<void*>(scale_b.data_ptr());
    void* alpha_ptr = static_cast<void*>(alpha.data_ptr());

    TORCH_CHECK(a_scale != nullptr);
    TORCH_CHECK(b_scale != nullptr);
    TORCH_CHECK(alpha_ptr != nullptr);

    cudaDataType_t aType = convert_torch_dtype(a.scalar_type());
    cudaDataType_t bType = convert_torch_dtype(b.scalar_type());
    cudaDataType_t outType = convert_torch_dtype(out.scalar_type());
    TORCH_CHECK(aType == bType);
    cudaDataType_t scaleAType = convert_torch_dtype(scale_a.scalar_type());
    cudaDataType_t scaleBType = convert_torch_dtype(scale_b.scalar_type());
    TORCH_CHECK(scaleAType == CUDA_R_8U);
    TORCH_CHECK(scaleBType == CUDA_R_8U);
    cudaDataType_t alphaType = convert_torch_dtype(alpha.scalar_type());
    TORCH_CHECK(alphaType == CUDA_R_32F);

    void const* bias_ptr = nullptr;
    if (bias.has_value())
    {
        auto const& bias_tensor = *bias;
        CHECK_TH_CUDA(bias_tensor);
        TORCH_CHECK(bias_tensor.device() == out.device(), "bias must reside on the same CUDA device as the output");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1-D");
        TORCH_CHECK(bias_tensor.sizes()[0] == n, "bias size must equal n=", n);
        TORCH_CHECK(bias_tensor.scalar_type() == out.scalar_type(), "bias dtype must match output dtype");
        bias_ptr = bias_tensor.data_ptr();
    }

    tensorrt_llm::kernels::cuda_core_gemm_nvfp4::Params params(a_ptr, b_ptr, out_ptr, m, n, k,
        reinterpret_cast<__nv_fp8_e4m3 const*>(a_scale), reinterpret_cast<__nv_fp8_e4m3 const*>(b_scale), aType,
        outType, reinterpret_cast<float const*>(alpha_ptr), bias_ptr, scale_a_swizzled);
    bool dispatched = tensorrt_llm::kernels::cuda_core_gemm_nvfp4::cudaCoreGemmDispatcher(params, stream);
    TORCH_CHECK(dispatched, "Failed to dispatch cudaCoreGemmLauncher");
}

} // namespace

Tensor& cuda_core_nvfp4_gemm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, std::optional<at::Tensor> const& bias, Tensor& out, bool scale_a_swizzled)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(alpha);
    CHECK_TH_CUDA(out);

    CHECK_INPUT(mat_a, FLOAT4_E2M1X2);
    CHECK_INPUT(mat_b, FLOAT4_E2M1X2);

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(mat_a.sizes()[0] == out.sizes()[0]);
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[1]);
    TORCH_CHECK(mat_b.sizes()[0] == out.sizes()[1]);

    CHECK_INPUT(scale_a, SF_DTYPE);
    CHECK_INPUT(scale_b, SF_DTYPE);
    TORCH_CHECK(mat_b.device() == mat_a.device() && alpha.device() == mat_a.device()
            && scale_a.device() == mat_a.device() && scale_b.device() == mat_a.device(),
        "GEMM tensors must reside on the activation device");
    TORCH_CHECK(mat_a.size(1) % 16 == 0, "FP4 vector loads require K divisible by 32");
    // Raw vector loads require aligned payload and scale pointers, including offset views.
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(mat_a.data_ptr()) % 16 == 0
            && reinterpret_cast<std::uintptr_t>(mat_b.data_ptr()) % 16 == 0,
        "FP4 payloads must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(scale_a.data_ptr()) % 2 == 0
            && reinterpret_cast<std::uintptr_t>(scale_b.data_ptr()) % 2 == 0,
        "Scale buffers must be 2-byte aligned");
    auto const scale_cols = mat_a.size(1) / 8;
    auto const padded_scale_cols = (scale_cols + 3) / 4 * 4;
    auto const activation_scale_size
        = scale_a_swizzled ? ((mat_a.size(0) + 127) / 128 * 128) * padded_scale_cols : mat_a.size(0) * scale_cols;
    TORCH_CHECK(scale_a.numel() >= activation_scale_size, "Activation scale buffer is too small for its layout");
    TORCH_CHECK(scale_b.numel() >= ((mat_b.size(0) + 127) / 128 * 128) * padded_scale_cols,
        "Weight scale buffer is too small for its 128x4 layout");
    if (out.numel() == 0)
    {
        return out;
    }
    cuda_core_nvfp4_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, alpha, bias, scale_a_swizzled);
    return out;
}

// mat_a: [M, K / 2], FLOAT4_E2M1X2
// mat_b: [N, K / 2], FLOAT4_E2M1X2
// out: [M, N], fp16/bf16/fp32
// scale_a: row-major [M, K/16], or padded 128x4 layout when scale_a_swizzled is true
// scale_b: padded 128x4 layout
// bias: fp16/bf16/fp32
// out_dtype: fp16/bf16/fp32
Tensor cuda_core_nvfp4_gemm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype,
    int64_t output_buffer_kind = 0, c10::optional<torch::List<int64_t>> group = c10::nullopt,
    bool scale_a_swizzled = false)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());

    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[0]};

    auto [out, _] = torch_ext::allocate_output(
        output_size, out_dtype_, mat_a.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);

    return cuda_core_nvfp4_gemm_out(mat_a, mat_b, scale_a, scale_b, alpha, bias, out, scale_a_swizzled);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cuda_core_nvfp4_gemm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor alpha, Tensor? bias,"
        " ScalarType? out_dtype, int output_buffer_kind=0, int[]? group=None, bool scale_a_swizzled=False)"
        " -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cuda_core_nvfp4_gemm", &tensorrt_llm::torch_ext::cuda_core_nvfp4_gemm);
}
