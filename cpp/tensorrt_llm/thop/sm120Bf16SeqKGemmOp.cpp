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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/sm120Bf16SeqKGemm/sm120Bf16SeqKGemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// out[M, N] = mat_a[M, K] @ weight[N, K]^T in BF16 with FP32 accumulation over ascending k16 steps:
// bitwise equal to cuBLASLt's non-split-K BF16 GEMM (F.linear / trtllm::cublas_mm without split-K).
// `config` selects the CTA tile (kernels::sm120_seqk_gemm::Config); it does not change the bits.
Tensor sm120_bf16_seqk_gemm(Tensor const& mat_a, Tensor const& weight, int64_t config)
{
    TORCH_CHECK(mat_a.dim() == 2 && weight.dim() == 2, "sm120_bf16_seqk_gemm: 2D inputs required");
    TORCH_CHECK(mat_a.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
        "sm120_bf16_seqk_gemm: BF16 inputs required");
    TORCH_CHECK(mat_a.is_cuda() && weight.is_cuda(), "sm120_bf16_seqk_gemm: CUDA tensors required");
    TORCH_CHECK(mat_a.size(1) == weight.size(1), "sm120_bf16_seqk_gemm: K mismatch");
    TORCH_CHECK(mat_a.stride(1) == 1 && weight.stride(1) == 1, "sm120_bf16_seqk_gemm: unit inner stride required");
    int64_t const M = mat_a.size(0);
    int64_t const N = weight.size(0);
    int64_t const K = mat_a.size(1);
    auto out = at::empty({M, N}, mat_a.options());
    if (M == 0)
    {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    int const status = kernels::sm120_seqk_gemm::run(mat_a.data_ptr(), weight.data_ptr(), out.data_ptr(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), mat_a.stride(0), weight.stride(0),
        out.stride(0), static_cast<int>(config), stream);
    TORCH_CHECK(status == 0, "sm120_bf16_seqk_gemm failed with status ", status, " (M=", M, ", N=", N, ", K=", K,
        ", config=", config, ")");
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("sm120_bf16_seqk_gemm(Tensor mat_a, Tensor weight, int config) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sm120_bf16_seqk_gemm", &tensorrt_llm::torch_ext::sm120_bf16_seqk_gemm);
}
