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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/flux3/fmha/flux3FmhaSm120.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kHeadDim = 128;

// Checks q, k, v (E4M3 [B, S, H, 128], contiguous, one shape) and the two one-element float32 device scales, and
// returns the launch parameters without outputs.
tensorrt_llm::kernels::flux3::Fp8AttentionParams makeArgs(at::Tensor const& q, at::Tensor const& k, at::Tensor const& v,
    at::Tensor const& scaleBmm1, at::Tensor const& scaleBmm2)
{
    CHECK_INPUT(q, at::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(k, at::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(v, at::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(scaleBmm1, at::ScalarType::Float);
    CHECK_INPUT(scaleBmm2, at::ScalarType::Float);
    TORCH_CHECK(q.dim() == 4 && q.size(3) == kHeadDim, "q must be [B, S, H, 128]");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k and v must have one shape");
    TORCH_CHECK(scaleBmm1.numel() == 1 && scaleBmm2.numel() == 1, "scale_bmm1 and scale_bmm2 must hold one element");
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() / 10 == 12, "the FLUX.3 FP8 attention requires SM12x");
    TORCH_CHECK(q.size(0) <= INT32_MAX && q.size(1) <= INT32_MAX && q.size(2) <= INT32_MAX, "shape too large");
    tensorrt_llm::kernels::flux3::Fp8AttentionParams args{};
    args.q = q.const_data_ptr();
    args.k = k.const_data_ptr();
    args.v = v.const_data_ptr();
    args.scale_bmm1 = scaleBmm1.const_data_ptr<float>();
    args.scale_bmm2 = scaleBmm2.const_data_ptr<float>();
    args.b = static_cast<int>(q.size(0));
    args.s = static_cast<int>(q.size(1));
    args.h = static_cast<int>(q.size(2));
    return args;
}

// (fp4 [B, S, H*64] uint8, sf flat uint8 [round_up(B*S, 128) * round_up(H*8, 4)]).
std::vector<at::Tensor> nvfp4Out(at::Tensor const& q, at::Tensor const& k, at::Tensor const& v,
    at::Tensor const& scaleBmm1, at::Tensor const& scaleBmm2, at::Tensor const& globalScale, int64_t qRows)
{
    auto args = makeArgs(q, k, v, scaleBmm1, scaleBmm2);
    CHECK_INPUT(globalScale, at::ScalarType::Float);
    TORCH_CHECK(globalScale.numel() == 1, "global_scale must hold one element");
    TORCH_CHECK(qRows >= 0 && qRows <= q.size(1), "q_rows must lie in [0, S], got ", qRows);
    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(q));
    int64_t const rows = q.size(0) * q.size(1);
    int64_t const width = q.size(2) * kHeadDim;
    int64_t const sfBytes = (rows + 127) / 128 * 128 * ((width / 16 + 3) / 4 * 4);
    at::Tensor fp4
        = at::detail::empty_cuda({q.size(0), q.size(1), width / 2}, at::ScalarType::Byte, q.device(), std::nullopt);
    at::Tensor sf = at::detail::empty_cuda({sfBytes}, at::ScalarType::Byte, q.device(), std::nullopt);
    args.o = nullptr;
    args.fp4 = fp4.data_ptr<uint8_t>();
    args.sf = sf.data_ptr<uint8_t>();
    args.fp4_global_scale = globalScale.const_data_ptr<float>();
    args.q_rows = static_cast<int>(qRows);
    tensorrt_llm::kernels::flux3::launchFp8AttentionSm120(args, at::cuda::getCurrentCUDAStream(q.get_device()));
    return {fp4, sf};
}

} // namespace

// FLUX.3 FP8 attention (kernels/flux3/fmha/flux3FmhaSm120.h), BF16 out: O [B, S, H, 128] =
// softmax(scale_bmm1 * Q K^T) V * scale_bmm2, non-causal, E4M3 Q/K/V [B, S, H, 128].
at::Tensor flux3_fp8_attention_bf16_out(at::Tensor const& q, at::Tensor const& k, at::Tensor const& v,
    at::Tensor const& scale_bmm1, at::Tensor const& scale_bmm2)
{
    auto args = makeArgs(q, k, v, scale_bmm1, scale_bmm2);
    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(q));
    at::Tensor out = at::detail::empty_cuda(q.sizes(), at::ScalarType::BFloat16, q.device(), std::nullopt);
    args.o = out.data_ptr();
    tensorrt_llm::kernels::flux3::launchFp8AttentionSm120(args, at::cuda::getCurrentCUDAStream(q.get_device()));
    return out;
}

// The same attention with O quantized to NVFP4 in the epilogue with global_scale: (fp4 [B, S, H*64] uint8,
// 128x4-swizzled E4M3 sf uint8 [round_up(B*S, 128) * H*8], padding rows zero). Bitwise FlashInfer's cute-dsl
// nvfp4_quantize of flux3_fp8_attention_bf16_out's O.
std::vector<at::Tensor> flux3_fp8_attention_nvfp4_out(at::Tensor const& q, at::Tensor const& k, at::Tensor const& v,
    at::Tensor const& scale_bmm1, at::Tensor const& scale_bmm2, at::Tensor const& global_scale)
{
    return nvfp4Out(q, k, v, scale_bmm1, scale_bmm2, global_scale, 0);
}

// flux3_fp8_attention_nvfp4_out computing only the Q-row prefix [0, q_rows) (rounded up to 192 rows) over every K/V
// row: fp4 rows [0, q_rows) and the SF row blocks holding them are bitwise the full launch's; later rows and the padded
// SF rows are not written. q_rows == 0 or S runs every row.
std::vector<at::Tensor> flux3_fp8_attention_nvfp4_out_q_rows(at::Tensor const& q, at::Tensor const& k,
    at::Tensor const& v, at::Tensor const& scale_bmm1, at::Tensor const& scale_bmm2, at::Tensor const& global_scale,
    int64_t q_rows)
{
    return nvfp4Out(q, k, v, scale_bmm1, scale_bmm2, global_scale, q_rows);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("flux3_fp8_attention_bf16_out(Tensor q, Tensor k, Tensor v, Tensor scale_bmm1, Tensor scale_bmm2) -> Tensor");
    m.def(
        "flux3_fp8_attention_nvfp4_out(Tensor q, Tensor k, Tensor v, Tensor scale_bmm1, Tensor scale_bmm2, "
        "Tensor global_scale) -> Tensor[]");
    m.def(
        "flux3_fp8_attention_nvfp4_out_q_rows(Tensor q, Tensor k, Tensor v, Tensor scale_bmm1, Tensor scale_bmm2, "
        "Tensor global_scale, int q_rows) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("flux3_fp8_attention_bf16_out", &tensorrt_llm::torch_ext::flux3_fp8_attention_bf16_out);
    m.impl("flux3_fp8_attention_nvfp4_out", &tensorrt_llm::torch_ext::flux3_fp8_attention_nvfp4_out);
    m.impl("flux3_fp8_attention_nvfp4_out_q_rows", &tensorrt_llm::torch_ext::flux3_fp8_attention_nvfp4_out_q_rows);
}
