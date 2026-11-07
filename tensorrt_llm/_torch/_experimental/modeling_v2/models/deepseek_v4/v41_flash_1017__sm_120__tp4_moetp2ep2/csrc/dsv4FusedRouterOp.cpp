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

#include "deepseek_v4/v41_flash_1017__sm_120__tp4_moetp2ep2/csrc/dsv4FusedRouter.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

namespace th = torch;
namespace tk = TRTLLM_MV2_NS::kernels::dsv4_fused_router;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

// DeepSeek-V4.1-Flash decode router + top-6 gate in one launch (1 <= M <= 8). Bitwise the same as
// dsv3_router_gemm_op(x, weight.t(), None, float32) followed by gate_forward(logits, bias, ..., topk=6, route_scale,
// is_hash=False). `counter` is a [1] int32 device buffer that is 0 between launches (the kernel resets it); launches
// sharing it must be ordered on one stream. Returns (indices [M,6] int32, weights [M,6] fp32, logits [M,384] fp32).
std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv4_fused_router(th::Tensor const& x, th::Tensor const& weight,
    th::Tensor const& bias, th::Tensor const& counter, double route_scale)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda() && counter.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device() && counter.device() == x.device(),
        "inputs must be on one device");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == th::kBFloat16 && x.size(1) == tk::kHiddenDim && x.is_contiguous(),
        "x must be a contiguous [M,5120] bf16 tensor");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= tk::kMaxTokens, "x must have 1..8 rows");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == th::kBFloat16 && weight.size(0) == tk::kNumExperts
            && weight.size(1) == tk::kHiddenDim && weight.is_contiguous(),
        "weight must be a contiguous [384,5120] bf16 tensor");
    TORCH_CHECK(bias.scalar_type() == th::kFloat32 && bias.numel() == tk::kNumExperts && bias.is_contiguous(),
        "bias must be a contiguous [384] fp32 tensor");
    TORCH_CHECK(counter.scalar_type() == th::kInt32 && counter.numel() >= 1, "counter must be int32");
    for (auto const& t : {x, weight, bias})
    {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "inputs must be 16-byte aligned");
    }

    c10::cuda::CUDAGuard const deviceGuard(x.device());
    int64_t const m = x.size(0);
    auto indices = th::empty({m, tk::kTopK}, x.options().dtype(th::kInt32));
    auto weights = th::empty({m, tk::kTopK}, x.options().dtype(th::kFloat32));
    auto logits = th::empty({m, tk::kNumExperts}, x.options().dtype(th::kFloat32));

    tk::FusedRouterParams params{};
    params.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.weight = static_cast<__nv_bfloat16 const*>(weight.data_ptr());
    params.bias = bias.data_ptr<float>();
    params.logits = logits.data_ptr<float>();
    params.outIndices = indices.data_ptr<int32_t>();
    params.outWeights = weights.data_ptr<float>();
    params.counter = counter.data_ptr<int32_t>();
    params.routeScale = static_cast<float>(route_scale);
    params.numTokens = static_cast<int>(m);
    tk::invokeDsv4FusedRouter(params, at::cuda::getCurrentCUDAStream(x.get_device()));
    return {indices, weights, logits};
}

// Stage 2: dsv4_fused_router plus what FlashInfer's mxfp8_q0_route_triton writes on its decode branch with
// expert_offset = slot_start and zero_buf (route_assign_decode_kernel + _mxfp8_q0_route_direct_kernel), into that
// route's Mxfp8Q0RouteWorkspace tensors: offsets [E+1], token_map / token_weights [M*6], dst_rows / scale_dst_rows
// [M,6], q_out [M*6,5120] e4m3, scale_out [10, padded_rows] int32 (contiguous); zero_buf is zero-filled.
std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv4_fused_router_route(th::Tensor const& x, th::Tensor const& weight,
    th::Tensor const& bias, th::Tensor const& counter, double route_scale, int64_t slot_start, int64_t num_local,
    th::Tensor const& offsets, th::Tensor const& token_map, th::Tensor const& token_weights,
    th::Tensor const& dst_rows, th::Tensor const& scale_dst_rows, th::Tensor const& q_out,
    th::Tensor const& scale_out, th::Tensor const& zero_buf)
{
    int64_t const m = x.size(0);
    int64_t const pairs = m * tk::kTopK;
    TORCH_CHECK(num_local >= 1 && num_local <= 1024, "num_local out of range");
    TORCH_CHECK(offsets.scalar_type() == th::kInt32 && offsets.is_contiguous() && offsets.numel() >= num_local + 1,
        "offsets must be int32 [num_local + 1]");
    TORCH_CHECK(token_map.scalar_type() == th::kInt32 && token_map.is_contiguous() && token_map.numel() >= pairs,
        "token_map must be int32 [M*6]");
    TORCH_CHECK(token_weights.scalar_type() == th::kFloat32 && token_weights.is_contiguous()
            && token_weights.numel() >= pairs,
        "token_weights must be fp32 [M*6]");
    TORCH_CHECK(dst_rows.scalar_type() == th::kInt32 && dst_rows.is_contiguous() && dst_rows.numel() == pairs,
        "dst_rows must be int32 [M,6]");
    TORCH_CHECK(scale_dst_rows.scalar_type() == th::kInt32 && scale_dst_rows.is_contiguous()
            && scale_dst_rows.numel() == pairs,
        "scale_dst_rows must be int32 [M,6]");
    TORCH_CHECK(q_out.element_size() == 1 && q_out.is_contiguous() && q_out.dim() == 2 && q_out.size(0) >= pairs
            && q_out.size(1) == tk::kHiddenDim,
        "q_out must be a contiguous [M*6,5120] 1-byte tensor");
    TORCH_CHECK(scale_out.scalar_type() == th::kInt32 && scale_out.is_contiguous() && scale_out.dim() == 2
            && scale_out.size(0) == tk::kHiddenDim / 512,
        "scale_out must be a contiguous [10, padded_rows] int32 tensor");
    TORCH_CHECK(zero_buf.is_contiguous() && (zero_buf.numel() * zero_buf.element_size()) % 8 == 0
            && reinterpret_cast<uintptr_t>(zero_buf.data_ptr()) % 8 == 0,
        "zero_buf must be contiguous, 8-byte aligned, a multiple of 8 bytes");
    for (auto const& t : {offsets, token_map, token_weights, dst_rows, scale_dst_rows, q_out, scale_out, zero_buf})
    {
        TORCH_CHECK(t.is_cuda() && t.device() == x.device(), "workspace tensors must be on x's device");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 16 == 0, "q_out must be 16-byte aligned");

    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda() && counter.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == th::kBFloat16 && x.size(1) == tk::kHiddenDim && x.is_contiguous(),
        "x must be a contiguous [M,5120] bf16 tensor");
    TORCH_CHECK(m >= 1 && m <= tk::kMaxTokens, "x must have 1..8 rows");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == th::kBFloat16 && weight.size(0) == tk::kNumExperts
            && weight.size(1) == tk::kHiddenDim && weight.is_contiguous(),
        "weight must be a contiguous [384,5120] bf16 tensor");
    TORCH_CHECK(bias.scalar_type() == th::kFloat32 && bias.numel() == tk::kNumExperts && bias.is_contiguous(),
        "bias must be a contiguous [384] fp32 tensor");
    TORCH_CHECK(counter.scalar_type() == th::kInt32 && counter.numel() >= 1, "counter must be int32");
    for (auto const& t : {x, weight, bias})
    {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "inputs must be 16-byte aligned");
    }

    c10::cuda::CUDAGuard const deviceGuard(x.device());
    auto indices = th::empty({m, tk::kTopK}, x.options().dtype(th::kInt32));
    auto weights = th::empty({m, tk::kTopK}, x.options().dtype(th::kFloat32));
    auto logits = th::empty({m, tk::kNumExperts}, x.options().dtype(th::kFloat32));

    tk::FusedRouterParams params{};
    params.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.weight = static_cast<__nv_bfloat16 const*>(weight.data_ptr());
    params.bias = bias.data_ptr<float>();
    params.logits = logits.data_ptr<float>();
    params.outIndices = indices.data_ptr<int32_t>();
    params.outWeights = weights.data_ptr<float>();
    params.counter = counter.data_ptr<int32_t>();
    params.routeScale = static_cast<float>(route_scale);
    params.numTokens = static_cast<int>(m);
    params.slotStart = static_cast<int>(slot_start);
    params.numLocalExperts = static_cast<int>(num_local);
    params.offsets = offsets.data_ptr<int32_t>();
    params.tokenMap = token_map.data_ptr<int32_t>();
    params.tokenWeights = token_weights.data_ptr<float>();
    params.dstRows = dst_rows.data_ptr<int32_t>();
    params.scaleDstRows = scale_dst_rows.data_ptr<int32_t>();
    params.qOut = static_cast<uint8_t*>(q_out.data_ptr());
    params.scaleOut = scale_out.data_ptr<int32_t>();
    params.paddedRows = static_cast<int>(scale_out.size(1));
    params.zeroBuf = static_cast<int64_t*>(zero_buf.data_ptr());
    params.zeroNumel = zero_buf.numel() * zero_buf.element_size() / 8;
    tk::invokeDsv4FusedRouter(params, at::cuda::getCurrentCUDAStream(x.get_device()));
    return {indices, weights, logits};
}

namespace
{

// K1 fold (TRTLLM_DSV4_ROUTER_K1_FOLD): checks the hc_ffn boundary's K1 inputs and fills the fold fields. `out` is the
// router input x: the launch writes layer_input there (mhc_lagged_layer_input's output for reducer_threads).
void fillK1Fold(tk::FusedRouterFoldParams& params, th::Tensor const& out, th::Tensor const& x_prev,
    th::Tensor const& residual, th::Tensor const& post, th::Tensor const& comb, th::Tensor const& pre,
    th::Tensor const& norm_weight, double norm_eps, int64_t reducer_threads)
{
    int64_t const m = out.size(0);
    TORCH_CHECK(m >= 1 && m <= tk::kMaxFoldTokens, "K1 fold: 1..4 rows");
    TORCH_CHECK(x_prev.scalar_type() == th::kBFloat16 && x_prev.is_contiguous() && x_prev.dim() == 2
            && x_prev.size(0) == m && x_prev.size(1) == tk::kHiddenDim,
        "K1 fold: x_prev must be a contiguous [M,5120] bf16 tensor");
    TORCH_CHECK(residual.scalar_type() == th::kBFloat16 && residual.is_contiguous() && residual.numel() == m * 4 * 5120,
        "K1 fold: residual must be a contiguous [M,4,5120] bf16 tensor");
    TORCH_CHECK(post.scalar_type() == th::kFloat32 && post.is_contiguous() && post.numel() == m * 4,
        "K1 fold: post must be a contiguous [M,4] fp32 tensor");
    TORCH_CHECK(comb.scalar_type() == th::kFloat32 && comb.is_contiguous() && comb.numel() == m * 16,
        "K1 fold: comb must be a contiguous [M,4,4] fp32 tensor");
    TORCH_CHECK(pre.scalar_type() == th::kFloat32 && pre.is_contiguous() && pre.numel() == m * 4,
        "K1 fold: pre must be a contiguous [M,4] fp32 tensor");
    TORCH_CHECK(norm_weight.scalar_type() == th::kBFloat16 && norm_weight.is_contiguous()
            && norm_weight.numel() == tk::kHiddenDim,
        "K1 fold: norm_weight must be a contiguous [5120] bf16 tensor");
    TORCH_CHECK(reducer_threads == 96 || reducer_threads == 224 || reducer_threads == 256 || reducer_threads == 480,
        "K1 fold: reducer_threads must be 96/224/256/480");
    for (auto const& t : {x_prev, residual, post, comb, pre, norm_weight})
    {
        TORCH_CHECK(t.is_cuda() && t.device() == out.device(), "K1 fold: inputs must be on x's device");
    }
    for (auto const& t : {x_prev, residual, norm_weight})
    {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "K1 fold: inputs must be 16-byte aligned");
    }
    params.k1X = static_cast<__nv_bfloat16 const*>(x_prev.data_ptr());
    params.k1Residual = static_cast<__nv_bfloat16 const*>(residual.data_ptr());
    params.k1Post = post.data_ptr<float>();
    params.k1Comb = comb.data_ptr<float>();
    params.k1Pre = pre.data_ptr<float>();
    params.k1NormWeight = static_cast<__nv_bfloat16 const*>(norm_weight.data_ptr());
    params.k1Out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.k1NormEps = static_cast<float>(norm_eps);
    params.k1ReducerThreads = static_cast<int>(reducer_threads);
}

} // namespace

// dsv4_fused_router with the hc_ffn mHC K1 folded in: writes layer_input to x (bitwise mhc_lagged_layer_input's for
// reducer_threads) and routes it (bitwise dsv4_fused_router of that x).
std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv4_fused_router_k1fold(th::Tensor const& x, th::Tensor const& x_prev,
    th::Tensor const& residual, th::Tensor const& post, th::Tensor const& comb, th::Tensor const& pre,
    th::Tensor const& norm_weight, double norm_eps, int64_t reducer_threads, th::Tensor const& weight,
    th::Tensor const& bias, th::Tensor const& counter, double route_scale)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda() && counter.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == th::kBFloat16 && x.size(1) == tk::kHiddenDim && x.is_contiguous(),
        "x must be a contiguous [M,5120] bf16 tensor");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == th::kBFloat16 && weight.size(0) == tk::kNumExperts
            && weight.size(1) == tk::kHiddenDim && weight.is_contiguous(),
        "weight must be a contiguous [384,5120] bf16 tensor");
    TORCH_CHECK(bias.scalar_type() == th::kFloat32 && bias.numel() == tk::kNumExperts && bias.is_contiguous(),
        "bias must be a contiguous [384] fp32 tensor");
    TORCH_CHECK(counter.scalar_type() == th::kInt32 && counter.numel() >= 1, "counter must be int32");
    for (auto const& t : {x, weight, bias})
    {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "inputs must be 16-byte aligned");
    }
    c10::cuda::CUDAGuard const deviceGuard(x.device());
    int64_t const m = x.size(0);
    auto indices = th::empty({m, tk::kTopK}, x.options().dtype(th::kInt32));
    auto weights = th::empty({m, tk::kTopK}, x.options().dtype(th::kFloat32));
    auto logits = th::empty({m, tk::kNumExperts}, x.options().dtype(th::kFloat32));

    tk::FusedRouterFoldParams params{};
    params.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.weight = static_cast<__nv_bfloat16 const*>(weight.data_ptr());
    params.bias = bias.data_ptr<float>();
    params.logits = logits.data_ptr<float>();
    params.outIndices = indices.data_ptr<int32_t>();
    params.outWeights = weights.data_ptr<float>();
    params.counter = counter.data_ptr<int32_t>();
    params.routeScale = static_cast<float>(route_scale);
    params.numTokens = static_cast<int>(m);
    fillK1Fold(params, x, x_prev, residual, post, comb, pre, norm_weight, norm_eps, reducer_threads);
    tk::invokeDsv4FusedRouterK1Fold(params, at::cuda::getCurrentCUDAStream(x.get_device()));
    return {indices, weights, logits};
}

// dsv4_fused_router_route with the hc_ffn mHC K1 folded in (see dsv4_fused_router_k1fold).
std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv4_fused_router_route_k1fold(th::Tensor const& x,
    th::Tensor const& x_prev, th::Tensor const& residual, th::Tensor const& post, th::Tensor const& comb,
    th::Tensor const& pre, th::Tensor const& norm_weight, double norm_eps, int64_t reducer_threads,
    th::Tensor const& weight, th::Tensor const& bias, th::Tensor const& counter, double route_scale, int64_t slot_start,
    int64_t num_local, th::Tensor const& offsets, th::Tensor const& token_map, th::Tensor const& token_weights,
    th::Tensor const& dst_rows, th::Tensor const& scale_dst_rows, th::Tensor const& q_out,
    th::Tensor const& scale_out, th::Tensor const& zero_buf)
{
    int64_t const m = x.size(0);
    int64_t const pairs = m * tk::kTopK;
    TORCH_CHECK(num_local >= 1 && num_local <= 1024, "num_local out of range");
    TORCH_CHECK(offsets.scalar_type() == th::kInt32 && offsets.is_contiguous() && offsets.numel() >= num_local + 1,
        "offsets must be int32 [num_local + 1]");
    TORCH_CHECK(token_map.scalar_type() == th::kInt32 && token_map.is_contiguous() && token_map.numel() >= pairs,
        "token_map must be int32 [M*6]");
    TORCH_CHECK(token_weights.scalar_type() == th::kFloat32 && token_weights.is_contiguous()
            && token_weights.numel() >= pairs,
        "token_weights must be fp32 [M*6]");
    TORCH_CHECK(dst_rows.scalar_type() == th::kInt32 && dst_rows.is_contiguous() && dst_rows.numel() == pairs,
        "dst_rows must be int32 [M,6]");
    TORCH_CHECK(scale_dst_rows.scalar_type() == th::kInt32 && scale_dst_rows.is_contiguous()
            && scale_dst_rows.numel() == pairs,
        "scale_dst_rows must be int32 [M,6]");
    TORCH_CHECK(q_out.element_size() == 1 && q_out.is_contiguous() && q_out.dim() == 2 && q_out.size(0) >= pairs
            && q_out.size(1) == tk::kHiddenDim,
        "q_out must be a contiguous [M*6,5120] 1-byte tensor");
    TORCH_CHECK(scale_out.scalar_type() == th::kInt32 && scale_out.is_contiguous() && scale_out.dim() == 2
            && scale_out.size(0) == tk::kHiddenDim / 512,
        "scale_out must be a contiguous [10, padded_rows] int32 tensor");
    TORCH_CHECK(zero_buf.is_contiguous() && (zero_buf.numel() * zero_buf.element_size()) % 8 == 0
            && reinterpret_cast<uintptr_t>(zero_buf.data_ptr()) % 8 == 0,
        "zero_buf must be contiguous, 8-byte aligned, a multiple of 8 bytes");
    for (auto const& t : {offsets, token_map, token_weights, dst_rows, scale_dst_rows, q_out, scale_out, zero_buf})
    {
        TORCH_CHECK(t.is_cuda() && t.device() == x.device(), "workspace tensors must be on x's device");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 16 == 0, "q_out must be 16-byte aligned");
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda() && counter.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == th::kBFloat16 && x.size(1) == tk::kHiddenDim && x.is_contiguous(),
        "x must be a contiguous [M,5120] bf16 tensor");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == th::kBFloat16 && weight.size(0) == tk::kNumExperts
            && weight.size(1) == tk::kHiddenDim && weight.is_contiguous(),
        "weight must be a contiguous [384,5120] bf16 tensor");
    TORCH_CHECK(bias.scalar_type() == th::kFloat32 && bias.numel() == tk::kNumExperts && bias.is_contiguous(),
        "bias must be a contiguous [384] fp32 tensor");
    TORCH_CHECK(counter.scalar_type() == th::kInt32 && counter.numel() >= 1, "counter must be int32");
    for (auto const& t : {x, weight, bias})
    {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "inputs must be 16-byte aligned");
    }

    c10::cuda::CUDAGuard const deviceGuard(x.device());
    auto indices = th::empty({m, tk::kTopK}, x.options().dtype(th::kInt32));
    auto weights = th::empty({m, tk::kTopK}, x.options().dtype(th::kFloat32));
    auto logits = th::empty({m, tk::kNumExperts}, x.options().dtype(th::kFloat32));

    tk::FusedRouterFoldParams params{};
    params.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.weight = static_cast<__nv_bfloat16 const*>(weight.data_ptr());
    params.bias = bias.data_ptr<float>();
    params.logits = logits.data_ptr<float>();
    params.outIndices = indices.data_ptr<int32_t>();
    params.outWeights = weights.data_ptr<float>();
    params.counter = counter.data_ptr<int32_t>();
    params.routeScale = static_cast<float>(route_scale);
    params.numTokens = static_cast<int>(m);
    params.slotStart = static_cast<int>(slot_start);
    params.numLocalExperts = static_cast<int>(num_local);
    params.offsets = offsets.data_ptr<int32_t>();
    params.tokenMap = token_map.data_ptr<int32_t>();
    params.tokenWeights = token_weights.data_ptr<float>();
    params.dstRows = dst_rows.data_ptr<int32_t>();
    params.scaleDstRows = scale_dst_rows.data_ptr<int32_t>();
    params.qOut = static_cast<uint8_t*>(q_out.data_ptr());
    params.scaleOut = scale_out.data_ptr<int32_t>();
    params.paddedRows = static_cast<int>(scale_out.size(1));
    params.zeroBuf = static_cast<int64_t*>(zero_buf.data_ptr());
    params.zeroNumel = zero_buf.numel() * zero_buf.element_size() / 8;
    fillK1Fold(params, x, x_prev, residual, post, comb, pre, norm_weight, norm_eps, reducer_threads);
    tk::invokeDsv4FusedRouterK1Fold(params, at::cuda::getCurrentCUDAStream(x.get_device()));
    return {indices, weights, logits};
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "dsv4_fused_router(Tensor x, Tensor weight, Tensor bias, Tensor(a!) counter, float route_scale) -> (Tensor "
        "indices, Tensor weights, Tensor logits)");
    m.def(
        "dsv4_fused_router_route(Tensor x, Tensor weight, Tensor bias, Tensor(a!) counter, float route_scale, "
        "int slot_start, int num_local, Tensor(b!) offsets, Tensor(c!) token_map, Tensor(d!) token_weights, "
        "Tensor(e!) dst_rows, Tensor(f!) scale_dst_rows, Tensor(g!) q_out, Tensor(h!) scale_out, "
        "Tensor(i!) zero_buf) -> (Tensor indices, Tensor weights, Tensor logits)");
    m.def(
        "dsv4_fused_router_k1fold(Tensor(j!) x, Tensor x_prev, Tensor residual, Tensor post, Tensor comb, "
        "Tensor pre, Tensor norm_weight, float norm_eps, int reducer_threads, Tensor weight, Tensor bias, "
        "Tensor(a!) counter, float route_scale) -> (Tensor indices, Tensor weights, Tensor logits)");
    m.def(
        "dsv4_fused_router_route_k1fold(Tensor(j!) x, Tensor x_prev, Tensor residual, Tensor post, Tensor comb, "
        "Tensor pre, Tensor norm_weight, float norm_eps, int reducer_threads, Tensor weight, Tensor bias, "
        "Tensor(a!) counter, float route_scale, int slot_start, int num_local, Tensor(b!) offsets, "
        "Tensor(c!) token_map, Tensor(d!) token_weights, Tensor(e!) dst_rows, Tensor(f!) scale_dst_rows, "
        "Tensor(g!) q_out, Tensor(h!) scale_out, Tensor(i!) zero_buf) -> (Tensor indices, Tensor weights, "
        "Tensor logits)");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("dsv4_fused_router", &TRTLLM_MV2_NS::torch_ext::dsv4_fused_router);
    m.impl("dsv4_fused_router_route", &TRTLLM_MV2_NS::torch_ext::dsv4_fused_router_route);
    m.impl("dsv4_fused_router_k1fold", &TRTLLM_MV2_NS::torch_ext::dsv4_fused_router_k1fold);
    m.impl("dsv4_fused_router_route_k1fold", &TRTLLM_MV2_NS::torch_ext::dsv4_fused_router_route_k1fold);
}
