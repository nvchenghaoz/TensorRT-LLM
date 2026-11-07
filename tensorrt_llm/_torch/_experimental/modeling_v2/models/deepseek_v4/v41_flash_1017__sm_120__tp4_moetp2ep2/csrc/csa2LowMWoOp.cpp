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
#include "tensorrt_llm/common/envUtils.h"
#include "deepseek_v4/v41_flash_1017__sm_120__tp4_moetp2ep2/csrc/csa2LowMWo.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! SM120 CSA2 decode wo_a: BF16 inverse RoPE + 1x128 UE8M0 quant + grouped low-M E4M3 GEMM in one launch.
//!
//! @param o BF16 [M, groups * groupWidth] attention output (unit column stride, 16-byte aligned rows). Not modified.
//! @param positions int32 or int64 [M] token positions.
//! @param cosSin FP32 rotary table, row `pos` = cos[(headDim - nope) / 2] then sin[(headDim - nope) / 2].
//! @param w E4M3 [groups, rank, groupWidth] (or [groups * rank, groupWidth]) contiguous.
//! @param wsc UE8M0 (uint8) [groups, rank / 128, groupWidth / 128] block-scale exponents.
//! @return BF16 [M, groups * rank], out[m, g * rank + r].
torch::Tensor csa2LowMWoA(torch::Tensor const& o, torch::Tensor const& positions, torch::Tensor const& cosSin,
    torch::Tensor const& w, torch::Tensor const& wsc, int64_t groups, int64_t headDim, int64_t nopeDim, int64_t warps,
    int64_t clusterK, int64_t rowTiles, bool pdl, std::optional<torch::Tensor> xqDbg,
    std::optional<torch::Tensor> xsDbg, int64_t debugFlags)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 120, "csa2_lowm_wo_a requires SM120, got SM", sm);
    TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.dim() == 2 && o.stride(1) == 1,
        "o must be a 2-D bf16 CUDA tensor with unit column stride");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0 && o.stride(0) % 8 == 0,
        "o rows must be 16-byte aligned");
    int64_t const m = o.size(0);
    TORCH_CHECK(groups >= 1 && o.size(1) % groups == 0, "o width must be groups * groupWidth");
    int64_t const gw = o.size(1) / groups;
    TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.is_contiguous() && w.numel() % (groups * gw) == 0,
        "w must be a contiguous e4m3 [groups, rank, groupWidth] tensor");
    int64_t const rank = w.numel() / (groups * gw);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "w must be 16-byte aligned");
    TORCH_CHECK((wsc.scalar_type() == at::kByte || wsc.scalar_type() == at::kFloat8_e8m0fnu) && wsc.is_contiguous()
            && wsc.numel() == groups * (rank / 128) * (gw / 128),
        "wsc must be contiguous UE8M0 [groups, rank / 128, groupWidth / 128]");
    TORCH_CHECK(positions.is_cuda() && positions.numel() == m && positions.is_contiguous()
            && (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong),
        "positions must be contiguous int32/int64 [M]");
    int64_t const half = (headDim - nopeDim) / 2;
    TORCH_CHECK(cosSin.is_cuda() && cosSin.scalar_type() == at::kFloat && cosSin.stride(-1) == 1
            && cosSin.dim() >= 2 && cosSin.numel() / cosSin.size(0) == 2 * half && cosSin.stride(0) % 4 == 0
            && reinterpret_cast<uintptr_t>(cosSin.data_ptr()) % 16 == 0,
        "cos_sin must be an fp32 [P, 2 * half_rope] table with 16-byte aligned rows");
    TORCH_CHECK(TRTLLM_MV2_NS::kernels::isCsa2LowMWoASupported(static_cast<int32_t>(warps),
                    static_cast<int32_t>(clusterK), static_cast<int32_t>(rowTiles), static_cast<int32_t>(m),
                    static_cast<int32_t>(groups), static_cast<int32_t>(rank), static_cast<int32_t>(gw),
                    static_cast<int32_t>(headDim), static_cast<int32_t>(nopeDim)),
        "csa2_lowm_wo_a: unsupported configuration (warps ", warps, ", cluster_k ", clusterK, ", row_tiles ", rowTiles,
        ") at M=", m);
    TORCH_CHECK(xqDbg.has_value() == xsDbg.has_value(), "xq_dbg and xs_dbg come together");
    if (xqDbg.has_value())
    {
        TORCH_CHECK(xqDbg->is_contiguous() && xqDbg->numel() == m * groups * gw && xqDbg->element_size() == 1
                && reinterpret_cast<uintptr_t>(xqDbg->data_ptr()) % 16 == 0,
            "xq_dbg must be a contiguous [M, groups, groupWidth] byte tensor");
        TORCH_CHECK(xsDbg->is_contiguous() && xsDbg->numel() == m * groups * (gw / 128) && xsDbg->element_size() == 1,
            "xs_dbg must be a contiguous [M, groups, groupWidth / 128] byte tensor");
    }
    c10::cuda::CUDAGuard const deviceGuard{o.device()};
    auto out = torch::empty({m, groups * rank}, o.options());
    TRTLLM_MV2_NS::kernels::invokeCsa2LowMWoA(reinterpret_cast<__nv_bfloat16 const*>(o.data_ptr()),
        static_cast<int32_t>(o.stride(0)), positions.data_ptr(), positions.scalar_type() == at::kLong,
        cosSin.data_ptr<float>(), static_cast<int32_t>(cosSin.stride(0)),
        reinterpret_cast<uint8_t const*>(w.data_ptr()), reinterpret_cast<uint8_t const*>(wsc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int32_t>(out.stride(0)),
        static_cast<int32_t>(m), static_cast<int32_t>(groups), static_cast<int32_t>(rank), static_cast<int32_t>(gw),
        static_cast<int32_t>(headDim), static_cast<int32_t>(nopeDim), static_cast<int32_t>(warps),
        static_cast<int32_t>(clusterK), static_cast<int32_t>(rowTiles), pdl && tensorrt_llm::common::getEnvEnablePDL(),
        at::cuda::getCurrentCUDAStream(o.get_device()),
        xqDbg.has_value() ? reinterpret_cast<uint8_t*>(xqDbg->data_ptr()) : nullptr,
        xsDbg.has_value() ? reinterpret_cast<uint8_t*>(xsDbg->data_ptr()) : nullptr,
        static_cast<int32_t>(debugFlags));
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "csa2_lowm_wo_a(Tensor o, Tensor positions, Tensor cos_sin, Tensor w, Tensor wsc, int groups, int head_dim, "
        "int nope_dim, int warps, int cluster_k, int row_tiles, bool pdl, Tensor(a!)? xq_dbg=None, "
        "Tensor(b!)? xs_dbg=None, int debug_flags=0) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("csa2_lowm_wo_a", &TRTLLM_MV2_NS::torch_ext::csa2LowMWoA);
}
