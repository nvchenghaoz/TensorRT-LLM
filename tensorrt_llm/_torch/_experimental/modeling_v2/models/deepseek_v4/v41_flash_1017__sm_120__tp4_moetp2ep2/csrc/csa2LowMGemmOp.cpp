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

#include "deepseek_v4/v41_flash_1017__sm_120__tp4_moetp2ep2/csrc/csa2LowMGemm.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! SM120 port for the CSA2 decode input projections: fused MXFP8 activation quantization + low-M MXFP8 GEMM.
//!
//! @param x BF16 [M, K] activation (unit column stride, 16-byte aligned rows). Not read when `xq` is given.
//! @param w E4M3 [N, K] weight, contiguous; @param wsf its 128x4-swizzled UE8M0 scales.
//! @param nSplit 0 for one [M, N] output, else rows [0, nSplit) go to the first and [nSplit, N) to a second
//!        contiguous output.
//! @param xq, xsf optional pre-quantized activation (`mxfp8_quantize(x, True)` outputs).
//! @param xqOut, xsfOut optional buffers receiving the in-kernel quantized activation in that layout.
std::vector<torch::Tensor> csa2Mxfp8LowMGemm(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor const& wsf,
    int64_t warps, int64_t clusterK, int64_t rowTiles, int64_t nSplit, bool pdl, std::optional<torch::Tensor> xq,
    std::optional<torch::Tensor> xsf, std::optional<torch::Tensor> xqOut, std::optional<torch::Tensor> xsfOut,
    std::optional<torch::Tensor> dbgTs, int64_t stages, int64_t debugFlags, std::optional<torch::Tensor> pfW,
    std::optional<torch::Tensor> pfSf, int64_t pfFrac)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 120, "csa2_mxfp8_low_m_gemm requires SM120, got SM", sm);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
        "x must be a 2-D bf16 CUDA tensor with unit column stride");
    TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.is_contiguous() && w.dim() == 2 && w.size(1) == x.size(1),
        "w must be a contiguous e4m3 [N, K] tensor");
    TORCH_CHECK(wsf.scalar_type() == at::kByte || wsf.scalar_type() == at::kFloat8_e8m0fnu,
        "wsf must be uint8 UE8M0 scales");
    int64_t const m = x.size(0);
    int64_t const n = w.size(0);
    int64_t const k = x.size(1);
    int64_t const sfNeed = (n + 127) / 128 * 128 * ((k / 32 + 3) / 4 * 4);
    TORCH_CHECK(wsf.is_contiguous() && wsf.numel() >= sfNeed, "wsf must be the contiguous swizzled buffer of ", sfNeed,
        " bytes");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0, "x rows must be 16-byte aligned");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wsf.data_ptr()) % 16 == 0,
        "w and wsf must be 16-byte aligned");
    TORCH_CHECK(TRTLLM_MV2_NS::kernels::isCsa2Mxfp8LowMGemmSupported(static_cast<int32_t>(warps),
                    static_cast<int32_t>(clusterK), static_cast<int32_t>(rowTiles), static_cast<int32_t>(m),
                    static_cast<int32_t>(stages)),
        "csa2_mxfp8_low_m_gemm: unsupported configuration (warps ", warps, ", cluster_k ", clusterK, ", row_tiles ",
        rowTiles, ", stages ", stages, ") at M=", m);
    TORCH_CHECK(nSplit >= 0 && nSplit < n, "n_split must be in [0, N)");
    int64_t const sfActNeed = (m + 127) / 128 * 128 * ((k / 32 + 3) / 4 * 4);
    TORCH_CHECK(xq.has_value() == xsf.has_value(), "xq and xsf come together");
    if (xq.has_value())
    {
        TORCH_CHECK(xq->is_cuda() && xq->is_contiguous() && xq->numel() == m * k && xq->element_size() == 1
                && reinterpret_cast<uintptr_t>(xq->data_ptr()) % 16 == 0,
            "xq must be a contiguous [M, K] e4m3 tensor");
        TORCH_CHECK(xsf->is_contiguous() && xsf->numel() >= sfActNeed && xsf->element_size() == 1,
            "xsf must be the swizzled activation scales of mxfp8_quantize(x, True)");
    }
    TORCH_CHECK(xqOut.has_value() == xsfOut.has_value(), "xq_out and xsf_out come together");
    if (xqOut.has_value())
    {
        TORCH_CHECK(xqOut->is_cuda() && xqOut->is_contiguous() && xqOut->numel() == m * k
                && xqOut->element_size() == 1 && reinterpret_cast<uintptr_t>(xqOut->data_ptr()) % 16 == 0,
            "xq_out must be a contiguous [M, K] byte tensor");
        TORCH_CHECK(xsfOut->is_contiguous() && xsfOut->numel() >= sfActNeed && xsfOut->element_size() == 1,
            "xsf_out is too small");
    }
    auto pfRange = [&](std::optional<torch::Tensor> const& t) -> std::pair<uint8_t const*, int64_t>
    {
        if (!t.has_value())
        {
            return {nullptr, 0};
        }
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->get_device() == x.get_device()
                && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
            "pf_w / pf_sf must be contiguous 16-byte aligned CUDA tensors on x's device");
        return {reinterpret_cast<uint8_t const*>(t->data_ptr()), t->numel() * t->element_size()};
    };
    auto const pf0 = pfRange(pfW);
    auto const pf1 = pfRange(pfSf);
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    int64_t const n1 = nSplit == 0 ? n : nSplit;
    auto out = torch::empty({m, n1}, x.options());
    std::vector<torch::Tensor> outs{out};
    __nv_bfloat16* out2Ptr = nullptr;
    int32_t ldo2 = 0;
    if (nSplit != 0)
    {
        auto out2 = torch::empty({m, n - nSplit}, x.options());
        out2Ptr = reinterpret_cast<__nv_bfloat16*>(out2.data_ptr());
        ldo2 = static_cast<int32_t>(out2.stride(0));
        outs.push_back(out2);
    }
    TRTLLM_MV2_NS::kernels::invokeCsa2Mxfp8LowMGemm(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<int32_t>(x.stride(0)), static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
        reinterpret_cast<uint8_t const*>(w.data_ptr()), reinterpret_cast<uint8_t const*>(wsf.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int32_t>(out.stride(0)), out2Ptr, ldo2,
        static_cast<int32_t>(nSplit == 0 ? n : nSplit),
        xq.has_value() ? reinterpret_cast<uint8_t const*>(xq->data_ptr()) : nullptr,
        xsf.has_value() ? reinterpret_cast<uint8_t const*>(xsf->data_ptr()) : nullptr,
        xqOut.has_value() ? reinterpret_cast<uint8_t*>(xqOut->data_ptr()) : nullptr,
        xsfOut.has_value() ? reinterpret_cast<uint8_t*>(xsfOut->data_ptr()) : nullptr, static_cast<int32_t>(warps),
        static_cast<int32_t>(clusterK), static_cast<int32_t>(rowTiles), pdl && tensorrt_llm::common::getEnvEnablePDL(),
        at::cuda::getCurrentCUDAStream(x.get_device()),
        dbgTs.has_value() ? reinterpret_cast<unsigned long long*>(dbgTs->data_ptr()) : nullptr,
        static_cast<int32_t>(stages), static_cast<int32_t>(debugFlags), pf0.first, pf0.second, pf1.first, pf1.second,
        static_cast<int32_t>(pfFrac));
    return outs;
}

//! SM120 CSA2 decode attention-input chain step B: q_norm (RMSNorm, bit for bit as FlashInfer rmsnorm_cute) +
//! MXFP8 quantization + low-M GEMM in one launch.
//!
//! @param x BF16 [M, K] raw (un-normalized) rows, unit column stride, 16-byte aligned rows (e.g. a column slice of the
//!        fused wq_a||wkv output).
//! @param normW BF16 [K] RMSNorm weight; @param eps its epsilon.
//! @return {qr [M, K] BF16 contiguous, out [M, n1]} or, with nSplit > 0, {qr, out [M, nSplit], out2 [M, N - nSplit]}.
std::vector<torch::Tensor> csa2QnormLowMGemm(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor const& wsf,
    torch::Tensor const& normW, double eps, int64_t warps, int64_t clusterK, int64_t rowTiles, int64_t nSplit,
    bool pdl, int64_t debugFlags, std::optional<torch::Tensor> xqOut, std::optional<torch::Tensor> xsfOut)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sm == 120, "csa2_qnorm_low_m_gemm requires SM120, got SM", sm);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
        "x must be a 2-D bf16 CUDA tensor with unit column stride");
    TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.is_contiguous() && w.dim() == 2 && w.size(1) == x.size(1),
        "w must be a contiguous e4m3 [N, K] tensor");
    TORCH_CHECK(wsf.scalar_type() == at::kByte || wsf.scalar_type() == at::kFloat8_e8m0fnu,
        "wsf must be uint8 UE8M0 scales");
    TORCH_CHECK(normW.scalar_type() == at::kBFloat16 && normW.is_contiguous() && normW.numel() == x.size(1)
            && reinterpret_cast<uintptr_t>(normW.data_ptr()) % 16 == 0,
        "norm_w must be a contiguous 16-byte aligned bf16 [K] tensor");
    int64_t const m = x.size(0);
    int64_t const n = w.size(0);
    int64_t const k = x.size(1);
    int64_t const sfNeed = (n + 127) / 128 * 128 * ((k / 32 + 3) / 4 * 4);
    TORCH_CHECK(wsf.is_contiguous() && wsf.numel() >= sfNeed, "wsf must be the contiguous swizzled buffer of ", sfNeed,
        " bytes");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0, "x rows must be 16-byte aligned");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wsf.data_ptr()) % 16 == 0,
        "w and wsf must be 16-byte aligned");
    TORCH_CHECK(TRTLLM_MV2_NS::kernels::isCsa2QnormLowMGemmSupported(static_cast<int32_t>(warps),
                    static_cast<int32_t>(clusterK), static_cast<int32_t>(rowTiles), static_cast<int32_t>(m),
                    static_cast<int32_t>(k)),
        "csa2_qnorm_low_m_gemm: unsupported configuration (warps ", warps, ", cluster_k ", clusterK, ", row_tiles ",
        rowTiles, ") at M=", m, " K=", k);
    TORCH_CHECK(nSplit >= 0 && nSplit < n, "n_split must be in [0, N)");
    TORCH_CHECK(xqOut.has_value() == xsfOut.has_value(), "xq_out and xsf_out come together");
    if (xqOut.has_value())
    {
        int64_t const sfActNeed = (m + 127) / 128 * 128 * ((k / 32 + 3) / 4 * 4);
        TORCH_CHECK(xqOut->is_cuda() && xqOut->is_contiguous() && xqOut->numel() == m * k
                && xqOut->element_size() == 1 && reinterpret_cast<uintptr_t>(xqOut->data_ptr()) % 16 == 0,
            "xq_out must be a contiguous [M, K] byte tensor");
        TORCH_CHECK(xsfOut->is_contiguous() && xsfOut->numel() >= sfActNeed && xsfOut->element_size() == 1,
            "xsf_out is too small");
    }
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    auto qr = torch::empty({m, k}, x.options());
    int64_t const n1 = nSplit == 0 ? n : nSplit;
    auto out = torch::empty({m, n1}, x.options());
    std::vector<torch::Tensor> outs{qr, out};
    __nv_bfloat16* out2Ptr = nullptr;
    int32_t ldo2 = 0;
    if (nSplit != 0)
    {
        auto out2 = torch::empty({m, n - nSplit}, x.options());
        out2Ptr = reinterpret_cast<__nv_bfloat16*>(out2.data_ptr());
        ldo2 = static_cast<int32_t>(out2.stride(0));
        outs.push_back(out2);
    }
    TRTLLM_MV2_NS::kernels::invokeCsa2QnormLowMGemm(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<int32_t>(x.stride(0)), static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
        reinterpret_cast<uint8_t const*>(w.data_ptr()), reinterpret_cast<uint8_t const*>(wsf.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(normW.data_ptr()), static_cast<float>(eps),
        reinterpret_cast<__nv_bfloat16*>(qr.data_ptr()), static_cast<int32_t>(qr.stride(0)),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int32_t>(out.stride(0)), out2Ptr, ldo2,
        static_cast<int32_t>(nSplit == 0 ? n : nSplit), static_cast<int32_t>(warps), static_cast<int32_t>(clusterK),
        static_cast<int32_t>(rowTiles), pdl && tensorrt_llm::common::getEnvEnablePDL(),
        at::cuda::getCurrentCUDAStream(x.get_device()), static_cast<int32_t>(debugFlags),
        xqOut.has_value() ? reinterpret_cast<uint8_t*>(xqOut->data_ptr()) : nullptr,
        xsfOut.has_value() ? reinterpret_cast<uint8_t*>(xsfOut->data_ptr()) : nullptr);
    return outs;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "csa2_mxfp8_low_m_gemm(Tensor x, Tensor w, Tensor wsf, int warps, int cluster_k, int row_tiles, int n_split, "
        "bool pdl, Tensor? xq=None, Tensor? xsf=None, Tensor(a!)? xq_out=None, Tensor(b!)? xsf_out=None, "
        "Tensor(c!)? dbg_ts=None, int stages=1, int debug_flags=0, Tensor? pf_w=None, Tensor? pf_sf=None, "
        "int pf_frac=256) -> Tensor[]");
    m.def(
        "csa2_qnorm_low_m_gemm(Tensor x, Tensor w, Tensor wsf, Tensor norm_w, float eps, int warps, int cluster_k, "
        "int row_tiles, int n_split, bool pdl, int debug_flags=0, Tensor(a!)? xq_out=None, Tensor(b!)? xsf_out=None) "
        "-> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("csa2_mxfp8_low_m_gemm", &TRTLLM_MV2_NS::torch_ext::csa2Mxfp8LowMGemm);
    m.impl("csa2_qnorm_low_m_gemm", &TRTLLM_MV2_NS::torch_ext::csa2QnormLowMGemm);
}
