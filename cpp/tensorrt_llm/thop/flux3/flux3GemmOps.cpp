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

// Torch bindings of the FLUX.3 (flux3_klein_nvfp4__sm_120__tp1) denoiser GEMM kernels in
// kernels/flux3/gemm. Their fakes for torch.compile live with the target's op entries
// (ops/gemm/_cpp_fakes.py).

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/flux3/gemm/flux3AdalnNvfp4Sm120.h"
#include "tensorrt_llm/kernels/flux3/gemm/flux3Fp8RowwiseGemmSm120.h"
#include "tensorrt_llm/kernels/flux3/gemm/flux3GateUpSwigluNvfp4Sm120.h"
#include "tensorrt_llm/kernels/flux3/gemm/flux3PairResidualSm120.h"
#include "tensorrt_llm/kernels/flux3/gemm/flux3QkvEpiFp8Sm120.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace kf = tensorrt_llm::kernels::flux3;

bool aligned(at::Tensor const& t, uintptr_t bytes)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}

int64_t swizzledSfBytes(int64_t rows, int64_t k)
{
    return (rows + 127) / 128 * 128 * ((k / 16 + 3) / 4 * 4);
}

void checkSm120(char const* op)
{
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() / 10 == 12, op, " requires SM12x");
}

at::Tensor emptyBytes(int64_t bytes, at::Device device)
{
    return at::detail::empty_cuda({std::max<int64_t>(bytes, 16)}, at::ScalarType::Byte, device, std::nullopt);
}

void checkNvfp4Operand(at::Tensor const& act, at::Tensor const& weight, at::Tensor const& actSf,
    at::Tensor const& weightSf, char const* what)
{
    CHECK_INPUT(act, FLOAT4_E2M1X2);
    CHECK_INPUT(weight, FLOAT4_E2M1X2);
    CHECK_INPUT(actSf, SF_DTYPE);
    CHECK_INPUT(weightSf, SF_DTYPE);
    TORCH_CHECK(act.dim() == 2 && weight.dim() == 2, what, ": activation and weight must be matrices");
    TORCH_CHECK(act.size(1) == weight.size(1), what, ": K mismatch");
    int64_t const k = act.size(1) * 2;
    TORCH_CHECK(actSf.numel() >= swizzledSfBytes(act.size(0), k), what,
        ": activation scales shorter than the padded swizzled rectangle");
    TORCH_CHECK(weightSf.numel() >= swizzledSfBytes(weight.size(0), k), what,
        ": weight scales shorter than the padded swizzled rectangle");
}

void checkScalar(at::Tensor const& t, char const* what)
{
    CHECK_INPUT(t, at::ScalarType::Float);
    TORCH_CHECK(t.numel() == 1, what, " must hold exactly one element");
}

} // namespace

// NVFP4 QKV GEMM (three planes of width W = weight.size(0) / 3) with the RMSNorm + RoPE + static-scale
// E4M3 epilogue for q/k, then the V quantize. Returns (qkv8 [3, M, W] float8_e4m3fn, scales [8] fp32,
// gemm [M, 3W] bf16 with only its v plane defined). keep_v = false lets the V quantize drop the v
// plane's L2 lines (its contents are then undefined).
std::tuple<at::Tensor, at::Tensor, at::Tensor> flux3_qkv_epi_fp8(at::Tensor const& act, at::Tensor const& weight,
    at::Tensor const& actSf, at::Tensor const& weightSf, at::Tensor const& qkvAlpha, at::Tensor const& qNorm,
    at::Tensor const& kNorm, at::Tensor const& ropeTab, at::Tensor const& ropeIdx, double smScale, bool keepV)
{
    checkSm120("flux3_qkv_epi_fp8");
    checkNvfp4Operand(act, weight, actSf, weightSf, "flux3_qkv_epi_fp8");
    CHECK_INPUT(qkvAlpha, at::ScalarType::Float);
    TORCH_CHECK(qkvAlpha.numel() == 3, "qkv_alpha must hold the three plane alphas");
    CHECK_INPUT(qNorm, at::ScalarType::BFloat16);
    CHECK_INPUT(kNorm, at::ScalarType::BFloat16);
    TORCH_CHECK(qNorm.numel() == 128 && kNorm.numel() == 128, "head_dim must be 128");
    CHECK_INPUT(ropeTab, at::ScalarType::Float);
    CHECK_INPUT(ropeIdx, at::ScalarType::Int);
    TORCH_CHECK(ropeTab.dim() == 2 && ropeTab.size(1) == 32, "rope_tab must be [entries, 32]");
    TORCH_CHECK(ropeIdx.dim() == 2 && ropeIdx.size(1) == 4, "rope_idx must be [rows, 4]");
    int64_t const m = act.size(0);
    int64_t const k = act.size(1) * 2;
    TORCH_CHECK(weight.size(0) % 3 == 0, "weight rows must be three planes");
    int64_t const n = weight.size(0) / 3;
    TORCH_CHECK(n % 128 == 0, "plane width must be a multiple of 128");
    TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(weightSf.numel() / 3 == n * (k / 16), "weight_scale must hold the three planes' scales");
    TORCH_CHECK(ropeIdx.size(0) >= m, "rope_idx rows < M");
    for (auto const* t : {&qNorm, &kNorm, &ropeTab, &ropeIdx})
    {
        TORCH_CHECK(aligned(*t, 16), "flux3_qkv_epi_fp8: 16-B alignment required");
    }

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(act));
    auto const device = act.device();
    at::Tensor qkv8 = at::detail::empty_cuda({3, m, n}, at::ScalarType::Float8_e4m3fn, device, std::nullopt);
    at::Tensor scales = at::detail::empty_cuda({8}, at::ScalarType::Float, device, std::nullopt);
    at::Tensor gemm = at::detail::empty_cuda({m, 3 * n}, at::ScalarType::BFloat16, device, std::nullopt);
    if (m == 0)
    {
        return {qkv8, scales, gemm};
    }
    auto stream = at::cuda::getCurrentCUDAStream(act.get_device());
    auto* qkv8Bytes = static_cast<uint8_t*>(qkv8.data_ptr());
    auto* vPlane = static_cast<char*>(gemm.data_ptr()) + 2 * m * n * 2;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(vPlane) % 128 == 0, "the v plane must be 128-B aligned");
    int64_t const nPart = (m + 127) / 128 * (n / 128);
    at::Tensor parts = at::detail::empty_cuda({nPart}, at::ScalarType::Float, device, std::nullopt);

    int const mi = static_cast<int>(m);
    int const ni = static_cast<int>(n);
    int const ki = static_cast<int>(k);
    size_t const wsBytes = kf::qkvEpiFp8GemmSm120(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, mi, ni, ki, nullptr, 0, stream);
    at::Tensor workspace = emptyBytes(static_cast<int64_t>(wsBytes), device);
    kf::qkvEpiFp8GemmSm120(act.const_data_ptr(), weight.const_data_ptr(), actSf.const_data_ptr(),
        weightSf.const_data_ptr(), qkvAlpha.const_data_ptr<float>(), qNorm.const_data_ptr(), kNorm.const_data_ptr(),
        ropeTab.const_data_ptr<float>(), ropeIdx.const_data_ptr<int>(), qkv8Bytes, vPlane, parts.data_ptr<float>(), mi,
        ni, ki, static_cast<char*>(workspace.data_ptr()), static_cast<size_t>(workspace.numel()), stream);

    // One wave of 2 CTAs (512 threads each) per SM for the V quantize.
    static thread_local int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int64_t const chunks = m * n / 8;
    int const grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(2 * smCount, (chunks + 511) / 512)));
    kf::vQuantSm120(vPlane, qkv8Bytes + 2 * m * n, parts.data_ptr<float>(), static_cast<int>(nPart),
        qNorm.const_data_ptr(), kNorm.const_data_ptr(), scales.data_ptr<float>(), m * n, static_cast<float>(smScale),
        grid, !keepV, stream);
    return {qkv8, scales, gemm};
}

// gate_up NVFP4 GEMM ([value, gate]-paired, v2w-permuted weight rows) + SwiGLU + NVFP4 quantize with
// down_proj's activation global scale out_scale, into out ([ceil128(M), N / 4] uint8) and out_sf
// (ceil128(M) * N / 32 uint8, 128x4-swizzled); every row of both, padding included, is written.
void flux3_gate_up_swiglu_nvfp4(at::Tensor const& act, at::Tensor const& weight, at::Tensor const& actSf,
    at::Tensor const& weightSf, at::Tensor const& alpha, at::Tensor const& outScale, at::Tensor& out,
    at::Tensor& outSf)
{
    checkSm120("flux3_gate_up_swiglu_nvfp4");
    checkNvfp4Operand(act, weight, actSf, weightSf, "flux3_gate_up_swiglu_nvfp4");
    checkScalar(alpha, "alpha");
    checkScalar(outScale, "out_scale");
    CHECK_INPUT(out, at::ScalarType::Byte);
    CHECK_INPUT(outSf, at::ScalarType::Byte);
    int64_t const m = act.size(0);
    int64_t const n = weight.size(0);
    int64_t const k = act.size(1) * 2;
    TORCH_CHECK(n % 128 == 0 && k % 128 == 0, "N and K must be multiples of 128");
    TORCH_CHECK((n / 2 / 16) % 4 == 0, "N / 32 must be a multiple of 4");
    int64_t const rowsPadded = (m + 127) / 128 * 128;
    TORCH_CHECK(out.dim() == 2 && out.size(0) == rowsPadded && out.size(1) == n / 4,
        "out must be [ceil128(M), N / 4]");
    TORCH_CHECK(outSf.numel() == rowsPadded * (n / 32), "out_sf must hold ceil128(M) * N / 32 bytes");
    TORCH_CHECK(aligned(out, 16) && (n / 4) % 16 == 0, "out must be 16-B aligned with 16-B rows");
    if (m == 0)
    {
        return;
    }
    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(act));
    auto stream = at::cuda::getCurrentCUDAStream(act.get_device());
    kf::gateUpSwigluNvfp4Sm120(act.const_data_ptr(), weight.const_data_ptr(), actSf.const_data_ptr(),
        weightSf.const_data_ptr(), alpha.const_data_ptr<float>(), outScale.const_data_ptr<float>(),
        out.data_ptr<uint8_t>(), outSf.data_ptr<uint8_t>(), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), static_cast<int>(rowsPadded), stream);
}

// out = x + gate[gate_index] * (attn_out(attn) + down(down_act)) as [M, N] bf16, one launch. down_act /
// attn_act: [M, K / 2] NVFP4 activations with their swizzled scales; gate: bf16 [R, N] row table;
// gate_index: uint8 or bool [M] (row 1 where true), or None (row 0). workspace: uint8, at least
// kPairResidualWorkspaceBytes, zero-filled once before its first use (outside graph capture) and then
// owned by this op. t256 / along_n pick the tile instance and the AlongN raster (bitwise equal).
at::Tensor flux3_nvfp4_pair_residual(at::Tensor const& downAct, at::Tensor const& downWeight,
    at::Tensor const& downActSf, at::Tensor const& downWeightSf, at::Tensor const& downAlpha,
    at::Tensor const& attnAct, at::Tensor const& attnWeight, at::Tensor const& attnActSf,
    at::Tensor const& attnWeightSf, at::Tensor const& attnAlpha, at::Tensor const& x, at::Tensor const& gate,
    std::optional<at::Tensor> const& gateIndex, at::Tensor& workspace, bool t256, bool alongN)
{
    checkSm120("flux3_nvfp4_pair_residual");
    checkNvfp4Operand(downAct, downWeight, downActSf, downWeightSf, "flux3_nvfp4_pair_residual (down)");
    checkNvfp4Operand(attnAct, attnWeight, attnActSf, attnWeightSf, "flux3_nvfp4_pair_residual (attn_out)");
    checkScalar(downAlpha, "down_alpha");
    checkScalar(attnAlpha, "attn_alpha");
    int64_t const m = downAct.size(0);
    int64_t const n = downWeight.size(0);
    TORCH_CHECK(attnAct.size(0) == m, "both groups must have the same M");
    TORCH_CHECK(attnWeight.size(0) == n, "both groups must have the same N");
    TORCH_CHECK(n % 128 == 0, "N must be a multiple of 128");
    TORCH_CHECK(downAct.size(1) * 2 % 128 == 0 && attnAct.size(1) * 2 % 128 == 0, "K must be a multiple of 128");
    CHECK_INPUT(x, at::ScalarType::BFloat16);
    CHECK_INPUT(gate, at::ScalarType::BFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == m && x.size(1) == n, "x must be [M, N]");
    TORCH_CHECK(gate.dim() == 2 && gate.size(0) >= 1 && gate.size(1) == n, "gate must be [R, N]");
    TORCH_CHECK(aligned(x, 16) && aligned(gate, 16), "x and gate must be 16-B aligned");
    uint8_t const* index = nullptr;
    if (gateIndex.has_value())
    {
        auto const& gi = gateIndex.value();
        TORCH_CHECK(gi.is_cuda() && gi.is_contiguous(), "gate_index must be a contiguous CUDA tensor");
        TORCH_CHECK(gi.scalar_type() == at::ScalarType::Byte || gi.scalar_type() == at::ScalarType::Bool,
            "gate_index must be uint8 or bool");
        TORCH_CHECK(gi.numel() >= m, "gate_index needs one entry per row");
        TORCH_CHECK(gate.size(0) <= 255, "a gate table indexed by uint8 has at most 255 rows");
        index = static_cast<uint8_t const*>(gi.data_ptr());
    }
    CHECK_INPUT(workspace, at::ScalarType::Byte);
    TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= kf::kPairResidualWorkspaceBytes,
        "workspace must hold kPairResidualWorkspaceBytes");

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(downAct));
    auto const device = downAct.device();
    at::Tensor mlpY = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, device, std::nullopt);
    at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, device, std::nullopt);
    if (m == 0)
    {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream(downAct.get_device());
    kf::nvfp4PairResidualSm120(downAct.const_data_ptr(), downWeight.const_data_ptr(), downActSf.const_data_ptr(),
        downWeightSf.const_data_ptr(), downAlpha.const_data_ptr<float>(), mlpY.data_ptr(), attnAct.const_data_ptr(),
        attnWeight.const_data_ptr(), attnActSf.const_data_ptr(), attnWeightSf.const_data_ptr(),
        attnAlpha.const_data_ptr<float>(), x.const_data_ptr(), gate.const_data_ptr(), index, out.data_ptr(),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(downAct.size(1) * 2),
        static_cast<int>(attnAct.size(1) * 2), workspace.data_ptr(), static_cast<size_t>(workspace.numel()), t256,
        alongN, stream);
    return out;
}

// bf16 [M, N] = BF16(sw * (sx * (a @ b^T))): a E4M3 [M, K], b E4M3 [N, K], sx fp32 [M], sw fp32 [N].
// arm: 1 = Pingpong 128x128x64, 3 = Cooperative 128x256x64, 4 = Cooperative 256x128x64.
at::Tensor flux3_fp8r_gemm(
    at::Tensor const& a, at::Tensor const& b, at::Tensor const& sx, at::Tensor const& sw, int64_t arm)
{
    checkSm120("flux3_fp8r_gemm");
    CHECK_INPUT(a, at::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(b, at::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(sx, at::ScalarType::Float);
    CHECK_INPUT(sw, at::ScalarType::Float);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be matrices");
    int64_t const m = a.size(0);
    int64_t const k = a.size(1);
    int64_t const n = b.size(0);
    TORCH_CHECK(b.size(1) == k, "K mismatch");
    TORCH_CHECK(sx.numel() == m && sw.numel() == n, "sx must hold M and sw N scales");
    TORCH_CHECK(k % 16 == 0 && n % 8 == 0, "K must be a multiple of 16 and N of 8");
    TORCH_CHECK(arm == 1 || arm == 3 || arm == 4, "arm must be 1, 3 or 4");
    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(a));
    at::Tensor d = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, a.device(), std::nullopt);
    if (m == 0)
    {
        return d;
    }
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    int const mi = static_cast<int>(m);
    int const ni = static_cast<int>(n);
    int const ki = static_cast<int>(k);
    int const armi = static_cast<int>(arm);
    size_t const wsBytes
        = kf::fp8RowwiseGemmSm120(nullptr, nullptr, nullptr, nullptr, nullptr, mi, ni, ki, armi, nullptr, 0, stream);
    at::Tensor workspace = emptyBytes(static_cast<int64_t>(wsBytes), a.device());
    kf::fp8RowwiseGemmSm120(a.const_data_ptr(), b.const_data_ptr(), sx.const_data_ptr<float>(),
        sw.const_data_ptr<float>(), d.data_ptr(), mi, ni, ki, armi, static_cast<char*>(workspace.data_ptr()),
        static_cast<size_t>(workspace.numel()), stream);
    return d;
}

// LayerNorm (identity affine) + adaLN modulate + NVFP4 quantize of x [..., 3072] bf16 with the warp-
// per-row kernel. scale / shift: [1, R, D] bf16 tables (unit last stride, 16-B aligned rows); select:
// [M] bool (row 1 where true) or uint8 (row index), or None (row 0). Returns (fp4 [M, D / 2] uint8,
// sf [ceil128(M) * D / 16] uint8, 128x4-swizzled, padding rows zero).
std::tuple<at::Tensor, at::Tensor> flux3_adaln_nvfp4_warp(at::Tensor const& x, std::optional<at::Tensor> const& select,
    at::Tensor const& scale, at::Tensor const& shift, at::Tensor const& globalScale, double eps)
{
    checkSm120("flux3_adaln_nvfp4_warp");
    CHECK_INPUT(x, at::ScalarType::BFloat16);
    TORCH_CHECK(x.size(-1) == kf::kAdalnHidden, "the adaLN head serves D = 3072");
    TORCH_CHECK(aligned(x, 16), "x must be 16-B aligned");
    checkScalar(globalScale, "global_scale");
    for (auto const* t : {&scale, &shift})
    {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::ScalarType::BFloat16, "scale / shift must be CUDA bf16");
        TORCH_CHECK(t->dim() == 3 && t->size(0) == 1 && t->size(2) == kf::kAdalnHidden && t->stride(2) == 1,
            "scale / shift must be [1, R, D] with a unit last stride");
        TORCH_CHECK(aligned(*t, 16), "scale / shift must be 16-B aligned");
    }
    int64_t const m = x.numel() / kf::kAdalnHidden;
    uint8_t const* sel = nullptr;
    bool selectIsIndex = false;
    if (select.has_value())
    {
        auto const& s = select.value();
        TORCH_CHECK(s.is_cuda() && s.is_contiguous() && s.numel() == m, "select must hold one entry per row");
        TORCH_CHECK(s.scalar_type() == at::ScalarType::Bool || s.scalar_type() == at::ScalarType::Byte,
            "select must be bool or uint8");
        TORCH_CHECK((scale.stride(1) * 2) % 16 == 0 && (shift.stride(1) * 2) % 16 == 0,
            "table rows must be 16-B aligned");
        selectIsIndex = s.scalar_type() == at::ScalarType::Byte;
        sel = static_cast<uint8_t const*>(s.data_ptr());
    }
    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(x));
    auto const device = x.device();
    int64_t const padded = (m + 127) / 128 * 128;
    at::Tensor fp4 = at::detail::empty_cuda({m, kf::kAdalnHidden / 2}, at::ScalarType::Byte, device, std::nullopt);
    at::Tensor sf = at::detail::empty_cuda({padded * (kf::kAdalnHidden / 16)}, at::ScalarType::Byte, device, std::nullopt);
    if (m == 0)
    {
        return {fp4, sf};
    }
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    kf::adalnNvfp4WarpSm120(x.const_data_ptr(), sel, selectIsIndex, scale.const_data_ptr(), shift.const_data_ptr(),
        scale.stride(1), shift.stride(1), globalScale.const_data_ptr<float>(), fp4.data_ptr<uint8_t>(),
        sf.data_ptr<uint8_t>(), static_cast<int>(m), static_cast<float>(eps), stream);
    return {fp4, sf};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "flux3_qkv_epi_fp8(Tensor act, Tensor weight, Tensor act_sf, Tensor weight_sf, Tensor qkv_alpha, "
        "Tensor q_norm, Tensor k_norm, Tensor rope_tab, Tensor rope_idx, float sm_scale, bool keep_v) "
        "-> (Tensor, Tensor, Tensor)");
    m.def(
        "flux3_gate_up_swiglu_nvfp4(Tensor act, Tensor weight, Tensor act_sf, Tensor weight_sf, Tensor alpha, "
        "Tensor out_scale, Tensor(a!) out, Tensor(b!) out_sf) -> ()");
    m.def(
        "flux3_nvfp4_pair_residual(Tensor down_act, Tensor down_weight, Tensor down_act_sf, Tensor down_weight_sf, "
        "Tensor down_alpha, Tensor attn_act, Tensor attn_weight, Tensor attn_act_sf, Tensor attn_weight_sf, "
        "Tensor attn_alpha, Tensor x, Tensor gate, Tensor? gate_index, Tensor(a!) workspace, bool t256, "
        "bool along_n) -> Tensor");
    m.def("flux3_fp8r_gemm(Tensor a, Tensor b, Tensor sx, Tensor sw, int arm) -> Tensor");
    m.def(
        "flux3_adaln_nvfp4_warp(Tensor x, Tensor? select, Tensor scale, Tensor shift, Tensor global_scale, "
        "float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("flux3_qkv_epi_fp8", &tensorrt_llm::torch_ext::flux3_qkv_epi_fp8);
    m.impl("flux3_gate_up_swiglu_nvfp4", &tensorrt_llm::torch_ext::flux3_gate_up_swiglu_nvfp4);
    m.impl("flux3_nvfp4_pair_residual", &tensorrt_llm::torch_ext::flux3_nvfp4_pair_residual);
    m.impl("flux3_fp8r_gemm", &tensorrt_llm::torch_ext::flux3_fp8r_gemm);
    m.impl("flux3_adaln_nvfp4_warp", &tensorrt_llm::torch_ext::flux3_adaln_nvfp4_warp);
}
