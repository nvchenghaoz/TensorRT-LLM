/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/kernels/marlin/marlin_nvfp4.h"
#include "tensorrt_llm/thop/outputTensor.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <torch/extension.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

// Persistent zeroed lock workspace for the dense Marlin GEMM.
//
// The kernel leaves every lock it uses at 0 when it completes (use_atomic_add
// is false: the last slice of each split-K reduction resets its lock in
// barrier_release, and blocks_per_sm == 1 keeps locks_off < sms), so a buffer
// zeroed once can be reused by every call that
// cannot overlap with another call on the same buffer. Calls on one stream are
// ordered (the kernel issues griddepcontrol.wait before any lock access, so
// this also holds under PDL), so each stream gets its own slot.
//
// The pool is created on the first non-capturing call on a device. A stream
// without a slot binds a free one with no allocation, which is what a CUDA
// graph capture stream gets (torch.cuda.graph captures on its own side stream
// that never runs eager work). With no pool yet or no free slot, the call falls
// back to a per-call at::zeros workspace.
constexpr int kMarlinWorkspaceSlots = 32;

struct MarlinWorkspacePool
{
    at::Tensor buffer; // [kMarlinWorkspaceSlots * sms] int32, zero-initialized
    int sms = 0;
    int nextSlot = 0;
    std::unordered_map<cudaStream_t, int> slotOfStream;
};

std::mutex& marlinWorkspaceMutex()
{
    static std::mutex m;
    return m;
}

// Intentionally leaked so no tensor is released after CUDA teardown at exit.
std::unordered_map<int, MarlinWorkspacePool*>& marlinWorkspacePools()
{
    static auto* pools = new std::unordered_map<int, MarlinWorkspacePool*>();
    return *pools;
}

// Returns this stream's zeroed slot (at least `sms` ints), or nullptr to fall back.
int* getPersistentMarlinWorkspace(Tensor const& like, int device, int sms, cudaStream_t stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    bool const capturing = status != cudaStreamCaptureStatusNone;

    std::lock_guard<std::mutex> guard(marlinWorkspaceMutex());
    auto& pools = marlinWorkspacePools();
    auto it = pools.find(device);
    MarlinWorkspacePool* pool = it == pools.end() ? nullptr : it->second;
    if (pool == nullptr)
    {
        if (capturing)
        {
            return nullptr;
        }
        pool = new MarlinWorkspacePool();
        pool->sms = sms;
        pool->buffer = at::zeros({kMarlinWorkspaceSlots * sms}, like.options().dtype(at::kInt));
        // Make the zero fill visible to every stream that later binds a slot.
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        pools[device] = pool;
    }
    if (pool->sms < sms)
    {
        return nullptr;
    }
    int slot = -1;
    auto sit = pool->slotOfStream.find(stream);
    if (sit != pool->slotOfStream.end())
    {
        slot = sit->second;
    }
    else
    {
        if (pool->nextSlot >= kMarlinWorkspaceSlots)
        {
            return nullptr;
        }
        slot = pool->nextSlot++;
        pool->slotOfStream.emplace(stream, slot);
    }
    return pool->buffer.data_ptr<int>() + static_cast<int64_t>(slot) * pool->sms;
}

} // namespace

// Marlin NVFP4 GEMM: W4A(4|16) interface.
//
// Accepts either FP4 E2M1 packed activations or BF16 activations, plus Marlin-tiled FP4 weights.
// FP4 path: dequantizes activations to BF16 using scale_a/alpha, then runs W4A16 Marlin GEMM.
// BF16 path: skips dequant, runs W4A16 Marlin GEMM directly (scale_a/alpha ignored).
//
// mat_a:              [M, K/2] FP4 E2M1 packed as uint8 (2 values per byte) OR [M, K] BF16
// mat_b:              Marlin-packed FP4 weights (int32), from gptq_marlin_repack
// scale_a:            activation block scales (FP8 E4M3, swizzled, stored as uint8)
// scale_b:            Marlin-processed weight block scales (from marlin_permute_scales + nvfp4_marlin_process_scales)
// alpha:              float32 global scale (applied to activation dequant only)
// weight_global_scale: BF16 Marlin-processed weight global scale (from nvfp4_marlin_process_global_scale)
// bias:               optional bias (not supported)
// out_dtype:          output dtype (fp16 or bf16)
// size_n:             output dimension N
// size_k:             reduction dimension K
// output_buffer_kind: output allocation kind (0=default, 1=userbuffers, 2=nccl_window)
// group:              communicator ranks, used when output_buffer_kind selects an NCCL window
// workspace_mode:     0 = persistent lock workspace, PDL launch when TRTLLM_ENABLE_PDL allows it;
//                     2 = as 0, plus split-K partials reduced in FP32 (per-call FP32 C_tmp) so each
//                     output is rounded to BF16 once.
// launch_hint:        0 = the stock launch configuration; 1 = the Qwen3.8 decode launch tables
//                     (marlin_nvfp4::kMarlinNvfp4LaunchHintQwen38).
Tensor marlin_nvfp4_gemm(Tensor const& mat_a, Tensor const& mat_b, std::optional<Tensor> const& scale_a,
    Tensor const& scale_b, std::optional<Tensor> const& alpha, Tensor const& weight_global_scale,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, int64_t size_n, int64_t size_k,
    int64_t output_buffer_kind = 0, c10::optional<torch::List<int64_t>> group = c10::nullopt,
    int64_t workspace_mode = 0, int64_t launch_hint = 0)
{
    TORCH_CHECK(workspace_mode == 0 || workspace_mode == 2, "workspace_mode must be 0 or 2, got ", workspace_mode);
    TORCH_CHECK(launch_hint == ::marlin_nvfp4::kMarlinNvfp4LaunchHintDefault
            || launch_hint == ::marlin_nvfp4::kMarlinNvfp4LaunchHintQwen38,
        "launch_hint must be 0 or 1, got ", launch_hint);
    CHECK_TH_CUDA(mat_a);
    TORCH_CHECK(mat_a.scalar_type() == FLOAT4_E2M1X2 || mat_a.scalar_type() == at::ScalarType::BFloat16,
        "mat_a must be FP4 E2M1X2 or BFloat16, got ", mat_a.scalar_type());
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(weight_global_scale);
    if (mat_a.scalar_type() != at::ScalarType::BFloat16)
    {
        TORCH_CHECK(scale_a.has_value() and alpha.has_value(), "scale_a must be provided for FP4 activations");
        CHECK_INPUT(scale_a.value(), SF_DTYPE); // e4m3
        CHECK_INPUT(alpha.value(), at::ScalarType::Float);
    }

    TORCH_CHECK(mat_a.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(!bias.has_value(), "bias is not supported yet");

    auto const out_dtype_ = out_dtype.value_or(at::ScalarType::Half);
    TORCH_CHECK(
        out_dtype_ == at::ScalarType::Half || out_dtype_ == at::ScalarType::BFloat16, "Output must be fp16 or bf16");

    int32_t m = mat_a.sizes()[0];
    int32_t n = static_cast<int32_t>(size_n);
    int32_t k = static_cast<int32_t>(size_k);

    // Allocate output
    auto [out, _] = torch_ext::allocate_output(
        {m, n}, out_dtype_, mat_a.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);

    if (m == 0)
        return out;

    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

    Tensor act_bf16;
    if (mat_a.scalar_type() == at::ScalarType::BFloat16)
    {
        // BF16 activations — skip FP4 dequant, use directly
        act_bf16 = mat_a;
    }
    else
    {
        // FP4 activations — dequantize to BF16
        act_bf16 = at::empty({m, k}, mat_a.options().dtype(at::ScalarType::BFloat16));
        ::marlin_nvfp4::dequantFp4Activations(mat_a.data_ptr(), scale_a.value().data_ptr(),
            reinterpret_cast<float const*>(alpha.value().data_ptr()), act_bf16.data_ptr(), m, k, stream);
    }

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, mat_a.get_device());

    bool const fp32Reduce = workspace_mode == 2;
    bool const enablePdl = tensorrt_llm::common::getEnvEnablePDL();

    Tensor workspace;
    int* workspacePtr = getPersistentMarlinWorkspace(mat_a, mat_a.get_device(), sms, stream);
    if (workspacePtr == nullptr)
    {
        workspace = at::zeros({sms}, mat_a.options().dtype(at::kInt));
        workspacePtr = reinterpret_cast<int*>(workspace.data_ptr());
    }

    int num_groups = k / 16;
    int group_size = 16;

    // workspace_mode 2: split-K partial sums reduce in an FP32 buffer (the template's
    // global_reduce_fp32) instead of chaining through the BF16 output, so each output is
    // rounded to BF16 once. One [tb_m x tb_n] FP32 tile per lock slot (locks_off < sms,
    // blocks_per_sm == 1), tb_m <= 64 rows, tb_n <= 256 columns. Slice 0 writes its tile
    // before any slice reads it, so the buffer needs no zero fill.
    Tensor cTmp;
    void* cTmpPtr = nullptr;
    if (fp32Reduce)
    {
        int64_t const tileRows = std::min<int64_t>((static_cast<int64_t>(m) + 15) / 16 * 16, 64);
        cTmp = at::empty({static_cast<int64_t>(sms) * tileRows * 256}, mat_a.options().dtype(at::kFloat));
        cTmpPtr = cTmp.data_ptr();
    }

    // Step 2: Run Marlin W4A16 GEMM
    // weight_global_scale is BF16, already Marlin-processed (includes 2^119 bias correction)
    ::marlin_nvfp4::marlinNvfp4Gemm(act_bf16.data_ptr(), mat_b.data_ptr(), out.data_ptr(), cTmpPtr, scale_b.data_ptr(),
        weight_global_scale.data_ptr(), m, n, k, workspacePtr, num_groups, group_size, fp32Reduce, stream, enablePdl,
        static_cast<int>(launch_hint));

    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "marlin_nvfp4_gemm(Tensor mat_a, Tensor mat_b, Tensor? scale_a, Tensor scale_b, Tensor? alpha,"
        " Tensor weight_global_scale, Tensor? bias, ScalarType? out_dtype,"
        " int size_n, int size_k, int output_buffer_kind=0, int[]? group=None, int workspace_mode=0,"
        " int launch_hint=0) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("marlin_nvfp4_gemm", &tensorrt_llm::torch_ext::marlin_nvfp4_gemm);
}
