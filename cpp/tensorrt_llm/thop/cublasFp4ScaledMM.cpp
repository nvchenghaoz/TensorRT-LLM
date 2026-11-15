/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/thop/outputTensor.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <unordered_map>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;
using tensorrt_llm::common::CublasMMWrapper;

// Helper function: Get or create a workspace tensor for the given device
// Workspace is reused across multiple GEMM calls to avoid repeated allocation
inline at::Tensor const& getWorkspaceTensor(c10::Device device)
{
    thread_local std::unordered_map<int, at::Tensor> workspace_tensors;
    int device_id = device.index();

    if (workspace_tensors.find(device_id) == workspace_tensors.end())
    {
        workspace_tensors[device_id]
            = torch::empty(CUBLAS_WORKSPACE_SIZE, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    }

    return workspace_tensors[device_id];
}

// Helper function: Convert PyTorch ScalarType to CUDA datatype for FP4 GEMM output
inline cudaDataType_t getCudaDataType(at::ScalarType dtype)
{
    if (dtype == at::ScalarType::Half)
    {
        return CUDA_R_16F;
    }
    else if (dtype == at::ScalarType::BFloat16)
    {
        return CUDA_R_16BF;
    }
    else if (dtype == at::ScalarType::Float)
    {
        return CUDA_R_32F;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "Unsupported output dtype for FP4 GEMM. Supported types: Float16, BFloat16, Float32");
        return CUDA_R_16BF; // Unreachable, but satisfy compiler
    }
}

void cublas_fp4_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias = c10::nullopt)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k_compressed = a.sizes()[1];
    int32_t k = k_compressed * 2;

    // Use device-aware thread-local CublasMMWrapper for FP4 GEMM
    at::cuda::CUDAGuard deviceGuard(a.device());

    thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
    auto& cublasWrapper = cublasWrappers[a.get_device()];
    if (!cublasWrapper)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    // Set FP4 configuration based on output tensor dtype
    cudaDataType_t outType = getCudaDataType(out.scalar_type());
    cublasWrapper->setFP4GemmConfig(outType);

    // Get workspace (reuse cached workspace for this device)
    auto const& workspace = getWorkspaceTensor(a.device());

    // Get stream
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    // Get data pointers
    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

    // Convert scaling factors to __nv_fp8_e4m3 format for cuBLASLt
    void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
    void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

    // Validate pointers
    TLLM_CHECK_WITH_INFO(a_sf_ptr != nullptr, "a_sf_ptr is null");
    TLLM_CHECK_WITH_INFO(b_sf_ptr != nullptr, "b_sf_ptr is null");

    // Validate alpha tensor before accessing data
    TLLM_CHECK_WITH_INFO(alpha.numel() > 0, "Alpha tensor is empty");
    TLLM_CHECK_WITH_INFO(alpha.dtype() == torch::kFloat32, "Alpha tensor must be float32");

    auto* alpha_ptr = alpha.data_ptr<float>();

    TLLM_CHECK_WITH_INFO(alpha_ptr != nullptr, "alpha_ptr is null");

    // Set workspace and stream
    cublasWrapper->setStream(stream);
    cublasWrapper->setWorkspace(ws_ptr);

    // Perform FP4 GEMM using CublasMMWrapper
    // Matrix layout conversion for cuBLASLt:
    //   PyTorch uses row-major layout: A[m, k] x B[n, k]^T = C[m, n]
    //   cuBLASLt expects column-major layout: B^T[k, n] x A^T[k, m] = C[m, n]
    // We achieve this conversion by:
    //   1. Swapping A and B matrices (b_ptr comes before a_ptr)
    //   2. Using CUBLAS_OP_T for first matrix, CUBLAS_OP_N for second
    //   3. Passing dimensions as (n, m, k) instead of (m, n, k)
    //   4. Swapping scaling factors to match (b_sf_ptr, a_sf_ptr)
    // Note: beta is always 0 and is managed internally by BlockScaleGemm
    void const* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined())
    {
        TLLM_CHECK_WITH_INFO(bias->is_cuda(), "bias must be a CUDA tensor for cuBLASLt epilogue");
        TLLM_CHECK_WITH_INFO(
            bias->device() == out.device(), "bias must reside on the same CUDA device as the GEMM output");
        TLLM_CHECK_WITH_INFO(bias->is_contiguous(), "bias must be contiguous for cuBLASLt epilogue");
        TLLM_CHECK_WITH_INFO(bias->dim() == 1 && bias->size(0) == n,
            "bias must be a 1-D tensor of shape [N] matching the GEMM output dim");
        TLLM_CHECK_WITH_INFO(
            bias->scalar_type() == out.scalar_type(), "bias dtype must match output dtype for cuBLASLt epilogue");
        bias_ptr = bias->data_ptr();
    }

    cublasWrapper->BlockScaleGemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr, k, // B matrix (swapped to first position)
        a_ptr, k,                                                              // A matrix (swapped to second position)
        out_ptr, n,                                                            // Output: C[m, n] in row-major
        b_sf_ptr, a_sf_ptr,                                                    // Scaling factors (also swapped)
        alpha_ptr,                                                             // Uses default algorithm (nullptr)
        bias_ptr);                                                             // Optional bias
}

} // namespace

// CublasLt FP4 GEMM Runner with auto-tuning support
class CublasLtFP4GemmRunner : public torch::CustomClassHolder
{
public:
    explicit CublasLtFP4GemmRunner(at::ScalarType outputDtype)
        : mOutputDtype(outputDtype)
    {
    }

    // Get number of heuristic algorithms for a given matrix shape
    int64_t getNumHeuristicAlgos(
        at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale, at::Tensor const& mat2_scale)
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2; // FP4 is 2 elements per byte
        int n = mat2.size(0);

        auto& cache = getOrCreateAlgoCache(m, k, n, mat1.device(), mat1_scale, mat2_scale);
        size_t num_algos = cache.heuristics.size();
        return static_cast<int64_t>(num_algos);
    }

    // Run GEMM with specified tactic (-1 for default/best). Optional `bias`
    // is fused via CUBLASLT_EPILOGUE_BIAS. The tactic indexes the heuristic
    // list queried at the actual m.
    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale,
        at::Tensor const& mat2_scale, at::Tensor const& alpha, int64_t output_buffer_kind, int64_t tactic,
        c10::optional<torch::List<int64_t>> group = c10::nullopt,
        c10::optional<torch::Tensor> bias = c10::nullopt) const
    {
        return runGemmBucketed(
            mat1, mat2, mat1_scale, mat2_scale, alpha, output_buffer_kind, tactic, /*heuristic_m=*/-1, group, bias);
    }

    // Run GEMM where `tactic` indexes the heuristic list queried at `heuristic_m`
    // (the autotuner bucket the tactic was profiled at). The selected algorithm is
    // validated on the actual (m, n, k) with cublasLtMatmulAlgoCheck; if it is not
    // supported there, the top-1 heuristic of the actual m runs instead.
    // heuristic_m <= 0 means "the actual m" (same as runGemm).
    at::Tensor runGemmBucketed(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale,
        at::Tensor const& mat2_scale, at::Tensor const& alpha, int64_t output_buffer_kind, int64_t tactic,
        int64_t heuristic_m, c10::optional<torch::List<int64_t>> group = c10::nullopt,
        c10::optional<torch::Tensor> bias = c10::nullopt) const
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2;
        int n = mat2.size(0);

        // Prepare output tensor
        std::vector<int64_t> output_size = {m, n};
        auto [out, _] = torch_ext::allocate_output(
            output_size, mOutputDtype, mat1.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);

        ResolvedAlgo const& resolved
            = resolveAlgo(m, static_cast<int>(heuristic_m), k, n, mat1.device(), mat1_scale, mat2_scale, tactic);

        // Execute GEMM (beta is always 0 and is managed internally)
        if (resolved.has_algo)
        {
            cublas_fp4_gemm_caller_with_algo(
                out, mat1, mat2, mat1_scale, mat2_scale, alpha, resolved.algo, mOutputDtype, bias);
        }
        else
        {
            // Fall back to default (no algorithm specified)
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: No valid algorithm found (tactic=%ld), falling back to default "
                "for shape (m=%d, n=%d, k=%d)",
                tactic, m, n, k);
            cublas_fp4_gemm_caller(out, mat1, mat2, mat1_scale, mat2_scale, alpha, bias);
        }

        return out;
    }

    // Debug/introspection: the cuBLASLt algorithm config runGemmBucketed would run for
    // (mat1.size(0), heuristic_m, tactic). Returns
    // [has_algo, fallback, algo_id, tile_id, stages_id, splitk_num, reduction_scheme, cta_swizzling,
    //  custom_option, inner_shape_id, cluster_shape_id]. No GEMM is executed.
    std::vector<int64_t> getAlgoConfig(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale,
        at::Tensor const& mat2_scale, int64_t tactic, int64_t heuristic_m) const
    {
        int m = mat1.size(0);
        int k = mat1.size(1) * 2;
        int n = mat2.size(0);
        ResolvedAlgo const& resolved
            = resolveAlgo(m, static_cast<int>(heuristic_m), k, n, mat1.device(), mat1_scale, mat2_scale, tactic);
        std::vector<int64_t> result{resolved.has_algo ? 1 : 0, resolved.fallback ? 1 : 0};
        if (!resolved.has_algo)
        {
            result.resize(11, -1);
            return result;
        }
        auto const* a = &resolved.algo;
        int32_t algoId = 0, tile = 0, stages = 0, splitK = 0, reduction = 0, swizzle = 0, custom = 0;
        uint16_t inner = 0, cluster = 0;
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(
            a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(
            a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster), nullptr);
        result.insert(result.end(), {algoId, tile, stages, splitK, reduction, swizzle, custom, inner, cluster});
        return result;
    }

private:
    struct ResolvedAlgo
    {
        cublasLtMatmulAlgo_t algo{};
        bool has_algo = false;
        bool fallback = false; // the requested tactic could not run at this m; actual-m top-1 is used
    };

    // (m, heuristic_m, k, n, device_id, output_dtype, tactic)
    using ResolvedKey = std::tuple<int, int, int, int, int, int, int64_t>;

    struct ResolvedKeyHash
    {
        size_t operator()(ResolvedKey const& key) const
        {
            size_t seed = 0;
            auto combine = [&seed](auto const& v)
            {
                using T = std::decay_t<decltype(v)>;
                seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };
            combine(std::get<0>(key));
            combine(std::get<1>(key));
            combine(std::get<2>(key));
            combine(std::get<3>(key));
            combine(std::get<4>(key));
            combine(std::get<5>(key));
            combine(std::get<6>(key));
            return seed;
        }
    };

    mutable std::unordered_map<ResolvedKey, ResolvedAlgo, ResolvedKeyHash> mResolvedCache;

    // Validate `algo` on the actual problem (n, m, k after the row-major swap).
    bool algoSupportedAt(int m, int k, int n, c10::Device device, at::Tensor const& mat1_scale,
        at::Tensor const& mat2_scale, cublasLtMatmulAlgo_t const& algo) const
    {
        at::cuda::CUDAGuard deviceGuard(device);
        auto cublasWrapper
            = std::make_shared<CublasMMWrapper>(getCublasHandle(), getCublasLtHandle(), nullptr, nullptr);
        cublasWrapper->setFP4GemmConfig(getCudaDataType(mOutputDtype));
        cublasWrapper->createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, 0);
        void* a_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(mat1_scale.data_ptr()));
        void* b_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(mat2_scale.data_ptr()));
        cublasWrapper->setScaleDescriptors(a_sf_ptr, b_sf_ptr);
        bool ok = cublasWrapper->checkTactic(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, algo);
        cublasWrapper->destroyDescriptors();
        return ok;
    }

    ResolvedAlgo const& resolveAlgo(int m, int heuristic_m, int k, int n, c10::Device device,
        at::Tensor const& mat1_scale, at::Tensor const& mat2_scale, int64_t tactic) const
    {
        if (heuristic_m <= 0)
        {
            heuristic_m = m;
        }
        ResolvedKey key
            = std::make_tuple(m, heuristic_m, k, n, device.index(), static_cast<int>(mOutputDtype), tactic);
        auto it = mResolvedCache.find(key);
        if (it != mResolvedCache.end())
        {
            return it->second;
        }

        ResolvedAlgo r;
        if (heuristic_m == m)
        {
            // Original semantics: index the heuristic list of the actual m.
            auto& cache = getOrCreateAlgoCache(m, k, n, device, mat1_scale, mat2_scale);
            if (tactic >= 0 && tactic < static_cast<int64_t>(cache.heuristics.size()))
            {
                r.algo = cache.heuristics[tactic].algo;
                r.has_algo = true;
            }
            else if (tactic == -1 && !cache.heuristics.empty())
            {
                int64_t best_idx
                    = cache.best_tactic < static_cast<int64_t>(cache.heuristics.size()) ? cache.best_tactic : 0;
                r.algo = cache.heuristics[best_idx].algo;
                r.has_algo = true;
            }
        }
        else
        {
            bool resolved = false;
            if (tactic >= 0)
            {
                // The tactic was profiled at heuristic_m: take the algorithm from that list.
                auto& hcache = getOrCreateAlgoCache(heuristic_m, k, n, device, mat1_scale, mat2_scale);
                if (tactic < static_cast<int64_t>(hcache.heuristics.size()))
                {
                    cublasLtMatmulAlgo_t const& candidate = hcache.heuristics[tactic].algo;
                    if (algoSupportedAt(m, k, n, device, mat1_scale, mat2_scale, candidate))
                    {
                        r.algo = candidate;
                        r.has_algo = true;
                        resolved = true;
                    }
                }
                if (!resolved)
                {
                    TLLM_LOG_WARNING(
                        "CublasLtFP4GemmRunner: tactic %ld profiled at m=%d is not usable at m=%d (n=%d, k=%d); "
                        "using the top heuristic of m=%d",
                        tactic, heuristic_m, m, n, k, m);
                    r.fallback = true;
                }
            }
            if (!resolved)
            {
                auto& cache = getOrCreateAlgoCache(m, k, n, device, mat1_scale, mat2_scale);
                if (!cache.heuristics.empty())
                {
                    int64_t best_idx
                        = cache.best_tactic < static_cast<int64_t>(cache.heuristics.size()) ? cache.best_tactic : 0;
                    r.algo = cache.heuristics[best_idx].algo;
                    r.has_algo = true;
                }
            }
        }
        TLLM_LOG_DEBUG(
            "CublasLtFP4GemmRunner: resolved tactic %ld (heuristic_m=%d) for shape (m=%d, n=%d, k=%d): has_algo=%d "
            "fallback=%d",
            tactic, heuristic_m, m, n, k, static_cast<int>(r.has_algo), static_cast<int>(r.fallback));
        auto [ins, _] = mResolvedCache.emplace(key, r);
        return ins->second;
    }

    struct AlgoCache
    {
        std::vector<cublasLtMatmulHeuristicResult_t> heuristics;
        int64_t best_tactic = 0; // Index of the best algorithm
    };

    // Cache key: (m, k, n, device_id, output_dtype) for algorithm list storage
    // Different output dtypes may have different optimal algorithms
    using ShapeKey = std::tuple<int, int, int, int, int>;

    struct ShapeKeyHash
    {
        size_t operator()(ShapeKey const& k) const
        {
            // Use boost-style hash_combine for better distribution
            size_t seed = 0;
            hash_combine(seed, std::get<0>(k));
            hash_combine(seed, std::get<1>(k));
            hash_combine(seed, std::get<2>(k));
            hash_combine(seed, std::get<3>(k));
            hash_combine(seed, std::get<4>(k));
            return seed;
        }

    private:
        // Standard hash combination algorithm (Boost-style)
        template <typename T>
        static void hash_combine(size_t& seed, T const& v)
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    };

    mutable std::unordered_map<ShapeKey, AlgoCache, ShapeKeyHash> mAlgoCache;
    at::ScalarType mOutputDtype;

    AlgoCache& getOrCreateAlgoCache(
        int m, int k, int n, c10::Device device, at::Tensor const& mat1_scale, at::Tensor const& mat2_scale) const
    {
        ShapeKey key = std::make_tuple(m, k, n, device.index(), static_cast<int>(mOutputDtype));

        if (mAlgoCache.find(key) == mAlgoCache.end())
        {
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Cache miss for shape (m=%d, k=%d, n=%d, device=%d, dtype=%d), creating new "
                "cache entry",
                m, k, n, device.index(), static_cast<int>(mOutputDtype));

            AlgoCache cache;

            // Create cublas wrapper
            at::cuda::CUDAGuard deviceGuard(device);
            auto cublasHandle = getCublasHandle();
            auto cublasLtHandle = getCublasLtHandle();
            auto cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

            // Set FP4 configuration
            cudaDataType_t outType = mOutputDtype == at::ScalarType::Half
                ? CUDA_R_16F
                : (mOutputDtype == at::ScalarType::BFloat16 ? CUDA_R_16BF : CUDA_R_32F);

            cublasWrapper->setFP4GemmConfig(outType);

            // Create descriptors
            cublasWrapper->createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, 0);

            // Use provided scale tensors for descriptor setup
            // FP4 GEMM always requires scale tensors
            void* a_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(mat1_scale.data_ptr()));
            void* b_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(mat2_scale.data_ptr()));

            // Set scale descriptors (required for FP4 GEMM heuristics)
            cublasWrapper->setScaleDescriptors(a_sf_ptr, b_sf_ptr);

            // Get heuristic algorithms
            auto heuristics = cublasWrapper->getTactics(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, k, k, n);

            // Filter valid algorithms
            for (auto const& h : heuristics)
            {
                if (h.state == CUBLAS_STATUS_SUCCESS && h.workspaceSize <= CUBLAS_WORKSPACE_SIZE)
                {
                    cache.heuristics.push_back(h);
                }
            }

            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Found %zu valid algorithms for shape (m=%d, k=%d, n=%d) on device %d",
                cache.heuristics.size(), m, k, n, device.index());

            if (cache.heuristics.empty())
            {
                TLLM_LOG_WARNING(
                    "CublasLtFP4GemmRunner: No valid cuBLASLt algorithms found for shape (m=%d, k=%d, n=%d), will fall "
                    "back to default",
                    m, k, n);
            }

            cublasWrapper->destroyDescriptors();

            mAlgoCache[key] = std::move(cache);
        }
        else
        {
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Cache hit for shape (m=%d, k=%d, n=%d, device=%d, dtype=%d), %zu algorithms "
                "available",
                m, k, n, device.index(), static_cast<int>(mOutputDtype), mAlgoCache[key].heuristics.size());
        }

        return mAlgoCache[key];
    }

    // Helper function to run GEMM with a specific algorithm
    static void cublas_fp4_gemm_caller_with_algo(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
        torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha,
        cublasLtMatmulAlgo_t const& algo, at::ScalarType output_dtype,
        c10::optional<torch::Tensor> const& bias = c10::nullopt)
    {
        int32_t m = a.sizes()[0];
        int32_t n = b.sizes()[0];
        int32_t k_compressed = a.sizes()[1];
        int32_t k = k_compressed * 2;

        at::cuda::CUDAGuard deviceGuard(a.device());

        thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
        auto& cublasWrapper = cublasWrappers[a.get_device()];
        if (!cublasWrapper)
        {
            auto cublasHandle = getCublasHandle();
            auto cublasLtHandle = getCublasLtHandle();
            cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
        }

        // Set FP4 configuration with correct output type
        cudaDataType_t outType = getCudaDataType(output_dtype);
        cublasWrapper->setFP4GemmConfig(outType);

        // Get workspace (reuse cached workspace for this device)
        auto const& workspace = getWorkspaceTensor(a.device());

        auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

        auto* a_ptr = static_cast<void*>(a.data_ptr());
        auto* b_ptr = static_cast<void*>(b.data_ptr());
        auto* out_ptr = static_cast<void*>(out.data_ptr());
        auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

        void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
        void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

        // Validate alpha tensor before accessing data
        TLLM_CHECK_WITH_INFO(alpha.numel() > 0, "Alpha tensor is empty");
        TLLM_CHECK_WITH_INFO(alpha.dtype() == torch::kFloat32, "Alpha tensor must be float32");

        auto* alpha_ptr = alpha.data_ptr<float>();

        TLLM_CHECK_WITH_INFO(alpha_ptr != nullptr, "alpha_ptr is null");

        cublasWrapper->setStream(stream);
        cublasWrapper->setWorkspace(ws_ptr);

        // Matrix layout conversion for cuBLASLt (same as in cublas_fp4_gemm_caller):
        //   PyTorch uses row-major layout: A[m, k] x B[n, k]^T = C[m, n]
        //   cuBLASLt expects column-major layout: B^T[k, n] x A^T[k, m] = C[m, n]
        // Conversion is achieved by:
        //   1. Swapping A and B matrices (b_ptr comes before a_ptr)
        //   2. Using CUBLAS_OP_T for first matrix, CUBLAS_OP_N for second
        //   3. Passing dimensions as (n, m, k) instead of (m, n, k)
        //   4. Swapping scaling factors to match matrices (b_sf_ptr, a_sf_ptr)

        void const* bias_ptr = nullptr;
        if (bias.has_value() && bias->defined())
        {
            TLLM_CHECK_WITH_INFO(bias->is_cuda(), "bias must be a CUDA tensor for cuBLASLt epilogue");
            TLLM_CHECK_WITH_INFO(
                bias->device() == out.device(), "bias must reside on the same CUDA device as the GEMM output");
            TLLM_CHECK_WITH_INFO(bias->is_contiguous(), "bias must be contiguous for cuBLASLt epilogue");
            TLLM_CHECK_WITH_INFO(bias->dim() == 1 && bias->size(0) == n,
                "bias must be a 1-D tensor of shape [N] matching the GEMM output dim");
            TLLM_CHECK_WITH_INFO(
                bias->scalar_type() == out.scalar_type(), "bias dtype must match output dtype for cuBLASLt epilogue");
            bias_ptr = bias->data_ptr();
        }

        // Use BlockScaleGemm with specified algorithm for autotuning
        // Note: beta is always 0 and is managed internally by BlockScaleGemm
        cublasWrapper->BlockScaleGemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr,
            k,                  // B matrix (swapped to first position)
            a_ptr, k,           // A matrix (swapped to second position)
            out_ptr, n,         // Output: C[m, n] in row-major
            b_sf_ptr, a_sf_ptr, // Scaling factors (also swapped)
            alpha_ptr,          // Alpha
            &algo,              // Use specified algorithm
            bias_ptr);          // Optional bias for CUBLASLT_EPILOGUE_BIAS
    }
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::CublasLtFP4GemmRunner>("CublasLtFP4GemmRunner")
        .def(torch::init<at::ScalarType>())
        .def("run_gemm", &tensorrt_llm::torch_ext::CublasLtFP4GemmRunner::runGemm)
        .def("run_gemm_bucketed", &tensorrt_llm::torch_ext::CublasLtFP4GemmRunner::runGemmBucketed)
        .def("get_algo_config", &tensorrt_llm::torch_ext::CublasLtFP4GemmRunner::getAlgoConfig)
        .def("get_num_heuristic_algos", &tensorrt_llm::torch_ext::CublasLtFP4GemmRunner::getNumHeuristicAlgos);
}
