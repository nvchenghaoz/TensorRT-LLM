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
#include "sm120Bf16SeqKGemm.h"
#include "sm120Bf16SeqKGemmKernel.cuh"

#include <cudaTypedefs.h>
#include <mutex>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::sm120_seqk_gemm
{

namespace
{

using sm120_seqk_detail::SeqKCfg;
using sm120_seqk_detail::sm120SeqKGemmKernel;

using CfgT160W176 = SeqKCfg<5, 11, 32, 4, 3>;
using CfgT160W208W8 = SeqKCfg<5, 13, 32, 4, 2>;
using CfgT160W208 = SeqKCfg<5, 13, 32, 4, 3>;
using CfgT128W176 = SeqKCfg<4, 11, 32, 4, 3>;

PFN_cuTensorMapEncodeTiled_v12000 encodeFn()
{
    static PFN_cuTensorMapEncodeTiled_v12000 fn = []() -> PFN_cuTensorMapEncodeTiled_v12000
    {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q)
                != cudaSuccess
            || q != cudaDriverEntryPointSuccess)
        {
            return nullptr;
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
    }();
    return fn;
}

bool encode(CUtensorMap* m, void const* ptr, int64_t rows, int64_t K, int64_t ld, int boxRows, int bk)
{
    auto fn = encodeFn();
    if (fn == nullptr)
    {
        return false;
    }
    cuuint64_t dims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(rows)};
    cuuint64_t strides[1] = {static_cast<cuuint64_t>(ld) * 2};
    cuuint32_t box[2] = {static_cast<cuuint32_t>(bk), static_cast<cuuint32_t>(boxRows)};
    cuuint32_t elemStrides[2] = {1, 1};
    // Out-of-bounds rows (M and N tails) are zero-filled; they never reach the output.
    CUresult r = fn(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, bk == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS;
}

struct MapKey
{
    void const* ptr;
    int64_t rows;
    int64_t K;
    int64_t ld;
    int box;
    int bk;

    bool operator==(MapKey const& o) const
    {
        return ptr == o.ptr && rows == o.rows && K == o.K && ld == o.ld && box == o.box && bk == o.bk;
    }
};

struct MapKeyHash
{
    size_t operator()(MapKey const& k) const
    {
        size_t h = std::hash<void const*>()(k.ptr);
        auto mix = [&h](int64_t v) { h ^= std::hash<int64_t>()(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix(k.rows);
        mix(k.K);
        mix(k.ld);
        mix(k.box);
        mix(k.bk);
        return h;
    }
};

// Weight descriptors are reused across calls (weights are persistent); activation descriptors are
// encoded per call because the activation buffer moves.
bool weightMap(CUtensorMap* out, void const* ptr, int64_t rows, int64_t K, int64_t ld, int boxRows, int bk)
{
    static std::mutex mu;
    static std::unordered_map<MapKey, CUtensorMap, MapKeyHash> cache;
    MapKey key{ptr, rows, K, ld, boxRows, bk};
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end())
    {
        *out = it->second;
        return true;
    }
    if (!encode(out, ptr, rows, K, ld, boxRows, bk))
    {
        return false;
    }
    if (cache.size() >= 4096)
    {
        cache.clear();
    }
    cache.emplace(key, *out);
    return true;
}

template <class C>
int launch(void const* x, void const* w, void* y, int M, int N, int K, int64_t ldx, int64_t ldw, int64_t ldy,
    cudaStream_t stream)
{
    if (K % C::BK != 0)
    {
        return kErrShape;
    }
    static cudaError_t attrErr = cudaFuncSetAttribute(
        sm120SeqKGemmKernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    if (attrErr != cudaSuccess)
    {
        return kErrAttr;
    }
    CUtensorMap tx;
    CUtensorMap tw;
    if (!encode(&tx, x, M, K, ldx, C::TM, C::BK) || !weightMap(&tw, w, N, K, ldw, C::TN, C::BK))
    {
        return kErrEncode;
    }
    int const ntm = (M + C::TM - 1) / C::TM;
    int const ntn = (N + C::TN - 1) / C::TN;
    sm120SeqKGemmKernel<C><<<ntm * ntn, C::kThreads, C::kSmem, stream>>>(
        tx, tw, reinterpret_cast<__nv_bfloat16*>(y), M, N, K, ldy, ntm);
    return cudaGetLastError() == cudaSuccess ? kOk : kErrLaunch;
}

} // namespace

int run(void const* x, void const* w, void* y, int M, int N, int K, int64_t ldx, int64_t ldw, int64_t ldy, int config,
    cudaStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || N % 8 != 0 || ldx % 8 != 0 || ldw % 8 != 0 || ldy % 8 != 0 || ldx < K
        || ldw < K || ldy < N || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(w) & 15)
        || (reinterpret_cast<uintptr_t>(y) & 15))
    {
        return kErrShape;
    }
    switch (config)
    {
    case kT160W176: return launch<CfgT160W176>(x, w, y, M, N, K, ldx, ldw, ldy, stream);
    case kT160W208W8: return launch<CfgT160W208W8>(x, w, y, M, N, K, ldx, ldw, ldy, stream);
    case kT160W208: return launch<CfgT160W208>(x, w, y, M, N, K, ldx, ldw, ldy, stream);
    case kT128W176: return launch<CfgT128W176>(x, w, y, M, N, K, ldx, ldw, ldy, stream);
    default: return kErrConfig;
    }
}

void tileShape(int config, int* tokens, int* weights)
{
    switch (config)
    {
    case kT160W176: *tokens = CfgT160W176::TM, *weights = CfgT160W176::TN; return;
    case kT160W208W8: *tokens = CfgT160W208W8::TM, *weights = CfgT160W208W8::TN; return;
    case kT160W208: *tokens = CfgT160W208::TM, *weights = CfgT160W208::TN; return;
    case kT128W176: *tokens = CfgT128W176::TM, *weights = CfgT128W176::TN; return;
    default: *tokens = 0, *weights = 0; return;
    }
}

} // namespace kernels::sm120_seqk_gemm

TRTLLM_NAMESPACE_END
