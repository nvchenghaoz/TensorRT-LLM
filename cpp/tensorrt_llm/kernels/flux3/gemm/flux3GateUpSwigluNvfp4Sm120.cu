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

// FLUX.3 SM120 NVFP4 gate_up GEMM with a register SwiGLU -> NVFP4 epilogue.
//
// Mainloop: the CUTLASS SM120 block-scaled TMA warp-specialized mainloop, Pingpong schedule with tile
// 128x128x128 (FP4 elements; two 4-warp consumer warpgroups alternate tiles, so one's epilogue
// overlaps the other's MMA), 1x1x1 cluster, the CLC persistent scheduler. The only change to the stock
// collective is the N permutation of the TiledMma, (atom col 8, warp 2, rep 8):(1, 64, 8): each warp
// owns 64 contiguous accumulator columns (two adjacent 16-output NVFP4 scale blocks of the
// pair-interleaved value/gate weight). The K order of the mma.sync chain is unchanged.
//
// Weight layout: gate_up rows are [value, gate] pairs (row 2o = up_o, row 2o + 1 = gate_o), then
// permuted within every 64-row group so that accumulator pair slot s = q + 4r + 16nb of a warp's span
// holds output 8q + 4nb + r (the load-time v2w permutation, gemm/_prepare.py): lane q of a quad then
// owns the 8 consecutive outputs 8q .. 8q + 7 of its warp's 32 outputs per row, block q >> 1. The amax
// needs one xor-1 shuffle, the e2m1 bytes pack lane-locally into one u32 per row, one xor-1 shuffle of a
// word forms an 8-B (row, block) chunk per lane (one STG.64), and every lane stores its own (row, block)
// scale byte.
//
// Epilogue arithmetic, per output element (PTX-pinned):
//   value = acc_even * alpha, gate = acc_odd * alpha            (mul.rn.f32)
//   h = (value * gate) / (ex2.approx.ftz(gate * -log2e) + 1)    (mul.rn, ex2.approx.ftz, add.rn, div.rn)
//   amax over the 16 outputs of the block; enc = e4m3(amax * inv / 6)   (cvt.rn.satfinite.e4m3x2)
//   mult = inv / max(float(enc), 1e-12)                          (div.rn)
//   byte = e2m1x2(high = h_odd * mult, low = h_even * mult)       (cvt.rn.satfinite.e2m1x2)
// FP4 data: one 8-B store per (row, scale block). Scales: 128x4-swizzled layout.

#include "flux3GateUpSwigluNvfp4Sm120.h"
#include "tensorrt_llm/common/assert.h"

#ifndef PLACEHOLDER_KERNELS

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::flux3
{
namespace gateUpSwiglu
{

using namespace cute;

// Exact-form PTX helpers (the module is built with fast math; inline PTX pins the forms).
__device__ __forceinline__ float mul_rn(float a, float b)
{
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float add_rn(float a, float b)
{
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float div_rn(float a, float b)
{
    float d;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float ex2_approx_ftz(float a)
{
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float abs_f32(float a)
{
    float d;
    asm("abs.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float max_f32(float a, float b)
{
    float d;
    asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

// E4M3 byte of x (cvt.rn.satfinite.e4m3x2.f32 with x in the low half).
__device__ __forceinline__ uint32_t e4m3_byte(float x)
{
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(x));
    return static_cast<uint32_t>(r) & 0xFFu;
}

// Exact float value of an E4M3 byte.
__device__ __forceinline__ float e4m3_to_f32(uint32_t byte)
{
    uint32_t h2;
    uint16_t in = static_cast<uint16_t>(byte);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    __half lo = __ushort_as_half(static_cast<unsigned short>(h2 & 0xFFFFu));
    return __half2float(lo);
}

// Byte = (e2m1(high) << 4) | e2m1(low).
__device__ __forceinline__ uint32_t e2m1x2_byte(float high, float low)
{
    uint32_t r;
    asm("{\n"
        ".reg .b8 t;\n"
        "cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n"
        "cvt.u32.u8 %0, t;\n"
        "}\n"
        : "=r"(r)
        : "f"(high), "f"(low));
    return r;
}

__device__ __forceinline__ float swiglu_h(float acc_even, float acc_odd, float factor)
{
    float const kNegLog2e = __int_as_float(0xBFB8AA3B); // -1.4426950f
    float value = mul_rn(factor, acc_even);
    float gate = mul_rn(factor, acc_odd);
    float prod = mul_rn(value, gate);
    float e = ex2_approx_ftz(mul_rn(gate, kNegLog2e));
    float denom = add_rn(e, 1.0f);
    return div_rn(prod, denom);
}

// Operator-style epilogue, wrapped by Sm90TmaWarpSpecializedAdapter.
template <class StrideC_, class StrideD_, class EpilogueSchedule_>
class SwigluFp4Epilogue
{
public:
    using EpilogueSchedule = EpilogueSchedule_;
    using DispatchPolicy = EpilogueSchedule_;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;
    using ElementC = cutlass::bfloat16_t;
    using StrideC = StrideC_;
    using ElementD = cutlass::bfloat16_t;
    using StrideD = StrideD_;
    using GmemTiledCopyC = void;
    using GmemTiledCopyD = void;

    struct SharedStorage
    {
    };

    using TensorStorage = SharedStorage;

    struct Arguments
    {
        uint8_t* out = nullptr;            // [rows_padded, N/4] FP4 bytes (pairs of outputs)
        uint8_t* out_sf = nullptr;         // 128x4-swizzled E4M3 scales, rows_padded x (N/32)
        float const* alpha = nullptr;      // input_gs * weight_gs (device scalar)
        float const* inv_scale = nullptr;  // down_proj's activation global scale (device scalar)
        int rows_padded = 0;               // rows rounded up to 128
        int out_row_bytes = 0;             // N / 4
        int sf_cols = 0;                   // N / 32 (multiple of 4)
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*)
    {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(
        ProblemShape const&, Arguments const&, void*, cudaStream_t, cutlass::CudaHostAdapter* = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const&, Arguments const&)
    {
        return true;
    }

    CUTLASS_HOST_DEVICE
    SwigluFp4Epilogue(Params const& params_, SharedStorage const& = SharedStorage())
        : params(params_)
    {
    }

    CUTLASS_DEVICE bool is_source_needed()
    {
        return false;
    }

    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine, class FrgLayout,
        class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL blk_coord_mnkl,
        cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TiledMma tiled_mma, ResidueMNK, int thread_idx, char*)
    {
        constexpr int BM = size<0>(BlockShapeMNK{});
        constexpr int BN = size<1>(BlockShapeMNK{});
        static_assert(size<0>(FrgLayout{}) == 4, "m16n8 C fragment expected");
        constexpr int MMA_M = size<1>(FrgLayout{});
        constexpr int MMA_N = size<2>(FrgLayout{});
        static_assert(MMA_N == 8, "each lane owns 8 consecutive pairs of a 32-output span");

        auto m_coord = get<0>(blk_coord_mnkl);
        auto n_coord = get<1>(blk_coord_mnkl);

        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        auto cD = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
        auto tCcD = thr_mma.partition_C(cD); // (4, MMA_M, MMA_N) -> (row, col) in tile

        float const factor = *params.alpha;
        float const inverse = *params.inv_scale;
        int const lane = threadIdx.x & 31;
        int const q = lane & 3;
        int const row_base = static_cast<int>(m_coord) * BM;
        int const col_base = static_cast<int>(n_coord) * BN;
        // Lane q: accumulator pair ni (0..7) of row half hh is output 8q + ni of its warp's 32 outputs
        // (value even, gate odd); outputs 0..15 = scale block 0 (q = 0, 1), 16..31 = block 1 (q = 2, 3).
        int const odd = q & 1;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < MMA_M; ++mi)
        {
            float h[2][8];
            CUTLASS_PRAGMA_UNROLL
            for (int hh = 0; hh < 2; ++hh)
            {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < 8; ++ni)
                {
                    h[hh][ni] = swiglu_h(accumulators(2 * hh, mi, ni), accumulators(2 * hh + 1, mi, ni), factor);
                }
            }
            uint32_t word[2];
            uint32_t enc[2];
            CUTLASS_PRAGMA_UNROLL
            for (int hh = 0; hh < 2; ++hh)
            {
                float m = max_f32(0.0f, abs_f32(h[hh][0]));
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 1; ni < 8; ++ni)
                {
                    m = max_f32(m, abs_f32(h[hh][ni]));
                }
                float const amax = max_f32(m, __shfl_xor_sync(0xffffffffu, m, 1));
                enc[hh] = e4m3_byte(div_rn(mul_rn(amax, inverse), 6.0f));
                float const mult = div_rn(inverse, max_f32(e4m3_to_f32(enc[hh]), 1.0e-12f));
                uint32_t w = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 4; ++j)
                {
                    w |= e2m1x2_byte(mul_rn(h[hh][2 * j + 1], mult), mul_rn(h[hh][2 * j], mult)) << (8 * j);
                }
                word[hh] = w; // bytes 4q..4q+3 of the row's 16-B (block_col even, +1) span
            }
            // Even lane: row g, its word + the odd partner's row-g word (8 B of its block).
            // Odd lane: row g+8, the even partner's row-g+8 word + its own.
            uint32_t const send = odd ? word[0] : word[1];
            uint32_t const recv = __shfl_xor_sync(0xffffffffu, send, 1);
            uint2 v;
            v.x = odd ? recv : word[0];
            v.y = odd ? word[1] : recv;
            int const row = row_base + get<0>(tCcD(2 * odd, mi, 0));
            // Warp's 64-column span + lane's block (q >> 1).
            int const span = get<1>(tCcD(0, mi, 0)) & ~63;
            int const block_col = ((col_base + span) >> 5) + (q >> 1);
            if (row < params.rows_padded)
            {
                *reinterpret_cast<uint2*>(params.out + static_cast<int64_t>(row) * params.out_row_bytes + block_col * 8)
                    = v;
                int64_t const idx = (block_col & 3) + (block_col >> 2) * 512 + (row & 31) * 16
                    + ((row & 127) >> 5) * 4 + static_cast<int64_t>(row >> 7) * 128 * params.sf_cols;
                params.out_sf[idx] = static_cast<uint8_t>(odd ? enc[1] : enc[0]);
            }
        }
    }

private:
    Params params;
};

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using AtomLayoutMNK = Layout<Shape<_2, _2, _1>>;

// Mainloop stages, pinned to what StageCountAutoCarveout<0> gives (CUTLASS 4.4.2 and 4.5 compute it the
// same way): sm120 smem 101376 B - CLC scheduler pipeline 48 B - 3 CLC responses 48 B = 101280 B; one
// stage = A 8192 + B 8192 (128x128 FP4 each) + SFA 1024 + SFB 1024 + pipeline 128 = 18560 B -> 5 stages.
constexpr int kStages = 5;

template <class StageCountType>
using Builder = cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, 32, ElementB, LayoutB, 32, float, TileShape,
    ClusterShape, StageCountType, cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
using StockBuilder = Builder<cutlass::gemm::collective::StageCount<kStages>>;
static_assert(Builder<cutlass::gemm::collective::StageCountAutoCarveout<0>>::PipelineStages == kStages,
    "pinned stage count differs from the auto carve-out the kernel was measured with");

// (atom col 8, warp 2, rep 8) -> col i + 64*w + 8*r: each warp owns 64 contiguous columns.
using PermTileN = Layout<Shape<_8, _2, _8>, Stride<_1, _64, _8>>;
using TiledMma = decltype(cute::make_tiled_mma(
    cute::rr_blockscaled_op_selector_sm120<typename StockBuilder::ElementA, typename StockBuilder::ElementB, float,
        typename StockBuilder::ElementSF, StockBuilder::SFVectorSize, StockBuilder::UseMxf8f6f4>(),
    AtomLayoutMNK{}, Tile<typename StockBuilder::PermTileM, PermTileN, typename StockBuilder::PermTileK>{}));

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<typename StockBuilder::DispatchPolicy, TileShape,
    cute::tuple<typename StockBuilder::ElementA, typename StockBuilder::ElementSF>, typename StockBuilder::StridePairA,
    cute::tuple<typename StockBuilder::ElementB, typename StockBuilder::ElementSF>, typename StockBuilder::StridePairB,
    TiledMma, typename StockBuilder::GmemTiledCopyPairA, typename StockBuilder::SmemLayoutAtomsA,
    typename StockBuilder::SmemCopyAtomsA, cute::identity, typename StockBuilder::GmemTiledCopyPairB,
    typename StockBuilder::SmemLayoutAtomsB, typename StockBuilder::SmemCopyAtomsB, cute::identity>;

using StrideCD = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    SwigluFp4Epilogue<StrideCD, StrideCD, cutlass::epilogue::NoSmemWarpSpecialized>>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
    CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;

} // namespace gateUpSwiglu
} // namespace kernels::flux3

TRTLLM_NAMESPACE_END

#endif // PLACEHOLDER_KERNELS

TRTLLM_NAMESPACE_BEGIN

namespace kernels::flux3
{

void gateUpSwigluNvfp4Sm120(void const* a, void const* b, void const* sfa, void const* sfb, float const* alpha,
    float const* outScale, uint8_t* out, uint8_t* outSf, int m, int n, int k, int rowsPadded, cudaStream_t stream)
{
#ifdef PLACEHOLDER_KERNELS
    TLLM_THROW("[FLUX.3 gate_up SwiGLU NVFP4 GEMM] not compiled for this architecture.");
#else
    using namespace gateUpSwiglu;
    using Gemm = GemmKernel;
    using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);
    args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(a);
    args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(b);
    args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sfa);
    args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sfb);
    args.mainloop.dA = cutlass::make_cute_packed_stride(typename Gemm::StrideA{}, {m, k, 1});
    args.mainloop.dB = cutlass::make_cute_packed_stride(typename Gemm::StrideB{}, {n, k, 1});
    args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
    args.epilogue.out = out;
    args.epilogue.out_sf = outSf;
    args.epilogue.alpha = alpha;
    args.epilogue.inv_scale = outScale;
    args.epilogue.rows_padded = rowsPadded;
    args.epilogue.out_row_bytes = n / 4;
    args.epilogue.sf_cols = n / 32;
    static thread_local int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    args.hw_info.device_id = tensorrt_llm::common::getDevice();
    args.hw_info.sm_count = smCount;
    if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>)
    {
        args.scheduler.max_swizzle_size = 1;
    }
    if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>)
    {
        using Enum_t = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = Enum_t::Heuristic;
    }

    TLLM_CHECK_WITH_INFO(Gemm::can_implement(args), "[FLUX.3 gate_up SwiGLU NVFP4 GEMM] cannot implement m=%d n=%d k=%d",
        m, n, k);
    TLLM_CHECK_WITH_INFO(
        Gemm::get_workspace_size(args) == 0, "[FLUX.3 gate_up SwiGLU NVFP4 GEMM] unexpected workspace");
    typename Gemm::Params params = Gemm::to_underlying_arguments(args, nullptr);
    dim3 const grid = Gemm::get_grid_shape(params);
    dim3 const block = Gemm::get_block_shape();
    int const smem = Gemm::SharedStorageSize;
    static thread_local bool attrSet = false;
    if (!attrSet)
    {
        TLLM_CUDA_CHECK(
            cudaFuncSetAttribute(cutlass::device_kernel<Gemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attrSet = true;
    }
    cutlass::device_kernel<Gemm><<<grid, block, smem, stream>>>(params);
    TLLM_CUDA_CHECK(cudaGetLastError());
#endif
}

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
