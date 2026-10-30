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

// FLUX.3 SM120 FP8 row-wise GEMM (the BF16 checkpoint-excluded block content_mode_blocks.image.0, run
// as per-token x per-channel E4M3): D[M,N] = BF16(sw[n] * (sx[m] * sum_k A[m,k] * B[n,k])).
//
// Same math as torch._scaled_mm's SM120 row-wise CUTLASS kernel (Cooperative 128x128x128, 2 smem
// stages): a sequential k32 MMA chain into one fp32 accumulator, then an EVT epilogue of two separate
// fp32 RN multiplies and a BF16 RN cast, associated as BF16(sw * (sx * acc)) -- the order that is
// bitwise torch._scaled_mm on every image.0 shape (sx * (sw * acc) differs in ~5e-6 of the elements by
// 1 ulp). The tiles differ (only the CTA -> output map changes, never an element's chain):
//   arm 1: Pingpong 128x128x64 (M < 8192),
//   arm 3: Cooperative 128x256x64 (M >= 8192, N >= 12288: gate_up),
//   arm 4: Cooperative 256x128x64 (M >= 8192, N < 12288: QKV, down, attn_out),
// all on the CLC persistent scheduler, launched with PDL.

#include "flux3Fp8RowwiseGemmSm120.h"
#include "tensorrt_llm/common/assert.h"

#ifndef PLACEHOLDER_KERNELS

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include <type_traits>

#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::flux3
{
namespace fp8Rowwise
{

using namespace cute;

template <class MainloopSchedule, class EpilogueSchedule, int TM, int TN, int TK, int Stages>
struct Fp8RowwiseSm120
{
    using Arch = cutlass::arch::Sm120;
    using ElementA = cutlass::float_e4m3_t;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 16;
    using ElementB = cutlass::float_e4m3_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 16;
    using ElementD = cutlass::bfloat16_t;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 8;
    using ElementC = void;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using TileShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
    using ClusterShape = cute::Shape<_1, _1, _1>;

    static constexpr auto RN = cutlass::FloatRoundStyle::round_to_nearest;
    using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float>;
    using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float>;
    using Acc = cutlass::epilogue::fusion::Sm90AccFetch;
    using MulF = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float, RN>;
    using MulD = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float, RN>;
    // BF16(sw * (sx * acc)): the weight scale is the outer multiply.
    using EVT = cutlass::epilogue::fusion::Sm90EVT<MulD, WScale, cutlass::epilogue::fusion::Sm90EVT<MulF, XScale, Acc>>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<Arch,
        cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute, ElementC, LayoutD, AlignmentD, ElementD, LayoutD, AlignmentD,
        EpilogueSchedule, EVT>::CollectiveOp;

    template <class StageCountType>
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<Arch, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        StageCountType, MainloopSchedule>::CollectiveOp;
    using CollectiveMainloop = Mainloop<cutlass::gemm::collective::StageCount<Stages>>;
    static_assert(Mainloop<cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                          sizeof(typename CollectiveEpilogue::SharedStorage))>>::DispatchPolicy::Stages
            == Stages,
        "pinned stage count differs from the auto carve-out the kernel was measured with");

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Mainloop stages, pinned to what StageCountAutoCarveout<sizeof(epilogue SharedStorage)> gives (CUTLASS
// 4.4.2 and 4.5 compute it the same way): sm120 smem 101376 B minus the epilogue (two 64x32 bf16 D
// stages = 8 KB plus the [TN] fp32 row-broadcast scales and the barriers, 8.7-10.3 KB). One stage of the
// 128x128x64 tile = A 8192 + B 8192 + pipeline 16 = 16400 B -> 5 stages; of the 128x256x64 / 256x128x64
// tiles = 24576 + 16 = 24592 B -> 3 stages.
using ArmC = Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized,
    128, 128, 64, 5>;
using ArmD = Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative, 128, 256, 64, 3>;
using ArmE = Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative, 256, 128, 64, 3>;

template <class Arm>
size_t runArm(void const* a, void const* b, float const* sx, float const* sw, void* d, int m, int n, int k,
    char* workspace, size_t workspaceBytes, cudaStream_t stream)
{
    using Gemm = typename Arm::Gemm;
    using GK = typename Gemm::GemmKernel;
    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);
    args.mainloop.ptr_A = static_cast<cutlass::float_e4m3_t const*>(a);
    args.mainloop.ptr_B = static_cast<cutlass::float_e4m3_t const*>(b);
    args.mainloop.dA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {m, k, 1});
    args.mainloop.dB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {n, k, 1});
    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {m, n, 1});
    args.epilogue.ptr_D = static_cast<cutlass::bfloat16_t*>(d);
    args.epilogue.dD = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {m, n, 1});
    args.hw_info.cluster_shape = dim3(1, 1, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    // Sm90EVT<MulD, WScale, Sm90EVT<MulF, XScale, Acc>>::Arguments is {WScale, {XScale, Acc, MulF},
    // MulD}; a broadcast's Arguments is {ptr, null_default, stride}.
    args.epilogue.thread = {{sw}, {{sx}, {}, {}}, {}};
    Gemm gemm;
    size_t const need = gemm.get_workspace_size(args);
    if (a == nullptr && b == nullptr && d == nullptr)
    {
        return need;
    }
    TLLM_CHECK_WITH_INFO(
        need <= workspaceBytes, "[FLUX.3 FP8 row-wise GEMM] workspace too small: need %zu, have %zu", need, workspaceBytes);
    auto st = gemm.can_implement(args);
    TLLM_CHECK_WITH_INFO(st == cutlass::Status::kSuccess, "[FLUX.3 FP8 row-wise GEMM] cannot implement m=%d n=%d k=%d: %s",
        m, n, k, cutlass::cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace, stream);
    TLLM_CHECK_WITH_INFO(st == cutlass::Status::kSuccess, "[FLUX.3 FP8 row-wise GEMM] initialize failed: %s",
        cutlass::cutlassGetStatusString(st));
    st = gemm.run(stream, nullptr, /*launch_with_pdl=*/true);
    TLLM_CHECK_WITH_INFO(
        st == cutlass::Status::kSuccess, "[FLUX.3 FP8 row-wise GEMM] run failed: %s", cutlass::cutlassGetStatusString(st));
    return need;
}

} // namespace fp8Rowwise
} // namespace kernels::flux3

TRTLLM_NAMESPACE_END

#endif // PLACEHOLDER_KERNELS

TRTLLM_NAMESPACE_BEGIN

namespace kernels::flux3
{

size_t fp8RowwiseGemmSm120(void const* a, void const* b, float const* sx, float const* sw, void* d, int m, int n,
    int k, int arm, char* workspace, size_t workspaceBytes, cudaStream_t stream)
{
#ifdef PLACEHOLDER_KERNELS
    TLLM_THROW("[FLUX.3 FP8 row-wise GEMM] not compiled for this architecture.");
#else
    using namespace fp8Rowwise;
    switch (arm)
    {
    case 1: return runArm<ArmC>(a, b, sx, sw, d, m, n, k, workspace, workspaceBytes, stream);
    case 3: return runArm<ArmD>(a, b, sx, sw, d, m, n, k, workspace, workspaceBytes, stream);
    case 4: return runArm<ArmE>(a, b, sx, sw, d, m, n, k, workspace, workspaceBytes, stream);
    default: TLLM_THROW("[FLUX.3 FP8 row-wise GEMM] arm must be 1, 3 or 4, got %d", arm);
    }
#endif
}

} // namespace kernels::flux3

TRTLLM_NAMESPACE_END
