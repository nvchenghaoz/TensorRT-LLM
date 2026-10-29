/*
 * FLUX.3 SM120 FP8-rowwise GEMM (idea 122): D[M,N] = BF16(sx[m] * sw[n] * sum_k A[m,k] * B[n,k])
 * A E4M3 [M,K] row-major, B E4M3 [N,K] K-major, sx FP32 [M], sw FP32 [N], fp32 accumulation.
 *
 * Add-only overlay. Same math as torch._scaled_mm's SM120 rowwise CUTLASS kernel (Cooperative
 * 128x128x128, 2 smem stages): a sequential k32 MMA chain into one FP32 accumulator, then an EVT
 * epilogue of two separate FP32 RN multiplies and a BF16 RN cast. The epilogue association is a
 * template choice (order): 1 -> BF16(sw * (sx * acc)), 0 -> BF16(sx * (sw * acc)).
 * Arms (all CLC cutlass::gemm::PersistentScheduler, StageCountAutoCarveout, PDL):
 *   0 = A: Cooperative 128x128x64; 1 = C: Pingpong 128x128x64; 2 = B: Cooperative 128x128x128.
 * Entries: fp8r_gemm(a, b, sx, sw, out, workspace, arm, order), fp8r_stages(arm),
 *          fp8r_smem(arm), fp8r_workspace_size(m, n, k, arm).
 */
#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "flashinfer/cutlass_utils.cuh"
#include "tvm_ffi_utils.h"

namespace flux3_fp8r_sm120 {

using namespace cute;

template <class MainloopSchedule, class EpilogueSchedule, int TM, int TN, int TK, bool OuterW>
struct Fp8RowwiseSm120 {
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
  using Inner = std::conditional_t<OuterW, XScale, WScale>;
  using Outer = std::conditional_t<OuterW, WScale, XScale>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<
      MulD, Outer, cutlass::epilogue::fusion::Sm90EVT<MulF, Inner, Acc>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutD, AlignmentD, ElementD, LayoutD, AlignmentD, EpilogueSchedule,
      EVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB, LayoutB,
      AlignmentB, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <bool OuterW>
using ArmA = typename Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecializedCooperative, 128, 128,
                                      64, OuterW>::Gemm;
template <bool OuterW>
using ArmC = typename Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                      cutlass::epilogue::TmaWarpSpecialized, 128, 128, 64,
                                      OuterW>::Gemm;
template <bool OuterW>
using ArmB = typename Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecializedCooperative, 128, 128,
                                      128, OuterW>::Gemm;

template <bool OuterW>
using ArmD = typename Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecializedCooperative, 128, 256,
                                      64, OuterW>::Gemm;
template <bool OuterW>
using ArmE = typename Fp8RowwiseSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecializedCooperative, 256, 128,
                                      64, OuterW>::Gemm;

template <class Gemm>
typename Gemm::Arguments make_args(void const* a, void const* b, void* d, int m, int n, int k) {
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
  return args;
}

template <bool OuterW, template <bool> class Arm>
struct Runner {
  using Gemm = Arm<OuterW>;
  static void run(void const* a, void const* b, float const* sx, float const* sw, void* d, int m,
                  int n, int k, void* ws, size_t ws_bytes, cudaStream_t stream) {
    auto args = make_args<Gemm>(a, b, d, m, n, k);
    float const* outer = OuterW ? sw : sx;
    float const* inner = OuterW ? sx : sw;
    // Sm90EVT<MulD, Outer, Sm90EVT<MulF, Inner, Acc>>::Arguments is {Outer, {Inner, Acc, MulF},
    // MulD}; a broadcast's Arguments is {ptr, null_default, stride}.
    args.epilogue.thread = {{outer}, {{inner}, {}, {}}, {}};
    Gemm gemm;
    size_t need = gemm.get_workspace_size(args);
    TVM_FFI_ICHECK(need <= ws_bytes) << "workspace too small: need " << need << " have "
                                     << ws_bytes;
    auto st = gemm.can_implement(args);
    TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
        << "can_implement failed: " << cutlass::cutlassGetStatusString(st);
    st = gemm.initialize(args, ws, stream);
    TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
        << "initialize failed: " << cutlass::cutlassGetStatusString(st);
    st = gemm.run(stream, nullptr, /*launch_with_pdl=*/true);
    TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
        << "run failed: " << cutlass::cutlassGetStatusString(st);
  }
  static size_t workspace(int m, int n, int k) {
    auto args = make_args<Gemm>(nullptr, nullptr, nullptr, m, n, k);
    return Gemm::get_workspace_size(args);
  }
};

void fp8r_gemm(TensorView a, TensorView b, TensorView sx, TensorView sw, TensorView out,
               TensorView workspace, int64_t arm, int64_t order) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_float8_e4m3fn) << "a must be float8_e4m3fn";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_float8_e4m3fn) << "b must be float8_e4m3fn";
  TVM_FFI_ICHECK_EQ(sx.dtype(), dl_float32) << "sx must be float32";
  TVM_FFI_ICHECK_EQ(sw.dtype(), dl_float32) << "sw must be float32";
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16) << "out must be bf16";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  int m = static_cast<int>(a.size(0));
  int k = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(0));
  TVM_FFI_ICHECK_EQ(b.size(1), k) << "K mismatch";
  TVM_FFI_ICHECK_EQ(out.size(0), m);
  TVM_FFI_ICHECK_EQ(out.size(1), n);
  TVM_FFI_ICHECK_EQ(sx.numel(), m);
  TVM_FFI_ICHECK_EQ(sw.numel(), n);
  TVM_FFI_ICHECK(k % 16 == 0 && n % 8 == 0) << "alignment";
  size_t have = static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
  cudaStream_t stream = get_stream(a.device());
  auto const* sxp = static_cast<float const*>(sx.data_ptr());
  auto const* swp = static_cast<float const*>(sw.data_ptr());
  void* ws = workspace.data_ptr();
#define FLUX3_FP8R_RUN(ARM, OW)                                                             \
  Runner<OW, ARM>::run(a.data_ptr(), b.data_ptr(), sxp, swp, out.data_ptr(), m, n, k, ws, \
                       have, stream)
  if (arm == 0) {
    if (order) FLUX3_FP8R_RUN(ArmA, true); else FLUX3_FP8R_RUN(ArmA, false);
  } else if (arm == 1) {
    if (order) FLUX3_FP8R_RUN(ArmC, true); else FLUX3_FP8R_RUN(ArmC, false);
  } else if (arm == 2) {
    if (order) FLUX3_FP8R_RUN(ArmB, true); else FLUX3_FP8R_RUN(ArmB, false);
  } else if (arm == 3) {
    FLUX3_FP8R_RUN(ArmD, true);
  } else if (arm == 4) {
    FLUX3_FP8R_RUN(ArmE, true);
  } else {
    TVM_FFI_ICHECK(false) << "bad arm " << arm;
  }
#undef FLUX3_FP8R_RUN
}

int64_t fp8r_workspace_size(int64_t m, int64_t n, int64_t k, int64_t arm) {
  int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
  size_t s = 0;
  if (arm == 0) {
    s = Runner<true, ArmA>::workspace(mi, ni, ki);
  } else if (arm == 1) {
    s = Runner<true, ArmC>::workspace(mi, ni, ki);
  } else {
    s = Runner<true, ArmB>::workspace(mi, ni, ki);
  }
  return static_cast<int64_t>(s);
}

template <class Gemm>
int64_t stages_of() {
  return static_cast<int64_t>(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages);
}
template <class Gemm>
int64_t smem_of() {
  return static_cast<int64_t>(Gemm::GemmKernel::SharedStorageSize);
}

int64_t fp8r_stages(int64_t arm) {
  return arm == 0   ? stages_of<ArmA<true>>()
         : arm == 1 ? stages_of<ArmC<true>>()
         : arm == 2 ? stages_of<ArmB<true>>()
         : arm == 3 ? stages_of<ArmD<true>>()
                    : stages_of<ArmE<true>>();
}
int64_t fp8r_smem(int64_t arm) {
  return arm == 0   ? smem_of<ArmA<true>>()
         : arm == 1 ? smem_of<ArmC<true>>()
         : arm == 2 ? smem_of<ArmB<true>>()
         : arm == 3 ? smem_of<ArmD<true>>()
                    : smem_of<ArmE<true>>();
}

}  // namespace flux3_fp8r_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8r_gemm, flux3_fp8r_sm120::fp8r_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8r_workspace_size, flux3_fp8r_sm120::fp8r_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8r_stages, flux3_fp8r_sm120::fp8r_stages);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8r_smem, flux3_fp8r_sm120::fp8r_smem);
