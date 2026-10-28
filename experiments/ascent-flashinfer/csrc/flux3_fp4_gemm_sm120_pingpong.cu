/*
 * FLUX.3 SM120 NVFP4 strided GEMM entries (add-only; idea 79).
 *
 * Same CUTLASS SM120 block-scaled NVFP4 kernel as FlashInfer's stock 128x128x128 DP instantiation
 * (fp4_gemm_template_sm120.h: LinearCombination<bf16,float,void,float>, ElementC void, TMA
 * epilogue, StageCountAutoCarveout, StaticPersistentScheduler, cluster 1x1x1), with the
 * mainloop schedule as a template parameter:
 *   - flux3_fp4_gemm_into_coop:     KernelTmaWarpSpecializedCooperative (stock schedule, control)
 *   - flux3_fp4_gemm_into_pingpong: KernelTmaWarpSpecializedPingpong (2 consumer warpgroups
 *     alternate tiles, so one tile's epilogue overlaps the next tile's mainloop)
 * D is written at out_base + offset_elems with row stride ldd (dD = (ldd, 1, ldd*m)).
 */
#include <cuda_bf16.h>

#include <cstdint>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"
#include "tvm_ffi_utils.h"

namespace flux3_sm120_pp {

using namespace cute;

template <class MainloopSchedule, class EpilogueSchedule, int TM, int TN, int TK>
struct Fp4GemmSm120 {
  using OutElementType = cutlass::bfloat16_t;
  using Arch = cutlass::arch::Sm120;
  using ClusterShape = cute::Shape<_1, _1, _1>;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using FusionOperation =
      cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>;
  using ThreadBlockShape = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassTensorOp, ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentC, EpilogueSchedule,
      FusionOperation>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::StaticPersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

#ifndef FLUX3_PP_ONLY
using CoopGemm = Fp4GemmSm120<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                              cutlass::epilogue::TmaWarpSpecialized, 128, 128, 128>::Gemm;
#endif
using PingpongGemm = Fp4GemmSm120<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                  cutlass::epilogue::TmaWarpSpecialized, 128, 128, 128>::Gemm;

template <class Gemm>
void run_strided(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
                 TensorView out_base, int64_t offset_elems, int64_t ldd, TensorView workspace) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be uint8 (packed e2m1)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be uint8 (packed e2m1)";
  TVM_FFI_ICHECK_EQ(sfa.dtype(), dl_uint8) << "sfa must be uint8";
  TVM_FFI_ICHECK_EQ(sfb.dtype(), dl_uint8) << "sfb must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(out_base.dtype(), dl_bfloat16) << "out must be bf16";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  int m = static_cast<int>(a.size(0));
  int k = static_cast<int>(a.size(1)) * 2;
  int n = static_cast<int>(b.size(0));
  TVM_FFI_ICHECK_EQ(b.size(1) * 2, k) << "K mismatch";
  TVM_FFI_ICHECK(ldd >= n && ldd % 8 == 0) << "bad ldd";
  TVM_FFI_ICHECK(offset_elems % 8 == 0) << "offset must be 16B aligned";
  TVM_FFI_ICHECK(offset_elems + (int64_t)(m - 1) * ldd + n <= out_base.numel())
      << "output range exceeds out_base";

  auto* d = static_cast<__nv_bfloat16*>(out_base.data_ptr()) + offset_elems;
  Gemm gemm;
  auto args = flashinfer::gemm::prepareGemmArgsImpl<Gemm>(
      d, a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      static_cast<float const*>(alpha.data_ptr()), m, n, k, 1);
  cute::get<0>(args.epilogue.dD) = ldd;
  cute::get<2>(args.epilogue.dD) = ldd * m;
  args.epilogue.dC = args.epilogue.dD;

  size_t need = gemm.get_workspace_size(args);
  size_t have = static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
  TVM_FFI_ICHECK(need <= have) << "workspace too small: need " << need << " have " << have;
  auto st = gemm.can_implement(args);
  TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
      << "can_implement failed: " << cutlass::cutlassGetStatusString(st);
  cudaStream_t stream = get_stream(a.device());
  char* ws = static_cast<char*>(workspace.data_ptr());
  st = gemm.initialize(args, ws, stream);
  TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
      << "initialize failed: " << cutlass::cutlassGetStatusString(st);
  st = gemm.run(args, ws, stream, nullptr, /*enablePDL=*/true);
  TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
      << "run failed: " << cutlass::cutlassGetStatusString(st);
}

#ifndef FLUX3_PP_ONLY
void gemm_into_coop(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
                    TensorView out_base, int64_t offset_elems, int64_t ldd,
                    TensorView workspace) {
  run_strided<CoopGemm>(a, b, sfa, sfb, alpha, out_base, offset_elems, ldd, workspace);
}
#endif

void gemm_into_pingpong(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                        TensorView alpha, TensorView out_base, int64_t offset_elems, int64_t ldd,
                        TensorView workspace) {
  run_strided<PingpongGemm>(a, b, sfa, sfb, alpha, out_base, offset_elems, ldd, workspace);
}

int64_t smem_pingpong() { return static_cast<int64_t>(PingpongGemm::GemmKernel::SharedStorageSize); }
int64_t stages_pingpong() {
  return static_cast<int64_t>(PingpongGemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages);
}
#ifndef FLUX3_PP_ONLY
int64_t smem_coop() { return static_cast<int64_t>(CoopGemm::GemmKernel::SharedStorageSize); }
int64_t stages_coop() {
  return static_cast<int64_t>(CoopGemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages);
}
#endif

}  // namespace flux3_sm120_pp

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_into_pingpong, flux3_sm120_pp::gemm_into_pingpong);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(smem_pingpong, flux3_sm120_pp::smem_pingpong);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stages_pingpong, flux3_sm120_pp::stages_pingpong);
#ifndef FLUX3_PP_ONLY
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_into_coop, flux3_sm120_pp::gemm_into_coop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(smem_coop, flux3_sm120_pp::smem_coop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stages_coop, flux3_sm120_pp::stages_coop);
#endif
