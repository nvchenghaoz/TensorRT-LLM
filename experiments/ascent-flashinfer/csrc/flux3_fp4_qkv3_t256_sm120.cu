/*
 * FLUX.3 SM120 NVFP4 batched L=3 QKV GEMM at TileShape 256x128x128 (idea 17, large-M variant).
 *
 * Add-only module (flux3_fp4_qkv3_t256_sm120); main's flux3_fp4_gemm_sm120_pingpong_dyn.cu is
 * untouched. Same contract as its gemm_qkv3_pingpong (A and its scales shared by the 3 planes,
 * per-plane alpha, planar BF16 D), but the CUTLASS SM120 block-scaled kernel uses a 256x128x128
 * FP4 tile with the Cooperative mainloop (two consumer warpgroups, 128x128 each, as FlashInfer's
 * stock tactic 25). Every output element sees the same k64 MMA chain over K and the same
 * BF16(alpha * acc) epilogue as the 128x128x128 kernels, so the output is bitwise equal to main's.
 * At M ~ 16.9K / 33.3K a 256-row tile halves B traffic per FLOP; at M ~ 4K main's kernel stays.
 * Entries: gemm_qkv3_t256 (dynamic: 1 = CLC PersistentScheduler, 0 = StaticPersistent),
 * qkv3_t256_workspace_size, t256_info (stages * 1e6 + smem bytes).
 */
#include <cuda_bf16.h>

#include <cstdint>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"
#include "tvm_ffi_utils.h"

namespace flux3_sm120_qkv3_t256 {

using namespace cute;

template <class TileSchedulerTag>
struct Fp4GemmSm120T256 {
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
  using ThreadBlockShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassTensorOp, ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentC,
      cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, TileSchedulerTag>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using GemmDyn = Fp4GemmSm120T256<cutlass::gemm::PersistentScheduler>::Gemm;
using GemmStatic = Fp4GemmSm120T256<cutlass::gemm::StaticPersistentScheduler>::Gemm;

template <class Args>
void set_sched(Args& args, int64_t raster, int64_t swizzle) {
  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = static_cast<int>(swizzle);
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using Enum_t = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order =
        raster == 1 ? Enum_t::AlongM : (raster == 2 ? Enum_t::AlongN : Enum_t::Heuristic);
  }
}

template <class Gemm>
auto qkv3_args(void* out, void const* a, void const* b, void const* sfa, void const* sfb,
               float const* alpha, int m, int n, int k, int64_t raster, int64_t swizzle) {
  auto args = flashinfer::gemm::prepareGemmArgsImpl<Gemm>(out, a, b, sfa, sfb, alpha, m, n, k, 3);
  // A and its scales are shared by the 3 planes: batch stride 0.
  cute::get<2>(args.mainloop.dA) = 0;
  cute::get<1>(cute::get<2>(args.mainloop.layout_SFA.stride())) = 0;
  // Per-plane alpha.
  args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 1};
  set_sched(args, raster, swizzle);
  return args;
}

template <class Gemm>
void run_qkv3(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
              TensorView out, TensorView workspace, int64_t raster, int64_t swizzle) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be uint8 (packed e2m1)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be uint8 (packed e2m1)";
  TVM_FFI_ICHECK_EQ(sfa.dtype(), dl_uint8) << "sfa must be uint8";
  TVM_FFI_ICHECK_EQ(sfb.dtype(), dl_uint8) << "sfb must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16) << "out must be bf16";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  TVM_FFI_ICHECK_EQ(alpha.numel(), 3) << "alpha must hold 3 plane scales";
  int m = static_cast<int>(a.size(0));
  int k = static_cast<int>(a.size(1)) * 2;
  TVM_FFI_ICHECK_EQ(b.size(0) % 3, 0) << "weight rows must be 3 planes";
  int n = static_cast<int>(b.size(0) / 3);
  TVM_FFI_ICHECK_EQ(b.size(1) * 2, k) << "K mismatch";
  TVM_FFI_ICHECK_EQ(n % 128, 0) << "plane width must be a multiple of 128 (SF atom)";
  TVM_FFI_ICHECK_EQ(k % 64, 0) << "K must be a multiple of 64 (SF atom)";
  TVM_FFI_ICHECK_EQ(sfb.numel() % 3, 0) << "sfb must hold 3 equal planes";
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(sfb.numel()) / 3, static_cast<int64_t>(n) * (k / 16))
      << "sfb plane size mismatch";
  TVM_FFI_ICHECK(out.numel() >= static_cast<int64_t>(3) * m * n) << "out too small";

  Gemm gemm;
  auto args = qkv3_args<Gemm>(out.data_ptr(), a.data_ptr(), b.data_ptr(), sfa.data_ptr(),
                              sfb.data_ptr(), static_cast<float const*>(alpha.data_ptr()), m, n,
                              k, raster, swizzle);
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
  st = gemm.run(stream, nullptr, /*launch_with_pdl=*/true);
  TVM_FFI_ICHECK(st == cutlass::Status::kSuccess)
      << "run failed: " << cutlass::cutlassGetStatusString(st);
}

void gemm_qkv3_t256(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
                    TensorView out, TensorView workspace, int64_t raster, int64_t swizzle,
                    int64_t dynamic) {
  if (dynamic) {
    run_qkv3<GemmDyn>(a, b, sfa, sfb, alpha, out, workspace, raster, swizzle);
  } else {
    run_qkv3<GemmStatic>(a, b, sfa, sfb, alpha, out, workspace, raster, swizzle);
  }
}

int64_t qkv3_t256_workspace_size(int64_t m, int64_t n, int64_t k, int64_t dynamic) {
  if (dynamic) {
    auto args = qkv3_args<GemmDyn>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                   static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                   0, 1);
    return static_cast<int64_t>(GemmDyn::get_workspace_size(args));
  }
  auto args = qkv3_args<GemmStatic>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                    0, 1);
  return static_cast<int64_t>(GemmStatic::get_workspace_size(args));
}

int64_t t256_info(int64_t dynamic) {
  if (dynamic) {
    return static_cast<int64_t>(GemmDyn::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages) *
               1000000 +
           static_cast<int64_t>(GemmDyn::GemmKernel::SharedStorageSize);
  }
  return static_cast<int64_t>(GemmStatic::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages) *
             1000000 +
         static_cast<int64_t>(GemmStatic::GemmKernel::SharedStorageSize);
}

}  // namespace flux3_sm120_qkv3_t256

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_qkv3_t256, flux3_sm120_qkv3_t256::gemm_qkv3_t256);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv3_t256_workspace_size,
                              flux3_sm120_qkv3_t256::qkv3_t256_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(t256_info, flux3_sm120_qkv3_t256::t256_info);
