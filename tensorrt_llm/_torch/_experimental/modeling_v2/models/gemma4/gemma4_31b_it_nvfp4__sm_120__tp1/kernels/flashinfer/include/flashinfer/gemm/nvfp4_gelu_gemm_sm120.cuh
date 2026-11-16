/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
// SM120 NVFP4 gate_up GEMM with a gelu_tanh(gate) * up + NVFP4-quantize epilogue.
//
// Mainloop: the CUTLASS SM120 block-scaled cooperative TMA mainloop with a 256x128x128 CTA tile,
// 1x1x1 cluster and StaticPersistentScheduler -- the same CollectiveBuilder arguments as
// TensorRT-LLM's FP4GemmRunner tactic 2 (nvfp4_nvfp4_gemm_template_sm120.h), so every accumulator
// is produced by the same MMA sequence along K.
//
// Weight layout: inside each 128-row N tile t, rows 32a + c (a in 0..3) hold gate (c < 16) or up
// (c >= 16) of output column j = 64t + 16a + (c % 16). Slice a of the tile (32 columns) therefore
// holds one 16-element NVFP4 block of gate and the matching block of up for every row.
//
// Epilogue (per tile, per slice a): every accumulator is turned into bf16_rn(fma(alpha, acc, 0)),
// the value the unfused GEMM's LinCombPerColBias epilogue (bias = 0, beta = 0) stores, and staged
// in shared memory as a 256 x 32 bf16 tile. Each of the 256 consumer threads then owns one row:
// it computes gelu_tanh(gate) * up, rounds to bf16, takes the block amax and quantizes exactly
// like _gelu_tanh_mul_fp4_kernel (tensorrt_llm/_torch/kernels/fused_ops/gelu_tanh_mul_fp4_quant.py),
// op for op (fma contraction of gate + c*gate^3 included), writing the e2m1 payload and the E4M3
// scale at its 128x4-swizzled offset. Scale rows in [M, pad128(M)) are written as zero.
// Compiled without fast-math so no op flushes denormals.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <stdexcept>
#include <string>

#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace flashinfer {
namespace gemm {
namespace gelu_epi {

using namespace cute;

__device__ __forceinline__ float rcp_approx_ftz(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Two e2m1 values in one byte, element `lo` in the low nibble.
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
  unsigned short r;
  asm volatile(
      "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
      : "=h"(r)
      : "f"(lo), "f"(hi));
  return static_cast<uint32_t>(r);
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

struct GeluQuantEpilogueOp {
  using ElementC = cutlass::bfloat16_t;  // nominal: no C operand is read
  using ElementD = cutlass::bfloat16_t;  // nominal: D is never written
  using ElementOutput = ElementD;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;
  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = StrideC;
  using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementD, 1, float, float>;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static constexpr int kTileM = 256;
  static constexpr int kTileN = 128;
  static constexpr int kSliceCols = 32;  // 16 gate + 16 up columns
  static constexpr int kNumThreads = 256;

  struct SharedStorage {
    alignas(128) uint16_t stage[kTileM * kSliceCols];  // bf16 bits, 16 KB
  };
  using TensorStorage = SharedStorage;

  struct Arguments {
    float const* alpha_ptr = nullptr;  // GEMM alpha (1 / (gs_act * gs_w))
    float const* gs_ptr = nullptr;     // down_proj input_scale (NVFP4 global scale)
    uint8_t* out_fp4 = nullptr;        // [M, N/4] packed e2m1
    uint8_t* out_sf = nullptr;         // [pad128(M) * nkt * 4] swizzled E4M3
    int64_t out_ld = 0;                // bytes per output row (N / 4)
    int64_t sf_nkt = 0;                // ceil((N/2/16) / 4)
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args,
                                                  void*) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*,
                                              cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto n = cute::get<1>(problem_shape);
    return (n % kTileN) == 0 && args.alpha_ptr != nullptr && args.gs_ptr != nullptr &&
           args.out_fp4 != nullptr && args.out_sf != nullptr;
  }

  CUTLASS_HOST_DEVICE
  explicit GeluQuantEpilogueOp(Params const& params_) : params(params_) {}

  CUTLASS_DEVICE
  bool is_source_needed() { return false; }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine,
            class FrgLayout, class TiledMma, class ResidueMNK>
  CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK,
                                 BlockCoordMNKL blk_coord_mnkl,
                                 cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
                                 TiledMma tiled_mma, ResidueMNK, int thread_idx, char* smem_buf) {
    static_assert(size<0>(BlockShapeMNK{}) == kTileM && size<1>(BlockShapeMNK{}) == kTileN,
                  "gelu epilogue expects a 256x128 CTA tile");
    static_assert(size(TiledMma{}) == kNumThreads, "gelu epilogue expects 256 MMA threads");

    int const M = static_cast<int>(get<0>(problem_shape_mnkl));
    int const m_coord = static_cast<int>(get<0>(blk_coord_mnkl));
    int const n_coord = static_cast<int>(get<1>(blk_coord_mnkl));

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor cD = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
    Tensor tCcD = thr_mma.partition_C(cD);
    CUTE_STATIC_ASSERT_V(size(tCcD) == size(accumulators));

    float const alpha = *params.alpha_ptr;
    float const gs = *params.gs_ptr;
    uint16_t* stage = reinterpret_cast<SharedStorage*>(smem_buf)->stage;

    int const row = thread_idx;  // 0..255: the tile row this thread quantizes
    int64_t const gm = int64_t(m_coord) * kTileM + row;
    uint32_t packed[8];
    uint32_t sfword = 0;
    float const rcp6 = rcp_approx_ftz(6.0f);
    float const rcpgs = rcp_approx_ftz(gs);

    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < kTileN / kSliceCols; ++a) {
      // 1) Stage slice a: bf16_rn(fma(alpha, acc, 0)), XOR-swizzled 16B chunks per row.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        int const m = get<0>(tCcD(i));
        int const n = get<1>(tCcD(i));
        if ((n >> 5) == a) {
          float const x = __fmaf_rn(alpha, accumulators(i), 0.0f);
          __nv_bfloat16 const b = __float2bfloat16_rn(x);
          int const c = n & 31;
          int const pq = (c >> 3) ^ ((m >> 1) & 3);
          stage[m * kSliceCols + pq * 8 + (c & 7)] = __bfloat16_as_ushort(b);
        }
      }
      cutlass::arch::NamedBarrier::sync(kNumThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

      // 2) This thread's row: 16 gate + 16 up bf16 values.
      uint4 ch[4];
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        int const pq = q ^ ((row >> 1) & 3);
        ch[q] = *reinterpret_cast<uint4 const*>(&stage[row * kSliceCols + pq * 8]);
      }
      uint32_t const* gw = reinterpret_cast<uint32_t const*>(&ch[0]);  // 8 words = 16 gate
      uint32_t const* uw = reinterpret_cast<uint32_t const*>(&ch[2]);  // 8 words = 16 up
      float v[16];
      float vmax = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < 16; ++e) {
        uint32_t const gbits = (gw[e >> 1] >> ((e & 1) * 16)) & 0xffffu;
        uint32_t const ubits = (uw[e >> 1] >> ((e & 1) * 16)) & 0xffffu;
        float const g = bf16_bits_to_float(gbits);
        float const u = bf16_bits_to_float(ubits);
        // gelu_tanh as the Triton act kernel's PTX computes it:
        //   mul(0.044715, g); mul(., g); fma(., g, g); mul(0.7978845608028654, .); tanh.approx
        float const c2 = __fmul_rn(__fmul_rn(0.044715f, g), g);
        float const s = __fmaf_rn(c2, g, g);
        float const inner = __fmul_rn(0.7978845608028654f, s);
        float const t = tanh_approx(inner);
        float const act = __fmul_rn(g, __fmul_rn(0.5f, __fadd_rn(1.0f, t)));
        float const vv = __bfloat162float(__float2bfloat16_rn(__fmul_rn(act, u)));
        v[e] = vv;
        vmax = fmaxf(vmax, fabsf(vv));
      }
      float const sfv = __fmul_rn(gs, __fmul_rn(vmax, rcp6));
      __nv_fp8_storage_t const s8 = __nv_cvt_float_to_fp8(sfv, __NV_SATFINITE, __NV_E4M3);
      __nv_fp8_e4m3 s8v;
      s8v.__x = s8;
      float const sf32 = static_cast<float>(s8v);
      float const oscale = vmax != 0.0f ? rcp_approx_ftz(__fmul_rn(sf32, rcpgs)) : 0.0f;
      uint32_t w0 = 0, w1 = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < 4; ++p) {
        w0 |= e2m1x2(__fmul_rn(v[2 * p], oscale), __fmul_rn(v[2 * p + 1], oscale)) << (8 * p);
        w1 |= e2m1x2(__fmul_rn(v[8 + 2 * p], oscale), __fmul_rn(v[8 + 2 * p + 1], oscale))
              << (8 * p);
      }
      packed[2 * a] = w0;
      packed[2 * a + 1] = w1;
      sfword |= static_cast<uint32_t>(s8) << (8 * a);

      // 3) The slice buffer is rewritten by the next slice (or the next tile).
      cutlass::arch::NamedBarrier::sync(kNumThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }

    // 64 output columns per tile -> 32 payload bytes and 4 scale bytes per row.
    if (gm < M) {
      uint4* dst = reinterpret_cast<uint4*>(params.out_fp4 + gm * params.out_ld + n_coord * 32);
      dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
      dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    }
    int64_t const m_pad = (int64_t(M) + 127) / 128 * 128;
    if (gm < m_pad) {
      int64_t const off = (gm / 128) * (params.sf_nkt * 512) + int64_t(n_coord) * 512 +
                          (gm % 32) * 16 + ((gm % 128) / 32) * 4;
      *reinterpret_cast<uint32_t*>(params.out_sf + off) = gm < M ? sfword : 0u;
    }
  }

 private:
  Params params;
};

struct GeluGemmSm120 {
  using CTAShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using ClusterShape = cute::Shape<_1, _1, _1>;
  using Arch = cutlass::arch::Sm120;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementAccumulator = float;

  using CollectiveEpilogue =
      cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<GeluQuantEpilogueOp>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, AlignmentA, ElementB,
      LayoutB, AlignmentB, ElementAccumulator, CTAShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, cutlass::gemm::StaticPersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr int kStages = CollectiveMainloop::DispatchPolicy::Stages;
};

inline int cached_sm_count() {
  static int sm_count = [] {
    int dev = 0, cnt = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cnt, cudaDevAttrMultiProcessorCount, dev);
    return cnt;
  }();
  return sm_count;
}

// Returns the CUTLASS workspace bytes when A, B and out_fp4 are all null; otherwise runs.
inline size_t runGeluGemmSm120(void const* A, void const* B, void const* sfa, void const* sfb,
                               float const* alpha, float const* gs, uint8_t* out_fp4,
                               uint8_t* out_sf, int m, int n, int k, char* workspace,
                               size_t workspace_bytes, cudaStream_t stream) {
  using Cfg = GeluGemmSm120;
  using Gemm = Cfg::Gemm;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(A);
  args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(B);
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sfa);
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sfb);
  args.mainloop.dA =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {m, k, 1});
  args.mainloop.dB =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
  args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);

  args.epilogue.alpha_ptr = alpha;
  args.epilogue.gs_ptr = gs;
  args.epilogue.out_fp4 = out_fp4;
  args.epilogue.out_sf = out_sf;
  args.epilogue.out_ld = n / 4;
  args.epilogue.sf_nkt = ((n / 2) / 16 + 3) / 4;

  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using Enum_t = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = Enum_t::Heuristic;
  }
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  args.hw_info.sm_count = cached_sm_count();

  Gemm gemm;
  if (A == nullptr && B == nullptr && out_fp4 == nullptr) {
    return gemm.get_workspace_size(args);
  }
  size_t const need = gemm.get_workspace_size(args);
  if (need > workspace_bytes) {
    throw std::runtime_error("[nvfp4_gelu_gemm_sm120] workspace too small: need " +
                             std::to_string(need) + ", got " + std::to_string(workspace_bytes));
  }
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("[nvfp4_gelu_gemm_sm120] can_implement: ") +
                             cutlassGetStatusString(st));
  }
  st = gemm.initialize(args, workspace, stream);
  if (st != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("[nvfp4_gelu_gemm_sm120] initialize: ") +
                             cutlassGetStatusString(st));
  }
  st = gemm.run(stream, nullptr, /*launch_with_pdl=*/false);
  if (st != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("[nvfp4_gelu_gemm_sm120] run: ") +
                             cutlassGetStatusString(st));
  }
  return need;
}

}  // namespace gelu_epi
}  // namespace gemm
}  // namespace flashinfer
