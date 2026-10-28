/*
 * FLUX.3 SM120 NVFP4 gate_up GEMM with a register SwiGLU -> NVFP4 epilogue (add-only module).
 *
 * Mainloop: CUTLASS SM120 block-scaled TMA warp-specialized mainloop (the same collective
 * FlashInfer's fp4_gemm instantiates), Pingpong schedule with tile 128x128x128 (FP4 elements) by
 * default (FLUX3_SCHED=0: cooperative 256x128x128), 1x1x1 cluster, StaticPersistent scheduler,
 * pinned (no autotune). The only change to the stock collective is the N permutation of the
 * TiledMma: each warp owns 32 contiguous accumulator columns (one 16-output NVFP4 scale block of
 * the pair-interleaved value/gate weight) instead of 16, so the scale-block amax is a quad
 * shuffle. The K order of the mma.sync chain is unchanged.
 *
 * Epilogue (registers only), per output element, matching the CuTe PACKED kernel's PTX:
 *   value = acc_even * alpha, gate = acc_odd * alpha            (mul.rn.f32)
 *   h = (value * gate) / (ex2.approx.ftz(gate * -log2e) + 1)    (mul.rn, ex2.approx.ftz, add.rn, div.rn)
 *   amax over the 16 outputs of the block; enc = e4m3(amax * inv / 6)   (cvt.rn.satfinite.e4m3x2)
 *   mult = inv / max(float(enc), 1e-12)                          (div.rn)
 *   byte = e2m1x2(high = h_odd * mult, low = h_even * mult)       (cvt.rn.satfinite.e2m1x2)
 * FP4 data: one 64-bit store per (row, scale block). Scales: 128x4-swizzled layout.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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
#include "tvm_ffi_utils.h"

namespace flux3_swiglu_fp4 {

using namespace cute;

// ---------------------------------------------------------------------------------------------
// Exact-form PTX helpers (the module is compiled with -use_fast_math; inline PTX pins the forms)
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float div_rn(float a, float b) {
  float d;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float ex2_approx_ftz(float a) {
  float d;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float abs_f32(float a) {
  float d;
  asm("abs.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float max_f32(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
// E4M3 byte of x (cvt.rn.satfinite.e4m3x2.f32 with x in the low half).
__device__ __forceinline__ uint32_t e4m3_byte(float x) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(x));
  return static_cast<uint32_t>(r) & 0xFFu;
}
// Exact float value of an E4M3 byte.
__device__ __forceinline__ float e4m3_to_f32(uint32_t byte) {
  uint32_t h2;
  uint16_t in = static_cast<uint16_t>(byte);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
  __half lo = __ushort_as_half(static_cast<unsigned short>(h2 & 0xFFFFu));
  return __half2float(lo);
}
// Byte = (e2m1(high) << 4) | e2m1(low).
__device__ __forceinline__ uint32_t e2m1x2_byte(float high, float low) {
  uint32_t r;
  asm(
      "{\n"
      ".reg .b8 t;\n"
      "cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n"
      "cvt.u32.u8 %0, t;\n"
      "}\n"
      : "=r"(r)
      : "f"(high), "f"(low));
  return r;
}

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float d;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

// Branch-free a / b, correctly rounded (so bitwise equal to div.rn.f32) when both operands are
// normal with exponents in [2^-100, 2^100]: reciprocal refinement plus two FMA remainder
// corrections (the fast path ptxas emits for div.rn). div_safe() says when it applies; callers
// batch the fallback to div.rn so the hot path has no per-division branch.
__device__ __forceinline__ float div_fast(float a, float b) {
  float r = rcp_approx_ftz(b);
  float e = fma_rn(-b, r, 1.0f);
  r = fma_rn(r, e, r);
  float q = mul_rn(a, r);
  float rem = fma_rn(-b, q, a);
  q = fma_rn(r, rem, q);
  rem = fma_rn(-b, q, a);
  q = fma_rn(r, rem, q);
  return q;
}
__device__ __forceinline__ bool div_safe(float a, float b) {
  uint32_t ea = (__float_as_uint(a) >> 23) & 0xFFu;
  uint32_t eb = (__float_as_uint(b) >> 23) & 0xFFu;
  return (ea - 27u) <= 200u && (eb - 27u) <= 200u;
}

// 0: div.rn.f32 (default; bitwise by construction). 1: branch-free division with a batched div.rn
// fallback (measured slower, job 971, and its safety range admits denormal quotients).
#ifndef FLUX3_DIV_MODE
#define FLUX3_DIV_MODE 0
#endif
// 1 (default): Pingpong 128x128x128 (two 4-warp consumer warpgroups alternate tiles, so one's
// epilogue overlaps the other's MMA). 0: cooperative 256x128x128 (8 MMA warps, FP4GemmRunner
// t2's tile): its epilogue is exposed after every tile and it lost to the CuTe kernel (job 1013).
#ifndef FLUX3_SCHED
#define FLUX3_SCHED 1
#endif
// Tile scheduler. 0 (default): StaticPersistentScheduler (sm_count persistent CTAs, fixed tile
// stride). 1: cutlass::gemm::PersistentScheduler, which CUTLASS maps on Sm120 to
// PersistentTileSchedulerSm100 (CLC dynamic work stealing). Same smem, no workspace, bitwise.
#ifndef FLUX3_TILESCHED
#define FLUX3_TILESCHED 0
#endif

struct SwigluParts {
  float prod;
  float denom;
};

__device__ __forceinline__ SwigluParts swiglu_parts(float acc_even, float acc_odd, float factor) {
#ifdef FLUX3_DIAG_NOMATH
  return {mul_rn(factor, acc_even), 1.0f};  // diagnostic only: no SwiGLU math
#endif
  float const kNegLog2e = __int_as_float(0xBFB8AA3B);  // -1.4426950f, as the CuTe kernel's PTX
  float value = mul_rn(factor, acc_even);
  float gate = mul_rn(factor, acc_odd);
  float prod = mul_rn(value, gate);
  float e = ex2_approx_ftz(mul_rn(gate, kNegLog2e));
  float denom = add_rn(e, 1.0f);
  return {prod, denom};
}

__device__ __forceinline__ float swiglu_h(float acc_even, float acc_odd, float factor) {
  SwigluParts p = swiglu_parts(acc_even, acc_odd, factor);
#ifdef FLUX3_DIAG_NOMATH
  return p.prod;
#endif
  return div_rn(p.prod, p.denom);
}

// ---------------------------------------------------------------------------------------------
// Epilogue (operator-style, wrapped by Sm90TmaWarpSpecializedAdapter)
// ---------------------------------------------------------------------------------------------
template <class StrideC_, class StrideD_, class EpilogueSchedule_>
class SwigluFp4Epilogue {
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

  struct SharedStorage {};
  using TensorStorage = SharedStorage;

  struct Arguments {
    uint8_t* out = nullptr;           // [rows_padded, N/4] FP4 bytes (pairs of outputs)
    uint8_t* out_sf = nullptr;        // 128x4-swizzled E4M3 scales, rows_padded x (N/32)
    float const* alpha = nullptr;     // input_gs * weight_gs (device scalar)
    float const* inv_scale = nullptr; // next projection's inverse input scale (device scalar)
    int rows_padded = 0;              // rows rounded up to 128
    int out_row_bytes = 0;            // N / 4
    int sf_cols = 0;                  // N / 32 (multiple of 4)
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
                                              cudaStream_t, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  SwigluFp4Epilogue(Params const& params_, SharedStorage const& = SharedStorage())
      : params(params_) {}

  CUTLASS_DEVICE bool is_source_needed() { return false; }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine,
            class FrgLayout, class TiledMma, class ResidueMNK>
  CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK blk_shape_MNK,
                                 BlockCoordMNKL blk_coord_mnkl,
                                 cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
                                 TiledMma tiled_mma, ResidueMNK, int thread_idx, char*) {
    constexpr int BM = size<0>(BlockShapeMNK{});
    constexpr int BN = size<1>(BlockShapeMNK{});
    static_assert(size<0>(FrgLayout{}) == 4, "m16n8 C fragment expected");
    constexpr int MMA_M = size<1>(FrgLayout{});
    constexpr int MMA_N = size<2>(FrgLayout{});
    static_assert(MMA_N % 4 == 0, "each warp must own whole 32-column scale blocks");

    auto m_coord = get<0>(blk_coord_mnkl);
    auto n_coord = get<1>(blk_coord_mnkl);

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto cD = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto tCcD = thr_mma.partition_C(cD);  // (4, MMA_M, MMA_N) -> (row, col) in tile

    float const factor = *params.alpha;
    float const inverse = *params.inv_scale;
    int const lane = threadIdx.x & 31;
    int const q = lane & 3;
    int const row_base = static_cast<int>(m_coord) * BM;
    int const col_base = static_cast<int>(n_coord) * BN;

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < MMA_M; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int nb = 0; nb < MMA_N / 4; ++nb) {
        float h[2][4];
#if FLUX3_DIV_MODE == 1
        SwigluParts parts[2][4];
        bool ok = true;
        CUTLASS_PRAGMA_UNROLL
        for (int hh = 0; hh < 2; ++hh) {
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < 4; ++r) {
            parts[hh][r] = swiglu_parts(accumulators(2 * hh, mi, nb * 4 + r),
                                        accumulators(2 * hh + 1, mi, nb * 4 + r), factor);
            h[hh][r] = div_fast(parts[hh][r].prod, parts[hh][r].denom);
            ok &= div_safe(parts[hh][r].prod, parts[hh][r].denom);
          }
        }
        if (!ok) {
          CUTLASS_PRAGMA_UNROLL
          for (int hh = 0; hh < 2; ++hh) {
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < 4; ++r) h[hh][r] = div_rn(parts[hh][r].prod, parts[hh][r].denom);
          }
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int hh = 0; hh < 2; ++hh) {
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < 4; ++r) {
            h[hh][r] = swiglu_h(accumulators(2 * hh, mi, nb * 4 + r),
                                accumulators(2 * hh + 1, mi, nb * 4 + r), factor);
          }
        }
#endif
        float amax[2];
        CUTLASS_PRAGMA_UNROLL
        for (int hh = 0; hh < 2; ++hh) {
          float m = max_f32(0.0f, abs_f32(h[hh][0]));
          CUTLASS_PRAGMA_UNROLL
          for (int r = 1; r < 4; ++r) m = max_f32(m, abs_f32(h[hh][r]));
          m = max_f32(m, __shfl_xor_sync(0xffffffffu, m, 1));
          amax[hh] = max_f32(m, __shfl_xor_sync(0xffffffffu, m, 2));
        }
        uint32_t enc[2];
        float mults[2];
#if FLUX3_DIV_MODE == 1
        {
          float x0 = mul_rn(amax[0], inverse);
          float x1 = mul_rn(amax[1], inverse);
          float s0 = div_fast(x0, 6.0f);
          float s1 = div_fast(x1, 6.0f);
          if (!(div_safe(x0, 6.0f) && div_safe(x1, 6.0f))) {
            s0 = div_rn(x0, 6.0f);
            s1 = div_rn(x1, 6.0f);
          }
          enc[0] = e4m3_byte(s0);
          enc[1] = e4m3_byte(s1);
          float d0 = max_f32(e4m3_to_f32(enc[0]), 1.0e-12f);
          float d1 = max_f32(e4m3_to_f32(enc[1]), 1.0e-12f);
          mults[0] = div_fast(inverse, d0);
          mults[1] = div_fast(inverse, d1);
          if (!(div_safe(inverse, d0) && div_safe(inverse, d1))) {
            mults[0] = div_rn(inverse, d0);
            mults[1] = div_rn(inverse, d1);
          }
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int hh = 0; hh < 2; ++hh) {
          enc[hh] = e4m3_byte(div_rn(mul_rn(amax[hh], inverse), 6.0f));
          mults[hh] = div_rn(inverse, max_f32(e4m3_to_f32(enc[hh]), 1.0e-12f));
        }
#endif
        uint32_t word_a[2];
        CUTLASS_PRAGMA_UNROLL
        for (int hh = 0; hh < 2; ++hh) {
          float const mult = mults[hh];
          uint32_t w = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < 4; ++r) {
            float const mine = h[hh][r];
            float const partner = __shfl_xor_sync(0xffffffffu, mine, 1);
            float const lo = (q & 1) ? partner : mine;
            float const hi = (q & 1) ? mine : partner;
            w |= e2m1x2_byte(mul_rn(hi, mult), mul_rn(lo, mult)) << (8 * r);
          }
          word_a[hh] = w;  // lanes q=0,1: bytes 0,2,4,6; lanes q=2,3: bytes 1,3,5,7
        }
        uint32_t word_b0 = __shfl_xor_sync(0xffffffffu, word_a[0], 2);
        uint32_t word_b1 = __shfl_xor_sync(0xffffffffu, word_a[1], 2);

        int const hh_store = q & 1;
        int const row = row_base + get<0>(tCcD(2 * hh_store, mi, nb * 4));
        int const col = col_base + get<1>(tCcD(0, mi, nb * 4));
        int const block_col = col >> 5;
        if (row < params.rows_padded) {
          if (q < 2) {
            uint32_t a = hh_store ? word_a[1] : word_a[0];
            uint32_t b = hh_store ? word_b1 : word_b0;
            uint2 v;
            v.x = __byte_perm(a, b, 0x5140);
            v.y = __byte_perm(a, b, 0x7362);
            *reinterpret_cast<uint2*>(params.out + static_cast<int64_t>(row) * params.out_row_bytes +
                                      block_col * 8) = v;
          } else {
            int64_t const idx = (block_col & 3) + (block_col >> 2) * 512 + (row & 31) * 16 +
                                ((row & 127) >> 5) * 4 +
                                static_cast<int64_t>(row >> 7) * 128 * params.sf_cols;
            params.out_sf[idx] = static_cast<uint8_t>(hh_store ? enc[1] : enc[0]);
          }
        }
      }
    }
  }

 private:
  Params params;
};

// ---------------------------------------------------------------------------------------------
// Kernel types
// ---------------------------------------------------------------------------------------------
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
#if FLUX3_SCHED == 1
using TileShape = Shape<_128, _128, _128>;
using BuilderSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using AtomLayoutMNK = Layout<Shape<_2, _2, _1>>;
#else
using TileShape = Shape<_256, _128, _128>;
using BuilderSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
#endif
using ClusterShape = Shape<_1, _1, _1>;

using StockBuilder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, 32, ElementB,
    LayoutB, 32, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<0>, BuilderSchedule>;

// N permutation: logical (atom col 8, warp 2, rep 4) -> col i + 32*w + 8*r, so each warp owns 32
// contiguous accumulator columns (one 16-output scale block of the interleaved weight).
using PermTileN = Layout<Shape<_8, _2, _4>, Stride<_1, _32, _8>>;
using TiledMma = decltype(cute::make_tiled_mma(
    cute::rr_blockscaled_op_selector_sm120<typename StockBuilder::ElementA,
                                           typename StockBuilder::ElementB, float,
                                           typename StockBuilder::ElementSF,
                                           StockBuilder::SFVectorSize, StockBuilder::UseMxf8f6f4>(),
    AtomLayoutMNK{},
    Tile<typename StockBuilder::PermTileM, PermTileN, typename StockBuilder::PermTileK>{}));

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    typename StockBuilder::DispatchPolicy, TileShape,
    cute::tuple<typename StockBuilder::ElementA, typename StockBuilder::ElementSF>,
    typename StockBuilder::StridePairA,
    cute::tuple<typename StockBuilder::ElementB, typename StockBuilder::ElementSF>,
    typename StockBuilder::StridePairB, TiledMma, typename StockBuilder::GmemTiledCopyPairA,
    typename StockBuilder::SmemLayoutAtomsA, typename StockBuilder::SmemCopyAtomsA, cute::identity,
    typename StockBuilder::GmemTiledCopyPairB, typename StockBuilder::SmemLayoutAtomsB,
    typename StockBuilder::SmemCopyAtomsB, cute::identity>;

using StrideCD = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    SwigluFp4Epilogue<StrideCD, StrideCD, cutlass::epilogue::NoSmemWarpSpecialized>>;

#if FLUX3_TILESCHED == 1
using TileSchedulerTag = cutlass::gemm::PersistentScheduler;
#else
using TileSchedulerTag = cutlass::gemm::StaticPersistentScheduler;
#endif

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                         CollectiveEpilogue, TileSchedulerTag>;

}  // namespace flux3_swiglu_fp4

namespace {

int cached_sm_count(int device) {
  static int counts[64] = {0};
  if (device < 0 || device >= 64) device = 0;
  if (counts[device] == 0) {
    counts[device] = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  }
  return counts[device];
}

}  // namespace

// a: [M, K/2] u8, b: [N, K/2] u8 (value/gate pair-interleaved rows), sfa/sfb: 128x4 E4M3 scales,
// alpha/inv_scale: f32[1] device, out: [rows_padded, N/4] u8, out_sf: rows_padded * N/32 u8.
void flux3_swiglu_fp4_gemm(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                           TensorView alpha, TensorView inv_scale, TensorView out,
                           TensorView out_sf, int64_t rows_padded) {
  using namespace flux3_swiglu_fp4;
  using Gemm = GemmKernel;
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  int const m = static_cast<int>(a.size(0));
  int const k = static_cast<int>(a.size(1)) * 2;
  int const n = static_cast<int>(b.size(0));
  TVM_FFI_ICHECK_EQ(b.size(1) * 2, k);
  TVM_FFI_ICHECK_EQ(n % 128, 0) << "N must be a multiple of 128";
  TVM_FFI_ICHECK(rows_padded >= m && rows_padded % 128 == 0);
  TVM_FFI_ICHECK_EQ(out.size(0), rows_padded);
  TVM_FFI_ICHECK_EQ(out.size(1), n / 4);
  TVM_FFI_ICHECK_EQ(out_sf.numel(), rows_padded * (n / 32));

  using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(a.data_ptr());
  args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(b.data_ptr());
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sfa.data_ptr());
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sfb.data_ptr());
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename Gemm::StrideA{}, {m, k, 1});
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename Gemm::StrideB{}, {n, k, 1});
  args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.out = static_cast<uint8_t*>(out.data_ptr());
  args.epilogue.out_sf = static_cast<uint8_t*>(out_sf.data_ptr());
  args.epilogue.alpha = static_cast<float const*>(alpha.data_ptr());
  args.epilogue.inv_scale = static_cast<float const*>(inv_scale.data_ptr());
  args.epilogue.rows_padded = static_cast<int>(rows_padded);
  args.epilogue.out_row_bytes = n / 4;
  args.epilogue.sf_cols = n / 32;
  int const device = a.device().device_id;
  args.hw_info.device_id = device;
  args.hw_info.sm_count = cached_sm_count(device);
  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using Enum_t = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = Enum_t::Heuristic;
  }

  TVM_FFI_ICHECK(Gemm::can_implement(args)) << "flux3_swiglu_fp4_gemm: cannot implement";
  TVM_FFI_ICHECK_EQ(Gemm::get_workspace_size(args), 0) << "unexpected workspace";
  typename Gemm::Params params = Gemm::to_underlying_arguments(args, nullptr);
  dim3 const grid = Gemm::get_grid_shape(params);
  dim3 const block = Gemm::get_block_shape();
  int const smem = Gemm::SharedStorageSize;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t err = cudaFuncSetAttribute(cutlass::device_kernel<Gemm>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TVM_FFI_ICHECK(err == cudaSuccess) << "cudaFuncSetAttribute: " << cudaGetErrorString(err);
    attr_set = true;
  }
  cudaStream_t stream = get_stream(a.device());
  cutlass::device_kernel<Gemm><<<grid, block, smem, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "flux3_swiglu_fp4_gemm launch: " << cudaGetErrorString(err);
}

__global__ void flux3_div_test_kernel(float const* a, float const* b, float* fast, float* rn,
                                      uint8_t* safe, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    fast[i] = flux3_swiglu_fp4::div_fast(a[i], b[i]);
    rn[i] = flux3_swiglu_fp4::div_rn(a[i], b[i]);
    safe[i] = flux3_swiglu_fp4::div_safe(a[i], b[i]) ? 1 : 0;
  }
}

// Unit test of the branch-free division against div.rn.f32 (elementwise over f32 inputs).
void flux3_div_test(TensorView a, TensorView b, TensorView fast, TensorView rn, TensorView safe) {
  int64_t n = a.numel();
  cudaStream_t stream = get_stream(a.device());
  flux3_div_test_kernel<<<1024, 256, 0, stream>>>(
      static_cast<float const*>(a.data_ptr()), static_cast<float const*>(b.data_ptr()),
      static_cast<float*>(fast.data_ptr()), static_cast<float*>(rn.data_ptr()),
      static_cast<uint8_t*>(safe.data_ptr()), n);
}

int64_t flux3_swiglu_fp4_smem_bytes() {
  return static_cast<int64_t>(flux3_swiglu_fp4::GemmKernel::SharedStorageSize);
}

int64_t flux3_swiglu_fp4_stages() {
  return static_cast<int64_t>(flux3_swiglu_fp4::CollectiveMainloop::DispatchPolicy::Stages);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flux3_swiglu_fp4_gemm, flux3_swiglu_fp4_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flux3_swiglu_fp4_smem_bytes, flux3_swiglu_fp4_smem_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flux3_swiglu_fp4_stages, flux3_swiglu_fp4_stages);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flux3_div_test, flux3_div_test);
