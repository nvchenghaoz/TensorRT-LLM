/*
 * FLUX.3 SM120 NVFP4 batched (L=3) QKV GEMM whose epilogue writes E4M3 Q/K directly (idea 14).
 *
 * Add-only module (main's flux3_fp4_gemm_sm120_pingpong_dyn.cu stays the kill-switch target).
 *
 * Mainloop: the CUTLASS SM120 block-scaled TMA warp-specialized Pingpong mainloop with tile
 * 128x128x128 (FP4 elements), the stock 2x2 warp layout and the CLC dynamic persistent scheduler,
 * as main's served QKV3 kernel. Only the N order of the TiledMma differs: identity (col = i + 8 wn
 * + 16 r, i = 2q + e the m16n8 atom column, wn the N warp, r the N repeat) instead of the stock
 * (8,2,2):(1,16,8) permutation; each 8x8 ldmatrix still reads 8 distinct rows mod 8. The K order
 * of the mma.sync chain per output element is unchanged, so every accumulator is bitwise main's.
 * The epilogue needs no TMA staging, so the mainloop gets 5 smem stages (main: 4).
 * (A 4x1 warp layout, which would keep a head's row inside one warp, spills the mainloop's
 * operand registers: 1.30-1.45 ms vs 0.59 ms at M=16896, idea 14 job 79.)
 *
 * Epilogue, plane p = the tile's L coordinate, x = bf16_rn(alpha[p] * acc) (main's BF16 output):
 *   p = 2 (v): store x (BF16) and fold |x| (max.f32, NaN-ignoring, init 0) into one per-tile
 *     partial max (warp butterfly, then the 4 warps through smem).
 *   p < 2 (q/k): exactly flux3_sm120_qknorm_rope_fp8's VEC mode-2 arithmetic (from its PTX):
 *     virtual lane l = col & 31 sums cols l, l+32, l+64, l+96 with fma.rn (exact bf16 squares),
 *     then the butterfly xor 16, 8, 4, 2, 1 (add.f32). Here col bit 4 = r bit 0 (local add),
 *     bit 3 = wn (the two N warps of a row exchange through smem: the wn=1 warp hands its partials
 *     to the wn=0 warp, which finishes the row and hands back inv_rms), bits 2,1 = q (quad
 *     shuffles lane^2, lane^1), bit 0 = e (local). Same grouping as the VEC tree; add is
 *     commutative, so every lane of a row sees the same sum.
 *     inv = rsqrt.approx.f32(fma.rn(ss, 2^-7, 1e-6)); even = (inv*x_e)*w_e; odd = (inv*x_o)*w_o
 *     rot_e = fma.rn(even, cos, -(odd*sin)); rot_o = fma.rn(odd, cos, even*sin); bf16 round each
 *     byte pair = cvt.rn.satfinite.e4m3x2(rot_o * f, rot_e * f), f the static factor:
 *     B' = mul_rn(mul_rn(sqrt(128), max|w|), 1 + 2^-6); s = bf16(max(div_rn(B', 448), min));
 *     f = div_rn(1, s)  (idea 268's STATIC_QK rule, computed in-kernel from the norm weights).
 *   RoPE row = token row (the route passes identity positions; cos_sin is [M, 128] FP32).
 * v_quant (second entry) then reduces the V partials, writes scales[0..7] exactly as the VEC
 * mode 2 does and quantizes V (16-B loads, 8-B stores, optional L2 discard of the dead V lines).
 * Its walk may run from the plane's last chunk down (reverse, idea 51): the GEMM writes the v
 * plane (L = 2) last, so its last lines are the ones still in L2; v8 is bitwise either way.
 */
#include <cuda_bf16.h>

#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/packed_stride.hpp"
#include "tvm_ffi_utils.h"

namespace flux3_qkv_epi {

using namespace cute;

// ---------------------------------------------------------------------------------------------
// Exact-form PTX helpers (the module may be compiled with -use_fast_math: inline PTX pins the
// non-ftz forms the CuTe DSL kernel uses).
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
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float div_rn(float a, float b) {
  float d;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float rsqrt_approx(float a) {
  float d;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float max_f32(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float max_abs(float acc, float x) {
  float d;
  asm("{\n.reg .f32 t;\nabs.f32 t, %2;\nmax.f32 %0, %1, t;\n}" : "=f"(d) : "f"(acc), "f"(x));
  return d;
}
__device__ __forceinline__ float neg_f32(float a) {
  float d;
  asm("neg.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ uint32_t bf16_bits(float x) {
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
  return static_cast<uint32_t>(h);
}
__device__ __forceinline__ float bits_to_f32(uint32_t bf16) { return __uint_as_float(bf16 << 16); }
__device__ __forceinline__ float bf16_round(float x) { return bits_to_f32(bf16_bits(x)); }
// e4m3(lo * f) in the low byte, e4m3(hi * f) in the high byte (as the DSL kernel's _e4m3x2).
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi, float f) {
  uint16_t r;
  asm("{\n.reg .f32 a, b;\nmul.rn.f32 a, %1, %3;\nmul.rn.f32 b, %2, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f32 %0, a, b;\n}"
      : "=h"(r)
      : "f"(hi), "f"(lo), "f"(f));
  return static_cast<uint32_t>(r);
}
__device__ __forceinline__ uint4 ldg_nc_u4(void const* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ void bar_sync(int id) {
  asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}

// Quad (4-lane) 4x4 transposes for wide stores. Lane q holds elements k = 0..3 of its row
// segment; afterwards it holds element k = q of lanes s = 0..3 (in s order). Pure data movement.
// 16-bit elements, in: A = (k0, k1), B = (k2, k3) (low half first); out: (s0, s1), (s2, s3).
__device__ __forceinline__ uint2 quad_t16(uint32_t A, uint32_t B, int q) {
  bool const q0 = q & 1;
  bool const q1 = (q >> 1) & 1;
  uint32_t const r1 = __shfl_xor_sync(0xffffffffu, q1 ? A : B, 2);
  uint32_t const X = q1 ? B : A;
  uint32_t const lo = (X & 0xFFFFu) | (r1 << 16);
  uint32_t const hi = (X >> 16) | (r1 & 0xFFFF0000u);
  uint32_t const K = q0 ? hi : lo;
  uint32_t const R = __shfl_xor_sync(0xffffffffu, q0 ? lo : hi, 1);
  uint32_t const a = K & 0xFFFFu, b = K >> 16, c = R & 0xFFFFu, d = R >> 16;
  uint32_t const P = q0 ? (c | (a << 16)) : (a | (c << 16));
  uint32_t const Q = q0 ? (d | (b << 16)) : (b | (d << 16));
  return q1 ? make_uint2(Q, P) : make_uint2(P, Q);
}
// 32-bit elements e0..e3; out: element k = q of lanes s = 0..3.
__device__ __forceinline__ uint4 quad_t32(uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3,
                                          int q) {
  bool const q0 = q & 1;
  bool const q1 = (q >> 1) & 1;
  uint32_t const y0 = __shfl_xor_sync(0xffffffffu, q1 ? e0 : e2, 2);
  uint32_t const y1 = __shfl_xor_sync(0xffffffffu, q1 ? e1 : e3, 2);
  uint32_t const x0 = q1 ? e2 : e0;
  uint32_t const x1 = q1 ? e3 : e1;
  uint32_t const a = q0 ? x1 : x0;  // lane q,     k = q
  uint32_t const b = q0 ? y1 : y0;  // lane q ^ 2, k = q
  uint32_t const c = __shfl_xor_sync(0xffffffffu, q0 ? x0 : x1, 1);  // lane q ^ 1
  uint32_t const d = __shfl_xor_sync(0xffffffffu, q0 ? y0 : y1, 1);  // lane q ^ 3
  uint32_t const P0 = q0 ? c : a, P1 = q0 ? a : c, Q0 = q0 ? d : b, Q1 = q0 ? b : d;
  return q1 ? make_uint4(Q0, Q1, P0, P1) : make_uint4(P0, P1, Q0, Q1);
}

#ifndef FLUX3_EPI_WIDE
#define FLUX3_EPI_WIDE 1
#endif

constexpr uint32_t kSqrt128Bits = 0x41350517u;   // fp32(sqrt(128)), as the DSL kernel's constant
constexpr uint32_t kMarginBits = 0x3F820000u;    // fp32(1 + 2^-6)
constexpr uint32_t kEpsBits = 0x358637BDu;       // fp32(1e-6)
constexpr uint32_t kInv128Bits = 0x3C000000u;    // 2^-7

// Static Q/K factor and scale (idea 268): every lane of the (full) warp gets the same values.
__device__ __forceinline__ void static_scale(__nv_bfloat16 const* norm, float min_scale,
                                             float& scale, float& factor) {
  int const lane = threadIdx.x & 31;
  uint2 w = *reinterpret_cast<uint2 const*>(norm + lane * 4);
  float m = 0.0f;
  m = max_abs(m, bits_to_f32(w.x & 0xFFFFu));
  m = max_abs(m, bits_to_f32(w.x >> 16));
  m = max_abs(m, bits_to_f32(w.y & 0xFFFFu));
  m = max_abs(m, bits_to_f32(w.y >> 16));
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) m = max_f32(m, __shfl_xor_sync(0xffffffffu, m, d));
  float b = mul_rn(mul_rn(__uint_as_float(kSqrt128Bits), m), __uint_as_float(kMarginBits));
  scale = bf16_round(max_f32(div_rn(b, 448.0f), min_scale));
  factor = div_rn(1.0f, scale);
}

// ---------------------------------------------------------------------------------------------
// Epilogue (operator-style, wrapped by Sm90TmaWarpSpecializedAdapter)
// ---------------------------------------------------------------------------------------------
struct QkvFp8EpiArgs {
  uint8_t* qk8 = nullptr;                    // E4M3 planes 0 (q), 1 (k): [2][rows][width]
  __nv_bfloat16* v = nullptr;                // BF16 v plane [rows][width]
  float* v_parts = nullptr;                  // [m_tiles * n_tiles] per-tile max|v|
  float const* alpha = nullptr;              // [3] plane alphas
  __nv_bfloat16 const* q_norm = nullptr;     // [128]
  __nv_bfloat16 const* k_norm = nullptr;     // [128]
  float const* cos_sin = nullptr;            // [rows][128] (cos 0..63, sin 64..127)
  int rows = 0;
  int width = 0;
  int n_tiles = 0;                           // width / 128
  float min_scale = 0.0f;
};
// Idea 84 (TRTLLM_FLUX3_QKV_EPI_ROPE_COMPACT): the compact per-axis RoPE table. Axis a (pairs
// 16a .. 16a+15) of row r is entry rope_idx[r][a] of rope_tab, 32 FP32 in the epilogue's lane
// order: slot s = q + 4 wn (= the lane's pair p0) holds {cos(16a+s), cos(16a+s+8), sin(16a+s),
// sin(16a+s+8)} at [4s, 4s+4). flux3_qkv_epi_fp8.build_rope_compact builds it from cos_sin and
// checks every index bit for bit on device (rope_tab[rope_idx[r][a]] == row r's axis-a slice).
struct QkvFp8EpiArgsCompact : QkvFp8EpiArgs {
  float const* rope_tab = nullptr;           // [entries][32]
  int const* rope_idx = nullptr;             // [rows][4] entry per axis
};

template <class StrideC_, class StrideD_, class EpilogueSchedule_, bool kCompact = false>
class QkvFp8Epilogue {
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
  using ThreadEpilogueOp =
      cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 1, float, float>;

  // Shared by both consumer warpgroups: their epilogues are serialized by the Pingpong kernel's
  // math_wg_order_barrier (the next epilogue starts after every thread of this one arrived).
  struct SharedStorage {
    float2 part[2][8][32];  // [wm][row slot][lane]: the wn=1 warp's xor16-level row partials
    float inv[2][8][8];     // [wm][row slot][group]: inv_rms handed back to the wn=1 warp
    float red[4];           // per-warp V max
  };
  using TensorStorage = SharedStorage;

  using Arguments = cute::conditional_t<kCompact, QkvFp8EpiArgsCompact, QkvFp8EpiArgs>;
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
  QkvFp8Epilogue(Params const& params_, SharedStorage const& = SharedStorage()) : params(params_) {}

  CUTLASS_DEVICE bool is_source_needed() { return false; }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine,
            class FrgLayout, class TiledMma, class ResidueMNK>
  CUTLASS_DEVICE void operator()(ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL blk_coord_mnkl,
                                 cute::Tensor<FrgEngine, FrgLayout> const& acc,
                                 TiledMma tiled_mma, ResidueMNK, int thread_idx, char* smem) {
#ifdef FLUX3_EPI_DIAG_NOOP
    // Diagnostic only: mainloop cost (keeps the accumulators live, writes nothing real).
    float sum = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc); ++i) sum += acc(i);
    if (sum == 1.2345f) params.v_parts[0] = sum;
    return;
#else
    constexpr int BM = size<0>(BlockShapeMNK{});
    constexpr int BN = size<1>(BlockShapeMNK{});
    static_assert(BM == 128 && BN == 128, "tile must be 128x128 (one head per N tile)");
    static_assert(size<0>(FrgLayout{}) == 4, "m16n8 C fragment expected");
    static_assert(size<1>(FrgLayout{}) == 4, "2x2 warps: MMA_M = 4");
    static_assert(size<2>(FrgLayout{}) == 8, "2x2 warps: MMA_N = 8");
    static_assert(!kCompact || FLUX3_EPI_WIDE, "the compact RoPE table needs the wide epilogue");

    int const m_tile = static_cast<int>(get<0>(blk_coord_mnkl));
    int const n_tile = static_cast<int>(get<1>(blk_coord_mnkl));
    int const plane = static_cast<int>(get<3>(blk_coord_mnkl));

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto cD = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto tCcD = thr_mma.partition_C(cD);  // (4, 4, 8) -> (row, col) in tile

    SharedStorage& ss_mem = *reinterpret_cast<SharedStorage*>(smem);
    int const lane = threadIdx.x & 31;
    int const group = lane >> 2;
    int const rows = params.rows;
    int const width = params.width;
    float const alpha = params.alpha[plane];
    int const row_base = m_tile * BM;
    // Column of (e = 0, ni = 0) is 2q + 8 wn; column of (e, ni) is col0 + e + 16 ni.
    int const col0 = get<1>(tCcD(0, 0, 0));
    int const wn = (col0 >> 3) & 1;
    int const wm = (get<0>(tCcD(0, 0, 0)) >> 4) & 1;  // stock 2x2: warp rows 16 wm + 32 mi + ..
    int const bar_id = 8 + ((static_cast<int>(threadIdx.x) >> 7) - 1);  // per consumer WG

    if (plane == 2) {
      // ---------------- V: BF16 store + per-tile max|bf16(v)| ----------------
      float amax = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < 4; ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          int const row = row_base + get<0>(tCcD(2 * h, mi, 0));
          bool const live = row < rows;
#if FLUX3_EPI_WIDE
          uint32_t wv[8];
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < 8; ++ni) {
            uint32_t lo = bf16_bits(alpha * acc(2 * h, mi, ni));
            uint32_t hi = bf16_bits(alpha * acc(2 * h + 1, mi, ni));
            if (live) {
              amax = max_abs(amax, bits_to_f32(lo));
              amax = max_abs(amax, bits_to_f32(hi));
            }
            wv[ni] = lo | (hi << 16);
          }
          // Quad transpose: lane q stores the 8 contiguous BF16 of ni = 4j + q
          // (cols 8 wn + 64 j + 16 q .. + 7), one 16-B store per (row, j).
          __nv_bfloat16* dst = params.v + static_cast<int64_t>(row) * width + n_tile * BN +
                               (col0 & 8) + 16 * (lane & 3);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 2; ++j) {
            uint4 o = quad_t32(wv[4 * j], wv[4 * j + 1], wv[4 * j + 2], wv[4 * j + 3], lane & 3);
#ifdef FLUX3_EPI_DIAG_NOVST
            // Diagnostic only: V values computed, (almost) never stored.
            if (live && (o.x ^ o.y ^ o.z ^ o.w) == 0x12345679u)
#else
            if (live)
#endif
              *reinterpret_cast<uint4*>(dst + 64 * j) = o;
          }
#else
          if (live) {
            __nv_bfloat16* dst =
                params.v + static_cast<int64_t>(row) * width + n_tile * BN + col0;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < 8; ++ni) {
              uint32_t lo = bf16_bits(alpha * acc(2 * h, mi, ni));
              uint32_t hi = bf16_bits(alpha * acc(2 * h + 1, mi, ni));
              amax = max_abs(amax, bits_to_f32(lo));
              amax = max_abs(amax, bits_to_f32(hi));
              *reinterpret_cast<uint32_t*>(dst + 16 * ni) = lo | (hi << 16);
            }
          }
#endif
        }
      }
#pragma unroll
      for (int d = 16; d >= 1; d >>= 1) amax = max_f32(amax, __shfl_xor_sync(0xffffffffu, amax, d));
      int const warp = thread_idx >> 5;
      if (lane == 0) ss_mem.red[warp] = amax;
      bar_sync(bar_id);
      if (thread_idx == 0) {
        float m = 0.0f;
        m = max_f32(m, ss_mem.red[0]);
        m = max_f32(m, ss_mem.red[1]);
        m = max_f32(m, ss_mem.red[2]);
        m = max_f32(m, ss_mem.red[3]);
        params.v_parts[m_tile * params.n_tiles + n_tile] = m;
      }
      return;
    }

    // ---------------- Q/K: RMSNorm + RoPE -> E4M3 ----------------
    __nv_bfloat16 const* norm = plane == 0 ? params.q_norm : params.k_norm;
    float scale_unused, factor;
#ifdef FLUX3_EPI_DIAG_NOSF
    // Diagnostic only: a plane-dependent constant instead of the per-tile static factor.
    factor = plane == 0 ? 0.0625f : 0.125f;
    (void)scale_unused;
#else
    static_scale(norm, params.min_scale, scale_unused, factor);
#endif
    float const eps = __uint_as_float(kEpsBits);
    float const inv128 = __uint_as_float(kInv128Bits);

    // Pass 1: per row slot (mi, h), the xor16-level partial of virtual lanes (q, e): chain over
    // t (ni = 2t + b), then the bit-4 add (b = 0, 1).
    float u[8][2];
#if FLUX3_EPI_WIDE
    // BF16(alpha * acc) once, packed (e0 low, e1 high) per (row slot, ni): the accumulators die.
    uint32_t pk[8][8];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 4; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < 8; ++ni) {
          pk[2 * mi + h][ni] = bf16_bits(alpha * acc(2 * h, mi, ni)) |
                               (bf16_bits(alpha * acc(2 * h + 1, mi, ni)) << 16);
        }
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < 8; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < 2; ++e) {
        float s[2];
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < 2; ++b) {
          float a = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int t = 0; t < 4; ++t) {
            uint32_t const w = pk[r][2 * t + b];
            float const x = e ? __uint_as_float(w & 0xFFFF0000u) : __uint_as_float(w << 16);
            a = fma_rn(x, x, a);
          }
          s[b] = a;
        }
        u[r][e] = add_rn(s[1], s[0]);
      }
    }
#else
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 4; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < 2; ++e) {
          float s[2];
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < 2; ++b) {
            float a = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int t = 0; t < 4; ++t) {
              float const x = bf16_round(alpha * acc(2 * h + e, mi, 2 * t + b));
              a = fma_rn(x, x, a);
            }
            s[b] = a;
          }
          u[2 * mi + h][e] = add_rn(s[1], s[0]);
        }
      }
    }
#endif
    float inv[8];
#ifdef FLUX3_EPI_DIAG_NOX
    // Diagnostic only: no cross-warp exchange (each warp finishes its own half-row sums).
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < 8; ++r) {
      float w0 = add_rn(u[r][0], u[r][0]);
      float w1 = add_rn(u[r][1], u[r][1]);
      w0 = add_rn(__shfl_xor_sync(0xffffffffu, w0, 2), w0);
      w1 = add_rn(__shfl_xor_sync(0xffffffffu, w1, 2), w1);
      w0 = add_rn(__shfl_xor_sync(0xffffffffu, w0, 1), w0);
      w1 = add_rn(__shfl_xor_sync(0xffffffffu, w1, 1), w1);
      inv[r] = rsqrt_approx(fma_rn(add_rn(w1, w0), inv128, eps));
    }
#else
    if (wn == 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 8; ++r) ss_mem.part[wm][r][lane] = make_float2(u[r][0], u[r][1]);
    }
    bar_sync(bar_id);
    if (wn == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 8; ++r) {
        float2 p = ss_mem.part[wm][r][lane];
        float w0 = add_rn(p.x, u[r][0]);  // xor 8 (bit 3 = wn)
        float w1 = add_rn(p.y, u[r][1]);
        w0 = add_rn(__shfl_xor_sync(0xffffffffu, w0, 2), w0);  // xor 4 (bit 2 = q bit 1)
        w1 = add_rn(__shfl_xor_sync(0xffffffffu, w1, 2), w1);
        w0 = add_rn(__shfl_xor_sync(0xffffffffu, w0, 1), w0);  // xor 2 (bit 1 = q bit 0)
        w1 = add_rn(__shfl_xor_sync(0xffffffffu, w1, 1), w1);
        float const ssum = add_rn(w1, w0);                     // xor 1 (bit 0 = e)
        inv[r] = rsqrt_approx(fma_rn(ssum, inv128, eps));
        if ((lane & 3) == 0) ss_mem.inv[wm][r][group] = inv[r];
      }
    }
    bar_sync(bar_id);
    if (wn == 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 8; ++r) inv[r] = ss_mem.inv[wm][r][group];
    }
#endif  // FLUX3_EPI_DIAG_NOX

    // Pass 2: RoPE pairs (col0 + 16 ni, +1) -> E4M3 pair, pair index p = col0 / 2 + 8 ni.
    uint32_t wpair[8];
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < 8; ++ni) {
      wpair[ni] = __ldg(reinterpret_cast<unsigned int const*>(norm + col0 + 16 * ni));
    }
    uint8_t* out_plane = params.qk8 + static_cast<int64_t>(plane) * rows * width + n_tile * BN;
    int const p0 = col0 >> 1;
#if FLUX3_EPI_WIDE
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < 8; ++r) {
      int const row = row_base + get<0>(tCcD(2 * (r & 1), r >> 1, 0));
      float const iv = inv[r];
      float const* cs_row = params.cos_sin + static_cast<int64_t>(row < rows ? row : 0) * 128 + p0;
      // kCompact: entry e[a] = {cos(ni = 2a), cos(2a + 1), sin(2a), sin(2a + 1)} of this lane.
      float4 ce[4];
      if constexpr (kCompact) {
        int4 const ix = __ldg(reinterpret_cast<int4 const*>(params.rope_idx) + (row < rows ? row : 0));
        float4 const* tab = reinterpret_cast<float4 const*>(params.rope_tab) + p0;
        ce[0] = __ldg(tab + static_cast<int64_t>(ix.x) * 8);
        ce[1] = __ldg(tab + static_cast<int64_t>(ix.y) * 8);
        ce[2] = __ldg(tab + static_cast<int64_t>(ix.z) * 8);
        ce[3] = __ldg(tab + static_cast<int64_t>(ix.w) * 8);
      }
      uint32_t hb[8];
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < 8; ++ni) {
        if constexpr (kCompact) {
          float const cv = (ni & 1) ? ce[ni >> 1].y : ce[ni >> 1].x;
          float const sv = (ni & 1) ? ce[ni >> 1].w : ce[ni >> 1].z;
          float const xe = __uint_as_float(pk[r][ni] << 16);
          float const xo = __uint_as_float(pk[r][ni] & 0xFFFF0000u);
          float const even = mul_rn(mul_rn(iv, xe), bits_to_f32(wpair[ni] & 0xFFFFu));
          float const odd = mul_rn(mul_rn(iv, xo), bits_to_f32(wpair[ni] >> 16));
          float const rot_e = fma_rn(even, cv, neg_f32(mul_rn(odd, sv)));
          float const rot_o = fma_rn(odd, cv, mul_rn(even, sv));
          hb[ni] = e4m3x2(bf16_round(rot_e), bf16_round(rot_o), factor);
          continue;
        }
#ifdef FLUX3_EPI_DIAG_NOCS
        // Diagnostic only: row-dependent register values instead of the cos_sin loads.
        float const cv = __int_as_float(0x3F000000 | ((row * 64 + p0 + 8 * ni) & 0x7FFFF));
        float const sv = __int_as_float(0x3E800000 | ((row * 64 + p0 + 8 * ni) & 0x7FFFF));
        (void)cs_row;
#else
        float const cv = __ldg(cs_row + 8 * ni);
        float const sv = __ldg(cs_row + 64 + 8 * ni);
#endif
        float const xe = __uint_as_float(pk[r][ni] << 16);
        float const xo = __uint_as_float(pk[r][ni] & 0xFFFF0000u);
        float const even = mul_rn(mul_rn(iv, xe), bits_to_f32(wpair[ni] & 0xFFFFu));
        float const odd = mul_rn(mul_rn(iv, xo), bits_to_f32(wpair[ni] >> 16));
        float const rot_e = fma_rn(even, cv, neg_f32(mul_rn(odd, sv)));
        float const rot_o = fma_rn(odd, cv, mul_rn(even, sv));
        hb[ni] = e4m3x2(bf16_round(rot_e), bf16_round(rot_o), factor);
      }
      // Quad transpose: lane q stores the 8 contiguous E4M3 of ni = 4j + q
      // (cols 8 wn + 64 j + 16 q .. + 7), one 8-B store per (row, j).
      uint8_t* dst = out_plane + static_cast<int64_t>(row) * width + (col0 & 8) + 16 * (lane & 3);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 2; ++j) {
        uint2 o = quad_t16(hb[4 * j] | (hb[4 * j + 1] << 16), hb[4 * j + 2] | (hb[4 * j + 3] << 16),
                           lane & 3);
#ifdef FLUX3_EPI_DIAG_NOQKST
        // Diagnostic only: E4M3 Q/K computed, (almost) never stored.
        if (row < rows && (o.x ^ o.y) == 0x12345679u)
#else
        if (row < rows)
#endif
          *reinterpret_cast<uint2*>(dst + 64 * j) = o;
      }
    }
#else
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 4; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        int const row = row_base + get<0>(tCcD(2 * h, mi, 0));
        if (row < rows) {
          float const iv = inv[2 * mi + h];
          float const* cs_row = params.cos_sin + static_cast<int64_t>(row) * 128 + p0;
          uint8_t* dst = out_plane + static_cast<int64_t>(row) * width + col0;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < 8; ++ni) {
            float const cv = __ldg(cs_row + 8 * ni);
            float const sv = __ldg(cs_row + 64 + 8 * ni);
            float const xe = bf16_round(alpha * acc(2 * h, mi, ni));
            float const xo = bf16_round(alpha * acc(2 * h + 1, mi, ni));
            float const even = mul_rn(mul_rn(iv, xe), bits_to_f32(wpair[ni] & 0xFFFFu));
            float const odd = mul_rn(mul_rn(iv, xo), bits_to_f32(wpair[ni] >> 16));
            float const rot_e = fma_rn(even, cv, neg_f32(mul_rn(odd, sv)));
            float const rot_o = fma_rn(odd, cv, mul_rn(even, sv));
            *reinterpret_cast<uint16_t*>(dst + 16 * ni) =
                static_cast<uint16_t>(e4m3x2(bf16_round(rot_e), bf16_round(rot_o), factor));
          }
        }
      }
    }
#endif  // FLUX3_EPI_WIDE
#endif
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
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using StrideCD = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using EpilogueOp = QkvFp8Epilogue<StrideCD, StrideCD, cutlass::epilogue::NoSmemWarpSpecialized>;

using StockBuilder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, 32, ElementB,
    LayoutB, 32, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename EpilogueOp::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>;

// Stock 2x2 warps; identity N order: (atom col i: 8, wn: 2, r: 8) -> i + 8 wn + 16 r.
using AtomLayoutMNK = Layout<Shape<_2, _2, _1>>;
#ifdef FLUX3_EPI_STOCK_PERM
using PermTileN = typename StockBuilder::PermTileN;  // diagnostic (epilogue must be NOOP)
#else
using PermTileN = Layout<Shape<_8, _2, _8>, Stride<_1, _8, _16>>;
#endif
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

using CollectiveEpilogue =
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp>;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                         CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Idea 84: the same kernel with the compact RoPE table (same mainloop, same smem carve-out).
using EpilogueOpCompact =
    QkvFp8Epilogue<StrideCD, StrideCD, cutlass::epilogue::NoSmemWarpSpecialized, true>;
static_assert(sizeof(typename EpilogueOpCompact::SharedStorage) ==
                  sizeof(typename EpilogueOp::SharedStorage),
              "compact epilogue must keep the mainloop's stage count");
using CollectiveEpilogueCompact =
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpilogueOpCompact>;
using GemmKernelCompact =
    cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                         CollectiveEpilogueCompact,
                                         cutlass::gemm::PersistentScheduler>;
using GemmCompact = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelCompact>;

int cached_sm_count(int device) {
  static int counts[64] = {0};
  if (device < 0 || device >= 64) device = 0;
  if (counts[device] == 0) {
    counts[device] = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  }
  return counts[device];
}

template <class GemmT = Gemm, class EpiArgsT = typename EpilogueOp::Arguments>
typename GemmT::Arguments make_args(void const* a, void const* b, void const* sfa, void const* sfb,
                                    int m, int n, int k, int device, EpiArgsT const& epi) {
  using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
  typename GemmT::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m, n, k, 3);
  args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(a);
  args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(b);
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sfa);
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sfb);
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {m, k, 3});
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {n, k, 3});
  // A and its scales are shared by the 3 planes: batch stride 0 (as main's qkv3_args).
  cute::get<2>(args.mainloop.dA) = 0;
  args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
  cute::get<1>(cute::get<2>(args.mainloop.layout_SFA.stride())) = 0;
  args.epilogue = epi;
  args.hw_info.device_id = device;
  args.hw_info.sm_count = cached_sm_count(device);
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using Enum_t = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = Enum_t::Heuristic;
  }
  return args;
}

// ---------------------------------------------------------------------------------------------
// V-only quantize: reduce the per-tile V partials, scales[0..7], E4M3 V.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void scale_factor(float amax, float min_scale, float& s, float& f) {
  s = bf16_round(max_f32(div_rn(amax, 448.0f), min_scale));
  f = div_rn(1.0f, s);
}

__device__ __forceinline__ uint32_t e4m3x4(uint32_t w0, uint32_t w1, float f) {
  uint32_t lo = e4m3x2(bits_to_f32(w0 & 0xFFFFu), bits_to_f32(w0 >> 16), f);
  uint32_t hi = e4m3x2(bits_to_f32(w1 & 0xFFFFu), bits_to_f32(w1 >> 16), f);
  return lo | (hi << 16);
}

constexpr int kVThreads = 512;

__global__ void __launch_bounds__(kVThreads) v_quant_kernel(
    __nv_bfloat16 const* __restrict__ v, uint8_t* __restrict__ v8,
    float const* __restrict__ parts, int n_part, __nv_bfloat16 const* __restrict__ q_norm,
    __nv_bfloat16 const* __restrict__ k_norm, float* __restrict__ scales, int64_t chunks,
    float sm_scale, float min_scale, int discard, int reverse) {
  __shared__ float red[kVThreads / 32];
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  float m = 0.0f;
  for (int i = tid; i < n_part; i += kVThreads) m = max_f32(m, parts[i]);
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) m = max_f32(m, __shfl_xor_sync(0xffffffffu, m, d));
  if (lane == 0) red[warp] = m;
  __syncthreads();
  float amax = 0.0f;
#pragma unroll
  for (int w = 0; w < kVThreads / 32; ++w) amax = max_f32(amax, red[w]);
  float s_v, f_v;
  scale_factor(amax, min_scale, s_v, f_v);
  if (blockIdx.x == 0 && warp == 0) {
    float s_q, f_q, s_k, f_k;
    static_scale(q_norm, min_scale, s_q, f_q);
    static_scale(k_norm, min_scale, s_k, f_k);
    if (lane == 0) {
      scales[0] = s_q;
      scales[1] = s_k;
      scales[2] = s_v;
      scales[3] = mul_rn(mul_rn(s_q, s_k), sm_scale);
      scales[4] = f_q;
      scales[5] = f_k;
      scales[6] = f_v;
      scales[7] = 0.0f;
    }
  }
  // 16-B chunks of 8 BF16; a warp covers 512 contiguous bytes (4 lines) per iteration.
  int64_t const stride = static_cast<int64_t>(gridDim.x) * kVThreads;
  // reverse (TRTLLM_FLUX3_QKV_EPI_VQ_REVERSE): walk the plane from its last chunk down, so the
  // lines the GEMM wrote last (the likeliest to still be in L2) are read first. Each element's
  // arithmetic is unchanged (same f_v), so v8 is bitwise the forward walk's.
  int const line_lane = reverse ? 7 : 0;  // the lane holding a line's first chunk
  for (int64_t c = static_cast<int64_t>(blockIdx.x) * kVThreads + tid; c < chunks; c += stride) {
    int64_t const idx = reverse ? (chunks - 1 - c) : c;
    uint4 w = ldg_nc_u4(v + idx * 8);
    uint2 o;
    o.x = e4m3x4(w.x, w.y, f_v);
    o.y = e4m3x4(w.z, w.w, f_v);
    *reinterpret_cast<uint2*>(v8 + idx * 8) = o;
    if (discard) {
      // Every lane of this line (8 lanes, chunks % 8 == 0) has consumed its loaded words.
      __syncwarp(__activemask());
      if ((lane & 7) == line_lane) {
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(v + idx * 8) : "memory");
      }
    }
  }
}

}  // namespace flux3_qkv_epi

using namespace flux3_qkv_epi;

// a [M, K/2] u8, b [3N, K/2] u8 (q, k, v planes), sfa/sfb swizzled E4M3 scales (sfb: 3 planes),
// alpha f32[3], q_norm/k_norm bf16[128], cos_sin f32[M, 128], qkv8 [3, M, N] 1-byte (planes 0, 1
// written), v bf16 [M, N] (v plane), v_parts f32[ceil(M/128) * N/128].
template <class GemmT, bool kCompact>
void qkv3_epi_fp8_impl(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
                       TensorView q_norm, TensorView k_norm, TensorView cos_sin, TensorView qkv8,
                       TensorView v, TensorView v_parts, TensorView workspace, double min_scale,
                       TensorView const* rope_tab, TensorView const* rope_idx) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be uint8 (packed e2m1)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be uint8 (packed e2m1)";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(alpha.numel(), 3);
  TVM_FFI_ICHECK_EQ(cos_sin.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(v.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(v_parts.dtype(), dl_float32);
  int const m = static_cast<int>(a.size(0));
  int const k = static_cast<int>(a.size(1)) * 2;
  TVM_FFI_ICHECK_EQ(b.size(0) % 3, 0);
  int const n = static_cast<int>(b.size(0) / 3);
  TVM_FFI_ICHECK_EQ(b.size(1) * 2, k);
  TVM_FFI_ICHECK_EQ(n % 128, 0);
  TVM_FFI_ICHECK_EQ(k % 64, 0);
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(sfb.numel()) / 3, static_cast<int64_t>(n) * (k / 16));
  TVM_FFI_ICHECK(q_norm.numel() == 128 && k_norm.numel() == 128) << "head_dim must be 128";
  TVM_FFI_ICHECK(cos_sin.numel() >= static_cast<int64_t>(m) * 128) << "cos_sin rows < M";
  TVM_FFI_ICHECK(qkv8.numel() >= static_cast<int64_t>(3) * m * n) << "qkv8 too small";
  TVM_FFI_ICHECK(v.numel() >= static_cast<int64_t>(m) * n) << "v too small";
  int const m_tiles = (m + 127) / 128;
  TVM_FFI_ICHECK(v_parts.numel() >= static_cast<int64_t>(m_tiles) * (n / 128))
      << "v_parts too small";
  for (auto* p : {qkv8.data_ptr(), v.data_ptr(), cos_sin.data_ptr(), q_norm.data_ptr(),
                  k_norm.data_ptr()}) {
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(p) % 16 == 0) << "16-B alignment required";
  }

  using EpiArgs = cute::conditional_t<kCompact, QkvFp8EpiArgsCompact, QkvFp8EpiArgs>;
  EpiArgs epi;
  if constexpr (kCompact) {
    TVM_FFI_ICHECK_EQ(rope_tab->dtype(), dl_float32) << "rope_tab must be float32";
    TVM_FFI_ICHECK_EQ(rope_idx->dtype(), dl_int32) << "rope_idx must be int32";
    TVM_FFI_ICHECK(rope_tab->ndim() == 2 && rope_tab->size(1) == 32) << "rope_tab [entries, 32]";
    TVM_FFI_ICHECK(rope_idx->ndim() == 2 && rope_idx->size(1) == 4) << "rope_idx [rows, 4]";
    TVM_FFI_ICHECK(rope_idx->size(0) >= m) << "rope_idx rows < M";
    for (auto* p : {rope_tab->data_ptr(), rope_idx->data_ptr()}) {
      TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(p) % 16 == 0) << "16-B alignment required";
    }
    epi.rope_tab = static_cast<float const*>(rope_tab->data_ptr());
    epi.rope_idx = static_cast<int const*>(rope_idx->data_ptr());
  }
  epi.qk8 = static_cast<uint8_t*>(qkv8.data_ptr());
  epi.v = static_cast<__nv_bfloat16*>(v.data_ptr());
  epi.v_parts = static_cast<float*>(v_parts.data_ptr());
  epi.alpha = static_cast<float const*>(alpha.data_ptr());
  epi.q_norm = static_cast<__nv_bfloat16 const*>(q_norm.data_ptr());
  epi.k_norm = static_cast<__nv_bfloat16 const*>(k_norm.data_ptr());
  epi.cos_sin = static_cast<float const*>(cos_sin.data_ptr());
  epi.rows = m;
  epi.width = n;
  epi.n_tiles = n / 128;
  epi.min_scale = static_cast<float>(min_scale);

  int const device = a.device().device_id;
  GemmT gemm;
  auto args = make_args<GemmT>(a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(), m, n, k,
                               device, epi);
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

void qkv3_epi_fp8(TensorView a, TensorView b, TensorView sfa, TensorView sfb, TensorView alpha,
                  TensorView q_norm, TensorView k_norm, TensorView cos_sin, TensorView qkv8,
                  TensorView v, TensorView v_parts, TensorView workspace, double min_scale) {
  qkv3_epi_fp8_impl<Gemm, false>(a, b, sfa, sfb, alpha, q_norm, k_norm, cos_sin, qkv8, v, v_parts,
                                 workspace, min_scale, nullptr, nullptr);
}

// Idea 84: as qkv3_epi_fp8, with Q/K RoPE read from the compact per-axis table: rope_tab f32
// [entries, 32] (QkvFp8EpiArgsCompact's lane order), rope_idx i32 [>= M, 4]. cos_sin is only
// shape-checked (build_rope_compact guarantees tab[idx] == cos_sin bit for bit).
void qkv3_epi_fp8_compact(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                          TensorView alpha, TensorView q_norm, TensorView k_norm,
                          TensorView cos_sin, TensorView qkv8, TensorView v, TensorView v_parts,
                          TensorView workspace, double min_scale, TensorView rope_tab,
                          TensorView rope_idx) {
  qkv3_epi_fp8_impl<GemmCompact, true>(a, b, sfa, sfb, alpha, q_norm, k_norm, cos_sin, qkv8, v,
                                       v_parts, workspace, min_scale, &rope_tab, &rope_idx);
}

// v bf16 [M, N] -> v8 (qkv8 plane 2) E4M3, scales f32[8]. discard: drop each consumed v line from
// L2 (the caller never reads v again; its contents become undefined).
void v_quant(TensorView v, TensorView v8, TensorView v_parts, int64_t n_part, TensorView q_norm,
             TensorView k_norm, TensorView scales, double sm_scale, double min_scale,
             int64_t grid, int64_t discard, int64_t reverse) {
  TVM_FFI_ICHECK_EQ(v.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(scales.dtype(), dl_float32);
  TVM_FFI_ICHECK(scales.numel() >= 8);
  int64_t const elems = v.numel();
  TVM_FFI_ICHECK_EQ(elems % 64, 0) << "v must be whole 128-B lines";
  TVM_FFI_ICHECK(v8.numel() >= elems);
  TVM_FFI_ICHECK(n_part > 0 && n_part <= v_parts.numel());
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 128 == 0 || discard == 0)
      << "discard needs 128-B aligned v";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(v8.data_ptr()) % 8 == 0);
  cudaStream_t stream = get_stream(v.device());
  v_quant_kernel<<<static_cast<int>(grid), kVThreads, 0, stream>>>(
      static_cast<__nv_bfloat16 const*>(v.data_ptr()), static_cast<uint8_t*>(v8.data_ptr()),
      static_cast<float const*>(v_parts.data_ptr()), static_cast<int>(n_part),
      static_cast<__nv_bfloat16 const*>(q_norm.data_ptr()),
      static_cast<__nv_bfloat16 const*>(k_norm.data_ptr()), static_cast<float*>(scales.data_ptr()),
      elems / 8, static_cast<float>(sm_scale), static_cast<float>(min_scale),
      static_cast<int>(discard), static_cast<int>(reverse));
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "v_quant launch: " << cudaGetErrorString(err);
}

int64_t qkv3_epi_workspace_size(int64_t m, int64_t n, int64_t k) {
  typename EpilogueOp::Arguments epi;
  auto args = make_args(nullptr, nullptr, nullptr, nullptr, static_cast<int>(m),
                        static_cast<int>(n), static_cast<int>(k), 0, epi);
  return static_cast<int64_t>(Gemm::get_workspace_size(args));
}
int64_t qkv3_epi_smem() { return static_cast<int64_t>(GemmKernel::SharedStorageSize); }
int64_t qkv3_epi_stages() {
  return static_cast<int64_t>(CollectiveMainloop::DispatchPolicy::Stages);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv3_epi_fp8, qkv3_epi_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv3_epi_fp8_compact, qkv3_epi_fp8_compact);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(v_quant, v_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv3_epi_workspace_size, qkv3_epi_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv3_epi_smem, qkv3_epi_smem);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv3_epi_stages, qkv3_epi_stages);
