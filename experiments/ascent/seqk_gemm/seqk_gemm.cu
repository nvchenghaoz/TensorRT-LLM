// Sequential-K BF16 GEMM for SM120 (mma.sync m16n8k16, FP32 accumulate, one RN cast to BF16).
// Y[M, N] = X[M, K] @ W[N, K]^T. Both operands K-major. Every output's FP32 accumulator walks the
// k16 steps in ascending K order, so the bits equal main's non-split-K cuBLASLt/cutlass dispatch
// (fact 707). Prototype for idea 73 stage 0 (built with torch cpp_extension in experiments only).
//
// CTA tile: TM tokens x TN weights, 8 warps (2 per SM sub-partition). SMSP s = warp & 3 owns a
// (TM/2) x (TN/2) quadrant = MT m16 token tiles x NTS n8 weight tiles; its two warps split the
// n8 tiles ceil/floor, so every SMSP issues the same MMA count per k16 step.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <map>
#include <tuple>

namespace seqk {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Byte offset of 16-byte chunk c of row r in a [rows][BK] bf16 tile, XOR-swizzled so that the
// 8 rows an ldmatrix phase reads at one logical chunk land in 8 distinct 16-byte bank groups.
template <int BK>
__device__ __forceinline__ uint32_t swz(int r, int c) {
  if constexpr (BK == 64) {
    return static_cast<uint32_t>((r * 8 + (c ^ (r & 7))) * 16);
  } else {
    static_assert(BK == 32, "BK");
    return static_cast<uint32_t>((r * 4 + (c ^ ((r >> 1) & 3))) * 16);
  }
}

template <int MT, int NTS, int BK, int ST>
struct Cfg {
  static constexpr int kThreads = 256;
  static constexpr int TM = 2 * MT * 16;
  static constexpr int TN = 2 * NTS * 8;
  static constexpr int CPR = BK / 8;
  static constexpr int kStageA = TM * BK * 2;
  static constexpr int kStageB = TN * BK * 2;
  static constexpr int kStage = kStageA + kStageB;
  static constexpr int kSmem = ST * kStage;
  static constexpr int NA = (NTS + 1) / 2;
  static constexpr int NB = NTS - NA;
};

template <class C, int BK>
__device__ __forceinline__ void load_stage(char* smem, int stage, const __nv_bfloat16* __restrict__ X,
                                           const __nv_bfloat16* __restrict__ W, int m0, int n0,
                                           int M, int N, int ldx, int ldw, int k0, int tid) {
  const uint32_t sA = smem_u32(smem + stage * C::kStage);
  const uint32_t sB = sA + C::kStageA;
  constexpr int kA = C::TM * C::CPR;
  constexpr int kB = C::TN * C::CPR;
#pragma unroll
  for (int i0 = 0; i0 < kA; i0 += C::kThreads) {
    int i = i0 + tid;
    if (kA % C::kThreads == 0 || i < kA) {
      int r = i / C::CPR, c = i % C::CPR;
      int gr = m0 + r;
      bool p = gr < M;
      const __nv_bfloat16* src = X + static_cast<size_t>(p ? gr : 0) * ldx + k0 + c * 8;
      cp_async16(sA + swz<BK>(r, c), src, p);
    }
  }
#pragma unroll
  for (int i0 = 0; i0 < kB; i0 += C::kThreads) {
    int i = i0 + tid;
    if (kB % C::kThreads == 0 || i < kB) {
      int r = i / C::CPR, c = i % C::CPR;
      int gr = n0 + r;
      bool p = gr < N;
      const __nv_bfloat16* src = W + static_cast<size_t>(p ? gr : 0) * ldw + k0 + c * 8;
      cp_async16(sB + swz<BK>(r, c), src, p);
    }
  }
}

template <class C, int MT, int NW, int BK, int ST>
__device__ __forceinline__ void mainloop(char* smem, const __nv_bfloat16* __restrict__ X,
                                         const __nv_bfloat16* __restrict__ W,
                                         __nv_bfloat16* __restrict__ Y, int M, int N, int K,
                                         int ldx, int ldw, int ldy, int m0, int n0, int wm0,
                                         int wn0, int tid, int lane) {
  float acc[MT][NW][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  const int ktiles = K / BK;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < ktiles) load_stage<C, BK>(smem, s, X, W, m0, n0, M, N, ldx, ldw, s * BK, tid);
    cp_async_commit();
  }

  // Per-lane ldmatrix row/chunk selectors.
  const int a_row = lane & 15;
  const int a_chk = lane >> 4;
  const int b_row = (lane & 7) + ((lane >> 4) << 3);
  const int b_chk = (lane >> 3) & 1;
  const int b2_row = lane & 7;

  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<ST - 2>();
    __syncthreads();
    {
      int nk = kt + ST - 1;
      if (nk < ktiles)
        load_stage<C, BK>(smem, nk % ST, X, W, m0, n0, M, N, ldx, ldw, nk * BK, tid);
      cp_async_commit();
    }
    const int stage = kt % ST;
    const uint32_t sA = smem_u32(smem + stage * C::kStage);
    const uint32_t sB = sA + C::kStageA;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      uint32_t a[MT][4];
      uint32_t b[NW][2];
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) {
        int r = wm0 + mi * 16 + a_row;
        ldsm_x4(sA + swz<BK>(r, kk * 2 + a_chk), a[mi][0], a[mi][1], a[mi][2], a[mi][3]);
      }
#pragma unroll
      for (int nj = 0; nj < NW / 2; ++nj) {
        int r = wn0 + nj * 16 + b_row;
        ldsm_x4(sB + swz<BK>(r, kk * 2 + b_chk), b[2 * nj][0], b[2 * nj][1], b[2 * nj + 1][0],
                b[2 * nj + 1][1]);
      }
      if constexpr (NW % 2 == 1) {
        int r = wn0 + (NW - 1) * 8 + b2_row;
        ldsm_x2(sB + swz<BK>(r, kk * 2 + b_chk), b[NW - 1][0], b[NW - 1][1]);
      }
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nj = 0; nj < NW; ++nj) mma_bf16(acc[mi][nj], a[mi], b[nj]);
    }
  }
  cp_async_wait<0>();

  // Epilogue: one RN cast per output, bf16x2 stores (N is a multiple of 8).
  const int qr = lane >> 2;
  const int qc = (lane & 3) * 2;
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
    for (int nj = 0; nj < NW; ++nj) {
      int col = n0 + wn0 + nj * 8 + qc;
      if (col < N) {
        int row = m0 + wm0 + mi * 16 + qr;
        if (row < M)
          *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(row) * ldy + col) =
              __floats2bfloat162_rn(acc[mi][nj][0], acc[mi][nj][1]);
        if (row + 8 < M)
          *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(row + 8) * ldy + col) =
              __floats2bfloat162_rn(acc[mi][nj][2], acc[mi][nj][3]);
      }
    }
  }
}

template <int MT, int NTS, int BK, int ST>
__global__ void __launch_bounds__(256, 1)
    seqk_gemm_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                     __nv_bfloat16* __restrict__ Y, int M, int N, int K, int ldx, int ldw,
                     int ldy, int ntm) {
  using C = Cfg<MT, NTS, BK, ST>;
  extern __shared__ __align__(128) char smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // Raster: consecutive CTAs share one weight tile (its second read hits L2).
  const int bid = blockIdx.x;
  const int tt = bid % ntm;
  const int wt = bid / ntm;
  const int m0 = tt * C::TM;
  const int n0 = wt * C::TN;
  const int s = warp & 3;
  const int half = warp >> 2;
  const int wm0 = (s & 1) * (MT * 16);
  const int wn_base = (s >> 1) * (NTS * 8);
  if (half == 0) {
    mainloop<C, MT, C::NA, BK, ST>(smem, X, W, Y, M, N, K, ldx, ldw, ldy, m0, n0, wm0, wn_base,
                                   tid, lane);
  } else {
    mainloop<C, MT, C::NB, BK, ST>(smem, X, W, Y, M, N, K, ldx, ldw, ldy, m0, n0, wm0,
                                   wn_base + C::NA * 8, tid, lane);
  }
}

template <int MT, int NTS, int BK, int ST>
void launch_cfg(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
  using C = Cfg<MT, NTS, BK, ST>;
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(K % BK == 0, "K must be a multiple of BK");
  TORCH_CHECK(N % 8 == 0, "N must be a multiple of 8");
  auto kern = seqk_gemm_kernel<MT, NTS, BK, ST>;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    init = true;
  }
  const int ntm = (M + C::TM - 1) / C::TM;
  const int ntn = (N + C::TN - 1) / C::TN;
  auto st = at::cuda::getCurrentCUDAStream();
  kern<<<ntm * ntn, C::kThreads, C::kSmem, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K, x.stride(0), w.stride(0),
      y.stride(0), ntm);
}

// ---------------------------------------------------------------------------------------------
// v2: one TMA producer warp + full/empty mbarrier ring (no CTA-wide barrier in the main loop),
// fragment double-buffering across k16 steps and stages, smem-staged coalesced epilogue.
// The K walk per output is unchanged: k16 steps ascending, FP32 accumulate, one RN cast.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t phase) {
  uint32_t done;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(a), "r"(phase)
        : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t mbar,
                                            int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, "
      "%4}], [%2];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(mbar), "r"(c0), "r"(c1)
      : "memory");
}

__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];\n" ::"l"(
                   reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1)
               : "memory");
}

template <int MT, int NTS, int BK, int ST, int WPS = 2, int PROBE = 0, int PF = 0>
struct Cfg2 {
  // PROBE 1: TMA + ldmatrix only (no MMA); PROBE 2: MMA on resident smem (no TMA, no waits).
  static constexpr int kProbe = PROBE;
  // PF > 0: also prefetch the weight k-tile PF tiles beyond the smem ring into L2 (TMA prefetch),
  // which keeps more DRAM requests in flight without smem.
  static constexpr int kPF = PF;
  // WPS warps per SM sub-partition; they split the SMSP quadrant's NTS n8 tiles.
  static constexpr int kWarps = 4 * WPS;
  static constexpr int kConsumers = 32 * kWarps;
  static constexpr int kThreads = kConsumers;
  static constexpr int nw(int j) { return NTS / WPS + (j < NTS % WPS ? 1 : 0); }
  static constexpr int off(int j) { return j == 0 ? 0 : off(j - 1) + nw(j - 1); }
  static constexpr int TM = 2 * MT * 16;
  static constexpr int TN = 2 * NTS * 8;
  static constexpr int kStageA = TM * BK * 2;
  static constexpr int kStageB = TN * BK * 2;
  static constexpr int kStage = kStageA + kStageB;
  static constexpr int kMain = ST * kStage;
  static constexpr int kEpiPitch = TN + 8;  // elements; 16B-aligned, conflict-free fragment writes
  static constexpr int kEpi = TM * kEpiPitch * 2;
  static constexpr int kBarOff = ((kMain > kEpi ? kMain : kEpi) + 127) / 128 * 128;
  static constexpr int kSmem = kBarOff + 2 * ST * 8 + 1024;  // + alignment slack
  static constexpr int NA = (NTS + 1) / 2;
  static constexpr int NB = NTS - NA;
  static_assert(kStageA % 1024 == 0 && kStageB % 1024 == 0, "swizzle atoms need 1KB alignment");
  static_assert((BK / 16) % 2 == 0, "fragment double buffer parity needs an even k16 count");
};

template <int MT, int NW, int BK>
__device__ __forceinline__ void load_frags(uint32_t sA, uint32_t sB, int kk, int wm0, int wn0,
                                           int lane, uint32_t (&a)[MT][4], uint32_t (&b)[NW][2]) {
  const int a_row = lane & 15;
  const int a_chk = lane >> 4;
  const int b_row = (lane & 7) + ((lane >> 4) << 3);
  const int b_chk = (lane >> 3) & 1;
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
    int r = wm0 + mi * 16 + a_row;
    ldsm_x4(sA + swz<BK>(r, kk * 2 + a_chk), a[mi][0], a[mi][1], a[mi][2], a[mi][3]);
  }
#pragma unroll
  for (int nj = 0; nj < NW / 2; ++nj) {
    int r = wn0 + nj * 16 + b_row;
    ldsm_x4(sB + swz<BK>(r, kk * 2 + b_chk), b[2 * nj][0], b[2 * nj][1], b[2 * nj + 1][0],
            b[2 * nj + 1][1]);
  }
  if constexpr (NW % 2 == 1) {
    int r = wn0 + (NW - 1) * 8 + (lane & 7);
    ldsm_x2(sB + swz<BK>(r, kk * 2 + b_chk), b[NW - 1][0], b[NW - 1][1]);
  }
}

template <class C, int MT, int NW, int BK, int ST>
__device__ __forceinline__ void consumer2(char* smem, uint32_t sbase, uint32_t bars,
                                          const CUtensorMap* tmX, const CUtensorMap* tmW,
                                          __nv_bfloat16* __restrict__ Y, int M, int N, int K,
                                          int ldy, int m0, int n0, int wm0, int wn0, int tid,
                                          int lane) {
  constexpr int KS = BK / 16;
  float acc[MT][NW][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
  uint32_t a[2][MT][4];
  uint32_t b[2][NW][2];
  const int ktiles = K / BK;
  if (C::kProbe != 2) mbar_wait(bars, 0);
  load_frags<MT, NW, BK>(sbase, sbase + C::kStageA, 0, wm0, wn0, lane, a[0], b[0]);
  for (int kt = 0; kt < ktiles; ++kt) {
    const int s = kt % ST;
    const uint32_t sA = sbase + s * C::kStage;
    // Thread 0 refills the stage every warp released at kt-1 with k-tile kt-1+ST.
    if (C::kProbe != 2 && tid == 0 && kt >= 1 && kt - 1 + ST < ktiles) {
      const int sp = (kt - 1) % ST;
      mbar_wait(bars + (ST + sp) * 8, ((kt - 1) / ST) & 1);
      const uint32_t full = bars + sp * 8;
      mbar_expect_tx(full, C::kStage);
      const uint32_t dst = sbase + sp * C::kStage;
      tma_load_2d(dst, tmX, full, (kt - 1 + ST) * BK, m0);
      tma_load_2d(dst + C::kStageA, tmW, full, (kt - 1 + ST) * BK, n0);
      if constexpr (C::kPF > 0) {
        if (kt - 1 + ST + C::kPF < ktiles) tma_prefetch_2d(tmW, (kt - 1 + ST + C::kPF) * BK, n0);
      }
    }
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      if (kk + 1 < KS) {
        load_frags<MT, NW, BK>(sA, sA + C::kStageA, kk + 1, wm0, wn0, lane, a[(kk + 1) & 1],
                               b[(kk + 1) & 1]);
      } else if (kt + 1 < ktiles) {
        const int s2 = (kt + 1) % ST;
        if (C::kProbe != 2) mbar_wait(bars + s2 * 8, ((kt + 1) / ST) & 1);
        const uint32_t sA2 = sbase + s2 * C::kStage;
        load_frags<MT, NW, BK>(sA2, sA2 + C::kStageA, 0, wm0, wn0, lane, a[0], b[0]);
      }
      if constexpr (C::kProbe != 1) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
          for (int nj = 0; nj < NW; ++nj) mma_bf16(acc[mi][nj], a[kk & 1][mi], b[kk & 1][nj]);
      } else {
        // loads-only probe: keep the fragments live without tensor work
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) acc[mi][0][0] += __uint_as_float(a[kk & 1][mi][0] & 1u);
        acc[0][0][1] += __uint_as_float(b[kk & 1][0][0] & 1u);
      }
    }
    __syncwarp();
    if (C::kProbe != 2 && lane == 0) mbar_arrive(bars + (ST + s) * 8);
  }

  // Epilogue: stage BF16 (one RN cast) through smem, then 16-byte coalesced stores.
  asm volatile("bar.sync 1, %0;\n" ::"n"(C::kThreads) : "memory");
  __nv_bfloat16* Cs = reinterpret_cast<__nv_bfloat16*>(smem);
  const int qr = lane >> 2;
  const int qc = (lane & 3) * 2;
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
    for (int nj = 0; nj < NW; ++nj) {
      int r = wm0 + mi * 16 + qr;
      int c = wn0 + nj * 8 + qc;
      *reinterpret_cast<__nv_bfloat162*>(Cs + r * C::kEpiPitch + c) =
          __floats2bfloat162_rn(acc[mi][nj][0], acc[mi][nj][1]);
      *reinterpret_cast<__nv_bfloat162*>(Cs + (r + 8) * C::kEpiPitch + c) =
          __floats2bfloat162_rn(acc[mi][nj][2], acc[mi][nj][3]);
    }
  }
  asm volatile("bar.sync 1, %0;\n" ::"n"(C::kThreads) : "memory");
  constexpr int kChunks = C::TN / 8;
  for (int i = tid; i < C::TM * kChunks; i += C::kConsumers) {
    int r = i / kChunks, c = i % kChunks;
    int gr = m0 + r, gc = n0 + c * 8;
    if (gr < M && gc < N)
      *reinterpret_cast<uint4*>(Y + static_cast<size_t>(gr) * ldy + gc) =
          *reinterpret_cast<const uint4*>(Cs + r * C::kEpiPitch + c * 8);
  }
}

template <int MT, int NTS, int BK, int ST, int WPS, int PROBE, int PF>
__global__ void __launch_bounds__(128 * WPS, 1)
    seqk_tma_kernel(const __grid_constant__ CUtensorMap tmX, const __grid_constant__ CUtensorMap tmW,
                    __nv_bfloat16* __restrict__ Y, int M, int N, int K, int ldy, int ntm) {
  using C = Cfg2<MT, NTS, BK, ST, WPS, PROBE, PF>;
  extern __shared__ __align__(1024) char smem_raw[];
  char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) &
                                       ~static_cast<uintptr_t>(1023));
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bars = sbase + C::kBarOff;  // full[ST] then empty[ST]
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bid = blockIdx.x;
  const int tt = bid % ntm;
  const int wt = bid / ntm;
  const int m0 = tt * C::TM;
  const int n0 = wt * C::TN;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      mbar_init(bars + s * 8, 1);
      mbar_init(bars + (ST + s) * 8, C::kWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    const int ktiles = K / BK;
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      if (PROBE != 2 && s < ktiles) {
        const uint32_t full = bars + s * 8;
        mbar_expect_tx(full, C::kStage);
        const uint32_t dst = sbase + s * C::kStage;
        tma_load_2d(dst, &tmX, full, s * BK, m0);
        tma_load_2d(dst + C::kStageA, &tmW, full, s * BK, n0);
      }
    }
    if constexpr (PF > 0 && PROBE != 2) {
#pragma unroll 1
      for (int t = ST; t < ST + PF && t < ktiles; ++t) tma_prefetch_2d(&tmW, t * BK, n0);
    }
  }
  __syncthreads();
  const int s4 = warp & 3;
  const int j = warp >> 2;
  const int wm0 = (s4 & 1) * (MT * 16);
  const int wn_base = (s4 >> 1) * (NTS * 8);
#define SEQK_RUN(J)                                                                         \
  consumer2<C, MT, C::nw(J), BK, ST>(smem, sbase, bars, &tmX, &tmW, Y, M, N, K, ldy, m0, n0, \
                                     wm0, wn_base + C::off(J) * 8, tid, lane)
  if (j == 0) {
    SEQK_RUN(0);
  }
  if constexpr (WPS > 1) {
    if (j == 1) SEQK_RUN(1);
  }
  if constexpr (WPS > 2) {
    if (j == 2) SEQK_RUN(2);
  }
  if constexpr (WPS > 3) {
    if (j == 3) SEQK_RUN(3);
  }
#undef SEQK_RUN
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                                     cudaEnableDefault, &q);
    TORCH_CHECK(e == cudaSuccess && q == cudaDriverEntryPointSuccess && p,
                "cuTensorMapEncodeTiled lookup failed");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

static CUtensorMap make_map(const void* ptr, int64_t rows, int64_t K, int64_t ld, int box_rows,
                            int BK) {
  static std::map<std::tuple<const void*, int64_t, int64_t, int64_t, int, int>, CUtensorMap> cache;
  auto key = std::make_tuple(ptr, rows, K, ld, box_rows, BK);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap m;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(ld) * 2};
  cuuint32_t box[2] = {static_cast<cuuint32_t>(BK), static_cast<cuuint32_t>(box_rows)};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
                           strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           BK == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  if (cache.size() > 4096) cache.clear();
  cache.emplace(key, m);
  return m;
}

template <int MT, int NTS, int BK, int ST, int WPS, int PROBE = 0, int PF = 0>
void launch_tma(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
  using C = Cfg2<MT, NTS, BK, ST, WPS, PROBE, PF>;
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(K % BK == 0, "K must be a multiple of BK");
  TORCH_CHECK(N % 8 == 0 && y.stride(0) % 8 == 0, "N and ldy must be multiples of 8");
  auto kern = seqk_tma_kernel<MT, NTS, BK, ST, WPS, PROBE, PF>;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    init = true;
  }
  CUtensorMap tx = make_map(x.data_ptr(), M, K, x.stride(0), C::TM, BK);
  CUtensorMap tw = make_map(w.data_ptr(), N, K, w.stride(0), C::TN, BK);
  const int ntm = (M + C::TM - 1) / C::TM;
  const int ntn = (N + C::TN - 1) / C::TN;
  auto st = at::cuda::getCurrentCUDAStream();
  kern<<<ntm * ntn, C::kThreads, C::kSmem, st>>>(tx, tw,
                                                 reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
                                                 M, N, K, y.stride(0), ntm);
}

}  // namespace seqk

// cfg ids: tile tokens x weights, BK, stages.
void seqk_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t cfg) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1);
  switch (cfg) {
    case 0: seqk::launch_cfg<5, 11, 64, 2>(x, w, y); break;  // 160x176 BK64 S2
    case 10: seqk::launch_tma<5, 11, 64, 2, 2>(x, w, y); break;  // TMA 160x176 BK64 S2, 8 warps
    case 11: seqk::launch_tma<5, 11, 32, 4, 2>(x, w, y); break;  // TMA 160x176 BK32 S4, 8 warps
    case 13: seqk::launch_tma<5, 11, 64, 2, 3>(x, w, y); break;  // TMA 160x176 BK64 S2, 12 warps
    case 14: seqk::launch_tma<5, 11, 32, 4, 3>(x, w, y); break;  // TMA 160x176 BK32 S4, 12 warps
    case 20: seqk::launch_tma<5, 11, 32, 4, 3, 1>(x, w, y); break;  // probe: loads only
    case 21: seqk::launch_tma<5, 11, 32, 4, 3, 2>(x, w, y); break;  // probe: compute only
    case 22: seqk::launch_tma<5, 11, 32, 4, 3, 1, 8>(x, w, y); break;  // probe: loads only, PF8
    case 30: seqk::launch_tma<5, 11, 32, 4, 3, 0, 4>(x, w, y); break;  // 12w BK32 S4 PF4
    case 31: seqk::launch_tma<5, 11, 32, 4, 3, 0, 8>(x, w, y); break;  // 12w BK32 S4 PF8
    case 32: seqk::launch_tma<5, 11, 32, 4, 3, 0, 16>(x, w, y); break;  // 12w BK32 S4 PF16
    case 33: seqk::launch_tma<5, 11, 64, 2, 3, 0, 4>(x, w, y); break;  // 12w BK64 S2 PF4
    case 34: seqk::launch_tma<5, 11, 64, 2, 3, 0, 8>(x, w, y); break;  // 12w BK64 S2 PF8
    default: TORCH_CHECK(false, "bad cfg");
  }
}

std::vector<int64_t> seqk_tile(int64_t cfg) {
  switch (cfg) {
    case 0: case 10: case 11: case 13: case 14: case 20: case 21: case 22: case 30: case 31:
    case 32: case 33: case 34:
      return {160, 176};
    default: return {0, 0};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("seqk_gemm", &seqk_gemm);
  m.def("seqk_tile", &seqk_tile);
}
