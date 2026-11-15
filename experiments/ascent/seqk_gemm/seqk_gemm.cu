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
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

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

}  // namespace seqk

// cfg ids: tile tokens x weights, BK, stages.
void seqk_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t cfg) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1);
  switch (cfg) {
    case 0: seqk::launch_cfg<5, 11, 64, 2>(x, w, y); break;  // 160x176 BK64 S2
    case 1: seqk::launch_cfg<5, 11, 32, 3>(x, w, y); break;  // 160x176 BK32 S3
    case 2: seqk::launch_cfg<5, 11, 32, 4>(x, w, y); break;  // 160x176 BK32 S4
    case 3: seqk::launch_cfg<4, 12, 64, 2>(x, w, y); break;  // 128x192 BK64 S2
    case 4: seqk::launch_cfg<4, 12, 32, 4>(x, w, y); break;  // 128x192 BK32 S4
    default: TORCH_CHECK(false, "bad cfg");
  }
}

std::vector<int64_t> seqk_tile(int64_t cfg) {
  switch (cfg) {
    case 0: case 1: case 2: return {160, 176};
    case 3: case 4: return {128, 192};
    default: return {0, 0};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("seqk_gemm", &seqk_gemm);
  m.def("seqk_tile", &seqk_tile);
}
