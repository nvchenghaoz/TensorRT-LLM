// STEP 0 prototype (branch 128 / idea 226): NVFP4 W4A4 gate_up GEMM on SM120
// with SwiGLU + NVFP4 (group-16, E4M3 SF, 128x4 swizzle) quantize in the
// epilogue. Mainloop identical to main's nvfp4_nvfp4_gemm_template_sm120.h
// (CollectiveBuilder Sm120 OpClassBlockScaledTensorOp, Cooperative,
// StaticPersistentScheduler, swizzle 1, raster heuristic). The epilogue is a
// register-only operator epilogue (Sm90TmaWarpSpecializedAdapter).
//
// Weight layout: gate_up rows permuted at load so that, inside every 128-wide
// N tile t, each thread's accumulator pair (c0,c1) of every n8 atom is
// (gate_j, up_j). With the SM120 tiled MMA (4x2 warps, N permutation
// (8,2,2):(1,16,8)), MMA_N index k (0..7) of a thread with warp-n nw and quad
// lane lq covers tile cols 32*(k/2) + 16*nw + 8*(k%2) + 2*lq + {0,1}.
// We assign it output j = 64*t + 32*nw + 16*(k/4) + 4*lq + (k%4), so a lane
// quad owns two whole 16-element SF blocks per row.
//
// Per-element math, bitwise copy of main (gate_up GEMM LinCombPerColBias ->
// bf16, then fused_silu_mul_nvfp4.py):
//   g = bf16_rn(alpha*acc + 0), u = bf16_rn(alpha*acc + 0)
//   sig = div.full(1, add(ex2.approx(mul(g, -log2e)), 1))
//   h = bf16_rn(mul(add(u, 0), mul(sig, g)))
//   tail = landing 13 quant (rcp.approx.ftz, mul.rn, cvt e4m3, cvt e2m1x2).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace swiglu_epi
{

__device__ __forceinline__ float rcp_approx_ftz(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float mul_rn(float a, float b)
{
    float y;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float add_rn(float a, float b)
{
    float y;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float ex2_approx(float a)
{
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(a));
    return y;
}

__device__ __forceinline__ float div_full(float a, float b)
{
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float bf16_round(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint32_t cvt_e4m3(float x)
{
    uint32_t r;
    asm("{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\ncvt.u32.u16 %0, t;\n}"
        : "=r"(r)
        : "f"(x), "f"(0.0f));
    return r & 0xff;
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b)
{
    uint32_t h2;
    uint16_t v = (uint16_t) b;
    asm("{\ncvt.rn.f16x2.e4m3x2 %0, %1;\n}" : "=r"(h2) : "h"(v));
    __half lo = __ushort_as_half((unsigned short) (h2 & 0xffff));
    return __half2float(lo);
}

__device__ __forceinline__ uint32_t cvt_e2m1x4(float e0, float e1, float e2, float e3)
{
    uint32_t r;
    asm("{\n.reg .b8 b0;\n.reg .b8 b1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "mov.b32 %0, {b0, b1, b0, b1};\n}"
        : "=r"(r)
        : "f"(e0), "f"(e1), "f"(e2), "f"(e3));
    return r & 0xffff;
}

__device__ __forceinline__ float swiglu_elem(float alpha, float acc_g, float acc_u)
{
    // GEMM epilogue (LinCombPerColBias, null bias, beta 0): alpha*acc + 0 -> bf16.
    float a = bf16_round(__fmaf_rn(alpha, acc_g, 0.0f));
    float b = bf16_round(__fmaf_rn(alpha, acc_u, 0.0f));
    // tl.sigmoid as Triton lowers it (job 1036 PTX).
    float t = mul_rn(a, -1.44269502162933349609375f); // 0fBFB8AA3B
    float ex = ex2_approx(t);
    float den = add_rn(ex, 1.0f);
    float sig = div_full(1.0f, den);
    float sa = mul_rn(sig, a);
    float bb = add_rn(b, 0.0f);
    return bf16_round(mul_rn(bb, sa));
}

template <class StrideC_, class StrideD_>
class SwigluNvfp4Epilogue
{
public:
    using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
    using DispatchPolicy = EpilogueSchedule;
    using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 1, float, float>;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;
    using ElementC = void;
    using StrideC = StrideC_;
    using ElementD = cutlass::bfloat16_t;
    using StrideD = StrideD_;
    using GmemElementC = ElementD;
    using GmemTiledCopyC = void;
    using GmemTiledCopyD = void;

    struct SharedStorage
    {
    };

    using TensorStorage = SharedStorage;

    struct Arguments
    {
        float const* alpha_ptr = nullptr;
        float const* out_scale_ptr = nullptr;
        uint8_t* fp4_ptr = nullptr;
        uint8_t* sf_ptr = nullptr;
        int num_k_tiles = 0;
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
    SwigluNvfp4Epilogue(Params const& params_, SharedStorage const& = SharedStorage())
        : params(params_)
    {
    }

    CUTLASS_DEVICE bool is_source_needed()
    {
        return false;
    }

    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine, class FrgLayout,
        class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK, BlockCoordMNKL blk_coord_mnkl,
        cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TiledMma tiled_mma, ResidueMNK, int thread_idx, char*)
    {
        constexpr int BLK_M = size<0>(BlockShapeMNK{});
        constexpr int BLK_N = size<1>(BlockShapeMNK{});
        static_assert(BLK_N == 128, "tile N must be 128");
        int const M = get<0>(problem_shape_mnkl);
        int const N = get<1>(problem_shape_mnkl);
        auto m_coord = get<0>(blk_coord_mnkl);
        auto n_coord = get<1>(blk_coord_mnkl);

        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor cD = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{}));
        Tensor tCcD = thr_mma.partition_C(cD);
        static_assert(size<0>(FrgLayout{}) == 4, "acc fragment (2,2)");
        constexpr int MMA_M = size<1>(FrgLayout{});
        constexpr int MMA_N = size<2>(FrgLayout{});
        static_assert(MMA_N == 8, "MMA_N must be 8");

        float const alpha = *params.alpha_ptr;
        float const s = *params.out_scale_ptr;
        float const r6 = rcp_approx_ftz(6.0f);
        float const rs = rcp_approx_ftz(s);

        int const n0 = get<1>(tCcD(0, 0, 0));
        int const nw = n0 / 16;
        int const lq = (n0 % 16) / 2;
#ifdef SWIGLU_EPI_CHECK
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < MMA_N; ++k)
        {
            int nn = get<1>(tCcD(0, 0, k));
            int ck = nn - n0;
            if (n0 >= 32 || (ck % 8) != 0 || get<1>(tCcD(1, 0, k)) != nn + 1
                || get<0>(tCcD(1, 0, k)) != get<0>(tCcD(0, 0, k)) || get<0>(tCcD(2, 0, k)) != get<0>(tCcD(0, 0, k)) + 8
                || get<1>(tCcD(2, 0, k)) != nn || get<1>(tCcD(3, 0, k)) != nn + 1)
            {
                printf("SWIGLU_EPI layout mismatch thr %d k %d n %d n0 %d\n", thread_idx, k, nn, n0);
                __trap();
            }
        }
#endif
        int const d = N / 2;
        int const m_pad = (M + 127) / 128 * 128;
        int const j_base = int(n_coord) * (BLK_N / 2) + 32 * nw + 4 * lq;
        int const m_base = int(m_coord) * BLK_M;
        int const nkt = params.num_k_tiles;

        CUTLASS_PRAGMA_UNROLL
        for (int im = 0; im < MMA_M; ++im)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int r2 = 0; r2 < 2; ++r2)
            {
                int const row = m_base + get<0>(tCcD(2 * r2, im, 0));
                // hv[sb][e]: tile col offset ck = 32a + 8b of MMA_N index k
                // (static after unrolling) -> output j_base + 16*(a/2) + 2*(a%2) + b.
                float hv[2][4];
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < MMA_N; ++k)
                {
                    int const ck = get<1>(tCcD(0, im, k)) - get<1>(tCcD(0, im, 0));
                    int const a_ = ck / 32;
                    int const b_ = (ck % 32) / 8;
                    hv[a_ / 2][2 * (a_ % 2) + b_]
                        = swiglu_elem(alpha, accumulators(2 * r2, im, k), accumulators(2 * r2 + 1, im, k));
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sb = 0; sb < 2; ++sb)
                {
                    float vmax = 0.0f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int e = 0; e < 4; ++e)
                    {
                        vmax = fmaxf(vmax, fabsf(hv[sb][e]));
                    }
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, 1));
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, 2));
                    float const sfv = mul_rn(s, mul_rn(vmax, r6));
                    uint32_t const sfb = cvt_e4m3(sfv);
                    float const sff = e4m3_to_f32(sfb);
                    float const osc = (vmax != 0.0f) ? rcp_approx_ftz(mul_rn(sff, rs)) : 0.0f;
                    uint32_t const packed = cvt_e2m1x4(mul_rn(hv[sb][0], osc), mul_rn(hv[sb][1], osc),
                        mul_rn(hv[sb][2], osc), mul_rn(hv[sb][3], osc));
                    int const j = j_base + 16 * sb;
                    if (row < M)
                    {
                        *reinterpret_cast<uint16_t*>(params.fp4_ptr + size_t(row) * (d / 2) + j / 2)
                            = (uint16_t) packed;
                    }
                    if (lq == 0 && row < m_pad)
                    {
                        int const c = j / 16;
                        size_t off = (size_t(row / 128) * nkt + c / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4
                            + c % 4;
                        params.sf_ptr[off] = row < M ? (uint8_t) sfb : (uint8_t) 0;
                    }
                }
            }
        }
    }

private:
    Params params;
};

template <int TM, int TN, int TK>
struct FusedGemm
{
    using CTAShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Arch = cutlass::arch::Sm120;
    using ElementA = cutlass::float_e2m1_t;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;
    using ElementB = cutlass::float_e2m1_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;
    using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using EpiOp = SwigluNvfp4Epilogue<StrideCD, StrideCD>;
    using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpiOp>;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<Arch,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementPairA, LayoutA, AlignmentA, ElementPairB, LayoutB,
        AlignmentB, float, CTAShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue, cutlass::gemm::StaticPersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <int TM, int TN, int TK>
void run_fused(torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& sfa, torch::Tensor const& sfb,
    torch::Tensor const& alpha, torch::Tensor const& out_scale, torch::Tensor& fp4, torch::Tensor& sf, bool pdl)
{
    using G = FusedGemm<TM, TN, TK>;
    using Gemm = typename G::Gemm;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    int const m = a.size(0);
    int const k = a.size(1) * 2;
    int const n = b.size(0);
    int const d = n / 2;
    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);
    args.mainloop.ptr_A = reinterpret_cast<typename G::ElementA const*>(a.data_ptr());
    args.mainloop.ptr_B = reinterpret_cast<typename G::ElementB const*>(b.data_ptr());
    args.mainloop.ptr_SFA = reinterpret_cast<cutlass::float_ue4m3_t const*>(sfa.data_ptr());
    args.mainloop.ptr_SFB = reinterpret_cast<cutlass::float_ue4m3_t const*>(sfb.data_ptr());
    args.mainloop.dA = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
    args.mainloop.dB = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
    args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
    args.epilogue.alpha_ptr = alpha.data_ptr<float>();
    args.epilogue.out_scale_ptr = out_scale.data_ptr<float>();
    args.epilogue.fp4_ptr = fp4.data_ptr<uint8_t>();
    args.epilogue.sf_ptr = sf.data_ptr<uint8_t>();
    args.epilogue.num_k_tiles = (d / 16 + 3) / 4;
    if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>)
    {
        args.scheduler.max_swizzle_size = 1;
    }
    if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>)
    {
        using Enum_t = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = Enum_t::Heuristic;
    }
    args.hw_info.cluster_shape = dim3(1, 1, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    static int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    args.hw_info.sm_count = sm_count;

    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cannot implement");
    size_t ws = gemm.get_workspace_size(args);
    auto workspace = torch::empty({(int64_t) std::max<size_t>(ws, 16)}, a.options().dtype(torch::kUInt8));
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    auto st = gemm.run(args, workspace.data_ptr(), stream, nullptr, pdl);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "run failed ", cutlass::cutlassGetStatusString(st));
}

std::vector<torch::Tensor> fused_gate_up_swiglu_nvfp4(torch::Tensor a, torch::Tensor b, torch::Tensor sfa,
    torch::Tensor sfb, torch::Tensor alpha, torch::Tensor out_scale, int64_t tile, bool pdl)
{
    c10::cuda::CUDAGuard g(a.device());
    int64_t m = a.size(0);
    int64_t n = b.size(0);
    int64_t d = n / 2;
    int64_t m_pad = (m + 127) / 128 * 128;
    int64_t nkt = (d / 16 + 3) / 4;
    auto fp4 = torch::empty({m, d / 2}, a.options().dtype(torch::kUInt8));
    auto sf = torch::empty({m_pad * nkt * 4}, a.options().dtype(torch::kUInt8));
    if (tile == 0)
        run_fused<256, 128, 128>(a, b, sfa, sfb, alpha, out_scale, fp4, sf, pdl);
    else
        run_fused<128, 128, 256>(a, b, sfa, sfb, alpha, out_scale, fp4, sf, pdl);
    return {fp4, sf};
}

} // namespace swiglu_epi

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_gate_up_swiglu_nvfp4", &swiglu_epi::fused_gate_up_swiglu_nvfp4);
}
