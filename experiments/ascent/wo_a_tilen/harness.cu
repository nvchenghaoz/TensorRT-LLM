// Standalone harness: SM120 FP8 1x128 block-scale strided BMM at TileN 128/64/32 (TileM 32/64, 4 stages).
// Mirrors launch_sm120_gemm_kernel in fp8_blockscale_gemm_kernel.cuh exactly, but templated on TileN.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "sm120_blockwise_gemm/sm120_fp8_gemm_1d1d.cuh"

template <int TileM, int TileN, int NumStages>
void launch_tile(void* mat_a, int64_t ld_a, int64_t stride_a, void* mat_b, int64_t ld_b, int64_t stride_b, void* mat_d,
    int64_t ld_d, int64_t stride_d, void* scales_a, void* scales_b, int num_problems, int shape_m, int shape_n,
    int shape_k, cudaStream_t stream, int num_sms)
{
    using KT = sm120_blockscaled_gemm::SM120BlockScaledBuilder<TileM, TileN, NumStages>;
    using GemmKernel = sm120_blockscaled_gemm::SM120BlockScaledKernel<KT>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape(shape_m, shape_n, shape_k, num_problems);
    auto ptr_A = reinterpret_cast<cute::float_e4m3_t*>(mat_a);
    auto ptr_B = reinterpret_cast<cute::float_e4m3_t*>(mat_b);
    auto ptr_SFA = reinterpret_cast<int32_t*>(scales_a);
    auto ptr_SFB = reinterpret_cast<int32_t*>(scales_b);
    auto ptr_D = reinterpret_cast<cute::bfloat16_t*>(mat_d);
    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    typename KT::StrideSFA dSFA = KT::deduce_sfa_layout(problem_shape).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);
    Arguments args = {ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD};
    Params kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;
    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = dim3(num_sms, 1, 1);
    cfg.blockDim = GemmKernel::get_block_shape();
    cfg.dynamicSmemBytes = GemmKernel::kSmemSize;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel_ptr, kernel_params);
    auto r = cudaGetLastError();
    TORCH_CHECK(r == cudaSuccess, "launch failed: ", cudaGetErrorString(r));
}

// a: fp8 [L, M, K] (any strides with last=1), b: fp8 [L, N, K], sfa int32, sfb int32, d: bf16 view [L, M, N]
void run(int tile_m, int tile_n, torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb,
    torch::Tensor d, int num_sms)
{
    int L = a.size(0), M = a.size(1), K = a.size(2), N = b.size(1);
    TORCH_CHECK(a.stride(2) == 1 && b.stride(2) == 1 && d.stride(2) == 1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
#define ARGS                                                                                                           \
    a.data_ptr(), a.stride(1), a.stride(0), b.data_ptr(), b.stride(1), b.stride(0), d.data_ptr(), d.stride(1),         \
        d.stride(0), sfa.data_ptr(), sfb.data_ptr(), L, M, N, K, stream, num_sms
    if (tile_m == 32 && tile_n == 128)
        launch_tile<32, 128, 4>(ARGS);
    else if (tile_m == 32 && tile_n == 64)
        launch_tile<32, 64, 4>(ARGS);
    else if (tile_m == 32 && tile_n == 32)
        launch_tile<32, 32, 4>(ARGS);
    else if (tile_m == 64 && tile_n == 128)
        launch_tile<64, 128, 4>(ARGS);
    else if (tile_m == 64 && tile_n == 64)
        launch_tile<64, 64, 4>(ARGS);
    else if (tile_m == 64 && tile_n == 32)
        launch_tile<64, 32, 4>(ARGS);
    else
        TORCH_CHECK(false, "unsupported tile");
#undef ARGS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run);
}
