/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "fp4_gemm_template.h"
#include "tensorrt_llm/common/config.h"
#ifndef PLACEHOLDER_KERNELS
#include "nvfp4_swiglu_gemm_sm120.h"
#endif
#include <stdexcept>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_BF16
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 64, 128, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 128, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 128, 256, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm100, __nv_bfloat16, 128, 256, 256, 4, 4, 1, _2SM)

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 128, 256, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(Sm103, __nv_bfloat16, 128, 256, 256, 4, 4, 1, _2SM)

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 128, 128, 128, 1, 1, 1)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 128, 128, 256, 1, 1, 1)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 256, 128, 128, 1, 1, 1)

template class CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>;
template class CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A8_MXFP4_MXFP8>;
template class CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W8A8_MXFP8_MXFP8>;

#endif

size_t nvfp4SwigluGemmSm120(void const* A, void const* B, void const* inputSf, void const* weightSf,
    float const* alpha, float const* outScale, uint8_t* fp4Out, uint8_t* sfOut, int m, int n, int k, int tile,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, bool enablePdl)
{
#ifdef PLACEHOLDER_KERNELS
    throw std::runtime_error("[TensorRT LLM Error][SwiGLU NVFP4 GEMM] not compiled for this architecture.");
#else
    if (tile == 0)
    {
        return swiglu_nvfp4_sm120::runSwigluNvfp4GemmSm120<256, 128, 128>(A, B, inputSf, weightSf, alpha, outScale,
            fp4Out, sfOut, m, n, k, workspace, workspaceBytes, stream, enablePdl);
    }
    if (tile == 2)
    {
        // 128x128x256 with the MMAs of tile rows 64-127 skipped (m <= 64 only).
        return swiglu_nvfp4_sm120::runSwigluNvfp4GemmSm120<128, 128, 256, true>(A, B, inputSf, weightSf, alpha,
            outScale, fp4Out, sfOut, m, n, k, workspace, workspaceBytes, stream, enablePdl);
    }
    return swiglu_nvfp4_sm120::runSwigluNvfp4GemmSm120<128, 128, 256>(A, B, inputSf, weightSf, alpha, outScale,
        fp4Out, sfOut, m, n, k, workspace, workspaceBytes, stream, enablePdl);
#endif
}

} // namespace cutlass_kernels
} // namespace kernels

TRTLLM_NAMESPACE_END
