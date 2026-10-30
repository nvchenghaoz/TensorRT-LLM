#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#

# FLUX.3 (flux3_klein_nvfp4__sm_120__tp1) denoiser GEMMs, their fused producers
# and epilogues. They run on SM120/121 only, so they are built for the sm_120
# family (placeholder kernels without it).
add_library(
  flux3_kernels_src OBJECT
  flux3QkvEpiFp8Sm120.cu flux3GateUpSwigluNvfp4Sm120.cu
  flux3PairResidualSm120.cu flux3Fp8RowwiseGemmSm120.cu flux3AdalnNvfp4Sm120.cu)
set_property(TARGET flux3_kernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET flux3_kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_cuda_architectures(flux3_kernels_src 120f)
# The CUTLASS GEMMs wait on their PDL predecessor (grid dependency control).
target_compile_options(flux3_kernels_src
                       PRIVATE "-DCUTLASS_ENABLE_GDC_FOR_SM100=1")
# Math flags per source, which the bitwise outputs require: --use_fast_math for
# the CUTLASS GEMMs (their epilogues pin the rounding-sensitive arithmetic with
# PTX; the plain fp32 products left, BF16(alpha * acc) in the pair and QKV
# epilogues and the FP8 row-wise EVT multiplies, are the ftz forms under fast
# math), and IEEE fp32 (--fmad=false --prec-div=true --prec-sqrt=true
# --ftz=false) for the adaLN kernel, which also pins every fp32 op with PTX.
set_source_files_properties(
  flux3QkvEpiFp8Sm120.cu flux3GateUpSwigluNvfp4Sm120.cu
  flux3PairResidualSm120.cu flux3Fp8RowwiseGemmSm120.cu
  PROPERTIES COMPILE_OPTIONS "--use_fast_math")
set_source_files_properties(
  flux3AdalnNvfp4Sm120.cu
  PROPERTIES COMPILE_OPTIONS
             "--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false")
if(NOT WIN32)
  target_compile_options(
    flux3_kernels_src PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
endif()
