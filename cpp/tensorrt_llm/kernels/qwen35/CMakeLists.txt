#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# Qwen3.5-family staircase target kernels. They run on SM120/121 only, so they
# are built for the sm_120 family (placeholder kernels without it). The context
# FMHA launcher (qwen35ContextFmhaSm120.cu) is built with the generated FMHA
# kernels instead, see contextFusedMultiHeadAttention/CMakeLists.txt.
add_library(qwen35_kernels_src OBJECT qwen35Nvfp4W4a16Gemv.cu
                                      qwen35Nvfp4GateUpSwigluSm120.cu)
set_property(TARGET qwen35_kernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET qwen35_kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_cuda_architectures(qwen35_kernels_src 120f)
# The CUTLASS GEMM waits on its PDL predecessor (grid dependency control).
target_compile_options(qwen35_kernels_src
                       PRIVATE "-DCUTLASS_ENABLE_GDC_FOR_SM100=1")
if(NOT WIN32)
  target_compile_options(
    qwen35_kernels_src
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
endif()
