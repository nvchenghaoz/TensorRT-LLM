# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# FLUX.3 (flux3_klein_nvfp4__sm_120__tp1) modeling_v2 target kernels. The GEMM
# family builds the flux3_kernels_src OBJECT library; the FMHA launcher in fmha/
# is compiled with the generated sm_120 context FMHA kernels instead.
add_subdirectory(gemm)
