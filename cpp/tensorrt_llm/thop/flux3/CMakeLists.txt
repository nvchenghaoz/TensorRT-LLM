# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# FLUX.3 modeling_v2 target bindings; they join existing family targets.

target_sources(th_common_attention PRIVATE flux3FmhaOp.cpp)
target_sources(th_common_gemm_quant PRIVATE flux3GemmOps.cpp)
