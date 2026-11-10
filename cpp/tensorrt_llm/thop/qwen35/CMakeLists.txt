# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# qwen3_5 modeling_v2 target bindings; they keep their existing family target.

target_sources(th_common_gemm_quant PRIVATE qwen35Nvfp4W4a16GemvOp.cpp
                                            qwen35Nvfp4GateUpSwigluOp.cpp)
