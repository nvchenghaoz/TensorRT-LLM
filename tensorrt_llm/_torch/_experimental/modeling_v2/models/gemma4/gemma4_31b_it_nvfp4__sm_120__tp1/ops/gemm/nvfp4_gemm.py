# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 x NVFP4 GEMM (the MLP projections), on the shared runners with this target's tuning.

`nvfp4_gemm(act_fp4, weight, act_sf, weight_scale, alpha, out_dtype, backends)` runs what the
shared `trtllm::nvfp4_gemm` runs, except that for the Gemma-4-31B MLP shapes it tunes on dense
M buckets with cold L2, adds a CUTLASS-only band for down_proj at M 513-1023, and pins the
tactic where the tuner's pick is a per-process coin flip (`_nvfp4_gemm_tuning`). Every tactic
those offer is bitwise equal on real activations over its range.
"""

from __future__ import annotations

from typing import List

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._nvfp4_gemm_tuning import nvfp4_gemm as _impl


class Nvfp4Gemm(OpWrapper):
    ARCHS = frozenset({Arch.SM_120})
    CELLS = tuple(
        Cell(why=f"{name} at {m} rows", spec=dict(n=n, k=k, m=m))
        for name, n, k in (("gate_up", 43008, 5376), ("down", 5376, 21504))
        for m in (1, 300, 1000, 2000, 4000, 8000)
    )

    def __call__(self, act_fp4, weight, act_sf, weight_scale, alpha, out_dtype, backends: List[str]):
        return _impl(act_fp4, weight, act_sf, weight_scale, alpha, out_dtype, backends)

    def reference(self, act_fp4, weight, act_sf, weight_scale, alpha, out_dtype, backends):
        """The shared op (its own tuned pick), which every tactic here matches bitwise."""
        return torch.ops.trtllm.nvfp4_gemm(
            act_fp4, weight, act_sf, weight_scale, alpha, out_dtype,
            allowed_backends=",".join(backends),
        )


nvfp4_gemm = Nvfp4Gemm()
