# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    from flashinfer.activation import gelu_tanh_and_mul, silu_and_mul
    from flashinfer.norm import (fused_add_rmsnorm, fused_add_rmsnorm_quant,
                                 gemma_fused_add_rmsnorm, gemma_rmsnorm,
                                 rmsnorm)
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    from flashinfer.utils import device_support_pdl

    _gemma_1row = None

    def _get_gemma_1row():
        # Lazy: avoids an import cycle through tensorrt_llm._torch.modules.
        global _gemma_1row
        if _gemma_1row is None:
            from ..modules import gemma_add_rmsnorm_1row
            _gemma_1row = gemma_add_rmsnorm_1row
        return _gemma_1row

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x, enable_pdl=get_env_enable_pdl())

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    @torch.library.custom_op("trtllm::flashinfer_gelu_tanh_and_mul",
                             mutates_args=())
    def flashinfer_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
        return gelu_tanh_and_mul(x, enable_pdl=get_env_enable_pdl())

    @flashinfer_gelu_tanh_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return rmsnorm(input, weight, eps, enable_pdl=get_env_enable_pdl())

    @flashinfer_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_gemma_rmsnorm",
                             mutates_args=())
    def flashinfer_gemma_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                                 eps: float) -> torch.Tensor:
        return gemma_rmsnorm(input,
                             weight,
                             eps,
                             enable_pdl=get_env_enable_pdl())

    @flashinfer_gemma_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        fused_add_rmsnorm(input,
                          residual,
                          weight,
                          eps,
                          enable_pdl=get_env_enable_pdl())

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm_quant",
                             mutates_args=("out", "residual"))
    def flashinfer_fused_add_rmsnorm_quant(out: torch.Tensor,
                                           input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           scale: torch.Tensor,
                                           eps: float) -> None:
        fused_add_rmsnorm_quant(out,
                                input,
                                residual,
                                weight,
                                scale,
                                eps,
                                enable_pdl=get_env_enable_pdl())

    @flashinfer_fused_add_rmsnorm_quant.register_fake
    def _(out: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
          weight: torch.Tensor, scale: torch.Tensor, eps: float) -> None:
        pass

    @torch.library.custom_op("trtllm::flashinfer_gemma_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_gemma_fused_add_rmsnorm(input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           eps: float) -> None:
        g1 = _get_gemma_1row()
        if g1.use_1row(input, residual):
            # Decode rows: exact 1-row-per-CTA geometry (bitwise to stock).
            tpr, arm = g1.routed_tpr_arm()
            enable_pdl = get_env_enable_pdl() and device_support_pdl(
                input.device)
            g1.gemma_fused_add_rmsnorm_1row(input,
                                            residual,
                                            weight,
                                            eps,
                                            enable_pdl,
                                            tpr=tpr,
                                            arm=arm)
            return
        gemma_fused_add_rmsnorm(input,
                                residual,
                                weight,
                                eps,
                                enable_pdl=get_env_enable_pdl())

    # Same math and outputs as flashinfer_gemma_fused_add_rmsnorm (bitwise); only
    # the PDL trigger moves to right after the kernel's loads land, so its PDL
    # dependent (e.g. a dense Marlin decode GEMM prefetching its weights before
    # griddepcontrol.wait) launches while the norm computes. CuTe-DSL kernel in
    # flashinfer/norm/kernels/fused_add_rmsnorm_early_trigger.py (a subclass of
    # the pinned FusedAddRMSNormKernel); falls back to the stock op without PDL
    # or on the CUDA-JIT norm path.
    @torch.library.custom_op(
        "trtllm::flashinfer_gemma_fused_add_rmsnorm_early_trigger",
        mutates_args=("input", "residual"))
    def flashinfer_gemma_fused_add_rmsnorm_early_trigger(
            input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
            eps: float) -> None:
        from flashinfer.norm import _use_cuda_norm
        from flashinfer.utils import device_support_pdl
        enable_pdl = get_env_enable_pdl() and device_support_pdl(input.device)
        if not enable_pdl or _use_cuda_norm(input.device):
            gemma_fused_add_rmsnorm(input,
                                    residual,
                                    weight,
                                    eps,
                                    enable_pdl=enable_pdl)
            return
        from flashinfer.norm.kernels.fused_add_rmsnorm_early_trigger import \
            gemma_fused_add_rmsnorm_early_trigger
        gemma_fused_add_rmsnorm_early_trigger(input,
                                              residual,
                                              weight,
                                              eps,
                                              enable_pdl=True)

    @torch.library.custom_op(
        "trtllm::flashinfer_apply_rope_with_cos_sin_cache_inplace",
        mutates_args=("query", "key"))
    def flashinfer_apply_rope_with_cos_sin_cache_inplace(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> None:
        apply_rope_with_cos_sin_cache_inplace(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

    @flashinfer_apply_rope_with_cos_sin_cache_inplace.register_fake
    def _(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ):
        return

    # mm_mxfp8 is newer than the entry points above, so probe it separately
    # rather than breaking this module's import on an older flashinfer build.
    try:
        from flashinfer import mm_mxfp8
    except ImportError:
        mm_mxfp8 = None

    if mm_mxfp8 is not None:

        # Wrap this into a custom op so torch.compile traces one opaque node
        # instead of inlining flashinfer's Python-level tactic lookup.
        @torch.library.custom_op("trtllm::flashinfer_mm_mxfp8", mutates_args=())
        def flashinfer_mm_mxfp8(act: torch.Tensor, act_scale: torch.Tensor,
                                weight: torch.Tensor,
                                weight_scale: torch.Tensor,
                                output_dtype: torch.dtype) -> torch.Tensor:
            """Run CUTLASS MXFP8 GEMM with row-major weights and swizzled scales."""
            # Argument order mirrors trtllm::mxfp8_mxfp8_gemm: weight arrives as
            # [N, K] and mm_mxfp8 wants [K, N]. Both scale buffers are the 1D
            # padded swizzled CUTLASS layout, hence use_8x4_sf_layout=False.
            return mm_mxfp8(act,
                            weight.t(),
                            act_scale,
                            weight_scale,
                            out_dtype=output_dtype,
                            use_8x4_sf_layout=False,
                            backend="cutlass")

        @flashinfer_mm_mxfp8.register_fake
        def _(act: torch.Tensor, act_scale: torch.Tensor, weight: torch.Tensor,
              weight_scale: torch.Tensor,
              output_dtype: torch.dtype) -> torch.Tensor:
            """Infer the GEMM output shape and dtype without invoking FlashInfer."""
            return act.new_empty((act.size(0), weight.size(0)),
                                 dtype=output_dtype)
