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
        gemma_fused_add_rmsnorm(input,
                                residual,
                                weight,
                                eps,
                                enable_pdl=get_env_enable_pdl())

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

        # SM12x small-M dispatch to FlashInfer's b12x (CuTe DSL warp-MMA)
        # MXFP8 GEMM. Its 16-row M tile streams the weight with N/64 CTAs
        # instead of CUTLASS's 128-row tile with N/32..N/128 CTAs. Same
        # m16n8k32 UE8M0 MMA with sequential BK128 K order and no split-K, so
        # the output is bitwise equal to the CUTLASS backend. Cold-L2 CUDA
        # graph timings (RTX PRO 6000, DeepSeek-V4.1 TP4 shapes, M=1..32):
        # wq_a 14.8->10.4us, wkv 14.3->10.2, wq_b 9.8->8.8, o_b 10.2->8.8,
        # index_wq_b 6.6->5.4. M > 32 (prefill) stays on CUTLASS, which is
        # faster there.
        _B12X_MAX_M = 32
        _B12X_TILE_M = 16
        _B12X_PRECOMPILE_N = (64, 128)
        _b12x_runners: dict = {}
        _b12x_tile_ok: dict = {}

        def _b12x_tactic(tile_n: int):
            return ((_B12X_TILE_M, tile_n), (1, 1), False, False, "sm120",
                    None)

        def _b12x_cache_key(tile_n: int, output_dtype: torch.dtype):
            # Mirrors the cache_key built in B12xMxfp8GemmRunner.forward.
            return (32, (_B12X_TILE_M, tile_n), (1, 1), False, False, "sm120",
                    None, True, output_dtype)

        def _b12x_get_runner(device: torch.device, output_dtype: torch.dtype):
            """Return (runner, sm_count) or None when b12x is unusable here."""
            key = (device.index, output_dtype)
            if key in _b12x_runners:
                return _b12x_runners[key]
            if torch.cuda.is_current_stream_capturing():
                # Creating the runner compiles kernels; never do that inside
                # graph capture. The caller falls back to CUTLASS.
                return None
            entry = None
            try:
                major, minor = torch.cuda.get_device_capability(device)
                if (major == 12 and output_dtype
                        in (torch.bfloat16, torch.float16)):
                    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import (
                        _b12x_gemm_mxfp8_runner, _b12x_mxfp8_dsl_supported)
                    from flashinfer.jit.cpp_ext import get_cuda_version
                    if (get_cuda_version().major >= 13
                            and _b12x_mxfp8_dsl_supported()):
                        runner = _b12x_gemm_mxfp8_runner(
                            major, minor, True, output_dtype)
                        sm_count = torch.cuda.get_device_properties(
                            device).multi_processor_count
                        # Compile every tile this dispatch can pick now, in
                        # eager mode, so graph capture never JIT-compiles.
                        k = 128
                        a = torch.zeros((1, k),
                                        dtype=torch.float8_e4m3fn,
                                        device=device)
                        w = torch.zeros((128, k),
                                        dtype=torch.float8_e4m3fn,
                                        device=device)
                        a_sf = torch.full((128 * 4, ),
                                          127,
                                          dtype=torch.uint8,
                                          device=device)
                        w_sf = torch.full((128 * 4, ),
                                          127,
                                          dtype=torch.uint8,
                                          device=device)
                        out = torch.empty((1, 128),
                                          dtype=output_dtype,
                                          device=device)
                        for tile_n in _B12X_PRECOMPILE_N:
                            runner.forward(
                                [a, w.t(), a_sf, w_sf, output_dtype, out, None],
                                tactic=_b12x_tactic(tile_n))
                        torch.cuda.current_stream(device).synchronize()
                        entry = (runner, sm_count)
            except Exception as error:  # pragma: no cover - optional path
                from tensorrt_llm.logger import logger
                logger.warning(
                    f"FlashInfer b12x MXFP8 GEMM unavailable ({error!r}); "
                    "using the CUTLASS backend.")
                entry = None
            _b12x_runners[key] = entry
            return entry

        def _b12x_try(act: torch.Tensor, act_scale: torch.Tensor,
                      weight: torch.Tensor, weight_scale: torch.Tensor,
                      output_dtype: torch.dtype):
            m, k = act.shape
            n = weight.shape[0]
            if (m == 0 or m > _B12X_MAX_M or k % 128 != 0
                    or act_scale.ndim != 1 or weight_scale.ndim != 1):
                return None
            entry = _b12x_get_runner(act.device, output_dtype)
            if entry is None:
                return None
            runner, sm_count = entry
            # 64-wide N tiles while one wave covers N; 128 beyond that.
            tile_n = 64 if (n + 63) // 64 <= sm_count else 128
            ok_key = (n, k, tile_n, output_dtype)
            ok = _b12x_tile_ok.get(ok_key)
            if ok is None:
                import cutlass
                from flashinfer.cute_dsl.utils import torch_to_cutlass_dtype
                from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import \
                    Sm120B12xBlockScaledDenseGemmKernel
                ok = bool(
                    Sm120B12xBlockScaledDenseGemmKernel.can_implement(
                        cutlass.Float8E4M3FN,
                        cutlass.Float8E8M0FNU,
                        32,
                        torch_to_cutlass_dtype(output_dtype),
                        (_B12X_TILE_M, tile_n),
                        (1, 1),
                        n,
                        k,
                        1,
                        "k",
                        "k",
                        "n",
                        swap_ab=False,
                    ))
                _b12x_tile_ok[ok_key] = ok
            if not ok:
                return None
            if torch.cuda.is_current_stream_capturing():
                from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import \
                    _B12X_MM_MXFP8_KERNEL_CACHE
                if ((act.device.index, _b12x_cache_key(tile_n, output_dtype))
                        not in _B12X_MM_MXFP8_KERNEL_CACHE):
                    return None
            out = torch.empty((m, n), dtype=output_dtype, device=act.device)
            runner.forward(
                [act, weight.t(), act_scale, weight_scale, output_dtype, out,
                 None],
                tactic=_b12x_tactic(tile_n))
            return out

        # M > 32 (eager prefill) host trim of the CUTLASS path. Stock
        # mm_mxfp8 pays, per call, the @backend_requirement checks, a new
        # TunableRunner subclass, the runner-factory dict and AutoTuner
        # .choose_one. Their result only depends on the shapes/dtypes and on
        # the AutoTuner's state, so the first call per exact shape runs the
        # stock path and records choose_one's tactic; later calls launch the
        # same TVM-FFI mxfp8_gemm with that tactic and the same workspace.
        # The memo is bypassed while the AutoTuner can choose per call
        # (tuning mode, skip_ops/bucket overrides, autotune_v2 contexts) and
        # is dropped whenever its caches change.
        _CUTLASS_MEMO_MIN_M = _B12X_MAX_M + 1
        _CUTLASS_MEMO_MAX_ENTRIES = 4096
        _cutlass_memo: dict = {}
        _cutlass_memo_state: list = [None, None]  # [tuner signature, deps]

        def _cutlass_memo_deps():
            """FlashInfer internals the memo needs; False when unavailable."""
            deps = _cutlass_memo_state[1]
            if deps is None:
                try:
                    from flashinfer.autotuner import AutoTuner
                    from flashinfer.gemm.gemm_base import (
                        _MM_MXFP8_TUNING_CONFIG, DEFAULT_WORKSPACE_SIZE,
                        _load_gemm_sm120_mxfp8_module,
                        get_cutlass_mxfp8_gemm_module)
                    from flashinfer.utils import _get_cache_buf
                    tuner = AutoTuner.get()
                    # Probe every attribute the signature reads.
                    (tuner.is_tuning_mode, tuner._in_v2_context,
                     tuner._skip_ops_local, tuner._override_local,
                     tuner._dirty_seq, tuner.profiling_cache,
                     tuner._file_configs, tuner._managed_cache,
                     tuner._winner_partitions, tuner._managed_decoded)
                    deps = (AutoTuner, _MM_MXFP8_TUNING_CONFIG,
                            DEFAULT_WORKSPACE_SIZE,
                            _load_gemm_sm120_mxfp8_module,
                            get_cutlass_mxfp8_gemm_module, _get_cache_buf)
                except Exception:  # pragma: no cover - other flashinfer
                    deps = False
                _cutlass_memo_state[1] = deps
            return deps

        def _cutlass_tuner_signature(tuner):
            """The AutoTuner state choose_one depends on, or None when it may
            pick per call (then the stock path runs)."""
            if tuner.is_tuning_mode or tuner._in_v2_context:
                return None
            if getattr(tuner._skip_ops_local, "stack", None):
                return None
            if getattr(tuner._override_local, "stack", None):
                return None
            return (tuner._dirty_seq, len(tuner.profiling_cache),
                    len(tuner._file_configs), id(tuner._managed_cache),
                    len(tuner._winner_partitions), len(tuner._managed_decoded))

        def _cutlass_memo_mm(act: torch.Tensor, act_scale: torch.Tensor,
                             weight: torch.Tensor, weight_scale: torch.Tensor,
                             output_dtype: torch.dtype):
            """Memoized CUTLASS mm_mxfp8 for M > 32; None -> run stock."""
            if (act.ndim != 2 or weight.ndim != 2 or act_scale.ndim != 1
                    or weight_scale.ndim != 1):
                return None
            m, k = act.shape
            if m < _CUTLASS_MEMO_MIN_M:
                return None
            if not (act.is_contiguous() and weight.is_contiguous()
                    and act_scale.is_contiguous()
                    and weight_scale.is_contiguous()):
                return None
            deps = _cutlass_memo_deps()
            if not deps:
                return None
            (auto_tuner, tuning_config, workspace_size, load_module,
             get_module, get_cache_buf) = deps
            tuner = auto_tuner.get()
            signature = _cutlass_tuner_signature(tuner)
            if signature is None:
                return None
            if signature != _cutlass_memo_state[0]:
                _cutlass_memo.clear()
                _cutlass_memo_state[0] = signature
            device = act.device
            n = weight.shape[0]
            key = (device, m, k, n, weight.shape[1], act_scale.shape[0],
                   weight_scale.shape[0], act.dtype, weight.dtype,
                   act_scale.dtype, weight_scale.dtype, output_dtype)
            entry = _cutlass_memo.get(key)
            if entry is None:
                # First call for this shape: the stock path validates the
                # inputs and computes the output; then record the tactic
                # choose_one picks for exactly the stock inputs.
                out = mm_mxfp8(act,
                               weight.t(),
                               act_scale,
                               weight_scale,
                               out_dtype=output_dtype,
                               use_8x4_sf_layout=False,
                               backend="cutlass")
                major, _ = torch.cuda.get_device_capability(device)
                if major == 12:
                    workspace = get_cache_buf("mm_mxfp8_workspace",
                                              workspace_size, device)
                    runner = get_module(major).cutlass_mxfp8_gemm_runner()
                    inputs = [
                        act,
                        weight.t(), act_scale, weight_scale, output_dtype,
                        out, workspace
                    ]
                    _, tactic = tuner.choose_one(
                        custom_op="mxfp8_gemm",
                        runners=[runner],
                        tuning_config=tuning_config,
                        inputs=inputs,
                    )
                    entry = (load_module(), tactic)
                else:
                    entry = (None, None)
                if len(_cutlass_memo) >= _CUTLASS_MEMO_MAX_ENTRIES:
                    _cutlass_memo.clear()
                # The lookups above may promote cache entries; re-read the
                # signature so the next call does not drop this entry.
                _cutlass_memo_state[0] = _cutlass_tuner_signature(tuner)
                _cutlass_memo[key] = entry
                return out
            module, tactic = entry
            if module is None:
                return None
            out = torch.empty((m, n), dtype=output_dtype, device=device)
            workspace = get_cache_buf("mm_mxfp8_workspace", workspace_size,
                                      device)
            # Same call as CutlassMxfp8GemmRunner.forward: b.T of weight.t()
            # is weight, and the swizzled 1D scales pass through.
            module.mxfp8_gemm(act, weight, act_scale, weight_scale, out,
                              workspace, tactic)
            return out

        # Wrap this into a custom op so torch.compile traces one opaque node
        # instead of inlining flashinfer's Python-level tactic lookup.
        @torch.library.custom_op("trtllm::flashinfer_mm_mxfp8", mutates_args=())
        def flashinfer_mm_mxfp8(act: torch.Tensor, act_scale: torch.Tensor,
                                weight: torch.Tensor,
                                weight_scale: torch.Tensor,
                                output_dtype: torch.dtype) -> torch.Tensor:
            """Run CUTLASS MXFP8 GEMM with row-major weights and swizzled scales."""
            # SM12x decode sizes: b12x small-M tiles (bitwise equal output).
            out = _b12x_try(act, act_scale, weight, weight_scale, output_dtype)
            if out is not None:
                return out
            # Prefill sizes: memoized CUTLASS dispatch (same tactic, same
            # kernel launch as the stock path below).
            out = _cutlass_memo_mm(act, act_scale, weight, weight_scale,
                                   output_dtype)
            if out is not None:
                return out
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
