# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from packaging.version import Version

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...utils import (
    ActivationType,
    Fp4QuantizedTensor,
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from .activation import MoEActivationSupport
from .fused_moe_cutlass import CutlassFusedMoE
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    require_comm_plan,
)
from .impl_environment import MoEDep
from .interface import _reject

# Shared MoE output buffer pool, keyed by (max_num_tokens, hidden_size, dtype,
# device). ``B12xMoEWrapper.__init__`` allocates a private
# ``(max_num_tokens, hidden_size)`` output tensor per instance; with one
# wrapper per MoE layer that is ``num_layers * max_num_tokens * hidden_size``
# bytes of GPU memory holding identical-shape buffers that are written
# sequentially. We fold them into a single shared buffer because MoE layers
# run sequentially on the same CUDA stream, and the wrapper consumes its
# previous output before the next layer is dispatched.
_SHARED_MOE_OUTPUT_BUF: dict = {}

# Shared wrapper scratch: the (static, dynamic) workspaces ``B12xMoEWrapper``
# preallocates for CUDA graphs, keyed by the wrapper's construction arguments
# and device. They hold only per-call state (routed-row bookkeeping, packed
# activations, barriers), and MoE layers run one at a time on one stream, so a
# single set serves every layer with the same configuration.
_SHARED_B12X_WORKSPACES: dict[tuple, tuple] = {}

# ActivationType -> b12x activation string. b12x currently exposes "relu2"
# (Nemotron-style x * relu(x)) and "silu" (SwiGLU-style x * silu(gate)).
_ACTIVATION_MAP = {
    ActivationType.Relu2: "relu2",
    ActivationType.Swiglu: "silu",
}

# FlashInfer's SM12x W4A16 fused MoE (0.6.17 and 0.6.18) selects a TC-decode
# "ultra-wide" FC2 tile of tile_n=512 / tile_k=32 for small m whenever that
# collapses FC2 to FC1's wave count (m=3 and m=4 for Qwen3.6-35B-A3B on the
# 188-SM RTX PRO 6000). tile_k=32 sits below the tile_k>=64 floor of its
# generic ``_candidate_tile_fits`` check, which the auto-selection skips but
# the ``force_tile_config`` re-pin across the custom-op boundary does not, so
# the kernel it just selected is rejected with "force_tile_config fc2 tile
# (tile_k=32, tile_n=512) does not fit ..." and CUDA-graph warmup aborts
# executor init (nvbug 6721561). Accepting the ultra tile in the re-validation
# under the same rule that produced it is the upstream fix; this shim mirrors
# it so the pinned wheel behaves like the fixed one. Drop it once the
# flashinfer pin carries ``_ultra_wide_fc2_tile_fits``.
_FLASHINFER_W4A16_ULTRA_WIDE_FC2_TILE = (512, 32, 256)  # (tile_n, tile_k, cta_threads)
_FLASHINFER_W4A16_ULTRA_WIDE_FC2_SINCE = Version("0.6.17")


def _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() -> bool:
    """Make flashinfer's W4A16 ``force_tile_config`` re-validation accept the
    ultra-wide FC2 tile its own auto-selection produces.

    Idempotent. Returns True when the shim is active, False when flashinfer is
    absent, predates the ultra-wide override, or already ships the fix.
    """
    try:
        import flashinfer
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel as kernel
    except ImportError:
        return False
    if Version(flashinfer.__version__) < _FLASHINFER_W4A16_ULTRA_WIDE_FC2_SINCE:
        return False
    if hasattr(kernel, "_ultra_wide_fc2_tile_fits"):
        return False
    upstream_fits = getattr(kernel, "_candidate_tile_fits", None)
    smem_footprint = getattr(kernel, "_shared_memory_footprint", None)
    scale_group_size = getattr(kernel, "_scale_group_size", None)
    if upstream_fits is None or smem_footprint is None or scale_group_size is None:
        return False
    if getattr(upstream_fits, "_trtllm_ultra_wide_fc2_shim", False):
        return True

    def candidate_tile_fits(
        *,
        problem_n: int,
        problem_k: int,
        cta_m_blocks: int,
        tile_n: int,
        tile_k: int,
        cta_threads: int,
        max_shared_mem: int,
        scale_format: str = "e4m3_k16",
        weight_layout: str = "packed",
        allow_logical_tail: bool = False,
    ) -> bool:
        if (int(tile_n), int(tile_k), int(cta_threads)) != _FLASHINFER_W4A16_ULTRA_WIDE_FC2_TILE:
            return upstream_fits(
                problem_n=problem_n,
                problem_k=problem_k,
                cta_m_blocks=cta_m_blocks,
                tile_n=tile_n,
                tile_k=tile_k,
                cta_threads=cta_threads,
                max_shared_mem=max_shared_mem,
                scale_format=scale_format,
                weight_layout=weight_layout,
                allow_logical_tail=allow_logical_tail,
            )
        # Same rule as the auto-selection override: exact N/K tiling, one scale
        # group per k-tile, and shared-memory fit.
        if int(problem_n) % int(tile_n) != 0 or int(problem_k) % int(tile_k) != 0:
            return False
        if int(tile_k) % int(scale_group_size(scale_format)) != 0:
            return False
        smem_bytes = smem_footprint(
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            scale_format=scale_format,
            weight_layout=weight_layout,
        )
        return int(smem_bytes) <= int(max_shared_mem)

    candidate_tile_fits._trtllm_ultra_wide_fc2_shim = True
    candidate_tile_fits._trtllm_upstream_fits = upstream_fits
    kernel._candidate_tile_fits = candidate_tile_fits
    logger.info_once(
        f"flashinfer {flashinfer.__version__}: accepting the SM12x W4A16 ultra-wide FC2 "
        "tile (tile_n=512, tile_k=32) in force_tile_config re-validation (nvbug 6721561).",
        key="flashinfer_w4a16_ultra_wide_fc2_tile_shim",
    )
    return True


def _routed_profile_inputs(
    num_experts: int, top_k: int, inputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Replace the tuner's synthesized inputs with realistic routing.

    The AutoTuner fills profiling inputs with zeros; b12x's cost depends on how
    many distinct experts the routing touches, so profile with ``top_k``
    distinct experts per token, normalized weights and non-zero activations.
    """
    x, token_selected_experts, token_final_scales = inputs
    scores, experts = torch.topk(
        torch.rand(x.shape[0], num_experts, device=x.device), top_k, dim=-1
    )
    return [
        torch.randn_like(x),
        experts.to(token_selected_experts.dtype),
        (scores / scores.sum(dim=-1, keepdim=True)).to(token_final_scales.dtype),
    ]


class _B12xRouteRunner(TunableRunner):
    """One NVFP4 MoE call through either of the backend's two routes.

    ``B12X`` runs FlashInfer's b12x kernels, which quantize the activations
    themselves; ``CUTLASS`` quantizes them to NVFP4 and runs the CUTLASS
    grouped GEMM. Which one is faster depends on the MoE shape and the token
    count, so the AutoTuner times both per power-of-two token bucket. b12x
    takes and returns bf16 only, so fp16 activations always run CUTLASS.
    """

    B12X = 0
    CUTLASS = 1

    tuning_config_cache: dict[tuple, TuningConfig] = {}

    def __init__(
        self,
        moe: "CuteDslB12xFusedMoE",
        ctx: MoERunContext,
        workspace: Optional[dict],
    ) -> None:
        super().__init__()
        self.moe = moe
        self.ctx = ctx
        self.workspace = workspace

    def unique_id(self) -> tuple:
        moe = self.moe
        return (
            moe.num_experts,
            moe.routing_method.top_k,
            moe.hidden_size,
            moe.intermediate_size_per_partition,
            moe.dtype,
            moe.activation_type,
        )

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list[int]:
        if inputs[0].dtype != torch.bfloat16:
            return [self.CUTLASS]
        return [self.B12X, self.CUTLASS]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in type(self).tuning_config_cache:

            def num_tokens(input_shapes: list[torch.Size]) -> int:
                return input_shapes[0][0]

            type(self).tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0,
                        0,
                        get_last_power_of_2_num_tokens_buckets,
                        last_positive_power_of_2,
                    ),
                ),
                constraint_specs=(
                    ConstraintSpec(1, 0, num_tokens),
                    ConstraintSpec(2, 0, num_tokens),
                ),
                inputs_pre_hook=functools.partial(
                    _routed_profile_inputs,
                    self.moe.num_experts,
                    self.moe.routing_method.top_k,
                ),
                use_cold_l2_cache=True,
            )
        return type(self).tuning_config_cache[key]

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        x, token_selected_experts, token_final_scales = inputs
        ctx = replace(
            self.ctx,
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
        )
        b12x_ok = x.dtype == torch.bfloat16
        if do_preparation:
            # b12x compiles its kernels and CUTLASS tunes its own tactics the
            # first time they see a shape; neither may happen inside the
            # CUDA graph that times the routes.
            if b12x_ok:
                self.moe._run_b12x(ctx)
            return self.moe._run_cutlass(ctx, self.workspace)
        if tactic == -1:  # no tuning result for this token count
            tactic = (
                self.B12X
                if b12x_ok and x.shape[0] < self.moe._UNTUNED_CUTLASS_MIN_TOKENS
                else self.CUTLASS
            )
        if tactic == self.CUTLASS:
            return self.moe._run_cutlass(ctx, self.workspace)
        return self.moe._run_b12x(ctx)


class CuteDslB12xFusedMoE(CutlassFusedMoE):
    """B12x NVFP4 fused-MoE backend for SM120 / SM121.

    NVFP4 calls run either FlashInfer's b12x kernels or CUTLASS NVFP4, chosen
    per token count by the AutoTuner (``_B12xRouteRunner``); W4A16_NVFP4 calls
    always run b12x.

    The only subclass of ``CutlassFusedMoE``, and the only backend for which
    that is a real dependency rather than a shortcut: the CUTLASS route
    (``_run_cutlass``) goes through ``CutlassFusedMoE.quantize_input`` /
    ``CutlassFusedMoE.run_moe``, which read the whole Cutlass execution state
    (chunking stream and events, ``use_fused_finalize``, the tuner flags, the
    LoRA slot helpers, ``_tuner_shapes``, ``_run_moe_w4a16_nvfp4``).

    ``CuteDslFusedMoE.run_moe_nvfp4*`` is never reached from here, so the
    ``AuxStreamType.MoeOutputMemset`` / ``EventType`` entries it needs are not
    set up and ``event_dict`` can be None. Restate them, and
    ``limit_when_absent`` below, before routing any CuteDSL path through this
    class.
    """

    # Inherited wholesale from CutlassFusedMoE, so every field is restated.
    # No code path here reads ``w3_w1_bias`` / ``w2_bias`` or fuses LoRA, and
    # ``supports_eplb`` stays False -- which is why ``can_implement`` has to
    # decline ``d.eplb_enabled`` explicitly, since the inherited
    # ``_supports_load_balancer()`` answers True.
    # ``supports_apply_router_weight_on_input`` is False where the parent says
    # True: only the NVFP4 prefill chunk reaches ``CutlassFusedMoE.run_moe``,
    # while the decode path hands ``token_final_scales`` straight to the
    # flashinfer b12x wrapper, which has no declared behaviour for the ``None``
    # the scheduler's fold leaves there.
    capabilities = MoEStaticCapability(
        supports_moe_lora=False,
        supports_dwdp=True,
        supports_expert_bias=False,
        supports_apply_router_weight_on_input=False,
    )

    # Same value the parent declares, pinned so a change there cannot silently
    # retarget this backend.
    input_requirement = MoEInputRequirement(routing_scales_dtype=torch.float32)

    # The kinds ``_ACTIVATION_MAP`` above gates on. The b12x decode kernel takes
    # no activation constants, so none are declared.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2})
    )

    # Read by ``ConfigurableMoE._reject_non_divisible_ep_backend()``; moot in
    # practice because ``can_implement`` rejects ``ep_size != 1`` outright.
    _supports_non_divisible_ep: bool = True

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        return self.has_nvfp4

    # SM versions on which the FlashInfer b12x NVFP4 MoE kernel is available.
    # SM120 = desktop Blackwell (RTX 5090 / GB202); SM121 = GB10 / DGX Spark.
    _SUPPORTED_SM_VERSIONS = frozenset({120, 121})

    # NVFP4 route when there is no tuning result (autotuner disabled, or a
    # token count the warmup did not tune): CUTLASS from this many tokens.
    _UNTUNED_CUTLASS_MIN_TOKENS = 64

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        sm_version = d.env.sm
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            sm_list = "/".join(f"SM{v}" for v in sorted(cls._SUPPORTED_SM_VERSIONS))
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"CuteDslB12xFusedMoE requires {sm_list}, got SM{sm_version}",
            )
        if p.quant_algo not in {QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4}:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"CuteDslB12xFusedMoE only supports NVFP4 or W4A16_NVFP4 quantization "
                f"(got quant_algo={p.quant_algo})",
            )
        if p.dtype_act not in {torch.float16, torch.bfloat16}:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CuteDslB12xFusedMoE NVFP4 requires float16 or bfloat16 "
                f"activation dtype (got {p.dtype_act})",
            )
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CuteDslB12xFusedMoE does not support swiglu_gptoss_style",
            )
        if p.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"CuteDslB12xFusedMoE does not support activation "
                f"{p.activation}; supported: {supported}",
            )
        # The decode kernel ships in the FlashInfer wheel.
        if not d.env.has_dep(MoEDep.FLASHINFER):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslB12xFusedMoE requires the flashinfer package",
            )
        # The only backend the construction-time allow-list turned down that
        # never said so during selection, so an EPLB run resolved to b12x and
        # then died in the factory. Declining here degrades with the usual
        # warning instead, matching VanillaMoE / TritonFusedMoE / MarlinFusedMoE.
        if d.eplb_enabled:
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED,
                "CuteDslB12xFusedMoE does not support the MoE load balancer",
            )
        # No expert-parallel dispatch/combine kernel: EP must stay at 1.
        if d.ep_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"CuteDslB12xFusedMoE requires ep_size == 1 (got {d.ep_size})",
            )
        # Attention-DP is a separate axis from EP: with moe_tp == tp the layer
        # can have ep_size == 1 and still sit behind a DP allgather /
        # reducescatter that the b12x wrapper has never been exercised under.
        # ``use_dp and parallel_size > 1`` is exactly ``mapping.dp_size > 1``
        # (``Mapping.dp_size`` is ``tp_size`` when attention-DP is on).
        if d.use_dp and d.parallel_size > 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"CuteDslB12xFusedMoE does not support attention-DP "
                f"(parallel_size={d.parallel_size})",
            )
        return MoEEligibility.ok()

    def __init__(self, *args, **kwargs):
        # ``ModelConfig`` is consumed by the inherited ``__init__`` for cache
        # / mapping setup but isn't kept on ``self``. b12x's wrapper needs the
        # ``use_cuda_graph`` flag at construction time, so capture it here
        # before delegating.
        model_config = kwargs.get("model_config", None)
        self._b12x_use_cuda_graph = bool(getattr(model_config, "use_cuda_graph", False))

        super().__init__(*args, **kwargs)

        # Only the W4A16 decode path reaches flashinfer's W4A16 tile selector.
        if self.quant_config is not None and self.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4:
            _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation()

        # No alltoall guard here: alltoall is picked by the wrapper's
        # communication strategy, and ``can_implement`` already rejects the
        # topologies that could pick it (ep_size != 1, attention-DP with
        # parallel_size > 1).
        self._b12x_weights: Optional[dict] = None
        self.b12x_wrapper = None

    def _get_quant_method(self):
        # Route NVFP4 to the b12x-aware quant method so weight prep
        # (SF un-normalization, ``convert_sf_to_mma_layout``,
        # ``B12xMoEWrapper`` instantiation) lives next to the rest of the
        # NVFP4 quant-method family, while every other quant algo (and the
        # unquantized fallback) continues to resolve via the parent.
        if (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
            and self.quant_config.layer_quant_mode.has_nvfp4()
        ):
            from .quantization import NVFP4CuteDslB12xFusedMoEMethod

            return NVFP4CuteDslB12xFusedMoEMethod()
        return super()._get_quant_method()

    # ``post_load_weights`` is inherited from ``CutlassFusedMoE`` and
    # dispatches to ``self.quant_method.transform_weights(self)`` — for this
    # backend ``self.quant_method`` is ``NVFP4CuteDslB12xFusedMoEMethod``
    # (see ``_get_quant_method`` override), which performs the SF un-normalization,
    # ``convert_sf_to_mma_layout`` reshape, ``B12xMoEWrapper`` instantiation,
    # and the cross-layer shared output buffer dance. The wrapper and the
    # bundled weight dict are attached to this module as ``self.b12x_wrapper``
    # / ``self._b12x_weights``, which ``_run_b12x`` consumes.

    @nvtx_range("[b12x] quantize_input")
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pass the bf16 / fp16 activations through unquantized.

        b12x quantizes them inside its kernels, and the CUTLASS route
        quantizes them in ``_run_cutlass`` once ``run_moe`` has picked the
        route for this call.
        """
        if isinstance(x, Fp4QuantizedTensor):
            raise ValueError(
                "CuteDslB12xFusedMoE does not accept Fp4QuantizedTensor input; "
                "it quantizes activations itself."
            )
        return x, None

    @nvtx_range("[b12x] run_moe")
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        # Check the plan before the tuner runs: it treats exceptions raised
        # while profiling as failed tactics.
        require_comm_plan(self, ctx)
        # Empty chunks skip the tuner: its profiling inputs need a token.
        if self.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4 or ctx.x.shape[0] == 0:
            return self._run_b12x(ctx)
        runner = _B12xRouteRunner(self, ctx, workspace)
        inputs = [ctx.x, ctx.token_selected_experts, ctx.token_final_scales]
        _, tactic = AutoTuner.get().choose_one(
            "CuteDslB12xFusedMoE::run_moe_nvfp4",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=tactic)

    def _run_cutlass(self, ctx: MoERunContext, workspace: Optional[dict]) -> torch.Tensor:
        """Quantize the activations to NVFP4 and run the CUTLASS grouped GEMM."""
        plan = require_comm_plan(self, ctx)
        x, x_sf = CutlassFusedMoE.quantize_input(
            self, ctx.x, post_quant_comm=not plan.input_sf_swizzled
        )
        # ``CutlassFusedMoE.run_moe`` forwards ``output_dtype`` straight into
        # the C++ ``trtllm::fused_moe`` op, which requires a concrete
        # high-precision ``ScalarType``. ``ConfigurableMoE.forward`` always
        # fills it; default to the activation dtype for direct callers.
        output_dtype = ctx.output_dtype if ctx.output_dtype is not None else ctx.x.dtype
        return CutlassFusedMoE.run_moe(
            self,
            replace(ctx, x=x, x_sf=x_sf, output_dtype=output_dtype),
            workspace=workspace,
        )

    def _run_b12x(self, ctx: MoERunContext) -> torch.Tensor:
        """Run FlashInfer's b12x kernels on the unquantized activations."""
        plan = require_comm_plan(self, ctx)
        if self.b12x_wrapper is None or self._b12x_weights is None:
            raise RuntimeError(
                "CuteDslB12xFusedMoE.run_moe called before process_weights_after_loading completed."
            )
        if ctx.x_sf is not None:
            raise ValueError(
                "CuteDslB12xFusedMoE expects unquantized input (x_sf=None); "
                "got a precomputed scale factor."
            )

        # Annotate the kwargs spread + wrapper entry separately so we can
        # attribute the per-layer Python dispatch cost vs. the kernel cost.
        with nvtx_range("[b12x] wrapper.run"):
            out = self.b12x_wrapper.run(
                x=ctx.x,
                token_selected_experts=ctx.token_selected_experts,
                token_final_scales=ctx.token_final_scales,
                **self._b12x_weights,
            )

        # B12xMoEWrapper allocates its own output buffer for CUDA-graph
        # compatibility. If the caller provided ``moe_output`` (e.g. an
        # alltoall workspace tensor), copy into it; ``can_implement`` rejects
        # the topologies that could pick alltoall, so this is a defensive
        # path for future workspace-driven uses.
        if plan.moe_output is not None:
            with nvtx_range("[b12x] out_copy"):
                plan.moe_output.copy_(out)
            return plan.moe_output
        return out
