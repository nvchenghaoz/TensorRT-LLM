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
"""CuteDslB12xFusedMoE gating and dispatch tests."""

import sys
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.moe.fused_moe import fused_moe_cute_dsl_b12x
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import (
    CuteDslB12xFusedMoE,
    _B12xRouteRunner,
    _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoECommPlan,
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    canonical_quant,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CuteDslB12xFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


# Spelled out rather than read from _SUPPORTED_SM_VERSIONS: deriving the input
# from the value under test hides a narrowing of that set, which is the
# direction that silently drops hardware support.
SUPPORTED_SM = [120, 121]


def _deployment(
    sm: int,
    *,
    flashinfer: bool = True,
    ep_size: int = 1,
    use_dp: bool = False,
    parallel_size: Optional[int] = None,
    eplb: bool = False,
) -> MoEDeployment:
    """Declare the machine rather than patching the probes that read it."""
    return MoEDeployment(
        ep_size=ep_size,
        tp_size=1,
        parallel_size=ep_size if parallel_size is None else parallel_size,
        use_dp=use_dp,
        num_slots=8,
        eplb_enabled=eplb,
        env=MoEEnvironment(
            sm=sm,
            available_deps=(MoEDep.FLASHINFER.value,) if flashinfer else (),
        ),
    )


def _problem(
    quant_algo=QuantAlgo.NVFP4, dtype=torch.bfloat16, swiglu_gptoss_style=None
) -> MoEProblem:
    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=dtype,
        hidden_size=2048,
        intermediate_size=2048,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=swiglu_gptoss_style,
    )


@pytest.mark.parametrize("sm_version", [80, 89, 90, 100, 103])
def test_can_implement_rejects_unsupported_sm(sm_version):
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(sm_version))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.SM_UNSUPPORTED
    assert f"SM{sm_version}" in verdict.detail


@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4])
def test_can_implement_accepts_supported_sm(sm_version, quant_algo):
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(quant_algo), _deployment(sm_version))
    assert verdict.eligible
    assert verdict.reject_reason is None


@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.FP8,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
    ],
)
def test_can_implement_rejects_non_nvfp4(quant_algo):
    """Only NVFP4 is supported; everything else must be turned away."""
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(quant_algo), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.QUANT_UNSUPPORTED


def test_can_implement_rejects_swiglu_gptoss_style():
    verdict = CuteDslB12xFusedMoE.can_implement(
        _problem(swiglu_gptoss_style=True), _deployment(120)
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_can_implement_rejects_unsupported_activation_dtype(dtype):
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(dtype=dtype), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DTYPE_UNSUPPORTED


def test_can_implement_rejects_missing_flashinfer():
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120, flashinfer=False))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DEP_MISSING


def test_can_implement_rejects_eplb():
    """EPLB is its own reject class, not a topology one: the machine is fine."""
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120, eplb=True))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.EPLB_UNSUPPORTED


def test_can_implement_rejects_expert_parallelism():
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120, ep_size=2))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED


def test_can_implement_rejects_attention_dp_without_expert_parallelism():
    """moe_tp == tp leaves ep_size at 1, so the EP gate alone would let this in."""
    verdict = CuteDslB12xFusedMoE.can_implement(
        _problem(), _deployment(120, ep_size=1, use_dp=True, parallel_size=2)
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED
    assert "attention-DP" in verdict.detail


# --------------------------------------------------------------------------
# NVFP4 route selection: b12x vs CUTLASS, chosen per token bucket by the
# AutoTuner through ``_B12xRouteRunner``. The runner is driven on a stub
# module so the route plumbing is checked without the full CutlassFusedMoE
# constructor (routing method, model config, weights).
# --------------------------------------------------------------------------


class _RouteStub:
    """Module stand-in that records which route ran."""

    _UNTUNED_CUTLASS_MIN_TOKENS = CuteDslB12xFusedMoE._UNTUNED_CUTLASS_MIN_TOKENS
    num_experts = 16
    hidden_size = 64
    intermediate_size_per_partition = 32
    dtype = torch.bfloat16
    activation_type = ActivationType.Swiglu
    routing_method = types.SimpleNamespace(top_k=4)

    def __init__(self):
        self.calls = []

    def _run_b12x(self, ctx):
        self.calls.append(("b12x", ctx.x.shape[0]))
        return "b12x"

    def _run_cutlass(self, ctx, workspace):
        self.calls.append(("cutlass", ctx.x.shape[0]))
        return "cutlass"


def _route_inputs(num_tokens: int, dtype: torch.dtype = torch.bfloat16):
    return [
        torch.zeros(num_tokens, _RouteStub.hidden_size, dtype=dtype),
        torch.zeros(num_tokens, _RouteStub.routing_method.top_k, dtype=torch.int32),
        torch.zeros(num_tokens, _RouteStub.routing_method.top_k, dtype=torch.float32),
    ]


def _route_runner(stub: _RouteStub, num_tokens: int) -> _B12xRouteRunner:
    x, experts, scales = _route_inputs(num_tokens)
    ctx = MoERunContext(token_selected_experts=experts, token_final_scales=scales, x=x, x_sf=None)
    return _B12xRouteRunner(stub, ctx, workspace=None)


@pytest.mark.parametrize(
    "num_tokens, route",
    [
        (1, "b12x"),
        (CuteDslB12xFusedMoE._UNTUNED_CUTLASS_MIN_TOKENS - 1, "b12x"),
        (CuteDslB12xFusedMoE._UNTUNED_CUTLASS_MIN_TOKENS, "cutlass"),
    ],
)
def test_route_without_tuning_result_uses_size_rule(num_tokens, route):
    stub = _RouteStub()
    runner = _route_runner(stub, num_tokens)
    assert runner(_route_inputs(num_tokens), tactic=-1) == route


@pytest.mark.parametrize(
    "tactic, route",
    [(_B12xRouteRunner.B12X, "b12x"), (_B12xRouteRunner.CUTLASS, "cutlass")],
)
@pytest.mark.parametrize("num_tokens", [1, 4096])
def test_route_tactic_overrides_size_rule(tactic, route, num_tokens):
    stub = _RouteStub()
    runner = _route_runner(stub, num_tokens)
    assert runner(_route_inputs(num_tokens), tactic=tactic) == route
    assert stub.calls == [(route, num_tokens)]


def test_route_runner_offers_both_routes():
    runner = _route_runner(_RouteStub(), 8)
    assert runner.get_valid_tactics(_route_inputs(8), profile=None) == [
        _B12xRouteRunner.B12X,
        _B12xRouteRunner.CUTLASS,
    ]


def test_route_fp16_activations_only_run_cutlass():
    """b12x's NVFP4 kernels are bf16-only, tuned or not."""
    stub = _RouteStub()
    runner = _route_runner(stub, 1)
    fp16_inputs = _route_inputs(1, torch.float16)
    assert runner.get_valid_tactics(fp16_inputs, profile=None) == [_B12xRouteRunner.CUTLASS]
    assert runner(fp16_inputs, tactic=-1) == "cutlass"
    runner(fp16_inputs, tactic=-1, do_preparation=True)
    assert stub.calls == [("cutlass", 1), ("cutlass", 1)]


def test_route_preparation_runs_both_routes_on_the_profiled_shape():
    stub = _RouteStub()
    runner = _route_runner(stub, 1)
    runner(_route_inputs(32), tactic=-1, do_preparation=True)
    assert stub.calls == [("b12x", 32), ("cutlass", 32)]


def test_route_tuning_config_keys_on_power_of_two_token_buckets():
    config = _route_runner(_RouteStub(), 8).get_tuning_config()
    (spec,) = config.dynamic_tensor_specs
    assert (spec.input_idx, spec.dim_idx) == (0, 0)
    assert sorted(spec.gen_tuning_buckets(4096)) == [2**i for i in range(13)]
    assert spec.map_to_tuning_buckets(96) == 64
    shapes = [torch.Size([96, 64]), torch.Size([1, 4]), torch.Size([1, 4])]
    assert [c.infer_shape(shapes) for c in config.constraint_specs] == [96, 96]


def test_route_profile_inputs_look_like_routing():
    x, experts, scales = (
        _route_runner(_RouteStub(), 8).get_tuning_config().inputs_pre_hook(_route_inputs(128))
    )
    top_k = _RouteStub.routing_method.top_k
    assert x.abs().sum() > 0
    assert experts.dtype == torch.int32 and experts.shape == (128, top_k)
    assert int(experts.min()) >= 0 and int(experts.max()) < _RouteStub.num_experts
    assert all(len(set(row.tolist())) == top_k for row in experts)
    assert scales.dtype == torch.float32
    torch.testing.assert_close(scales.sum(dim=-1), torch.ones(128))


def test_nvfp4_quantize_input_leaves_activations_unquantized():
    moe = object.__new__(CuteDslB12xFusedMoE)
    moe.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    x = torch.empty(4096, 1024)

    with patch.object(
        CutlassFusedMoE,
        "quantize_input",
        side_effect=AssertionError("the CUTLASS route quantizes in run_moe"),
    ):
        out, out_sf = CuteDslB12xFusedMoE.quantize_input(moe, x)

    assert out is x
    assert out_sf is None


def test_quantize_input_rejects_prequantized_input():
    moe = object.__new__(CuteDslB12xFusedMoE)
    moe.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    fp4 = Fp4QuantizedTensor(torch.empty(1, 512, dtype=torch.uint8), torch.empty(1, 64))
    with pytest.raises(ValueError, match="Fp4QuantizedTensor"):
        CuteDslB12xFusedMoE.quantize_input(moe, fp4)


@pytest.mark.parametrize(
    "quant_algo, num_tokens", [(QuantAlgo.W4A16_NVFP4, 4096), (QuantAlgo.NVFP4, 0)]
)
def test_run_moe_skips_route_tuning(quant_algo, num_tokens):
    """W4A16 has no CUTLASS route, and empty chunks have nothing to profile."""
    moe = object.__new__(CuteDslB12xFusedMoE)
    moe.quant_config = QuantConfig(quant_algo=quant_algo)
    x, experts, scales = _route_inputs(num_tokens)
    ctx = MoERunContext(
        token_selected_experts=experts,
        token_final_scales=scales,
        x=x,
        x_sf=None,
        comm_plan=MoECommPlan(
            input_sf_swizzled=True,
            enable_alltoall=False,
            moe_output=None,
            payload_in_workspace=False,
        ),
    )
    with (
        patch.object(CuteDslB12xFusedMoE, "_run_b12x", return_value="b12x") as run_b12x,
        patch.object(
            AutoTuner, "choose_one", side_effect=AssertionError("route must not be tuned")
        ),
    ):
        assert CuteDslB12xFusedMoE.run_moe(moe, ctx) == "b12x"
    run_b12x.assert_called_once_with(ctx)


def test_w4a16_nvfp4_prefill_quantize_input_stays_on_b12x():
    moe = object.__new__(CuteDslB12xFusedMoE)
    moe.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    x = torch.empty(CuteDslB12xFusedMoE._UNTUNED_CUTLASS_MIN_TOKENS, 1024)

    with patch.object(
        CutlassFusedMoE,
        "quantize_input",
        side_effect=AssertionError("W4A16_NVFP4 prefill must not route through CUTLASS"),
    ):
        out, out_sf = CuteDslB12xFusedMoE.quantize_input(moe, x)

    assert out is x
    assert out_sf is None


class _FakeB12xWrapper:
    """Records construction arguments; stands in for flashinfer.B12xMoEWrapper."""

    calls = []

    def __init__(self, **kwargs):
        self._moe_output = None
        self._static_workspace = object()
        self._dynamic_workspace = object()
        self.device = "cuda:0"
        self.calls.append(kwargs)


def _install_fake_flashinfer(monkeypatch):
    def _convert_sf_to_mma_layout(scales, *, m, k, num_groups):
        return scales

    _FakeB12xWrapper.calls = []
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.B12xMoEWrapper = _FakeB12xWrapper
    cute_dsl = types.ModuleType("flashinfer.cute_dsl")
    utils = types.ModuleType("flashinfer.cute_dsl.utils")
    utils.convert_sf_to_mma_layout = _convert_sf_to_mma_layout
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl", cute_dsl)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl.utils", utils)
    # Start every test from an empty cross-layer pool.
    monkeypatch.setattr(fused_moe_cute_dsl_b12x, "_SHARED_B12X_WORKSPACES", {})


def _nvfp4_b12x_module(
    *,
    quant_algo: QuantAlgo = QuantAlgo.W4A16_NVFP4,
    num_experts: int = 2,
    hidden_size: int = 128,
    logical_intermediate_size: int = 1856,
    padded_intermediate_size: int = 1920,
    max_num_tokens: int = 8,
) -> torch.nn.Module:
    """An NVFP4 (W4A4) or W4A16_NVFP4 MoE module with the attributes the b12x weight prep reads."""

    class _RoutingMethod:
        experts_per_token = 4

    module = torch.nn.Module()
    module.num_experts = num_experts
    module.hidden_size = hidden_size
    module.intermediate_size_per_partition = logical_intermediate_size
    module.moe_max_num_tokens = max_num_tokens
    module.routing_method = _RoutingMethod()
    module.activation_type = ActivationType.Swiglu
    module.quant_config = QuantConfig(quant_algo=quant_algo)
    module.w3_w1_weight = torch.empty(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.int64,
    )
    module.w2_weight = torch.empty(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.int64,
    )
    module.w3_w1_weight_scale = torch.ones(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    module.w2_weight_scale = torch.ones(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    module.fc31_alpha = torch.tensor([0.25, 0.5], dtype=torch.float32)
    module.fc2_alpha = torch.tensor([0.125, 0.25], dtype=torch.float32)
    module.fc31_input_scale = torch.tensor(2.0, dtype=torch.float32)
    module.fc2_input_scale = torch.tensor(4.0, dtype=torch.float32)
    return module


def _transform(module: torch.nn.Module) -> None:
    with patch.object(NVFP4CutlassFusedMoEMethod, "transform_weights", return_value=None):
        NVFP4CuteDslB12xFusedMoEMethod().transform_weights(module)


def test_w4a16_nvfp4_post_load_uses_modelopt_scale_contract(monkeypatch):
    _install_fake_flashinfer(monkeypatch)
    padded_intermediate_size = 1920
    module = _nvfp4_b12x_module(padded_intermediate_size=padded_intermediate_size)
    w3_w1_weight_scale = module.w3_w1_weight_scale.clone()
    w2_weight_scale = module.w2_weight_scale.clone()

    _transform(module)

    assert _FakeB12xWrapper.calls
    wrapper_kwargs = _FakeB12xWrapper.calls[0]
    assert wrapper_kwargs.get("quant_mode") == "w4a16", wrapper_kwargs
    assert wrapper_kwargs["intermediate_size"] == padded_intermediate_size
    assert module._b12x_weights["fc2_input_scale"] is None
    assert torch.equal(
        module._b12x_weights["w1_weight_sf"].float(),
        w3_w1_weight_scale.float(),
    )
    assert torch.equal(
        module._b12x_weights["w2_weight_sf"].float(),
        w2_weight_scale.float(),
    )
    assert torch.allclose(
        module._b12x_weights["w1_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )
    assert torch.allclose(
        module._b12x_weights["w2_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )


def test_b12x_layers_share_scratch_workspaces(monkeypatch):
    """MoE layers with the same wrapper configuration share one set of scratch."""
    _install_fake_flashinfer(monkeypatch)
    layers = [_nvfp4_b12x_module() for _ in range(3)]
    other = _nvfp4_b12x_module(max_num_tokens=16)
    for module in layers + [other]:
        _transform(module)

    first = layers[0].b12x_wrapper
    for module in layers[1:]:
        assert module.b12x_wrapper._static_workspace is first._static_workspace
        assert module.b12x_wrapper._dynamic_workspace is first._dynamic_workspace
    # A different max_num_tokens sizes the scratch differently, so it keeps its own.
    assert other.b12x_wrapper._static_workspace is not first._static_workspace
    assert other.b12x_wrapper._dynamic_workspace is not first._dynamic_workspace


# ---------------------------------------------------------------------------
# flashinfer W4A16 ultra-wide FC2 tile re-validation shim (nvbug 6721561)
# ---------------------------------------------------------------------------

# FC2 problem of Qwen3.6-35B-A3B (N=hidden_size, K=moe_intermediate_size) and
# the RTX PRO 6000 opt-in shared memory minus flashinfer's 512-byte reserve:
# the exact geometry the post-merge failure re-pinned.
_QWEN36_FC2 = dict(problem_n=2048, problem_k=512)
_RTX_PRO_6000_SMEM = 101376 - 512
_ULTRA_FC2 = dict(tile_n=512, tile_k=32, cta_threads=256)


def _flashinfer_w4a16_kernel():
    kernel = pytest.importorskip("flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel")
    if hasattr(kernel, "_ultra_wide_fc2_tile_fits"):
        pytest.skip("installed flashinfer ships the ultra-wide FC2 tile fix; drop the shim")
    return kernel


def _reset_to_stock_validator(kernel, monkeypatch) -> None:
    """Start every test from flashinfer's own validator, whichever test or
    backend construction installed the shim earlier in this process."""
    fits = kernel._candidate_tile_fits
    monkeypatch.setattr(
        kernel, "_candidate_tile_fits", getattr(fits, "_trtllm_upstream_fits", fits)
    )


def _fits(kernel, **overrides) -> bool:
    kwargs = dict(cta_m_blocks=1, max_shared_mem=_RTX_PRO_6000_SMEM, **_QWEN36_FC2, **_ULTRA_FC2)
    kwargs.update(overrides)
    return kernel._candidate_tile_fits(**kwargs)


def test_w4a16_ultra_wide_fc2_shim_accepts_repinned_ultra_tile(monkeypatch):
    kernel = _flashinfer_w4a16_kernel()
    _reset_to_stock_validator(kernel, monkeypatch)
    assert not _fits(kernel), (
        "flashinfer's stock validator accepts the ultra-wide FC2 tile; the shim in "
        "fused_moe_cute_dsl_b12x.py is no longer needed"
    )
    stock_wide_fc2 = _fits(kernel, tile_n=256, tile_k=64)

    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is True

    # The failing re-pin now passes ...
    assert _fits(kernel)
    assert _fits(kernel, scale_format="e8m0_k32")
    # ... under the override's own rule: exact N/K tiling and shared-memory fit.
    assert not _fits(kernel, problem_n=2048 + 256)
    assert not _fits(kernel, problem_k=512 + 16)
    assert not _fits(kernel, max_shared_mem=16 * 1024)
    # Everything that is not the ultra-wide FC2 tile still goes to flashinfer.
    assert _fits(kernel, tile_n=256, tile_k=64) is stock_wide_fc2
    assert not _fits(kernel, tile_n=256, tile_k=32, cta_threads=128)
    assert not _fits(kernel, tile_n=512, tile_k=32, cta_threads=128)


def test_w4a16_ultra_wide_fc2_shim_is_idempotent(monkeypatch):
    kernel = _flashinfer_w4a16_kernel()
    _reset_to_stock_validator(kernel, monkeypatch)
    stock = kernel._candidate_tile_fits

    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is True
    installed = kernel._candidate_tile_fits
    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is True

    assert kernel._candidate_tile_fits is installed
    assert installed._trtllm_upstream_fits is stock


def test_w4a16_ultra_wide_fc2_shim_yields_to_upstream_fix(monkeypatch):
    kernel = _flashinfer_w4a16_kernel()
    _reset_to_stock_validator(kernel, monkeypatch)
    stock = kernel._candidate_tile_fits
    monkeypatch.setattr(kernel, "_ultra_wide_fc2_tile_fits", lambda **_: True, raising=False)

    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is False
    assert kernel._candidate_tile_fits is stock
