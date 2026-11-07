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
"""``trtllm.cutlass.grouped_gemm.w4a8_mxfp4_mxfp8``."""

import os
from dataclasses import replace

import torch

from tensorrt_llm.logger import logger

from ....utils import ActivationType
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext, require_comm_plan
from ..impl_identity import register_moe_impl
from ..quantization import W4A8MXFP4MXFP8CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags, quantize_input_mxfp8
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor

#: CUTLASS calls (more tokens than the FlashInfer decode chain takes) of at most this many host-known
#: tokens hand the BF16 activations straight to ``fused_moe``: its row expansion (expandInputRowsKernel,
#: need_mxfp8_quant) quantizes them to MXFP8 with the same cvt_warp_fp16_to_mxfp8 the standalone
#: ``mxfp8_quantize`` kernel uses, so one launch per layer (and the FP8 copy's write + re-read) goes away,
#: bitwise. Negative (the default) folds at every CUTLASS token count; 0 disables the fold.
_MXFP8_FOLD_MAX_TOKENS = int(os.environ.get("TRTLLM_MOE_MXFP8_QUANT_FOLD_MAX_TOKENS", "-1"))
#: ... and of at least this many: below it the fold's measured gain is within run-to-run noise
#: (SM120 EP4 rank, M=128 paired runs read 2-4% slower cross-process), so small prefill chunks keep the
#: standalone quant.
_MXFP8_FOLD_MIN_TOKENS = int(os.environ.get("TRTLLM_MOE_MXFP8_QUANT_FOLD_MIN_TOKENS", "1024"))


@register_moe_impl
class TrtllmCutlassW4a8Mxfp4Mxfp8Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a8_mxfp4_mxfp8``.

    Activations are MXFP8-quantized per forward.

    On SM120/121, decode calls (at most ``FI_DECODE_MAX_TOKENS`` tokens) run FlashInfer's
    SM12x CuTe-DSL chain instead (``w4a8_mxfp4_mxfp8_sm12x_decode``): it quantizes the
    BF16 activations itself, so ``quantize_input`` hands those calls through unquantized and
    ``run_moe`` recognizes them by their missing scale factors. Prefill stays on CUTLASS.
    """

    descriptor = cutlass_descriptor(
        "w4a8_mxfp4_mxfp8",
        "CUTLASS grouped GEMM over MXFP4 weights with dynamic MXFP8 activations.",
    )
    sm_support = SmSupport(allowed=frozenset({100, 103, 107, 120, 121}))
    supported_dtypes = HP_DTYPES
    supports_gptoss_style = True
    requires_oss_cutlass_build = True
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(use_mxfp8_act_scaling=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return W4A8MXFP4MXFP8CutlassFusedMoEMethod()

    # ------------------------------------------------------------------
    # SM12x FlashInfer decode chain
    # ------------------------------------------------------------------

    def _fi_decode_layer_reason(self) -> str:
        """Why this layer cannot take the FlashInfer chain, or '' when it can."""
        if self.activation_type != ActivationType.Swiglu:
            return f"activation {self.activation_type!r} is not SwiGLU"
        if getattr(self, "act_alpha", None) is not None or getattr(self, "act_beta", None) is not None:
            return "SwiGLU alpha/beta are not supported"
        if getattr(self, "act_clamp_after_silu", False):
            return "post-SiLU clamp is not supported"
        if self.w3_w1_bias is not None or self.w2_bias is not None:
            return "expert bias is not supported"
        if getattr(self, "_moe_lora_enabled", False):
            return "routed-expert LoRA is not supported"
        if not self.use_fused_finalize:
            return "finalize fusion is disabled"
        if self.num_slots != self.num_experts:
            return "the load balancer remaps expert slots"
        if self.dtype not in (None, torch.bfloat16):
            return f"dtype {self.dtype} is not bfloat16"
        if self.unpadded_hidden_size not in (None, self.hidden_size):
            return "hidden size is padded"
        if self.hidden_size % 128 or self.intermediate_size_per_partition % 128:
            return "hidden/intermediate size not a multiple of 128"
        return ""

    def _fi_swiglu_limit(self):
        clamp = getattr(self, "act_clamp", None)
        if clamp is None:
            return None
        if isinstance(clamp, torch.Tensor):
            vals = clamp.float().flatten()
            if vals.numel() == 0:
                return None
            first = float(vals[0])
            if not bool(torch.all(vals == first)):
                return False  # non-uniform: the FlashInfer kernel bakes one value
            clamp = first
        clamp = float(clamp)
        if clamp == float("inf"):
            return None
        return clamp

    def _build_fi_decode_state(self) -> None:
        self._fi_decode = None
        from .w4a8_mxfp4_mxfp8_sm12x_decode import Sm12xW4a8DecodeState, fi_decode_available

        ok, reason = fi_decode_available(self.w3_w1_weight.device)
        if ok:
            reason = self._fi_decode_layer_reason()
        limit = self._fi_swiglu_limit() if not reason else None
        if not reason and limit is False:
            reason = "per-expert SwiGLU clamp is not uniform"
        if reason:
            logger.info_once(
                f"W4A8 MXFP4xMXFP8 SM12x FlashInfer decode chain off: {reason}",
                key="w4a8_mxfp4_mxfp8_sm12x_fi_decode_off",
            )
            return
        state = Sm12xW4a8DecodeState(
            w3_w1_weight=self.w3_w1_weight,
            w3_w1_weight_scale=self.w3_w1_weight_scale,
            w2_weight=self.w2_weight,
            w2_weight_scale=self.w2_weight_scale,
            num_local_experts=self.expert_size_per_partition,
            slot_start=self.slot_start,
            swiglu_limit=limit,
            enable_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1",
        )
        state.warmup(self.routing_method.get_experts_per_token())
        state.warmup_prefill(self.routing_method.get_experts_per_token())
        self._fi_decode = state
        logger.info_once(
            "W4A8 MXFP4xMXFP8 SM12x FlashInfer decode chain on: decode calls of at most "
            f"{self._fi_max_tokens()} tokens bypass CUTLASS; extra scale-factor copy "
            f"{state.extra_bytes / 2**20:.1f} MiB per layer",
            key="w4a8_mxfp4_mxfp8_sm12x_fi_decode_on",
        )

    @staticmethod
    def _fi_max_tokens() -> int:
        from .w4a8_mxfp4_mxfp8_sm12x_decode import FI_DECODE_MAX_TOKENS

        return FI_DECODE_MAX_TOKENS

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        self._build_fi_decode_state()

    def _takes_fi_decode(self, x) -> bool:
        return (
            getattr(self, "_fi_decode", None) is not None
            and isinstance(x, torch.Tensor)
            and x.dtype == torch.bfloat16
            and 0 < x.shape[0] <= self._fi_max_tokens()
        )

    def _takes_fi_chain(self, x) -> bool:
        """Decode calls the FlashInfer chain takes (_takes_fi_decode), plus the prefill calls its
        gate (fi_prefill_takes: token range and TRTLLM_SM12X_W4A8_FI_PREFILL mode) sends there."""
        if self._takes_fi_decode(x):
            return True
        if (
            getattr(self, "_fi_decode", None) is None
            or not isinstance(x, torch.Tensor)
            or x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.shape[1] != self.hidden_size
            or not x.is_contiguous()
        ):
            return False
        from .w4a8_mxfp4_mxfp8_sm12x_decode import fi_prefill_takes

        return fi_prefill_takes(x.shape[0])

    def _folds_quant_into_cutlass(self, x) -> bool:
        """BF16 call above the FlashInfer decode range whose MXFP8 quant ``fused_moe`` does in-kernel."""
        if _MXFP8_FOLD_MAX_TOKENS == 0 or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            return False
        m = x.shape[0] if x.dim() == 2 else 0
        return (
            m > self._fi_max_tokens()
            and m >= _MXFP8_FOLD_MIN_TOKENS
            and (_MXFP8_FOLD_MAX_TOKENS < 0 or m <= _MXFP8_FOLD_MAX_TOKENS)
            and x.shape[1] == self.hidden_size
            and x.shape[1] % self.quant_method.weight_alignment == 0
            and self.unpadded_hidden_size in (None, self.hidden_size)
            and x.is_contiguous()
        )

    def quantize_input(self, x, post_quant_comm: bool = True, **kwargs):
        """MXFP8-quantize for CUTLASS; hand decode calls through in BF16 when they take the
        FlashInfer chain, which quantizes them itself, and CUTLASS calls above that range in
        BF16 too: ``fused_moe``'s row expansion quantizes them in-kernel, bitwise as here.
        Calls followed by communication (``post_quant_comm``) always quantize here."""
        if not post_quant_comm and (self._takes_fi_chain(x) or self._folds_quant_into_cutlass(x)):
            return x, None
        return quantize_input_mxfp8(self, x, post_quant_comm=post_quant_comm, **kwargs)

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        if ctx.x_sf is not None or not self._takes_fi_chain(ctx.x):
            return super().run_moe(ctx, workspace=workspace)
        plan = require_comm_plan(self, ctx)
        if (
            not ctx.do_finalize
            or plan.enable_alltoall
            or ctx.lora_params
            or ctx.token_selected_experts is None
            or ctx.token_final_scales is None
            or ctx.output_dtype not in (None, torch.bfloat16)
        ):
            # Not a call the chain serves: quantize now and run CUTLASS as before.
            if ctx.x_ready_event is not None:
                ctx.x_ready_event.wait()
            xq, xsf = quantize_input_mxfp8(self, ctx.x, post_quant_comm=False)
            ctx = replace(
                ctx, x=xq, x_sf=xsf, x_ready_event=None,
                comm_plan=replace(plan, input_sf_swizzled=True),
            )
            return super().run_moe(ctx, workspace=workspace)
        if ctx.x_ready_event is not None:
            ctx.x_ready_event.wait()
        out = self._fi_decode.run(ctx.x, ctx.token_selected_experts, ctx.token_final_scales)
        if plan.moe_output is not None:
            plan.moe_output.copy_(out)
            return plan.moe_output
        return out
