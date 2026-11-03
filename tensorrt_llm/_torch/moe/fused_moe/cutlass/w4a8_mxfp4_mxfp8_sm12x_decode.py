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
"""SM120/121 decode chain for W4A8 MXFP4 x MXFP8 MoE: FlashInfer's SM12x CuTe-DSL ops.

The CUTLASS grouped GEMM has one SM120 FP8xFP4 tile (128x128x128B), so a decode batch of
1-8 tokens pads every active expert to 128 rows, and the chain spends ~7 launches per layer
on permute / strides / activation / finalize. FlashInfer 0.7.2 ships the three-op chain
``mxfp8_q0_route_triton`` -> ``cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4`` (8x128 swap-AB tile,
SwiGLU + MXFP8 quant in the epilogue) -> ``cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4`` (routing
weight + scatter into the BF16 output).  Its unified runner refuses expert parallelism and
allocates per call, so this module calls the ops directly, the way
``SM12xMxfp8Mxfp4Runner.forward`` does, with rank-local expert ids (``-1`` for experts on
another rank: the route kernels give such a pair no row).

The FP4 weight bytes are shared with the CUTLASS layout; only the ue8m0 scale factors are
re-laid out (TRT-LLM 128x4 swizzle -> FlashInfer MN-major int32 packs), once at load.
"""

from __future__ import annotations

import os
import threading

import torch

from tensorrt_llm.logger import logger

#: Calls with at most this many tokens take the FlashInfer chain; larger ones (prefill)
#: stay on CUTLASS. 32 tokens x top-6 = 192 pairs keeps the route on its decode kernel
#: (<= 256 pairs) and the GEMMs on the small-M tiles.
FI_DECODE_MAX_TOKENS = int(os.environ.get("TRTLLM_SM12X_W4A8_FI_DECODE_MAX_TOKENS", "32"))

_ENABLE_ENV = "TRTLLM_SM12X_W4A8_FI_DECODE"

_LOCK = threading.Lock()
_OPS = None  # (route, make_ws, fc1, out_sf_shape, fc2, FiActivation) once imported
_WARMED: set = set()
_WORKSPACES: dict = {}


def _load_ops():
    global _OPS
    if _OPS is None:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
            cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
            out_sf_shape,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize import (
            cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
            make_mxfp8_q0_route_workspace,
            mxfp8_q0_route_triton,
        )
        from flashinfer.tllm_enums import ActivationType as FiActivationType

        _OPS = (
            mxfp8_q0_route_triton,
            make_mxfp8_q0_route_workspace,
            cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
            out_sf_shape,
            cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
            FiActivationType,
        )
    return _OPS


def fi_decode_available(device: torch.device | None = None) -> tuple[bool, str]:
    """The SM12x runner's ``_check_support`` conditions that do not depend on the layer."""
    if os.environ.get(_ENABLE_ENV, "1") == "0":
        return False, f"disabled by {_ENABLE_ENV}=0"
    if not torch.cuda.is_available():
        return False, "no CUDA"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((12, 0), (12, 1)):
        return False, f"SM{major}{minor} is not SM120/121"
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available
        from flashinfer.jit.cpp_ext import get_cuda_version

        if get_cuda_version().major < 13:
            return False, "FlashInfer SM12x MXFP8xMXFP4 needs CUDA >= 13"
        if not is_cute_dsl_available():
            return False, "CuTe DSL is not available"
        _load_ops()
    except Exception as e:  # noqa: BLE001 - any import problem means "not available"
        return False, f"FlashInfer SM12x ops unavailable: {e!r}"
    return True, ""


def swizzled_sf_to_fi(sf_swizzled: torch.Tensor, num_experts: int, rows: int, k: int) -> torch.Tensor:
    """TRT-LLM swizzled ue8m0 SF of a (rows, k) MXFP4 weight -> FlashInfer (E, k/128, 4*rows) uint8."""
    raw = torch.ops.trtllm.block_scale_interleave_reverse(
        sf_swizzled.view(torch.uint8).reshape(num_experts, rows, k // 32)
    )
    out = raw.view(torch.int32).transpose(1, 2).contiguous().view(torch.uint8)
    del raw
    return out


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _workspace(x: torch.Tensor, local_ids: torch.Tensor, num_experts: int):
    """Route workspace shared by every layer: layers run one after another on one stream."""
    key = (x.device, x.shape[0], x.shape[1], local_ids.shape[1], num_experts)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _load_ops()[1](x, local_ids, num_experts)
        _WORKSPACES[key] = ws
    return ws


class Sm12xW4a8DecodeState:
    """Per-layer FlashInfer views of the expert weights plus the call."""

    def __init__(
        self,
        *,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        num_local_experts: int,
        slot_start: int,
        swiglu_limit: float | None,
        enable_pdl: bool,
    ):
        e = num_local_experts
        n1 = w3_w1_weight.shape[1]  # 2 * intermediate
        k1 = w3_w1_weight.view(torch.uint8).reshape(e, n1, -1).shape[2] * 2  # hidden
        n2 = w2_weight.shape[1]  # hidden
        k2 = w2_weight.view(torch.uint8).reshape(e, n2, -1).shape[2] * 2  # intermediate
        self.num_experts, self.hidden, self.inter = e, k1, k2
        self.slot_start = slot_start
        self.swiglu_limit = swiglu_limit
        self.enable_pdl = enable_pdl
        # FP4 data: same storage as the CUTLASS weights.
        self.w1 = w3_w1_weight.view(torch.uint8).reshape(e, n1, k1 // 2)
        self.w2 = w2_weight.view(torch.uint8).reshape(e, n2, k2 // 2)
        self.w1_sf = swizzled_sf_to_fi(w3_w1_weight_scale, e, n1, k1)
        self.w2_sf = swizzled_sf_to_fi(w2_weight_scale, e, n2, k2)

    @property
    def extra_bytes(self) -> int:
        return self.w1_sf.numel() + self.w2_sf.numel()

    def local_ids(self, token_selected_experts: torch.Tensor) -> torch.Tensor:
        ids = token_selected_experts.to(torch.int32) - self.slot_start
        return torch.where(
            (ids >= 0) & (ids < self.num_experts), ids, torch.full_like(ids, -1)
        )

    def run(self, x: torch.Tensor, token_selected_experts: torch.Tensor,
            token_final_scales: torch.Tensor) -> torch.Tensor:
        route, _, fc1, out_sf_shape, fc2, fi_act = _load_ops()
        m = x.shape[0]
        local = self.local_ids(token_selected_experts)
        weights = token_final_scales.to(torch.float32)
        pairs = local.numel()
        ws = _workspace(x, local, self.num_experts)
        # q1, sf1 and the output all have to start at zero: FC2 reads rows of q1 that FC1
        # leaves unwritten (stale bytes there are NaN often enough to poison real rows), the
        # padded rows of sf1 likewise, and FC2 scatter-adds into the output. One zeroed
        # buffer, one memset.
        sf1_shape = out_sf_shape(pairs, self.inter, self.num_experts)
        n_q1 = pairs * self.inter
        n_sf1 = sf1_shape[0] * sf1_shape[1] * 4
        off_sf1 = _align(n_q1, 256)
        off_out = _align(off_sf1 + n_sf1, 256)
        buf = torch.zeros(off_out + m * self.hidden * 2, dtype=torch.uint8, device=x.device)
        q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, self.inter)
        sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
        out = buf[off_out:].view(torch.bfloat16).view(m, self.hidden)
        offsets, token_map, route_weights, a_q, a_sf = route(
            x, local, weights, self.num_experts, workspace=ws, enable_pdl=self.enable_pdl
        )
        fc1(a_q, a_sf, self.w1, self.w1_sf, offsets, tune=False, enable_pdl=self.enable_pdl,
            out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=self.swiglu_limit)
        fc2(q1, sf1, self.w2, self.w2_sf, offsets, token_map, route_weights, m, tune=False,
            enable_pdl=self.enable_pdl, out=out)
        return out

    def warmup(self, top_k: int) -> None:
        """JIT every kernel and Triton specialization decode can hit, outside any graph capture.

        The CuTe-DSL kernels are keyed by tile and SM count only, so one compile serves every
        layer and every decode batch; the Triton route kernels specialize on scalar
        arguments, so each batch size up to the threshold is run once.
        """
        key = (self.hidden, self.inter, self.num_experts, top_k, self.enable_pdl,
               self.swiglu_limit)
        with _LOCK:
            if key in _WARMED:
                return
            dev = self.w1.device
            for m in range(1, FI_DECODE_MAX_TOKENS + 1):
                x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=dev)
                ids = (torch.arange(m * top_k, device=dev, dtype=torch.int32)
                       % self.num_experts + self.slot_start).view(m, top_k)
                w = torch.full((m, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
                self.run(x, ids, w)
            torch.cuda.synchronize(dev)
            _WARMED.add(key)
            logger.info(
                f"SM12x W4A8 MXFP4xMXFP8 FlashInfer decode chain warmed up for 1..{FI_DECODE_MAX_TOKENS} "
                f"tokens (hidden={self.hidden}, inter={self.inter}, experts={self.num_experts})"
            )
