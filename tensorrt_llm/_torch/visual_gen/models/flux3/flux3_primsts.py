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

"""PyTorch custom ops wrapping the vendored PrimsTS GEMM kernels."""

from __future__ import annotations

import os
from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .flux3_gemm_config import autotune_enabled, gemm_config
from .nvfp4_metadata import find_nvfp4_owner, nvfp4_owner

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm import PrimsTsGemmConfig
    from tensorrt_llm._torch.visual_gen.prims_ts.gemm.api import PreparedFp4Linear


def is_sm120(device: torch.device) -> bool:
    """SM120/SM121 run FLUX.3 GEMMs through the CuTe DSL kernels instead of PrimsTS."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))


# SM120 plain NVFP4 linears (attn_out, down) run TRT-LLM's CUTLASS SM120 block-scaled GEMM
# (fp4_gemm, CtaShape128x128x128B = config 0) instead of the CuTe DSL kernel: same 128x4 scale
# layouts, same FP32 k=64 block-scaled accumulation and BF16(acc * alpha) epilogue, so outputs are
# bitwise identical, ~40% faster at M=4096+T. The tactic is pinned (M-independent, no autotuning
# or heuristic query inside a request). TRTLLM_FLUX3_SM120_LINEAR=cute restores the CuTe kernel.
_SM120_LINEAR_CUTLASS_TACTIC = 0


@lru_cache(maxsize=1)
def _sm120_linear_backend() -> str:
    import os

    backend = os.environ.get("TRTLLM_FLUX3_SM120_LINEAR", "cutlass").strip().lower()
    if backend not in ("cutlass", "trtllm", "cute"):
        raise ValueError(
            f"TRTLLM_FLUX3_SM120_LINEAR must be 'cutlass', 'trtllm' or 'cute', got {backend!r}"
        )
    return backend


@lru_cache(maxsize=1)
def _sm120_fp4_gemm_runner():
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

    return torch.classes.trtllm.FP4GemmRunner(
        torch.bfloat16, int(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4)
    )


# SM120 plain linears can instead run FlashInfer's stock SM120 CUTLASS NVFP4 GEMM
# (flashinfer.jit.gemm.gen_gemm_sm120_module_cutlass_fp4, entry fp4_gemm) at a pinned tactic.
# Stock tactic index = tile_idx * 4 + {0 swap-AB DP, 1 DP, 2 swap-AB StreamK, 3 StreamK}; tile 4 is
# TileShape 128x128x128 (FP4 elements; FlashInfer's enum name CtaShape128x128x64B counts K in bytes).
# Index 18 = 128x128x128, swap-AB, CUTLASS StreamKScheduler (default heuristic decomposition,
# deterministic reduction): it splits the wave tail's K range across CTAs, so its outputs are a
# REORDERED fp32 accumulation of t0's (run-to-run deterministic). Index 17 is bitwise to t0.
# TRTLLM_FLUX3_SM120_LINEAR_FI_TACTIC=<idx> picks another stock tactic; TRTLLM_FLUX3_SM120_LINEAR=trtllm
# restores TRT-LLM FP4GemmRunner tactic 0.
_SM120_LINEAR_FI_TACTIC = 18
# Workspace for the stock entry. Its runGemm wants the max over all 32 stock tactics for (m, n, k)
# and allocates a fresh buffer on EVERY call when the one passed in is smaller, so it is allocated
# once per (device, stream) at model load, sized well above that max for M up to ~16K rows.
_SM120_LINEAR_FI_WORKSPACE_BYTES = 128 * 1024 * 1024


@lru_cache(maxsize=1)
def _sm120_linear_fi_tactic() -> int:
    import os

    return int(os.environ.get("TRTLLM_FLUX3_SM120_LINEAR_FI_TACTIC", _SM120_LINEAR_FI_TACTIC))


@lru_cache(maxsize=1)
def _sm120_fi_fp4_module():
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4

    return gen_gemm_sm120_module_cutlass_fp4().build_and_load()


_SM120_FI_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}


def _sm120_fi_workspace(device: torch.device) -> torch.Tensor:
    # One buffer per stream: StreamK keeps partials and barrier flags in it, so two GEMMs running
    # concurrently on different streams must not share one.
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, torch.cuda.current_stream(index).cuda_stream)
    ws = _SM120_FI_WORKSPACES.get(key)
    if ws is None:
        ws = torch.empty(_SM120_LINEAR_FI_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _SM120_FI_WORKSPACES[key] = ws
    return ws


def sm120_linear_preload(device: torch.device) -> None:
    """Load the FlashInfer SM120 GEMM module and allocate its workspace at model load."""
    if _sm120_linear_backend() == "cutlass":
        _sm120_fi_fp4_module()
        _sm120_fi_workspace(device)
        # The block tail pair runs attn_out on its own side stream: give it its workspace now too.
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.stream(_sm120_tail_pair_stream(index)):
            _sm120_fi_workspace(device)
        # The one-launch NVFP4 tail pair: build its module and allocate its workspace now.
        from . import flux3_sm120_lpt_pair

        flux3_sm120_lpt_pair.preload(device)
        # The pair with the gated residual in attn_out's epilogue (zeroed workspace: flags/epoch).
        from . import flux3_sm120_lpt_pair_residual

        flux3_sm120_lpt_pair_residual.preload(device)
        # The one-launch tail with the gated residual for M >= 24576 (two accumulators per
        # output tile; TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH).
        from . import flux3_sm120_tail_one_launch

        flux3_sm120_tail_one_launch.preload(device)
        if flux3_sm120_lpt_pair.enabled() and attn_nvfp4_out_enabled():
            # The FMHA nvfp4_out variant module (attn_out's activation from the FMHA epilogue).
            attn_nvfp4_out_module()
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass

    if flux3_swiglu_fp4_cutlass.enabled():
        # Build/load the gate_up SwiGLU->NVFP4 module now, outside any timed request.
        flux3_swiglu_fp4_cutlass.get_module()


def _sm120_cutlass_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar

    # Cached device scalar (CUDA-graph safe), the same value the CuTe epilogue multiplies by.
    alpha = _scalar(activation.device, input_global_scale * weight_global_scale)
    if _sm120_linear_backend() == "cutlass":
        out = torch.empty(
            (activation.shape[0], weight.shape[0]), device=activation.device, dtype=torch.bfloat16
        )
        _sm120_fi_fp4_module().fp4_gemm(
            activation.detach(),
            weight.detach(),
            activation_scale.detach(),
            weight_scale.detach(),
            alpha,
            out,
            _sm120_fi_workspace(activation.device),
            _sm120_linear_fi_tactic(),
        )
        return out
    return _sm120_fp4_gemm_runner().run_gemm(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        alpha,
        0,  # BufferKind.DEFAULT
        _SM120_LINEAR_CUTLASS_TACTIC,
        None,
        None,
    )


@lru_cache(maxsize=1)
def _sm120_qkv_backend() -> str:
    import os

    backend = os.environ.get("TRTLLM_FLUX3_SM120_QKV", "cutlass").strip().lower()
    if backend not in ("cutlass", "cute"):
        raise ValueError(f"TRTLLM_FLUX3_SM120_QKV must be 'cutlass' or 'cute', got {backend!r}")
    return backend


# The SM120 CUTLASS QKV path writes Q, K and V as three contiguous [M, N/3] planes (one
# [M, N] buffer, plane p at element p*M*N/3), so attention's q/k/v .contiguous() are no-ops.
# False restores the packed q|k|v rows (module switch for A/B checks; set it before preparing
# the model and tracing, since qkv_forward reads the layout decision at prepare time).
SM120_QKV_PLANAR = True


def _sm120_qkv_covers(columns: int, q_dim: int, k_dim: int) -> bool:
    """Shapes the SM120 CUTLASS QKV path (GEMMs + qknorm_rope pass) handles."""
    return columns % 3 == 0 and (columns // 3) % 512 == 0 and q_dim == 128 and k_dim == 128


def sm120_qkv_planar(
    device: torch.device, columns: int, q_dim: int, k_dim: int | None = None
) -> bool:
    """True when nvfp4_qkv_qknorm_rope on ``device`` returns three contiguous Q/K/V planes.

    The one layout decision shared by the op (nvfp4_qkv_qknorm_rope) and its reader
    (qkv_planes): SM120, CUTLASS QKV backend, covered shape and the planar switch. The CuTe
    fallback and every other device keep the packed / config-decided layouts.
    """
    return (
        SM120_QKV_PLANAR
        and is_sm120(device)
        and _sm120_qkv_backend() == "cutlass"
        and _sm120_qkv_covers(columns, q_dim, q_dim if k_dim is None else k_dim)
    )


@lru_cache(maxsize=8)
def _sm120_qkv_side_streams(device_index: int) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    return torch.cuda.Stream(device_index), torch.cuda.Stream(device_index)


def _sm120_cutlass_qkv(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    qkv_scale: torch.Tensor | None,
    planar: bool = False,
) -> torch.Tensor | None:
    """CUTLASS NVFP4 GEMM(s) + per-head QK-RMSNorm/RoPE pass, bitwise equal to the fused CuTe op.

    The CUTLASS epilogue is BF16(acc * alpha) like the CuTe one, and the pass repeats the CuTe QKV
    epilogue's norm/RoPE arithmetic on those BF16 values. Output: BF16 [M, 3*H*128]; planar=True
    lays it out as three contiguous [M, H*128] planes q, k, v (``output.view(3, M, H*128)``),
    planar=False as packed q|k|v rows (row stride 3*H*128). Same values either way.
    One alpha (qkv_scale None): one GEMM over all columns; packed runs the pass in place, planar
    writes a new buffer (v copied by the pass). Per-plane alphas: q on the current stream, k and
    v on two side streams forked from and joined to it (fills the GEMMs' wave tails), then the
    pass writes q, k and v into the output.
    Returns None for shapes it does not cover (the caller falls back to the CuTe kernel).
    """
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
        qknorm_rope,
    )

    columns = weight.shape[0]
    width = columns // 3
    if not _sm120_qkv_covers(columns, q_norm.numel(), k_norm.numel()):
        return None
    runner = _sm120_fp4_gemm_runner()
    activation = activation.detach()
    activation_scale = activation_scale.detach()
    weight = weight.detach()
    weight_scale = weight_scale.detach()
    cos_sin = cos_sin.detach()
    positions = positions.detach()
    if qkv_scale is None:
        alpha = _scalar(activation.device, input_global_scale * weight_global_scale)
        out = runner.run_gemm(
            activation, weight, activation_scale, weight_scale, alpha, 0,
            _SM120_LINEAR_CUTLASS_TACTIC, None, None,
        )
        if planar:
            dst = torch.empty_like(out)
            return qknorm_rope(
                out[:, :width], out[:, width:2 * width], out[:, 2 * width:], dst,
                q_norm, k_norm, cos_sin, positions, planar=True,
            )
        return qknorm_rope(
            out[:, :width], out[:, width:2 * width], out[:, 2 * width:], out,
            q_norm, k_norm, cos_sin, positions,
        )
    qkv_scale = qkv_scale.detach()
    segment_scale = weight_scale.numel() // 3

    from . import flux3_sm120_strided_gemm as strided

    if planar and strided.inplace_enabled():
        # QKV in place (case 1): plane i's GEMM writes plane i of the one planar output (FlashInfer
        # strided SM120 entry, bitwise to t0); then norm+RoPE runs in place on planes 0/1 and v
        # (already plane 2) is not touched.
        rows = activation.shape[0]
        out = torch.empty((rows, columns), device=activation.device, dtype=torch.bfloat16)
        planes_view = out.view(3, rows, width)

        if strided.batched():
            # Idea 82: the three planes as ONE batched (L=3) Pingpong launch on the current
            # stream (no side-stream fork), bitwise equal to the per-plane launches.
            strided.gemm_qkv3(
                activation, weight, activation_scale, weight_scale, qkv_scale, out
            )
            return qknorm_rope(
                planes_view[0], planes_view[1], planes_view[2], out,
                q_norm, k_norm, cos_sin, positions, planar=True,
            )

        def gemm_plane(i: int) -> None:
            strided.gemm_into(
                activation,
                weight[i * width:(i + 1) * width],
                activation_scale,
                weight_scale[i * segment_scale:(i + 1) * segment_scale],
                qkv_scale[i:i + 1],
                out,
                i * rows * width,
                width,
                i,
            )

        current = torch.cuda.current_stream(activation.device)
        side = _sm120_qkv_side_streams(activation.device.index)
        for stream in side:
            stream.wait_stream(current)
        gemm_plane(0)
        for i, stream in enumerate(side):
            with torch.cuda.stream(stream):
                gemm_plane(i + 1)
        for stream in side:
            current.wait_stream(stream)
        return qknorm_rope(
            planes_view[0], planes_view[1], planes_view[2], out,
            q_norm, k_norm, cos_sin, positions, planar=True,
        )

    def gemm(i: int) -> torch.Tensor:
        return runner.run_gemm(
            activation,
            weight[i * width:(i + 1) * width],
            activation_scale,
            weight_scale[i * segment_scale:(i + 1) * segment_scale],
            qkv_scale[i:i + 1],
            0,  # BufferKind.DEFAULT
            _SM120_LINEAR_CUTLASS_TACTIC,
            None,
            None,
        )

    current = torch.cuda.current_stream(activation.device)
    side = _sm120_qkv_side_streams(activation.device.index)
    for stream in side:
        stream.wait_stream(current)
    q = gemm(0)
    planes = []
    for i, stream in enumerate(side):
        with torch.cuda.stream(stream):
            planes.append(gemm(i + 1))
    for stream in side:
        current.wait_stream(stream)
    for plane in planes:
        plane.record_stream(current)
    k, v = planes
    out = torch.empty((activation.shape[0], columns), device=activation.device, dtype=torch.bfloat16)
    return qknorm_rope(q, k, v, out, q_norm, k_norm, cos_sin, positions, planar=planar)


# Block tail pair (single blocks and image mode blocks on SM120): ff.down_proj and attn_out are
# independent once attention and gate_up have run, so one op launches down on the current stream
# and attn_out (with its activation quantize) on a side stream forked from and joined to it.
# Each GEMM is ~4.1-4.2 waves of 128x128 tiles on 188 SMs; launched second, attn_out's CTAs fill
# down's last partial wave. Same kernels, same inputs, so the outputs are bitwise unchanged.
# Module-level so a microbenchmark can time attn_out-first.
_SM120_TAIL_PAIR_DOWN_FIRST = True


@lru_cache(maxsize=8)
def _sm120_tail_pair_stream(device_index: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device_index)


def _sm120_tail_pair(device: torch.device, launch_down, launch_attn):
    current = torch.cuda.current_stream(device)
    side = _sm120_tail_pair_stream(device.index)
    side.wait_stream(current)
    if _SM120_TAIL_PAIR_DOWN_FIRST:
        down = launch_down()
        with torch.cuda.stream(side):
            attn = launch_attn()
    else:
        with torch.cuda.stream(side):
            attn = launch_attn()
        down = launch_down()
    current.wait_stream(side)
    # attn was allocated on the side stream; keep its block until the current stream is done.
    attn.record_stream(current)
    return attn, down


@torch.library.custom_op(
    "trtllm::flux3_sm120_nvfp4_out_pair", mutates_args=(), device_types="cuda"
)
def nvfp4_out_pair(
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    attn: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(attn_out(attn), down(down_activation)) on the SM120 CUTLASS NVFP4 LINEAR path.

    down takes the packed-SwiGLU FP4 output and its scales; attn is BF16 [M, K] and is quantized
    with attn_out's static scale, as PreparedNVFP4.forward does.
    """
    down_owner = nvfp4_owner(down_weight)
    attn_owner = nvfp4_owner(attn_weight)
    down_weight, down_weight_scale = down_weight.detach(), down_weight_scale.detach()
    attn_weight, attn_weight_scale = attn_weight.detach(), attn_weight_scale.detach()

    def launch_down() -> torch.Tensor:
        return _sm120_cutlass_linear(
            down_activation.detach(),
            down_weight,
            down_activation_scale.detach(),
            down_weight_scale,
            down_owner.input_global_scale,
            down_owner.weight_global_scale,
        )

    def launch_attn() -> torch.Tensor:
        data, scales = nvfp4_quantize_prepared(attn.detach(), attn_weight)
        return _sm120_cutlass_linear(
            data,
            attn_weight,
            scales,
            attn_weight_scale,
            attn_owner.input_global_scale,
            attn_owner.weight_global_scale,
        )

    if _sm120_linear_backend() == "cutlass":
        from . import flux3_sm120_lpt_pair as lpt

        if lpt.enabled():
            # One persistent launch: down's 128x128 tiles (K=9216) first, then attn_out's
            # (K=3072) from a dynamic tile counter. Per output tile the MMA sequence is the stock
            # DP kernel's (bitwise to t0/t17). TRTLLM_FLUX3_SM120_PAIR_KERNEL=0 restores the
            # two-stream pair below.
            from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import (
                _scalar,
            )

            device = attn.device
            data, scales = nvfp4_quantize_prepared(attn.detach(), attn_weight)
            rows_down = down_activation.shape[0]
            rows_attn = data.shape[0]
            down_out = torch.empty(
                (rows_down, down_weight.shape[0]), device=device, dtype=torch.bfloat16
            )
            attn_out = torch.empty(
                (rows_attn, attn_weight.shape[0]), device=device, dtype=torch.bfloat16
            )
            lpt.run(
                down_activation.detach(),
                down_weight,
                down_activation_scale.detach(),
                down_weight_scale,
                _scalar(device, down_owner.input_global_scale * down_owner.weight_global_scale),
                down_out,
                data,
                attn_weight,
                scales,
                attn_weight_scale,
                _scalar(device, attn_owner.input_global_scale * attn_owner.weight_global_scale),
                attn_out,
            )
            return attn_out, down_out

    return _sm120_tail_pair(attn.device, launch_down, launch_attn)


@nvfp4_out_pair.register_fake
def _nvfp4_out_pair_fake(
    down_activation,
    down_weight,
    down_activation_scale,
    down_weight_scale,
    attn,
    attn_weight,
    attn_weight_scale,
):
    return (
        attn.new_empty((attn.shape[0], attn_weight.shape[0]), dtype=torch.bfloat16),
        attn.new_empty((down_activation.shape[0], down_weight.shape[0]), dtype=torch.bfloat16),
    )


def _fp8r_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, slot: int = 0
) -> torch.Tensor:
    """PreparedFP8Sm120.forward for a 2D BF16 input: fp8r quantize + rowwise cuBLASLt GEMM."""
    from . import flux3_fp8r_gemm
    from .fp8 import quantize_fp8r

    data, scales = quantize_fp8r(x)
    scales = scales.float().reshape(-1).contiguous()
    if flux3_fp8r_gemm.enabled():
        # Idea 122: SM120 CUTLASS FP8-rowwise GEMM, bitwise equal to torch._scaled_mm.
        return flux3_fp8r_gemm.gemm(data.contiguous(), weight, scales, scale, slot=slot)
    return torch._scaled_mm(
        data.contiguous(),
        weight.t(),
        scale_a=scales.reshape(-1, 1),
        scale_b=scale.reshape(1, -1),
        out_dtype=torch.bfloat16,
    )


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8r_out_pair", mutates_args=(), device_types="cuda"
)
def fp8r_out_pair(
    down_input: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    attn: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(attn_out(attn), down(down_input)) for FP8R (PreparedFP8Sm120) blocks, BF16 2D inputs."""
    down_input, attn = down_input.detach(), attn.detach()
    down_weight, down_scale = down_weight.detach(), down_scale.detach()
    attn_weight, attn_scale = attn_weight.detach(), attn_scale.detach()
    return _sm120_tail_pair(
        attn.device,
        lambda: _fp8r_linear(down_input, down_weight, down_scale),
        lambda: _fp8r_linear(attn, attn_weight, attn_scale, slot=1),
    )


@fp8r_out_pair.register_fake
def _fp8r_out_pair_fake(down_input, down_weight, down_scale, attn, attn_weight, attn_scale):
    return (
        attn.new_empty((attn.shape[0], attn_weight.shape[0]), dtype=torch.bfloat16),
        attn.new_empty((down_input.shape[0], down_weight.shape[0]), dtype=torch.bfloat16),
    )


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8r_out_pair_q", mutates_args=(), device_types="cuda"
)
def fp8r_out_pair_q(
    down_data: torch.Tensor,
    down_scales: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    attn: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8r_out_pair with down's input already E4M3-quantized (data [M, K], FP32 scales [M]),
    e.g. by the fused swiglu_fp8r_quant. Same GEMMs as fp8r_out_pair, so bitwise equal."""
    down_data = down_data.detach().contiguous()
    down_scales = down_scales.detach().float().reshape(-1).contiguous()
    attn = attn.detach()
    down_weight, down_scale = down_weight.detach(), down_scale.detach()
    attn_weight, attn_scale = attn_weight.detach(), attn_scale.detach()

    def launch_down() -> torch.Tensor:
        from . import flux3_fp8r_gemm

        if flux3_fp8r_gemm.enabled():
            # Idea 122: SM120 CUTLASS FP8-rowwise GEMM, bitwise equal to torch._scaled_mm.
            return flux3_fp8r_gemm.gemm(down_data, down_weight, down_scales, down_scale)
        return torch._scaled_mm(
            down_data,
            down_weight.t(),
            scale_a=down_scales.reshape(-1, 1),
            scale_b=down_scale.reshape(1, -1),
            out_dtype=torch.bfloat16,
        )

    return _sm120_tail_pair(
        attn.device,
        launch_down,
        lambda: _fp8r_linear(attn, attn_weight, attn_scale, slot=1),
    )


@fp8r_out_pair_q.register_fake
def _fp8r_out_pair_q_fake(
    down_data, down_scales, down_weight, down_scale, attn, attn_weight, attn_scale
):
    return (
        attn.new_empty((attn.shape[0], attn_weight.shape[0]), dtype=torch.bfloat16),
        attn.new_empty((down_data.shape[0], down_weight.shape[0]), dtype=torch.bfloat16),
    )


@lru_cache(maxsize=32)
def _token_positions(rows: int, device: torch.device) -> torch.Tensor:
    """Cache identity RoPE indices across blocks and denoising steps."""
    return torch.arange(rows, device=device, dtype=torch.int64)


def _version(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    # Offload can replace storage without changing tensor identity/version.
    return (
        tensor.device,
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        None if tensor.is_inference() else tensor._version,
    )


def prepare_nvfp4_gemm(
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    epilogue: str = "linear",
    output_inverse_scale: torch.Tensor | None = None,
    q_norm: torch.Tensor | None = None,
    k_norm: torch.Tensor | None = None,
    qkv_scale: torch.Tensor | None = None,
    config: PrimsTsGemmConfig | None = None,
) -> PreparedFp4Linear:
    """Cache launch descriptors on the adapter, or the weight for standalone calls."""
    versions = tuple(
        _version(t) for t in (weight, scale, output_inverse_scale, q_norm, k_norm, qkv_scale)
    )
    owner = find_nvfp4_owner(weight)
    cache_owner = weight if owner is None else owner
    cache = getattr(cache_owner, "_prepared_gemm_cache", None)
    if cache is None:
        cache = OrderedDict()
        cache_owner._prepared_gemm_cache = cache
    key = (epilogue, input_global_scale, weight_global_scale, versions, config)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    # Drop descriptors for replaced storage or changed auxiliary tensors.
    for old_key in list(cache):
        same_output = (old_key[3][2] is None) == (output_inverse_scale is None)
        if old_key[3][:2] != versions[:2] or (
            old_key[0] == epilogue and same_output and old_key[3] != versions
        ):
            del cache[old_key]
    prepared = _prepared_nvfp4_gemm(
        weight,
        scale,
        input_global_scale,
        weight_global_scale,
        epilogue,
        output_inverse_scale,
        q_norm,
        k_norm,
        qkv_scale,
        config,
    )
    cache[key] = prepared
    if len(cache) > 8:
        cache.popitem(last=False)
    return prepared


def _prepared_nvfp4_gemm(
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    epilogue: str,
    output_inverse_scale: torch.Tensor | None,
    q_norm: torch.Tensor | None,
    k_norm: torch.Tensor | None,
    qkv_scale: torch.Tensor | None,
    config: PrimsTsGemmConfig | None,
) -> PreparedFp4Linear:
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    args = (weight.detach(), scale.detach(), input_global_scale, weight_global_scale)
    if epilogue == "linear":
        return gemm.prepare_fp4_linear(*args, config=config)
    if epilogue == "swiglu":
        return gemm.prepare_fp4_linear_swiglu(
            *args,
            out_dtype=torch.bfloat16 if output_inverse_scale is None else torch.uint8,
            output_quant_scale=output_inverse_scale,
            config=config,
        )
    if epilogue == "qkv_qknorm_rope":
        if q_norm is None or k_norm is None:
            raise ValueError("QKV epilogue requires Q/K normalization weights")
        heads = weight.shape[0] // (3 * q_norm.numel())
        return gemm.prepare_fp4_qkv_qknorm_rope(
            *args,
            q_norm.detach(),
            k_norm.detach(),
            num_q_heads=heads,
            num_kv_heads=heads,
            head_dim=q_norm.numel(),
            is_neox=False,
            qkv_scale=qkv_scale,
            config=config,
        )
    raise ValueError(f"Unsupported PrimsTS epilogue: {epilogue}")


def pack_swiglu_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [gate | value] rows and their scales into [value, gate] pairs."""
    if weight.ndim != 2 or weight.shape[0] % 2 or scale.shape != (weight.shape[0],):
        raise ValueError("SwiGLU requires [2H,K] weights and [2H] row scales")
    half = weight.shape[0] // 2
    packed = torch.stack((weight[half:], weight[:half]), dim=1).reshape_as(weight)
    scales = torch.stack((scale[half:], scale[:half]), dim=1).reshape_as(scale)
    return packed.contiguous(), scales.float().contiguous()


def _gemm_config(
    x: torch.Tensor,
    weight: torch.Tensor,
    epilogue: str,
    head_dim: int | None = None,
    *,
    qkv_scale: bool = False,
    packed_output: bool = False,
) -> PrimsTsGemmConfig | None:
    major, minor = torch.cuda.get_device_capability(x.device)
    fp4 = weight.dtype == torch.uint8
    return gemm_config(
        major * 10 + minor,
        "nvfp4_e2m1" if fp4 else "fp8_e4m3",
        epilogue,
        x.shape[0],
        weight.shape[0],
        weight.shape[1] * (2 if fp4 else 1),
        "nvfp4_e2m1" if packed_output else "bf16",
        head_dim,
        qkv_scale,
        autotune_enabled(),
    )


def qkv_gemm_config(
    activation: torch.Tensor,
    weight: torch.Tensor,
    head_dim: int,
    *,
    qkv_scale: bool = False,
) -> PrimsTsGemmConfig | None:
    """Config the fused QKV GEMM runs with, None on SM120 and under autotuning.

    The config decides the output layout. With separate_qkv_output the kernel writes
    Q, K and V as three [M, N/3] planes, otherwise the output is packed [M, N].
    """
    if is_sm120(activation.device):
        return None
    return _gemm_config(activation, weight, "qkv_qknorm_rope", head_dim, qkv_scale=qkv_scale)


def qkv_planes(
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    head_dim: int,
    *,
    qkv_scale: bool = False,
    planar: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q, K and V of a fused QKV GEMM ``output``, each [M, N/3], as the GEMM laid them out.

    The QKV ops return the GEMM buffer itself because a custom op may not return views of
    its own output, so this is the one place that reads the layout: under
    separate_qkv_output, or from the SM120 CUTLASS NVFP4 path (``sm120_qkv_planar``), the
    buffer holds three contiguous [M, N/3] planes, otherwise it is the packed [M, N] tensor
    and is split along the last dimension. ``planar`` is the SM120 decision precomputed at
    prepare time (primsts.qkv_forward passes it); None derives it from the NVFP4 (uint8)
    weight. The config lookup is memoized, so it costs a dictionary hit.
    """
    width = output.shape[-1] // 3
    if planar is None:
        planar = weight.dtype == torch.uint8 and sm120_qkv_planar(
            weight.device, weight.shape[0], head_dim
        )
    if planar:
        q, k, v = output.view(3, -1, width)
        return q, k, v
    config = qkv_gemm_config(activation, weight, head_dim, qkv_scale=qkv_scale)
    if config is not None and config.separate_qkv_output:
        q, k, v = output.view(3, -1, width)
    else:
        q, k, v = output.split(width, dim=-1)
    return q, k, v


@torch.library.custom_op("trtllm::flux3_primsts_fp8_linear", mutates_args=(), device_types="cuda")
def fp8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 [M,K] times [N,K], with FP32 [M]/[N] scales, returning BF16 [M,N]."""
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    return gemm.fp8_linear(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        config=_gemm_config(activation, weight, "linear"),
    )


@fp8_linear.register_fake
def _fp8_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_primsts_fp8_swiglu", mutates_args=(), device_types="cuda")
def fp8_linear_swiglu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 [M,K] times paired [2H,K] weights, returning BF16 SwiGLU [M,H]."""
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    return gemm.fp8_linear_swiglu(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        config=_gemm_config(activation, weight, "swiglu"),
    )


@fp8_linear_swiglu.register_fake
def _fp8_linear_swiglu_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0] // 2), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_primsts_fp8_qkv_rope", mutates_args=(), device_types="cuda")
def fp8_qkv_qknorm_rope(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    """FP8 QKV GEMM with BF16 per-head RMSNorm and interleaved RoPE.

    Norm weights are [128], cos_sin is FP32 [M,128] (half cos, half sin),
    positions is int64 [M] or None for identity indices, and output is BF16 [M,3*heads*128]
    in the layout ``qkv_planes`` reads.
    The kernel has fixed RMSNorm epsilon 1e-6 and equal Q/KV head counts.
    """
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    if positions is None:
        positions = _token_positions(activation.shape[0], activation.device)
    head_dim = q_norm.numel()
    heads = weight.shape[0] // (3 * head_dim)
    return gemm.fp8_qkv_qknorm_rope(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions.detach(),
        num_q_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        is_neox=False,
        config=qkv_gemm_config(activation, weight, head_dim),
    )


@fp8_qkv_qknorm_rope.register_fake
def _fp8_qkv_qknorm_rope_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def nvfp4_scale_size(rows: int, k: int) -> int:
    """Bytes in FlashInfer's padded 128x4 E4M3 block-scale layout."""
    return ((rows + 127) // 128) * 128 * ((k // 16 + 3) // 4) * 4


def pack_nvfp4_swiglu_weight(
    weight: torch.Tensor,
    linear_scales: torch.Tensor,
    *,
    interleaved_64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert native gate/up rows to PrimsTS adjacent [value, gate] pairs.

    TRT-LLM's fused CuteDSL path already uses [value64, gate64] groups;
    ordinary linears use [gate | value]. Apply the same permutation to scales.
    """
    n = weight.shape[0]
    if n % 128 or linear_scales.shape[0] != n:
        raise ValueError("NVFP4 SwiGLU requires aligned weight and block-scale rows")

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        if interleaved_64:
            return (
                tensor.reshape(n // 128, 2, 64, -1).transpose(1, 2).reshape_as(tensor).contiguous()
            )
        half = n // 2
        return torch.stack((tensor[half:], tensor[:half]), dim=1).reshape_as(tensor).contiguous()

    return pack(weight), pack(linear_scales)


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_quantize", mutates_args=(), device_types="cuda"
)
def nvfp4_quantize(
    activation: torch.Tensor, inverse_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with the checkpoint's inverse global scale, without row shuffling."""
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    data, scales = quantize(
        activation.detach(),
        inverse_scale.detach(),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    return data.view(torch.uint8).contiguous(), scales.view(torch.uint8).reshape(-1).contiguous()


@nvfp4_quantize.register_fake
def _nvfp4_quantize_fake(
    activation: torch.Tensor,
    inverse_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = activation.shape
    return (
        activation.new_empty((m, k // 2), dtype=torch.uint8),
        activation.new_empty((nvfp4_scale_size(m, k),), dtype=torch.uint8),
    )


@torch.library.custom_op("trtllm::flux3_primsts_nvfp4_linear", mutates_args=(), device_types="cuda")
def nvfp4_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    if is_sm120(activation.device):
        if _sm120_linear_backend() != "cute":
            return _sm120_cutlass_linear(
                activation,
                weight,
                activation_scale,
                weight_scale,
                input_global_scale,
                weight_global_scale,
            )
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

        return flux3_sm120_nvfp4.run(
            "linear",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        config=_gemm_config(activation, weight, "linear"),
    )
    return prepared(activation.detach(), activation_scale.detach())


@nvfp4_linear.register_fake
def _nvfp4_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_primsts_nvfp4_swiglu", mutates_args=(), device_types="cuda")
def nvfp4_linear_swiglu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    if is_sm120(activation.device):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass

        if flux3_swiglu_fp4_cutlass.weight_is_v2w(weight):
            weight, weight_scale = flux3_swiglu_fp4_cutlass.unpermute_weight_v2w(
                weight, weight_scale
            )
        return flux3_sm120_nvfp4.run(
            "swiglu",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        "swiglu",
        config=_gemm_config(activation, weight, "swiglu"),
    )
    return prepared(activation.detach(), activation_scale.detach())


@nvfp4_linear_swiglu.register_fake
def _nvfp4_swiglu_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0] // 2), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_swiglu_packed", mutates_args=(), device_types="cuda"
)
def nvfp4_swiglu_packed(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    output_inverse_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU with fused NVFP4 output using the next projection's static scale."""
    if is_sm120(activation.device):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass

        # CUTLASS t2 mainloop + register SwiGLU->NVFP4 epilogue, bitwise to run('packed').
        if flux3_swiglu_fp4_cutlass.enabled() and flux3_swiglu_fp4_cutlass.supported(
            activation, weight
        ):
            return flux3_swiglu_fp4_cutlass.swiglu_packed(
                activation,
                weight,
                activation_scale,
                weight_scale,
                flux3_sm120_nvfp4._scalar(
                    activation.device, input_global_scale * weight_global_scale
                ),
                output_inverse_scale,
            )
        if flux3_swiglu_fp4_cutlass.weight_is_v2w(weight):
            weight, weight_scale = flux3_swiglu_fp4_cutlass.unpermute_weight_v2w(
                weight, weight_scale
            )
        return flux3_sm120_nvfp4.run(
            "packed",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            output_inverse_scale=output_inverse_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        "swiglu",
        output_inverse_scale,
        config=_gemm_config(activation, weight, "swiglu", packed_output=True),
    )
    return prepared(activation.detach(), activation_scale.detach())


@nvfp4_swiglu_packed.register_fake
def _nvfp4_swiglu_packed_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    output_inverse_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = activation.shape[0], weight.shape[0] // 2
    return activation.new_empty((m, n // 2)), activation.new_empty((nvfp4_scale_size(m, n),))


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_qkv_rope", mutates_args=(), device_types="cuda"
)
def nvfp4_qkv_qknorm_rope(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    qkv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """NVFP4 QKV GEMM with per-head RMSNorm and RoPE, BF16 [M,N] in the layout ``qkv_planes`` reads."""
    if positions is None:
        positions = _token_positions(activation.shape[0], activation.device)

    if is_sm120(activation.device):
        if _sm120_qkv_backend() == "cutlass":
            output = _sm120_cutlass_qkv(
                activation,
                weight,
                activation_scale,
                weight_scale,
                input_global_scale,
                weight_global_scale,
                q_norm,
                k_norm,
                cos_sin,
                positions,
                qkv_scale,
                planar=sm120_qkv_planar(
                    activation.device, weight.shape[0], q_norm.numel(), k_norm.numel()
                ),
            )
            if output is not None:
                return output
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

        return flux3_sm120_nvfp4.run(
            "qkv",
            activation,
            weight,
            activation_scale,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_sin=cos_sin,
            positions=positions,
            qkv_scale=qkv_scale,
        )

    prepared = prepare_nvfp4_gemm(
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        "qkv_qknorm_rope",
        q_norm=q_norm,
        k_norm=k_norm,
        qkv_scale=qkv_scale,
        config=qkv_gemm_config(activation, weight, q_norm.numel(), qkv_scale=qkv_scale is not None),
    )
    return prepared(
        activation.detach(),
        activation_scale.detach(),
        cos_sin=cos_sin.detach(),
        positions=positions.detach(),
    )


@nvfp4_qkv_qknorm_rope.register_fake
def _nvfp4_qkv_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    qkv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::flux3_nvfp4_quantize_prepared", mutates_args=())
def nvfp4_quantize_prepared(
    activation: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a host scale without exposing a per-layer float to Dynamo."""
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    data, scales = quantize(
        activation.detach(),
        1.0 / nvfp4_owner(weight).input_global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    return data.view(torch.uint8).contiguous(), scales.view(torch.uint8).reshape(-1).contiguous()


@nvfp4_quantize_prepared.register_fake
def _nvfp4_quantize_prepared_fake(activation, weight):
    return _nvfp4_quantize_fake(activation, weight)


def _prepared_activation(activation, activation_scale, original, source_weight, weight):
    if (
        source_weight is not None
        and nvfp4_owner(source_weight).input_global_scale != nvfp4_owner(weight).input_global_scale
    ):
        return nvfp4_quantize_prepared(
            original.reshape(-1, original.shape[-1]).contiguous(), weight
        )
    return activation, activation_scale


@torch.library.custom_op("trtllm::flux3_nvfp4_linear_prepared", mutates_args=())
def nvfp4_linear_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    original: torch.Tensor | None = None,
    source_weight: torch.Tensor | None = None,
    swiglu: bool = False,
) -> torch.Tensor:
    owner = nvfp4_owner(weight)
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    op = nvfp4_linear_swiglu if swiglu else nvfp4_linear
    return op(
        activation,
        weight,
        activation_scale,
        weight_scale,
        owner.input_global_scale,
        owner.weight_global_scale,
    )


@nvfp4_linear_prepared.register_fake
def _nvfp4_linear_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    original=None,
    source_weight=None,
    swiglu=False,
):
    return activation.new_empty(
        (activation.shape[0], weight.shape[0] // (2 if swiglu else 1)), dtype=torch.bfloat16
    )


@torch.library.custom_op("trtllm::flux3_nvfp4_swiglu_prepared", mutates_args=())
def nvfp4_swiglu_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_inverse_scale: torch.Tensor,
    original: torch.Tensor,
    source_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner = nvfp4_owner(weight)
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    return nvfp4_swiglu_packed(
        activation,
        weight,
        activation_scale,
        weight_scale,
        owner.input_global_scale,
        owner.weight_global_scale,
        output_inverse_scale,
    )


@nvfp4_swiglu_prepared.register_fake
def _nvfp4_swiglu_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    output_inverse_scale,
    original,
    source_weight=None,
):
    return _nvfp4_swiglu_packed_fake(
        activation, weight, activation_scale, weight_scale, 1.0, 1.0, output_inverse_scale
    )


# Side-stream gate_up beside attention (idea 161). The block's packed-SwiGLU gate_up has no data
# path to or from attention, so this op issues it on a side stream inside the attention op and
# joins it before returning (attention's consumer, the down || attn_out tail pair, needs both).
# On the default direct NVFP4-QKV -> E4M3 path (FLUX3_QKV_FP8_DIRECT) there is no K1/K2, so the
# fork (fp8_prequant_attention_side_gate_up) sits right before FMHA in every mode. On the
# BF16-QKV path (fp8_attention_side_gate_up):
# Mode 1: gate_up becomes ready right after QKV (before the attention K1/K2 quantize and FMHA).
# Mode 2: gate_up becomes ready after K2, just before the FMHA launch (beside FMHA).
# Mode 3 (default): mode 2 on a dedicated gate_up stream with the current stream's priority (mode
# 1/2 use the tail-pair stream; under graph capture both are priority 0, the lowest, so no side
# stream can be lower than the capture stream). Same kernels and inputs as the sequential path,
# so outputs are bitwise.
SIDE_GATE_UP_COUNTERS = {"issued": 0, "joined": 0, "sync_fallback": 0}


@lru_cache(maxsize=8)
def _side_gate_up_equal_priority_stream(device_index: int, priority: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device_index, priority=priority)


def _side_gate_up_stream(device: torch.device, mode: int) -> torch.cuda.Stream:
    if mode == 3:
        current = torch.cuda.current_stream(device)
        return _side_gate_up_equal_priority_stream(device.index, current.priority)
    return _sm120_tail_pair_stream(device.index)


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8_attention_side_gate_up", mutates_args=(), device_types="cuda"
)
def fp8_attention_side_gate_up(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    up_activation: torch.Tensor,
    up_weight: torch.Tensor,
    up_activation_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    output_inverse_scale: torch.Tensor,
    original: torch.Tensor,
    source_weight: torch.Tensor | None,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(visual_gen_fp8_attention_sm120(q, k, v), nvfp4_swiglu_prepared(up...)) with the gate_up
    on a side stream (mode 1/2/3 above), joined before return. Bitwise to the two ops in order."""
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as _fi
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    device = q.device
    owner = nvfp4_owner(up_weight)
    up_activation, up_activation_scale = _prepared_activation(
        up_activation, up_activation_scale, original, source_weight, up_weight
    )
    _, seq_len, heads, head_dim = q.shape

    def attention_quantize():
        return quantize_qkv_e4m3(q, k, v, sm_scale)

    def attention_fmha(q8, k8, v8, scales):
        output = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
        _fi.run_fmha_v2_fp8_sm12x(
            q8, k8, v8, output, heads, head_dim, seq_len,
            scale_bmm1_d=scales[3:4], scale_bmm2_d=scales[2:3],
            causal=is_causal, return_lse=False,
        )
        return output

    if not _side_gate_up_supported(device, mode, up_activation, up_weight):
        # Sequential fallback: the main path's order (attention, then gate_up).
        SIDE_GATE_UP_COUNTERS["sync_fallback"] += 1
        attn = attention_fmha(*attention_quantize())
        packed, packed_scales = nvfp4_swiglu_packed(
            up_activation, up_weight, up_activation_scale, up_weight_scale,
            owner.input_global_scale, owner.weight_global_scale, output_inverse_scale,
        )
        return attn, packed, packed_scales

    launch_gate_up, join, outputs = _side_gate_up_fork(
        device, mode, up_activation, up_weight, up_activation_scale, up_weight_scale,
        output_inverse_scale, owner,
    )
    fmha_first = side_gate_up_order() == "fmha_first"
    if mode == 1:
        if fmha_first:
            launch_gate_up.fork()
            attn = attention_fmha(*attention_quantize())
            launch_gate_up.enqueue()
        else:
            launch_gate_up()
            attn = attention_fmha(*attention_quantize())
    else:
        quantized = attention_quantize()
        if fmha_first:
            launch_gate_up.fork()
            attn = attention_fmha(*quantized)
            launch_gate_up.enqueue()
        else:
            launch_gate_up()
            attn = attention_fmha(*quantized)
    join()
    return attn, *outputs


def _side_gate_up_fork(
    device, mode, up_activation, up_weight, up_activation_scale, up_weight_scale,
    output_inverse_scale, owner,
):
    """(launch, join, (packed, scales)) for the CUTLASS packed-SwiGLU gate_up on the side stream.

    Outputs are allocated here, on the current stream, before the fork: no side-stream allocation
    and no record_stream (which defers frees past graph capture). launch() forks the side stream
    from the current one and enqueues gate_up on it; join() makes the current stream wait for it.
    The caller keeps the inputs alive until join (they are its op arguments)."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

    up_activation = up_activation.detach().contiguous()
    up_activation_scale = up_activation_scale.detach()
    up_weight, up_weight_scale = up_weight.detach(), up_weight_scale.detach()
    alpha = _scalar(device, owner.input_global_scale * owner.weight_global_scale)
    rows = up_activation.shape[0]
    out_buffer, out_scale = swc.allocate_outputs(rows, up_weight.shape[0], device)
    current = torch.cuda.current_stream(device)
    side = _side_gate_up_stream(device, mode)

    def fork():
        # Event on the current stream at this point; the side stream (gate_up) waits on it.
        side.wait_stream(current)

    def enqueue():
        with torch.cuda.stream(side):
            swc.swiglu_packed(
                up_activation, up_weight, up_activation_scale, up_weight_scale, alpha,
                output_inverse_scale, output=out_buffer, output_scale=out_scale,
            )
        SIDE_GATE_UP_COUNTERS["issued"] += 1

    def launch():
        fork()
        enqueue()

    def join():
        current.wait_stream(side)
        SIDE_GATE_UP_COUNTERS["joined"] += 1

    # fmha_first callers: fork() before the FMHA launch, enqueue() after it (same dependency,
    # FMHA captured/enqueued first).
    launch.fork, launch.enqueue = fork, enqueue
    return launch, join, (out_buffer[:rows], out_scale)


def side_gate_up_order() -> str:
    """TRTLLM_FLUX3_SIDE_GATE_UP_ORDER (idea 234): fmha_first (default) records the fork event
    before FMHA, enqueues/captures FMHA, then enqueues the side gate_up waiting on that event, so
    gate_up's CTAs fill FMHA's last-wave tail. gate_up_first is the previous order (gate_up
    enqueued before FMHA). Same kernels, inputs and dependencies: outputs are bitwise."""
    value = os.environ.get("TRTLLM_FLUX3_SIDE_GATE_UP_ORDER", "fmha_first").strip().lower()
    if value not in ("fmha_first", "gate_up_first"):
        raise ValueError(
            f"TRTLLM_FLUX3_SIDE_GATE_UP_ORDER must be fmha_first or gate_up_first, got {value!r}"
        )
    return value


def _side_gate_up_supported(device, mode, up_activation, up_weight) -> bool:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

    return (
        is_sm120(device)
        and mode in (1, 2, 3)
        and swc.enabled()
        and swc.supported(up_activation, up_weight)
    )


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up",
    mutates_args=(),
    device_types="cuda",
)
def fp8_prequant_attention_side_gate_up(
    qkv8: torch.Tensor,
    scales: torch.Tensor,
    batch: int,
    heads: int,
    up_activation: torch.Tensor,
    up_weight: torch.Tensor,
    up_activation_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    output_inverse_scale: torch.Tensor,
    original: torch.Tensor,
    source_weight: torch.Tensor | None,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(visual_gen_fp8_attention_sm120_prequant(qkv8, scales, batch, heads),
    nvfp4_swiglu_prepared(up...)) with gate_up forked onto the side stream right before the FMHA
    launch (the direct NVFP4-QKV -> E4M3 path has no K1/K2, so modes 1 and 2 coincide; mode 3
    uses the dedicated stream), joined before return. Bitwise to the two ops in order."""
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (
        visual_gen_fp8_attention_sm120_prequant as prequant,
    )

    device = qkv8.device
    owner = nvfp4_owner(up_weight)
    up_activation, up_activation_scale = _prepared_activation(
        up_activation, up_activation_scale, original, source_weight, up_weight
    )
    if not _side_gate_up_supported(device, mode, up_activation, up_weight):
        SIDE_GATE_UP_COUNTERS["sync_fallback"] += 1
        attn = prequant(qkv8, scales, batch, heads)
        packed, packed_scales = nvfp4_swiglu_packed(
            up_activation, up_weight, up_activation_scale, up_weight_scale,
            owner.input_global_scale, owner.weight_global_scale, output_inverse_scale,
        )
        return attn, packed, packed_scales
    launch_gate_up, join, outputs = _side_gate_up_fork(
        device, mode, up_activation, up_weight, up_activation_scale, up_weight_scale,
        output_inverse_scale, owner,
    )
    if side_gate_up_order() == "fmha_first":
        launch_gate_up.fork()
        attn = prequant(qkv8, scales, batch, heads)
        launch_gate_up.enqueue()
    else:
        launch_gate_up()
        attn = prequant(qkv8, scales, batch, heads)
    join()
    return attn, *outputs


@fp8_prequant_attention_side_gate_up.register_fake
def _fp8_prequant_attention_side_gate_up_fake(
    qkv8, scales, batch, heads, up_activation, up_weight, up_activation_scale, up_weight_scale,
    output_inverse_scale, original, source_weight, mode,
):
    _, rows, width = qkv8.shape
    packed, packed_scales = _nvfp4_swiglu_packed_fake(
        up_activation, up_weight, up_activation_scale, up_weight_scale, 1.0, 1.0,
        output_inverse_scale,
    )
    attn = torch.empty((batch, rows // batch, heads, width // heads), dtype=torch.bfloat16,
                       device=qkv8.device)
    return attn, packed, packed_scales


@lru_cache(maxsize=1)
def attn_nvfp4_out_enabled() -> bool:
    """TRTLLM_FLUX3_ATTN_NVFP4_OUT: the FP8-direct FMHA of NVFP4 tail-pair blocks writes attn_out's
    NVFP4 activation (FP4 + 128x4 SF) from its epilogue instead of BF16 O + a quantize launch
    (fmha_v2_flux3 lever nvfp4_out; bitwise to main). '0' = main's path."""
    import os

    return os.environ.get("TRTLLM_FLUX3_ATTN_NVFP4_OUT", "1").strip() != "0"


def attn_nvfp4_out_module(key: torch.Tensor | None = None, value: torch.Tensor | None = None):
    """The routed fmha_v2 flux3 module plus lever nvfp4_out: the levers main's
    run_fmha_v2_fp8_sm12x picks for these K/V (load_trim_levers), or ROUTED_LEVERS at preload."""
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx

    levers = fx.ROUTED_LEVERS
    if key is not None and value is not None:
        levers = fx.load_trim_levers(levers, key, value)
    return fx.get_flux3_d128_module((*levers, "nvfp4_out"))


def _prequant_attention_nvfp4_out(qkv8, scales, batch, heads, attn_weight, q_rows: int = 0):
    """visual_gen_fp8_attention_sm120_prequant's fmha_v2 call (routed levers + nvfp4_out) writing
    nvfp4_quantize_prepared(O, attn_weight) directly: (fp4 [B, S, W/2] uint8, sf flat uint8).

    q_rows > 0 (B = 1, last single block's keep_rows): the G3 launch computes only the Q-row
    prefix (fmha_v2_flux3_d128_nvfp4_out_q_rows) over every K/V row; only fp4 rows [0, q_rows)
    and the leading q_rows / 128 SF row blocks are defined (bitwise the full launch's)."""
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx

    _, rows, width = qkv8.shape
    seq_len = rows // batch
    head_dim = width // heads
    q8, k8, v8 = (plane.view(batch, seq_len, heads, head_dim) for plane in qkv8.unbind(0))
    fp4 = torch.empty((batch, seq_len, width // 2), dtype=torch.uint8, device=qkv8.device)
    sf = torch.empty((fx.nvfp4_scale_bytes(rows, width),), dtype=torch.uint8, device=qkv8.device)
    if q_rows > 0 and batch == 1:
        fx.fmha_v2_flux3_d128_nvfp4_out_q_rows(
            q8, k8, v8, fp4, sf, nvfp4_owner(attn_weight).inverse_scale, heads, head_dim, seq_len,
            min(int(q_rows), seq_len), scale_bmm1_d=scales[3:4], scale_bmm2_d=scales[2:3],
            module=attn_nvfp4_out_module(k8, v8),
        )
        return fp4, sf
    fx.fmha_v2_flux3_d128_nvfp4_out(
        q8, k8, v8, fp4, sf, nvfp4_owner(attn_weight).inverse_scale, heads, head_dim, seq_len,
        scale_bmm1_d=scales[3:4], scale_bmm2_d=scales[2:3], module=attn_nvfp4_out_module(k8, v8),
    )
    return fp4, sf


@torch.library.custom_op(
    "trtllm::flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up",
    mutates_args=(),
    device_types="cuda",
)
def fp8_prequant_attention_nvfp4_out_side_gate_up(
    qkv8: torch.Tensor,
    scales: torch.Tensor,
    batch: int,
    heads: int,
    up_activation: torch.Tensor,
    up_weight: torch.Tensor,
    up_activation_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    output_inverse_scale: torch.Tensor,
    original: torch.Tensor,
    source_weight: torch.Tensor | None,
    mode: int,
    attn_weight: torch.Tensor,
    q_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8_prequant_attention_side_gate_up whose FMHA writes attn_out's NVFP4 activation
    (nvfp4_quantize_prepared(O, attn_weight): fp4 [B, S, W/2], 128x4 sf) instead of BF16 O. The
    fork, side stream, gate_up kernel and join are fp8_prequant_attention_side_gate_up's.
    Returns (fp4, sf, packed, packed_scales), bitwise to the two ops plus the quantize.
    q_rows > 0: the FMHA computes only the Q-row prefix (_prequant_attention_nvfp4_out)."""
    device = qkv8.device
    owner = nvfp4_owner(up_weight)
    up_activation, up_activation_scale = _prepared_activation(
        up_activation, up_activation_scale, original, source_weight, up_weight
    )
    if not _side_gate_up_supported(device, mode, up_activation, up_weight):
        SIDE_GATE_UP_COUNTERS["sync_fallback"] += 1
        fp4, sf = _prequant_attention_nvfp4_out(qkv8, scales, batch, heads, attn_weight, q_rows)
        packed, packed_scales = nvfp4_swiglu_packed(
            up_activation, up_weight, up_activation_scale, up_weight_scale,
            owner.input_global_scale, owner.weight_global_scale, output_inverse_scale,
        )
        return fp4, sf, packed, packed_scales
    launch_gate_up, join, outputs = _side_gate_up_fork(
        device, mode, up_activation, up_weight, up_activation_scale, up_weight_scale,
        output_inverse_scale, owner,
    )
    if side_gate_up_order() == "fmha_first":
        launch_gate_up.fork()
        fp4, sf = _prequant_attention_nvfp4_out(qkv8, scales, batch, heads, attn_weight, q_rows)
        launch_gate_up.enqueue()
    else:
        launch_gate_up()
        fp4, sf = _prequant_attention_nvfp4_out(qkv8, scales, batch, heads, attn_weight, q_rows)
    join()
    return fp4, sf, *outputs


@fp8_prequant_attention_nvfp4_out_side_gate_up.register_fake
def _fp8_prequant_attention_nvfp4_out_side_gate_up_fake(
    qkv8, scales, batch, heads, up_activation, up_weight, up_activation_scale, up_weight_scale,
    output_inverse_scale, original, source_weight, mode, attn_weight, q_rows=0,
):
    _, rows, width = qkv8.shape
    packed, packed_scales = _nvfp4_swiglu_packed_fake(
        up_activation, up_weight, up_activation_scale, up_weight_scale, 1.0, 1.0,
        output_inverse_scale,
    )
    fp4 = torch.empty((batch, rows // batch, width // 2), dtype=torch.uint8, device=qkv8.device)
    sf = qkv8.new_empty((((rows + 127) // 128) * 128 * ((width // 16 + 3) // 4) * 4,),
                        dtype=torch.uint8)
    return fp4, sf, packed, packed_scales


@torch.library.custom_op(
    "trtllm::flux3_sm120_nvfp4_out_pair_prequant", mutates_args=(), device_types="cuda"
)
def nvfp4_out_pair_prequant(
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    attn_data: torch.Tensor,
    attn_scales: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """nvfp4_out_pair with attn_out's activation already quantized (attn_data [M, K/2] uint8,
    attn_scales: 128x4 sf, as nvfp4_quantize_prepared returns them): the same LPT launch (or, with
    TRTLLM_FLUX3_SM120_PAIR_KERNEL=0 / a non-cutlass backend, the same two-stream pair) minus the
    quantize."""
    down_owner = nvfp4_owner(down_weight)
    attn_owner = nvfp4_owner(attn_weight)
    down_weight, down_weight_scale = down_weight.detach(), down_weight_scale.detach()
    attn_weight, attn_weight_scale = attn_weight.detach(), attn_weight_scale.detach()
    data, scales = attn_data.detach(), attn_scales.detach()

    if _sm120_linear_backend() == "cutlass":
        from . import flux3_sm120_lpt_pair as lpt

        if lpt.enabled():
            from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import (
                _scalar,
            )

            device = data.device
            rows_down = down_activation.shape[0]
            rows_attn = data.shape[0]
            down_out = torch.empty(
                (rows_down, down_weight.shape[0]), device=device, dtype=torch.bfloat16
            )
            attn_out = torch.empty(
                (rows_attn, attn_weight.shape[0]), device=device, dtype=torch.bfloat16
            )
            lpt.run(
                down_activation.detach(),
                down_weight,
                down_activation_scale.detach(),
                down_weight_scale,
                _scalar(device, down_owner.input_global_scale * down_owner.weight_global_scale),
                down_out,
                data,
                attn_weight,
                scales,
                attn_weight_scale,
                _scalar(device, attn_owner.input_global_scale * attn_owner.weight_global_scale),
                attn_out,
            )
            return attn_out, down_out

    def launch_down() -> torch.Tensor:
        return _sm120_cutlass_linear(
            down_activation.detach(),
            down_weight,
            down_activation_scale.detach(),
            down_weight_scale,
            down_owner.input_global_scale,
            down_owner.weight_global_scale,
        )

    def launch_attn() -> torch.Tensor:
        return _sm120_cutlass_linear(
            data,
            attn_weight,
            scales,
            attn_weight_scale,
            attn_owner.input_global_scale,
            attn_owner.weight_global_scale,
        )

    return _sm120_tail_pair(data.device, launch_down, launch_attn)


@nvfp4_out_pair_prequant.register_fake
def _nvfp4_out_pair_prequant_fake(
    down_activation,
    down_weight,
    down_activation_scale,
    down_weight_scale,
    attn_data,
    attn_scales,
    attn_weight,
    attn_weight_scale,
):
    return (
        attn_data.new_empty((attn_data.shape[0], attn_weight.shape[0]), dtype=torch.bfloat16),
        attn_data.new_empty(
            (down_activation.shape[0], down_weight.shape[0]), dtype=torch.bfloat16
        ),
    )


@torch.library.custom_op(
    "trtllm::flux3_sm120_nvfp4_tail_one_launch", mutates_args=(), device_types="cuda"
)
def nvfp4_tail_one_launch(
    down_activation: torch.Tensor,
    down_weight: torch.Tensor,
    down_activation_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    attn_data: torch.Tensor,
    attn_scales: torch.Tensor,
    attn_weight: torch.Tensor,
    attn_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    select: torch.Tensor | None,
    gate_mode: int,
) -> torch.Tensor:
    """residual + gate * (attn_out(attn_data) + down(down_activation)) in ONE SM120 launch
    (flux3_sm120_tail_one_launch), bitwise equal to nvfp4_out_pair_prequant followed by the
    compiled block's residual pointwise. The operands are nvfp4_out_pair_prequant's; residual is
    the contiguous BF16 block input ([..., N]); gate is the BF16 modulation gate: gate_mode 0 one
    [N] row, 1 a [K, N] table indexed per row by select (bool/uint8, True/1 = row 1), 2 per row
    (same shape as residual)."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar

    from . import flux3_sm120_tail_one_launch as tol

    down_owner = nvfp4_owner(down_weight)
    attn_owner = nvfp4_owner(attn_weight)
    device = attn_data.device
    n = attn_weight.shape[0]
    out = torch.empty(residual.shape, device=device, dtype=residual.dtype)
    tol.run(
        attn_data.detach(),
        attn_weight.detach(),
        attn_scales.detach(),
        attn_weight_scale.detach(),
        _scalar(device, attn_owner.input_global_scale * attn_owner.weight_global_scale),
        down_activation.detach(),
        down_weight.detach(),
        down_activation_scale.detach(),
        down_weight_scale.detach(),
        _scalar(device, down_owner.input_global_scale * down_owner.weight_global_scale),
        residual.detach(),
        gate.detach().reshape(-1, n),
        None if select is None else select.detach().reshape(-1),
        out,
        gate_mode=int(gate_mode),
    )
    return out


@nvfp4_tail_one_launch.register_fake
def _nvfp4_tail_one_launch_fake(
    down_activation,
    down_weight,
    down_activation_scale,
    down_weight_scale,
    attn_data,
    attn_scales,
    attn_weight,
    attn_weight_scale,
    residual,
    gate,
    select,
    gate_mode,
):
    return residual.new_empty(residual.shape)


@fp8_attention_side_gate_up.register_fake
def _fp8_attention_side_gate_up_fake(
    q, k, v, sm_scale, is_causal, up_activation, up_weight, up_activation_scale,
    up_weight_scale, output_inverse_scale, original, source_weight, mode,
):
    packed, scales = _nvfp4_swiglu_packed_fake(
        up_activation, up_weight, up_activation_scale, up_weight_scale, 1.0, 1.0,
        output_inverse_scale,
    )
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device), packed, scales


@torch.library.custom_op("trtllm::flux3_nvfp4_qkv_prepared", mutates_args=())
def nvfp4_qkv_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    qkv_scale: torch.Tensor | None,
    original: torch.Tensor,
    source_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    owner = nvfp4_owner(weight)
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    return nvfp4_qkv_qknorm_rope(
        activation,
        weight,
        activation_scale,
        weight_scale,
        owner.input_global_scale,
        owner.weight_global_scale,
        q_norm,
        k_norm,
        cos_sin,
        None,
        qkv_scale,
    )


@nvfp4_qkv_prepared.register_fake
def _nvfp4_qkv_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    q_norm,
    k_norm,
    cos_sin,
    qkv_scale,
    original,
    source_weight=None,
):
    return activation.new_empty((activation.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def sm120_qkv_fp8_covers(device: torch.device, columns: int, q_dim: int, k_dim: int) -> bool:
    """Shapes/routes nvfp4_qkv_rope_fp8 handles: main's per-plane in-place planar QKV route."""
    from . import flux3_sm120_strided_gemm as strided

    return sm120_qkv_planar(device, columns, q_dim, k_dim) and strided.inplace_enabled()


def _sm120_qkv_planar_gemms(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    qkv_scale: torch.Tensor,
) -> torch.Tensor:
    """The GEMM half of _sm120_cutlass_qkv's in-place planar route, unchanged: plane i's GEMM
    writes plane i (element offset i*M*W) of one fresh [M, 3W] BF16 buffer, q on the current
    stream, k and v on the two QKV side streams forked from and joined to it."""
    from . import flux3_sm120_strided_gemm as strided

    columns = weight.shape[0]
    width = columns // 3
    rows = activation.shape[0]
    segment_scale = weight_scale.numel() // 3
    out = torch.empty((rows, columns), device=activation.device, dtype=torch.bfloat16)

    if strided.batched():
        # Idea 82: the three planes as ONE batched (L=3) Pingpong launch on the current stream
        # (no side-stream fork), bitwise equal to the per-plane launches below.
        strided.gemm_qkv3(activation, weight, activation_scale, weight_scale, qkv_scale, out)
        return out

    def gemm_plane(i: int) -> None:
        strided.gemm_into(
            activation,
            weight[i * width:(i + 1) * width],
            activation_scale,
            weight_scale[i * segment_scale:(i + 1) * segment_scale],
            qkv_scale[i:i + 1],
            out,
            i * rows * width,
            width,
            i,
        )

    current = torch.cuda.current_stream(activation.device)
    side = _sm120_qkv_side_streams(activation.device.index)
    for stream in side:
        stream.wait_stream(current)
    gemm_plane(0)
    for i, stream in enumerate(side):
        with torch.cuda.stream(stream):
            gemm_plane(i + 1)
    for stream in side:
        current.wait_stream(stream)
    return out


def _qkv_epi_fp8_route(activation, weight, q_norm, k_norm, cos_sin, static_qk=True) -> bool:
    """Idea 14's fused route applies exactly where main's chain is the batched QKV3 GEMM +
    qknorm_rope_fp8 VEC/FOLD with the static Q/K scale (TRTLLM_FLUX3_QKV_EPI_FP8=0: never)."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
        flux3_sm120_qknorm_rope_fp8 as qr8,
    )

    from . import flux3_qkv_epi_fp8
    from . import flux3_sm120_strided_gemm as strided

    if not static_qk:  # dynamic Q/K amax: only main's two-pass chain computes it
        return False
    if not (flux3_qkv_epi_fp8.enabled() and strided.batched()):
        return False
    if not (qr8.FOLD and qr8.VEC and qr8.STATIC_QK):
        return False
    rows, columns = activation.shape[0], weight.shape[0]
    if not flux3_qkv_epi_fp8.covers(rows, columns, q_norm.numel()) or k_norm.numel() != 128:
        return False
    if (columns // 3 // 128) % 8:  # VEC's head-pair layout (main falls back otherwise)
        return False
    if (cos_sin.dtype != torch.float32 or cos_sin.dim() != 2 or cos_sin.shape[1] != 128
            or cos_sin.shape[0] < rows or not cos_sin.is_contiguous()):
        return False
    if q_norm.dtype != torch.bfloat16 or k_norm.dtype != torch.bfloat16:
        return False
    if not (q_norm.is_contiguous() and k_norm.is_contiguous()):
        return False
    return all(t.data_ptr() % 16 == 0 for t in (q_norm, k_norm, cos_sin))


def nvfp4_qkv_rope_fp8_impl(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    qkv_scale: torch.Tensor,
    sm_scale: float,
    keep_v: bool = True,
    static_qk: bool = True,
    rope_tab: torch.Tensor | None = None,
    rope_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(qkv8 [3, M, W] E4M3, scales [8] FP32, gemm [M, 3W] BF16 planar GEMM output).

    static_qk=True: the module's STATIC_QK rule (main); False: dynamic Q/K amax for this call.
    rope_tab / rope_idx: flux3_qkv_epi_fp8.build_rope_compact(cos_sin) or None; read only by the
    fused epilogue route (same values as cos_sin, idea 84).

    Same GEMMs as main's route; then flux3_sm120_qknorm_rope_fp8 instead of the in-place BF16
    norm/RoPE pass + fp8_sm12x K1/K2. gemm.view(3, M, W)[2] is v, the plane main hands PAG.
    Nothing reads gemm's q/k planes afterwards: qknorm_rope_fp8 may drop them from L2
    (discard_qk; contents undefined). keep_v=False (PAG off) lets it drop v's lines too."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope_fp8 import (
        qknorm_rope_fp8,
    )

    rows = activation.shape[0]
    width = weight.shape[0] // 3
    if _qkv_epi_fp8_route(activation, weight, q_norm, k_norm, cos_sin, static_qk):
        # Idea 14: Q/K E4M3 straight from the QKV GEMM epilogue + a V-only quantize; bitwise
        # main's qkv8/scales and gemm v plane (q/k planes of gemm undefined, as main's L2 discard).
        # `positions` is the identity _token_positions table here (the epilogue indexes cos_sin
        # by row).
        from . import flux3_qkv_epi_fp8

        return flux3_qkv_epi_fp8.qkv_rope_fp8(
            activation.detach(),
            weight.detach(),
            activation_scale.detach(),
            weight_scale.detach(),
            qkv_scale.detach(),
            q_norm.detach(),
            k_norm.detach(),
            cos_sin.detach(),
            sm_scale,
            keep_v,
            None if rope_tab is None else rope_tab.detach(),
            None if rope_idx is None else rope_idx.detach(),
        )
    out = _sm120_qkv_planar_gemms(
        activation.detach(),
        weight.detach(),
        activation_scale.detach(),
        weight_scale.detach(),
        qkv_scale.detach(),
    )
    planes = out.view(3, rows, width)
    qkv8, scales = qknorm_rope_fp8(
        planes[0],
        planes[1],
        planes[2],
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions.detach(),
        sm_scale,
        discard_qk=True,
        keep_v=keep_v,
        static_qk=None if static_qk else False,
    )
    return qkv8, scales, out


@torch.library.custom_op(
    "trtllm::flux3_primsts_nvfp4_qkv_rope_fp8", mutates_args=(), device_types="cuda"
)
def nvfp4_qkv_rope_fp8_prepared(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    qkv_scale: torch.Tensor,
    original: torch.Tensor,
    source_weight: torch.Tensor | None,
    sm_scale: float,
    keep_v: bool = True,
    static_qk: bool = True,
    rope_tab: torch.Tensor | None = None,
    rope_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """nvfp4_qkv_prepared + fp8_sm12x's quantize, straight from the NVFP4 QKV GEMM to E4M3.

    Same activation handling as nvfp4_qkv_prepared (_prepared_activation; `original` is read only
    when the source weight's input scale differs). All three outputs are fresh buffers (no view
    of an input or of another output)."""
    # Per-plane alphas come from qkv_scale (the global scales only matter for one-alpha QKV).
    activation, activation_scale = _prepared_activation(
        activation, activation_scale, original, source_weight, weight
    )
    return nvfp4_qkv_rope_fp8_impl(
        activation,
        weight,
        activation_scale,
        weight_scale,
        q_norm,
        k_norm,
        cos_sin,
        _token_positions(activation.shape[0], activation.device),
        qkv_scale,
        sm_scale,
        keep_v,
        static_qk,
        rope_tab,
        rope_idx,
    )


@nvfp4_qkv_rope_fp8_prepared.register_fake
def _nvfp4_qkv_rope_fp8_prepared_fake(
    activation,
    weight,
    activation_scale,
    weight_scale,
    q_norm,
    k_norm,
    cos_sin,
    qkv_scale,
    original,
    source_weight,
    sm_scale,
    keep_v=True,
    static_qk=True,
    rope_tab=None,
    rope_idx=None,
):
    rows, columns = activation.shape[0], weight.shape[0]
    return (
        activation.new_empty((3, rows, columns // 3), dtype=torch.float8_e4m3fn),
        activation.new_empty((8,), dtype=torch.float32),
        activation.new_empty((rows, columns), dtype=torch.bfloat16),
    )
