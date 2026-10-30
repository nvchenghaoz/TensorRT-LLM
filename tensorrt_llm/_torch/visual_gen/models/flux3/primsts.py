# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 PrimsTS weight preparation and dispatch."""

import functools
import math
import os
from typing import NamedTuple

import torch
from torch import nn

from tensorrt_llm._torch.modules.linear import (
    FP8RowwiseLinearMethod,
    Fp8RowwiseQuantizedTensor,
    Linear,
    NVFP4LinearMethod,
    TensorParallelMode,
    UnquantizedLinearMethod,
)
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
    fp8_linear,
    fp8_linear_swiglu,
    fp8_qkv_qknorm_rope,
    fp8r_out_pair,
    fp8r_out_pair_q,
    is_sm120,
    nvfp4_out_pair,
    nvfp4_linear_prepared,
    nvfp4_qkv_prepared,
    nvfp4_quantize_prepared,
    nvfp4_scale_size,
    nvfp4_swiglu_prepared,
    pack_nvfp4_swiglu_weight,
    pack_swiglu_weight,
    prepare_nvfp4_gemm,
    qkv_planes,
    sm120_linear_preload,
    sm120_qkv_planar,
)
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import (
    Flux3FP8RowwiseLinearMethod,
    quantize_fp8r,
    swiglu_fp8r_quant,
)
from tensorrt_llm.logger import logger

from . import flux3_fp8r_gemm as _fp8r_gemm
from .flux3_fp8r_qkv import _sm120_fp8r_qknorm_rope_eligible, _sm120_fp8r_qkv_forward
from .nvfp4_metadata import register_nvfp4_owner


def _enabled() -> bool:
    return os.environ.get("TRTLLM_FLUX3_PRIMS_TS", "1") != "0"


def _supported_nvfp4_device(weight: torch.Tensor) -> bool:
    return is_sm120(weight.device) or (
        weight.is_cuda
        and torch.cuda.get_device_capability(weight.device) in ((10, 0), (10, 3), (10, 7))
    )


def _has_prims_ts(names: tuple[str, ...]) -> bool:
    from tensorrt_llm._torch.visual_gen.prims_ts import gemm

    return all(callable(getattr(gemm, name, None)) for name in names)


def _has_fp8_prims_ts() -> bool:
    return _has_prims_ts(("fp8_linear", "fp8_linear_swiglu", "fp8_qkv_qknorm_rope"))


def _has_nvfp4_prims_ts() -> bool:
    return _has_prims_ts(
        ("prepare_fp4_linear", "prepare_fp4_linear_swiglu", "prepare_fp4_qkv_qknorm_rope")
    )


class NVFP4Activation(NamedTuple):
    original: torch.Tensor
    quantized: torch.Tensor
    scales: torch.Tensor
    quantized_by: torch.Tensor


PrimsTSInput = torch.Tensor | Fp8RowwiseQuantizedTensor | NVFP4Activation


class NVFP4AttnOut(NamedTuple):
    """An attention output held only as attn_out's NVFP4 activation (written by the FMHA epilogue,
    TRTLLM_FLUX3_ATTN_NVFP4_OUT): data uint8 [B, S, H*D/2], scales flat 128x4 E4M3 bytes. Exactly
    nvfp4_quantize_prepared(attn, attn_out weight) of the BF16 attention output it replaces."""

    data: torch.Tensor
    scales: torch.Tensor


def original_input(x: PrimsTSInput) -> torch.Tensor:
    return x.original if isinstance(x, (Fp8RowwiseQuantizedTensor, NVFP4Activation)) else x


class PreparedNVFP4(nn.Module):
    """Static checkpoint scales and 128x4 weight layouts prepared before tracing."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        input_global_scale: float,
        weight_global_scale: float,
        *,
        swiglu: bool = False,
        qkv_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("qkv_scale", qkv_scale, persistent=False)
        self.swiglu = swiglu
        self.input_global_scale = input_global_scale
        self.weight_global_scale = weight_global_scale
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.register_buffer(
            "scale", scale.detach().view(torch.uint8).reshape(-1).contiguous(), persistent=False
        )
        self.register_buffer(
            "inverse_scale",
            torch.tensor([1.0 / input_global_scale], device=weight.device, dtype=torch.float32),
            persistent=False,
        )

        register_nvfp4_owner(self)

    def _apply(self, fn, recurse=True):
        cache = getattr(self, "_prepared_gemm_cache", None)
        if cache is not None:
            cache.clear()
        return super()._apply(fn, recurse=recurse)

    def can_run(self, x: torch.Tensor) -> bool:
        return (
            not torch.is_grad_enabled()
            and x.is_cuda
            and x.device == self.weight.device
            and x.dtype == torch.bfloat16
            and x.shape[-1] == self.weight.shape[1] * 2
            and x.numel() > 0
        )

    def activation(self, x: PrimsTSInput) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, NVFP4Activation):
            return x.quantized, x.scales
        original = original_input(x)
        return nvfp4_quantize_prepared(
            original.reshape(-1, original.shape[-1]).contiguous(), self.weight
        )

    def forward(self, x: PrimsTSInput) -> torch.Tensor:
        original = original_input(x)
        data, scales = self.activation(x)
        output = nvfp4_linear_prepared(
            data,
            self.weight,
            scales,
            self.scale,
            original,
            x.quantized_by if isinstance(x, NVFP4Activation) else None,
            self.swiglu,
        )
        return output.reshape(*original.shape[:-1], output.shape[-1])


def _local_projection(layer: Linear) -> bool:
    """Adapters bypass Linear.forward, so they must not bypass a collective."""
    mapping = getattr(layer, "mapping", None)
    if mapping is None or mapping.tp_size == 1:
        return True
    return not (
        (layer.tp_mode == TensorParallelMode.ROW and layer.reduce_output)
        or (layer.tp_mode == TensorParallelMode.COLUMN and layer.gather_output)
    )


def _eligible_nvfp4(layer: Linear) -> bool:
    return (
        isinstance(layer, Linear)
        and type(layer.quant_method) is NVFP4LinearMethod
        and _supported_nvfp4_device(layer.weight)
        and layer.bias is None
        and _local_projection(layer)
        and layer.pre_quant_scale is None
        and (
            not layer.force_dynamic_quantization
            or getattr(layer, "_flux3_nvfp4_missing_input_scale", False)
        )
        and layer.input_scale is not None
        and layer.input_scale.numel() == 1
        and layer.weight_scale_2.numel() == 1
        and layer.weight.dtype == torch.uint8
        and layer.weight.shape[1] % 128 == 0
        and layer.weight_scale.numel()
        == nvfp4_scale_size(layer.weight.shape[0], layer.weight.shape[1] * 2)
        and layer.dtype in (None, torch.bfloat16)
    )


def _prepare_nvfp4(layer: Linear, swiglu: bool) -> PreparedNVFP4:
    # FLUX.3 PrimsTS uses a unit global scale when the checkpoint omits it.
    # Keep native projection state intact for feature-off and fallback execution.
    inverse_scale = (
        1.0
        if getattr(layer, "_flux3_nvfp4_missing_input_scale", False)
        else float(layer.input_scale.item())
    )
    weight_global_scale = float(layer.weight_scale_2.item())
    if not all(math.isfinite(s) and s > 0 for s in (inverse_scale, weight_global_scale)):
        raise ValueError("PrimsTS requires finite positive NVFP4 checkpoint global scales")
    weight, scale = layer.weight.detach(), layer.weight_scale.detach()
    if swiglu:
        scale = unswizzle_sf(scale, weight.shape[0], weight.shape[1] * 2)
        weight, scale = pack_nvfp4_swiglu_weight(
            weight,
            scale,
            interleaved_64=layer.use_cute_dsl_blockscaling_mm,
        )
        scale = torch.ops.trtllm.block_scale_interleave(scale)
    return PreparedNVFP4(weight, scale, 1.0 / inverse_scale, weight_global_scale, swiglu=swiglu)


def _qkv_epilogue_eligible(attn: nn.Module) -> bool:
    return (
        attn.head_dim == 128
        and attn.local_num_attention_heads == attn.local_num_key_value_heads
        and attn.norm_q.variance_epsilon == 1e-6
        and attn.norm_k.variance_epsilon == 1e-6
        and attn.norm_q.weight.dtype == torch.bfloat16
        and attn.norm_k.weight.dtype == torch.bfloat16
        and attn.norm_q.weight.shape == (128,)
        and attn.norm_k.weight.shape == (128,)
    )


def prepare_nvfp4_block(block: nn.Module) -> None:
    """Prepare image/joint NVFP4 linears without changing the checkpoint topology."""
    if block._fuse_proj_out:
        return
    attn = block.attn
    attn._flux3_primsts_nvfp4_qkv = None
    attn._flux3_primsts_nvfp4_qkv_planar = False
    qkv = getattr(attn, "qkv_proj", None)
    projections = [qkv] if qkv is not None else [attn.to_q, attn.to_k, attn.to_v]
    layers = [*projections, block.attn_out, block.ff.gate_up_proj, block.ff.down_proj]
    available = (
        _enabled() and any(_eligible_nvfp4(layer) for layer in layers) and _has_nvfp4_prims_ts()
    )
    for layer in layers:
        layer._flux3_primsts_nvfp4 = None
        if not available or not _eligible_nvfp4(layer):
            continue
        swiglu = layer is block.ff.gate_up_proj
        if swiglu and not (
            block.ff.activation is torch.nn.functional.silu
            and block.ff._is_plain_swiglu()
            and block.ff.swiglu_limit is None
        ):
            continue
        layer._flux3_primsts_nvfp4 = _prepare_nvfp4(layer, swiglu)
        adapter = layer._flux3_primsts_nvfp4
        if is_sm120(adapter.weight.device):
            if not swiglu:
                sm120_linear_preload(adapter.weight.device)
                # QKV in place: build the strided GEMM module and its workspaces at load.
                from .flux3_sm120_strided_gemm import preload as sm120_strided_preload

                sm120_strided_preload(adapter.weight.device)
                # Idea 14: the QKV GEMM with the E4M3 Q/K epilogue (JIT build at load).
                from .flux3_qkv_epi_fp8 import preload as qkv_epi_fp8_preload

                qkv_epi_fp8_preload(adapter.weight.device)
            continue
        prepare_nvfp4_gemm(
            adapter.weight,
            adapter.scale,
            adapter.input_global_scale,
            adapter.weight_global_scale,
            "swiglu" if swiglu else "linear",
        )
    up = getattr(block.ff.gate_up_proj, "_flux3_primsts_nvfp4", None)
    down = getattr(block.ff.down_proj, "_flux3_primsts_nvfp4", None)
    if up is not None and down is not None and up.swiglu and not is_sm120(up.weight.device):
        prepare_nvfp4_gemm(
            up.weight,
            up.scale,
            up.input_global_scale,
            up.weight_global_scale,
            "swiglu",
            down.inverse_scale,
        )
    if not _qkv_epilogue_eligible(attn):
        return
    adapters = [getattr(layer, "_flux3_primsts_nvfp4", None) for layer in projections]
    if any(adapter is None for adapter in adapters):
        return
    first = adapters[0]
    if any(a.input_global_scale != first.input_global_scale for a in adapters[1:]):
        return
    if len(adapters) == 1:
        attn._flux3_primsts_nvfp4_qkv = first
    else:
        weight = torch.cat([a.weight for a in adapters], dim=0)
        scales = torch.cat(
            [unswizzle_sf(a.scale, a.weight.shape[0], a.weight.shape[1] * 2) for a in adapters],
            dim=0,
        )
        scale = torch.ops.trtllm.block_scale_interleave(scales)
        attn._flux3_primsts_nvfp4_qkv = PreparedNVFP4(
            weight,
            scale,
            first.input_global_scale,
            first.weight_global_scale,
            qkv_scale=torch.tensor(
                [a.input_global_scale * a.weight_global_scale for a in adapters],
                device=weight.device,
                dtype=torch.float32,
            ),
        )
    adapter = attn._flux3_primsts_nvfp4_qkv
    # Static layout decision of the SM120 CUTLASS QKV op (three contiguous planes), resolved
    # here so traced code reads a plain bool (no env/lru_cache lookups under Dynamo).
    attn._flux3_primsts_nvfp4_qkv_planar = sm120_qkv_planar(
        adapter.weight.device,
        adapter.weight.shape[0],
        attn.norm_q.weight.numel(),
        attn.norm_k.weight.numel(),
    )
    if is_sm120(adapter.weight.device):
        return
    prepare_nvfp4_gemm(
        adapter.weight,
        adapter.scale,
        adapter.input_global_scale,
        adapter.weight_global_scale,
        "qkv_qknorm_rope",
        q_norm=attn.norm_q.weight,
        k_norm=attn.norm_k.weight,
        qkv_scale=adapter.qkv_scale,
    )


def verify_nvfp4_coverage(blocks: list[tuple[str, nn.Module]]) -> None:
    """Require complete image/joint NVFP4 coverage unless PrimsTS is disabled."""
    if not _enabled():
        return
    blocks = [
        (name, block)
        for name, block in blocks
        if any(
            isinstance(layer, Linear)
            and type(layer.quant_method) is NVFP4LinearMethod
            and _supported_nvfp4_device(layer.weight)
            for layer in block.modules()
        )
    ]
    if not blocks:
        return
    missing = []
    for name, block in blocks:
        if block._fuse_proj_out:
            missing.append(f"{name}.proj_out (combined projection)")
            continue
        for label, layer in (
            ("attn_out", block.attn_out),
            ("ff.gate_up_proj", block.ff.gate_up_proj),
            ("ff.down_proj", block.ff.down_proj),
        ):
            adapter = getattr(layer, "_flux3_primsts_nvfp4", None)
            if adapter is None or (label == "ff.gate_up_proj" and not adapter.swiglu):
                missing.append(f"{name}.{label}")
        if getattr(block.attn, "_flux3_primsts_nvfp4_qkv", None) is None:
            missing.append(f"{name}.attn (QKV+QKNorm+RoPE)")
    if missing:
        raise RuntimeError(
            "FLUX.3 NVFP4 PrimsTS coverage incomplete: "
            + ", ".join(missing)
            + ". Check PrimsTS availability, projection shapes, and scales, "
            "or set TRTLLM_FLUX3_PRIMS_TS=0 before model loading for native execution."
        )
    logger.info(
        f"FLUX.3 NVFP4 PrimsTS coverage verified: {len(blocks)} image/joint blocks; "
        f"{len(blocks)} each of QKV+QKNorm+RoPE, fused MLP-up, attention-out, and MLP-down."
    )


def shared_nvfp4_input(attn: nn.Module, mlp_in: Linear, x: torch.Tensor) -> NVFP4Activation | None:
    adapter = getattr(mlp_in, "_flux3_primsts_nvfp4", None)
    qkv = getattr(attn, "_flux3_primsts_nvfp4_qkv", None)
    if qkv is None:
        qkv_layer = getattr(attn, "qkv_proj", None)
        if qkv_layer is None:
            qkv_layer = getattr(attn, "to_q", None)
        qkv = getattr(qkv_layer, "_flux3_primsts_nvfp4", None)
    if adapter is None or qkv is None:
        return None
    if not adapter.can_run(x) or not qkv.can_run(x):
        return None
    data, scales = adapter.activation(x)
    return NVFP4Activation(x, data, scales, adapter.weight)


def separate_qkv_forward(attn: nn.Module, x: PrimsTSInput) -> tuple[torch.Tensor, ...] | None:
    fused = getattr(attn, "_flux3_sm120_fp8_qkv", None)
    if fused is not None and fused.can_run(original_input(x)):
        return tuple(fused(x).split(fused.weight.shape[0] // 3, dim=-1))
    layers = [attn.to_q, attn.to_k, attn.to_v]
    if not any(getattr(layer, "_flux3_primsts_nvfp4", None) is not None for layer in layers):
        return None
    return tuple(linear_forward(layer, x) for layer in layers)


def nvfp4_mlp_forward(ff: nn.Module, x: PrimsTSInput) -> torch.Tensor | None:
    up = getattr(ff.gate_up_proj, "_flux3_primsts_nvfp4", None)
    down = getattr(ff.down_proj, "_flux3_primsts_nvfp4", None)
    original = original_input(x)
    if up is None or down is None or not up.swiglu or not up.can_run(original):
        return None
    data, scales = up.activation(x)
    packed, output_scales = nvfp4_swiglu_prepared(
        data,
        up.weight,
        scales,
        up.scale,
        down.inverse_scale,
        original,
        x.quantized_by if isinstance(x, NVFP4Activation) else None,
    )
    output = nvfp4_linear_prepared(
        packed,
        down.weight,
        output_scales,
        down.scale,
    )
    return output.reshape(*original.shape[:-1], output.shape[-1])


class PreparedFP8(nn.Module):
    """FP8 weight layout prepared before tracing; refresh after weights or scales change."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        swiglu: bool = False,
        fp32_scales: bool = False,
    ):
        super().__init__()
        self.swiglu = swiglu
        self.fp32_scales = fp32_scales
        if swiglu:
            weight, scale = pack_swiglu_weight(weight.detach(), scale.detach())
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        self.register_buffer("scale", scale.detach().float().contiguous(), persistent=False)

    def can_run(self, x: torch.Tensor) -> bool:
        return (
            not torch.is_grad_enabled()
            and x.is_cuda
            and x.device == self.weight.device
            and x.dtype == torch.bfloat16
            and x.shape[-1] == self.weight.shape[1]
            and x.numel() > 0
        )

    def activation(self, x: torch.Tensor | Fp8RowwiseQuantizedTensor):
        if isinstance(x, Fp8RowwiseQuantizedTensor):
            return x.quantized.contiguous(), x.scales.float().reshape(-1).contiguous()
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if self.fp32_scales:
            data, scales = quantize_fp8r(flat)
        elif FP8RowwiseLinearMethod.use_tunable_quantize:
            data, scales = torch.ops.trtllm.tunable_fp8_per_token_quant(flat)
        else:
            data, scales = torch.ops.tensorrt_llm.quantize_e4m3_activation(flat)
        return data.contiguous(), scales.float().reshape(-1).contiguous()

    def forward(self, x: torch.Tensor | Fp8RowwiseQuantizedTensor) -> torch.Tensor:
        original = x.original if isinstance(x, Fp8RowwiseQuantizedTensor) else x
        data, scales = self.activation(x)
        op = fp8_linear_swiglu if self.swiglu else fp8_linear
        output = op(data, self.weight, scales, self.scale)
        return output.reshape(*original.shape[:-1], output.shape[-1])


def _sm120_fp8_excluded_enabled() -> bool:
    return _enabled() and os.environ.get("TRTLLM_FLUX3_SM120_FP8_EXCLUDED", "1") != "0"


def quantize_fp8r_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel E4M3 weights with FP32 amax/448 scales.

    Bitwise identical to the BFL fp8r package variant (variants/fp8r) for the layers
    the NVFP4 variant keeps in BF16.
    """
    w = weight.detach().float()
    scale = torch.clamp(w.abs().amax(dim=1) / 448.0, min=1e-12)
    data = (w / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.contiguous(), scale.contiguous()


class PreparedFP8Sm120(PreparedFP8):
    """FP8R (per-token x per-channel E4M3) execution of a BF16 checkpoint-excluded linear on SM120.

    Activations are quantized with the fp8r recipe (quantize_fp8r) and the GEMM runs
    through cuBLASLt rowwise-scaled FP8 (torch._scaled_mm) with FP32 accumulation.
    With swiglu the output is SwiGLU over the native [gate | up] layout.
    """

    def __init__(self, weight: torch.Tensor, *, swiglu: bool = False):
        data, scale = quantize_fp8r_weight(weight)
        super().__init__(data, scale, swiglu=False, fp32_scales=True)
        self.swiglu = swiglu

    def gemm(self, data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """The rowwise-scaled FP8 GEMM alone (no SwiGLU), BF16 [M, N]."""
        if _fp8r_gemm.enabled():
            # Idea 122: SM120 CUTLASS FP8-rowwise GEMM, bitwise equal to torch._scaled_mm.
            return torch.ops.trtllm.flux3_sm120_fp8r_gemm(data, self.weight, scales, self.scale)
        return torch._scaled_mm(
            data,
            self.weight.t(),
            scale_a=scales.reshape(-1, 1),
            scale_b=self.scale.reshape(1, -1),
            out_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor | Fp8RowwiseQuantizedTensor) -> torch.Tensor:
        original = x.original if isinstance(x, Fp8RowwiseQuantizedTensor) else x
        data, scales = self.activation(x)
        output = self.gemm(data, scales)
        if self.swiglu:
            output = swiglu(output)
        return output.reshape(*original.shape[:-1], output.shape[-1])


# Fuse the FP8R MLP's SwiGLU with down_proj's per-token E4M3 quantization (bitwise equal to
# silu_and_mul -> quantize_fp8r). Module-level so an in-process A/B can flip it.
_FP8R_FUSED_SWIGLU_QUANT = os.environ.get("TRTLLM_FLUX3_FP8R_FUSED_SWIGLU_QUANT", "1") != "0"


def sm120_fp8_mlp_forward(ff: nn.Module, x: PrimsTSInput) -> torch.Tensor | None:
    """gate_up GEMM -> fused SwiGLU + E4M3 quantize -> down GEMM for FP8R (SM120) blocks."""
    if not _FP8R_FUSED_SWIGLU_QUANT:
        return None
    up = getattr(ff.gate_up_proj, "_flux3_primsts_fp8", None)
    down = getattr(ff.down_proj, "_flux3_primsts_fp8", None)
    if not (
        isinstance(up, PreparedFP8Sm120)
        and isinstance(down, PreparedFP8Sm120)
        and up.swiglu
        and not down.swiglu
        and up.weight.shape[0] == 2 * down.weight.shape[1]
    ):
        return None
    original = original_input(x)
    if isinstance(x, NVFP4Activation) or not up.can_run(original):
        return None
    data, scales = up.activation(x)
    hidden, hidden_scales = swiglu_fp8r_quant(up.gemm(data, scales))
    output = down.gemm(hidden, hidden_scales)
    return output.reshape(*original.shape[:-1], output.shape[-1])


def _excluded_bf16(layer: nn.Module) -> bool:
    return (
        isinstance(layer, Linear)
        and type(layer.quant_method) is UnquantizedLinearMethod
        and layer.weight.is_cuda
        and is_sm120(layer.weight.device)
        and layer.weight.dtype == torch.bfloat16
        and layer.bias is None
        and _local_projection(layer)
        and layer.weight.shape[1] % 16 == 0
        and layer.weight.shape[0] % 16 == 0
        and layer.dtype in (None, torch.bfloat16)
    )


_EXCLUDED_NVFP4_ENV = "TRTLLM_FLUX3_EXCLUDED_NVFP4"
# single_blocks.24-27: the largest subset whose Qwen3-VL judge count matched main's (43/43) in
# branch 105's STEP 1 (job 1270); "all" (adds content_mode_blocks.image.0) lost one answer.
_EXCLUDED_NVFP4_DEFAULT = "single"


def _excluded_bf16_block(block: nn.Module) -> bool:
    """A BF16 checkpoint-excluded image/joint block whose six projections are plain BF16."""
    if block._fuse_proj_out or getattr(block, "_allreduce", None) is not None:
        return False
    attn = block.attn
    if getattr(attn, "qkv_proj", None) is not None:
        return False
    projections = [getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")]
    layers = [*projections, block.attn_out, block.ff.gate_up_proj, block.ff.down_proj]
    return (
        all(_excluded_bf16(layer) for layer in layers)
        and all(layer.weight.shape[1] % 128 == 0 for layer in layers)
        and block.ff.activation is torch.nn.functional.silu
        and block.ff._is_plain_swiglu()
        and block.ff.swiglu_limit is None
    )


def _excluded_nvfp4_selection(names: list[str]) -> tuple[str, list[str]]:
    """(env value, selected names) from TRTLLM_FLUX3_EXCLUDED_NVFP4.

    none / 0 / off: no block (main's FP8R path); all: every excluded block; single: the
    single_blocks.* ones; otherwise a comma list of block names.
    """
    value = os.environ.get(_EXCLUDED_NVFP4_ENV, _EXCLUDED_NVFP4_DEFAULT).strip()
    key = value.lower()
    if key in ("", "none", "0", "off", "false"):
        return value, []
    if key in ("all", "1"):
        return value, list(names)
    if key == "single":
        return value, [n for n in names if n.startswith("single_blocks.")]
    wanted = {v.strip() for v in value.split(",") if v.strip()}
    return value, [n for n in names if n in wanted]


def _convert_bf16_linear_to_nvfp4(layer: Linear, quant_config) -> None:
    """Replace a BF16 Linear's weight with an NVFP4 W4A4 copy in the checkpoint's layout.

    ModelOpt recipe (the checkpoint loader's own quantize_nvfp4): per-tensor weight_scale_2 =
    amax/(6*448), per-16 E4M3 block scales (128x4 swizzled like checkpoint NVFP4 layers), and
    a unit activation global scale, as for checkpoint NVFP4 layers without input_scale.
    """
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
    from tensorrt_llm.quantization.utils import fp4_utils

    weight = layer.weight.detach()
    device = weight.device
    with torch.no_grad():
        qweight, scale, weight_scale_2 = quantize_nvfp4(weight)
        scale = torch.ops.trtllm.block_scale_interleave(
            scale.view(fp4_utils.float4_sf_dtype).contiguous()
        )
        weight_scale_2 = weight_scale_2.reshape(1).to(device=device, dtype=torch.float32)
    del weight
    layer.quant_config = quant_config
    layer.quant_method = NVFP4LinearMethod()
    layer.scaling_vector_size = fp4_utils.NVFP4_SF_VEC_SIZE
    layer.weight = nn.Parameter(qweight.contiguous(), requires_grad=False)
    layer.weight_scale = nn.Parameter(
        scale.reshape(-1).view(fp4_utils.float4_sf_dtype).contiguous(), requires_grad=False
    )
    layer.weight_scale_2 = nn.Parameter(weight_scale_2.clone(), requires_grad=False)
    layer.input_scale = nn.Parameter(
        torch.ones(1, device=device, dtype=torch.float32), requires_grad=False
    )
    layer.inv_input_scale = nn.Parameter(
        torch.full((1,), 1.0 / 6.0, device=device, dtype=torch.float32), requires_grad=False
    )
    layer.alpha = nn.Parameter(weight_scale_2.clone(), requires_grad=False)
    layer.scalar_alpha = float(weight_scale_2.item())
    layer.kv_scales = nn.Parameter(
        torch.ones(3, device=device, dtype=torch.float32), requires_grad=False
    )
    layer.inv_kv_scales = nn.Parameter(
        torch.ones(3, device=device, dtype=torch.float32), requires_grad=False
    )
    layer.pre_quant_scale = None
    layer._flux3_nvfp4_missing_input_scale = True


def convert_excluded_blocks_to_nvfp4(
    blocks: list[tuple[str, nn.Module]], quant_config
) -> tuple[str, list[str]]:
    """PRECISION: give the selected BF16 checkpoint-excluded blocks load-time NVFP4 W4A4 weights.

    Runs before prepare_nvfp4_block, so a converted block takes the full NVFP4 block route
    (PrimsTS adapters, CUTLASS SwiGLU gate_up, NVFP4 tail pair, fused adaLN+NVFP4 quant) and
    prepare_sm120_fp8_excluded_block leaves it alone. TRTLLM_FLUX3_EXCLUDED_NVFP4=none keeps
    every excluded block BF16 (main's FP8R route).
    """
    if not _enabled():
        return "", []
    excluded = [
        name for name, block in blocks if _excluded_bf16_block(getattr(block, "_orig_mod", block))
    ]
    value, selected = _excluded_nvfp4_selection(excluded)
    by_name = dict(blocks)
    for name in selected:
        block = getattr(by_name[name], "_orig_mod", by_name[name])
        attn = block.attn
        for layer in (
            attn.to_q,
            attn.to_k,
            attn.to_v,
            block.attn_out,
            block.ff.gate_up_proj,
            block.ff.down_proj,
        ):
            _convert_bf16_linear_to_nvfp4(layer, quant_config)
    if selected:
        torch.cuda.empty_cache()
    return value, selected


def prepare_sm120_fp8_excluded_block(block: nn.Module) -> bool:
    """Run a BF16 checkpoint-excluded image/joint block of an NVFP4 package as FP8R on SM120.

    Returns True when the block's QKV, attn_out, MLP-up and MLP-down were switched.
    """
    attn = block.attn
    attn._flux3_sm120_fp8_qkv = None
    attn._flux3_sm120_fp8r_qknorm_rope = False
    if block._fuse_proj_out or not _sm120_fp8_excluded_enabled():
        return False
    projections = [getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")]
    layers = [*projections, block.attn_out, block.ff.gate_up_proj, block.ff.down_proj]
    if not all(_excluded_bf16(layer) for layer in layers):
        return False
    if any(
        getattr(layer, "_flux3_primsts_fp8", None) is not None
        or getattr(layer, "_flux3_primsts_nvfp4", None) is not None
        for layer in layers
    ):
        return False
    if not (
        block.ff.activation is torch.nn.functional.silu
        and block.ff._is_plain_swiglu()
        and block.ff.swiglu_limit is None
    ):
        return False
    with torch.no_grad():
        attn._flux3_sm120_fp8_qkv = PreparedFP8Sm120(
            torch.cat([layer.weight for layer in projections], dim=0)
        )
        block.attn_out._flux3_primsts_fp8 = PreparedFP8Sm120(block.attn_out.weight)
        block.ff.gate_up_proj._flux3_primsts_fp8 = PreparedFP8Sm120(
            block.ff.gate_up_proj.weight, swiglu=True
        )
        block.ff.down_proj._flux3_primsts_fp8 = PreparedFP8Sm120(block.ff.down_proj.weight)
    attn._flux3_sm120_fp8r_qknorm_rope = _sm120_fp8r_qknorm_rope_eligible(attn)
    # Idea 122 (TRTLLM_FLUX3_FP8R_GEMM): build the FP8R GEMM and rebind the FP8R QKV op, at load.
    _fp8r_gemm.install(attn._flux3_sm120_fp8_qkv.weight.device)
    return True


def sm120_fp8_shared_input(
    attn: nn.Module, mlp_in: Linear, x: torch.Tensor
) -> Fp8RowwiseQuantizedTensor | None:
    qkv = getattr(attn, "_flux3_sm120_fp8_qkv", None)
    up = getattr(mlp_in, "_flux3_primsts_fp8", None)
    if not (isinstance(qkv, PreparedFP8Sm120) and isinstance(up, PreparedFP8Sm120)):
        return None
    if not (qkv.can_run(x) and up.can_run(x)):
        return None
    data, scales = quantize_fp8r(x.reshape(-1, x.shape[-1]))
    return Fp8RowwiseQuantizedTensor(x, data, scales)


def _eligible_fp8(layer: Linear) -> bool:
    return (
        isinstance(layer, Linear)
        and isinstance(layer.quant_method, FP8RowwiseLinearMethod)
        and layer.weight.is_cuda
        and torch.cuda.get_device_capability(layer.weight.device) in ((10, 0), (10, 3), (10, 7))
        and layer.bias is None
        and _local_projection(layer)
        and layer.weight.dtype == torch.float8_e4m3fn
        and layer.weight.shape[1] % 128 == 0
        and layer.weight.shape[0] % 512 == 0
        and layer.weight_scale.shape == (layer.weight.shape[0],)
        and layer.dtype in (None, torch.bfloat16)
    )


def prepare_fp8_block(block: nn.Module) -> None:
    """Prepare a selected image/joint block; leave unsupported linears native."""
    if block._fuse_proj_out:
        return
    attn = block.attn
    attn._primsts_fp8_qkv_rope = False
    layers = [block.attn_out, block.ff.gate_up_proj, block.ff.down_proj]
    qkv = getattr(attn, "qkv_proj", None)
    if qkv is not None:
        layers.append(qkv)
    # Resolve optional PrimsTS APIs before tracing.
    available = _enabled() and any(_eligible_fp8(layer) for layer in layers) and _has_fp8_prims_ts()
    for layer in layers:
        # Clear stale adapters after quantization or LoRA changes.
        layer._flux3_primsts_fp8 = None
        if not available or not _eligible_fp8(layer):
            continue
        swiglu = layer is block.ff.gate_up_proj
        if swiglu and not (
            block.ff.activation is torch.nn.functional.silu
            and block.ff._is_plain_swiglu()
            and block.ff.swiglu_limit is None
        ):
            continue
        layer._flux3_primsts_fp8 = PreparedFP8(
            layer.weight,
            layer.weight_scale,
            swiglu=swiglu,
            fp32_scales=isinstance(layer.quant_method, Flux3FP8RowwiseLinearMethod),
        )
    if qkv is not None and getattr(qkv, "_flux3_primsts_fp8", None) is not None:
        attn._primsts_fp8_qkv_rope = _qkv_epilogue_eligible(attn)


def _sm120_tail_pair_enabled() -> bool:
    return _enabled() and os.environ.get("TRTLLM_FLUX3_SM120_TAIL_PAIR", "1") != "0"


def prepare_sm120_tail_pair(block: nn.Module) -> bool:
    """Set block._sm120_tail_pair, the static switch of sm120_tail_pair_forward.

    True when attn_out and ff.down_proj are both SM120 CUTLASS NVFP4 adapters (with the packed
    SwiGLU gate_up) or both FP8R adapters, on a local (tp_size 1) block. Call after every
    adapter preparation and before compiling; forward reads only the flag.
    """
    eligible = False
    if (
        not block._fuse_proj_out
        and getattr(block, "_allreduce", None) is None
        and _sm120_tail_pair_enabled()
    ):
        from .flux3_primsts import _sm120_linear_backend

        layers = (block.attn_out, block.ff.gate_up_proj, block.ff.down_proj)
        out, up, down = (getattr(layer, "_flux3_primsts_nvfp4", None) for layer in layers)
        fp8 = [getattr(layer, "_flux3_primsts_fp8", None) for layer in layers]
        if all(a is None for a in fp8) and all(a is not None for a in (out, up, down)):
            eligible = (
                up.swiglu
                and not out.swiglu
                and not down.swiglu
                and is_sm120(out.weight.device)
                and _sm120_linear_backend() == "cutlass"
                and all(_local_projection(layer) for layer in layers)
            )
        elif all(isinstance(a, PreparedFP8Sm120) for a in fp8):
            out, up, down = fp8
            eligible = up.swiglu and not out.swiglu and not down.swiglu
    block._sm120_tail_pair = bool(eligible)
    # Side-stream gate_up beside attention (flux3_primsts.fp8_attention_side_gate_up): NVFP4
    # tail-pair blocks only. TRTLLM_FLUX3_SIDE_GATE_UP=0 keeps the sequential gate_up (main's graph).
    block._sm120_side_gate_up = (
        _side_gate_up_mode()
        if block._sm120_tail_pair
        and getattr(block.attn_out, "_flux3_primsts_nvfp4", None) is not None
        else 0
    )
    block._sm120_attn_nvfp4_out = _sm120_attn_nvfp4_out_eligible(block)
    block._sm120_pair_residual = _sm120_pair_residual_eligible(block)
    block._sm120_tail_one_launch = _sm120_tail_one_launch_eligible(block)
    if block._sm120_tail_one_launch:
        from . import flux3_sm120_tail_one_launch as tol

        # Static M-range bucket of the one-launch tail (read in forward as a plain int).
        block._sm120_tail_one_launch_min_m = tol.min_rows()
    return block._sm120_tail_pair


def _sm120_tail_one_launch_eligible(block: nn.Module) -> bool:
    """Static switch of sm120_tail_one_launch_forward (TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH): NVFP4
    tail-pair blocks (SM120 CUTLASS NVFP4 attn_out and down) with no TP all-reduce."""
    if not getattr(block, "_sm120_tail_pair", False):
        return False
    if getattr(block, "_allreduce", None) is not None:
        return False
    from . import flux3_sm120_tail_one_launch as tol
    from .flux3_primsts import _sm120_linear_backend

    return bool(
        tol.enabled()
        and _sm120_linear_backend() == "cutlass"
        and getattr(block.attn_out, "_flux3_primsts_nvfp4", None) is not None
        and getattr(block.ff.down_proj, "_flux3_primsts_nvfp4", None) is not None
    )


def _sm120_pair_residual_eligible(block: nn.Module) -> bool:
    """Static switch of sm120_tail_pair_residual_forward (TRTLLM_FLUX3_SM120_PAIR_RESIDUAL): NVFP4
    tail-pair blocks whose pair runs as the one-launch LPT kernel."""
    out = getattr(block.attn_out, "_flux3_primsts_nvfp4", None)
    if not block._sm120_tail_pair or out is None:
        return False
    from . import flux3_sm120_lpt_pair_residual as lptr

    return bool(lptr.available(out.weight.device))


def _sm120_attn_nvfp4_out_eligible(block: nn.Module) -> bool:
    """Static switch of the FMHA NVFP4 output epilogue (TRTLLM_FLUX3_ATTN_NVFP4_OUT): NVFP4
    tail-pair blocks with the side-stream gate_up whose pair runs as the one-launch LPT kernel."""
    if not getattr(block, "_sm120_side_gate_up", 0):
        return False
    from . import flux3_sm120_lpt_pair as lpt
    from .flux3_primsts import _sm120_linear_backend, attn_nvfp4_out_enabled

    return bool(
        attn_nvfp4_out_enabled()
        and lpt.enabled()
        and _sm120_linear_backend() == "cutlass"
        and getattr(block.attn_out, "_flux3_primsts_nvfp4", None) is not None
        # Inputs: block._flux3_cond_stream (static, set at prepare for the i2i reference-stream
        # blocks content_mode_blocks.image_cond) and TRTLLM_FLUX3_COND_ATTN_NFO. A subordinate
        # opt-out only: it can turn nfo off for cond blocks, never on.
        and (not getattr(block, "_flux3_cond_stream", False) or _cond_attn_nfo_enabled())
    )


@functools.lru_cache(maxsize=None)
def _cond_attn_nfo_enabled() -> bool:
    """TRTLLM_FLUX3_COND_ATTN_NFO for the prepared image_cond blocks (M = 16384 at 2048^2).
    1 (default): follow the global FMHA NVFP4-out switch (attn_nvfp4_out_enabled) like image 1-4
    (G3 NVFP4-out epilogue; no separate nvfp4_quantize_prepared before the LPT pair+residual).
    0: BF16-out FMHA + side gate_up, attn_out quantized before the LPT pair (branch 11's default,
    measured against the old 4-warp nvfp4_out kernel before G3 had the NVFP4-out epilogue).
    Outputs are bitwise equal either way."""
    return os.environ.get("TRTLLM_FLUX3_COND_ATTN_NFO", "1").strip() not in ("", "0")


def sm120_attn_nvfp4_out_weight(block: nn.Module) -> torch.Tensor | None:
    """attn_out's NVFP4 weight when the block's FMHA may write attn_out's activation (the
    fp8-direct side-gate_up call), else None. Callers must not read the BF16 attention output."""
    if not getattr(block, "_sm120_attn_nvfp4_out", False):
        return None
    return block.attn_out._flux3_primsts_nvfp4.weight


def _side_gate_up_mode() -> int:
    # Default 3: issued after the attention quantize (K2), right before FMHA, on a dedicated stream
    # (idea 161 STEP 0, job 1232: -22.4 us/block weighted vs 0; mode 2 -15.0, mode 1 -5.3).
    mode = int(os.environ.get("TRTLLM_FLUX3_SIDE_GATE_UP", "3"))
    if mode not in (0, 1, 2, 3):
        raise ValueError(f"TRTLLM_FLUX3_SIDE_GATE_UP must be 0-3, got {mode}")
    return mode


def sm120_side_gate_up_args(block: nn.Module, mlp_input: PrimsTSInput) -> tuple | None:
    """The gate_up operands of fp8_attention_side_gate_up for an NVFP4 tail-pair block (the same
    operands sm120_tail_pair_forward passes nvfp4_swiglu_prepared), or None."""
    if not getattr(block, "_sm120_side_gate_up", 0):
        return None
    up = getattr(block.ff.gate_up_proj, "_flux3_primsts_nvfp4", None)
    down = getattr(block.ff.down_proj, "_flux3_primsts_nvfp4", None)
    if up is None or down is None or not isinstance(mlp_input, NVFP4Activation):
        return None
    original = original_input(mlp_input)
    if not up.can_run(original):
        return None
    data, scales = up.activation(mlp_input)
    return (
        data,
        up.weight,
        scales,
        up.scale,
        down.inverse_scale,
        original,
        mlp_input.quantized_by,
        block._sm120_side_gate_up,
    )


def sm120_tail_pair_forward(
    block: nn.Module,
    attn: torch.Tensor,
    mlp_input: PrimsTSInput,
    gate_up: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(attn_out(attn), ff(mlp_input)) as 2D [M, N] tensors, with down and attn_out run as one
    two-stream op, or None (caller runs the sequential path). Only for blocks whose
    _sm120_tail_pair flag is set. gate_up runs exactly as in _flux3_gated_mlp_forward, unless
    ``gate_up`` (packed, scales) was already computed beside attention.
    """
    if isinstance(attn, NVFP4AttnOut):
        # The FMHA already wrote attn_out's NVFP4 activation (sm120_attn_nvfp4_out_weight).
        from .flux3_primsts import nvfp4_out_pair_prequant

        out = block.attn_out._flux3_primsts_nvfp4
        down = block.ff.down_proj._flux3_primsts_nvfp4
        packed, output_scales = gate_up
        return nvfp4_out_pair_prequant(
            packed,
            down.weight,
            output_scales,
            down.scale,
            attn.data.reshape(-1, attn.data.shape[-1]),
            attn.scales,
            out.weight,
            out.scale,
        )
    original = original_input(mlp_input)
    attn_2d = attn.reshape(-1, attn.shape[-1]).contiguous()
    out = getattr(block.attn_out, "_flux3_primsts_nvfp4", None)
    if out is not None:
        up = block.ff.gate_up_proj._flux3_primsts_nvfp4
        down = block.ff.down_proj._flux3_primsts_nvfp4
        if gate_up is not None:
            packed, output_scales = gate_up
            if not out.can_run(attn):
                y = nvfp4_linear_prepared(packed, down.weight, output_scales, down.scale)
                return out(attn).reshape(-1, out.weight.shape[0]), y
            return nvfp4_out_pair(
                packed, down.weight, output_scales, down.scale, attn_2d, out.weight, out.scale
            )
        if not (up.can_run(original) and out.can_run(attn)):
            return None
        data, scales = up.activation(mlp_input)
        packed, output_scales = nvfp4_swiglu_prepared(
            data,
            up.weight,
            scales,
            up.scale,
            down.inverse_scale,
            original,
            mlp_input.quantized_by if isinstance(mlp_input, NVFP4Activation) else None,
        )
        return nvfp4_out_pair(
            packed, down.weight, output_scales, down.scale, attn_2d, out.weight, out.scale
        )
    out = block.attn_out._flux3_primsts_fp8
    up = block.ff.gate_up_proj._flux3_primsts_fp8
    down = block.ff.down_proj._flux3_primsts_fp8
    if not (up.can_run(original) and out.can_run(attn)):
        return None
    if _FP8R_FUSED_SWIGLU_QUANT:
        # Fused SwiGLU + E4M3 quantize (bitwise equal to silu_and_mul -> quantize_fp8r) before
        # the fork; down's GEMM then takes the quantized hidden directly.
        if up.weight.shape[0] != 2 * down.weight.shape[1]:
            return None
        data, scales = up.activation(mlp_input)
        hidden, hidden_scales = swiglu_fp8r_quant(up.gemm(data, scales))
        if hidden.shape[-1] != down.weight.shape[1]:
            return None
        return fp8r_out_pair_q(
            hidden, hidden_scales, down.weight, down.scale, attn_2d, out.weight, out.scale
        )
    hidden = up(mlp_input)
    if not down.can_run(hidden):
        return None
    return fp8r_out_pair(
        hidden.reshape(-1, hidden.shape[-1]),
        down.weight,
        down.scale,
        attn_2d,
        out.weight,
        out.scale,
    )


def sm120_tail_pair_residual_forward(
    block: nn.Module,
    attn: torch.Tensor,
    mlp_input: PrimsTSInput,
    gate_up: tuple[torch.Tensor, torch.Tensor] | None,
    x: torch.Tensor,
    gate: torch.Tensor,
    gate_index: torch.Tensor | None,
) -> torch.Tensor | None:
    """``x + gate[gate_index] * (attn_out(attn) + ff(mlp_input))`` (shaped like x) as ONE LPT
    pair launch with the gated residual in attn_out's epilogue
    (flux3_sm120_lpt_pair_residual.py), or None (caller runs sm120_tail_pair_forward + the
    Inductor residual). NVFP4 tail-pair blocks only; gate is a [1, R, D] row table with a
    [1, S, 1] bool/uint8 row index, or a [1, 1, D] row without one.
    """
    if not getattr(block, "_sm120_pair_residual", False):
        return None
    from . import flux3_sm120_lpt_pair_residual as lptr

    out = block.attn_out._flux3_primsts_nvfp4
    width = out.weight.shape[0]
    rows = x.numel() // width if x.shape[-1] == width else -1
    if rows <= 0 or x.dtype != torch.bfloat16 or not lptr.gate_form(gate, gate_index, rows, width):
        return None
    down = block.ff.down_proj._flux3_primsts_nvfp4
    if isinstance(attn, NVFP4AttnOut):
        packed, output_scales = gate_up
        data = attn.data.reshape(-1, attn.data.shape[-1])
        scales = attn.scales
    else:
        attn_rows = attn.numel() // attn.shape[-1]
        if attn_rows != rows or not out.can_run(attn):
            return None
        if gate_up is not None:
            packed, output_scales = gate_up
        else:
            up = block.ff.gate_up_proj._flux3_primsts_nvfp4
            original = original_input(mlp_input)
            if not up.can_run(original):
                return None
            act, act_scales = up.activation(mlp_input)
            packed, output_scales = nvfp4_swiglu_prepared(
                act,
                up.weight,
                act_scales,
                up.scale,
                down.inverse_scale,
                original,
                mlp_input.quantized_by if isinstance(mlp_input, NVFP4Activation) else None,
            )
        data = attn.reshape(-1, attn.shape[-1]).contiguous()
        scales = None
    if data.shape[0] != rows or packed.shape[0] != rows:
        return None
    y = lptr.nvfp4_out_pair_residual(
        packed,
        down.weight,
        output_scales,
        down.scale,
        data,
        scales,
        out.weight,
        out.scale,
        x,
        gate,
        gate_index,
    )
    return y.reshape(x.shape)


def _tail_one_launch_gate(
    residual: torch.Tensor, gate: torch.Tensor, select: torch.Tensor | None, n: int
) -> int | None:
    """gate_mode of nvfp4_tail_one_launch for these layouts, or None (keep main's path)."""
    if (
        residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or residual.shape[-1] != n
        or gate.dtype != torch.bfloat16
        or not gate.is_contiguous()
        or gate.shape[-1] != n
    ):
        return None
    rows = residual.numel() // n
    if select is not None:
        # [1, K, N] table + [1, S, 1] per-row index (bool: True picks row 1, as torch.where).
        if (
            gate.dim() != 3
            or gate.shape[0] != 1
            or select.dtype not in (torch.bool, torch.uint8)
            or not select.is_contiguous()
            or select.numel() != rows
            or residual.dim() != 3
            or residual.shape[0] != 1
        ):
            return None
        return 1
    if gate.numel() == n:
        return 0
    if gate.shape == residual.shape:
        return 2
    return None


def sm120_tail_one_launch_forward(
    block: nn.Module,
    attn: torch.Tensor,
    mlp_input: PrimsTSInput,
    gate_up: tuple[torch.Tensor, torch.Tensor] | None,
    residual: torch.Tensor,
    gate: torch.Tensor,
    select: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """residual + gate[select] * (attn_out(attn) + ff(mlp_input)) as the one-launch SM120 block
    tail (nvfp4_tail_one_launch: two accumulators per output tile, the residual in the epilogue),
    bitwise equal to sm120_tail_pair_forward + the block's residual pointwise, or None (caller
    runs main's path). Only for blocks whose _sm120_tail_one_launch flag is set and tails in its
    static M-range bucket (default M >= 24576); gate forms: per-token [1, S, D], [1, K, D] table +
    per-row bool/uint8 index, [1, 1, D] row. Every check runs before any work is issued."""
    if not getattr(block, "_sm120_tail_one_launch", False):
        return None
    out = getattr(block.attn_out, "_flux3_primsts_nvfp4", None)
    down = getattr(block.ff.down_proj, "_flux3_primsts_nvfp4", None)
    if out is None or down is None:
        return None
    n = out.weight.shape[0]
    gate_mode = _tail_one_launch_gate(residual, gate, select, n)
    if gate_mode is None or down.weight.shape[0] != n:
        return None
    rows = residual.numel() // n
    if rows < getattr(block, "_sm120_tail_one_launch_min_m", 1 << 30):
        return None  # outside this route's M-range bucket: main's tail path
    from .flux3_primsts import nvfp4_tail_one_launch

    if isinstance(attn, NVFP4AttnOut):
        # The FMHA already wrote attn_out's NVFP4 activation (sm120_attn_nvfp4_out_weight).
        packed, output_scales = gate_up
        attn_data = attn.data.reshape(-1, attn.data.shape[-1])
        if attn_data.shape[0] != rows or packed.shape[0] != rows:
            return None
        attn_scales = attn.scales
    else:
        if attn.numel() // attn.shape[-1] != rows or not out.can_run(attn):
            return None
        if gate_up is not None:
            packed, output_scales = gate_up
            if packed.shape[0] != rows:
                return None
        else:
            up = block.ff.gate_up_proj._flux3_primsts_nvfp4
            original = original_input(mlp_input)
            if original.shape[0] != rows or not up.can_run(original):
                return None
            data, scales = up.activation(mlp_input)
            packed, output_scales = nvfp4_swiglu_prepared(
                data,
                up.weight,
                scales,
                up.scale,
                down.inverse_scale,
                original,
                mlp_input.quantized_by if isinstance(mlp_input, NVFP4Activation) else None,
            )
        attn_2d = attn.reshape(-1, attn.shape[-1]).contiguous()
        attn_data, attn_scales = nvfp4_quantize_prepared(attn_2d, out.weight)
    return nvfp4_tail_one_launch(
        packed,
        down.weight,
        output_scales,
        down.scale,
        attn_data,
        attn_scales,
        out.weight,
        out.scale,
        residual,
        gate,
        select,
        gate_mode,
    )


def linear_forward(layer: Linear, x: PrimsTSInput) -> torch.Tensor:
    adapter = getattr(layer, "_flux3_primsts_fp8", None)
    if adapter is None:
        adapter = getattr(layer, "_flux3_primsts_nvfp4", None)
    original = original_input(x)
    if adapter is not None and adapter.can_run(original):
        return adapter(x)
    return layer(original if isinstance(x, NVFP4Activation) else x)


def rope_cos_sin_table(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The fused QK-norm+RoPE ops' table: [rows, 128], the cos half then the sin
    half (every other column of the repeat-interleaved head_dim cos/sin)."""
    return torch.cat((cos.reshape(-1, 128)[:, ::2], sin.reshape(-1, 128)[:, ::2]), dim=-1)


def qkv_forward(
    attn: nn.Module,
    x: PrimsTSInput,
    freqs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return Q, K and V after the fused projection, or None when the input
    requires native dispatch. Each is ``[..., heads * head_dim]`` and the tuned
    configs make them contiguous. ``freqs`` is ``(cos, sin)`` or ``(cos, sin,
    cos_sin)`` with ``cos_sin`` the per-request FP32 rope_cos_sin_table, which
    then is not rebuilt here."""
    adapter = getattr(attn, "_flux3_primsts_nvfp4_qkv", None)
    if adapter is None:
        if getattr(attn, "_flux3_sm120_fp8r_qknorm_rope", False):
            return _sm120_fp8r_qkv_forward(attn, x, original_input(x), freqs)
        if not getattr(attn, "_primsts_fp8_qkv_rope", False):
            return None
        adapter = attn.qkv_proj._flux3_primsts_fp8
    original = original_input(x)
    cos, sin = freqs[0], freqs[1]
    if not adapter.can_run(original) or cos.numel() != original.numel() // original.shape[-1] * 128:
        return None
    if sin.shape != cos.shape:
        return None
    data, scales = adapter.activation(x)
    cos_sin = freqs[2] if len(freqs) > 2 else rope_cos_sin_table(cos, sin)
    args = (
        data,
        adapter.weight,
        scales,
        adapter.scale,
        attn.norm_q.weight.contiguous(),
        attn.norm_k.weight.contiguous(),
        cos_sin.float().contiguous(),
    )
    if isinstance(adapter, PreparedNVFP4):
        output = nvfp4_qkv_prepared(
            *args,
            adapter.qkv_scale,
            original,
            x.quantized_by if isinstance(x, NVFP4Activation) else None,
        )
        planar = bool(getattr(attn, "_flux3_primsts_nvfp4_qkv_planar", False))
    else:
        output = fp8_qkv_qknorm_rope(*args, None)
        planar = False
    planes = qkv_planes(
        output,
        data,
        adapter.weight,
        128,
        qkv_scale=getattr(adapter, "qkv_scale", None) is not None,
        planar=planar,
    )
    return tuple(t.reshape(*original.shape[:-1], t.shape[-1]) for t in planes)


# NVFP4 QKV straight to E4M3 for the SM120 per-tensor FP8 attention backend: one op runs main's
# planar QKV GEMMs and writes E4M3 q/k/v + scales directly (no BF16 norm+RoPE q/k planes, no
# fp8_sm12x K1/K2), then the same fmha_v2 call. Bitwise equal to main's path (qkv_forward ->
# _attend -> FlashInferAttention FP8). Module-level so one process can compare (Dynamo guards on it);
# TRTLLM_FLUX3_QKV_FP8_DIRECT=0 keeps main's path.
FLUX3_QKV_FP8_DIRECT = os.environ.get("TRTLLM_FLUX3_QKV_FP8_DIRECT", "1").strip() != "0"
# Last single block (keep_rows > 0): its NVFP4-out G3 FMHA computes only the Q-row prefix
# [0, keep_rows) over all S K/V rows (fmha_v2_flux3 q_rows launch; i2i S=32848 grid.x 172 -> 86,
# t2i S=16464 keeps 86). Bitwise the first keep_rows rows of the full launch, which are the only
# rows the keep_rows tail reads. Module-level (Dynamo guards on it);
# TRTLLM_FLUX3_LAST_BLOCK_QPREFIX=0 keeps landing 18's full-row FMHA.
FLUX3_LAST_BLOCK_QPREFIX = os.environ.get("TRTLLM_FLUX3_LAST_BLOCK_QPREFIX", "1").strip() != "0"


def _fp8_direct_backend(attn: nn.Module):
    """The attention's FlashInfer SM12x per-tensor FP8 backend, or None."""
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention

    backend = getattr(attn, "attn", None)
    if type(backend) is not FlashInferAttention or not getattr(backend, "_fp8_sm12x", False):
        return None
    if backend.head_dim != 128 or backend.num_heads != backend.num_kv_heads:
        return None
    return backend


def qkv_fp8_attention_forward(
    attn: nn.Module,
    x: PrimsTSInput,
    freqs: tuple[torch.Tensor, ...],
    full_sequence: bool,
    side_gate_up: tuple | None = None,
    attn_nvfp4_out: torch.Tensor | None = None,
    v_live: bool = True,
    keep_rows: int = 0,
) -> tuple | None:
    """(attention output [B, S, H*D] BF16, planar BF16 QKV GEMM output [M, 3W]) or None.

    keep_rows > 0 (last single block, _Flux3LastRowsSingleStreamBlock; a multiple of 128 < S):
    only the side-gate_up NVFP4-out call at B = 1 is covered. QKV and the FMHA run on all S rows as
    in main; the side gate_up runs on the first keep_rows rows of the shared NVFP4 activation, and
    the result is (NVFP4AttnOut of the first keep_rows rows (views), gemm, (packed, scales) of
    keep_rows rows), every value bitwise the first rows of main's. None (before launching anything)
    for any other call.

    Only the GEMM output's v plane is defined on return (its q/k lines may be dropped from L2,
    flux3_sm120_qknorm_rope_fp8.L2DISCARD); v_live=False (the caller never reads v: PAG off)
    lets v's lines go too.

    With side_gate_up (sm120_side_gate_up_args) the result gets a third item, the block's packed
    gate_up (packed, scales), forked onto a side stream right before FMHA
    (flux3_primsts.fp8_prequant_attention_side_gate_up).

    Covers exactly the calls where main runs: qkv_forward's PreparedNVFP4 per-plane in-place planar
    route, then one FlashInferAttention FP8 (non-causal, no LSE) call on the whole [B, S] q/k/v
    (``full_sequence``: sample_bounds None, or one full-range segment at B=1 on the _attend fast
    path) with S >= the backend's FP8 minimum. Anything else returns None (main's path).
    Like qkv_forward it never reads the BF16 values of ``x`` (the fused adaLN head may not store
    them); `original` goes to _prepared_activation only, as in nvfp4_qkv_prepared.
    """
    if not FLUX3_QKV_FP8_DIRECT or not full_sequence:
        return None
    if keep_rows and (side_gate_up is None or attn_nvfp4_out is None or keep_rows % 128):
        return None
    adapter = getattr(attn, "_flux3_primsts_nvfp4_qkv", None)
    if not isinstance(adapter, PreparedNVFP4) or adapter.qkv_scale is None:
        return None
    if not getattr(attn, "_flux3_primsts_nvfp4_qkv_planar", False):
        return None
    backend = _fp8_direct_backend(attn)
    if backend is None:
        return None
    original = original_input(x)
    if original.dim() != 3:
        return None
    batch, seq_len = original.shape[0], original.shape[1]
    cos, sin = freqs[0], freqs[1]
    if not adapter.can_run(original) or cos.numel() != original.numel() // original.shape[-1] * 128:
        return None
    if sin.shape != cos.shape:
        return None
    if seq_len < backend._fp8_min_seq_len or seq_len <= 0:
        return None
    if keep_rows and (batch != 1 or keep_rows >= seq_len):
        return None
    q_dim, k_dim = attn.norm_q.weight.numel(), attn.norm_k.weight.numel()
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
        nvfp4_qkv_rope_fp8_prepared,
        sm120_qkv_fp8_covers,
    )

    columns = adapter.weight.shape[0]
    if not sm120_qkv_fp8_covers(original.device, columns, q_dim, k_dim):
        return None
    heads = columns // 3 // 128
    if heads != backend.num_heads:
        return None
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (
        visual_gen_fp8_attention_sm120_prequant,
    )

    data, scales = adapter.activation(x)
    cos_sin = freqs[2] if len(freqs) > 2 else rope_cos_sin_table(cos, sin)
    qkv8, fp8_scales, gemm = nvfp4_qkv_rope_fp8_prepared(
        data,
        adapter.weight,
        scales,
        adapter.scale,
        attn.norm_q.weight.contiguous(),
        attn.norm_k.weight.contiguous(),
        cos_sin.float().contiguous(),
        adapter.qkv_scale,
        original,
        x.quantized_by if isinstance(x, NVFP4Activation) else None,
        backend.scale,
        bool(v_live),
        # False (i2i reference-stream blocks): dynamic Q/K amax, as main's generic route there.
        bool(getattr(attn, "_flux3_static_qk", True)),
    )
    if side_gate_up is not None:
        from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
            fp8_prequant_attention_side_gate_up,
        )

        up_data, up_weight, up_scales, up_weight_scale, inverse_scale, up_original, source, mode = (
            side_gate_up
        )
        if keep_rows:
            from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
                fp8_prequant_attention_nvfp4_out_side_gate_up,
            )

            # gate_up on the first keep_rows rows of the shared NVFP4 activation: its data rows,
            # the leading keep_rows / 128 row blocks of the flat 128x4 scales, and the BF16 rows.
            n = keep_rows
            up_cols = (up_data.shape[-1] * 2 // 16 + 3) // 4 * 4
            fp4, sf, packed, packed_scales = fp8_prequant_attention_nvfp4_out_side_gate_up(
                qkv8, fp8_scales, batch, heads, up_data[:n], up_weight,
                up_scales.reshape(-1)[: n * up_cols], up_weight_scale, inverse_scale,
                up_original.reshape(-1, up_original.shape[-1])[:n], source, mode,
                attn_nvfp4_out, n if FLUX3_LAST_BLOCK_QPREFIX else 0,
            )
            # The attention output's first n rows: fp4 rows and the leading n / 128 row blocks.
            out_cols = (fp4.shape[-1] * 2 // 16 + 3) // 4 * 4
            return NVFP4AttnOut(fp4[:, :n], sf[: n * out_cols]), gemm, (packed, packed_scales)
        if attn_nvfp4_out is not None and batch == 1:
            # attn_out's NVFP4 activation straight from the FMHA epilogue (sm120_attn_nvfp4_out).
            from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
                fp8_prequant_attention_nvfp4_out_side_gate_up,
            )

            fp4, sf, packed, packed_scales = fp8_prequant_attention_nvfp4_out_side_gate_up(
                qkv8, fp8_scales, batch, heads, up_data, up_weight, up_scales, up_weight_scale,
                inverse_scale, up_original, source, mode, attn_nvfp4_out,
            )
            return NVFP4AttnOut(fp4, sf), gemm, (packed, packed_scales)
        output, packed, packed_scales = fp8_prequant_attention_side_gate_up(
            qkv8, fp8_scales, batch, heads, up_data, up_weight, up_scales, up_weight_scale,
            inverse_scale, up_original, source, mode,
        )
        return output.flatten(2), gemm, (packed, packed_scales)
    output = visual_gen_fp8_attention_sm120_prequant(qkv8, fp8_scales, batch, heads)
    return output.flatten(2), gemm
