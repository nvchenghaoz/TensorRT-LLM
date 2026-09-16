# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 projection storage with batched GEMM selection."""

from typing import ClassVar

import torch
import torch.nn.functional as F

from ...autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...custom_ops.lossless_bf16_gemm import lossless_bf16_gemm
from ...utils import is_torch_compiling
from ..linear import Linear, UnquantizedLinearMethod
from ..low_m_gemm import LOW_M_GEMM_ACTIVE


def _pack_chunk(
    weight: torch.Tensor, reassociated: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = weight.view(torch.int16).int().reshape(-1, 128) & 65535
    exponent = (raw >> 7) & 255
    base = exponent.max(1).values
    delta = base[:, None] - exponent
    escaped = delta >= 7
    counts = escaped.sum(1).long()
    offsets = counts.cumsum(0) - counts
    exceptions = exponent[escaped].to(torch.uint8).contiguous()
    codes = delta.clamp_max(7)
    decoded_exponents = base[:, None] - codes
    decoded_exponents[escaped] = exceptions.int()
    decoded = (raw & 0x807F) | (decoded_exponents << 7)
    if not torch.equal(decoded, raw):
        raise RuntimeError("Lossless BF16 packing changed a checkpoint weight")
    if reassociated:
        # A zero-based uint32 exception offset and three byte-sized
        # prefixes locate exceptions within each 32-value group.
        groups = escaped.reshape(-1, 4, 32).sum(2)
        prefixes = groups.cumsum(1) - groups
        metadata = (
            base.long()
            | (offsets << 8)
            | (prefixes[:, 1] << 40)
            | (prefixes[:, 2] << 48)
            | (prefixes[:, 3] << 56)
        )
        codes = codes.reshape(-1, 16, 8)
        pairs = sum(codes[:, :, j] << (j * 3) for j in range(8))
    else:
        metadata = base.long() | torch.where(counts > 0, (offsets + 1) << 8, 0)
        codes = codes.reshape(-1, 4, 32).transpose(1, 2)
        words = sum(codes[:, :, j] << (j * 3) for j in range(4))
        pairs = words[:, ::2] | (words[:, 1::2] << 12)
    packed_codes = (
        torch.stack((pairs & 255, (pairs >> 8) & 255, (pairs >> 16) & 255), dim=-1)
        .to(torch.uint8)
        .reshape(-1, 48)
    )
    signs_mantissas = ((raw & 127) | ((raw >> 8) & 128)).to(torch.uint8)
    if not reassociated:
        signs_mantissas = signs_mantissas.reshape(-1, 4, 32).transpose(1, 2).reshape(-1, 128)
    data = torch.cat((signs_mantissas, packed_codes), dim=-1).contiguous()
    return data, metadata, exceptions


@torch.no_grad()
def pack_lossless_bf16_weights(
    weight: torch.Tensor, *, reassociated: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache an exact layout in bounded chunks without modifying the parameter.

    ``reassociated`` selects adjacent-value attention storage; the default
    preserves the vocabulary head's K-stride-32 summation order.
    """
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
        raise ValueError("weight must be a contiguous BF16 matrix")
    if weight.shape[0] == 0 or weight.shape[1] == 0 or weight.shape[1] % 128:
        raise ValueError("weight dimensions must be positive with K divisible by 128")
    data_chunks, metadata_chunks, exception_chunks = [], [], []
    exception_offset = 0
    for start in range(0, weight.shape[0], 1024):
        data, metadata, exceptions = _pack_chunk(weight[start : start + 1024], reassociated)
        if reassociated:
            # Zero-based headers must advance even when their local offset is zero.
            adjusted = metadata + (exception_offset << 8)
        else:
            adjusted = torch.where(
                (metadata >> 8) != 0, metadata + (exception_offset << 8), metadata
            )
        data_chunks.append(data)
        metadata_chunks.append(adjusted)
        exception_chunks.append(exceptions)
        exception_offset += exceptions.numel()
    if exception_offset >= (1 << 31):
        raise ValueError("too many lossless BF16 exceptions")
    return torch.cat(data_chunks), torch.cat(metadata_chunks), torch.cat(exception_chunks)


class _ProjectionRunner(TunableRunner):
    def __init__(self, packed: bool, reassociated: bool) -> None:
        self.packed = packed
        self.reassociated = reassociated

    def unique_id(self) -> tuple[bool, bool]:
        return (self.packed, self.reassociated)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile, **kwargs: object
    ) -> list[int]:
        return [0]

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        x, weight, data, metadata, exceptions = inputs
        if self.packed:
            return lossless_bf16_gemm(
                x, data, metadata, exceptions, weight.shape[0], self.reassociated
            )
        return F.linear(x, weight)


class _PackedBF16LinearMethod(UnquantizedLinearMethod):
    supports_nccl_symmetric_memory_window_output: ClassVar[bool] = False

    def __init__(self, original: UnquantizedLinearMethod, reassociated: bool) -> None:
        self.original = original
        self.reassociated = reassociated
        self.runners = [
            _ProjectionRunner(False, reassociated),
            _ProjectionRunner(True, reassociated),
        ]
        # Warmup runs the maximum batch; profile the decode sizes before graph
        # capture too. Untuned larger shapes keep the original GEMM. Use events
        # for timing: a profiling graph would retain many large logit tensors.
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=0,
                    dim_idx=0,
                    gen_tuning_buckets=tuple(1 << i for i in range(9)),
                    map_to_tuning_buckets=lambda m: 1 << max(0, (m - 1).bit_length()),
                ),
            ),
            use_cold_l2_cache=True,
            use_cuda_graph=False,
        )

    def cache_derived_state(self, module: Linear) -> None:
        self.original.cache_derived_state(module)
        data, metadata, exceptions = pack_lossless_bf16_weights(
            module.weight.contiguous(), reassociated=self.reassociated
        )
        module.register_buffer("_packed_bf16_data", data, persistent=False)
        module.register_buffer("_packed_bf16_metadata", metadata, persistent=False)
        module.register_buffer("_packed_bf16_exceptions", exceptions, persistent=False)

    def transform_weights(self, module: Linear) -> None:
        self.original.transform_weights(module)
        self.cache_derived_state(module)

    def apply(self, module: Linear, input: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if (
            not input.is_cuda
            or input.dtype != torch.bfloat16
            or bias is not None
            or torch.is_grad_enabled()
            or is_torch_compiling()
            or module.use_custom_cublas_mm
            or module.use_cute_dsl_bf16_gemm
            or LOW_M_GEMM_ACTIVE
        ):
            return self.original.apply(module, input, bias)
        x = input.reshape(-1, input.shape[-1]).contiguous()
        inputs = [
            x,
            module.weight,
            module._packed_bf16_data,
            module._packed_bf16_metadata,
            module._packed_bf16_exceptions,
        ]
        runner, tactic = AutoTuner.get().choose_one(
            "trtllm::packed_bf16_projection_cute_dsl", self.runners, self.tuning_config, inputs
        )
        return runner(inputs, tactic=tactic).reshape(*input.shape[:-1], module.weight.shape[0])


def configure_packed_bf16(module: Linear, *, reassociated: bool) -> None:
    """Attach optional storage after weight loading, including TP projection shards."""
    if (
        type(module.quant_method) is not UnquantizedLinearMethod
        or not module.weight.is_cuda
        or module.weight.dtype != torch.bfloat16
        or module.weight.shape[1] % 128
        or module.bias is not None
        or module.use_custom_cublas_mm
        or module.use_cute_dsl_bf16_gemm
        or LOW_M_GEMM_ACTIVE
    ):
        return
    module.quant_method = _PackedBF16LinearMethod(module.quant_method, reassociated)
    module.quant_method.cache_derived_state(module)
