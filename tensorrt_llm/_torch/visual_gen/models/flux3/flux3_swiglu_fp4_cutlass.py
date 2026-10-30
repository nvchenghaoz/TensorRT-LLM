# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 SM120 NVFP4 gate_up as a CUTLASS GEMM with a register SwiGLU -> NVFP4 epilogue.

Add-only FlashInfer JIT module (csrc/flux3_fp4_gemm_swiglu_fp4_sm120.cu, shipped through the
experiments/ascent-flashinfer overlay): the stock SM120 block-scaled mainloop, Pingpong schedule at
tile 128x128x128 (pinned, no autotune; the cooperative t2 256x128x128 build exposes the epilogue
after every tile and lost to the CuTe kernel), and a register epilogue that writes the packed FP4
bytes and the 128x4-swizzled E4M3 scales of SwiGLU(acc) directly. Output bytes and scales are
bitwise equal to flux3_sm120_nvfp4.run('packed') (same fp32 per-element PTX forms).

The env TRTLLM_FLUX3_SM120_SWIGLU_CUTLASS=0 routes nvfp4_swiglu_packed back to the CuTe kernel.
"""

import functools
import os

import torch

MODULE_NAME = "fp4_gemm_swiglu_fp4_sm120"
SOURCE_NAME = "flux3_fp4_gemm_swiglu_fp4_sm120.cu"


def enabled() -> bool:
    return os.environ.get("TRTLLM_FLUX3_SM120_SWIGLU_CUTLASS", "1") != "0"


def gen_spec(suffix: str = "", flags: tuple[str, ...] = ()):
    """JitSpec of the module; suffix/flags build diagnostic variants under another name."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    return gen_jit_spec(
        MODULE_NAME + suffix,
        [jit_env.FLASHINFER_CSRC_DIR / SOURCE_NAME],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1", *flags],
        extra_cflags=["-DFAST_BUILD"],
    )


@functools.cache
def tile_sched() -> str:
    """TRTLLM_FLUX3_SM120_SWIGLU_SCHED: dynamic (default, CLC work stealing) or static (main's)."""
    value = os.environ.get("TRTLLM_FLUX3_SM120_SWIGLU_SCHED", "dynamic").strip().lower()
    if value not in ("dynamic", "static"):
        raise ValueError(f"TRTLLM_FLUX3_SM120_SWIGLU_SCHED must be dynamic or static, got {value!r}")
    return value


@functools.cache
def epi_vec() -> bool:
    """TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC: 1 (default) = 16-B FP4 / u16 SF epilogue stores
    (FLUX3_EPI_VEC=1, separate JIT module names, bitwise equal); 0 = main's modules byte for byte."""
    value = os.environ.get("TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC", "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC must be 0 or 1, got {value!r}")
    return value == "1"


@functools.cache
def epi_v2() -> bool:
    """TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2: 1 (default) = lane-local epilogue (FLUX3_EPI_V2=1 on top of
    EPI_VEC; modules _ev2 / _dyn_ev2, bitwise equal); 0 = the EPI_VEC modules (_ev / _dyn_ev) byte
    for byte. Ignored when EPI_VEC is off."""
    value = os.environ.get("TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2", "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2 must be 0 or 1, got {value!r}")
    return value == "1"


@functools.cache
def _build(suffix: str, flags: tuple[str, ...]):
    return gen_spec(suffix, flags).build_and_load()


def get_module_static(vec: bool | None = None, v2: bool | None = None):
    """StaticPersistentScheduler build (main's module, same name, when vec is off)."""
    if vec is None:
        vec = epi_vec()
    if not vec:
        return _build("", ())
    if v2 is None:
        v2 = epi_v2()
    if v2:
        return _build("_ev2", ("-DFLUX3_EPI_VEC=1", "-DFLUX3_EPI_V2=1"))
    return _build("_ev", ("-DFLUX3_EPI_VEC=1",))


def get_module_dynamic(vec: bool | None = None, v2: bool | None = None):
    """CLC dynamic persistent scheduler build (cutlass::gemm::PersistentScheduler on Sm120)."""
    if vec is None:
        vec = epi_vec()
    if not vec:
        return _build("_dyn", ("-DFLUX3_TILESCHED=1",))
    if v2 is None:
        v2 = epi_v2()
    if v2:
        return _build("_dyn_ev2", ("-DFLUX3_TILESCHED=1", "-DFLUX3_EPI_VEC=1", "-DFLUX3_EPI_V2=1"))
    return _build("_dyn_ev", ("-DFLUX3_TILESCHED=1", "-DFLUX3_EPI_VEC=1"))


def get_module():
    """The module selected by TRTLLM_FLUX3_SM120_SWIGLU_SCHED; only that one is built."""
    return get_module_dynamic() if tile_sched() == "dynamic" else get_module_static()


def _scale_size(rows: int, columns: int) -> int:
    return ((rows + 127) // 128) * 128 * ((columns // 16 + 3) // 4) * 4


def _u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def supported(activation: torch.Tensor, weight: torch.Tensor) -> bool:
    columns = weight.shape[0]
    return (
        activation.dim() == 2
        and weight.dim() == 2
        and columns % 128 == 0
        and (activation.shape[1] * 2) % 128 == 0
        and (columns // 2 // 16) % 4 == 0
    )


def allocate_outputs(
    rows: int, columns: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed FP4 [rows padded to 128, columns/4] u8, swizzled scales u8) for swiglu_packed."""
    padded_rows = ((rows + 127) // 128) * 128
    output = torch.empty((padded_rows, columns // 4), device=device, dtype=torch.uint8)
    output_scale = torch.empty(
        (_scale_size(padded_rows, columns // 2),), device=device, dtype=torch.uint8
    )
    return output, output_scale


def swiglu_packed(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_inverse_scale: torch.Tensor,
    module=None,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (FP4 bytes [rows, N/4] as a view of a 128-row padded buffer, scales) like run('packed').

    output/output_scale: optional preallocated buffers (allocate_outputs), e.g. made on the current
    stream before a side-stream launch so no allocation happens on the side stream.
    """
    rows = activation.shape[0]
    columns = weight.shape[0]
    padded_rows = ((rows + 127) // 128) * 128
    if output is None or output_scale is None:
        output, output_scale = allocate_outputs(rows, columns, activation.device)
    else:
        assert output.shape == (padded_rows, columns // 4) and output.dtype == torch.uint8
        assert output_scale.numel() == _scale_size(padded_rows, columns // 2)
    (module or get_module()).flux3_swiglu_fp4_gemm(
        _u8(activation.contiguous()),
        _u8(weight),
        _u8(activation_scale),
        _u8(weight_scale),
        alpha,
        output_inverse_scale,
        output,
        output_scale,
        padded_rows,
    )
    return output[:rows], output_scale
