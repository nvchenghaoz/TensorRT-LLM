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
    """TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2 (read at weight preparation): 1 (default) = lane-local
    epilogue: prepare permutes the packed gate_up weight rows (permute_weight_v2w) and those weights
    run the FLUX3_EPI_V2=2 modules (_ev2w / _dyn_ev2w; same MMA permutation as EPI_VEC, bitwise
    equal outputs); 0 = unpermuted weights on the EPI_VEC modules (_ev / _dyn_ev) byte for byte.
    Needs EPI_VEC; ignored when EPI_VEC is off."""
    value = os.environ.get("TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2", "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2 must be 0 or 1, got {value!r}")
    return value == "1"


@functools.cache
def _build(suffix: str, flags: tuple[str, ...]):
    return gen_spec(suffix, flags).build_and_load()


def get_module_static(vec: bool | None = None, v2: bool | None = None):
    """StaticPersistentScheduler build (main's module, same name, when vec is off). v2: the module
    for v2w-permuted weights."""
    if vec is None:
        vec = epi_vec()
    if not vec:
        return _build("", ())
    if v2 is None:
        v2 = epi_v2()
    if v2:
        return _build("_ev2w", ("-DFLUX3_EPI_VEC=1", "-DFLUX3_EPI_V2=2"))
    return _build("_ev", ("-DFLUX3_EPI_VEC=1",))


def get_module_dynamic(vec: bool | None = None, v2: bool | None = None):
    """CLC dynamic persistent scheduler build (cutlass::gemm::PersistentScheduler on Sm120). v2: the
    module for v2w-permuted weights."""
    if vec is None:
        vec = epi_vec()
    if not vec:
        return _build("_dyn", ("-DFLUX3_TILESCHED=1",))
    if v2 is None:
        v2 = epi_v2()
    if v2:
        return _build("_dyn_ev2w", ("-DFLUX3_TILESCHED=1", "-DFLUX3_EPI_VEC=1", "-DFLUX3_EPI_V2=2"))
    return _build("_dyn_ev", ("-DFLUX3_TILESCHED=1", "-DFLUX3_EPI_VEC=1"))


def get_module(v2: bool | None = None):
    """The module selected by TRTLLM_FLUX3_SM120_SWIGLU_SCHED; only that one is built. v2=None:
    the module prepared weights use under the current env (v2w when EPI_VEC and EPI_V2 are on)."""
    if v2 is None:
        v2 = v2w_prepare_enabled()
    return get_module_dynamic(None, v2) if tile_sched() == "dynamic" else get_module_static(None, v2)


def v2w_prepare_enabled() -> bool:
    """Whether weight preparation permutes packed gate_up weights for the v2w modules."""
    return enabled() and epi_vec() and epi_v2()


def weight_is_v2w(weight: torch.Tensor) -> bool:
    """True when weight is a prepared gate_up weight that permute_weight_v2w reordered (flag on its
    registered NVFP4 owner, so offload moves are followed)."""
    from tensorrt_llm._torch.visual_gen.models.flux3.nvfp4_metadata import find_nvfp4_owner

    owner = find_nvfp4_owner(weight)
    return bool(getattr(owner, "_flux3_v2w", False)) if owner is not None else False


def maybe_permute_for_v2w(weight: torch.Tensor, weight_scale: torch.Tensor):
    """(weight, scale, permuted) for a packed gate_up weight at preparation: v2w-permuted when
    v2w_prepare_enabled() and the shape is one the CUTLASS module serves."""
    columns, half_k = weight.shape
    if not (v2w_prepare_enabled() and columns % 128 == 0 and (half_k * 2) % 128 == 0
            and (columns // 2 // 16) % 4 == 0):
        return weight, weight_scale, False
    w, s = permute_weight_v2w(weight, weight_scale)
    return w, s, True


def v2w_row_permutation(columns: int, device) -> torch.Tensor:
    """Source row of every packed gate_up weight row for FLUX3_EPI_V2=2 (served MMA permutation):
    within each 64-row group, row 2*s + e (pair slot s = q + 4r + 16nb, e = value/gate) takes
    the interleaved row 2*o + e of output o = 8q + 4nb + r."""
    assert columns % 64 == 0
    p = torch.arange(64, device=device)
    e, s = p & 1, p >> 1
    q, r, nb = s & 3, (s >> 2) & 3, s >> 4
    local = 2 * (8 * q + 4 * nb + r) + e
    rows = torch.arange(columns, device=device)
    return (rows // 64) * 64 + local[rows % 64]


def _sf_index(rows: int, kb: int, device) -> torch.Tensor:
    """[rows, kb] -> byte offset in the 128x4-swizzled scale layout (rows % 128 == 0, kb % 4 == 0)."""
    n = torch.arange(rows, device=device).unsqueeze(1)
    k = torch.arange(kb, device=device).unsqueeze(0)
    return (k & 3) + (k >> 2) * 512 + (n & 31) * 16 + ((n & 127) >> 5) * 4 + (n >> 7) * 128 * kb


def permute_weight_v2w(weight: torch.Tensor, weight_scale: torch.Tensor):
    """(weight, 128x4-swizzled scale) with rows permuted by v2w_row_permutation. Exact (bytes)."""
    w = _u8(weight)
    columns, half_k = w.shape
    kb = half_k * 2 // 16
    assert columns % 128 == 0 and kb % 4 == 0
    perm = v2w_row_permutation(columns, w.device)
    sf = _u8(weight_scale).reshape(-1)
    assert sf.numel() == columns * kb
    idx = _sf_index(columns, kb, w.device)
    sf_rows = sf[idx][perm]
    out_sf = torch.empty_like(sf)
    out_sf[idx] = sf_rows
    return w[perm].contiguous().view(weight.dtype), out_sf.view(weight_scale.dtype).reshape(
        weight_scale.shape)


def unpermute_weight_v2w(weight: torch.Tensor, weight_scale: torch.Tensor):
    """Inverse of permute_weight_v2w (for non-v2w consumers of a permuted weight). Exact (bytes)."""
    w = _u8(weight)
    columns, half_k = w.shape
    kb = half_k * 2 // 16
    perm = v2w_row_permutation(columns, w.device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(columns, device=w.device)
    sf = _u8(weight_scale).reshape(-1)
    idx = _sf_index(columns, kb, w.device)
    sf_rows = sf[idx][inv]
    out_sf = torch.empty_like(sf)
    out_sf[idx] = sf_rows
    return w[inv].contiguous().view(weight.dtype), out_sf.view(weight_scale.dtype).reshape(
        weight_scale.shape)


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
    if module is None:
        module = get_module(weight_is_v2w(weight))
    module.flux3_swiglu_fp4_gemm(
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
