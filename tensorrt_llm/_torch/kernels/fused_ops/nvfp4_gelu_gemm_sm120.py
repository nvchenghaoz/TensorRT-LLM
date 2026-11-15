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
"""SM120 NVFP4 gate_up GEMM with a gelu_tanh(gate)*up + NVFP4-quantize epilogue.

The fused kernel (FlashInfer JIT module ``nvfp4_gelu_gemm_sm120``) runs the
same CUTLASS SM120 block-scaled mainloop as TRT-LLM's FP4GemmRunner tactic 2
(256x128x128 CTA tile) and quantizes ``gelu_tanh(gate) * up`` for down_proj in
its epilogue, so the [M, 2I] bf16 gate_up output is never written. Its outputs
are byte-identical to ``nvfp4_gemm`` followed by ``gelu_tanh_mul_fp4_quant``.

It needs the gate_up weight rows (FP4 payload and the 128x4-swizzled block
scales) in a per-128-row-tile interleave: fused row ``128t + 32a + c`` holds
gate (``c < 16``) or up (``c >= 16``) of output column ``64t + 16a + c % 16``.
Each output column is still the same dot product, so a plain GEMM on the
permuted rows produces the same values in permuted columns; the unfused path
reads them back through ``gelu_tanh_mul_fp4_quant_permuted`` below, a copy of
the act kernel with only its gather addresses changed.
"""

import os
from typing import Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ...autotuner import AutoTuner
from ...custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
from .gelu_tanh_mul_fp4_quant import _launch_config

_ENV_KILL = "TRTLLM_GEMMA4_GATEUP_GELU_EPI"

_module = None
_module_error: Optional[str] = None


def gelu_gemm_epi_env_enabled() -> bool:
    return os.environ.get(_ENV_KILL, "1") != "0"


def gelu_gemm_epi_supported() -> bool:
    """SM120/121 only; the kill switch TRTLLM_GEMMA4_GATEUP_GELU_EPI=0 disables it."""
    if not gelu_gemm_epi_env_enabled():
        return False
    try:
        return get_sm_version() in (120, 121)
    except Exception:  # noqa: BLE001
        return False


def get_gelu_gemm_module():
    """Build (once) and return the FlashInfer JIT module, or None if unavailable."""
    global _module, _module_error
    if _module is not None or _module_error is not None:
        return _module
    try:
        from flashinfer.jit.gemm.core import gen_nvfp4_gelu_gemm_sm120_module

        _module = gen_nvfp4_gelu_gemm_sm120_module().build_and_load()
    except Exception as e:  # noqa: BLE001
        _module_error = f"{type(e).__name__}: {e}"
        logger.warning(f"nvfp4_gelu_gemm_sm120 unavailable, using the unfused path: {_module_error}")
        _module = None
    return _module


def gate_up_perm(n_total: int, device: torch.device) -> torch.Tensor:
    """perm[r] = source (unpermuted) gate_up row placed at fused row r."""
    assert n_total % 128 == 0
    inter = n_total // 2
    r = torch.arange(n_total, device=device, dtype=torch.int64)
    t, a, c = r // 128, (r % 128) // 32, r % 32
    j = 64 * t + 16 * a + (c % 16)
    return torch.where(c < 16, j, inter + j)


def _sf_offsets(rows: int, nvec: int, device: torch.device) -> torch.Tensor:
    nkt = (nvec + 3) // 4
    mm = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    kk = torch.arange(nvec, device=device, dtype=torch.int64)[None, :]
    return ((mm // 128) * (nkt * 512) + (kk // 4) * 512 + (mm % 32) * 16 +
            ((mm % 128) // 32) * 4 + (kk % 4)).reshape(-1)


@torch.no_grad()
def permute_gate_up_rows(weight: torch.Tensor, weight_scale: torch.Tensor,
                         in_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return gate_up FP4 rows and swizzled SF rows in the fused-epilogue order."""
    n = weight.shape[0]
    perm = gate_up_perm(n, weight.device)
    w_u8 = weight.view(torch.uint8)
    w_new = w_u8[perm].contiguous().view(weight.dtype)
    nvec = in_features // 16
    off = _sf_offsets(n, nvec, weight.device)
    flat = weight_scale.contiguous().view(torch.uint8).reshape(-1)
    assert flat.numel() >= int(off.max()) + 1
    lin = flat[off].reshape(n, nvec)
    new_flat = flat.clone()
    new_flat[off] = lin[perm].reshape(-1)
    ws_new = new_flat.view(weight_scale.dtype).reshape(weight_scale.shape)
    return w_new, ws_new


def nvfp4_gelu_gemm_fp4(
        act_fp4: torch.Tensor,
        act_sf: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        global_scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused gate_up GEMM + gelu_tanh*up + NVFP4 quant on permuted gate_up rows.

    Returns (fp4 [M, I/2] uint8, sf [pad128(M) * nkt * 4] uint8 swizzled). The
    kernel writes every byte of both, scale padding rows as zero.
    """
    mod = get_gelu_gemm_module()
    m = act_fp4.shape[0]
    n = weight.shape[0]
    i = n // 2
    nkt = (i // 16 + 3) // 4
    if out is None:
        out = torch.empty((m, i // 2), dtype=torch.uint8, device=act_fp4.device)
    if sf is None:
        sf = torch.empty((((m + 127) // 128) * 128 * nkt * 4, ), dtype=torch.uint8,
                         device=act_fp4.device)
    if m == 0:
        return out, sf
    ws = _workspace(act_fp4.device)
    mod.nvfp4_gelu_gemm(act_fp4.view(torch.uint8), weight.view(torch.uint8),
                        act_sf.view(torch.uint8), weight_scale.view(torch.uint8),
                        alpha, global_scale, out, sf, ws)
    return out, sf


_ws_cache = {}


def ensure_gelu_gemm_workspace(device: torch.device) -> None:
    """Allocate the fused op's workspace now (at weight load), never lazily inside a
    CUDA-graph capture."""
    _workspace(device)


def _workspace(device: torch.device) -> torch.Tensor:
    key = str(device)
    ws = _ws_cache.get(key)
    if ws is None:
        ws = torch.empty((1 << 20, ), dtype=torch.uint8, device=device)
        _ws_cache[key] = ws
    return ws


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _gelu_tanh_mul_fp4_permuted_kernel(
    x_ptr,  # [M, 2I] bf16, gate_up output on the fused-epilogue row order
    out_ptr,  # [M, I//2] uint8, two e2m1 per byte (element 0 = low nibble)
    sf_ptr,  # [pad128(M) * 4 * NKT] uint8, swizzled 128x4 layout
    gs_ptr,  # [1] fp32 global scale (down_proj.input_scale)
    M,
    M_PAD,  # pad128(M): rows [M, M_PAD) get zero scales (the 128-row padding)
    SX,  # x row stride (elements)
    NKT,  # numKTiles = ceil((IDIM/16) / 4)
    IDIM: tl.constexpr,  # intermediate size (elements, multiple of 64)
    BM: tl.constexpr,
    BK: tl.constexpr,  # multiple of 64
):
    # Identical to _gelu_tanh_mul_fp4_kernel (gelu_tanh_mul_fp4_quant.py) except
    # for the gather: output column j reads gate at 128*(j//64) + 32*((j%64)//16)
    # + j%16 and up 16 columns further.
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    smask = rows < M_PAD
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = rmask[:, None] & (cols[None, :] < IDIM)
    gcols = (cols // 64) * 128 + ((cols % 64) // 16) * 32 + (cols % 16)
    # Runs of 16 contiguous source columns starting at multiples of 16: lets the
    # loads vectorize like the contiguous kernel's (load addressing only).
    gcols = tl.max_contiguous(tl.multiple_of(gcols, 16), 16)

    gate = tl.load(x_ptr + rows[:, None] * SX + gcols[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + rows[:, None] * SX + 16 + gcols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [inner], dtype=tl.float32, is_pure=True, pack=1
    )
    act = gate * (0.5 * (1.0 + t))
    v = (act * up).to(tl.bfloat16).to(tl.float32)

    v16 = tl.reshape(v, (BM, BK // 16, 16))
    vmax = tl.max(tl.abs(v16), axis=2)  # [BM, BK/16]

    gs = tl.load(gs_ptr)
    rcp6 = _rcp_approx(tl.full((BM, BK // 16), 6.0, tl.float32))
    sf8 = (gs * (vmax * rcp6)).to(tl.float8e4nv)
    rcpgs = _rcp_approx(tl.full((BM, BK // 16), 0.0, tl.float32) + gs)
    oscale = _rcp_approx(sf8.to(tl.float32) * rcpgs)
    oscale = tl.where(vmax != 0.0, oscale, 0.0)

    kvec = (tl.program_id(1) * (BK // 16) + tl.arange(0, BK // 16)).to(tl.int64)
    kvmask = smask[:, None] & (kvec[None, :] < NKT * 4)
    m2 = rows[:, None]
    k2 = kvec[None, :]
    sfoff = (
        (m2 // 128) * (NKT * 512)
        + (k2 // 4) * 512
        + (m2 % 32) * 16
        + ((m2 % 128) // 32) * 4
        + (k2 % 4)
    )
    tl.store(sf_ptr + sfoff, sf8.to(tl.uint8, bitcast=True), mask=kvmask)

    y = tl.reshape(v16 * oscale[:, :, None], (BM, BK))
    lo, hi = tl.split(tl.reshape(y, (BM, BK // 2, 2)))
    byte = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,f,f",
        [lo, hi],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)

    ocols = tl.program_id(1) * (BK // 2) + tl.arange(0, BK // 2)
    omask = rmask[:, None] & (ocols[None, :] * 2 < IDIM)
    tl.store(out_ptr + rows[:, None] * (IDIM // 2) + ocols[None, :], byte, mask=omask)


def gelu_tanh_mul_fp4_quant_permuted(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """gelu_tanh_mul_fp4_quant for a gate_up output on the fused-epilogue row order.

    ``out`` / ``sf_out`` follow gelu_tanh_mul_fp4_quant's contract (caller-owned
    contiguous uint8 destinations of exactly the returned shapes, e.g. the row
    slices gate_up_gelu_fp4_chunked passes; SF slices start on a 128-row tile).
    """
    assert x.dim() == 2 and x.stride(-1) == 1
    assert x.dtype == torch.bfloat16
    assert global_scale.dtype == torch.float32
    m, two_i = x.shape
    i = two_i // 2
    assert two_i % 128 == 0, "intermediate size must be a multiple of 64"

    nkt = triton.cdiv(i // 16, 4)
    m_pad = ((m + 127) // 128) * 128
    if out is None:
        out = torch.empty((m, i // 2), dtype=torch.uint8, device=x.device)
    else:
        assert out.shape == (m, i // 2) and out.dtype == torch.uint8 and out.is_contiguous()
    # The kernel writes every SF byte, padding rows/columns included (as 0x00).
    if sf_out is None:
        sf = torch.empty((m_pad * nkt * 4,), dtype=torch.uint8, device=x.device)
    else:
        assert sf_out.shape == (m_pad * nkt * 4,) and sf_out.dtype == torch.uint8
        assert sf_out.is_contiguous()
        sf = sf_out
    if m == 0:
        return out, sf
    bm, bk, num_warps = _launch_config(m, x.device)
    grid = (triton.cdiv(m_pad, bm), triton.cdiv(i, bk))
    _gelu_tanh_mul_fp4_permuted_kernel[grid](
        x, out, sf, global_scale, m, m_pad, x.stride(0), nkt, IDIM=i, BM=bm, BK=bk,
        num_warps=num_warps
    )
    return out, sf


# --------------------------------------------------------------------------
# Dispatch on permuted gate_up rows (used by _Gemma4GeluQuantMLP and its tests)
# --------------------------------------------------------------------------
EPI_TACTIC = ("cutlass", 2)
_GEMM_OP = "trtllm::nvfp4_gemm::gemm"
_GEMM_DENSE_OP = "trtllm::nvfp4_gemm::gemm_dense"
_epi_decisions: Dict[Tuple, bool] = {}
_epi_cache_len = -1
# Calls per path, for A/B and test accounting.
path_calls = {"fused": 0, "chunked": 0, "unchunked": 0}


def clear_epi_decisions() -> None:
    """Drop the memoized per-shape pick decisions (tests that edit tuner entries)."""
    global _epi_cache_len
    _epi_decisions.clear()
    _epi_cache_len = -1


def _cached_tactic(tuner, op, runner, inputs, tuning_config):
    hit, _, tactic, _ = tuner.profiling_cache.search_cache(
        op, [runner], tuple(tuner._get_input_sizes(inputs)), tuning_config)
    if isinstance(tactic, list):
        tactic = tuple(tactic)
    return bool(hit), tactic


def _main_chunk_runs(act_fp4, act_sf, chunk_rows) -> bool:
    """Structural conditions under which gate_up_gelu_fp4_chunked runs its chunk loop
    (given a CUTLASS full-M pick)."""
    from .gate_up_gelu_fp4_chunked import chunk_bounds

    if chunk_rows is None or chunk_rows <= 0 or chunk_rows % 128 != 0:
        return False
    if act_fp4.dim() != 2 or not act_fp4.is_contiguous() or not act_sf.is_contiguous():
        return False
    m, k2 = act_fp4.shape
    if len(chunk_bounds(m, chunk_rows)) < 2:
        return False
    nvec = (k2 * 2) // 16
    m_pad = (m + 127) // 128 * 128
    return act_sf.numel() == m_pad // 128 * (((nvec + 3) // 4) * 4 * 128)


def main_gate_up_tactic(act_fp4, act_sf, weight, weight_scale, alpha, output_dtype,
                        allowed_backends, chunk_rows: Optional[int] = None):
    """The gate_up GEMM tactic main's _Gemma4GeluQuantMLP chain runs for these inputs in
    inference mode, or None when it would run the cache-miss fallback.

    Mirrors main's order: the M-chunked chain (chunk_rows given and the standard full-M pick is
    a CUTLASS tactic) runs the standard pick per chunk; otherwise trtllm::nvfp4_gemm runs the
    dense-bucket pick (TRTLLM_NVFP4_DENSE_TUNING, M in the shape's dense range, dense cache hit)
    or else the standard pick.
    """
    from ...custom_ops.torch_custom_ops import _nvfp4_dense_range, _nvfp4_dense_tuning_config

    tuner = AutoTuner.get()
    backends = list(allowed_backends)
    inputs = [act_fp4, weight, act_sf, weight_scale, alpha]
    runner = NVFP4GemmUnifiedRunner(0, output_dtype, backends)
    hit, std = _cached_tactic(tuner, _GEMM_OP, runner, inputs, NVFP4GemmUnifiedRunner.tuning_config)
    if (chunk_rows is not None and hit and isinstance(std, tuple) and len(std) == 2
            and std[0] == "cutlass" and _main_chunk_runs(act_fp4, act_sf, chunk_rows)):
        return std
    dense_range = _nvfp4_dense_range(weight.shape[0], act_fp4.shape[1] * 2)
    if dense_range is not None:
        lo, hi = dense_range
        if lo <= act_fp4.shape[0] <= hi:
            druner = NVFP4GemmUnifiedRunner(0, output_dtype, backends, dense_min_m=lo)
            dhit, dt = _cached_tactic(tuner, _GEMM_DENSE_OP, druner, inputs,
                                      _nvfp4_dense_tuning_config(lo, hi))
            if dhit and dt != -1:
                return dt
    return std if hit else None


def epi_pick_is_fused(act_fp4, act_sf, weight, weight_scale, alpha, output_dtype,
                      allowed_backends, chunk_rows: Optional[int] = None) -> bool:
    """True iff the gate_up tactic main runs for these inputs (main_gate_up_tactic) is EPI_TACTIC."""
    global _epi_cache_len
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        return False  # tune exactly what main tunes
    cache = tuner.profiling_cache
    n_entries = len(cache)
    if n_entries != _epi_cache_len:
        _epi_decisions.clear()
        _epi_cache_len = n_entries
    key = (act_fp4.shape[0], act_fp4.shape[1], weight.shape[0], tuple(allowed_backends), output_dtype,
           chunk_rows)
    use = _epi_decisions.get(key)
    if use is None:
        tactic = main_gate_up_tactic(act_fp4, act_sf, weight, weight_scale, alpha, output_dtype,
                                     key[3], chunk_rows)
        use = isinstance(tactic, tuple) and tactic == EPI_TACTIC
        _epi_decisions[key] = use
    return use


def gate_up_gelu_fp4_permuted(
    act_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    allowed_backends,
    down_input_scale: torch.Tensor,
    chunk_rows: Optional[int],
    gemm: Callable[[], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """gate_up -> gelu_tanh*up -> NVFP4 quant on PERMUTED gate_up rows.

    The same order as main's chain, with every activation on the permuted gather:
      (a) "fused": the tactic main runs (main_gate_up_tactic: the chunked chain's full-M
          pick, else nvfp4_gemm's dense or standard pick) == ('cutlass', 2) ->
          nvfp4_gelu_gemm_fp4;
      (b) "chunked": chunk_rows given (main's chunk-band conditions hold) ->
          gate_up_gelu_fp4_chunked with gelu_tanh_mul_fp4_quant_permuted
          (out=/sf_out= slices; None when it does not apply);
      (c) "unchunked": gemm() (the tuned nvfp4_gemm on the permuted rows) ->
          gelu_tanh_mul_fp4_quant_permuted.
    Every path is byte-identical to main's chain on the unpermuted rows.
    """
    if epi_pick_is_fused(act_fp4, act_sf, weight, weight_scale, alpha, output_dtype,
                         allowed_backends, chunk_rows):
        fp4, sf = nvfp4_gelu_gemm_fp4(act_fp4, act_sf, weight, weight_scale, alpha,
                                      down_input_scale)
        path_calls["fused"] += 1
        return fp4, sf, "fused"
    if chunk_rows is not None:
        from .gate_up_gelu_fp4_chunked import gate_up_gelu_fp4_chunked

        res = gate_up_gelu_fp4_chunked(act_fp4, act_sf, weight, weight_scale, alpha,
                                       output_dtype, allowed_backends, down_input_scale,
                                       chunk_rows, act_fn=gelu_tanh_mul_fp4_quant_permuted)
        if res is not None:
            path_calls["chunked"] += 1
            return res[0], res[1], "chunked"
    fp4, sf = gelu_tanh_mul_fp4_quant_permuted(gemm(), down_input_scale)
    path_calls["unchunked"] += 1
    return fp4, sf, "unchunked"
