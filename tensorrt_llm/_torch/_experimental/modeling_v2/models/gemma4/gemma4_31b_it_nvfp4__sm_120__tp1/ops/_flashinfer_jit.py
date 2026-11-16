# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT modules built from this target's FlashInfer kernel sources (`kernels/flashinfer/`).

The target keeps its own copies of the FlashInfer FA2 prefill and NVFP4 GEMM sources it changes
(see `kernels/flashinfer/README.md`) and compiles them with FlashInfer's JIT toolchain against
the installed `flashinfer-python` (the version TensorRT-LLM pins, `FLASHINFER_VERSION`). The
installed package is not modified:

* `include_dir()` materializes, in FlashInfer's writable generated-sources directory, the
  installed `include/` tree with the target's headers laid over it; the target's modules put it
  ahead of the installed one, so every header (including relative includes between FlashInfer
  headers) resolves inside one consistent tree.
* `prefill_module(...)` is the FA2 batch-prefill module FlashInfer's own
  `get_batch_prefill_module("fa2", ...)` builds, generated from the target's
  `batch_prefill.cu` and paged-kernel template under a `mv2_gemma4_` URI (its own cache entry
  and torch op names).
* `gelu_gemm_module()` is the NVFP4 gate_up GEMM with the GELU-tanh + NVFP4 epilogue.

`Fa2PrefillWrapper` is FlashInfer's paged prefill wrapper whose plan picks `prefill_module`
for the configurations this target builds, and FlashInfer's own module otherwise.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import shutil
from pathlib import Path

import flashinfer
import flashinfer.jit.attention.modules as _fi_modules
import flashinfer.prefill as _fi_prefill
import torch
from flashinfer.jit import env as _jit_env
from flashinfer.jit.core import current_compilation_context, gen_jit_spec

#: The FlashInfer release the sources under `kernels/flashinfer/` are a patch of.
FLASHINFER_VERSION = "0.7.3"

KERNELS = Path(__file__).resolve().parents[1] / "kernels" / "flashinfer"
_URI_PREFIX = "mv2_gemma4_"


def available() -> bool:
    """The installed FlashInfer is the release the target's sources patch."""
    return flashinfer.__version__.split("+")[0] == FLASHINFER_VERSION


@functools.cache
def _digest() -> str:
    h = hashlib.sha256()
    for path in sorted(KERNELS.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(KERNELS)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _overlay(installed: Path, ours: Path, name: str, symlink: bool) -> Path:
    """`installed` with `ours` laid over it, once per source digest, built beside and renamed
    into place so a concurrent builder never sees a partial tree."""
    dest = _jit_env.FLASHINFER_GEN_SRC_DIR / f"{_URI_PREFIX}{name}_{_digest()}"
    if dest.exists():
        return dest
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    if symlink:
        tmp.mkdir(parents=True)
        for entry in installed.iterdir():
            (tmp / entry.name).symlink_to(entry)
    else:
        shutil.copytree(installed, tmp, symlinks=True)
    for src in ours.rglob("*"):
        if src.is_file():
            dst = tmp / src.relative_to(ours)
            if dst.is_symlink():
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    try:
        tmp.rename(dest)
    except OSError:  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


@functools.cache
def include_dir() -> Path:
    """The installed `include/` tree with the target's headers laid over it."""
    return _overlay(_jit_env.FLASHINFER_INCLUDE_DIR, KERNELS / "include", "include", symlink=False)


@functools.cache
def _csrc_dir() -> Path:
    """The installed `csrc/` (symlinked) with the target's launcher and template laid over it:
    what the module generator reads its sources from."""
    return _overlay(_jit_env.FLASHINFER_CSRC_DIR, KERNELS / "csrc", "csrc", symlink=True)


@contextlib.contextmanager
def _patched(obj, **attrs):
    old = {name: getattr(obj, name) for name in attrs}
    for name, value in attrs.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(obj, name, value)


_FI_PREFILL_URI = _fi_modules.get_batch_prefill_uri


def _prefixed_uri(*args, **kwargs) -> str:
    return _URI_PREFIX + _FI_PREFILL_URI(*args, **kwargs)


def _target_spec(generator):
    """FlashInfer's spec generator, reading the target's sources and compiling against the
    overlaid include tree, under the target's URI."""

    def generate(*args, **kwargs):
        with (
            _patched(_jit_env, FLASHINFER_CSRC_DIR=_csrc_dir()),
            _patched(_fi_modules, get_batch_prefill_uri=_prefixed_uri),
        ):
            spec = generator(*args, **kwargs)
        spec.extra_include_dirs = [include_dir()] + list(spec.extra_include_dirs or [])
        return spec

    return generate


def supported_prefill_args(args) -> bool:
    """`get_batch_prefill_module("fa2", *args)` arguments this target builds its module for:
    bf16 Q and output over an e4m3 pool, int32 indices, head 256 or 512, no RoPE, soft cap,
    fp16 QK reduction or per-token scales (sliding window either way)."""
    (dtype_q, dtype_kv, dtype_o, dtype_idx, head_qk, head_vo, pos_enc, _window, soft_cap,
     fp16_qk, *rest) = args
    token_head_sf = rest[0] if rest else False
    return (
        dtype_q == torch.bfloat16
        and dtype_kv == torch.float8_e4m3fn
        and dtype_o == torch.bfloat16
        and dtype_idx == torch.int32
        and head_qk == head_vo
        and head_qk in (256, 512)
        and pos_enc == 0
        and not soft_cap
        and not fp16_qk
        and not token_head_sf
    )


@functools.cache
def prefill_module(*args):
    """The FA2 batch-prefill module for `args` (`get_batch_prefill_module`'s, after "fa2"),
    built from the target's sources: the same plan / paged_run / ragged_run surface."""
    with (
        _patched(
            _fi_prefill,
            _gen_batch_prefill_primary_module=_target_spec(
                _fi_prefill._gen_batch_prefill_primary_module
            ),
            _gen_batch_prefill_independent_paged_module=_target_spec(
                _fi_prefill._gen_batch_prefill_independent_paged_module
            ),
            get_batch_prefill_uri=_prefixed_uri,
        ),
    ):
        return _fi_prefill.get_batch_prefill_module.__wrapped__("fa2", *args)


def _module_getter(backend, *args):
    if backend == "fa2" and available() and supported_prefill_args(args):
        return prefill_module(*args)
    return _FI_GET_BATCH_PREFILL_MODULE(backend, *args)


_FI_GET_BATCH_PREFILL_MODULE = _fi_prefill.get_batch_prefill_module


class Fa2PrefillWrapper(flashinfer.BatchPrefillWithPagedKVCacheWrapper):
    """FlashInfer's FA2 paged prefill wrapper on this target's prefill module.

    `plan` chooses the module as the base class does, with the target's module for the
    configurations `supported_prefill_args` admits (and FlashInfer's own module if the installed
    release is not the one the target's sources patch); `run` runs whatever `plan` chose. Every
    change in the target's kernels is bitwise equal to FlashInfer's output, so the choice moves
    speed only.
    """

    def plan(self, *args, **kwargs):
        with _patched(_fi_prefill, get_batch_prefill_module=_module_getter):
            return super().plan(*args, **kwargs)


@functools.cache
def gelu_gemm_module():
    """The SM120 NVFP4 gate_up GEMM with the gelu_tanh(gate) * up + NVFP4-quantize epilogue
    (`kernels/flashinfer/csrc/nvfp4_gelu_gemm_sm120.cu`), built without fast math: the epilogue
    reproduces the unfused activation kernel bit for bit, so no op may flush denormals."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    return gen_jit_spec(
        f"{_URI_PREFIX}nvfp4_gelu_gemm_sm120_{_digest()}",
        [KERNELS / "csrc" / "nvfp4_gelu_gemm_sm120.cu"],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP4"],
        extra_cflags=["-DFAST_BUILD"],
        extra_include_paths=[include_dir()],
        use_fast_math=False,
    ).build_and_load()
