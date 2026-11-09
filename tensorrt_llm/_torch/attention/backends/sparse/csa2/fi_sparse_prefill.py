# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 SM120 shared-context prefill on FlashInfer's DeepSeek-V4.1 FullBF16 direct sparse prefill.

The shared-context route stages a BF16 bank (``stage_shared_rows``: SharedCompact +
two SharedDecode launches), then runs FA2 paged prefill over one-row pages and a sink
epilogue (``CSA2FlashInfer.run_shared``). FlashInfer's SM120 sparse-MLA
``backend="sparse", compute_precision="bf16"`` direct prefill reads CSA2's packed
``[rows, 528]`` SWA and ``[rows, 288]`` main pools in place (page_size=1 views are
byte-identical to the DSV4.1 footer layouts), dequantizes exactly to BF16 on chip,
and runs BF16 QK / PV with fp32 accumulation and softmax plus the same sink epilogue
``out *= sigmoid(lse - sink)``. Empty rows are exactly zero.

This module adds one Triton launch that turns a shared-plan query's selections into
the int32 physical slots FlashInfer takes (``-1`` masks a slot), with exactly the
validity rules of ``SharedCompactKernel`` and the physical rows ``SharedDecodeKernel``
would decode, so the attention sees the same rows (in selection order, with holes
masked instead of compacted): the same math in another order.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_HEAD_DIM = 512

_RUNTIME_INTS = [
    "swa_rs",
    "swa_cs",
    "swa_capacity",
    "swa_pages_cols",
    "swa_pages_stride",
    "swa_page_size",
    "main_rs",
    "main_cs",
    "main_capacity",
    "table_rows",
    "table_cols",
    "table_stride",
    "page_size",
    "max_positions",
    "request_count",
]


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def _shared_prefill_index_kernel(
    swa_ptr,
    swa_rs,
    swa_cs,
    swa_capacity,
    positions_ptr,
    requests_ptr,
    starts_ptr,
    swa_offsets_ptr,
    main_offsets_ptr,
    swa_pages_ptr,
    swa_pages_cols,
    swa_pages_stride,
    swa_page_size,
    swa_out,
    main_ptr,
    main_rs,
    main_cs,
    main_capacity,
    table_ptr,
    table_rows,
    table_cols,
    table_stride,
    page_size,
    max_positions,
    visible_ptr,
    main_out,
    request_count,
    SWA_W: tl.constexpr,
    SWA_P2: tl.constexpr,
    MAIN_W: tl.constexpr,
    MAIN_P2: tl.constexpr,
    HAS_MAIN: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    request = tl.load(requests_ptr + q).to(tl.int64)
    request_valid = (request >= 0) & (request < request_count)
    safe_request = tl.minimum(tl.maximum(request, 0), request_count - 1)
    position = tl.load(positions_ptr + q).to(tl.int64)
    swa_start = tl.load(starts_ptr + safe_request).to(tl.int64)
    swa_begin = tl.load(swa_offsets_ptr + safe_request).to(tl.int64)
    swa_end = tl.load(swa_offsets_ptr + safe_request + 1).to(tl.int64)
    # SharedCompactKernel: a SWA column is kept when its slot is a pool row and its
    # logical row lies inside the request's SWA bank domain.
    cols = tl.arange(0, SWA_P2)
    live = cols < SWA_W
    slot = tl.load(swa_ptr + q * swa_rs + cols * swa_cs, mask=live, other=-1).to(tl.int64)
    logical = position - SWA_W + 1 + cols
    row = swa_begin + logical - swa_start
    valid = (slot >= 0) & (slot < swa_capacity) & (row >= swa_begin) & (row < swa_end)
    valid = valid & request_valid & live
    # SharedDecodeKernel: the bank row decodes logical ``row - begin + start`` from
    # the request's SWA page table.
    column = logical // swa_page_size
    valid = valid & (logical >= 0) & (column < swa_pages_cols)
    page = tl.load(
        swa_pages_ptr + safe_request * swa_pages_stride + column, mask=valid, other=-1
    ).to(tl.int64)
    physical = page * swa_page_size + logical % swa_page_size
    valid = valid & (page >= 0) & (physical < swa_capacity)
    tl.store(swa_out + q * SWA_W + cols, tl.where(valid, physical, -1).to(tl.int32), mask=live)
    if HAS_MAIN:
        main_begin = tl.load(main_offsets_ptr + safe_request).to(tl.int64)
        main_end = tl.load(main_offsets_ptr + safe_request + 1).to(tl.int64)
        visible = tl.load(visible_ptr + q).to(tl.int64)
        mcols = tl.arange(0, MAIN_P2)
        mlive = mcols < MAIN_W
        x = tl.load(main_ptr + q * main_rs + mcols * main_cs, mask=mlive, other=-1).to(tl.int64)
        # ``map_logical_slot`` with the plan's query request, visibility applied.
        lx = tl.maximum(x, 0)
        mpage = lx // page_size
        has_table = (table_rows > 0) & (table_cols > 0)
        table_request = tl.minimum(tl.maximum(request, 0), table_rows - 1)
        safe_page = tl.minimum(mpage, table_cols - 1)
        mphys = tl.load(
            table_ptr + table_request * table_stride + safe_page,
            mask=mlive & has_table & (x >= 0),
            other=-1,
        ).to(tl.int64)
        ok = has_table & (x >= 0) & (x < max_positions) & (mpage < table_cols)
        ok = ok & (request >= 0) & (request < table_rows) & (mphys >= 0) & (x < visible)
        mslot = tl.where(ok, mphys * page_size + lx % page_size, -1)
        ok = (mslot >= 0) & (mslot < main_capacity) & (x >= 0) & (x < main_end - main_begin)
        ok = ok & request_valid & mlive
        tl.store(
            main_out + q * MAIN_W + mcols, tl.where(ok, mslot, -1).to(tl.int32), mask=mlive
        )


def prepare_shared_prefill_indices(
    swa_pool,
    swa_slots,
    main_pool,
    main_logical,
    mapping,
    plan,
    swa_out,
    main_out=None,
    direct=False,
):
    """Write int32 physical SWA / main slots (``-1`` = masked) for one shared-plan tile.

    ``direct`` launches through the cached CompiledKernel on a specialization-key hit.
    """
    count, swa_width = swa_slots.shape
    has_main = main_logical is not None and main_logical.shape[1] > 0
    if has_main:
        table, page_size, max_positions, _, visible = mapping
        table_stride = table.stride(0) if table.shape[0] > 1 else table.shape[1]
        table_rows, table_cols = table.shape
        main_width = int(main_logical.shape[1])
    else:
        table, page_size, max_positions, visible = swa_out, 1, 0, plan.query_positions
        table_stride, table_rows, table_cols = 0, 0, 0
        main_width = 1
    pages = plan.swa_pages
    pages_stride = pages.stride(0) if pages.shape[0] > 1 else pages.shape[1]
    args = (
        swa_slots,
        swa_slots.stride(0),
        swa_slots.stride(1) if swa_width > 1 else 1,
        int(swa_pool.shape[0]),
        plan.query_positions,
        plan.query_requests,
        plan.swa_starts,
        plan.swa_offsets,
        plan.main_offsets,
        pages,
        int(pages.shape[1]),
        int(pages_stride),
        int(plan.swa_page_size),
        swa_out,
        main_logical if has_main else swa_slots,
        main_logical.stride(0) if has_main else 0,
        (main_logical.stride(1) if main_width > 1 else 1) if has_main else 1,
        int(main_pool.shape[0]) if has_main else 0,
        table,
        int(table_rows),
        int(table_cols),
        int(table_stride),
        int(page_size),
        int(max_positions),
        visible,
        main_out if has_main else swa_out,
        int(plan.num_requests),
    )
    consts = (
        int(swa_width),
        triton.next_power_of_2(max(int(swa_width), 1)),
        main_width,
        triton.next_power_of_2(main_width),
        has_main,
    )
    _launch_index_kernel(count, args, consts, direct)


# Positions of the tensor (pointer) arguments of ``_shared_prefill_index_kernel``; every
# other runtime argument is an int listed in ``_RUNTIME_INTS`` (not specialized).
_PTR_ARGS = (0, 4, 5, 6, 7, 8, 9, 13, 14, 18, 24, 25)
_INT_ARGS = tuple(i for i in range(27) if i not in _PTR_ARGS)
_INDEX_KERNELS: dict[tuple, object] = {}
_INDEX_KERNELS_MAX = 64


def _launch_index_kernel(count, args, consts, direct):
    """Launch the index kernel; on a specialization-key hit, through the cached CompiledKernel.

    Triton specializes this kernel on its constexprs, on each pointer's dtype and 16-byte
    alignment, and on each int's 32/64-bit width (``do_not_specialize`` drops the value
    specializations). The key below holds exactly those, so a hit runs the same binary
    the JIT dispatcher would pick, with the same arguments, minus its Python binding.
    """
    key = None
    if direct and getattr(triton.knobs.runtime, "launch_enter_hook", None) is None:
        device = torch.cuda.current_device()
        key = (
            device,
            consts,
            tuple((args[i].dtype, args[i].data_ptr() % 16 == 0) for i in _PTR_ARGS),
            all(-(2**31) <= args[i] < 2**31 for i in _INT_ARGS),
        )
        kernel = _INDEX_KERNELS.get(key)
        if kernel is not None:
            kernel.run(
                count,
                1,
                1,
                torch._C._cuda_getCurrentRawStream(device),
                kernel.function,
                kernel.packed_metadata,
                None,
                None,
                triton.knobs.runtime.launch_exit_hook,
                *args,
                *consts,
            )
            return
    kernel = _shared_prefill_index_kernel[(count,)](
        *args,
        SWA_W=consts[0],
        SWA_P2=consts[1],
        MAIN_W=consts[2],
        MAIN_P2=consts[3],
        HAS_MAIN=consts[4],
        num_warps=4,
    )
    if key is not None and kernel is not None and not torch.cuda.is_current_stream_capturing():
        if len(_INDEX_KERNELS) >= _INDEX_KERNELS_MAX:
            _INDEX_KERNELS.clear()
        _INDEX_KERNELS[key] = kernel


@functools.lru_cache(maxsize=None)
def fi_prefill_available(device_index: int) -> bool:
    """Whether FlashInfer's SM120 DSv4.1 FullBF16 sparse prefill can serve this device."""
    if torch.cuda.get_device_capability(device_index)[0] != 12:
        return False
    try:
        from flashinfer.mla import SparseMLASm120Wrapper  # noqa: F401
    except Exception:  # noqa: BLE001 - an older FlashInfer has no such route
        return False
    return True


def _pool_ok(pool, row_bytes) -> bool:
    return (
        pool is not None
        and pool.is_cuda
        and pool.dtype == torch.uint8
        and pool.ndim == 2
        and pool.shape[1] == row_bytes
        and pool.stride(1) == 1
        and pool.stride(0) == row_bytes
        and pool.data_ptr() % 16 == 0
        and 0 < pool.shape[0] < 2**31
    )


def supported(q, swa_pool, swa_slots, main_pool, main_logical, mapping, plan) -> bool:
    count, heads, dim = q.shape
    if not q.is_cuda or not fi_prefill_available(q.device.index):
        return False
    if dim != _HEAD_DIM or q.dtype != torch.bfloat16 or not q.is_contiguous():
        return False
    if count <= 64 or heads > 128 or plan is None:
        return False
    if not _pool_ok(swa_pool, 528) or swa_slots is None or swa_slots.ndim != 2:
        return False
    if swa_slots.shape != (count, 128) or swa_slots.dtype not in (torch.int32, torch.int64):
        return False
    if plan.swa_pages.dtype != torch.int32 or plan.swa_pages.ndim != 2:
        return False
    if plan.swa_pages.shape[1] > 1 and plan.swa_pages.stride(1) != 1:
        return False
    if main_logical is not None and main_logical.shape[1] > 0:
        if mapping is None or not _pool_ok(main_pool, 288):
            return False
        if main_logical.ndim != 2 or main_logical.shape[0] != count:
            return False
        if main_logical.dtype not in (torch.int32, torch.int64):
            return False
        table = mapping[0]
        if table.dtype != torch.int32 or table.ndim != 2:
            return False
        if table.shape[1] > 1 and table.stride(1) != 1:
            return False
    return True


class FISharedPrefill:
    """Per-shape int32 index scratch, one FullBF16 wrapper per cache form, and the launch."""

    _MAX_SHAPES = 8

    def __init__(self) -> None:
        self._buffers: dict[tuple, dict[str, torch.Tensor]] = {}
        self._wrappers: dict[tuple, object] = {}
        # (PreparedCall, mid, mlse) per call-signature key, LRU like the index scratch.
        self._prepared: dict[tuple, tuple] = {}
        self._views: dict[str, tuple] = {}
        self.direct_calls = 0
        self.wrapper_calls = 0
        from flashinfer.mla._sparse_mla_sm120 import _calibration, _prepared

        self._calibration = _calibration
        self._tuning_enabled = _prepared.tuning_enabled

    def _view(self, name, pool, row_bytes):
        # The page_size=1 cache view of a persistent pool, reused while the pool is.
        cached = self._views.get(name)
        if cached is not None and cached[0] is pool:
            return cached[1]
        view = pool.view(pool.shape[0], 1, row_bytes)
        self._views[name] = (pool, view)
        return view

    def _wrapper(self, dual: bool, device):
        key = (dual, device)
        wrapper = self._wrappers.get(key)
        if wrapper is None:
            from flashinfer.mla import SparseMLASm120Wrapper

            wrapper = SparseMLASm120Wrapper(
                kv_cache_format="fp8",
                kv_scale_format="ue8m0_g32",
                extra_kv_fp4=dual,
                compute_precision="bf16",
                device=device,
                backend="sparse",
            )
            self._wrappers[key] = wrapper
        return wrapper

    def _scratch(self, count, heads, swa_width, main_width, device):
        key = (count, heads, swa_width, main_width, device)
        # Pop and re-insert: the dict's order is the LRU order.
        buffers = self._buffers.pop(key, None)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the CSA2 FlashInfer prefill shape before capture")
            # Eager prefill shapes vary per step; retain the most recent few only.
            while len(self._buffers) >= self._MAX_SHAPES:
                del self._buffers[next(iter(self._buffers))]
            buffers = {
                "swa": torch.empty((count, swa_width), dtype=torch.int32, device=device),
                "main": torch.empty((count, max(main_width, 1)), dtype=torch.int32, device=device),
                "lse": torch.empty((count, heads), dtype=torch.float32, device=device),
            }
        self._buffers[key] = buffers
        return buffers

    def __call__(self, q, swa_pool, swa_slots, main_pool, main_logical, mapping, plan, sink, scale, out):
        count, heads, _ = q.shape
        swa_width = int(swa_slots.shape[1])
        has_main = main_logical is not None and main_logical.shape[1] > 0
        main_width = int(main_logical.shape[1]) if has_main else 0
        buffers = self._scratch(count, heads, swa_width, main_width, q.device)
        prepare_shared_prefill_indices(
            swa_pool,
            swa_slots,
            main_pool if has_main else None,
            main_logical if has_main else None,
            mapping,
            plan,
            buffers["swa"],
            buffers["main"] if has_main else None,
            direct=True,
        )
        if sink.dtype != torch.float32 or not sink.is_contiguous():
            sink = sink.float().contiguous()
        cache = self._view("swa", swa_pool, 528)
        extra = self._view("main", main_pool, 288) if has_main else None
        extra_indices = buffers["main"] if has_main else None
        # Every component of FlashInfer's per-call signature key (shape, stride, dtype,
        # device of q, caches, indices, output, sink, LSE) is fixed by this key: q,
        # the pools and the scratch are contiguous (``supported``/``_pool_ok``).
        key = (
            has_main,
            count,
            heads,
            swa_width,
            main_width,
            q.device,
            cache.shape[0],
            extra.shape[0] if has_main else 0,
            out.dtype,
            out.stride(),
            sink.shape,
            float(scale),
        )
        entry = self._prepared.pop(key, None)
        if (
            entry is not None
            and entry[0].epoch == self._calibration._constants_version
            and not self._tuning_enabled(False)
        ):
            self._prepared[key] = entry
            prepared, mid, mlse = entry
            prepared.execute(
                q,
                cache,
                buffers["swa"],
                out,
                scale,
                None,
                sink,
                extra,
                extra_indices,
                None,
                mid,
                mlse,
                buffers["lse"],
                1.0,
            )
            self.direct_calls += 1
            return out
        wrapper = self._wrapper(has_main, q.device)
        wrapper.run(
            q,
            cache,
            buffers["swa"],
            out,
            scale,
            attn_sink=sink,
            extra_kv_cache=extra,
            extra_indices=extra_indices,
            out_lse=buffers["lse"],
        )
        self.wrapper_calls += 1
        if not torch.cuda.is_current_stream_capturing():
            last = getattr(wrapper, "_last_prepared", None)
            wrapper._last_prepared = None
            if last is not None:
                # wrapper.run just executed exactly this PreparedCall with these scratch
                # views; later calls with the same key replay it directly.
                while len(self._prepared) >= self._MAX_SHAPES:
                    del self._prepared[next(iter(self._prepared))]
                self._prepared[key] = last
        return out
