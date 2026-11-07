# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 SM120 decode attention on FlashInfer's Cake DeepSeek-V4.1 mixed-cache sparse-MLA decode.

Generation-only CSA2 rows on SM120/SM121 otherwise run ``StageCompact`` /
``StageDequant`` (packed rows -> BF16 staging pools) followed by the two-launch
Triton split sink attention. CSA2's persistent caches are byte-identical to the
DeepSeek-V4.1 ``fp8_dsv41_fp4_ca`` footer formats at page_size=1:

* SWA rows ``[rows, 528]``: 512 E4M3 + 16 UE8M0 scales per 32 channels;
* main (compressed) rows ``[rows, 288]``: 256 B packed E2M1 (low nibble = even
  channel) + 32 E4M3 scales per 16 channels.

So the Cake kernel reads both pools in place and dequantizes on chip (exact:
E4M3 x 2^e and E2M1 x E4M3 are representable in BF16), with BF16 QK / PV,
fp32 softmax and accumulation, and the same attention-sink epilogue
``out *= sigmoid(lse - sink)``. That is the staging + split-sink math in
another order (split partials are BF16, merged in fp32). This module adds one
small launch that turns the layer's selections into the int32 physical-slot
indices Cake takes (``-1`` masks a slot; out-of-pool slots are masked like the
staging kernels mask them) and drives the decode with caller-owned split scratch,
so it is CUDA-graph safe once a shape has been warmed up.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

_HEAD_DIM = 512

# (max count, render) buckets for 16-head dual fp32-partials calls under TRTLLM_CSA2_CAKE_F32_VARIANT=auto;
# counts above the last bucket keep the plain `default` body.
_F32_FOLD_BUCKETS = ((16, "sa+pw+lq+el"),)

# Decode counts whose fp32-partials Cake decode body (`default`, or the h16 `sa+lq+el` / `sa+pw+lq+el`
# fold render) fires its PDL trigger on entry (binding ``enable_pdl`` bit 2), so the PDL merge and
# csa2_lowm_wo_a behind it launch while the body runs. Launch timing only (every dependent waits for
# the whole body grid before it reads or writes global memory). TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY=1 / =0
# forces it on / off for every count. Default ON only at the counts with a measured same-slot win
# (cold-L2 40-layer chain, experiments/ascent/cake_q1/chain_332.py); other counts launch as before.
_BODY_TRIGGER_ENTRY_COUNTS: frozenset[int] = frozenset((1, 2, 3, 4))
_BODY_TRIGGER_ENTRY_VARIANTS = frozenset(("default", "sa+lq+el", "sa+pw+lq+el"))


def _body_trigger_entry_mode(value: bool | None) -> bool | None:
    """None = per-count default (``_BODY_TRIGGER_ENTRY_COUNTS``), True/False = forced."""
    if value is not None:
        return bool(value)
    env = os.environ.get("TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY", "")
    if env == "":
        return None
    return env != "0"


# Every runtime argument stays unspecialized: eager mixed steps see new batch shapes
# (page-table rows, slice offsets), and a value/alignment specialization would JIT a
# new variant mid-serving instead of reusing the one warmed up before capture.
_RUNTIME_INTS = [
    "swa_rs",
    "swa_cs",
    "swa_capacity",
    "main_rs",
    "main_cs",
    "main_capacity",
    "table_rows",
    "table_cols",
    "table_stride",
    "page_size",
    "max_positions",
]
_RUNTIME_PTRS = [
    "swa_ptr",
    "swa_out",
    "main_ptr",
    "main_out",
    "table_ptr",
    "requests_ptr",
    "visible_ptr",
]


@triton.jit(do_not_specialize=_RUNTIME_INTS, do_not_specialize_on_alignment=_RUNTIME_PTRS)
def _cake_index_prep_kernel(
    swa_ptr,
    swa_rs,
    swa_cs,
    swa_capacity,
    swa_out,
    main_ptr,
    main_rs,
    main_cs,
    main_capacity,
    main_out,
    table_ptr,
    table_rows,
    table_cols,
    table_stride,
    page_size,
    max_positions,
    requests_ptr,
    visible_ptr,
    SWA_W: tl.constexpr,
    SWA_P2: tl.constexpr,
    MAIN_W: tl.constexpr,
    MAIN_P2: tl.constexpr,
    HAS_MAIN: tl.constexpr,
    MAPPED: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, SWA_P2)
    live = cols < SWA_W
    slot = tl.load(swa_ptr + q * swa_rs + cols * swa_cs, mask=live, other=-1).to(tl.int64)
    ok = (slot >= 0) & (slot < swa_capacity)
    tl.store(swa_out + q * SWA_W + cols, tl.where(ok, slot, -1).to(tl.int32), mask=live)
    if HAS_MAIN:
        cols = tl.arange(0, MAIN_P2)
        live = cols < MAIN_W
        x = tl.load(main_ptr + q * main_rs + cols * main_cs, mask=live, other=-1).to(tl.int64)
        if MAPPED:
            # ``map_logical_slot`` (csa2/kernel.py) for every entry, visibility applied.
            request = tl.load(requests_ptr + q).to(tl.int64)
            visible = tl.load(visible_ptr + q).to(tl.int64)
            logical = tl.maximum(x, 0)
            page = logical // page_size
            safe_request = tl.minimum(tl.maximum(request, 0), table_rows - 1)
            safe_page = tl.minimum(page, table_cols - 1)
            has_table = (table_rows > 0) & (table_cols > 0)
            physical = tl.load(
                table_ptr + safe_request * table_stride + safe_page,
                mask=live & has_table & (x >= 0),
                other=-1,
            ).to(tl.int64)
            valid = has_table & (x >= 0) & (x < max_positions) & (page < table_cols)
            valid = valid & (request >= 0) & (request < table_rows) & (physical >= 0)
            valid = valid & (x < visible)
            slot = tl.where(valid, physical * page_size + logical % page_size, -1)
        else:
            slot = x
        ok = (slot >= 0) & (slot < main_capacity)
        tl.store(main_out + q * MAIN_W + cols, tl.where(ok, slot, -1).to(tl.int32), mask=live)


def prepare_cake_indices(
    swa_slots, swa_capacity, swa_out, main=None, main_capacity=0, main_out=None, mapping=None
):
    """Write int32 physical slots (``-1`` = masked) for Cake, one launch.

    ``swa_slots`` ``[count, swa_width]`` int32/int64 physical SWA rows; ``main``
    ``[count, main_width]`` either physical main rows (``mapping is None``) or
    logical positions mapped through ``mapping = (table, page_size,
    max_positions, requests, visible)`` exactly as the staging kernels do.
    """
    count, swa_width = swa_slots.shape
    has_main = main is not None and main.shape[1] > 0
    mapped = has_main and mapping is not None
    if mapped:
        table, page_size, max_positions, requests, visible = mapping
        table_stride = table.stride(0) if table.shape[0] > 1 else table.shape[1]
        table_rows, table_cols = table.shape
    else:
        table, page_size, max_positions = swa_out, 1, 0
        requests = visible = swa_out
        table_stride, table_rows, table_cols = 0, 0, 0
    main_width = int(main.shape[1]) if has_main else 1
    _cake_index_prep_kernel[(count,)](
        swa_slots,
        swa_slots.stride(0),
        swa_slots.stride(1) if swa_width > 1 else 1,
        int(swa_capacity),
        swa_out,
        main if has_main else swa_slots,
        main.stride(0) if has_main else 0,
        (main.stride(1) if main.shape[1] > 1 else 1) if has_main else 1,
        int(main_capacity),
        main_out if has_main else swa_out,
        table,
        int(table_rows),
        int(table_cols),
        int(table_stride),
        int(page_size),
        int(max_positions),
        requests,
        visible,
        SWA_W=int(swa_width),
        SWA_P2=triton.next_power_of_2(max(int(swa_width), 1)),
        MAIN_W=main_width,
        MAIN_P2=triton.next_power_of_2(main_width),
        HAS_MAIN=has_main,
        MAPPED=mapped,
        num_warps=4,
    )


def _flat_pool(pool: torch.Tensor, row_bytes: int) -> torch.Tensor:
    """Flat uint8 span of a ``[rows, row_bytes]`` pool view (page_size=1 footer layout)."""
    span = (pool.shape[0] - 1) * pool.stride(0) + row_bytes
    return pool.as_strided((span,), (1,), pool.storage_offset())


def _cache_launch(q, swa_pool, main_pool, swa_width, main_width, variant, scale, info):
    """Resolve the exact plan the validated public call used, for direct binding launches."""
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as cdm

    plan = cdm._resolve_plan(
        num_tokens=int(q.shape[0]),
        num_heads=int(q.shape[1]),
        topk=swa_width,
        extra_topk=main_width,
        dual=main_pool is not None,
        ragged=False,
        device=q.device,
        num_splits=None,
        max_splits=16,
        variant=variant,
        page_size=1,
        extra_page_size=1 if main_pool is not None else 0,
    )
    if (plan.variant, plan.num_splits, plan.chunks_per_block) != (
        info["variant"],
        info["num_splits"],
        info["chunks_per_block"],
    ):
        return None  # never take the direct path on a plan the public call did not run
    key = (
        swa_pool.data_ptr(),
        swa_pool.shape,
        swa_pool.stride(),
        None if main_pool is None else (main_pool.data_ptr(), main_pool.shape, main_pool.stride()),
        scale,
    )
    entry = cdm.get_cake_sparse_mla_sm120_dsv41_mixed_module().raw_decode
    return key, entry, plan, cdm._resolve_enable_pdl(None, q.device)


@functools.lru_cache(maxsize=None)
def cake_available(device_index: int) -> bool:
    """Whether the Cake SM120 DSv4.1 mixed-cache decode can serve this device."""
    if os.environ.get("TRTLLM_CSA2_CAKE_DECODE", "1") == "0":
        return False
    if torch.cuda.get_device_capability(device_index)[0] != 12:
        return False
    try:
        from flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_available,
        )
        from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (  # noqa: F401
            cake_sparse_mla_sm120_dsv41_mixed_decode,
        )
    except Exception:  # noqa: BLE001 - an older FlashInfer has no such route
        return False
    return bool(cake_sparse_mla_sm120_dsv41_mixed_available())


@functools.lru_cache(maxsize=None)
def _supported_heads() -> tuple[int, ...]:
    from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
        cake_sparse_mla_sm120_dsv41_mixed_supported_heads,
    )

    return tuple(cake_sparse_mla_sm120_dsv41_mixed_supported_heads())


def supported(q, swa_pool, swa_slots, main_pool, main_slots) -> bool:
    count, heads, dim = q.shape
    if not q.is_cuda or not cake_available(q.device.index):
        return False
    if dim != _HEAD_DIM or q.dtype != torch.bfloat16 or not q.is_contiguous() or count == 0:
        return False
    if heads not in _supported_heads():
        return False
    if (
        swa_pool is None
        or swa_slots is None
        or swa_pool.dtype != torch.uint8
        or swa_pool.ndim != 2
        or swa_pool.shape[1] != 528
        or swa_pool.stride(1) != 1
        or swa_pool.stride(0) % 16
        or swa_pool.data_ptr() % 16
        or swa_slots.ndim != 2
        or swa_slots.shape[0] != count
        or swa_slots.shape[1] < 1
        or swa_slots.dtype not in (torch.int32, torch.int64)
        or swa_pool.shape[0] >= 2**31
    ):
        return False
    if main_slots is not None and main_slots.shape[1] > 0:
        if (
            main_pool is None
            or main_pool.dtype != torch.uint8
            or main_pool.ndim != 2
            or main_pool.shape[1] != 288
            or main_pool.stride(1) != 1
            or main_pool.stride(0) % 16
            or main_pool.data_ptr() % 16
            or main_slots.ndim != 2
            or main_slots.shape[0] != count
            or main_slots.dtype not in (torch.int32, torch.int64)
            or main_pool.shape[0] >= 2**31
        ):
            return False
    return True


class CakeDecode:
    """Per-shape scratch (int32 indices, split partials, LSE) and the Cake launch.

    ``f32_partials`` (default on) keeps the split-K partials in fp32: the 16-head
    ``default`` Cake bodies write them and the binding's fp32 merge combines them
    (FlashInfer overlay), so the only BF16 rounding is the output's, as in the
    split sink path. Off, the stock BF16 partials and dispatched variant run.
    """

    def __init__(
        self,
        f32_partials: bool | None = None,
        f32_variant: str | None = None,
        body_trigger_entry: bool | None = None,
    ) -> None:
        self._buffers: dict[tuple, dict[str, torch.Tensor]] = {}
        if f32_partials is None:
            f32_partials = os.environ.get("TRTLLM_CSA2_CAKE_F32_PARTIALS", "1") != "0"
        self.f32_partials = bool(f32_partials)
        if f32_variant is None:
            f32_variant = os.environ.get("TRTLLM_CSA2_CAKE_F32_VARIANT", "auto")
        self.f32_variant = str(f32_variant)
        self.body_trigger_entry = _body_trigger_entry_mode(body_trigger_entry)

    def _entry_trigger_bit(self, count: int, heads: int, plan, enable_pdl) -> int:
        """Binding enable_pdl bit 2: only fp32-partials calls on the bodies that take it, PDL on."""
        if not enable_pdl or not self._use_f32(heads) or plan.variant not in _BODY_TRIGGER_ENTRY_VARIANTS:
            return 0
        mode = self.body_trigger_entry
        on = (count in _BODY_TRIGGER_ENTRY_COUNTS) if mode is None else mode
        return 4 if on else 0

    def _f32_variant_for(self, count: int, has_main: bool) -> str:
        """The exact Cake render that writes the fp32 partials for one 16-head call.

        ``auto``: dual (SWA + main) calls take the h16 fold render (``sa+lq+el`` /
        ``sa+pw+lq+el``, FlashInfer overlay) in the count buckets where it is faster,
        else the plain ``default`` body; ``default`` restores the plain body everywhere;
        any other value pins that label on dual calls. Every render computes the same
        fp32 partials (exact renders of one kernel).
        """
        if not has_main or self.f32_variant == "default":
            return "default"
        if self.f32_variant != "auto":
            return self.f32_variant
        for limit, label in _F32_FOLD_BUCKETS:
            if count <= limit:
                return label
        return "default"

    def _scratch(self, count, heads, swa_width, main_width, device):
        key = (count, heads, swa_width, main_width, device)
        buffers = self._buffers.get(key)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the CSA2 Cake decode shape before CUDA Graph capture")
            from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
                cake_sparse_mla_sm120_dsv41_mixed_num_chunks,
                get_cake_sparse_mla_sm120_dsv41_mixed_module,
            )

            get_cake_sparse_mla_sm120_dsv41_mixed_module()  # JIT build/load outside capture
            splits = max(cake_sparse_mla_sm120_dsv41_mixed_num_chunks(swa_width, main_width), 1)
            buffers = {
                "swa": torch.full((count, swa_width), -1, dtype=torch.int32, device=device),
                "main": torch.full((count, max(main_width, 1)), -1, dtype=torch.int32, device=device),
                "lse": torch.empty((count, heads), dtype=torch.float32, device=device),
                "mid_out": torch.empty(
                    (count, heads, splits, _HEAD_DIM),
                    dtype=torch.float32 if self._use_f32(heads) else torch.bfloat16,
                    device=device,
                ),
                "mid_lse": torch.empty((count, heads, splits), dtype=torch.float32, device=device),
            }
            self._buffers[key] = buffers
        return buffers

    def _use_f32(self, heads: int) -> bool:
        return self.f32_partials and heads == 16

    def __call__(
        self,
        q,
        swa_pool,
        swa_slots,
        main_pool,
        main_slots,
        sink,
        scale,
        out,
        mapping=None,
        pdl_consumer=False,
    ):
        """``out[i]`` = sink attention of ``q[i]`` over its selected SWA and main rows.

        ``pdl_consumer``: the next launch is a PDL kernel that reads ``out`` only after its
        griddepcontrol.wait (csa2_lowm_wo_a), so the fp32 split merge may trigger its
        dependents first (binding ``enable_pdl`` bit 1). Launch attributes only; same math.
        """
        from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_decode,
        )

        count, heads, _ = q.shape
        swa_width = int(swa_slots.shape[1])
        has_main = main_slots is not None and main_slots.shape[1] > 0
        main_width = int(main_slots.shape[1]) if has_main else 0
        buffers = self._scratch(count, heads, swa_width, main_width, q.device)
        prepare_cake_indices(
            swa_slots,
            swa_pool.shape[0],
            buffers["swa"],
            main_slots if has_main else None,
            main_pool.shape[0] if has_main else 0,
            buffers["main"] if has_main else None,
            mapping if has_main else None,
        )
        if sink.dtype != torch.float32 or not sink.is_contiguous():
            sink = sink.float().contiguous()
        launch = buffers.get("launch")
        if launch is not None and launch[0] == (
            swa_pool.data_ptr(),
            swa_pool.shape,
            swa_pool.stride(),
            None if not has_main else (main_pool.data_ptr(), main_pool.shape, main_pool.stride()),
            float(scale),
        ):
            # Same shape and pools as the validated first call: launch the binding directly
            # (eager mixed steps pay no per-call validation / plan resolution).
            entry, plan, enable_pdl = launch[1:]
            entry(
                q,
                _flat_pool(swa_pool, 528),
                buffers["swa"],
                _flat_pool(main_pool, 288) if has_main else None,
                buffers["main"] if has_main else None,
                None,
                None,
                sink,
                out,
                buffers["lse"],
                buffers["mid_out"] if plan.num_splits > 1 else None,
                buffers["mid_lse"] if plan.num_splits > 1 else None,
                1,
                swa_pool.stride(0),
                1 if has_main else 0,
                main_pool.stride(0) if has_main else 0,
                plan.chunks_per_block,
                plan.decode_body,
                plan.merge_body,
                plan.hb_major,
                float(scale),
                1.0,
                int(bool(enable_pdl))
                | (2 if pdl_consumer else 0)
                | self._entry_trigger_bit(count, heads, plan, enable_pdl),
            )
            return out
        variant = self._f32_variant_for(count, has_main) if self._use_f32(heads) else None
        info = cake_sparse_mla_sm120_dsv41_mixed_decode(
            q,
            swa_pool,
            buffers["swa"],
            out,
            buffers["lse"],
            float(scale),
            attn_sink=sink,
            extra_kv_cache=main_pool if has_main else None,
            extra_indices=buffers["main"] if has_main else None,
            mid_out=buffers["mid_out"],
            mid_lse=buffers["mid_lse"],
            variant=variant,
        )
        if not torch.cuda.is_current_stream_capturing():
            buffers["launch"] = _cache_launch(
                q, swa_pool, main_pool if has_main else None, swa_width, main_width, variant,
                float(scale), info,
            )
        return out


__all__ = ["CakeDecode", "cake_available", "prepare_cake_indices", "supported"]
