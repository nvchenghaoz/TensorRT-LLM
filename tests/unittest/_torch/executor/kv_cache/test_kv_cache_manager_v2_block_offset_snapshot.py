# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVCacheManagerV2 block-offset copy through a host-gathered pinned snapshot.

Drives the real ``copy_batch_block_offsets`` on a bare manager with the real C++
IndexMapper and a pinned page table of a served shape (2 pools x 65 rows x 2 x 1156
blocks), and compares the snapshot path with the stock zero-copy kernel bitwise and
with a CPU reference.
"""

import random

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as m
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    IndexMapper,
    copy_batch_block_offsets_to_device,
)

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA device", allow_module_level=True)

P, SRC_MAX, B, DST_MAX = 2, 65, 1156, 32
SCALES, OFFS = [48, 32], [0, 1]
SENTINEL = 0x5A5A5A5A
BAD = -1


def _manager(stream: torch.cuda.Stream) -> m.KVCacheManagerV2:
    mgr = m.KVCacheManagerV2.__new__(m.KVCacheManagerV2)
    mgr.sparse_metadata_batch = None
    mgr.max_copy_beam_width = 1
    mgr.index_mapper = IndexMapper(SRC_MAX, 1, 0)
    mgr.kv_cache_type = m.CacheTypeCpp.SELF
    mgr._use_per_layer_page_tables = False
    mgr.host_kv_cache_block_offsets = torch.zeros(P, SRC_MAX, 2, B, dtype=torch.int32).pin_memory()
    mgr.index_scales = torch.tensor(SCALES, dtype=torch.int32).pin_memory()
    mgr.kv_offset = torch.tensor(OFFS, dtype=torch.int32).pin_memory()
    mgr._stream = stream
    mgr.block_offset_snapshot_copy = True
    return mgr


def _expected(host: torch.Tensor, copy_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    k = host[:, copy_idx, 0, :].long()
    scales = torch.tensor(SCALES).view(P, 1, 1).long()
    offsets = torch.tensor(OFFS).view(P, 1, 1).long()
    return (
        torch.where(k == BAD, 0, k * scales).int(),
        torch.where(k == BAD, 0, k * scales + offsets).int(),
    )


def _fill_row(host: torch.Tensor, slot: int, num_blocks: int, rng: random.Random) -> None:
    for p in range(P):
        vals = torch.tensor([rng.randrange(0, 60000) for _ in range(num_blocks)], dtype=torch.int32)
        vals[torch.rand(num_blocks) < 0.03] = BAD
        host[p, slot, 0, :num_blocks] = vals
        host[p, slot, 1, :num_blocks] = vals + 7  # the V plane is never read
        host[p, slot, 0, num_blocks:] = 0


@pytest.mark.parametrize("num_seqs", [8, 16, 32])
def test_bitwise_vs_stock_kernel(num_seqs: int) -> None:
    rng = random.Random(7)
    stream = torch.cuda.Stream()
    mgr = _manager(stream)
    rids = list(range(100, 100 + SRC_MAX))
    for rid in rids:
        mgr.index_mapper.add_new_sequence(rid)
    for slot in range(SRC_MAX):
        _fill_row(mgr.host_kv_cache_block_offsets, slot, rng.randrange(1000, B + 1), rng)
    batch = rng.sample(rids, num_seqs)
    candidate = torch.empty(P, DST_MAX, 2, B, dtype=torch.int32, device="cuda")
    stock = torch.empty_like(candidate)
    first = None
    for _ in range(10):
        with torch.cuda.stream(stream):
            candidate.fill_(SENTINEL)
            stock.fill_(SENTINEL)
            mgr.copy_batch_block_offsets(candidate, batch, 1, 0, num_seqs)
        stream.synchronize()
        copy_idx = mgr.index_mapper.get_copy_index(batch, 0, 1)
        copy_batch_block_offsets_to_device(
            mgr.host_kv_cache_block_offsets,
            stock,
            copy_idx,
            mgr.index_scales,
            mgr.kv_offset,
            stream.cuda_stream,
        )
        stream.synchronize()
        assert torch.equal(candidate, stock)
        got = candidate.cpu()
        want_k, want_v = _expected(mgr.host_kv_cache_block_offsets, copy_idx.tolist())
        assert torch.equal(got[:, :num_seqs, 0], want_k)
        assert torch.equal(got[:, :num_seqs, 1], want_v)
        assert bool((got[:, num_seqs:] == SENTINEL).all()), "rows outside the batch are untouched"
        first = got if first is None else first
        assert torch.equal(first, got)


def test_queued_copy_does_not_see_the_next_batch() -> None:
    """The host rewrites the page table and the copy index for batch B while batch A's
    copy is still queued; A's destination must hold A's offsets, also through a chain
    of six copies over the three-slot snapshot ring."""
    rng = random.Random(3)
    stream = torch.cuda.Stream()
    mgr = _manager(stream)
    host = mgr.host_kv_cache_block_offsets
    rids = list(range(500, 532))
    for rid in rids:
        mgr.index_mapper.add_new_sequence(rid)
    for slot in range(32):
        _fill_row(host, slot, 1090, rng)
    torch.cuda.synchronize()

    batch_a, batch_b = rids[:24], list(reversed(rids[8:32]))
    want_a = _expected(host, mgr.index_mapper.get_copy_index(batch_a, 0, 1).tolist())
    dst_a = torch.zeros(P, DST_MAX, 2, B, dtype=torch.int32, device="cuda")
    dst_b = torch.zeros_like(dst_a)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        mgr.copy_batch_block_offsets(dst_a, batch_a, 1, 0, 24)
    for slot in range(32):
        _fill_row(host, slot, 1090, rng)
    want_b = _expected(host, mgr.index_mapper.get_copy_index(batch_b, 0, 1).tolist())
    with torch.cuda.stream(stream):
        mgr.copy_batch_block_offsets(dst_b, batch_b, 1, 0, 24)
    torch.cuda.synchronize()
    for dst, (want_k, want_v) in ((dst_a, want_a), (dst_b, want_b)):
        got = dst.cpu()
        assert torch.equal(got[:, :24, 0], want_k) and torch.equal(got[:, :24, 1], want_v)

    dsts = [torch.zeros_like(dst_a) for _ in range(6)]
    wants = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
    for j, dst in enumerate(dsts):
        batch = rng.sample(rids, 16 + j)
        wants.append(
            (_expected(host, mgr.index_mapper.get_copy_index(batch, 0, 1).tolist()), len(batch))
        )
        with torch.cuda.stream(stream):
            mgr.copy_batch_block_offsets(dst, batch, 1, 0, len(batch))
        for slot in range(32):
            _fill_row(host, slot, 1090, rng)
    torch.cuda.synchronize()
    for dst, ((want_k, want_v), n) in zip(dsts, wants):
        got = dst.cpu()
        assert torch.equal(got[:, :n, 0], want_k) and torch.equal(got[:, :n, 1], want_v)


def test_not_taken_under_graph_capture_or_without_opt_in() -> None:
    stream = torch.cuda.Stream()
    mgr = _manager(stream)
    dst = torch.zeros(P, DST_MAX, 2, B, dtype=torch.int32, device="cuda")
    assert mgr._can_copy_block_offsets_via_snapshot(dst)
    graph, capture_stream, x = (
        torch.cuda.CUDAGraph(),
        torch.cuda.Stream(),
        torch.zeros(4, device="cuda"),
    )
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph, stream=capture_stream):
        inside = mgr._can_copy_block_offsets_via_snapshot(dst)
        x.add_(1)
    assert not inside
    mgr.block_offset_snapshot_copy = False
    assert not mgr._can_copy_block_offsets_via_snapshot(dst)
