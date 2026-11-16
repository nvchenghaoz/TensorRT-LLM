# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The read-rows index arithmetic of `ops/attention/_flashinfer_paged.py`, on the CPU.

The last layer's context attention runs each request's trailing n query rows only. n is the
smallest count whose packed rows (n * group) exceed the full-row plan's query tile, so the
planner keeps that tile, and for which the dropped rows ((seq_len - n) * group) fill whole
tiles, so the kept last row sits at the same in-tile offset as in the full-row plan.

`paged_attention` itself (the PagedAttention entry) is not driven here: its cells need the
engine's KV cache manager and FlashInfer metadata, and the target's GSM8K accuracy test
(`accuracy/modeling_v2/gemma4/gemma4_31b_it_nvfp4__sm_120__tp1`) runs it end to end. The
target's FA2 prefill kernels are checked bitwise against FlashInfer's in
`../test_modeling_v2_flashinfer_jit.py`.
"""

import math

import numpy as np
import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1.ops.attention._flashinfer_paged import (  # noqa: E501
    READ_ROWS_CTA_TILES_Q,
    build_read_rows_indices,
    read_rows_subset_lengths,
)

pytestmark = pytest.mark.cpu_only

#: (q heads per kv head, query tile): full attention 32 / 4 at both FA2 tiles; sliding 32 / 16.
_PLANS = [(8, t) for t in READ_ROWS_CTA_TILES_Q] + [(2, 64), (1, 16)]
_SEQ_LENS = np.array([9, 16, 17, 300, 1000, 1023, 2000, 8192])


@pytest.mark.parametrize("group, tile", _PLANS)
def test_subset_lengths_keep_the_tile_and_the_in_tile_offset(group, tile):
    n = read_rows_subset_lengths(_SEQ_LENS[_SEQ_LENS > tile], group, tile)
    seq_lens = _SEQ_LENS[_SEQ_LENS > tile]
    assert n is not None and n.dtype == np.int64
    assert (n * group > tile).all()
    assert ((seq_lens - n) * group % tile == 0).all()
    assert (n <= seq_lens).all()
    # Smallest such count: the congruence repeats every tile / gcd(tile, group) rows.
    period = tile // math.gcd(tile, group)
    assert ((n - period) * group <= tile).all()


@pytest.mark.parametrize("group, tile", _PLANS)
def test_subset_lengths_are_minimal_by_search(group, tile):
    for seq_len in range(1, 400):
        got = read_rows_subset_lengths(np.array([seq_len]), group, tile)
        want = next(
            (
                n
                for n in range(1, seq_len + 1)
                if n * group > tile and (seq_len - n) * group % tile == 0
            ),
            None,
        )
        assert (None if got is None else int(got[0])) == want, (seq_len, got, want)


def test_one_short_request_declines_the_whole_step():
    group, tile = 8, 64
    assert read_rows_subset_lengths(np.array([2000, 8]), group, tile) is None
    assert read_rows_subset_lengths(np.array([2000, 9]), group, tile) is not None


def test_indices_pick_each_requests_trailing_rows():
    seq_lens = np.array([300, 1000, 17, 2000])
    qo_indptr = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32)
    n = read_rows_subset_lengths(seq_lens, 8, 64)
    sub_indptr, q_rows, keep, dst = build_read_rows_indices(qo_indptr, n)

    assert sub_indptr.dtype == np.int32 and q_rows.dtype == keep.dtype == dst.dtype == np.int64
    assert sub_indptr.tolist() == [0, *np.cumsum(n).tolist()]
    want_rows = np.concatenate([np.arange(e - c, e) for e, c in zip(qo_indptr[1:], n)])
    assert q_rows.tolist() == want_rows.tolist()
    assert keep.tolist() == (sub_indptr[1:] - 1).tolist()
    assert dst.tolist() == (qo_indptr[1:] - 1).tolist()
    # The kept row of each sub-request is its request's last row.
    assert q_rows[keep].tolist() == dst.tolist()
