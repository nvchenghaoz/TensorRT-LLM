"""Repaired copies of two csa2 test_indexer.py tests (tests/ is protected for workers).

1. test_gathered_decode_graph_grows_past_warmup_prefix: its hook counted only
   manager.gather_indexer_keys, but the indexer gathers through
   csa2/kernel.py gather_owner_index_keys (fused IndexKeyGather) whenever
   fused_keys() returns a geometry, so ``calls == []`` on main and branch alike.
   The repair counts BOTH routes (keys gathered per call: slots.numel() on the
   manager route, rows * width on the fused route). Everything else is kept:
   3x eager warmup, ``calls == [128] * 3``, capture, replays at visible
   96/96/0/65 with the changed page table, _assert_state_selection and
   storage == 77.
2. test_constant_rows_retained_only_for_captured_graphs encoded the bug (rows
   are cached only while capturing, before their fill has run). Replaced by
   (a)-(d): (a) eager right after capture, before any replay, is correct;
   (b) after an eager call a capture reuses the row and records no fill;
   (c) a capture-time row with no eager entry is correct after replay, even
   after 200 eager geometries; (d) two graphs capture one key with no eager
   entry and replaying only the second is correct.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "unittest"))
_spec = importlib.util.spec_from_file_location(
    "csa2_test_indexer_orig",
    ROOT / "tests/unittest/_torch/attention/sparse/csa2/test_indexer.py",
)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@needs_cuda
@torch.inference_mode()
def test_gathered_decode_graph_grows_past_warmup_prefix_repaired(monkeypatch):
    layout, manager, metadata, state = T._single_request_state()
    metadata.is_cuda_graph = True
    indexer = indexer_mod.CSA2Indexer(layout, 0, 8, 128)
    calls = []
    routes = []
    original = manager.gather_indexer_keys

    def gather(owner, slots):
        calls.append(slots.numel())
        routes.append("manager")
        return original(owner, slots)

    monkeypatch.setattr(manager, "gather_indexer_keys", gather)
    fused_original = indexer_mod.gather_owner_index_keys

    def fused(pages, table, requests, page_size, max_positions, width, queries, *a, **kw):
        calls.append(int(requests.shape[0]) * int(width))
        routes.append("fused")
        return fused_original(pages, table, requests, page_size, max_positions, width, queries,
                              *a, **kw)

    monkeypatch.setattr(indexer_mod, "gather_owner_index_keys", fused)
    # Exercise the bounded fallback independently of native head coverage.
    monkeypatch.setattr(metadata, "prepare_indexer", lambda layer: None)
    monkeypatch.setattr(
        indexer,
        "_call_paged_mqa_logits",
        lambda *a, **kw: pytest.fail("Unavailable native staging requires gathered fallback"),
    )
    for _ in range(3):
        output = indexer(state, 0, 1)
    print(f"[repaired] routes={routes} calls={calls}")
    assert calls == [128] * 3
    torch.testing.assert_close(
        output[0, :2], torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 1)
    for length, second_page in ((96, 1), (96, 2), (0, -1), (65, 1)):
        metadata.csa2_visible_lengths[0].fill_(length)
        metadata.csa2_global_page_tables[0][0, 1] = second_page
        graph.replay()
        torch.cuda.synchronize()
        T._assert_state_selection(output, state, 0)
        if length > 64:
            assert bool((output >= 64).any())
    assert bool((manager.storage == 77).all())


def _capture_copy(indexer, count, value, out, device):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            out.copy_(indexer._constant_rows(count, value, device))
    torch.cuda.synchronize()
    return graph


def _full(count, value, device):
    return torch.full((count,), value, dtype=torch.int32, device=device)


@needs_cuda
def test_constant_rows_a_eager_after_capture_before_replay():
    indexer = T._prepared_indexer()
    device = torch.device("cuda")
    out = torch.zeros(7, dtype=torch.int32, device=device)
    graph = _capture_copy(indexer, 7, 16384, out, device)
    eager = indexer._constant_rows(7, 16384, device)
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, _full(7, 16384, device), rtol=0, atol=0)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _full(7, 16384, device), rtol=0, atol=0)


@needs_cuda
def test_constant_rows_b_capture_reuses_eager_row_without_fill(monkeypatch):
    indexer = T._prepared_indexer()
    device = torch.device("cuda")
    eager = indexer._constant_rows(5, 3, device)
    torch.testing.assert_close(eager, _full(5, 3, device), rtol=0, atol=0)
    assert indexer._constant_rows(5, 3, device) is eager
    fills = []
    original = torch.full

    def counting(*a, **kw):
        fills.append(a)
        return original(*a, **kw)

    out = torch.zeros(5, dtype=torch.int32, device=device)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.synchronize()
        monkeypatch.setattr(torch, "full", counting)
        with torch.cuda.graph(graph, stream=stream):
            captured = indexer._constant_rows(5, 3, device)
            out.copy_(captured)
        monkeypatch.setattr(torch, "full", original)
    torch.cuda.synchronize()
    assert captured is eager
    assert fills == []
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _full(5, 3, device), rtol=0, atol=0)


@needs_cuda
def test_constant_rows_c_captured_row_survives_eager_geometries():
    indexer = T._prepared_indexer()
    device = torch.device("cuda")
    out = torch.zeros(7, dtype=torch.int32, device=device)
    graph = _capture_copy(indexer, 7, 3, out, device)
    for count in range(1, 201):
        value = 3 if count % 50 == 7 else count + 11
        rows = indexer._constant_rows(count, value, device)
        torch.testing.assert_close(rows, _full(count, value, device), rtol=0, atol=0)
    # Recycle freed blocks so a stale row would be overwritten.
    junk = [torch.full((1 << 16,), -5, dtype=torch.int32, device=device) for _ in range(8)]
    del junk
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _full(7, 3, device), rtol=0, atol=0)


@needs_cuda
def test_constant_rows_d_two_graphs_one_key_replay_second_only():
    indexer = T._prepared_indexer()
    device = torch.device("cuda")
    out1 = torch.zeros(6, dtype=torch.int32, device=device)
    out2 = torch.zeros(6, dtype=torch.int32, device=device)
    first = _capture_copy(indexer, 6, 9, out1, device)
    second = _capture_copy(indexer, 6, 9, out2, device)
    second.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, _full(6, 9, device), rtol=0, atol=0)
    del first
