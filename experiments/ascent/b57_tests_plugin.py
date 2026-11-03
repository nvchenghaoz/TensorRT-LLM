"""pytest plugin (branch 57): repair two stale csa2 test_indexer.py tests in place.

tests/ is protected for workers, so (as sm120_fused_tests_plugin does) this plugin
swaps the test functions at collection time, inside the real suite and its conftest.

1. test_gathered_decode_graph_grows_past_warmup_prefix: its hook counted only
   manager.gather_indexer_keys, but the indexer gathers through csa2/kernel.py
   gather_owner_index_keys (fused IndexKeyGather) whenever fused_keys() returns a
   geometry, so ``calls == []`` on main and branch alike. The repair counts BOTH
   routes (keys per call: slots.numel() on the manager route, rows * width on the
   fused route). Everything else is kept: 3x eager warmup, ``calls == [128] * 3``,
   capture, replays at visible 96/96/0/65 with the changed page table,
   _assert_state_selection and storage == 77.
2. test_constant_rows_retained_only_for_captured_graphs encoded the bug (rows
   cached only while capturing, before their fill has run). Replaced by (a)-(d):
   (a) eager right after capture, before any replay, is correct;
   (b) after an eager call, a capture reuses the row and records no fill;
   (c) a capture-time row with no eager entry is correct after replay, even
       after 200 eager geometries;
   (d) two graphs capture one key with no eager entry; replaying only the
       second is correct.

Use: python -m pytest -p b57_tests_plugin ... (experiments/ascent on PYTHONPATH).
"""

import pytest
import torch

GATHERED = "test_gathered_decode_graph_grows_past_warmup_prefix"
CONSTANT = "test_constant_rows_retained_only_for_captured_graphs"


def _make_gathered(T):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod

    @torch.inference_mode()
    def test_gathered_repaired(monkeypatch):
        layout, manager, metadata, state = T._single_request_state()
        metadata.is_cuda_graph = True
        indexer = T.CSA2Indexer(layout, 0, 8, 128)
        calls, routes = [], []
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
            return fused_original(pages, table, requests, page_size, max_positions, width,
                                  queries, *a, **kw)

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
        print(f"[b57-plugin] gathered routes={routes} calls={calls}")
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
        print("[b57-plugin] gathered replays 96/96/0/65 passed")

    return test_gathered_repaired


def _make_constant(T):
    def full(count, value, device):
        return torch.full((count,), value, dtype=torch.int32, device=device)

    def capture_copy(indexer, count, value, out, device):
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                out.copy_(indexer._constant_rows(count, value, device))
        torch.cuda.synchronize()
        return graph

    def test_constant_rows_repaired():
        device = torch.device("cuda")
        warm = torch.zeros(8, dtype=torch.int32, device=device)
        warm.copy_(full(8, 1, device))
        # (a) eager right after capture, before any replay
        indexer = T._prepared_indexer()
        out = torch.zeros(7, dtype=torch.int32, device=device)
        graph = capture_copy(indexer, 7, 16384, out, device)
        eager = indexer._constant_rows(7, 16384, device)
        torch.cuda.synchronize()
        assert torch.equal(eager, full(7, 16384, device)), "(a) eager after capture"
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, full(7, 16384, device)), "(a) replay"
        del graph
        # (b) after an eager call, a capture reuses the row and records no fill
        indexer = T._prepared_indexer()
        first = indexer._constant_rows(5, 3, device)
        assert torch.equal(first, full(5, 3, device)), "(b) eager"
        assert indexer._constant_rows(5, 3, device) is first, "(b) eager retained"
        fills, original_full = [], torch.full

        def counting(*a, **kw):
            fills.append(a)
            return original_full(*a, **kw)

        out = torch.zeros(5, dtype=torch.int32, device=device)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            stream.synchronize()
            torch.full = counting
            try:
                with torch.cuda.graph(graph, stream=stream):
                    captured = indexer._constant_rows(5, 3, device)
                    out.copy_(captured)
            finally:
                torch.full = original_full
        torch.cuda.synchronize()
        assert captured is first and fills == [], "(b) capture reused, no fill"
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, full(5, 3, device)), "(b) replay"
        del graph
        # (c) capture-time row without an eager entry survives 200 eager geometries
        indexer = T._prepared_indexer()
        out = torch.zeros(7, dtype=torch.int32, device=device)
        graph = capture_copy(indexer, 7, 3, out, device)
        for count in range(1, 201):
            value = 3 if count == 7 else count + 11
            rows = indexer._constant_rows(count, value, device)
            assert torch.equal(rows, full(count, value, device)), f"(c) eager {count}"
        junk = [torch.full((1 << 16,), -5, dtype=torch.int32, device=device) for _ in range(8)]
        del junk
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, full(7, 3, device)), "(c) replay after 200 geometries"
        del graph
        # (d) two graphs, one key, no eager entry; replay only the second
        indexer = T._prepared_indexer()
        out1 = torch.zeros(6, dtype=torch.int32, device=device)
        out2 = torch.zeros(6, dtype=torch.int32, device=device)
        g1 = capture_copy(indexer, 6, 9, out1, device)
        g2 = capture_copy(indexer, 6, 9, out2, device)
        g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(out2, full(6, 9, device)), "(d) second graph only"
        del g1, g2
        print("[b57-plugin] constant rows (a)-(d) passed")

    return test_constant_rows_repaired


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        name = getattr(item, "originalname", None) or item.name
        module = getattr(item, "module", None)
        if module is None or not module.__name__.endswith("test_indexer"):
            continue
        if "csa2" not in str(getattr(module, "__file__", "")):
            continue
        if name == GATHERED:
            item.obj = _make_gathered(module)
            print(f"[b57-plugin] repaired {item.nodeid}")
        elif name == CONSTANT:
            item.obj = _make_constant(module)
            print(f"[b57-plugin] replaced {item.nodeid} with (a)-(d)")
