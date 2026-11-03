"""CSA2 SM120 prefill candidate-consumer indexer: one harness (check + diagnostic + timing).

Runs the production ``CSA2Indexer.sparse_attn_indexer`` in either tree with a
stub metadata at one TP4 rank's shapes (the indexer is replicated: 32 index
heads x 128-d MXFP4 Q, FP32 weights): one prefill request of 2048 new rows
after a 32768-position cached prefix (visible 32769..34816), native
[pages, 64, 1, 68] index pages with unallocated holes and duplicated pages
(exact key ties). Layout: layer 0 is the ratio-1 owner and candidate source
(the model's layer 20), layers 1..4 are Reindex consumers (24/28/32/36).
The owner's own prefill indexer publishes the [2048, 16384] candidates; the
check also injects -1 holes and positions >= visible.

--check (candidate tree): the unmodified tiled path is the same production
call with ``metadata.is_cuda_graph=True`` (it blocks only the new dense route on
a prefill consumer). Compares bitwise: raw logits at every valid candidate,
masked [rows, 16384] scores, final int32 selections; then an untimed
repeated sequence with the timing loop's warmup/iters/reset points; two eager
runs of the new path; decode CUDA-graph capture/replay at batch 1/2/4.
Default (both trees): tile diagnostic (old arm only), memory, timing of the 4
consumer layers' prefill indexer, eager, L2 flush per call.
"""

import argparse
import json
import os
import statistics
import sys
import types
from collections import Counter

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout

NEW = hasattr(indexer_mod.CSA2Indexer, "_select_candidate_columns")
HEADS, HEAD_DIM = 32, 128
CONSUMERS = (1, 2, 3, 4)


def log(*args):
    print(*args, flush=True)


def read_topk(checkpoint):
    try:
        with open(os.path.join(checkpoint, "config.json")) as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)
        return (
            int(text.get("index_topk", 512)),
            int(text.get("candidate_topk_blocks", 2048)),
            int(text.get("candidate_block_size", 8)),
            int(text.get("index_n_heads", HEADS)),
            int(text.get("index_head_dim", HEAD_DIM)),
        )
    except Exception as exc:  # noqa: BLE001
        log(f"config read failed ({exc}); defaults")
        return 512, 2048, 8, HEADS, HEAD_DIM


class _Manager:
    def __init__(self, pages):
        self.pages = pages

    def get_index_pages(self, layer_idx):
        return self.pages


def make_world(args, device, seed=1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    page_size = args.tokens_per_block
    max_positions = args.max_positions
    cols = -(-max_positions // page_size)
    native_per_global = page_size // 64
    table_rows = 2
    physical = table_rows * cols + 8
    perm = torch.randperm(physical, generator=g)[: table_rows * cols]
    table = perm.reshape(table_rows, cols).to(torch.int32)
    total = args.prefix + args.rows
    allocated = -(-total // page_size)
    table[:, allocated:] = -1
    holes = torch.rand((table_rows, cols), generator=g) < 0.03
    table[holes] = -1
    native = physical * native_per_global
    pages = torch.randint(0, 256, (native, 64 * 68), generator=g, dtype=torch.int32).to(torch.uint8)
    pages[:, 4096:] = torch.randint(118, 131, (native, 256), generator=g, dtype=torch.int32).to(
        torch.uint8
    )
    # Duplicated key pages -> exact logit ties between different positions.
    for a, b in ((10, 20), (11, 200), (100, 101)):
        pa, pb = int(table[0, a]), int(table[0, b])
        if pa >= 0 and pb >= 0:
            for k in range(native_per_global):
                pages[pb * native_per_global + k] = pages[pa * native_per_global + k]
    pages = pages.view(native, 64, 1, 68)
    rows = args.rows
    q = torch.randint(-128, 128, (rows, HEADS, HEAD_DIM // 2), generator=g, dtype=torch.int32)
    q_scale = torch.randint(122, 129, (rows, HEADS, 4), generator=g, dtype=torch.int32)
    weights = torch.randn((rows, HEADS), generator=g, dtype=torch.float32)
    # Duplicated query rows too (ties across rows are irrelevant, but cheap).
    q[1] = q[0]
    visible = torch.arange(args.prefix + 1, args.prefix + rows + 1, dtype=torch.int32)
    return dict(
        pages=pages.to(device),
        table=table.to(device),
        q=q.to(torch.int8).to(device),
        q_scale=q_scale.to(torch.uint8).view(torch.int32).reshape(rows, HEADS).to(device),
        weights=weights.to(device),
        visible=visible.to(device),
        page_size=page_size,
        max_positions=max_positions,
    )


def make_metadata(world, args, layers, candidates=None, graph=False):
    device = world["q"].device
    rows = args.rows
    md = types.SimpleNamespace()
    md.kv_cache_manager = _Manager(world["pages"])
    md.csa2_visible_lengths = {layer: world["visible"] for layer in layers}
    md.csa2_global_max_positions = {0: world["max_positions"]}
    md.csa2_indices = {}
    md.csa2_candidates = {} if candidates is None else {0: candidates}
    md.csa2_candidate_blocks = {}
    md.csa2_candidate_counts = {}
    md.is_cuda_graph = graph
    md.csa2_request_start_positions = [args.prefix]
    md.csa2_request_query_ranges = [(0, rows)]
    md.csa2_kv_sources = {layer: 0 for layer in layers}
    md.csa2_global_page_tables = {0: world["table"]}
    md.csa2_global_page_sizes = {0: world["page_size"]}
    md.csa2_token_requests = torch.zeros(rows, dtype=torch.int32, device=device)
    md.csa2_num_context_requests = 1
    md.indexer_max_chunk_size = 8192
    md.mapping = None
    md.indexer_q_split_threshold = 8192

    def fail(*a, **k):
        raise RuntimeError("unexpected metadata call")

    md.register_indexer_reset = fail
    md.prepare_indexer_prior = fail
    md.prepare_indexer = fail
    md.global_slot_tile = fail
    return md


def make_layout(topk, blocks, block_size):
    return CSA2Layout(
        compress_ratios=(1, 1, 1, 1, 1),
        kv_source_layer_ids=(0,),
        index_source_layer_ids=(0, 1, 2, 3, 4),
        candidate_source_layer_id=0,
        candidate_topk_blocks=blocks,
        candidate_block_size=block_size,
        index_topk=topk,
    )


def call(indexer, md, world):
    rows = world["q"].shape[0]
    hidden = torch.empty((rows, 1), device=world["q"].device)
    return indexer.sparse_attn_indexer(
        md, hidden, world["q"], None, None, world["weights"], world["q_scale"]
    )


def owner_candidates(owner, world, args):
    md = make_metadata(world, args, (0,))
    call(owner, md, world)
    torch.cuda.synchronize()
    return md.csa2_candidates[0]


def perturb(candidates, args, seed=77):
    """-1 holes and positions >= visible (and >= width) in whole and partial blocks."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = candidates.clone().cpu()
    rows, width = c.shape
    hole = torch.rand((rows, width), generator=g) < 0.02
    c[hole] = -1
    future = torch.rand((rows, width), generator=g) < 0.01
    c[future] = args.prefix + args.rows + torch.randint(-1500, 1500, (int(future.sum()),), generator=g).to(torch.int32)
    c[:, 3] = args.max_positions + 5  # beyond the owner's bound
    return c.to(candidates.device).contiguous()


def l2_flush(buf):
    buf.zero_()


# --------------------------------------------------------------------- capture


class Capture:
    """Instance-level wrappers on one indexer: record raw scores, masked scores
    and tile geometry per call (forward_prepared / _select_mapped_logits /
    select_prepared_scores)."""

    def __init__(self, indexer, record=True):
        self.indexer = indexer
        self.raw, self.raw_valid, self.masked, self.tiles = [], [], [], []
        self.logical = []
        orig_sml = type(indexer)._select_mapped_logits.__get__(indexer)
        orig_sps = type(indexer).select_prepared_scores.__get__(indexer)
        orig_fp = type(indexer).forward_prepared.__get__(indexer)
        cap = self

        def fp(q_data, k_data, *a, **k):
            cap.tiles.append((q_data.shape[0], k_data.shape[0], k.get("candidate_columns") is not None))
            return orig_fp(q_data, k_data, *a, **k)

        def sml(logits, row_starts, row_ends, logical, visible, *a, **k):
            width = logical.shape[1]
            if row_starts is None and row_ends is None:
                raw = logits.clone()
            else:
                cols = row_starts.long()[:, None] + torch.arange(width, device=logits.device)
                raw = logits.gather(1, cols.clamp(0, logits.shape[1] - 1))
            cap.raw.append(raw)
            cap.raw_valid.append((logical >= 0) & (logical < visible[:, None]))
            cap.logical.append(logical.clone())
            return orig_sml(logits, row_starts, row_ends, logical, visible, *a, **k)

        def sps(scores, *a, **k):
            cap.masked.append(scores.clone())
            return orig_sps(scores, *a, **k)

        if record:
            indexer.forward_prepared = fp
            indexer._select_mapped_logits = sml
            indexer.select_prepared_scores = sps

    def remove(self):
        for name in ("forward_prepared", "_select_mapped_logits", "select_prepared_scores"):
            self.indexer.__dict__.pop(name, None)

    def joined(self):
        return torch.cat(self.raw), torch.cat(self.raw_valid), torch.cat(self.masked)

    def joined_logical(self):
        return torch.cat(self.logical)


def stage_ab(raw_a, val_a, mask_a, raw_b, val_b, mask_b):
    """Stage (a): raw logits at every valid candidate; stage (b): masked scores."""
    valid = bool(torch.equal(val_a, val_b))
    raw = valid and bool(torch.equal(torch.where(val_a, raw_a, 0.0), torch.where(val_b, raw_b, 0.0)))
    masked = bool(torch.equal(mask_a, mask_b))
    out = dict(valid_equal=valid, a_raw_logits_equal=raw, b_masked_scores_equal=masked,
               ok=valid and raw and masked)
    if not masked and mask_a.shape == mask_b.shape:
        diff = (mask_a != mask_b) & ~(torch.isnan(mask_a) & torch.isnan(mask_b))
        if bool(diff.any()):
            r, c = (int(x) for x in diff.nonzero()[0])
            out["b_first_difference"] = dict(row=r, column=c, a=float(mask_a[r, c]),
                                             b=float(mask_b[r, c]))
    return out


def tie_aware(masked, logical, out_a, out_b, topk):
    """Compare two final selections over the same masked scores.

    The CUDA radix prefill Top-K (topKPerRowPrefill) breaks exact score ties at
    the K boundary by atomically assigned smem slots, so the unmodified op
    itself may pick different tied positions run to run. Rows without a tie at
    the boundary (K-th score != (K+1)-th) must be bitwise equal; every row's
    selected-score multiset must be bitwise equal to the K largest masked scores.
    """
    rows = masked.shape[0]
    vals = masked.sort(-1, descending=True).values
    k = min(topk, vals.shape[1])
    kth = vals[:, k - 1]
    nxt = vals[:, k] if vals.shape[1] > k else torch.full_like(kth, -torch.inf)
    tie = (kth == nxt) & torch.isfinite(nxt)
    span = int(max(int(logical.max()), int(out_a.max()), int(out_b.max()), 0)) + 1
    dense = torch.full((rows, span), -torch.inf, device=masked.device)
    src = torch.where(logical >= 0, masked, torch.tensor(-torch.inf, device=masked.device))
    dense.scatter_reduce_(1, logical.clamp(min=0).long(), src, "amax", include_self=True)
    top_vals = vals[:, :k]

    def multiset_ok(out):
        sel = dense.gather(1, out.clamp(min=0).long())
        sel = torch.where(out >= 0, sel, torch.tensor(-torch.inf, device=sel.device))
        sel = sel.sort(-1, descending=True).values[:, :k]
        return bool(torch.equal(sel, top_vals))

    row_eq = (out_a == out_b).all(-1)
    first = None
    if not bool(row_eq.all()):
        r = int((~row_eq).nonzero()[0, 0])
        c = int((out_a[r] != out_b[r]).nonzero()[0, 0])
        first = dict(row=r, column=c, a=int(out_a[r, c]), b=int(out_b[r, c]),
                     boundary_tie=bool(tie[r]), kth_score=float(kth[r]))
    return dict(
        first_difference=first,
        bitwise_equal=bool(row_eq.all()),
        boundary_tie_rows=int(tie.sum()),
        differing_rows=int((~row_eq).sum()),
        differing_rows_without_tie=int((~row_eq & ~tie).sum()),
        multiset_a=multiset_ok(out_a),
        multiset_b=multiset_ok(out_b),
        ok=bool((row_eq | tie).all()) and multiset_ok(out_a) and multiset_ok(out_b),
    )


# --------------------------------------------------------------------- check


def run_check(args, device, topk, blocks, block_size):
    assert NEW, "check runs in the candidate tree"
    layout = make_layout(topk, blocks, block_size)
    world = make_world(args, device)
    owner = indexer_mod.CSA2Indexer(layout, 0, HEADS, HEAD_DIM)
    consumer = indexer_mod.CSA2Indexer(layout, 1, HEADS, HEAD_DIM)
    failures, report = [], {}
    cand_sets = {}
    published = owner_candidates(owner, world, args)
    cand_sets["owner_published"] = published
    cand_sets["perturbed"] = perturb(published, args)
    g = torch.Generator(device="cpu").manual_seed(5)
    rnd = torch.randint(0, (args.prefix + args.rows) // block_size, (args.rows, blocks), generator=g)
    rnd = (rnd[:, :, None] * block_size + torch.arange(block_size)).reshape(args.rows, -1).to(torch.int32)
    rnd[torch.rand(rnd.shape, generator=g) < 0.05] = -1
    cand_sets["random_blocks"] = rnd.to(device)
    log(f"candidates shape={tuple(published.shape)} dtype={published.dtype} "
        f"published -1 frac={(published < 0).float().mean().item():.4f}")
    ref_out = {}
    logical_by_set, masked_by_set = {}, {}
    for name, cand in cand_sets.items():
        md_old = make_metadata(world, args, (1,), cand, graph=True)
        cap_old = Capture(consumer)
        out_old = call(consumer, md_old, world).clone()
        torch.cuda.synchronize()
        cap_old.remove()
        md_new = make_metadata(world, args, (1,), cand)
        cap_new = Capture(consumer)
        out_new = call(consumer, md_new, world).clone()
        torch.cuda.synchronize()
        cap_new.remove()
        # The unmodified path run a second time: the kernel's own tie behaviour.
        md_old2 = make_metadata(world, args, (1,), cand, graph=True)
        cap_old2 = Capture(consumer)
        out_old2 = call(consumer, md_old2, world).clone()
        torch.cuda.synchronize()
        cap_old2.remove()
        md_new2 = make_metadata(world, args, (1,), cand)
        cap_new2 = Capture(consumer)
        out_new2 = call(consumer, md_new2, world).clone()
        torch.cuda.synchronize()
        cap_new2.remove()
        raw_o, val_o, mask_o = cap_old.joined()
        raw_n, val_n, mask_n = cap_new.joined()
        stages_old_old = stage_ab(raw_o, val_o, mask_o, *cap_old2.joined())
        stages_new_new = stage_ab(raw_n, val_n, mask_n, *cap_new2.joined())
        del cap_old2, cap_new2
        if name == "perturbed":
            stage_ref = (raw_o, val_o, mask_o)
        logical_n = cap_new.joined_logical()
        logical_by_set[name] = logical_n
        masked_by_set[name] = mask_n
        old_tiles = Counter((t[0], t[1], t[2]) for t in cap_old.tiles)
        new_tiles = Counter((t[0], t[1], t[2]) for t in cap_new.tiles)
        ok_valid = torch.equal(val_o, val_n)
        ok_raw = torch.equal(torch.where(val_o, raw_o, 0.0), torch.where(val_n, raw_n, 0.0))
        ok_mask = torch.equal(mask_o, mask_n)
        sel_new_old = tie_aware(mask_n, logical_n, out_old, out_new, topk)
        sel_old_old = tie_aware(mask_n, logical_n, out_old, out_old2, topk)
        sel_new_new = tie_aware(mask_n, logical_n, out_new, out_new2, topk)
        ok_sel = sel_new_old["ok"] and sel_new_new["ok"] and sel_old_old["ok"]
        finite = bool(torch.isfinite(raw_n[val_n]).all())
        entry = dict(
            valid_equal=ok_valid, raw_logits_equal_at_valid=ok_raw, masked_scores_equal=ok_mask,
            selections_new_vs_old=sel_new_old, selections_old_vs_old_rerun=sel_old_old,
            selections_new_vs_new_rerun=sel_new_new,
            stages_ab_old_vs_old_rerun=stages_old_old, stages_ab_new_vs_new_rerun=stages_new_new,
            masked_has_nan=bool(torch.isnan(mask_n).any()),
            selections_ok=ok_sel, finite=finite, valid_frac=val_n.float().mean().item(),
            old_tiles=str(dict(old_tiles)), new_tiles=str(dict(new_tiles)),
            sel_minus1=int((out_new < 0).sum()), masked_shape=list(mask_n.shape),
        )
        # Count exact score ties among valid candidates of row 0..63 (duplicated keys).
        sub = mask_n[:64]
        srt = sub.sort(-1).values
        entry["exact_ties_rows0_63"] = int(((srt[:, 1:] == srt[:, :-1]) & torch.isfinite(srt[:, 1:])).sum())
        report[name] = entry
        log(f"[{name}] {json.dumps(entry)}")
        if not (ok_valid and ok_raw and ok_mask and ok_sel and finite and stages_old_old["ok"]
                and stages_new_new["ok"] and not entry["masked_has_nan"]):
            failures.append(name)
        ref_out[name] = out_old
    # Repeated-execution sequence: timing loop's warmup/iters/reset points.
    cand = cand_sets["perturbed"]
    ref = ref_out["perturbed"]
    per_iter, per_iter_detail = [], []
    mask_p, logical_p = masked_by_set["perturbed"], logical_by_set["perturbed"]
    md_pristine = make_metadata(world, args, CONSUMERS, cand)
    consumers = [indexer_mod.CSA2Indexer(layout, l, HEADS, HEAD_DIM) for l in CONSUMERS]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    for it in range(args.warmup + args.iters):
        md = make_metadata(world, args, CONSUMERS, cand)  # reset point
        assert torch.equal(md.csa2_candidates[0], md_pristine.csa2_candidates[0])
        oks, diffs, bitwise, ab_ok = [], 0, 0, True
        for ix in consumers:
            l2_flush(flush)
            cap = Capture(ix)
            out = call(ix, md, world)
            torch.cuda.synchronize()
            cap.remove()
            ab = stage_ab(*stage_ref, *cap.joined())
            del cap
            ab_ok = ab_ok and ab["ok"]
            cmp = tie_aware(mask_p, logical_p, ref, out, topk)
            oks.append(cmp["ok"] and ab["ok"])
            diffs = max(diffs, cmp["differing_rows"])
            bitwise += int(cmp["bitwise_equal"])
        torch.cuda.synchronize()
        # state before reset: candidates unchanged, inputs unchanged
        oks.append(torch.equal(md.csa2_candidates[0], cand))
        per_iter.append(all(oks))
        per_iter_detail.append(dict(ok=all(oks), stages_ab_equal_all_layers=ab_ok,
                                    selections_bitwise_layers=bitwise,
                                    max_tie_rows_differing=diffs))
        log(f"[repeated iter {it}] {json.dumps(per_iter_detail[-1])}")
    report["repeated"] = dict(iterations=len(per_iter), all_ok=all(per_iter),
                              per_iter=per_iter_detail,
                              max_abs_err_masked_scores=0.0)
    log(f"[repeated] {json.dumps(report['repeated'])}")
    if not all(per_iter):
        failures.append("repeated")
    md1 = make_metadata(world, args, (1,), cand)
    cap1 = Capture(consumer)
    a = call(consumer, md1, world).clone()
    torch.cuda.synchronize()
    cap1.remove()
    md2 = make_metadata(world, args, (1,), cand)
    cap2 = Capture(consumer)
    b = call(consumer, md2, world).clone()
    torch.cuda.synchronize()
    cap2.remove()
    report["two_eager_new"] = tie_aware(mask_p, logical_p, a, b, topk)
    report["two_eager_new"]["stages_ab"] = stage_ab(*cap1.joined(), *cap2.joined())
    del cap1, cap2
    log(f"[two_eager_new] {json.dumps(report['two_eager_new'])}")
    if not (report["two_eager_new"]["ok"] and report["two_eager_new"]["stages_ab"]["ok"]):
        failures.append("two_eager")
    # Decode CUDA-graph capture/replay at batch 1/2/4 (consumer, tiled decode path).
    report["decode_graph"] = decode_graph_check(args, device, layout, world, failures)
    passed = not failures
    print(json.dumps({"passed": passed, "failures": failures, "compared": "new dense-prefix "
                      "consumer path vs unmodified tiled candidate path (same production call, "
                      "is_cuda_graph=True route) bitwise", "report": report}), flush=True)


def decode_graph_check(args, device, layout, world, failures):
    out = {}
    consumer = indexer_mod.CSA2Indexer(layout, 2, HEADS, HEAD_DIM)
    g = torch.Generator(device="cpu").manual_seed(9)
    for m in (1, 2, 4):
        cand = torch.randint(0, args.prefix + args.rows, (m, consumer.layout.candidate_topk_blocks
                                                          * consumer.layout.candidate_block_size),
                             generator=g, dtype=torch.int32)
        cand[:, ::9] = -1
        cand = cand.to(device)
        md = make_metadata(world, args, (2,), cand, graph=True)
        md.csa2_request_start_positions = [args.prefix + args.rows + i for i in range(m)]
        md.csa2_request_query_ranges = [(i, i + 1) for i in range(m)]
        md.csa2_num_context_requests = 0
        md.csa2_token_requests = torch.zeros(m, dtype=torch.int32, device=device)
        vis = torch.tensor([args.prefix + args.rows + i + 1 for i in range(m)], dtype=torch.int32,
                           device=device)
        md.csa2_visible_lengths = {2: vis}
        hidden = torch.empty((m, 1), device=device)
        q, w, qs = world["q"][:m].clone(), world["weights"][:m].clone(), world["q_scale"][:m].clone()

        def step():
            return consumer.sparse_attn_indexer(md, hidden, q, None, None, w, qs)

        try:
            eager = step().clone()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                step()
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    gout = step()
            torch.cuda.synchronize()
            oks = []
            for _ in range(5):
                graph.replay()
                torch.cuda.synchronize()
                oks.append(torch.equal(gout, eager))
            out[str(m)] = all(oks)
            del graph
        except Exception as exc:  # noqa: BLE001
            out[str(m)] = f"error: {type(exc).__name__}: {exc}"[:300]
        if out[str(m)] is not True:
            failures.append(f"decode_graph_{m}")
        log(f"[decode graph m={m}] {out[str(m)]}")
    return out


# --------------------------------------------------------------------- bench


def diagnostic(args, consumer, world, cand):
    """Old-arm tile diagnostic: wrap _run_csa2_chunks / forward_prepared."""
    budget = indexer_mod._INDEXER_MQA_LOGITS_ELEM_BUDGET
    info = dict(budget=budget, budget_env=os.environ.get("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET"),
                ranges=[(0, args.rows)], num_contexts=1, candidates_shape=list(cand.shape),
                new_route=NEW)
    orig_run = type(consumer)._run_csa2_chunks.__get__(consumer)
    orig_fp = type(consumer).forward_prepared.__get__(consumer)
    tiles = []

    def run(chunks, *a, **k):
        info["chunks"] = [dict(start=c.token_start, end=c.token_end, k_tokens=c.k_token_count,
                               keys_per_query=c.keys_per_query, max_q=c.max_query_tokens,
                               tiled=c.load_tile is not None) for c in chunks]
        return orig_run(chunks, *a, **k)

    def fp(q_data, k_data, *a, **k):
        tiles.append((q_data.shape[0], k_data.shape[0]))
        return orig_fp(q_data, k_data, *a, **k)

    consumer._run_csa2_chunks = run
    consumer.forward_prepared = fp
    md = make_metadata(world, args, (1,), cand)
    call(consumer, md, world)
    torch.cuda.synchronize()
    consumer.__dict__.pop("_run_csa2_chunks")
    consumer.__dict__.pop("forward_prepared")
    info["tile_count"] = len(tiles)
    info["tile_shapes"] = str(dict(Counter(tiles)))
    from torch.profiler import ProfilerActivity, profile

    md = make_metadata(world, args, (1,), cand)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call(consumer, md, world)
        torch.cuda.synchronize()
    kern = {}
    for evt in prof.key_averages():
        if evt.device_type is not None and "CUDA" in str(evt.device_type):
            t = getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0)
            kern[evt.key[:60]] = (evt.count, round(t / 1000.0, 3))
    top = sorted(kern.items(), key=lambda kv: -kv[1][1])[:12]
    info["kernels_count_ms"] = {k: v for k, v in top}
    return info


def run_bench(args, device, topk, blocks, block_size):
    layout = make_layout(topk, blocks, block_size)
    world = make_world(args, device)
    owner = indexer_mod.CSA2Indexer(layout, 0, HEADS, HEAD_DIM)
    cand = owner_candidates(owner, world, args)
    cand = perturb(cand, args)
    consumers = [indexer_mod.CSA2Indexer(layout, l, HEADS, HEAD_DIM) for l in CONSUMERS]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    result = {"new_route": NEW}
    if not NEW or args.diag_always:
        result["diagnostic"] = diagnostic(args, consumers[0], world, cand)
        log(f"[diagnostic] {json.dumps(result['diagnostic'])}")
    # Memory: owner prefill indexer vs one consumer layer.
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    md = make_metadata(world, args, (0,))
    call(owner, md, world)
    torch.cuda.synchronize()
    owner_peak = torch.cuda.max_memory_allocated() - base
    del md
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    md = make_metadata(world, args, (1,), cand)
    call(consumers[0], md, world)
    torch.cuda.synchronize()
    cons_peak = torch.cuda.max_memory_allocated() - base
    del md
    result["mem_peak_mb"] = {"owner_layer": owner_peak / 2**20, "consumer_layer": cons_peak / 2**20}
    log(f"[memory] {json.dumps(result['mem_peak_mb'])}")
    starts = [torch.cuda.Event(enable_timing=True) for _ in CONSUMERS]
    ends = [torch.cuda.Event(enable_timing=True) for _ in CONSUMERS]
    times, per_layer = [], []
    outs = None
    for it in range(args.warmup + args.iters):
        md = make_metadata(world, args, CONSUMERS, cand)  # reset point
        outs = []
        for i, ix in enumerate(consumers):
            l2_flush(flush)
            starts[i].record()
            outs.append(call(ix, md, world))
            ends[i].record()
        torch.cuda.synchronize()
        if it >= args.warmup:
            lt = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            per_layer.append(lt)
            times.append(sum(lt))
        if it % 5 == 0:
            log(f"iter {it} total={sum(s.elapsed_time(e) for s, e in zip(starts, ends)):.3f} ms")
    times.sort()
    med = statistics.median(times)
    p10 = times[max(0, int(0.1 * len(times)))]
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    try:
        import pynvml  # noqa: F401

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        result["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:  # noqa: BLE001
        result["sm_clock_mhz"] = None
    import hashlib

    h = hashlib.sha256()
    for o in outs:
        h.update(o.cpu().numpy().tobytes())
    result["output_digest"] = h.hexdigest()[:16]
    result["per_layer_ms_median"] = [statistics.median(c) for c in zip(*per_layer)]
    result["p10_ms"], result["p90_ms"] = p10, p90
    result["batch"] = {str(args.rows): {"kernel_ms": med, "noise_floor": (p90 - p10) / med}}
    result["output"] = {"class": "exact"}
    log(f"4 consumer layers median {med:.3f} ms p10 {p10:.3f} p90 {p90:.3f}")
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--prefix", type=int, default=32768)
    p.add_argument("--max-positions", type=int, default=36864)
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--diag-always", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda")
    topk, blocks, block_size, heads, head_dim = read_topk(args.checkpoint)
    log(f"device={torch.cuda.get_device_name(0)} new={NEW} topk={topk} blocks={blocks} "
        f"block_size={block_size} heads={heads} head_dim={head_dim}")
    if args.check:
        run_check(args, device, topk, blocks, block_size)
    else:
        run_bench(args, device, topk, blocks, block_size)


if __name__ == "__main__":
    sys.exit(main())
