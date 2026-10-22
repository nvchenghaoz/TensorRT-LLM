"""Correctness gates for idea 30 (KVCacheManagerV2 block-offset copy via a host snapshot).

Drives the real KVCacheManagerV2.copy_batch_block_offsets (snapshot path) on a bare manager
object with the real C++ IndexMapper, real pinned page table of the served shape
(2 pools x 65 rows x 2 x 1156 blocks, index scales [48, 32], kv offsets [0, 1]) and compares
with main's path (copy_batch_block_offsets_to_device, the zero-copy kernel) bitwise.

G2a  10 warmup + 50 sentinel-restored calls at 8/16/32 seqs: dst bitwise equal to main's path
     and to iteration 1, untouched rows keep the sentinel.
G2b  replayed c32 decode sequence (400 steps): admissions with slot reuse (partially rewritten
     rows keep stale tails), finishes, block appends every 32 tokens, BAD_PAGE_INDEX entries,
     batch reordering, contexts first. The candidate runs on a stream stalled by random
     torch.cuda._sleep so it lags the host by several steps (snapshot ring wraps); main's path
     runs synchronously on its own stream. Every step's full dst is compared bitwise, and the
     scheduled rows against a CPU reference.
G1   overlap race: stall the stream, enqueue batch A's copy, then rewrite the whole host page
     table and the IndexMapper copy index for batch B (and enqueue B). A's dst must hold A's
     offsets. Also a 6-deep chain through the 3-slot ring. Run with the snapshot path (must
     pass) and with the stock path (TLLM_KV_V2_BLOCK_OFFSET_SNAPSHOT off; expected to show the
     hazard).
G3   inside torch.cuda.graph capture the snapshot path is not taken (no event op captured).
"""
import json
import random
import sys

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as m
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    IndexMapper, copy_batch_block_offsets_to_device)

P, SRC_MAX, B, DST_MAX = 2, 65, 1156, 32
SCALES, OFFS = [48, 32], [0, 1]
SENT = 0x5A5A5A5A
BAD = -1


def log(*a):
    print(*a, flush=True)


def make_mgr(stream):
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
    return mgr


def expected(host, cidx_list, n):
    k = host[:, cidx_list, 0, :].long()
    sc = torch.tensor(SCALES).view(P, 1, 1).long()
    of = torch.tensor(OFFS).view(P, 1, 1).long()
    ek = torch.where(k == BAD, 0, k * sc).int()
    ev = torch.where(k == BAD, 0, k * sc + of).int()
    return ek, ev


def ref_copy(mgr, dst, rids, nctx, stream):
    cidx = mgr.index_mapper.get_copy_index(rids, nctx, 1)
    copy_batch_block_offsets_to_device(mgr.host_kv_cache_block_offsets, dst, cidx, mgr.index_scales, mgr.kv_offset,
                                       stream.cuda_stream)
    stream.synchronize()
    return cidx.tolist()


def fill_row(host, slot, nblocks, rng, keep_tail=False):
    for p in range(P):
        vals = torch.tensor([rng.randrange(0, 60000) for _ in range(nblocks)], dtype=torch.int32)
        if nblocks:
            bad = torch.rand(nblocks) < 0.03
            vals[bad] = BAD
        host[p, slot, 0, :nblocks] = vals
        host[p, slot, 1, :nblocks] = vals + 7  # V plane: never read by either path
        if not keep_tail:
            host[p, slot, 0, nblocks:] = 0


def g2a(stream):
    rng = random.Random(7)
    mgr = make_mgr(stream)
    rids = list(range(100, 100 + SRC_MAX))
    for r in rids:
        mgr.index_mapper.add_new_sequence(r)
    for s in range(SRC_MAX):
        fill_row(mgr.host_kv_cache_block_offsets, s, rng.randrange(1000, B + 1), rng)
    out = dict()
    ok = True
    for n in (8, 16, 32):
        batch = rng.sample(rids, n)
        dc = torch.empty(P, DST_MAX, 2, B, dtype=torch.int32, device="cuda")
        dr = torch.empty_like(dc)
        first = None
        max_diff_ref = 0
        drift = 0
        torch.cuda.synchronize()
        for it in range(60):
            with torch.cuda.stream(stream):
                dc.fill_(SENT)
                dr.fill_(SENT)
                mgr.copy_batch_block_offsets(dc, batch, 1, 0, n)
            stream.synchronize()
            cl = ref_copy(mgr, dr, batch, 0, stream)
            eq = torch.equal(dc, dr)
            ek, ev = expected(mgr.host_kv_cache_block_offsets, cl, n)
            dcc = dc.cpu()
            eq_cpu = torch.equal(dcc[:, :n, 0], ek) and torch.equal(dcc[:, :n, 1], ev)
            sent_ok = bool((dcc[:, n:] == SENT).all())
            if first is None:
                first = dcc.clone()
            elif not torch.equal(first, dcc):
                drift += 1
            if not eq:
                max_diff_ref = max(max_diff_ref, int((dc != dr).sum()))
            ok &= eq and eq_cpu and sent_ok
        out[str(n)] = dict(iters=60, bitwise_vs_main=max_diff_ref == 0, mismatched_elems_max=max_diff_ref,
                           drift_vs_iter1=drift, cpu_ref_ok=bool(eq_cpu), untouched_rows_keep_sentinel=sent_ok)
        log("G2a n=%d" % n, json.dumps(out[str(n)]))
    return ok, out


def g2b(stream, ref_stream, steps=400):
    rng = random.Random(11)
    mgr = make_mgr(stream)
    host = mgr.host_kv_cache_block_offsets
    active = dict()  # rid -> dict(slot, tokens)
    next_rid = [1000]
    dc = torch.full((P, DST_MAX, 2, B), SENT, dtype=torch.int32, device="cuda")
    dr = torch.full_like(dc, SENT)
    torch.cuda.synchronize()
    cand_clones, ref_clones = [], []
    stats = dict(admitted=0, finished=0, slot_reuse=0, appends=0, max_batch=0, bad_entries_seen=0, batch_changes=0)
    used_slots = set()
    prev_batch = None
    for step in range(steps):
        # finishes
        for rid in list(active):
            if rng.random() < 0.02:
                mgr.index_mapper.remove_sequence(rid)
                del active[rid]
                stats["finished"] += 1
        # admissions (contexts first in the batch)
        ctx = []
        while len(active) + len(ctx) < DST_MAX and rng.random() < (0.9 if len(active) < 24 else 0.3):
            rid = next_rid[0]
            next_rid[0] += 1
            slot = mgr.index_mapper.add_new_sequence(rid)
            if slot in used_slots:
                stats["slot_reuse"] += 1
            used_slots.add(slot)
            nb = rng.randrange(1, 1097)
            # slot reuse: rewrite only the new prefix, keep the previous request's stale tail half the time
            fill_row(host, slot, nb, rng, keep_tail=rng.random() < 0.5)
            ctx.append(rid)
            active[rid] = dict(slot=slot, tokens=nb * 32 - rng.randrange(0, 32))
            stats["admitted"] += 1
        # generation rows advance; append a block every 32 tokens
        gens = [r for r in active if r not in ctx]
        for rid in gens:
            a = active[rid]
            a["tokens"] += 1
            if a["tokens"] % 32 == 1:
                nb = (a["tokens"] + 31) // 32
                if nb <= B:
                    for p in range(P):
                        host[p, a["slot"], 0, nb - 1] = BAD if rng.random() < 0.03 else rng.randrange(0, 60000)
                    stats["appends"] += 1
        if rng.random() < 0.2:
            rng.shuffle(gens)
        batch = ctx + gens
        if not batch:
            continue
        n = len(batch)
        stats["max_batch"] = max(stats["max_batch"], n)
        if batch != prev_batch:
            stats["batch_changes"] += 1
        prev_batch = list(batch)
        # main's path, synchronous on its own stream (reads the host state of this step)
        cl = ref_copy(mgr, dr, batch, len(ctx), ref_stream)
        ek, ev = expected(host, cl, n)
        stats["bad_entries_seen"] += int((host[:, cl, 0, :] == BAD).sum())
        ref_clones.append(dr.clone())
        torch.cuda.synchronize()
        # candidate, lagging behind the host: stall the stream first
        if rng.random() < 0.5:
            torch.cuda._sleep(rng.randrange(1000, 400000))
        with torch.cuda.stream(stream):
            mgr.copy_batch_block_offsets(dc, batch, 1, len(ctx), n)
            cand_clones.append((dc.clone(), ek, ev, n))
    torch.cuda.synchronize()
    nbad = 0
    first_bad = None
    for i, ((c, ek, ev, n), r) in enumerate(zip(cand_clones, ref_clones)):
        good = torch.equal(c, r)
        cc = c.cpu()
        good_cpu = torch.equal(cc[:, :n, 0], ek) and torch.equal(cc[:, :n, 1], ev)
        if not (good and good_cpu):
            nbad += 1
            if first_bad is None:
                first_bad = i
    res = dict(steps=len(cand_clones), steps_mismatched=nbad, first_mismatch=first_bad, **stats)
    log("G2b", json.dumps(res))
    return nbad == 0, res


def g1(stream, use_snapshot):
    m._BLOCK_OFFSET_SNAPSHOT_COPY = use_snapshot
    rng = random.Random(3)
    mgr = make_mgr(stream)
    host = mgr.host_kv_cache_block_offsets
    rids = list(range(500, 532))
    for r in rids:
        mgr.index_mapper.add_new_sequence(r)
    for s in range(32):
        fill_row(host, s, 1090, rng)
    torch.cuda.synchronize()
    # single A/B race
    batch_a = rids[:24]
    batch_b = list(reversed(rids[8:32]))
    ca = mgr.index_mapper.get_copy_index(batch_a, 0, 1).tolist()
    ea = expected(host, ca, 24)
    da = torch.zeros(P, DST_MAX, 2, B, dtype=torch.int32, device="cuda")
    db = torch.zeros_like(da)
    dsts = [torch.zeros_like(da) for _ in range(6)]
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        mgr.copy_batch_block_offsets(da, batch_a, 1, 0, 24)
    # host prepares batch B while A is still queued: whole page table rewritten, copy index rewritten
    for s in range(32):
        fill_row(host, s, 1090, rng)
    cb = mgr.index_mapper.get_copy_index(batch_b, 0, 1).tolist()
    eb = expected(host, cb, 24)
    with torch.cuda.stream(stream):
        mgr.copy_batch_block_offsets(db, batch_b, 1, 0, 24)
    torch.cuda.synchronize()
    dac, dbc = da.cpu(), db.cpu()
    a_ok = torch.equal(dac[:, :24, 0], ea[0]) and torch.equal(dac[:, :24, 1], ea[1])
    b_ok = torch.equal(dbc[:, :24, 0], eb[0]) and torch.equal(dbc[:, :24, 1], eb[1])
    # chain of 6 through the 3-slot ring while the stream is stalled
    chain_ok = True
    exps = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
    for j in range(6):
        bj = rng.sample(rids, 16 + j)
        cj = mgr.index_mapper.get_copy_index(bj, 0, 1).tolist()
        exps.append((expected(host, cj, len(bj)), len(bj)))
        with torch.cuda.stream(stream):
            mgr.copy_batch_block_offsets(dsts[j], bj, 1, 0, len(bj))
        for s in range(32):
            fill_row(host, s, 1090, rng)
    torch.cuda.synchronize()
    for d, ((ek, ev), n) in zip(dsts, exps):
        dcpu = d.cpu()
        chain_ok &= torch.equal(dcpu[:, :n, 0], ek) and torch.equal(dcpu[:, :n, 1], ev)
    m._BLOCK_OFFSET_SNAPSHOT_COPY = True
    res = dict(path="snapshot" if use_snapshot else "stock(main)", a_holds_a=bool(a_ok), b_holds_b=bool(b_ok),
               ring_chain6_ok=bool(chain_ok))
    log("G1", json.dumps(res))
    return a_ok and b_ok and chain_ok, res


def g3(stream):
    mgr = make_mgr(stream)
    d = torch.zeros(P, DST_MAX, 2, B, dtype=torch.int32, device="cuda")
    outside = mgr._can_copy_block_offsets_via_snapshot(d)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    inside = None
    x = torch.zeros(4, device="cuda")
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            inside = mgr._can_copy_block_offsets_via_snapshot(d)
            x.add_(1)
    res = dict(snapshot_path_outside_capture=bool(outside), snapshot_path_inside_capture=bool(inside))
    log("G3", json.dumps(res))
    return outside and not inside, res


def main():
    torch.cuda.init()
    stream = torch.cuda.Stream()
    ref_stream = torch.cuda.Stream()
    assert m._BLOCK_OFFSET_SNAPSHOT_COPY, "snapshot path disabled by env"
    ok2a, r2a = g2a(stream)
    ok2b, r2b = g2b(stream, ref_stream)
    ok1, r1 = g1(stream, True)
    ok1s, r1s = g1(stream, False)
    ok3, r3 = g3(stream)
    passed = bool(ok2a and ok2b and ok1 and ok3)
    print(json.dumps(dict(passed=passed, compared="device kv_cache_block_offsets vs main's "
                          "copy_batch_block_offsets_to_device (bitwise) and a CPU reference",
                          G2a_sentinel_10_plus_50=r2a, G2b_replayed_c32_sequence=r2b,
                          G1_overlap_race_snapshot=r1, G1_overlap_race_stock_main=r1s,
                          G1_stock_shows_hazard=not ok1s, G3_capture_guard=r3)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
