"""Branch 57 check: SM120 CSA2 split-key decode indexer launch cuts + _constant_rows fix.

Runs in either tree (baseline = main 1d696e72, candidate = branch 57). The
reference is main's select_split_decode path written out verbatim here
(``ref_select``: expand-copy replicas, DeepGEMM split-key logits, as_strided
``.contiguous()`` assembly, ``mask_positions_``, ``select_prepared_scores``,
``finalize_selection_``) on its OWN indexer, run eagerly only, so its constant
rows are always filled.

Per layer kind (r2 = ratio 2, r1 = ratio 1, cand = candidate tile) and decode
rows m (last row is a padding row, token_requests = -1, when m > 1):

A. production order of job 374: 3 eager warmups, CUDA-graph capture, then an
   eager call BEFORE any replay, compared with the reference (main reads an
   unwritten capture-time row here), then the first replay;
B. eager x2 vs reference: hook (masked) scores bitwise, valid bitwise,
   int32 selections torch.equal (tie-aware only on exact K-boundary ties);
C. 10-step decode window (visible lengths advance, q / scales / weights /
   key pages change in place): graph replay vs eager reference every step;
D. warmup + iters replays at the final state with the timing loop's reset
   point, every replay compared; caching-allocator allocations during bare
   replays must be 0;
E. launches of one layer (warm eager call and one replay of a plain graph),
   with kernel names.

Constant-row unit checks (fresh indexers): (a) eager right after capture,
before any replay, is correct; (b) after an eager call a capture reuses the
row and records no torch.full; (c) a capture-time row with no eager entry is
correct after replay, even after 200 eager geometries; (d) two graphs capture
one key with no eager entry, and replaying only the second is correct.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csa2_split_decode_bench as sdb  # noqa: E402
from csa2_prefill_consumer_bench import tie_aware  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel as kernel_mod  # noqa: E402
import csa2_index_gather_bench as gb  # noqa: E402

NEW = hasattr(kernel_mod, "mask_split_positions")
log = sdb.log


def bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def ref_replicas(q_data, q_scale, weights, reps):
    rows, heads = q_data.shape[0], q_data.shape[1]
    q = q_data.view(torch.int8)
    q = q[:, None].expand(rows, reps, heads, q.shape[2]).contiguous()
    q = q.view(rows * reps, heads, q.shape[3])
    scale = q_scale.contiguous().view(torch.int32).reshape(rows, heads)
    scale = scale[:, None].expand(rows, reps, heads).contiguous().view(rows * reps, heads)
    w = weights[:, None].expand(rows, reps, heads).contiguous().view(rows * reps, heads)
    return q, scale, w


def ref_logits(ind, q_data, q_scale, weights, k_data, k_scale, width):
    rows = q_data.shape[0]
    splits = indexer_mod._split_decode_count(rows, width)
    block = indexer_mod._SPLIT_DECODE_BLOCK_Q
    chunk = width // splits
    reps = splits * block
    bounds = ind._split_decode_bounds(rows, width, splits, q_data.device)
    q, scale, w = ref_replicas(q_data, q_scale, weights, reps)
    out = indexer_mod._fp8_fp4_mqa_logits(
        (q, scale),
        (k_data.view(torch.int8), k_scale.contiguous().view(torch.int32).reshape(-1)),
        w, bounds[0], bounds[1], clean_logits=False, max_seqlen_k=chunk,
    )
    stride = out.stride(0)
    return (out.as_strided((rows, splits, chunk), (reps * stride, block * stride, 1))
            .contiguous().view(rows, width))


def ref_select(ref_ind, c):
    """main's select_split_decode -> _select_mapped_logits fused path, eager."""
    inp = c.inp
    keys, scales, mapped = c.new_keys()
    width = mapped.shape[1]
    logits = ref_logits(ref_ind, inp["q"], inp["q_scale"], inp["weights"], keys, scales, width)
    valid = kernel_mod.mask_positions_(logits, mapped, inp["visible"])
    out = torch.full((c.m, sdb.TOPK), -7, dtype=torch.int32, device=keys.device)
    ri = torch.full((c.m, 10, sdb.TOPK), -3, dtype=torch.int32, device=keys.device)
    rl = torch.full((c.m, 10, sdb.TOPK), 1.5, dtype=torch.float32, device=keys.device)
    ref_ind.select_prepared_scores(logits, out, None, None, is_prefill=False,
                                   radix_aux_indices=ri, radix_aux_logits=rl)
    kernel_mod.finalize_selection_(out, width, scores=logits, positions=mapped)
    torch.cuda.synchronize()
    return out.clone(), logits.clone(), valid.clone(), mapped.clone()


VALIDS = []
_ORIG = {}


def _install_valid_recorder():
    """Record the boolean validity the path computes (clone, also inside captures).

    Wraps mask_positions_ (main, and the branch's eager path) and, on the branch,
    mask_split_positions (its graph-only fused path; compile_only calls return None)."""
    orig_mp = indexer_mod.mask_positions_

    def rec_mp(*a, **k):
        valid = orig_mp(*a, **k)
        if RECORD[0]:
            VALIDS.append(valid.clone())
            PATHS.append("mask_positions_")
        return valid

    indexer_mod.mask_positions_ = rec_mp
    if NEW:
        orig_ms = indexer_mod.mask_split_positions

        def rec_ms(*a, **k):
            r = orig_ms(*a, **k)
            if r is not None and RECORD[0]:
                VALIDS.append(r[1].clone())
                PATHS.append("mask_split_positions")
            return r

        indexer_mod.mask_split_positions = rec_ms


PATHS = []


RECORD = [False]


def run_new(c, record):
    RECORD[0] = record
    VALIDS.clear()
    PATHS.clear()
    c.hooked = []
    try:
        c.new(record=record)
    finally:
        RECORD[0] = False
    if record:
        return c.hooked[0], VALIDS[0]
    return None, None


def sel_ok(scores, mapped, a, b):
    if torch.equal(a, b):
        return True, None
    t = tie_aware(scores, mapped, a, b, sdb.TOPK)
    return bool(t["ok"]), t


def alloc_count():
    return torch.cuda.memory_stats().get("allocation.all.allocated", 0)


def compare(ref, out, scores, valid):
    r_out, r_sc, r_valid, r_map = ref
    ok_sel, tie = sel_ok(r_sc, r_map, r_out, out)
    sc = same(scores, r_sc)
    va = same(valid, r_valid)
    return dict(sel=ok_sel, sel_equal=bool(torch.equal(r_out, out)), scores=sc, valid=va,
                ok=bool(ok_sel and sc and va), tie=tie)


def mutate(c, g):
    inp = c.inp
    m = c.m
    inp["visible"].add_(1)
    inp["q"].copy_(torch.randint(-128, 128, inp["q"].shape, generator=g, dtype=torch.int32)
                   .to(torch.int8))
    qs = torch.randint(122, 129, (m, inp["q_scale"].shape[1], 4), generator=g,
                       dtype=torch.int32).to(torch.uint8).view(torch.int32).reshape(m, -1)
    inp["q_scale"].copy_(qs)
    inp["weights"].copy_(torch.randn(inp["weights"].shape, generator=g))
    pages = inp["pages"].reshape(inp["pages"].shape[0], -1)
    npages, per = pages.shape
    pages = pages.view(-1)
    idx = torch.randint(0, npages, (4096,), generator=g) * per + torch.randint(
        0, 4096, (4096,), generator=g)
    pages.index_copy_(0, idx.to(pages.device),
                      torch.randint(0, 256, (4096,), generator=g, dtype=torch.int32)
                      .to(torch.uint8).to(pages.device))


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and "memset" not in ev.name.lower() \
                and "memcpy" not in ev.name.lower():
            k = ev.name[:80]
            names[k] = names.get(k, 0) + 1
    return sum(names.values()), names


def run_case(kind, m, args, device, failures):
    c = sdb.Case(kind, m, args, device)
    ref_ind = gb.make_indexer(2 if kind == "r2" else 1)
    info = dict(kind=kind, m=m, width=c.width,
                splits=indexer_mod._split_decode_count(m, c.width))
    # A. job 374's order: eager warmup, capture, eager BEFORE any replay.
    ref0 = ref_select(ref_ind, c)
    for _ in range(3):
        c.reset()
        run_new(c, False)
    torch.cuda.synchronize()
    c.reset()
    p_sc, p_valid = run_new(c, True)
    info["eager_mask_path"] = list(PATHS)
    torch.cuda.synchronize()
    pre = compare(ref0, c.out.clone(), p_sc, p_valid)
    info["A_eager_before_capture"] = pre["ok"]
    if not pre["ok"]:
        failures.append(f"A eager-before-capture {kind} m{m}")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        c.reset()
        run_new(c, False)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            RECORD[0] = True
            VALIDS.clear()
            PATHS.clear()
            c.hooked = []
            try:
                c.new(record=True)
            finally:
                RECORD[0] = False
            g_sc, g_valid = c.hooked[0], VALIDS[0]
            info["graph_mask_path"] = list(PATHS)
    torch.cuda.synchronize()
    c.reset()
    e_sc, e_valid = run_new(c, True)
    torch.cuda.synchronize()
    a = compare(ref0, c.out.clone(), e_sc, e_valid)
    info["A_eager_after_capture_before_replay"] = {k: v for k, v in a.items() if k != "tie"}
    if not a["ok"]:
        info["A_selected_counts"] = [int(x) for x in (c.out >= 0).sum(1).tolist()]
        failures.append(f"A eager-after-capture {kind} m{m}")
    c.reset()
    graph.replay()
    torch.cuda.synchronize()
    a1 = compare(ref0, c.out.clone(), g_sc, g_valid)
    info["A_first_replay"] = a1["ok"]
    if not a1["ok"]:
        failures.append(f"A first replay {kind} m{m}")
    # B. eager x2 vs reference
    ok_b = True
    for _ in range(2):
        c.reset()
        sc, va = run_new(c, True)
        torch.cuda.synchronize()
        b = compare(ref0, c.out.clone(), sc.clone(), va.clone())
        ok_b &= b["ok"]
        if b["tie"]:
            info["B_tie"] = b["tie"]
    info["B_eager_x2"] = ok_b
    if not ok_b:
        failures.append(f"B eager {kind} m{m}")
    # C. 10-step decode window, graph captured at warmup state
    gen = torch.Generator(device="cpu").manual_seed(9000 + 10 * m + len(kind))
    bad_steps, ties = [], 0
    for step in range(10):
        mutate(c, gen)
        r = ref_select(ref_ind, c)
        c.reset()
        graph.replay()
        torch.cuda.synchronize()
        cc = compare(r, c.out.clone(), g_sc, g_valid)
        ties += int(not cc["sel_equal"])
        if not cc["ok"]:
            bad_steps.append(dict(step=step, **{k: v for k, v in cc.items() if k != "tie"}))
    info["C_window_bad_steps"] = bad_steps
    info["C_window_tie_steps"] = ties
    if bad_steps:
        failures.append(f"C window {kind} m{m}: {bad_steps[:2]}")
    # D. warmup + iters replays at the final state
    r = ref_select(ref_ind, c)
    total = args.warmup + args.iters
    bad = 0
    for _ in range(total):
        c.reset()
        graph.replay()
        torch.cuda.synchronize()
        if not compare(r, c.out, g_sc, g_valid)["ok"]:
            bad += 1
    torch.cuda.synchronize()
    a0 = alloc_count()
    for _ in range(total):
        c.reset()
        graph.replay()
    torch.cuda.synchronize()
    info["D_replays"] = total
    info["D_bad_replays"] = bad
    info["D_allocs_during_bare_replays"] = alloc_count() - a0
    if bad or info["D_allocs_during_bare_replays"]:
        failures.append(f"D replays {kind} m{m}: bad={bad}")
    del graph
    # E. launches: warm eager call and one replay of a plain graph
    n_eager, k_eager = kernels_of(lambda: run_new(c, False))
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run_new(c, False)
        stream.synchronize()
        with torch.cuda.graph(g2, stream=stream):
            run_new(c, False)
    torch.cuda.synchronize()
    g2.replay()
    n_graph, k_graph = kernels_of(g2.replay)
    del g2
    info["E_launches_eager"] = n_eager
    info["E_launches_graph"] = n_graph
    info["E_kernels_eager"] = k_eager
    info["E_kernels_graph"] = k_graph
    log(json.dumps(info))
    del c, ref_ind
    torch.cuda.empty_cache()
    return info


def constant_rows_checks(device, failures):
    res = {}
    count, value = 4, 16384

    def cap(ind, key_count, key_value, out):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            s.synchronize()
            with torch.cuda.graph(g, stream=s):
                out.copy_(ind._constant_rows(key_count, key_value, device))
        torch.cuda.synchronize()
        return g

    expect = torch.full((count,), value, dtype=torch.int32, device=device)
    # warm the copy kernel outside captures
    tmp = torch.zeros(count, dtype=torch.int32, device=device)
    tmp.copy_(expect)
    # (a) eager right after capture, before any replay
    ind = gb.make_indexer(1)
    out = torch.zeros(count, dtype=torch.int32, device=device)
    g = cap(ind, count, value, out)
    eager = ind._constant_rows(count, value, device).clone()
    torch.cuda.synchronize()
    res["a_eager_after_capture"] = bool(torch.equal(eager, expect))
    g.replay()
    torch.cuda.synchronize()
    res["a_replay"] = bool(torch.equal(out, expect))
    del g
    # (b) after eager warmup, a capture reuses the row and records no fill
    ind = gb.make_indexer(1)
    first = ind._constant_rows(count, value, device)
    calls = [0]
    orig_full = torch.full

    def counting_full(*a, **k):
        calls[0] += 1
        return orig_full(*a, **k)

    out = torch.zeros(count, dtype=torch.int32, device=device)
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        s.synchronize()
        torch.full = counting_full
        try:
            with torch.cuda.graph(g, stream=s):
                got = ind._constant_rows(count, value, device)
                out.copy_(got)
        finally:
            torch.full = orig_full
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    res["b_reused"] = got is first
    res["b_fills_recorded"] = calls[0]
    res["b_eager_first_correct"] = bool(torch.equal(first, expect))
    res["b_replay"] = bool(torch.equal(out, expect))
    del g
    # (c) capture-time row without eager entry, correct after 200 eager geometries
    ind = gb.make_indexer(1)
    out = torch.zeros(count, dtype=torch.int32, device=device)
    g = cap(ind, count, value, out)
    eager_bad = 0
    for i in range(200):
        cnt, val = 1 + i % 9, 1000 + i
        if i % 50 == 0:
            cnt, val = count, value  # the captured key itself, eagerly
        r = ind._constant_rows(cnt, val, device)
        if not torch.equal(r, torch.full((cnt,), val, dtype=torch.int32, device=device)):
            eager_bad += 1
    junk = [torch.full((1 << 16,), -5, dtype=torch.int32, device=device) for _ in range(8)]
    del junk
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    res["c_eager_bad"] = eager_bad
    res["c_replay_after_200"] = bool(torch.equal(out, expect))
    del g
    # (d) two graphs, one key, no eager entry; replay only the second
    ind = gb.make_indexer(1)
    out1 = torch.zeros(count, dtype=torch.int32, device=device)
    out2 = torch.zeros(count, dtype=torch.int32, device=device)
    g1 = cap(ind, count, value, out1)
    g2 = cap(ind, count, value, out2)
    g2.replay()
    torch.cuda.synchronize()
    res["d_second_only"] = bool(torch.equal(out2, expect))
    del g1, g2
    ok = (res["a_eager_after_capture"] and res["a_replay"] and res["b_reused"]
          and res["b_fills_recorded"] == 0 and res["b_eager_first_correct"] and res["b_replay"]
          and res["c_eager_bad"] == 0 and res["c_replay_after_200"] and res["d_second_only"])
    res["ok"] = bool(ok)
    if not ok:
        failures.append(f"constant_rows {res}")
    log("constant_rows " + json.dumps(res))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=36864)
    p.add_argument("--context", type=int, default=34816)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=60)
    args = p.parse_args()
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)} new={NEW}")
    _install_valid_recorder()
    failures = []
    cr = constant_rows_checks(device, failures)
    cases = []
    for kind in ("r2", "r1", "cand"):
        for m in [int(x) for x in args.batches.split(",")]:
            cases.append(run_case(kind, m, args, device, failures))
    steps = {}
    for m in [int(x) for x in args.batches.split(",")]:
        by = {c["kind"]: c for c in cases if c["m"] == m}
        steps[str(m)] = {k: 3 * by["r2"][k] + by["r1"][k] + 4 * by["cand"][k]
                         for k in ("E_launches_eager", "E_launches_graph")}
    print(json.dumps({
        "passed": not failures, "tree": "branch57" if NEW else "main",
        "failures": failures[:20],
        "compared": "select_split_decode vs main's path written out verbatim on its own eager "
                    "indexer: hook-masked scores and valid bitwise, int32 selections torch.equal "
                    "(tie-aware on K-boundary ties); eager-after-capture before replay, eager x2, "
                    "10-step advancing window, warmup+iters replays with reset points, no alloc "
                    "after capture; _constant_rows (a)-(d)",
        "launches_per_step_8_layers": steps,
        "constant_rows": cr,
        "cases": cases}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
