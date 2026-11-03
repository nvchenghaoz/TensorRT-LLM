"""CSA2 decode top-512: CUDA radix (topKPerRowDecode) vs CuTe-DSL radix filter.

Modes:
  engines  - call both engines directly (TopK module) at rows x widths; time and compare.
  indexer  - build a CSA2Indexer from the checkpoint config and time its decode
             selection path (select_prepared_scores) as the tree's gate picks it.
  check    - untimed repeated-execution correctness (indexer path) vs the CUDA radix
             reference, graph capture/replay included.
Synthetic fp32 scores (fixed seed), a -inf tail of invisible columns like the decode mask.
"""

import argparse
import json
import os
import time

import torch

ROWS = (1, 2, 4, 8)
WIDTHS = (36864, 18432, 16384)
TOPK = 512


def log(*a):
    print(*a, flush=True)


def make_scores(rows, width, seed, ties=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(rows, width, generator=g, device="cuda", dtype=torch.float32)
    if ties:
        s = torch.round(s * 8) / 8  # heavy ties at the K boundary
    # invisible tail like mask_positions_ (decode context < width)
    for r in range(rows):
        t = (r * 997 + seed * 131) % 2048
        if t:
            s[r, width - t :] = -torch.inf
    return s.contiguous()


def radix_aux(rows):
    return (
        torch.empty(rows, 10, TOPK, dtype=torch.int32, device="cuda"),
        torch.empty(rows, 10, TOPK, dtype=torch.float32, device="cuda"),
    )


def graph_time(fn, inner=20, reps=25):
    """Per-call us: capture `inner` calls in one graph, replay `reps` times."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / inner)
    times.sort()
    med = times[len(times) // 2]
    p10 = times[int(0.1 * (len(times) - 1))]
    p90 = times[int(0.9 * (len(times) - 1))]
    return med, (p90 - p10) / med


def sorted_sel(idx):
    return torch.sort(idx.long(), dim=1).values


def compare(ref_idx, idx, scores):
    """(exact_rows, tie_rows_multiset_equal, bad_rows)."""
    bad = []
    tie_ok = 0
    exact = 0
    a, b = sorted_sel(ref_idx), sorted_sel(idx)
    for r in range(scores.shape[0]):
        if torch.equal(a[r], b[r]):
            exact += 1
            continue
        if (a[r] < 0).any() or (b[r] < 0).any() or (b[r] >= scores.shape[1]).any():
            bad.append(r)
            continue
        if len(set(b[r].tolist())) != b.shape[1]:
            bad.append(r)
            continue
        sa = torch.sort(scores[r][a[r]]).values
        sb = torch.sort(scores[r][b[r]]).values
        if torch.equal(sa, sb):
            tie_ok += 1
        else:
            bad.append(r)
    return exact, tie_ok, bad


def mode_engines(args):
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
    import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers ops)

    log("sm", torch.cuda.get_device_capability())
    cuda = TopK(TOPK, decode_implementation=TopKImplementation.CUDA_RADIX)
    dsl = TopK(TOPK, decode_implementation=TopKImplementation.CUTE_DSL_RADIX)
    out = {}
    ok = True
    for width in WIDTHS:
        for rows in ROWS:
            scores = make_scores(rows, width, seed=rows * 7 + width)
            lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
            ai, al = radix_aux(rows)
            o_c = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            o_d = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")

            def call(eng, sc, o):
                eng(sc, o, is_prefill=False, sequence_lengths=lens, scan_lengths=lens,
                    next_n=1, max_seq_len=width, radix_aux_indices=ai, radix_aux_logits=al)

            t0 = time.time()
            try:
                call(dsl, scores, o_d)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                log(f"DSL FAILED rows={rows} width={width}: {type(e).__name__}: {e}")
                out[f"{rows}x{width}"] = {"dsl_error": f"{type(e).__name__}: {e}"[:600]}
                ok = False
                continue
            log(f"compile+first dsl {rows}x{width}: {time.time() - t0:.1f}s")
            call(cuda, scores, o_c)
            torch.cuda.synchronize()
            ex, tie, bad = compare(o_c, o_d, scores)
            tc, nc = graph_time(lambda: call(cuda, scores, o_c))
            td, nd = graph_time(lambda: call(dsl, scores, o_d))
            ts = make_scores(rows, width, seed=rows + 3, ties=True)
            o_c2 = torch.empty_like(o_c)
            o_d2 = torch.empty_like(o_d)
            call(cuda, ts, o_c2)
            call(dsl, ts, o_d2)
            torch.cuda.synchronize()
            ex2, tie2, bad2 = compare(o_c2, o_d2, ts)
            rec = {"cuda_us": round(tc, 2), "cuda_nf": round(nc, 4), "dsl_us": round(td, 2),
                   "dsl_nf": round(nd, 4), "exact_rows": ex, "tie_rows": tie, "bad_rows": len(bad),
                   "tie_case": {"exact": ex2, "tie_equal": tie2, "bad": len(bad2)}}
            ok = ok and not bad and not bad2
            out[f"{rows}x{width}"] = rec
            log(f"{rows}x{width}: {rec}")
    print(json.dumps({"passed": ok, "results": out}))


def build_indexer(checkpoint, mode_wanted):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Params

    with open(os.path.join(checkpoint, "config.json")) as f:
        cfg = json.load(f)
    layout = CSA2Layout.from_hf_config(cfg)
    text = cfg.get("text_config", cfg)
    heads = text.get("index_n_heads", 64)
    head_dim = text.get("index_head_dim", 128)
    for i in range(len(layout.compress_ratios)):
        if layout.compress_ratios[i] == 0:
            continue
        if layout.layer(i).mode == mode_wanted:
            return CSA2Indexer(layout, i, max(1, heads // 4), head_dim, CSA2Params(layout=layout))
    raise RuntimeError(f"no layer of mode {mode_wanted}")


def indexer_select(indexer, scores, out, ai, al):
    indexer.select_prepared_scores(scores, out, None, None, is_prefill=False,
                                   radix_aux_indices=ai, radix_aux_logits=al)
    return out


def mode_indexer(args):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    indexers = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
    impls = {}
    for m, ix in indexers.items():
        impls[str(m)] = [ix.top_k.decode_implementation.value, ix.top_k.prefill_implementation.value]
    log("decode/prefill impls", impls)
    # 8 indexer layers per step: 3 ratio-2 FULL @18432, 1 ratio-1 FULL @36864,
    # 4 candidate consumers (REINDEX) @16384.
    step = [(CSA2Mode.FULL, 18432)] * 3 + [(CSA2Mode.FULL, 36864)] + [(CSA2Mode.REINDEX, 16384)] * 4
    batch = {}
    per_shape = {}
    for rows in ROWS:
        bufs = []
        for li, (m, width) in enumerate(step):
            scores = make_scores(rows, width, seed=rows * 13 + li)
            ai, al = radix_aux(rows)
            o = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            bufs.append((indexers[m], scores, o, ai, al))

        def run_step():
            for ix, s, o, ai, al in bufs:
                indexer_select(ix, s, o, ai, al)

        t, nf = graph_time(run_step, inner=5, reps=30)
        batch[str(rows)] = {"kernel_ms": round(t / 1000.0, 5), "noise_floor": round(nf, 4)}
        log(f"rows={rows}: 8-layer decode top-K {t:.2f} us (nf {nf:.3f})")
        for m, width in ((CSA2Mode.FULL, 36864), (CSA2Mode.FULL, 18432), (CSA2Mode.REINDEX, 16384)):
            s = make_scores(rows, width, seed=rows + width)
            ai, al = radix_aux(rows)
            o = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            tt, nn = graph_time(lambda: indexer_select(indexers[m], s, o, ai, al))
            per_shape[f"{rows}x{width}"] = {"us": round(tt, 2), "nf": round(nn, 4)}
            log(f"  {rows}x{width}: {tt:.2f} us (nf {nn:.3f})")
    print(json.dumps({"batch": batch, "per_shape_us": per_shape, "impls": impls,
                      "output": {"class": "exact"}}))


def mode_check(args):
    """Untimed: warmup + iterations like timing; every iteration compared to CUDA radix."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    indexers = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
    impls = {str(m): ix.top_k.decode_implementation.value for m, ix in indexers.items()}
    pre = {str(m): ix.top_k.prefill_implementation.value for m, ix in indexers.items()}
    log("decode impls", impls, "prefill impls", pre)
    ref = TopK(TOPK, decode_implementation=TopKImplementation.CUDA_RADIX)
    failures = []
    stats = {"exact_rows": 0, "tie_rows": 0, "rows": 0, "iterations": 0, "graph_replays": 0}
    for m, widths in ((CSA2Mode.FULL, (36864, 18432)), (CSA2Mode.REINDEX, (16384,))):
        ix = indexers[m]
        for width in widths:
            for rows in ROWS:
                for ties in (False, True):
                    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
                    ai, al = radix_aux(rows)
                    ai2, al2 = radix_aux(rows)
                    o = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
                    o_ref = torch.empty_like(o)
                    s_buf = make_scores(rows, width, seed=1)
                    # 8-step decode window with changing scores (cache advancing)
                    for step in range(8):
                        s_buf.copy_(make_scores(rows, width, seed=100 + step + rows, ties=ties))
                        pristine = s_buf.clone()
                        for it in range(3 + 5):  # 3 warmup + 5 timed-like iterations, all checked
                            o.fill_(-7)
                            indexer_select(ix, s_buf, o, ai, al)
                            ref(pristine, o_ref, is_prefill=False, sequence_lengths=lens,
                                scan_lengths=lens, next_n=1, max_seq_len=width,
                                radix_aux_indices=ai2, radix_aux_logits=al2)
                            torch.cuda.synchronize()
                            if not torch.equal(s_buf, pristine):
                                failures.append(f"{m}/{width}/{rows}: scores mutated")
                            ex, tie, bad = compare(o_ref, o, pristine)
                            stats["exact_rows"] += ex
                            stats["tie_rows"] += tie
                            stats["rows"] += rows
                            stats["iterations"] += 1
                            if bad:
                                failures.append(f"{m}/{width}/{rows}/ties={ties}/step{step}/it{it}: rows {bad}")
                            if not ties and tie:
                                failures.append(f"{m}/{width}/{rows}/step{step}: non-tie rows differ")
                    # graph capture/replay over changing inputs
                    g_out = torch.empty_like(o)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        indexer_select(ix, s_buf, g_out, ai, al)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        indexer_select(ix, s_buf, g_out, ai, al)
                    for step in range(8):
                        s_buf.copy_(make_scores(rows, width, seed=500 + step, ties=ties))
                        g_out.fill_(-7)
                        g.replay()
                        ref(s_buf, o_ref, is_prefill=False, sequence_lengths=lens, scan_lengths=lens,
                            next_n=1, max_seq_len=width, radix_aux_indices=ai2, radix_aux_logits=al2)
                        torch.cuda.synchronize()
                        ex, tie, bad = compare(o_ref, g_out, s_buf)
                        stats["graph_replays"] += 1
                        if bad or (not ties and tie):
                            failures.append(f"graph {m}/{width}/{rows}/ties={ties}/step{step}: {bad} tie={tie}")
                    log(f"{m} width={width} rows={rows} ties={ties}: failures so far {len(failures)}")
    print(json.dumps({"passed": not failures,
                      "compared": "CSA2Indexer decode selection vs CUDA radix indexer_topk_decode "
                                  "(sorted sets; tie rows by selected-score multiset)",
                      "decode_impls": impls, "prefill_impls": pre, "stats": stats,
                      "failures": failures[:30]}))


def mode_profile(args):
    """One 8-layer decode top-K step at rows=4: the tree's CSA2Indexer path (NVTX new_step)
    and the CUDA radix TopK (NVTX old_step, what main runs), each 5x eagerly."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    indexers = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
    ref = TopK(TOPK, decode_implementation=TopKImplementation.CUDA_RADIX)
    rows = 4
    step = [(CSA2Mode.FULL, 18432)] * 3 + [(CSA2Mode.FULL, 36864)] + [(CSA2Mode.REINDEX, 16384)] * 4
    bufs = []
    for li, (m, width) in enumerate(step):
        ai, al = radix_aux(rows)
        bufs.append((indexers[m], make_scores(rows, width, seed=li), torch.empty(
            rows, TOPK, dtype=torch.int32, device="cuda"), ai, al,
            torch.full((rows,), width, dtype=torch.int32, device="cuda")))
    for _ in range(2):  # compile outside the ranges
        for ix, s, o, ai, al, lens in bufs:
            indexer_select(ix, s, o, ai, al)
    torch.cuda.synchronize()
    for _ in range(5):
        torch.cuda.nvtx.range_push("new_step")
        for ix, s, o, ai, al, lens in bufs:
            indexer_select(ix, s, o, ai, al)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("old_step")
        for ix, s, o, ai, al, lens in bufs:
            ref(s, o, is_prefill=False, sequence_lengths=lens, scan_lengths=lens, next_n=1,
                max_seq_len=s.shape[1], radix_aux_indices=ai, radix_aux_logits=al)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    print(json.dumps({"done": True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["engines", "indexer", "check", "profile"])
    p.add_argument("--checkpoint", default="")
    a = p.parse_args()
    {"engines": mode_engines, "indexer": mode_indexer, "check": mode_check,
     "profile": mode_profile}[a.mode](a)
