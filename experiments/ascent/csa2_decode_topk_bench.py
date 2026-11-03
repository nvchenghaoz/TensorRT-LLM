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


_FLUSH = {}


def _l2_flush():
    buf = _FLUSH.get("b")
    if buf is None:
        buf = _FLUSH["b"] = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    buf.zero_()


def _sm_clock_mhz():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:  # noqa: BLE001
        try:
            return int(torch.cuda.clock_rate())
        except Exception:  # noqa: BLE001
            return -1


def graph_time_flushed(fn, reps=60):
    """One step captured in a graph; L2 flushed (256 MB zero) before every replay, outside the
    timed events. Returns (median_us, p10_us, p90_us, sm_clock_mhz sampled during the reps)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        _l2_flush()
        g.replay()
    torch.cuda.synchronize()
    times, clocks = [], []
    for i in range(reps):
        _l2_flush()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        if i % 15 == 7:
            clocks.append(_sm_clock_mhz())
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    times.sort()
    med = times[len(times) // 2]
    p10 = times[int(0.1 * (len(times) - 1))]
    p90 = times[int(0.9 * (len(times) - 1))]
    return med, p10, p90, (min(clocks) if clocks else -1)


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

        med, p10, p90, clk = graph_time_flushed(run_step)
        tw, nfw = graph_time(run_step, inner=5, reps=30)
        nf = (p90 - p10) / med
        batch[str(rows)] = {"kernel_ms": round(med / 1000.0, 5), "noise_floor": round(nf, 4),
                            "p10_ms": round(p10 / 1000.0, 5), "p90_ms": round(p90 / 1000.0, 5),
                            "sm_clock_mhz_min": clk,
                            "warm_l2_kernel_ms": round(tw / 1000.0, 5), "warm_l2_noise": round(nfw, 4)}
        log(f"rows={rows}: 8-layer decode top-K L2-flushed median {med:.2f} us "
            f"(p10 {p10:.2f}, p90 {p90:.2f}, nf {nf:.3f}, sm clk {clk} MHz); warm-L2 {tw:.2f} us")
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


STEP = None  # filled lazily (needs CSA2Mode)


def _step_layers():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    return [(CSA2Mode.FULL, 18432)] * 3 + [(CSA2Mode.FULL, 36864)] + [(CSA2Mode.REINDEX, 16384)] * 4


def _trace_section(indexers, old_indexers):
    """torch.profiler (CUPTI) trace of one 8-layer decode top-K step, rows 4, both arms through
    CSA2Indexer.select_prepared_scores with explicit row ends (as production passes them):
    new_step = this tree's indexer (CuTe-DSL decode), old_step = same indexer class with the
    decode implementation set to CUDA_RADIX (what main runs). Reduced in-job."""
    from torch.profiler import ProfilerActivity, profile, record_function
    import tempfile

    rows = 4
    bufs = []
    for li, (m, width) in enumerate(_step_layers()):
        ai, al = radix_aux(rows)
        bufs.append((m, make_scores(rows, width, seed=li),
                     torch.empty(rows, TOPK, dtype=torch.int32, device="cuda"), ai, al,
                     torch.full((rows,), width, dtype=torch.int32, device="cuda")))

    def new_step():
        for m, s, o, ai, al, lens in bufs:
            indexers[m].select_prepared_scores(s, o, None, lens, is_prefill=False,
                                               radix_aux_indices=ai, radix_aux_logits=al)

    def old_step():
        for m, s, o, ai, al, lens in bufs:
            old_indexers[m].select_prepared_scores(s, o, None, lens, is_prefill=False,
                                                   radix_aux_indices=ai, radix_aux_logits=al)

    for _ in range(2):
        new_step()
        old_step()
    torch.cuda.synchronize()
    reps = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            for name, fn in (("new_step", new_step), ("old_step", old_step)):
                torch.cuda.synchronize()
                with record_function(name):
                    fn()
                    torch.cuda.synchronize()
    path = os.path.join(tempfile.mkdtemp(), "trace.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        ev = json.load(f)
    ev = ev.get("traceEvents", ev)
    wins = [(e["name"], float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)))
            for e in ev if e.get("cat") == "user_annotation" and e.get("name") in ("new_step", "old_step")]
    kern = [e for e in ev if e.get("cat") == "kernel"]
    out = {n: {} for n in ("new_step", "old_step")}
    unassigned = 0
    for k in kern:
        t0 = float(k["ts"])
        hit = None
        for n, a, b in wins:
            if a <= t0 <= b:
                hit = n
                break
        if hit is None:
            unassigned += 1
            continue
        nm = k["name"][:120]
        d = out[hit].setdefault(nm, {"count": 0, "total_us": 0.0})
        d["count"] += 1
        d["total_us"] += float(k.get("dur", 0))
    summary = {}
    for n, d in out.items():
        per_step = {nm: {"per_step": v["count"] / reps, "us_per_step": round(v["total_us"] / reps, 2)}
                    for nm, v in sorted(d.items(), key=lambda x: -x[1]["total_us"])}
        summary[n] = {"kernels": per_step,
                      "launches_per_step": sum(v["count"] for v in d.values()) / reps,
                      "kernel_us_per_step": round(sum(v["total_us"] for v in d.values()) / reps, 2)}
    bad_words = ("pad", "copy", "elementwise", "fill", "cat", "contiguous")
    new_names = list(out["new_step"].keys())
    extra = [nm for nm in new_names if any(w in nm.lower() for w in bad_words)]
    only_topk = (len(new_names) >= 1 and not extra
                 and summary["new_step"]["launches_per_step"] == len(bufs))
    return {"rows": rows, "layers": len(bufs), "reps": reps, "windows": len(wins),
            "unassigned_kernels": unassigned, "ranges": summary,
            "new_step_extra_copy_pad_kernels": extra,
            "new_step_one_topk_launch_per_layer_only": only_topk}


def _coarse_bin(x):
    # to_coarse_key: fp32 -> fp16, ordered transform of the 16 bits, high 8 bits. The ordered
    # transform is a bijection that maps equal high bytes to equal high bytes, so two values
    # share a coarse bin iff their fp16 bit patterns share the high byte.
    return (x.half().view(torch.int16).to(torch.int32) >> 8) & 0xFF


def _overflow_inputs(rows, width, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "all_equal":
        s = torch.full((rows, width), 0.5, device="cuda")
    elif kind == "few_distinct_dense":
        # every column one of 4 values in the same coarse bin (fp16 0x3C00..0x3C03)
        v = 1.0 + torch.randint(0, 4, (rows, width), generator=g, device="cuda").float() * 2.0 ** -10
        s = v
    elif kind == "few_distinct_tail":
        # dense block of 4 near-top values + randn rest + -inf invisible tail
        s = torch.randn(rows, width, generator=g, device="cuda") * 0.01
        n = width * 3 // 4
        s[:, :n] = 1.0 + torch.randint(0, 4, (rows, n), generator=g, device="cuda").float() * 2.0 ** -10
        for r in range(rows):
            t = (r * 997 + seed) % 2048
            if t:
                s[r, width - t:] = -torch.inf
    else:
        raise ValueError(kind)
    return s.contiguous().float()


def _kernel_geometry(rows, width):
    import cutlass
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as C
    from tensorrt_llm._torch.kernels.blackwell.top_k.filtered_top_k_decode_varlen import (
        FilteredTopKKernelVarlenDecode,
    )
    import math as _m

    cs = C.CuteDSLTopKDecodeRadixFilterSPMultiCTARunner.auto_cluster_size(width, rows, True)
    bucket = 1 << (width - 1).bit_length()
    try:
        return _kernel_geometry_inner(rows, width, cs, bucket, cutlass, FilteredTopKKernelVarlenDecode, _m)
    except Exception as e:  # noqa: BLE001
        log("geometry introspection failed", e)
        return {"cluster_size": cs, "chunk_cols_per_cta": _m.ceil(bucket / cs) if cs >= 2 else width,
                "smem_candidate_capacity_S": -1, "reread_compiled_in": None,
                "error": f"{type(e).__name__}: {e}"[:300]}


def _kernel_geometry_inner(rows, width, cs, bucket, cutlass, FilteredTopKKernelVarlenDecode, _m):
    if cs >= 2:
        chunk = _m.ceil(bucket / cs)
        k = FilteredTopKKernelVarlenDecode(cutlass.Float32, bucket, TOPK, 1, chunk_size_per_cta=chunk,
                                           num_ctas_per_row=cs, overflow_policy="REREAD",
                                           single_pass_multi_cta=True)
    else:
        chunk = width
        k = FilteredTopKKernelVarlenDecode(cutlass.Float32, bucket, TOPK, 1,
                                           large_occupancy=rows > torch.cuda.get_device_properties(0).multi_processor_count,
                                           overflow_policy="REREAD")
    return {"cluster_size": cs, "chunk_cols_per_cta": chunk,
            "smem_candidate_capacity_S": int(k.filtered_topk_smem_input_size),
            "reread_compiled_in": bool(k.enable_reread)}


def _threshold_bin_counts(s, chunk):
    """Per row: the coarse bin of the K-th largest value, and the max per-CTA-chunk count of
    elements falling into it (what the kernel writes to its SMEM candidate buffer)."""
    rows, width = s.shape
    kth = torch.topk(s, TOPK, dim=1).values[:, -1:]
    tb = _coarse_bin(kth)
    bins = _coarse_bin(s)
    inbin = (bins == tb)
    mx = 0
    for c0 in range(0, width, chunk):
        mx = max(mx, int(inbin[:, c0:c0 + chunk].sum(1).max().item()))
    return int(inbin.sum(1).max().item()), mx


def _run_overflow_case(ix, ref, rows, width, kind, failures, stats, label):
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    ai, al = radix_aux(rows)
    ai2, al2 = radix_aux(rows)
    o = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    o_ref = torch.empty_like(o)
    s_buf = _overflow_inputs(rows, width, kind, seed=rows * 31 + width)
    pristine = s_buf.clone()
    for it in range(3 + 5):
        o.fill_(-7)
        if ix is None:
            ref_dsl = stats["_dsl"]
            ref_dsl(s_buf, o, is_prefill=False, sequence_lengths=lens, scan_lengths=lens, next_n=1,
                    max_seq_len=width, radix_aux_indices=ai, radix_aux_logits=al)
        else:
            indexer_select(ix, s_buf, o, ai, al)
        ref(pristine, o_ref, is_prefill=False, sequence_lengths=lens, scan_lengths=lens, next_n=1,
            max_seq_len=width, radix_aux_indices=ai2, radix_aux_logits=al2)
        torch.cuda.synchronize()
        if not torch.equal(s_buf, pristine):
            failures.append(f"{label}: scores mutated")
        ex, tie, bad = compare(o_ref, o, pristine)
        stats["iterations"] += 1
        stats["rows"] += rows
        stats["exact_rows"] += ex
        stats["tie_rows"] += tie
        if bad:
            failures.append(f"{label}/it{it}: rows {bad}")
    if ix is not None:
        g_out = torch.empty_like(o)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            indexer_select(ix, s_buf, g_out, ai, al)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            indexer_select(ix, s_buf, g_out, ai, al)
        for step in range(4):
            s_buf.copy_(_overflow_inputs(rows, width, kind, seed=900 + step))
            g_out.fill_(-7)
            g.replay()
            ref(s_buf, o_ref, is_prefill=False, sequence_lengths=lens, scan_lengths=lens, next_n=1,
                max_seq_len=width, radix_aux_indices=ai2, radix_aux_logits=al2)
            torch.cuda.synchronize()
            ex, tie, bad = compare(o_ref, g_out, s_buf)
            stats["graph_replays"] += 1
            if bad:
                failures.append(f"graph {label}/step{step}: {bad}")
    return s_buf


def mode_evalchecks(args):
    """Evaluator items: (1) per-range kernel list trace, (2) forced-overflow inputs (production
    budget: bin count vs capacity; forced small budget: REREAD actually compiled in and taken),
    (3) raw (unsorted) index-row equality vs CUDA radix."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as C
    from tensorrt_llm._torch.kernels.blackwell.top_k import filtered_top_k_varlen_util as U

    indexers = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
    impls = {str(m): ix.top_k.decode_implementation.value for m, ix in indexers.items()}
    pre = {str(m): ix.top_k.prefill_implementation.value for m, ix in indexers.items()}
    log("decode impls", impls, "prefill impls", pre)
    ref = TopK(TOPK, decode_implementation=TopKImplementation.CUDA_RADIX)
    dsl = TopK(TOPK, decode_implementation=TopKImplementation.CUTE_DSL_RADIX)
    result = {"decode_impls": impls, "prefill_impls": pre}
    failures = []

    # (1) trace
    try:
        old_ix = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
        for ix in old_ix.values():
            ix.top_k.decode_implementation = TopKImplementation.CUDA_RADIX
        result["trace_old_arm_impl"] = {str(m): ix.top_k.decode_implementation.value
                                        for m, ix in old_ix.items()}
        result["trace"] = _trace_section(indexers, old_ix)
        log("trace", json.dumps(result["trace"])[:3000])
        if not result["trace"]["new_step_one_topk_launch_per_layer_only"]:
            failures.append("trace: new_step has extra kernels")
    except Exception as e:  # noqa: BLE001
        result["trace"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        failures.append("trace error")
        log("trace error", e)

    # (2a) forced-overflow inputs on the production config (indexer path)
    geo = {}
    stats = {"exact_rows": 0, "tie_rows": 0, "rows": 0, "iterations": 0, "graph_replays": 0}
    bins = {}
    for m, width in ((CSA2Mode.FULL, 36864), (CSA2Mode.FULL, 18432), (CSA2Mode.REINDEX, 16384)):
        for rows in ROWS:
            gm = _kernel_geometry(rows, width)
            geo[f"{rows}x{width}"] = gm
            for kind in ("all_equal", "few_distinct_dense", "few_distinct_tail"):
                label = f"prod/{kind}/{rows}x{width}"
                s = _run_overflow_case(indexers[m], ref, rows, width, kind, failures, stats, label)
                s0 = _overflow_inputs(rows, width, kind, seed=rows * 31 + width)
                tot, per_cta = _threshold_bin_counts(s0, gm["chunk_cols_per_cta"])
                bins[label] = {"threshold_bin_row_max": tot, "threshold_bin_per_cta_max": per_cta,
                               "capacity_S": gm["smem_candidate_capacity_S"],
                               "would_overflow": (gm["smem_candidate_capacity_S"] < 0
                                                  or per_cta > gm["smem_candidate_capacity_S"])}
            log(f"prod {rows}x{width} geo={gm}: failures so far {len(failures)}")
    result["production_overflow"] = {"geometry": geo, "stats": stats, "threshold_bins": bins,
                                     "any_case_reaches_overflow": any(v["would_overflow"] for v in bins.values())}

    # (2b) shrink the candidate budget so REREAD is compiled in and the bin overflows; prove the
    # SM120 REREAD fallback is exact. Kernel caches are cleared before and after.
    caches = [C.CuteDSLTopKDecodeRadixFilterSPMultiCTARunner.kernel_cache,
              C.CuteDSLTopKDecodeSingleCTARunner.kernel_cache]
    dev = torch.cuda.current_device()
    forced = {"stats": {"exact_rows": 0, "tie_rows": 0, "rows": 0, "iterations": 0,
                        "graph_replays": 0, "_dsl": dsl}, "geometry": {}, "threshold_bins": {}}
    try:
        for c in caches:
            c.clear()
        U._input_idx_budget_cache[dev] = 16 * 1024  # S = 16 KB / (2 bufs * 2 B) = 4096 < 8192 chunk
        for width in (36864, 18432, 16384):
            for rows in ROWS:
                gm = _kernel_geometry(rows, width)
                forced["geometry"][f"{rows}x{width}"] = gm
                for kind in ("all_equal", "few_distinct_dense", "few_distinct_tail"):
                    label = f"forced/{kind}/{rows}x{width}"
                    _run_overflow_case(None, ref, rows, width, kind, failures, forced["stats"], label)
                    s0 = _overflow_inputs(rows, width, kind, seed=rows * 31 + width)
                    tot, per_cta = _threshold_bin_counts(s0, gm["chunk_cols_per_cta"])
                    forced["threshold_bins"][label] = {
                        "threshold_bin_per_cta_max": per_cta, "capacity_S": gm["smem_candidate_capacity_S"],
                        "reread_compiled_in": gm["reread_compiled_in"],
                        "overflow_taken": gm["reread_compiled_in"] and per_cta > gm["smem_candidate_capacity_S"]}
                log(f"forced {rows}x{width} geo={gm}: failures so far {len(failures)}")
        forced["all_cases_took_reread"] = all(v["overflow_taken"] for v in forced["threshold_bins"].values())
    except Exception as e:  # noqa: BLE001
        forced["error"] = f"{type(e).__name__}: {e}"[:500]
        failures.append("forced-overflow error")
        log("forced error", e)
    finally:
        U._input_idx_budget_cache.pop(dev, None)
        for c in caches:
            c.clear()
    forced["stats"].pop("_dsl", None)
    result["forced_reread"] = forced

    # (3) raw (unsorted) row order vs CUDA radix, random (non-tie) inputs, production config
    raw = {}
    for m, width in ((CSA2Mode.FULL, 36864), (CSA2Mode.FULL, 18432), (CSA2Mode.REINDEX, 16384)):
        for rows in ROWS:
            s = make_scores(rows, width, seed=77 + rows)
            lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
            ai, al = radix_aux(rows)
            o = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            o_ref = torch.empty_like(o)
            indexer_select(indexers[m], s, o, ai, al)
            ref(s, o_ref, is_prefill=False, sequence_lengths=lens, scan_lengths=lens, next_n=1,
                max_seq_len=width, radix_aux_indices=ai, radix_aux_logits=al)
            torch.cuda.synchronize()
            raw_eq = sum(int(torch.equal(o[r], o_ref[r])) for r in range(rows))
            set_eq = sum(int(torch.equal(sorted_sel(o)[r], sorted_sel(o_ref)[r])) for r in range(rows))
            raw[f"{rows}x{width}"] = {"raw_rows_equal": raw_eq, "sorted_rows_equal": set_eq, "rows": rows}
            if set_eq != rows:
                failures.append(f"raw-section set mismatch {rows}x{width}")
    result["raw_order"] = {"per_shape": raw,
                           "all_raw_equal": all(v["raw_rows_equal"] == v["rows"] for v in raw.values()),
                           "all_sets_equal": all(v["sorted_rows_equal"] == v["rows"] for v in raw.values())}
    result["passed"] = not failures
    result["failures"] = failures[:30]
    result["compared"] = ("CSA2Indexer / CuTe-DSL decode top-K vs CUDA radix indexer_topk_decode; "
                          "3 warmup + 5 iterations each compared, graph replay; tie rows by "
                          "selected-score multiset")
    print(json.dumps({"passed": result.pop("passed"), **result}))


def mode_padcheck(args):
    """Top-lead item 3 evidence: eager torch.profiler kernel lists (launch order) of one 8-layer
    decode top-K step, old (CUDA radix) vs new, rows 1/2/4/8; F.pad call counter under CUDA-graph
    capture; scores alignment; near-constant scores at default policy and REREAD_ALWAYS."""
    from torch.profiler import ProfilerActivity, profile
    import torch.nn.functional as F
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    new_ix = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
    old_ix = {m: build_indexer(args.checkpoint, m) for m in (CSA2Mode.FULL, CSA2Mode.REINDEX)}
    for ix in old_ix.values():
        ix.top_k.decode_implementation = TopKImplementation.CUDA_RADIX
    impls = {"new": {str(m): ix.top_k.decode_implementation.value for m, ix in new_ix.items()},
             "old": {str(m): ix.top_k.decode_implementation.value for m, ix in old_ix.items()}}
    log("impls", impls)
    failures = []
    pad_calls = {"n": 0}
    orig_pad = F.pad

    def counting_pad(*a, **k):
        pad_calls["n"] += 1
        return orig_pad(*a, **k)

    F.pad = counting_pad
    bad_words = ("pad", "copy", "elementwise", "fill")
    steps = {}
    try:
        for rows in ROWS:
            bufs = []
            for li, (m, width) in enumerate(_step_layers()):
                ai, al = radix_aux(rows)
                bufs.append((m, make_scores(rows, width, seed=li + rows),
                             torch.empty(rows, TOPK, dtype=torch.int32, device="cuda"), ai, al,
                             torch.full((rows,), width, dtype=torch.int32, device="cuda")))
            align = [{"width": s.shape[1], "stride0_mod8": s.stride(0) % 8,
                      "ptr_mod32": s.data_ptr() % 32, "width_mod8": s.shape[1] % 8}
                     for _, s, _, _, _, _ in bufs]

            def step(ixs):
                for m, s, o, ai, al, lens in bufs:
                    ixs[m].select_prepared_scores(s, o, None, lens, is_prefill=False,
                                                  radix_aux_indices=ai, radix_aux_logits=al)

            for _ in range(2):
                step(new_ix)
                step(old_ix)
            torch.cuda.synchronize()
            rec = {"align": align}
            for arm, ixs in (("old", old_ix), ("new", new_ix)):
                pad_calls["n"] = 0
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    step(ixs)
                    torch.cuda.synchronize()
                names = [e.name for e in sorted(
                    (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                    key=lambda e: e.time_range.start)]
                rec[arm] = {"kernels_in_launch_order": [n[:90] for n in names],
                            "launches": len(names),
                            "pad_copy_elementwise_kernels": sum(
                                1 for n in names if any(w in n.lower() for w in bad_words)),
                            "F_pad_calls_eager": pad_calls["n"]}
            # graph capture of the new step with the F.pad counter
            pad_calls["n"] = 0
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                step(new_ix)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            pad_calls["n"] = 0
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step(new_ix)
            rec["new_F_pad_calls_under_capture"] = pad_calls["n"]
            g.replay()
            torch.cuda.synchronize()
            n_new = rec["new"]["pad_copy_elementwise_kernels"]
            n_old = rec["old"]["pad_copy_elementwise_kernels"]
            rec["new_adds_pad_copy_kernels"] = n_new > n_old
            if n_new > n_old or rec["new_F_pad_calls_under_capture"] or rec["new"]["F_pad_calls_eager"]:
                failures.append(f"rows {rows}: pad/copy found {rec}")
            if rec["new"]["launches"] != len(bufs):
                failures.append(f"rows {rows}: new launches {rec['new']['launches']} != {len(bufs)}")
            steps[str(rows)] = rec
            log(f"rows {rows}: old {rec['old']['launches']} launches, new {rec['new']['launches']}, "
                f"pad under capture {rec['new_F_pad_calls_under_capture']}")
    finally:
        F.pad = orig_pad

    # near-constant scores 1 + 1e-6 * randn, default policy and REREAD_ALWAYS
    ref = TopK(TOPK, decode_implementation=TopKImplementation.CUDA_RADIX)
    near = {}
    for width in WIDTHS:
        for rows in ROWS:
            g_ = torch.Generator(device="cuda").manual_seed(rows * 5 + width)
            s = (1.0 + 1e-6 * torch.randn(rows, width, generator=g_, device="cuda")).float().contiguous()
            lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
            ai, al = radix_aux(rows)
            o_ref = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
            ref(s, o_ref, is_prefill=False, sequence_lengths=lens, scan_lengths=lens, next_n=1,
                max_seq_len=width, radix_aux_indices=ai, radix_aux_logits=al)
            gm = _kernel_geometry(rows, width)
            _, per_cta = _threshold_bin_counts(s, gm["chunk_cols_per_cta"])
            rec = {"threshold_bin_per_cta_max": per_cta, "capacity_S": gm["smem_candidate_capacity_S"],
                   "reread_compiled_in": gm["reread_compiled_in"], "cluster_size": gm["cluster_size"]}
            for pol in ("REREAD", "REREAD_ALWAYS"):
                o = torch.empty_like(o_ref)
                tot = {"exact": 0, "tie": 0, "bad": 0, "iterations": 0}
                try:
                    for it in range(3 + 5):
                        o.fill_(-7)
                        torch.ops.trtllm.cute_dsl_indexer_topk_decode(s, lens, o, TOPK, 1,
                                                                      overflow_policy=pol)
                        torch.cuda.synchronize()
                        ex, tie, bad = compare(o_ref, o, s)
                        tot["exact"] += ex
                        tot["tie"] += tie
                        tot["bad"] += len(bad)
                        tot["iterations"] += 1
                        if bad:
                            failures.append(f"near-const {pol} {rows}x{width} it{it}: rows {bad}")
                except Exception as e:  # noqa: BLE001
                    tot["error"] = f"{type(e).__name__}: {e}"[:300]
                    failures.append(f"near-const {pol} {rows}x{width} error")
                rec[pol] = tot
            near[f"{rows}x{width}"] = rec
            log(f"near-const {rows}x{width}: {rec}")
    print(json.dumps({"passed": not failures, "impls": impls, "steps": steps,
                      "near_constant": near, "failures": failures[:30],
                      "compared": "kernel lists old (CUDA radix) vs new (CuTe-DSL) per 8-layer step; "
                                  "near-constant scores vs CUDA radix, tie-aware (selected-score multiset)"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["engines", "indexer", "check", "profile", "evalchecks", "padcheck"])
    p.add_argument("--checkpoint", default="")
    a = p.parse_args()
    {"engines": mode_engines, "indexer": mode_indexer, "check": mode_check,
     "profile": mode_profile, "evalchecks": mode_evalchecks, "padcheck": mode_padcheck}[a.mode](a)
