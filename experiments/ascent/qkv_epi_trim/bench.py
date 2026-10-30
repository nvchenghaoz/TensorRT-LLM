"""Branch 54 / idea 84: compact per-axis RoPE table for the fused QKV epilogue.

Synthetic NVFP4 activations/weights at the served shapes (N = K = 3072); the RoPE cos_sin tables
are the model's own (rope.make_cos_sin: Flux3PosEmbed -> rope_cos_sin_table) for each served M.

  pair   the served op, epi.qkv_rope_fp8(keep_v=False), with the compact table when the tree has
         it (candidate) and without (base): run_gpu_pair arm. Prints the evidence JSON.
  diag   candidate only, one process: GEMM launch full / compact / nocs probe / noop floor,
         op full / op compact, interleaved in a seeded random order; build_rope_compact cost.
  check  bitwise matrix vs the levers-off route (full table) and main's two-pass chain.
  kill   run under TRTLLM_FLUX3_QKV_EPI_FP8=0: impl with compact tensors == main's chain.
Env: B_MS, B_R, B_ITERS. Last stdout line: JSON.
"""

import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idea14_epi"))
from op import SM, N, compare, main_chain, make_inputs  # noqa: E402
from rope import make_cos_sin, rope_config  # noqa: E402

CKPT = os.environ.get("B_CKPT")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stats(t):
    t = sorted(t)
    med = t[len(t) // 2]
    return med, (t[int(0.9 * len(t)) - 1] - t[int(0.1 * len(t))]) / med


def inputs_for(m, dev, cond=False):
    a, w, sfa, sfb, alpha, qn, kn, _cs, pos = make_inputs(m, dev)
    cs = make_cos_sin(m, dev, CKPT, cond=cond)
    return a, w, sfa, sfb, alpha, qn, kn, cs, pos


def time_graphs(fns, R, ITERS, seed=54):
    for f in fns.values():
        f()
        f()
    torch.cuda.synchronize()
    graphs = {}
    for name, f in fns.items():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(R):
                f()
        graphs[name] = g
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    # Warm the clocks before timing (the first M measured otherwise shows a wide spread).
    t_end = time.time() + float(os.environ.get("B_WARM_S", "3"))
    while time.time() < t_end:
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
    rng = random.Random(seed)
    times = {n: [] for n in graphs}
    order = list(graphs)
    for _ in range(ITERS):
        rng.shuffle(order)
        for name in order:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[name].replay()
            e1.record()
            e1.synchronize()
            times[name].append(e0.elapsed_time(e1) / R)
    del graphs
    return {n: stats(t) for n, t in times.items()}


def setup():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    epi.preload(dev)
    has = hasattr(epi, "build_rope_compact")
    log(f"rope config {rope_config(CKPT)}; compact in tree: {has}; "
        f"ROPE_COMPACT={getattr(epi, 'ROPE_COMPACT', None)} VQ_REVERSE={epi.VQ_REVERSE}")
    return epi, strided, dev, has


def pair(ms):
    epi, _, dev, has = setup()
    R = int(os.environ.get("B_R", "4"))
    ITERS = int(os.environ.get("B_ITERS", "60"))
    batch = {}
    for m in ms:
        a, w, sfa, sfb, alpha, qn, kn, cs, _ = inputs_for(m, dev)
        extra = ()
        if has:
            t0 = time.time()
            compact = epi.build_rope_compact(cs)
            torch.cuda.synchronize()
            log(f"M{m} build_rope_compact {1e3 * (time.time() - t0):.1f} ms, entries "
                f"{None if compact is None else compact[0].shape[0]}")
            assert compact is not None, "compact table not built"
            extra = compact
        fn = (lambda: epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False, *extra))
        res = time_graphs({"op": fn, "op_aa": fn}, R, ITERS)
        med, nf = res["op"]
        log(f"M{m} op {med:.4f} ms nf {nf:.4f}; A/A {res['op_aa'][0]:.4f}")
        batch[str(m)] = {"kernel_ms": round(med, 5), "noise_floor": round(max(nf, 1e-4), 4),
                         "aa_ms": round(res["op_aa"][0], 5), "compact": bool(extra)}
        del a, w, sfa, sfb
        torch.cuda.empty_cache()
    return {"batch": batch, "output": {"class": "exact"}}


def diag(ms):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR

    from b54_variants import build_all

    epi, strided, dev, has = setup()
    assert has
    mod = epi.module()
    var = build_all(["noop", "nocs"], log)
    R = int(os.environ.get("B_R", "4"))
    ITERS = int(os.environ.get("B_ITERS", "40"))
    minsf = float(_MIN_SCALING_FACTOR)
    out = {}
    for m in ms:
        a, w, sfa, sfb, alpha, qn, kn, cs, _ = inputs_for(m, dev)
        builds = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            tab, idx = epi.build_rope_compact(cs)
            torch.cuda.synchronize()
            builds.append(1e3 * (time.time() - t0))
        qkv8 = torch.empty((3, m, N), dtype=torch.uint8, device=dev)
        v = torch.empty((m, N), dtype=torch.bfloat16, device=dev)
        parts = torch.zeros(((m + 127) // 128) * (N // 128) + 8, dtype=torch.float32, device=dev)
        ws = strided.workspace(dev, 0)
        base = (a, w, sfa, sfb, alpha, qn, kn, cs, qkv8, v, parts, ws, minsf)
        fns = {
            "gemm_full": lambda: mod.qkv3_epi_fp8(*base),
            "gemm_compact": lambda: mod.qkv3_epi_fp8_compact(*base, tab, idx),
            "gemm_nocs": lambda: var["nocs"].qkv3_epi_fp8(*base),
            "gemm_noop": lambda: var["noop"].qkv3_epi_fp8(*base),
            "op_full": lambda: epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False),
            "op_compact": lambda: epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False,
                                                   tab, idx),
        }
        res = time_graphs(fns, R, ITERS)
        row = {n: {"ms": round(v_[0], 4), "nf": round(v_[1], 4)} for n, v_ in res.items()}
        row["entries"] = int(idx.unique().numel())
        row["own_entry_rows"] = int((idx.long() == (torch.arange(m, device=dev).unsqueeze(1) * 4
                                                    + torch.arange(4, device=dev))).sum())
        row["build_ms"] = [round(b, 2) for b in builds]
        row["save_compact_ms"] = round(row["gemm_full"]["ms"] - row["gemm_compact"]["ms"], 4)
        row["save_nocs_ms"] = round(row["gemm_full"]["ms"] - row["gemm_nocs"]["ms"], 4)
        log(f"M{m}", json.dumps(row))
        out[str(m)] = row
        torch.cuda.empty_cache()
    return {"diag": out}


def _snap(t):
    return tuple(x.clone() for x in t)


def check_matrix(ms):
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
        flux3_sm120_qknorm_rope_fp8 as qr8,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    epi, strided, dev, has = setup()
    assert has
    fails, counts = [], {"eager": 0, "replay": 0, "op": 0}
    cases = []
    for m in ms:
        cases.append((m, False))
        if m == 16384:
            cases.append((m, True))  # the i2i image_cond table (l = -1)
    data = {}
    for m, cond in cases:
        inp = inputs_for(m, dev, cond)
        cs = inp[7]
        tab, idx = epi.build_rope_compact(cs)
        os.environ[epi._ROPE_COMPACT_FORCE_MISMATCH_ENV] = "1"
        forced = epi.build_rope_compact(cs)
        os.environ.pop(epi._ROPE_COMPACT_FORCE_MISMATCH_ENV)
        if forced is not None:
            fails.append(f"M{m} c{int(cond)}: forced mismatch still built a table")
        # Independent check of the table: rebuild cos_sin from it in plain torch.
        cols = epi._rope_columns(dev)
        rebuilt = torch.empty_like(cs.view(torch.int32))
        for ax in range(4):
            rebuilt[:, cols[ax]] = tab.view(torch.int32)[idx[:, ax].long()]
        if not torch.equal(rebuilt, cs.view(torch.int32)):
            fails.append(f"M{m} c{int(cond)}: tab[idx] != cos_sin")
        a, w, sfa, sfb, alpha, qn, kn, _, pos = inp
        refs = {}
        for keep_v in (True, False):
            ref = _snap(epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v))
            refs[keep_v] = ref
            chain = _snap(main_chain(strided, qr8, inp, True))
            compare(f"M{m} c{int(cond)} kv{int(keep_v)} full-route vs two-pass chain", chain, ref,
                    fails, check_v=keep_v)
        log(f"M{m} c{int(cond)}: entries {tab.shape[0]}, refs ready, fails {len(fails)}")
        data[(m, cond)] = (inp, tab, idx, refs, [t.clone() for t in inp] + [tab.clone(),
                                                                            idx.clone()])
    # Device-side verification failure: corrupt one non-key word of a row whose axis-2 slice
    # points at another row; that row must fall back to its own entry, outputs still bitwise.
    key0 = list(data)[0]
    (inp, tab, idx, refs, _) = data[key0]
    m0 = key0[0]
    def own_entries(n):
        return torch.arange(n, device=dev).unsqueeze(1) * 4 + torch.arange(4, device=dev)

    own = own_entries(m0)
    fallback_rows = {f"{k[0]}{'c' if k[1] else ''}":
                     int((d[2].long() == own_entries(d[2].shape[0])).sum())
                     for k, d in data.items()}
    nz = (idx[:, 2].long() != own[:, 2]).nonzero().flatten()
    r = int(nz[len(nz) // 2])
    cs2 = inp[7].clone()
    col = int(epi._rope_columns(dev)[2][5])
    cs2.view(torch.int32)[r, col] ^= 1
    tab2, idx2 = epi.build_rope_compact(cs2)
    if int(idx2[r, 2]) != 4 * r + 2:
        fails.append(f"mismatch row {r}: idx {int(idx2[r, 2])} != own entry {4 * r + 2}")
    cols = epi._rope_columns(dev)
    rebuilt = torch.empty_like(cs2.view(torch.int32))
    for ax in range(4):
        rebuilt[:, cols[ax]] = tab2.view(torch.int32)[idx2[:, ax].long()]
    if not torch.equal(rebuilt, cs2.view(torch.int32)):
        fails.append("mismatch case: tab[idx] != corrupted cos_sin")
    a, w, sfa, sfb, alpha, qn, kn, _, pos = inp
    ref2 = _snap(epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs2, SM, True))
    got2 = epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs2, SM, True, tab2, idx2)
    torch.cuda.synchronize()
    compare(f"M{m0} device-mismatch fallback", ref2, got2, fails)
    log(f"mismatch test row {r}: fails {len(fails)}; own-entry rows per case {fallback_rows}")
    del cs2, tab2, idx2, ref2, got2
    # Alternating M order, eager calls and graph replays, keep_v both ways, repeated rounds.
    graphs = {}
    for rnd in range(3):
        order = list(data) if rnd % 2 == 0 else list(reversed(list(data)))
        for key in order:
            (inp, tab, idx, refs, pristine) = data[key]
            m, cond = key
            a, w, sfa, sfb, alpha, qn, kn, cs, pos = inp
            for keep_v in (True, False):
                tag = f"r{rnd} M{m} c{int(cond)} kv{int(keep_v)}"
                call0 = None
                for i in range(3):
                    got = epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v, tab, idx)
                    torch.cuda.synchronize()
                    compare(f"{tag} eager{i}", refs[keep_v], got, fails, check_v=keep_v)
                    if call0 is None:
                        call0 = _snap(got)
                    else:
                        compare(f"{tag} eager{i} vs call0", call0, got, fails, check_v=keep_v)
                    counts["eager"] += 1
                # The custom op (torch.ops path the model uses) with the compact tensors.
                got = fp.nvfp4_qkv_rope_fp8_impl(a, w, sfa, sfb, qn, kn, cs, pos, alpha, SM,
                                                 keep_v, True, tab, idx)
                torch.cuda.synchronize()
                compare(f"{tag} impl", refs[keep_v], got, fails, check_v=keep_v)
                counts["op"] += 1
                # ROPE_COMPACT off with the tensors passed: falls back to the full table.
                epi.ROPE_COMPACT = False
                got = epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v, tab, idx)
                epi.ROPE_COMPACT = True
                torch.cuda.synchronize()
                compare(f"{tag} lever-off", refs[keep_v], got, fails, check_v=keep_v)
                gk = (key, keep_v)
                if gk not in graphs:
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v, tab, idx)
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        gout = epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v,
                                                tab, idx)
                    graphs[gk] = (g, gout)
                g, gout = graphs[gk]
                for r in range(2):
                    for t in gout:
                        if t.dtype == torch.float32:
                            t.fill_(-1.0)
                        else:
                            t.view(torch.uint8).fill_(0xFF)
                    g.replay()
                    torch.cuda.synchronize()
                    compare(f"{tag} replay{r}", refs[keep_v], gout, fails, check_v=keep_v)
                    counts["replay"] += 1
            for t, p0 in zip(list(inp) + [tab, idx], pristine):
                if not torch.equal(t, p0):
                    fails.append(f"r{rnd} M{m} c{int(cond)}: input mutated")
        log(f"round {rnd}: fails {len(fails)} counts {counts}")
    return {"passed": not fails, "fails": fails[:30], "counts": counts,
            "cases": [f"{m}{'c' if c else ''}" for m, c in data],
            "compared": "compact-table fused QKV op (qkv8 q/k/v bytes, scales[8], kept v plane) "
                        "bitwise vs the levers-off route (full cos_sin table, landing 22 kernel) "
                        "and that route vs main's two-pass chain (gemm_qkv3 + qknorm_rope_fp8); "
                        "eager x3 vs call0, custom-op impl, ROPE_COMPACT=0 fallback with tensors "
                        "passed, CUDA-graph replays, keep_v both ways, 3 rounds alternating M "
                        "order, inputs/tables unchanged; forced verification mismatch builds no "
                        "table; tab[idx] rebuilds cos_sin bit for bit"}


def _poison(outs):
    for t in outs:
        if t.dtype == torch.float32:
            t.fill_(float("nan"))
        else:
            t.view(torch.uint8).fill_(0xFF)  # NaN in E4M3 and BF16


def repeat(_ms):
    """Untimed served-order repeat sequence: per round t2i (16464 single, 16384 image mode) then
    i2i (32848 single, 16384 image_cond, 16384 image mode), each a CUDA-graph replay of the
    compact-table op (keep_v False and True) plus one eager call, pristine inputs restored and
    outputs NaN-poisoned before every call; bitwise vs round 0 and vs the levers-off arm (full
    cos_sin table, landing 22 kernel). B_ROUNDS rounds (default 50)."""
    epi, strided, dev, has = setup()
    assert has
    rounds = int(os.environ.get("B_ROUNDS", "50"))
    seq = [("t2i", 16464, False), ("t2i", 16384, False), ("i2i", 32848, False),
           ("i2i", 16384, True), ("i2i", 16384, False)]
    fails, n_cmp = [], 0
    cases = {}
    for _, m, cond in seq:
        if (m, cond) in cases:
            continue
        inp = inputs_for(m, dev, cond)
        tab, idx = epi.build_rope_compact(inp[7])
        live = list(inp) + [tab, idx]
        pristine = [t.clone() for t in live]
        a, w, sfa, sfb, alpha, qn, kn, cs, _ = inp
        per_kv = {}
        for keep_v in (False, True):
            ref = _snap(epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v, tab, idx)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v, tab, idx)
            per_kv[keep_v] = {"ref": ref, "g": g, "out": gout, "r0": None, "e0": None}
        cases[(m, cond)] = (live, pristine, per_kv)
    log(f"captured {len(cases)} cases x 2 keep_v; running {rounds} rounds")
    for rnd in range(rounds):
        for mode, m, cond in seq:
            live, pristine, per_kv = cases[(m, cond)]
            for t, p in zip(live, pristine):
                t.copy_(p)
            a, w, sfa, sfb, alpha, qn, kn, cs, _, tab, idx = live
            for keep_v, st in per_kv.items():
                tag = f"r{rnd} {mode} M{m}{'c' if cond else ''} kv{int(keep_v)}"
                _poison(st["out"])
                st["g"].replay()
                got_e = epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v, tab, idx)
                torch.cuda.synchronize()
                compare(f"{tag} replay vs levers-off", st["ref"], st["out"], fails, keep_v)
                compare(f"{tag} eager vs levers-off", st["ref"], got_e, fails, keep_v)
                if st["r0"] is None:
                    st["r0"], st["e0"] = _snap(st["out"]), _snap(got_e)
                else:
                    compare(f"{tag} replay vs round 0", st["r0"], st["out"], fails, keep_v)
                    compare(f"{tag} eager vs round 0", st["e0"], got_e, fails, keep_v)
                n_cmp += 4
                del got_e
            for t, p in zip(live, pristine):
                if not torch.equal(t, p):
                    fails.append(f"r{rnd} M{m}: input/table mutated")
        if rnd % 10 == 0 or rnd == rounds - 1:
            log(f"round {rnd}: fails {len(fails)} comparisons {n_cmp}")
    return {"passed": not fails, "fails": fails[:30], "rounds": rounds, "comparisons": n_cmp,
            "sequence": [f"{mo}:{m}{'c' if c else ''}" for mo, m, c in seq],
            "compared": "compact-table fused QKV op, served order t2i->i2i per round, CUDA-graph "
                        "replay + eager per case and keep_v, pristine inputs restored, outputs "
                        "NaN-poisoned; qkv8/scales/kept v bitwise vs round 0 and vs the "
                        "levers-off arm (full cos_sin, landing 22 kernel)"}


def kill(ms):
    """TRTLLM_FLUX3_QKV_EPI_FP8=0: impl given compact tensors still runs main's chain."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
        flux3_sm120_qknorm_rope_fp8 as qr8,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    epi, strided, dev, has = setup()
    fails = []
    if epi.enabled() or os.environ.get("TRTLLM_FLUX3_QKV_EPI_FP8") != "0":
        fails.append("kill switch not active")
    if epi.build_rope_compact(make_cos_sin(16464, dev, CKPT)) is not None:
        fails.append("compact table built with the route off")
    calls = {"n": 0}
    orig = epi.qkv_rope_fp8

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    epi.qkv_rope_fp8 = counted
    epi.ENABLED = True  # build a table anyway, to pass it in
    for m in ms:
        inp = inputs_for(m, dev)
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = inp
        tab, idx = epi.build_rope_compact(cs)
        epi.ENABLED = False
        for keep_v in (True, False):
            ref = _snap(main_chain(strided, qr8, inp, True))
            got = fp.nvfp4_qkv_rope_fp8_impl(a, w, sfa, sfb, qn, kn, cs, pos, alpha, SM, keep_v,
                                             True, tab, idx)
            torch.cuda.synchronize()
            compare(f"M{m} kv{int(keep_v)} killswitch", ref, got, fails, check_v=keep_v)
        epi.ENABLED = True
    epi.ENABLED = False
    if calls["n"]:
        fails.append(f"fused route called {calls['n']} times")
    return {"passed": not fails, "fails": fails, "Ms": ms,
            "compared": "TRTLLM_FLUX3_QKV_EPI_FP8=0: nvfp4_qkv_rope_fp8_impl given compact "
                        "tensors == main's gemm_qkv3 + qknorm_rope_fp8 chain bitwise; fused route "
                        "never called; build_rope_compact returns None"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pair"
    ms = [int(x) for x in os.environ.get("B_MS", "16384,16464,32848").split(",")]
    fn = {"pair": pair, "diag": diag, "check": check_matrix, "kill": kill,
          "repeat": repeat}[mode]
    print(json.dumps(fn(ms)), flush=True)


if __name__ == "__main__":
    main()
