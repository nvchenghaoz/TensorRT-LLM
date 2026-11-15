"""Table generator v3 for the SM120 BF16 cuBLASLt range table (Gemma-4-31B attention projections).

Served token counts: the frozen benchmark sends exact-length prompts (ISL 300/1k/2k/4k/8k) and packs up to
max_num_tokens=8192, so forward M values are S = {300k} U {1000k} U {2000k} U {4000k} U {8000} (<= 8192).
Table per shape (K, N):
  * exact entry at every served M in S: the best config at that M, re-validated against F.linear with
    interleaved CUDA-event timing (warmup 5, 20 timed, x2) among the sweep's top 4; kept only if it beats
    F.linear by >= 1% (else the M is not routed, i.e. stays F.linear, or below 1280 / heuristic above);
  * M in [1280, 8192] between served points: 64-row grid intervals (G[i-1], G[i]] get the config minimizing
    max(t_cfg / t_F.linear) over both endpoints; if that worst ratio > 1.0 the interval keeps the cuBLASLt heuristic;
  * below 1280 only the exact served M are routed (Python token gate); everything else stays on F.linear.
Pool: union of cuBLASLt heuristic lists (32 per query) over all shapes and grid M; pruned by a pre-pass at S to
configs within 3% of the best at any served M (plus each grid point's heuristic top-1). In-place (atomic) split-K
reductions are excluded. Cold L2: 4 rotated weight copies (>= 704 MB per shape), CUDA events.
Prints @@TABLE (C++) and @@ROUTES (Python token gate) blocks and an off-grid validation.

usage: python sweep3.py <checkpoint>
"""
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt import Lt, read_shapes  # noqa: E402

NCOPY = 4
LO = 1280
HI = 8192
EXACT_THR = 0.97  # exact served-M entry kept only if >= 3% faster than F.linear (interleaved re-validation)
INTERVAL_THR = 0.99  # interval pinned only if its monotonicity bound (below) is >= 1% under F.linear
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def timeit(fn, warm, iters):
    for i in range(warm):
        fn(i)
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for i, (s, e) in enumerate(evs):
        s.record()
        fn(warm + i)
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in evs)
    return t[len(t) // 2]


def served():
    s = set()
    for isl in (300, 1000, 2000, 4000, 8000):
        k = 1
        while isl * k <= HI:
            s.add(isl * k)
            k += 1
    return sorted(s)


def main():
    ckpt = sys.argv[1]
    shapes, info = read_shapes(ckpt)
    log("config", json.dumps(info), "shapes", json.dumps(shapes))
    lt = Lt()
    log("cublasLt", lt.path, lt.version, torch.cuda.get_device_name())
    S = served()
    grid = sorted(set(list(range(LO, HI + 1, 64)) + [m for m in S if m >= LO]))
    log("served M", S)
    pool = set()
    top1 = {}
    for sname, (N, K) in shapes.items():
        for M in sorted(set(grid + S)):
            h = lt.heuristics(M, N, K, 32)
            for (cfg8, inner), wsz, waves, orig in h:
                # nvjet only, no in-place (atomic) split-K reduction; the bitwise gate below decides the rest
                if cfg8[0] in (66, 67) and cfg8[4] != 1:
                    pool.add(cfg8)
            if h and h[0][0][0][0] in (66, 67) and h[0][0][0][4] != 1:
                top1.setdefault(sname, set()).add(h[0][0][0])
    pool = sorted(pool)
    algos = {c: lt.make_algo(c) for c in pool}
    log("pool size", len(pool))
    tables, routes, summary = {}, {}, {}
    for sname, (N, K) in shapes.items():
        torch.manual_seed(0)
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NCOPY)]
        xmax = torch.randn(HI, K, device="cuda", dtype=torch.bfloat16)
        obuf = torch.empty(HI, N, device="cuda", dtype=torch.bfloat16)

        nonbitwise = {}

        def mismatches(M, c, j=0):
            """Number of output elements whose raw BF16 bits differ from main's F.linear output at M."""
            x, out = xmax[:M], obuf[:M]
            ref = F.linear(x, Ws[j])
            lt.matmul(x, Ws[j], out, algos[c])
            return int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())

        def bitwise(M, c):
            """Config c gives exactly main's F.linear bits at M (checked on two weight copies)."""
            return all(mismatches(M, c, j) == 0 for j in (0, 1))

        # ---- bitwise study for the fact entry: main's kernel vs every pooled config at selected M
        study_ms = [300, 1000, 1100, 1152, 1200, 4800, 7040, 7500, 7999, 8000, 8050, 8150, 8192]
        try:
            from torch.profiler import ProfilerActivity, profile

            def kname(fn):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                return ";".join(sorted({e.name[:90] for e in prof.events() if e.device_type.name == "CUDA"}))
        except Exception:  # noqa: BLE001
            def kname(fn):
                return "?"
        for M in study_ms:
            x, out = xmax[:M], obuf[:M]
            main_k = kname(lambda: F.linear(x, Ws[0]))
            n_el = M * N
            rows = []
            for c in pool:
                if not lt.check(M, N, K, algos[c]):
                    continue
                mm = mismatches(M, c)
                rows.append((mm, c, kname(lambda: lt.matmul(x, Ws[0], out, algos[c]))))
            n_bw = sum(1 for r in rows if r[0] == 0)
            log(f"STUDY {sname} M={M} main={main_k} : {n_bw}/{len(rows)} configs bitwise")
            for mm, c, kn in rows:
                grp = "b" if c[3] != 1 or c[4] != 0 else "a"
                log(f"STUDY   {sname} M={M} cfg={list(c)} {lt.name(c)} group={grp} kernel={kn} "
                    f"bitwise={'YES' if mm == 0 else 'NO'} mismatched={mm}/{n_el}")

        def measure(M, cfgs, warm=3, iters=12):
            x, out = xmax[:M], obuf[:M]
            fl0 = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), warm, iters)
            row = {}
            for c in cfgs:
                if c in algos and lt.check(M, N, K, algos[c]):
                    if not bitwise(M, c):
                        nonbitwise[(M, c)] = True
                        continue
                    row[c] = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, algos[c]), warm, iters)
            fl1 = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), warm, iters)
            return min(fl0, fl1), row

        # ---- pre-pass at served M with the whole pool -> shortlist
        pre = {}
        short = set(top1.get(sname, set()))
        for M in S:
            fl, row = measure(M, pool)
            pre[M] = (fl, row)
            if not row:
                log(f"{sname} pre M={M}: F.linear {fl:.4f}; no bitwise config")
                continue
            b = min(row.values())
            short |= {c for c, t in row.items() if t <= 1.03 * b}
            log(f"{sname} pre M={M}: F.linear {fl:.4f} best {b:.4f} {(fl / b - 1) * 100:+.1f}% "
                f"{lt.name(min(row, key=row.get))}")
        short = sorted(short)
        log(f"{sname} shortlist {len(short)}")
        # ---- exact served entries: re-validate top-4 interleaved
        exact = {}
        for M in S:
            fl, row = pre[M]
            cands = sorted(row, key=row.get)[:4]
            x, out = xmax[:M], obuf[:M]
            best = None
            res = {}
            for rep in range(2):
                f = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 5, 20)
                res.setdefault("fl", []).append(f)
                for c in cands:
                    res.setdefault(c, []).append(timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, algos[c]), 5, 20))
            flv = min(res["fl"])
            for c in cands:
                r = min(res[c]) / flv
                if best is None or r < best[1]:
                    best = (c, r)
            if best is None:
                best = (None, 9.0)
            exact[M] = best
            log(f"{sname} exact M={M}: F.linear {flv:.4f} best {lt.name(best[0]) if best[0] else '-'} "
                f"ratio {best[1]:.3f}")
        # ---- dense intervals for M >= LO
        tim = {}
        for M in grid:
            if M in pre:
                fl, row = pre[M]
                row = {c: t for c, t in row.items() if c in short}
            else:
                fl, row = measure(M, short)
            tim[M] = (fl, row)
        intervals = []
        for i, M in enumerate(grid):
            # Bound for every m in (G[i-1], G[i]]: a fixed kernel is no faster at m than at G[i], and F.linear
            # is no faster at m than at G[i-1], so ratio(m) <= t_cfg(G[i]) / t_F.linear(G[i-1]).
            prev = grid[i - 1] if i > 0 else M
            best = None
            for c in short:
                if c not in tim[M][1]:
                    continue
                if i > 0 and c not in tim[prev][1]:
                    continue
                sc = max(tim[M][1][c] / tim[prev][0], tim[M][1][c] / tim[M][0])
                if best is None or sc < best[1]:
                    best = (c, sc)
            if best is None:
                best = (None, 9.0)
            intervals.append((M, best[0] if best[1] <= INTERVAL_THR else None, best[1]))
            log(f"{sname} interval (.., {M}]: {lt.name(best[0]) if best[0] else '-'} worst ratio {best[1]:.3f}"
                f"{'' if best[1] <= INTERVAL_THR else ' -> F.linear'}")
        # ---- assemble entries {m_max: cfg or None}
        ents = []
        small_routed = []
        for M in [m for m in S if m < LO]:
            c, r = exact[M]
            ents.append((M - 1, None))
            if r <= EXACT_THR:
                ents.append((M, c))
                small_routed.append(M)
            else:
                ents.append((M, None))
        ents.append((LO - 1, None))
        for M, c, sc in intervals:
            if M in exact:
                ce, re_ = exact[M]
                ents.append((M - 1, c))
                ents.append((M, ce if re_ <= EXACT_THR else c))
            else:
                ents.append((M, c))
        # drop empty / duplicate m_max, merge consecutive equal configs
        clean = []
        for m, c in ents:
            if clean and m <= clean[-1][0]:
                continue
            if clean and clean[-1][1] == c:
                clean[-1] = (m, c)
            else:
                clean.append((m, c))
        tables[sname] = clean
        # Python token gate: exactly the pinned ranges; everything else (heuristic entries included) stays F.linear
        ranges = []
        prev = 0
        for m, c in clean:
            if c is not None:
                if ranges and ranges[-1][1] == prev:
                    ranges[-1] = (ranges[-1][0], m)
                else:
                    ranges.append((prev + 1, m))
            prev = m
        routes[sname] = ranges

        def lookup(m):
            if not any(lo <= m <= hi for lo, hi in ranges):
                return "F.linear"
            for mm, c in clean:
                if m <= mm:
                    return c
            return "F.linear"

        log(f"{sname} table {len(clean)} entries; small routed {small_routed}")
        for m, c in clean:
            log(f"   m<={m}: {lt.name(c) if c else 'heuristic'}")
        # ---- off-grid + served validation, interleaved, warm 5 / 20 timed
        random.seed(17)
        vals = sorted(set(S + [m + 1 for m in S if m < HI] + [g + 1 for g in grid if g < HI]
                          + [1281, 1500, 4100, 4500, 7040, 7500, 7999, 8050, 8191] + random.sample(range(LO, HI), 24)))
        worst = (0.0, None)
        gains = []
        bw_fail = []
        for M in vals:
            c = lookup(M)
            if c == "F.linear":
                continue
            x, out = xmax[:M], obuf[:M]
            a = algos[c] if c is not None else None
            fl = min(timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 5, 20) for _ in range(2))
            tc = min(timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, a), 5, 20) for _ in range(2))
            bw = c is not None and bitwise(M, c)
            if not bw:
                bw_fail.append(M)
            r = tc / fl
            gains.append(1 - r)
            worst = max(worst, (r, M))
            log(f"  validate {sname} M={M}{' (served)' if M in S else ''}: F.linear {fl:.4f} "
                f"{lt.name(c) if c else 'heuristic'} {tc:.4f} ratio {r:.3f} bitwise={bw}")
        log(f"{sname} configs rejected as non-bitwise: {len(nonbitwise)}; routed M not bitwise: {bw_fail}")
        summary[sname] = {"entries": len(clean), "worst_ratio": round(worst[0], 4), "worst_M": worst[1],
                          "nonbitwise_rejected": len(nonbitwise), "routed_not_bitwise": bw_fail,
                          "mean_gain": round(sum(gains) / max(len(gains), 1), 4), "small_routed": small_routed,
                          "routed_ranges": len(ranges)}
        log(f"{sname} summary {summary[sname]}")
        del Ws, xmax, obuf
        torch.cuda.empty_cache()
    lines = ["inline const RangeAlgoListType sm120_bf16_range_algo_list = {"]
    for sname, (N, K) in shapes.items():
        lines.append(f"    // {sname}: N={N} K={K}")
        lines.append(f"    {{{{{K}, {N}}},")
        lines.append("        {")
        for m, c in tables[sname]:
            if c is None:
                lines.append(f"            {{{m}, {{-1, 0, 0, 0, 0, 0, 0, 0}}}}, // cuBLASLt heuristic")
            else:
                lines.append(f"            {{{m}, {{{', '.join(str(v) for v in c)}}}}}, // {lt.name(c)}")
        lines.append("        }},")
    lines.append("};")
    print("@@TABLE_BEGIN", flush=True)
    print("\n".join(lines), flush=True)
    print("@@TABLE_END", flush=True)
    print("@@ROUTES_BEGIN", flush=True)
    for sname, (N, K) in shapes.items():
        rs = ", ".join(f"({lo}, {hi})" for lo, hi in routes[sname])
        print(f"    # {sname}\n    ({K}, {N}): ({rs}),", flush=True)
    print("@@ROUTES_END", flush=True)
    print(json.dumps({"done": True, "pool": len(pool), "summary": summary}))


if __name__ == "__main__":
    main()
