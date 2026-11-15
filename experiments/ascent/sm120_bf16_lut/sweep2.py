"""Stage-1b dense sweep + table generation for the SM120 BF16 cuBLASLt range table.

1. Pool = union of cuBLASLt heuristic results (32 per query) over every grid M and every shape.
2. For each shape and grid M (step 64, 128..8192 + protocol M), time every valid pooled config and F.linear
   (F.linear timed before and after the configs; the faster of the two is the reference).
   Cold L2: 4 rotated weight copies per shape (>= 704 MB, > 2x the 128 MiB L2). CUDA events, warmup 3, 12 timed.
3. Table: grid interval (G[i-1], G[i]] gets the config minimizing max(t_cfg/t_F.linear) over both endpoints.
   Configs using the in-place (atomic) split-K reduction are excluded. Consecutive equal configs are merged.
4. Emulated validation at off-grid M: the table's config vs F.linear, interleaved.
Prints the C++ table between @@TABLE_BEGIN/@@TABLE_END.

usage: python sweep2.py <checkpoint> [step]
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
T0 = time.time()
CFG_COMMENT = "{algo, tile, stages, numsK, reduction, swizzle, custom, cga}"


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


def build_table(grid, tim, pool):
    """tim[M] = {"fl": t, "c": {ci: t}} -> list of (m_max, ci, score)."""
    entries = []
    for i, M in enumerate(grid):
        ends = [grid[i - 1], M] if i > 0 else [M]
        best = None
        for ci in tim[M]["c"]:
            if pool[ci][4] == 1:  # in-place split-K reduction: excluded (non-deterministic)
                continue
            if not all(ci in tim[e]["c"] for e in ends):
                continue
            score = max(tim[e]["c"][ci] / tim[e]["fl"] for e in ends)
            if best is None or score < best[1]:
                best = (ci, score)
        entries.append((M, best[0], best[1]))
    merged = []
    for M, ci, sc in entries:
        if merged and merged[-1][1] == ci:
            merged[-1] = (M, ci, max(merged[-1][2], sc))
        else:
            merged.append((M, ci, sc))
    return merged


def lookup(table, m):
    for m_max, ci, _ in table:
        if m <= m_max:
            return ci
    return None


def main():
    ckpt = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    shapes, info = read_shapes(ckpt)
    if len(sys.argv) > 3:
        keep = sys.argv[3].split(",")
        shapes = {k: v for k, v in shapes.items() if k in keep}
    log("config", json.dumps(info), "shapes", json.dumps(shapes))
    lt = Lt()
    log("cublasLt", lt.path, lt.version, torch.cuda.get_device_name())
    grid = sorted(set(list(range(128, 8193, step)) + [300, 1000, 1200, 2000, 4000, 4800, 8000, 8100]))
    pool = set()
    for sname, (N, K) in shapes.items():
        for M in grid:
            for (cfg8, inner), wsz, waves, orig in lt.heuristics(M, N, K, 32):
                pool.add(cfg8)
    pool = sorted(pool)
    log("pool size", len(pool))
    for ci, c in enumerate(pool):
        log(f"  pool[{ci}]", lt.name(c), list(c))
    algos = [lt.make_algo(c) for c in pool]
    tables = {}
    summary = {}
    for sname, (N, K) in shapes.items():
        torch.manual_seed(0)
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NCOPY)]
        xmax = torch.randn(max(grid), K, device="cuda", dtype=torch.bfloat16)
        obuf = torch.empty(max(grid), N, device="cuda", dtype=torch.bfloat16)
        tim = {}
        for M in grid:
            x = xmax[:M]
            out = obuf[:M]
            t_fl0 = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 3, 12)
            row = {}
            for ci in range(len(pool)):
                if not lt.check(M, N, K, algos[ci]):
                    continue
                row[ci] = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, algos[ci]), 3, 12)
            t_fl1 = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 3, 12)
            tim[M] = {"fl": min(t_fl0, t_fl1), "c": row}
            bi = min(row, key=row.get)
            log(f"{sname} M={M}: F.linear {tim[M]['fl']:.4f} ({t_fl0:.4f}/{t_fl1:.4f}) best {row[bi]:.4f} "
                f"[{bi}] {lt.name(pool[bi])} {(tim[M]['fl'] / row[bi] - 1) * 100:+.1f}%")
        table = build_table(grid, tim, pool)
        tables[sname] = table
        log(f"{sname} table ({len(table)} entries):")
        for m_max, ci, sc in table:
            log(f"   m<={m_max}: [{ci}] {lt.name(pool[ci])} worst ratio {sc:.3f}")
        # ---- emulated validation at off-grid M (table config vs F.linear, interleaved, warm 5 / 20 timed)
        random.seed(hash(sname) & 0xFFFF)
        offgrid = sorted(set([129, 301, 999, 1001, 1199, 1201, 1999, 2001, 3999, 4001, 4100, 4500, 4799, 4801,
                              7040, 7500, 7999, 8050, 8101, 8191] + random.sample(range(130, 8192), 24)))
        worst = (0.0, None)
        gains = []
        for M in offgrid:
            x = xmax[:M]
            out = obuf[:M]
            ci = lookup(table, M)
            a = algos[ci]
            assert lt.check(M, N, K, a), (sname, M, ci)
            fl0 = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 5, 20)
            tc0 = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, a), 5, 20)
            fl1 = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 5, 20)
            tc1 = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, a), 5, 20)
            fl, tc = min(fl0, fl1), min(tc0, tc1)
            r = tc / fl
            gains.append(1 - r)
            worst = max(worst, (r, M))
            log(f"  validate {sname} M={M}: F.linear {fl:.4f} table[{ci}] {tc:.4f} ratio {r:.3f}")
        summary[sname] = {"entries": len(table), "worst_offgrid_ratio": round(worst[0], 4), "worst_M": worst[1],
                          "mean_offgrid_gain": round(sum(gains) / len(gains), 4)}
        log(f"{sname} validation summary {summary[sname]}")
        del Ws, xmax, obuf
        torch.cuda.empty_cache()
    # ---- C++ table
    lines = ["inline const RangeAlgoListType sm120_bf16_range_algo_list = {"]
    for sname, (N, K) in shapes.items():
        lines.append(f"    // {sname}: N={N} K={K}; {{m_max, {CFG_COMMENT}}}")
        lines.append(f"    {{{{{K}, {N}}},")
        lines.append("        {")
        lines.append("            {127, {-1, 0, 0, 0, 0, 0, 0, 0}},")
        for m_max, ci, sc in tables[sname]:
            c = pool[ci]
            lines.append(f"            {{{m_max}, {{{', '.join(str(v) for v in c)}}}}}, // {lt.name(c)}")
        lines.append("        }},")
    lines.append("};")
    print("@@TABLE_BEGIN", flush=True)
    print("\n".join(lines), flush=True)
    print("@@TABLE_END", flush=True)
    print(json.dumps({"done": True, "pool": len(pool), "grid": len(grid), "summary": summary}))


if __name__ == "__main__":
    main()
