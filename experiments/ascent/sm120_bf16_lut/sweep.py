"""Stage-1 sweep: F.linear vs cuBLASLt heuristic vs every heuristic-returned algo, Gemma4 attention projections.

usage: python sweep.py <checkpoint> [phaseA_only]
Cold L2: 4 rotated weight copies per shape (>= 704 MB). CUDA events, warmup 5, timed 20 (phase A).
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt import Lt, read_shapes  # noqa: E402

PROTO_M = [300, 512, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
DENSE = sorted(set(list(range(256, 8193, 256)) + PROTO_M + [320, 352, 1152, 4097, 7936]))
NCOPY = 4
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
    n = len(t)
    return dict(med=t[n // 2], p10=t[int(n * 0.1)], p90=t[min(n - 1, int(n * 0.9))])


def main():
    ckpt = sys.argv[1]
    phase_a_only = len(sys.argv) > 2 and sys.argv[2] == "a"
    shapes, info = read_shapes(ckpt)
    log("config", json.dumps(info))
    log("shapes (N,K)", json.dumps(shapes))
    lt = Lt()
    log("cublasLt", lt.path, "version", lt.version, "hdr", lt.hdr, "tiles parsed", len(lt.tiles))
    log("gpu", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    results = {}
    for sname, (N, K) in shapes.items():
        torch.manual_seed(0)
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NCOPY)]
        xmax = torch.randn(max(DENSE), K, device="cuda", dtype=torch.bfloat16)
        obuf = torch.empty(max(DENSE), N, device="cuda", dtype=torch.bfloat16)
        res = {"N": N, "K": K, "A": {}, "B": {}}
        cand_scores = {}
        # ---------------- phase A: all heuristic algos at protocol M
        for M in PROTO_M:
            x = xmax[:M]
            out = obuf[:M]
            tl = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 5, 20)
            th = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out), 5, 20)
            ref_null = out.clone()
            heur = lt.heuristics(M, N, K, 32)
            row = {"flinear": tl, "lt_heur": th, "heur0": lt.name(heur[0][0][0]) if heur else None, "algos": []}
            seen = set()
            for (cfg8, inner), wsz, waves, orig in heur:
                if cfg8 in seen:
                    continue
                seen.add(cfg8)
                algo = lt.make_algo(cfg8)
                if not lt.check(M, N, K, algo):
                    row["algos"].append({"cfg": cfg8, "name": lt.name(cfg8), "ok": False})
                    continue
                t = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, algo), 5, 20)
                # compare reconstructed vs heuristic struct on the same weight copy
                lt.matmul(x, Ws[0], out, orig)
                o_orig = out.clone()
                lt.matmul(x, Ws[0], out, algo)
                same = bool(torch.equal(o_orig, out))
                row["algos"].append({"cfg": cfg8, "inner": inner, "name": lt.name(cfg8), "ok": True, "t": t,
                                     "ws": wsz, "waves": round(waves, 3), "recon_bitwise": same})
            row["algos"].sort(key=lambda a: a.get("t", {}).get("med", 1e9))
            best = row["algos"][0] if row["algos"] else None
            if best:
                bt = best["t"]["med"]
                for a in row["algos"]:
                    if a.get("ok") and a["t"]["med"] <= bt * 1.02:
                        cand_scores[a["cfg"]] = cand_scores.get(a["cfg"], 0) + 1
                gain = tl["med"] / bt - 1
                log(f"{sname} M={M}: F.linear {tl['med']:.4f} lt_heur {th['med']:.4f} best {bt:.4f} "
                    f"({best['name']}, recon_bitwise={best['recon_bitwise']}) gain_vs_flinear {gain * 100:+.1f}% "
                    f"heur0={row['heur0']} n_algos={len(row['algos'])}")
                for a in row["algos"][:6]:
                    if a.get("ok"):
                        log(f"    {a['t']['med']:.4f} [{a['t']['p10']:.4f},{a['t']['p90']:.4f}] {a['name']} "
                            f"inner={a['inner']} ws={a['ws']} waves={a['waves']} cfg={list(a['cfg'])}")
            res["A"][M] = row
        # ---------------- phase B: shortlisted configs over a dense M grid
        if not phase_a_only:
            short = sorted(cand_scores, key=lambda c: -cand_scores[c])[:12]
            log(f"{sname} shortlist", [lt.name(c) for c in short])
            for M in DENSE:
                x = xmax[:M]
                out = obuf[:M]
                tl = timeit(lambda i: F.linear(x, Ws[i % NCOPY]), 3, 12)
                th = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out), 3, 12)
                row = {"flinear": tl["med"], "lt_heur": th["med"], "c": {}}
                for c in short:
                    algo = lt.make_algo(c)
                    if not lt.check(M, N, K, algo):
                        continue
                    row["c"][lt.name(c)] = timeit(lambda i: lt.matmul(x, Ws[i % NCOPY], out, algo), 3, 12)["med"]
                bn, bt = min(row["c"].items(), key=lambda kv: kv[1]) if row["c"] else (None, 1e9)
                log(f"{sname} dense M={M}: F.linear {tl['med']:.4f} lt_heur {th['med']:.4f} best {bt:.4f} {bn} "
                    f"{(tl['med'] / bt - 1) * 100:+.1f}%")
                res["B"][M] = row
            res["shortlist"] = [[lt.name(c), list(c)] for c in short]
        results[sname] = res
        del Ws, xmax, obuf
        torch.cuda.empty_cache()
    # ---------------- ceiling 8192^3
    N = K = M = 8192
    Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tl = timeit(lambda i: F.linear(x, Ws[i % 4]), 5, 20)
    best = (1e9, None)
    for (cfg8, inner), wsz, waves, orig in lt.heuristics(M, N, K, 32):
        algo = lt.make_algo(cfg8)
        if not lt.check(M, N, K, algo):
            continue
        t = timeit(lambda i: lt.matmul(x, Ws[i % 4], out, algo), 5, 20)["med"]
        best = min(best, (t, lt.name(cfg8)))
    fl = 2 * M * N * K
    log(f"CEILING 8192^3: F.linear {tl['med']:.4f} ms = {fl / tl['med'] / 1e9:.1f} TFLOPS; best algo {best[0]:.4f} "
        f"ms = {fl / best[0] / 1e9:.1f} TFLOPS ({best[1]})")
    results["ceiling"] = {"flinear_ms": tl["med"], "best_ms": best[0], "best": best[1]}
    path = os.environ.get("SWEEP_OUT", "/tmp/sweep_results.json")
    with open(path, "w") as f:
        json.dump(results, f, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    log("wrote", path)
    summ = {s: {M: round(r["A"][M]["flinear"]["med"] / r["A"][M]["algos"][0]["t"]["med"] - 1, 4)
                for M in r["A"]} for s, r in results.items() if s != "ceiling"}
    print(json.dumps({"gain_best_vs_flinear": summ, "ceiling": results["ceiling"]}))


if __name__ == "__main__":
    main()
