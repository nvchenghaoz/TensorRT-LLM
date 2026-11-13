"""Stage 0 for idea 68: exhaustive-ish cuBLASLt BF16 config sweep at MiniMax-H3 TP2 per-rank shapes.

For each DiT Linear shape (QKV at full M; out / gate_up / down at the chunked-overlap chunk M) this
 1. times F.linear (the main path) and unpinned trtllm::cublas_mm,
 2. enumerates cuBLASLt configs: the heuristic list (128 requested), then every algo id x tile x
    stages (split-K 1 only: split-K changes the K summation, never bitwise) validated with
    cublasLtMatmulAlgoCheck, then one-at-a-time swizzle / custom option / cluster shape variations
    around the fastest bitwise configs, then their small joint product,
 3. quick-times every valid config once and checks it bitwise against F.linear on the same inputs,
 4. re-times F.linear, cublas_mm and the best bitwise configs interleaved (rounds x iters, medians),
 5. re-checks the finalists bitwise on randn / realistic / outlier inputs at two Ms per shape, and
    runs an untimed repeat (same warmup + iters) comparing every output to F.linear.
Synthetic tensors only. Last stdout line is a JSON summary.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt import Lt  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
OUT_DIR = os.environ.get("AUTO_DEV_ARTIFACTS", os.environ.get("ASCENT_ARTIFACTS", "/tmp"))
SM = None

# name: (N, K, activation kind, [primary M, second M])
SHAPES = {
    "qkv": (10752, 5376, "act", [37966, 37838]),
    "out": (5376, 3584, "act", [9600, 9166]),
    "gate_up": (14336, 5376, "act", [9600, 9166]),
    "down": (5376, 7168, "swiglu", [9600, 9166]),
}


def log(*a):
    print(*a, flush=True)


def make_act(M, K, kind, dist, gen):
    if dist == "randn":
        return torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    if kind == "swiglu":
        a = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32) * 1.5
        b = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32) * 1.5
        x = F.silu(a) * b
        del a, b
    else:
        x = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32)
        col_scale = torch.exp(0.5 * torch.randn(K, generator=gen, device="cuda"))
        col_shift = 0.3 * torch.randn(K, generator=gen, device="cuda")
        x = x * col_scale + col_shift
    if dist == "outlier":
        idx = torch.randperm(K, generator=gen, device="cuda")[:8]
        x[:, idx] *= 50.0
    return x.to(torch.bfloat16).contiguous()


def make_weight(N, K, gen):
    w = torch.randn(N, K, generator=gen, device="cuda", dtype=torch.float32) / (K ** 0.5)
    w = w * torch.exp(0.3 * torch.randn(N, 1, generator=gen, device="cuda"))
    return w.to(torch.bfloat16).contiguous()


def ev_time(fn, n):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in zip(s, e)]


def stats(ts):
    ts = sorted(ts)
    med = statistics.median(ts)
    p10 = ts[max(0, int(0.1 * (len(ts) - 1)))]
    p90 = ts[min(len(ts) - 1, int(round(0.9 * (len(ts) - 1))))]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "spread": (p90 - p10) / med, "n": len(ts)}


def key_of(cfg):
    return tuple(cfg.get(k) for k in ("ID", "TILE_ID", "STAGES_ID", "SPLITK_NUM", "REDUCTION_SCHEME",
                                      "CTA_SWIZZLING", "CUSTOM_OPTION", "CLUSTER_SHAPE_ID",
                                      "INNER_SHAPE_ID"))


def run_shape(name, lt, args, summary):
    N, K, kind, Ms = SHAPES[name]
    M = Ms[0]
    flop = 2.0 * M * N * K
    gen = torch.Generator(device="cuda")
    gen.manual_seed(4242 + N + K)
    w = make_weight(N, K, gen)
    x = make_act(M, K, kind, "realistic", gen)
    ref = F.linear(x, w)
    y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    prob = lt.problem(M, N, K)
    t0 = time.time()

    base = stats(ev_time(lambda: F.linear(x, w), 3) and ev_time(lambda: F.linear(x, w), 8))
    base_ms = base["median_ms"]
    log(f"[{name}] M={M} N={N} K={K}: F.linear quick {base_ms:.4f} ms "
        f"({flop / base_ms / 1e9:.1f} TFLOPS)")

    seen = {}
    counts = {"heuristic_returned": 0, "enum_checked": 0, "enum_valid": 0, "enum_skipped_small_tile": 0,
              "neigh_checked": 0, "neigh_valid": 0, "timed": 0, "bitwise": 0, "slow_skipped": 0,
              "errors": 0, "timed_faster_quick": 0, "bitwise_faster_quick": 0}

    def evaluate(cfg, source):
        k = key_of(cfg)
        if k in seen:
            return seen[k]
        a = lt.make(cfg)
        rec = {"cfg": cfg, "desc": lt.describe(cfg), "source": source}
        seen[k] = rec
        if a is None:
            rec["error"] = "make"
            counts["errors"] += 1
            return rec
        chk = lt.check(prob, a)
        if chk is None:
            rec["error"] = "check"
            return rec
        rec.update(chk)
        fn = lt.matmul_fn(prob, a, x, w, y)
        try:
            fn()  # untimed: lazy module load of a new kernel
            y.fill_(float("nan"))
            t1 = ev_time(fn, 1)[0]
            rec["bitwise"] = bool(torch.equal(y, ref))
            if t1 > args.slow_factor * base_ms:
                rec["slow_ms"] = t1
                counts["slow_skipped"] += 1
                return rec
            ts = ev_time(fn, args.quick_iters)
            rec["quick_ms"] = statistics.median(ts)
            counts["timed"] += 1
            if rec["bitwise"]:
                counts["bitwise"] += 1
            if rec["quick_ms"] < base_ms:
                counts["timed_faster_quick"] += 1
                if rec["bitwise"]:
                    counts["bitwise_faster_quick"] += 1
        except Exception as e:  # noqa
            rec["error"] = str(e)[:200]
            counts["errors"] += 1
            torch.cuda.synchronize()
        return rec

    # 1. heuristic list
    heur = lt.heuristics(prob, 128)
    counts["heuristic_returned"] = len(heur)
    heur_recs = []
    for i, h in enumerate(heur):
        cfg = lt.get_cfg(h.algo)
        rec = evaluate(cfg, f"heur{i}")
        rec["heur_idx"] = i
        heur_recs.append(rec)
    log(f"[{name}] heuristics: {len(heur)}; #0 {heur_recs[0]['desc'] if heur_recs else None} "
        f"quick {heur_recs[0].get('quick_ms') if heur_recs else None} bitwise "
        f"{heur_recs[0].get('bitwise') if heur_recs else None}")
    for r in heur_recs[:12]:
        log(f"   h{r['heur_idx']:3d} {r['desc']} waves {r.get('waves')} quick {r.get('quick_ms')} "
            f"bitwise {r.get('bitwise')} {r.get('error', '')}")

    # 2. algo id x tile x stages (split-K 1)
    ids = lt.algo_ids()
    caps = {}
    seed_defaults = {}
    for r in heur_recs:
        c = r["cfg"]
        seed_defaults.setdefault(c["ID"], []).append(c)
    for aid in ids:
        a = lt.init(aid)
        if a is None:
            continue
        tiles = lt.cap_array(a, "TILE_IDS") or [0]
        stages = lt.cap_array(a, "STAGES_IDS") or [0]
        cmax = lt.cap_scalar(a, "CUSTOM_OPTION_MAX")
        swz = lt.cap_scalar(a, "CTA_SWIZZLING_SUPPORT")
        caps[aid] = {"tiles": len(tiles), "stages": len(stages), "custom_max": cmax, "swizzle": swz}
        templates = [{"SPLITK_NUM": 1, "REDUCTION_SCHEME": 0, "CTA_SWIZZLING": 0, "CUSTOM_OPTION": 0,
                      "CLUSTER_SHAPE_ID": 0, "INNER_SHAPE_ID": 0}]
        for c in seed_defaults.get(aid, [])[:2]:
            templates.append({k: c[k] for k in ("CTA_SWIZZLING", "CUSTOM_OPTION", "CLUSTER_SHAPE_ID",
                                                "INNER_SHAPE_ID")} | {"SPLITK_NUM": 1,
                                                                      "REDUCTION_SCHEME": 0})
        joint_custom = (cmax or 0) > 0 and len(tiles) > 1
        for ti, t in enumerate(tiles):
            if ti % 50 == 49:
                log(f"[{name}] algo {aid}: tile {ti + 1}/{len(tiles)} checked {counts['enum_checked']} "
                    f"valid {counts['enum_valid']} timed {counts['timed']} ({time.time() - t0:.0f}s)")
            dims = lt.tile_dims(t)
            if t != 0 and dims is not None and dims[0] * dims[1] < args.min_tile_area:
                counts["enum_skipped_small_tile"] += len(stages) * len(templates)
                continue
            if joint_custom:
                # nvjet-style algos: the custom option selects the kernel variant for a tile, so
                # enumerate tile x stages x custom jointly (cluster 1x1x1 first, then AUTO).
                for s in stages:
                    for cu in range(0, min(cmax, args.max_custom) + 1):
                        for cg in (2, 0):
                            cfg = {"ID": aid, "TILE_ID": t, "STAGES_ID": s, "SPLITK_NUM": 1,
                                   "REDUCTION_SCHEME": 0, "CTA_SWIZZLING": 0, "CUSTOM_OPTION": cu,
                                   "CLUSTER_SHAPE_ID": cg, "INNER_SHAPE_ID": 0}
                            if key_of(cfg) in seen:
                                break
                            counts["enum_checked"] += 1
                            al = lt.make(cfg)
                            if al is None or lt.check(prob, al) is None:
                                continue
                            counts["enum_valid"] += 1
                            evaluate(cfg, "enum")
                            break
                continue
            for s in stages:
                for tpl in templates:
                    cfg = dict(tpl, ID=aid, TILE_ID=t, STAGES_ID=s)
                    if key_of(cfg) in seen:
                        continue
                    counts["enum_checked"] += 1
                    al = lt.make(cfg)
                    if al is None or lt.check(prob, al) is None:
                        seen[key_of(cfg)] = {"cfg": cfg, "error": "check", "source": "enum"}
                        continue
                    counts["enum_valid"] += 1
                    evaluate(cfg, "enum")
        log(f"[{name}] algo {aid}: caps {caps[aid]}; checked so far {counts['enum_checked']} "
            f"valid {counts['enum_valid']} timed {counts['timed']} bitwise {counts['bitwise']} "
            f"({time.time() - t0:.0f}s)")

    def ranked_bitwise():
        return sorted([r for r in seen.values() if r.get("bitwise") and "quick_ms" in r],
                      key=lambda r: r["quick_ms"])

    # 3. neighbourhood of the fastest bitwise configs
    for seed in ranked_bitwise()[:args.seeds]:
        aid = seed["cfg"]["ID"]
        a = lt.init(aid)
        cmax = caps.get(aid, {}).get("custom_max") or 0
        swz = caps.get(aid, {}).get("swizzle") or 0
        variations = []
        for v in range(0, min(cmax, args.max_custom) + 1):
            variations.append(("CUSTOM_OPTION", v))
        for v in sorted(lt.clusters) or list(range(0, 40)):
            variations.append(("CLUSTER_SHAPE_ID", v))
        for v in range(0, 2 if swz else 1):
            variations.append(("CTA_SWIZZLING", v))
        for attr, v in variations:
            cfg = dict(seed["cfg"])
            cfg[attr] = v
            if key_of(cfg) in seen:
                continue
            counts["neigh_checked"] += 1
            al = lt.make(cfg)
            if al is None or lt.check(prob, al) is None:
                seen[key_of(cfg)] = {"cfg": cfg, "error": "check", "source": "neigh"}
                continue
            counts["neigh_valid"] += 1
            evaluate(cfg, "neigh")
        log(f"[{name}] neighbourhood of {seed['desc']}: checked {counts['neigh_checked']} valid "
            f"{counts['neigh_valid']} ({time.time() - t0:.0f}s)")

    # 4. joint product of the best per-attribute values among the top bitwise configs
    top = ranked_bitwise()[:args.seeds]
    vals = {k: [] for k in ("CUSTOM_OPTION", "CLUSTER_SHAPE_ID", "CTA_SWIZZLING")}
    for r in top:
        for k in vals:
            if r["cfg"][k] not in vals[k]:
                vals[k].append(r["cfg"][k])
    for r in top[:3]:
        for cu in vals["CUSTOM_OPTION"][:3]:
            for cg in vals["CLUSTER_SHAPE_ID"][:3]:
                for sw in vals["CTA_SWIZZLING"][:2]:
                    cfg = dict(r["cfg"], CUSTOM_OPTION=cu, CLUSTER_SHAPE_ID=cg, CTA_SWIZZLING=sw)
                    if key_of(cfg) in seen:
                        continue
                    counts["neigh_checked"] += 1
                    al = lt.make(cfg)
                    if al is None or lt.check(prob, al) is None:
                        seen[key_of(cfg)] = {"cfg": cfg, "error": "check", "source": "joint"}
                        continue
                    counts["neigh_valid"] += 1
                    evaluate(cfg, "joint")

    allrank = sorted([r for r in seen.values() if "quick_ms" in r], key=lambda r: r["quick_ms"])
    log(f"[{name}] sweep done: {counts} ({time.time() - t0:.0f}s)")
    log(f"[{name}] fastest 8 (any numerics):")
    for r in allrank[:8]:
        log(f"   {r['quick_ms']:.4f} bitwise {r['bitwise']} waves {r.get('waves'):.2f} {r['desc']}")
    rb = ranked_bitwise()
    log(f"[{name}] fastest 8 bitwise:")
    for r in rb[:8]:
        log(f"   {r['quick_ms']:.4f} waves {r.get('waves'):.2f} {r['desc']} ({r['source']})")

    # 5. careful interleaved timing
    finalists = []
    if heur_recs and "quick_ms" in heur_recs[0]:
        finalists.append(("lt_heur0", heur_recs[0]))
    for i, r in enumerate(rb[:args.finalists]):
        if heur_recs and r is heur_recs[0]:
            continue
        finalists.append((f"bitwise{i}", r))
    arms = [("F.linear", lambda: F.linear(x, w)),
            ("cublas_mm", lambda: torch.ops.trtllm.cublas_mm(x, w.t(), None, None))]
    for lab, r in finalists:
        arms.append((lab, lt.matmul_fn(prob, lt.make(r["cfg"]), x, w, y)))
    tims = {lab: [] for lab, _ in arms}
    for _ in range(2):
        for lab, fn in arms:
            fn()
    torch.cuda.synchronize()
    for rnd in range(args.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for lab, fn in order:
            fn()
            tims[lab] += ev_time(fn, args.iters)
    table = {}
    for lab, _ in arms:
        st = stats(tims[lab])
        st["tflops"] = flop / st["median_ms"] / 1e9
        table[lab] = st
    for lab, r in finalists:
        table[lab]["desc"] = r["desc"]
        table[lab]["cfg"] = r["cfg"]
        table[lab]["waves"] = r.get("waves")
        if r.get("waves") is not None:
            table[lab]["last_wave_fill"] = r["waves"] - math.floor(r["waves"]) or 1.0
    fl = table["F.linear"]["median_ms"]
    for lab in table:
        table[lab]["vs_flinear"] = table[lab]["median_ms"] / fl - 1.0
        log(f"[{name}] {lab:10s} {table[lab]['median_ms']:.4f} ms spread {table[lab]['spread']:.4f} "
            f"{table[lab]['tflops']:.1f} TFLOPS vs F.linear {100 * table[lab]['vs_flinear']:+.2f}% "
            f"{table[lab].get('desc', '')} waves {table[lab].get('waves')}")

    # 6. bitwise on 3 distributions x 2 Ms + untimed repeat, for every finalist faster than F.linear
    verify = {}
    for lab, r in finalists:
        verify[lab] = {"passed": True, "cases": []}
    for Mv in Ms:
        probv = lt.problem(Mv, N, K)
        yv = torch.empty(Mv, N, dtype=torch.bfloat16, device="cuda")
        for dist in ("randn", "realistic", "outlier"):
            xv = make_act(Mv, K, kind, dist, gen)
            refv = F.linear(xv, w)
            xp = xv.clone()
            for lab, r in finalists:
                fn = lt.matmul_fn(probv, lt.make(r["cfg"]), xv, w, yv)
                bad = 0
                nrep = (2 + args.iters) if dist == "realistic" else 1
                for _ in range(nrep):
                    yv.fill_(float("nan"))
                    fn()
                    if not torch.equal(yv, refv):
                        bad += 1
                if not torch.equal(xv, xp):
                    bad += 1000
                verify[lab]["cases"].append({"M": Mv, "dist": dist, "runs": nrep, "mismatches": bad})
                if bad:
                    verify[lab]["passed"] = False
            del xv, refv, xp
        del yv
    for lab in verify:
        log(f"[{name}] verify {lab}: passed {verify[lab]['passed']} "
            f"{[(c['M'], c['dist'], c['mismatches']) for c in verify[lab]['cases']]}")

    # kernel names (torch profiler) for F.linear and the finalists
    names = {}
    try:
        from torch.profiler import ProfilerActivity, profile
        for lab, fn in arms:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            names[lab] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
            log(f"[{name}] kernel {lab}: {names[lab]}")
    except Exception as e:  # noqa
        log("profiler failed", e)

    out = {"M": M, "Ms": Ms, "N": N, "K": K, "flop": flop, "counts": counts, "caps": caps,
           "table": table, "verify": verify, "kernels": names,
           "top_any": [{"quick_ms": r["quick_ms"], "bitwise": r["bitwise"], "waves": r.get("waves"),
                        "desc": r["desc"]} for r in allrank[:20]],
           "top_bitwise": [{"quick_ms": r["quick_ms"], "waves": r.get("waves"), "desc": r["desc"],
                            "cfg": r["cfg"]} for r in rb[:20]]}
    summary[name] = out
    with open(os.path.join(OUT_DIR, f"sweep_{name}.json"), "w") as f:
        json.dump({"shape": out, "all": [dict(r, cfg=r["cfg"]) for r in seen.values()]}, f, default=str)
    del x, y, ref, w
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="out,down,gate_up,qkv")
    ap.add_argument("--quick-iters", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--finalists", type=int, default=4)
    ap.add_argument("--max-custom", type=int, default=255)
    ap.add_argument("--min-tile-area", type=int, default=64 * 128)
    ap.add_argument("--slow-factor", type=float, default=1.6)
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401  (registers trtllm ops)
    torch.cuda.init()
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    F.linear(a, a)
    torch.cuda.synchronize()
    lt = Lt()
    log(f"cublasLt libs loaded: {lt.libs}; version {lt.version}; headers {lt.headers[:3]}; "
        f"enum mismatches {getattr(lt, 'enum_mismatch', None)}; tiles {len(lt.tiles)} stages "
        f"{len(lt.stages)} clusters {len(lt.clusters)}")
    log(f"device {torch.cuda.get_device_name()} cap {torch.cuda.get_device_capability()} torch "
        f"{torch.__version__} blas backend {torch.backends.cuda.preferred_blas_library()}")
    summary = {}
    for name in args.shapes.split(","):
        run_shape(name, lt, args, summary)
    brief = {}
    for name, s in summary.items():
        t = s["table"]
        best = min((lab for lab in t if lab.startswith("bitwise") and s["verify"].get(lab, {}).get("passed")),
                   key=lambda lab: t[lab]["median_ms"], default=None)
        brief[name] = {"M": s["M"], "flinear_ms": t["F.linear"]["median_ms"],
                       "flinear_spread": t["F.linear"]["spread"],
                       "cublas_mm_ms": t["cublas_mm"]["median_ms"],
                       "best_bitwise": best, "best_ms": t[best]["median_ms"] if best else None,
                       "best_vs_flinear": t[best]["vs_flinear"] if best else None,
                       "best_desc": t[best].get("desc") if best else None,
                       "counts": s["counts"]}
    print(json.dumps({"stage0": brief}))


if __name__ == "__main__":
    main()
