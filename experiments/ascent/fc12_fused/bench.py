"""Idea 201 timing: the decode-MoE leaf chain (route_assign -> q0 -> GEMM kernel(s)) per layer.

BENCH_MODE = leaf (default: Sm12xW4a8DecodeState.run, i.e. whatever this tree's production
leaf does; used by run_gpu_pair so BASE = main's fc1+fc2, CAND = the fused kernel), or
base / base_fc1 / fused (chains of this tree, for A/B diagnostics in one process).
BENCH_SHAPE = ep4 (96 x 2304, the primary layout) or ep2tp2 (192 x 1152).

BENCH_LAYERS (default 8) layers with DISTINCT synthetic weights run back to back in one CUDA
graph, as consecutive MoE layers do; their active weights (D x 18.8 MB per layer at EP4) do
not fit in L2 together, so every layer streams cold weights.  A 512 MB READ-ONLY flush runs
before every replay (pitfall 1356).  Kineto per node: span = first route_assign start -> last
chain kernel end; per-layer time = span / layers; p50 over BENCH_REPLAYS replays; noise =
(p90 - p10) / p50 of that one case.  Each layer routes its own expert set (same D).
Cases: per-D rows (forced D distinct local experts: M=1 D 0..6, M=4 D 0..12) and the
uniform top-6 busiest-rank mix at M = 1, 2, 4 (BENCH_SEEDS routings, rank 0 = busiest).
Last stdout line: {"batch": {"1"|"2"|"4": {"kernel_ms", "noise_floor"}}, "rows": ..., "output"}.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common12 import H, SHAPE, Runner, capture, log, make_state, routing, uniform_routing  # noqa: E402

MODE = os.environ.get("BENCH_MODE", "leaf")
REPLAYS = int(os.environ.get("BENCH_REPLAYS", "30"))
SEEDS = int(os.environ.get("BENCH_SEEDS", "6"))
LAYERS = int(os.environ.get("BENCH_LAYERS", "8"))
FLUSH = None


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def trace(g):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        FLUSH.sum()
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPLAYS):
            FLUSH.sum()
            g.replay()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"b201_{os.getpid()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    evs = sorted((e for e in tr.get("traceEvents", [])
                  if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")),
                 key=lambda e: float(e["ts"]))
    reps, cur = [], None
    for e in evs:
        if "reduce_kernel" in e["name"] and float(e["dur"]) > 100.0:
            if cur:
                reps.append(cur)
            cur = []
        elif cur is not None:
            cur.append(e)
    if cur:
        reps.append(cur)
    spans, names = [], set()
    for r in reps:
        st_i = next((i for i, e in enumerate(r) if "route_assign" in e["name"]), None)
        if st_i is None:
            continue
        # The chain's own nodes only: the next flush's cast/memset also land in this group.
        r = [e for e in r[st_i:] if "at::native" not in e["name"] and "Memset" not in e["name"]]
        t0 = float(r[0]["ts"])
        spans.append((max(float(e["ts"]) + float(e["dur"]) for e in r) - t0) / LAYERS)
        for e in r:
            names.add(e["name"][:48])
    p50 = pct(spans, 0.5)
    return {"layer_us": round(p50, 3), "noise": round((pct(spans, 0.9) - pct(spans, 0.1)) / p50, 4),
            "n": len(spans), "kernels": sorted(names)}


def main():
    global FLUSH
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    layers = []
    for li in range(LAYERS):
        leaf, st = make_state(seed=li)
        st.warmup(6)
        layers.append((st, None if MODE == "leaf" else Runner(leaf, st)))
    torch.cuda.synchronize()
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    log(f"BENCH arm={'tree leaf (BASE=main fc1+fc2 / CAND=fused per tree)' if MODE == 'leaf' else MODE} "
        f"shape={SHAPE} layers={LAYERS} replays={REPLAYS}")

    def chain(x, routes):
        out = None
        for (st, runner), (ids, w) in zip(layers, routes):
            out = st.run(x, ids, w) if runner is None else runner(x, ids, w, MODE)[2]
        return out

    rows = []
    kernels = set()
    for m, dset in ((1, range(0, 7)), (4, range(0, 13))):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        for d in dset:
            routes = [routing(m, d, seed=li) for li in range(LAYERS)]
            g = capture(lambda: chain(x, routes))
            r = trace(g)
            del g
            kernels |= set(r["kernels"])
            rows.append({"case": f"M{m}_D{d}", "layer_us": r["layer_us"], "noise": r["noise"]})
            log(f"[{MODE}] {SHAPE} M{m} D{d}: {r['layer_us']} us/layer noise {r['noise']} n {r['n']}")
    log(f"[{MODE}] kernels: {sorted(kernels)}")
    batch = {}
    for m in ((1, 2, 4) if os.environ.get("BENCH_UNIFORM", "1") == "1" else ()):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        sp, nz, ds = [], [], []
        for s in range(SEEDS):
            rr = [uniform_routing(m, 1000 * s + li) for li in range(LAYERS)]
            routes = [(a, b) for a, b, _ in rr]
            dl = [c for _, _, c in rr]
            g = capture(lambda: chain(x, routes))
            r = trace(g)
            del g
            sp.append(r["layer_us"])
            nz.append(r["noise"])
            ds.append(sum(dl) / len(dl))
            rows.append({"case": f"uni_M{m}_s{s}", "meanD": round(ds[-1], 2),
                         "layer_us": r["layer_us"], "noise": r["noise"]})
        mean = sum(sp) / len(sp)
        batch[str(m)] = {"kernel_ms": round(mean / 1000.0, 6), "noise_floor": max(nz)}
        log(f"[{MODE}] {SHAPE} uniform M{m}: mean D {[round(v, 2) for v in ds]} us/layer {sp} "
            f"-> mean {mean:.3f}, max noise {max(nz)}")
    print(json.dumps({"batch": batch, "rows": rows, "mode": MODE, "shape": SHAPE,
                      "layers": LAYERS, "output": {"class": "reordered"}}))


if __name__ == "__main__":
    main()
