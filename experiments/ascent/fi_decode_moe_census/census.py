"""Analysis idea 134: per-node census of the landed FlashInfer SM12x decode-MoE chain.

One EP4 rank of DeepSeek-V4.1-Flash (96 local experts, top-6, hidden 5120, inter 2304),
the production leaf ``Sm12xW4a8DecodeState.run`` captured in a CUDA graph, cold L2 (512 MB
write before every replay).  For M = 1/2/4/8 and forced distinct-local-expert counts D,
plus the routing of experiments/ascent/w4a8_fi_chain/bench.py (pair 537/538) and a uniform
random top-6 routing mapped so the busiest rank is rank 0:

  * CUDA-event time of the whole graph replay (ground truth), p10/p50/p90;
  * a Kineto trace of the same graph: every node (name, grid, block), its duration,
    its gap after the previous node's end, and its exposed time;
  * upper-bound probes (deliberately skipping work, timing only): route+fc1+fc2 without
    the prologue elementwise nodes and the zero memset; fc1 alone; fc2 alone.

Usage: census.py [--ncu]   (--ncu: run M in {1,4} x D in {3,6} once each, between
cudaProfilerStart/Stop, for profile_kernels capture=region).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w4a8_fi_chain"))

import torch  # noqa: E402

from common import (E, E_GLOBAL, H, I, TOPK, log, make_routing, make_weights,  # noqa: E402
                    trtllm_layout)

FC1_BYTES = 2 * I * H // 2 + 2 * I * H // 32  # 11.80 MB codes + 0.74 MB ue8m0
FC2_BYTES = H * I // 2 + H * I // 32
PEAK = 1507.17e9
ITERS = int(os.environ.get("CENSUS_ITERS", "60"))
TRACE_REPLAYS = 12


def forced_routing(m, d, seed):
    """Exactly d distinct local experts (rank 0) over m tokens; other picks off-rank."""
    g = torch.Generator(device="cpu")
    g.manual_seed(7000 + 31 * m + d + seed)
    assert d <= TOPK * m
    experts = torch.randperm(E, generator=g)[:d].tolist()
    per_tok = [[] for _ in range(m)]
    for i, e in enumerate(experts):
        per_tok[i % m].append(e)
    if d < m:  # tokens without a local pick reuse the set (pairs = max(d, m))
        for t in range(d, m):
            per_tok[t].append(experts[t % d])
    ids = torch.empty(m, TOPK, dtype=torch.int32)
    for t in range(m):
        loc = per_tok[t]
        rem = (torch.randperm(E_GLOBAL - E, generator=g)[: TOPK - len(loc)] + E).tolist()
        row = torch.tensor(loc + rem)
        ids[t] = row[torch.randperm(TOPK, generator=g)].to(torch.int32)
    w = torch.rand(m, TOPK, generator=g) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return ids.cuda(), w.cuda()


def uniform_busiest(m, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(9000 + 17 * m + seed)
    ids = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK] for _ in range(m)]).to(torch.int32)
    counts = [len(set(ids[(ids // E) == r].tolist())) for r in range(4)]
    r = max(range(4), key=lambda i: counts[i])
    ids = (ids - r * E) % E_GLOBAL
    w = torch.rand(m, TOPK, generator=g) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return ids.to(torch.int32).cuda(), w.cuda(), counts


def distinct_local(ids):
    v = ids[(ids >= 0) & (ids < E)]
    return int(torch.unique(v).numel()), int(v.numel())


def capture(fn, args):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(*args)
    torch.cuda.synchronize()
    return g


FLUSH = None


def flush():
    FLUSH.zero_()


def time_g(g, iters=ITERS, warmup=10):
    ts = []
    for i in range(warmup + iters):
        flush()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warmup:
            ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    n = len(ts)
    med = ts[n // 2]
    return {"p50_us": round(med, 2), "p10_us": round(ts[n // 10], 2),
            "p90_us": round(ts[(9 * n) // 10], 2),
            "noise": round((ts[(9 * n) // 10] - ts[n // 10]) / med, 4)}


def short(name):
    n = name
    for key, lab in (("route_assign", "route_assign_decode"), ("q0_route_direct", "q0_route_direct"),
                     ("Fc1", "fc1_act_q1"), ("fc1", "fc1_act_q1"), ("Fc2", "fc2_finalize"),
                     ("fc2", "fc2_finalize"), ("Finalize", "fc2_finalize")):
        if key in n:
            return lab
    if "FillFunctor" in n or "fill" in n.lower():
        return "fill"
    if "where" in n.lower():
        return "ew_where"
    for key in ("CompareGE", "CompareGT", "CompareLT", "CompareLE", "ge_", "lt_", "BitwiseAnd",
                "bitwise_and", "AUnaryFunctor", "BUnaryFunctor", "BinaryFunctor", "sub", "add"):
        if key in n:
            return "ew:" + key
    return n[:60]


def trace_graph(g, tag):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(TRACE_REPLAYS):
            flush()
            g.replay()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"census_{os.getpid()}_{tag}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    evs = [e for e in tr.get("traceEvents", [])
           if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    evs.sort(key=lambda e: float(e["ts"]))
    replays, cur = [], None
    for e in evs:
        grid = (e.get("args", {}).get("grid") or [0])[0]
        if "FillFunctor<int>" in e["name"] and float(e["dur"]) > 100.0 and grid > 1000:  # 512 MB flush
            if cur is not None:
                replays.append(cur)
            cur = {"flush_end": float(e["ts"]) + float(e["dur"]), "nodes": []}
        elif cur is not None:
            cur["nodes"].append(e)
    if cur is not None:
        replays.append(cur)
    replays = [r for r in replays if r["nodes"]][1:]  # drop the first
    if not replays:
        return None
    nn = min(len(r["nodes"]) for r in replays)
    cols = []
    for j in range(nn):
        durs, gaps, exps = [], [], []
        for r in replays:
            nodes = r["nodes"]
            e = nodes[j]
            st, du = float(e["ts"]), float(e["dur"])
            prev_end = max([float(x["ts"]) + float(x["dur"]) for x in nodes[:j]] or [r["flush_end"]])
            durs.append(du)
            gaps.append(st - prev_end)
            exps.append(max(0.0, st + du - max(st, prev_end)))
        durs.sort(); gaps.sort(); exps.sort()
        e0 = replays[0]["nodes"][j]
        a = e0.get("args", {})
        cols.append({"node": short(e0["name"]), "grid": a.get("grid"), "block": a.get("block"),
                     "dur_us": round(durs[len(durs) // 2], 2),
                     "gap_us": round(gaps[len(gaps) // 2], 2),
                     "exposed_us": round(exps[len(exps) // 2], 2),
                     "name": e0["name"][:140]})
    spans = sorted(float(r["nodes"][nn - 1]["ts"]) + float(r["nodes"][nn - 1]["dur"]) - r["flush_end"]
                   for r in replays)
    return {"n_nodes": nn, "node_counts": sorted({len(r["nodes"]) for r in replays}),
            "span_from_flush_end_us": round(spans[len(spans) // 2], 2), "nodes": cols}


class Probes:
    """Timing-only upper-bound probes (skip prologue/memset; single kernels)."""

    def __init__(self, st):
        from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
        self.leaf, self.st = leaf, st
        self.cache = {}

    def prep(self, x, ids, w):
        route, _, fc1, out_sf_shape, fc2, fi_act = self.leaf._load_ops()
        st = self.st
        local = st.local_ids(ids)
        pairs = local.numel()
        ws = self.leaf._workspace(x, local, st.num_experts)
        q1 = torch.zeros(pairs, st.inter, dtype=torch.float8_e4m3fn, device=x.device)
        sf1 = torch.zeros(out_sf_shape(pairs, st.inter, st.num_experts), dtype=torch.int32, device=x.device)
        out = torch.zeros(x.shape[0], st.hidden, dtype=torch.bfloat16, device=x.device)
        offsets, token_map, rw, a_q, a_sf = route(x, local, w, st.num_experts, workspace=ws,
                                                  enable_pdl=st.enable_pdl)
        torch.cuda.synchronize()
        return dict(route=route, fc1=fc1, fc2=fc2, act=fi_act.Swiglu, local=local, ws=ws, q1=q1,
                    sf1=sf1, out=out, offsets=offsets, token_map=token_map, rw=rw, a_q=a_q, a_sf=a_sf)

    def no_prologue(self, x, ids, w, p):
        st = self.st
        offsets, token_map, rw, a_q, a_sf = p["route"](x, p["local"], w, st.num_experts,
                                                       workspace=p["ws"], enable_pdl=st.enable_pdl)
        p["fc1"](a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=st.enable_pdl,
                 out_q=p["q1"], out_sf=p["sf1"], activation=p["act"], swiglu_limit=st.swiglu_limit)
        p["fc2"](p["q1"], p["sf1"], st.w2, st.w2_sf, offsets, token_map, rw, x.shape[0], tune=False,
                 enable_pdl=st.enable_pdl, out=p["out"])

    def fc1_only(self, x, ids, w, p):
        st = self.st
        p["fc1"](p["a_q"], p["a_sf"], st.w1, st.w1_sf, p["offsets"], tune=False,
                 enable_pdl=st.enable_pdl, out_q=p["q1"], out_sf=p["sf1"], activation=p["act"],
                 swiglu_limit=st.swiglu_limit)

    def fc2_only(self, x, ids, w, p):
        st = self.st
        p["fc2"](p["q1"], p["sf1"], st.w2, st.w2_sf, p["offsets"], p["token_map"], p["rw"],
                 x.shape[0], tune=False, enable_pdl=st.enable_pdl, out=p["out"])


def main():
    global FLUSH
    ncu = "--ncu" in sys.argv
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import (
        Sm12xW4a8DecodeState)

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    st = Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=pdl)
    log("state built, pdl", pdl)
    FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    if ncu:
        cfgs = [(m, d) for m in (1, 4) for d in (3, 6)]
        runs = []
        for m, d in cfgs:
            x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
            ids, w = forced_routing(m, d, 0)
            for _ in range(3):
                st.run(x, ids, w)
            runs.append((m, d, x, ids, w))
        torch.cuda.synchronize()
        log("ncu order: " + ", ".join(f"M{m}/D{d}" for m, d, *_ in runs))
        torch.cuda.profiler.start()
        for m, d, x, ids, w in runs:
            flush()
            st.run(x, ids, w)
            torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print(json.dumps({"ncu_order": [f"M{m}/D{d}" for m, d, *_ in runs]}))
        return

    probes = Probes(st)
    results = []
    for m in (1, 2, 4, 8):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        dset = tuple(range(1, 13)) if m == 4 else (1, 2, 3, 6, 12)
        if os.environ.get("CENSUS_QUICK"):
            dset = (1, 2, 3, 6, 12)
        cases = [(f"D{d}", *forced_routing(m, d, 0)) for d in dset if d <= TOPK * m]
        ids_b, w_b = make_routing(m, {1: 3, 2: 2, 4: 2, 8: 2}[m], seed=m)
        cases.append(("bench537", ids_b, w_b))
        ids_u, w_u, cnt = uniform_busiest(m, 0)
        cases.append(("uniform_busiest", ids_u, w_u))
        for label, ids, w in cases:
            d, pairs = distinct_local(ids)
            g = capture(st.run, (x, ids, w))
            t = time_g(g)
            tr = trace_graph(g, f"m{m}_{label}")
            p = probes.prep(x, ids, w)
            pr = {}
            for nm, fn in (("route_fc1_fc2_no_prologue", probes.no_prologue),
                           ("fc1_only", probes.fc1_only), ("fc2_only", probes.fc2_only)):
                gp = capture(lambda a, b, c, fn=fn: fn(a, b, c, p), (x, ids, w))
                pr[nm] = time_g(gp, iters=40)
                del gp
            floor = d * (FC1_BYTES + FC2_BYTES) / PEAK * 1e6
            rec = {"M": m, "routing": label, "E_local": d, "local_pairs": pairs,
                   "chain_event": t, "weight_floor_us": round(floor, 2), "probes": pr, "trace": tr}
            if tr:
                for nd in tr["nodes"]:
                    if nd["node"] == "fc1_act_q1":
                        nd["dram_pct_weights"] = round(100 * d * FC1_BYTES / (nd["dur_us"] * 1e-6) / PEAK, 1)
                    if nd["node"] == "fc2_finalize":
                        nd["dram_pct_weights"] = round(100 * d * FC2_BYTES / (nd["dur_us"] * 1e-6) / PEAK, 1)
            results.append(rec)
            nodes_s = " | ".join(f"{n['node']} {n['dur_us']}/{n['gap_us']}/{n['exposed_us']}"
                                 for n in (tr["nodes"] if tr else []))
            log(f"M={m} {label} D={d} pairs={pairs} chain p50 {t['p50_us']} us (noise {t['noise']}) "
                f"floor {floor:.1f} | noprol {pr['route_fc1_fc2_no_prologue']['p50_us']} "
                f"fc1 {pr['fc1_only']['p50_us']} fc2 {pr['fc2_only']['p50_us']} || nodes dur/gap/exp: "
                + nodes_s)
            del g
    first = next((r for r in results if r["trace"]), None)
    if first:
        log("node names (M=%d %s):" % (first["M"], first["routing"]))
        for nd in first["trace"]["nodes"]:
            log("  ", nd["node"], nd["grid"], nd["block"], nd["name"])
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    with open(os.path.join(out_dir, "census.json"), "w") as f:
        json.dump(results, f, indent=1)
    summ = [{"M": r["M"], "routing": r["routing"], "D": r["E_local"],
             "chain_us": r["chain_event"]["p50_us"], "noise": r["chain_event"]["noise"],
             "floor_us": r["weight_floor_us"],
             "noprol_us": r["probes"]["route_fc1_fc2_no_prologue"]["p50_us"],
             "fc1_us": r["probes"]["fc1_only"]["p50_us"], "fc2_us": r["probes"]["fc2_only"]["p50_us"]}
            for r in results]
    print(json.dumps({"summary": summ}))


if __name__ == "__main__":
    main()
