"""Idea 201 Stage 0: how much of the FI SM12x decode-MoE chain a persistent fc1+fc2 kernel can recover.

One rank (BENCH_SHAPE=ep4: 96 x 2304; ep2tp2: 192 x 1152), synthetic MXFP4 weights, the
production leaf's call sequence (route_assign_decode -> q0_route_direct -> fc1_act_q1 ->
fc2_finalize, PDL on) captured in a CUDA graph, 512 MB READ-ONLY flush before every replay
(pitfall 1356), Kineto per-node timing (CUDA events tick in ~2 us steps here).

Per D (distinct local experts, 0..12) and M (1, 4), four graphs:
  A  chain        : production order (the BASE).
  B  no_fc2       : route + q0 + fc1 only.             A - B = exposed fc2 cost.
  C  concurrent   : fc2 forked onto a 2nd stream right after the route, so it does not wait
                    for fc1 at all (timing only, numerics deliberately wrong).  The fused kernel
                    keeps the fc1 -> fc2 dependency, so A - C is a hard UPPER BOUND on what it
                    can recover (fc1 tail + fc2 ramp + boundary + any weight-stream overlap).
  R  fc2_l2warm   : A with the active experts' fc2 weights read into L2 before the route
                    (outside the span): A - R bounds what streaming fc2 weights during fc1's
                    tail could buy if it were free.
Spans are measured from the route_assign node's start to the last fc1/fc2 node's end.
Also: Monte-Carlo busiest-rank D distribution (uniform top-6 of 384) per layout and M, and
the D-weighted recoverable cost at that mix.
"""

import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))

import torch  # noqa: E402

import common as _common  # noqa: E402

SHAPE = os.environ.get("BENCH_SHAPE", "ep4")
if SHAPE == "ep2tp2":
    _common.E, _common.I = 192, 1152

from common import E, E_GLOBAL, H, I, TOPK, log, make_weights, trtllm_layout  # noqa: E402

REPLAYS = int(os.environ.get("BENCH_REPLAYS", "30"))
FC1_BYTES = 2 * I * H // 2 + 2 * I * H // 32
FC2_BYTES = H * I // 2 + H * I // 32
PEAK = 1508e9
FLUSH = None


def routing(m, d, seed=0):
    """Exactly d distinct local experts in [0, E) over m tokens (d=0: an empty rank)."""
    g = random.Random(7000 + 31 * m + d + seed)
    assert d <= TOPK * m
    experts = g.sample(range(E), d)
    per_tok = [[] for _ in range(m)]
    for i, e in enumerate(experts):
        per_tok[i % m].append(e)
    if 0 < d < m:
        for t in range(d, m):
            per_tok[t].append(experts[t % d])
    ids = []
    for t in range(m):
        loc = per_tok[t]
        rem = [E + v for v in g.sample(range(E_GLOBAL - E), TOPK - len(loc))]
        row = loc + rem
        g.shuffle(row)
        ids.append(row)
    w = torch.rand(m, TOPK, generator=torch.Generator().manual_seed(d + 100 * m)) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return torch.tensor(ids, dtype=torch.int32).cuda(), w.cuda(), experts


def d_distribution(e_local, m, trials=20000):
    g = random.Random(5)
    ranks = E_GLOBAL // e_local
    hist = {}
    for _ in range(trials):
        sets = [set() for _ in range(ranks)]
        for _t in range(m):
            for e in g.sample(range(E_GLOBAL), TOPK):
                sets[e // e_local].add(e)
        d = max(len(s) for s in sets)
        hist[d] = hist.get(d, 0) + 1
    return {k: round(v / trials, 4) for k, v in sorted(hist.items())}


class Chain:
    def __init__(self, st, leaf):
        self.st, self.leaf = st, leaf
        self.side = torch.cuda.Stream()

    def __call__(self, x, ids, w, mode, warm=()):
        st, leaf = self.st, self.leaf
        route, _, fc1, out_sf_shape, fc2, fi_act = leaf._load_ops()
        m = x.shape[0]
        pairs = ids.numel()
        ws = leaf._workspace(x, ids, st.num_experts)
        sf1_shape = out_sf_shape(pairs, st.inter, st.num_experts)
        n_q1 = pairs * st.inter
        n_sf1 = sf1_shape[0] * sf1_shape[1] * 4
        off_sf1 = leaf._align(n_q1, 256)
        off_out = leaf._align(off_sf1 + n_sf1, 256)
        buf = torch.empty(off_out + m * st.hidden * 2, dtype=torch.uint8, device=x.device)
        q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, st.inter)
        sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
        out = buf[off_out:].view(torch.bfloat16).view(m, st.hidden)
        for e in warm:  # read-only L2 warm of fc2 weights (before the timed span)
            st.w2[e].view(torch.int32).sum()
            st.w2_sf[e].view(torch.int32).sum()
        offsets, token_map, rw, a_q, a_sf = route(
            x, ids, w, st.num_experts, workspace=ws, enable_pdl=st.enable_pdl,
            expert_offset=st.slot_start, zero_buf=buf)

        def run_fc2():
            fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m, tune=False,
                enable_pdl=st.enable_pdl, out=out)

        if mode == "concurrent":
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                run_fc2()
        fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=st.enable_pdl,
            out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=st.swiglu_limit)
        if mode in ("chain", "fc2_l2warm"):
            run_fc2()
        if mode == "concurrent":
            torch.cuda.current_stream().wait_stream(self.side)
        return out


def capture(fn):
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
    torch.cuda.synchronize()
    return g


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def trace(g):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        FLUSH.sum()
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPLAYS):
            FLUSH.sum()
            g.replay()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"s0_{os.getpid()}.json")
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
    rows = []
    for r in reps:
        st_i = next((i for i, e in enumerate(r) if "route_assign" in e["name"]), None)
        if st_i is None:
            continue
        r = r[st_i:]
        gem = [e for e in r if "fc1" in e["name"].lower() or "fc2" in e["name"].lower()]
        if not gem:
            continue
        t0 = float(r[0]["ts"])
        en = max(float(e["ts"]) + float(e["dur"]) for e in gem)
        f1 = [e for e in gem if "fc1" in e["name"].lower()]
        f2 = [e for e in gem if "fc2" in e["name"].lower()]
        row = {"span": en - t0}
        if f1:
            row["fc1_dur"] = float(f1[0]["dur"])
            row["fc1_end"] = float(f1[0]["ts"]) + float(f1[0]["dur"]) - t0
            row["fc1_start"] = float(f1[0]["ts"]) - t0
        if f2:
            row["fc2_dur"] = float(f2[0]["dur"])
            row["fc2_start"] = float(f2[0]["ts"]) - t0
        rows.append(row)
    out = {"n": len(rows)}
    for k in ("span", "fc1_dur", "fc1_end", "fc1_start", "fc2_dur", "fc2_start"):
        v = [r[k] for r in rows if k in r]
        if v:
            p50 = pct(v, 0.5)
            out[k] = round(p50, 3)
            if k == "span":
                out["noise"] = round((pct(v, 0.9) - pct(v, 0.1)) / max(p50, 1e-9), 4)
    return out


def main():
    global FLUSH
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    assert leaf._route_fuses_prologue()
    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    chain = Chain(st, leaf)
    log(f"STAGE0 shape={SHAPE} E={E} I={I} fc1 {FC1_BYTES/1e6:.2f} MB fc2 {FC2_BYTES/1e6:.2f} MB/expert")
    dist = {m: d_distribution(E, m) for m in (1, 2, 4)}
    log("busiest-rank D distribution (uniform top-6):", json.dumps(dist))
    table = []
    for m, dset in ((1, range(0, 7)), (4, range(0, 13))):
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        for d in dset:
            ids, w, act = routing(m, d)
            res = {}
            for mode in ("chain", "no_fc2", "concurrent", "fc2_l2warm"):
                warm = act if mode == "fc2_l2warm" else ()
                g = capture(lambda: chain(x, ids, w, mode, warm))
                res[mode] = trace(g)
                del g
            a = res["chain"]["span"]
            row = {"M": m, "D": d, "A_chain": a, "B_no_fc2": res["no_fc2"]["span"],
                   "C_concurrent": res["concurrent"]["span"], "R_fc2warm": res["fc2_l2warm"]["span"],
                   "exposed_fc2": round(a - res["no_fc2"]["span"], 3),
                   "fc2_floor": round(d * FC2_BYTES / PEAK * 1e6, 3),
                   "chain_floor": round(d * (FC1_BYTES + FC2_BYTES) / PEAK * 1e6, 3),
                   "UB_overlap": round(a - res["concurrent"]["span"], 3),
                   "UB_l2warm": round(a - res["fc2_l2warm"]["span"], 3),
                   "noise": max(r.get("noise", 0) for r in res.values()), "detail": res}
            table.append(row)
            log(f"{SHAPE} M{m} D{d}: A {a} | B {row['B_no_fc2']} | C {row['C_concurrent']} | "
                f"R {row['R_fc2warm']} | exposed_fc2 {row['exposed_fc2']} (floor {row['fc2_floor']}) | "
                f"UB_overlap {row['UB_overlap']} UB_l2warm {row['UB_l2warm']} | noise {row['noise']} | "
                f"A fc1 {res['chain'].get('fc1_dur')} end {res['chain'].get('fc1_end')} "
                f"fc2 start {res['chain'].get('fc2_start')} dur {res['chain'].get('fc2_dur')} | "
                f"C fc2 start {res['concurrent'].get('fc2_start')} dur {res['concurrent'].get('fc2_dur')}")
    # D-weighted recoverable at each M's busiest-rank mix, using the M=4 rows (flat in M)
    by_d = {r["D"]: r for r in table if r["M"] == 4}
    by_d1 = {r["D"]: r for r in table if r["M"] == 1}
    mix = {}
    for m, dd in dist.items():
        tot = {"UB_overlap": 0.0, "UB_l2warm": 0.0, "exposed_fc2": 0.0, "A_chain": 0.0, "cov": 0.0}
        for d, p in dd.items():
            r = (by_d1 if m == 1 and d in by_d1 else by_d).get(d)
            if r is None:
                continue
            tot["cov"] += p
            for k in ("UB_overlap", "UB_l2warm", "exposed_fc2", "A_chain"):
                tot[k] += p * r[k]
        mix[m] = {k: round(v / max(tot["cov"], 1e-9), 3) if k != "cov" else round(v, 4)
                  for k, v in tot.items()}
        log(f"{SHAPE} mix M{m}: {mix[m]}")
    od = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    with open(os.path.join(od, f"stage0_{SHAPE}.json"), "w") as f:
        json.dump({"table": table, "dist": dist, "mix": mix}, f, indent=1)
    print(json.dumps({"shape": SHAPE, "mix": mix,
                      "rows": [[r["M"], r["D"], r["A_chain"], r["UB_overlap"], r["UB_l2warm"],
                                r["exposed_fc2"]] for r in table]}))


if __name__ == "__main__":
    main()
