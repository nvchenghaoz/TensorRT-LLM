"""Idea 143 paired microbenchmark: FI SM12x decode-MoE leaf (route -> fc1_act_q1 -> fc2_finalize).

Runs identically in BASE (main) and CAND (branch) trees.  One EP4 rank of DeepSeek-V4.1-Flash
(96 local experts, H 5120, I 2304), the production leaf Sm12xW4a8DecodeState.run captured in a
CUDA graph; before every replay a 512 MB READ-ONLY flush (clean L2, pitfall 1356).  Timing is
from a Kineto trace of the graph replays (CUDA events on this box tick in ~2 us steps): per
replay the span from the first leaf node's start to the last node's end, and the fc1 / fc2 node
durations; p50 over the replays and noise = (p90 - p10) / p50.

Cases: forced distinct-local-expert counts D at M = 1 and 4, and the bench routing
(experiments/ascent/w4a8_fi_chain make_routing) at M = 1, 2, 4, 8, 16, 32.

Last stdout line: {"batch": {"<case>": {"kernel_ms": chain p50 ms, "noise_floor": ..}}, ...}
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

import torch  # noqa: E402

from common import E, H, log, make_routing, make_weights, trtllm_layout  # noqa: E402

TAG = os.environ.get("BENCH_TAG", "RUN")
REPLAYS = int(os.environ.get("BENCH_REPLAYS", "40"))
FLUSH = None


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
    path = os.path.join(tempfile.gettempdir(), f"bp_{os.getpid()}.json")
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
    spans, f1, f2, names = [], [], [], None
    for r in reps:
        # leaf nodes only: from the first node after the flush to the end of fc2 (the
        # next flush's own small kernels follow and are not part of the leaf)
        last = max((i for i, e in enumerate(r) if "fc2" in e["name"].lower()), default=None)
        if last is None:
            continue
        r = r[: last + 1]
        st = min(float(e["ts"]) for e in r)
        en = max(float(e["ts"]) + float(e["dur"]) for e in r)
        spans.append(en - st)
        f1.append(sum(float(e["dur"]) for e in r if "fc1" in e["name"].lower()))
        f2.append(sum(float(e["dur"]) for e in r if "fc2" in e["name"].lower()))
        names = names or [e["name"][:90] for e in r]
    def summ(v):
        p50 = pct(v, 0.5)
        return round(p50, 3), round((pct(v, 0.9) - pct(v, 0.1)) / max(p50, 1e-9), 4)
    return summ(spans), summ(f1), summ(f2), len(spans), names


def main():
    global FLUSH
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
    from census import forced_routing

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    sk = os.environ.get("TRTLLM_SM12X_FI_SPLITK", "<unset>")
    log(f"{TAG}: leaf module {leaf.__file__}; TRTLLM_SM12X_FI_SPLITK={sk}")
    cases = []
    dset = [int(v) for v in os.environ.get("BENCH_D", "1,2,3,4,5,6,7,8,9,10,11,12").split(",")]
    for m in (1, 4):
        for d in dset:
            if d <= 6 * m:
                cases.append((f"M{m}_D{d}", m, *forced_routing(m, d, 0)))
    for m in (1, 2, 4, 8, 16, 32):
        ids, w = make_routing(m, {1: 3, 2: 2}.get(m, 2), seed=m)
        cases.append((f"bench_M{m}", m, ids, w))
    batch, table = {}, []
    for label, m, ids, w in cases:
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        d = int(torch.unique(ids[(ids >= 0) & (ids < E)]).numel())
        g = capture(lambda: st.run(x, ids, w))
        (cs, cn), (f1, f1n), (f2, f2n), n, names = trace(g)
        del g
        batch[label] = {"kernel_ms": round(cs / 1000.0, 6), "noise_floor": cn}
        row = {"case": label, "M": m, "D": d, "chain_us": cs, "chain_noise": cn, "fc1_us": f1,
               "fc1_noise": f1n, "fc2_us": f2, "fc2_noise": f2n, "replays": n}
        table.append(row)
        log(f"{TAG} {label} D={d}: chain {cs} us (noise {cn}) | fc1 {f1} us | fc2 {f2} us "
            f"| replays {n} | kernels: {' / '.join(nm[:40] for nm in (names or []))}")
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    with open(os.path.join(out_dir, f"bench_pair_{TAG}.json"), "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}, "tag": TAG,
                      "table": [[r["case"], r["chain_us"], r["fc1_us"], r["fc2_us"]] for r in table]}))


if __name__ == "__main__":
    main()
