"""Idea 135: FI SM12x W4A8 decode MoE prologue fold. Chain of L layers of one EP4 rank.

Modes (argv[1]):
  pair    time Sm12xW4a8DecodeState.run chained over L layers in one CUDA graph, L2 flushed
          before each replay; per-layer ms. Last line: evidence_format JSON.
  stage0  BASE sizing probe: full run chain vs an "essential-only" chain (route/fc1/fc2 with
          precomputed local ids and preallocated, never re-zeroed buffers: wrong numerics,
          timing upper bound) vs a "prologue-only" chain (local_ids + torch.zeros).
Shapes: H=5120, I=2304, 96 local of 384 experts, top-6 (DeepSeek-V4.1-Flash EP4 rank).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "w4a8_fi_chain"))

import torch  # noqa: E402

from common import E, E_GLOBAL, H, I, TOPK, log, make_weights, trtllm_layout  # noqa: E402

L = int(os.environ.get("BENCH_LAYERS", "4"))
ITERS = int(os.environ.get("BENCH_ITERS", "100"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "1,2,4,8,16,32").split(",")]
N_LOCAL = {1: 3, 2: 2, 3: 2, 4: 2, 6: 2, 8: 2, 12: 2, 16: 2, 24: 2, 32: 2}


def routing(m, n_local, slot_start, seed, device="cuda"):
    g = torch.Generator(device="cpu")
    g.manual_seed(5000 + seed)
    others = torch.tensor([e for e in range(E_GLOBAL) if not slot_start <= e < slot_start + E])
    ids = torch.empty(m, TOPK, dtype=torch.int32)
    for t in range(m):
        loc = torch.randperm(E, generator=g)[:n_local] + slot_start
        rem = others[torch.randperm(others.numel(), generator=g)[: TOPK - n_local]]
        row = torch.cat([loc, rem])
        ids[t] = row[torch.randperm(TOPK, generator=g)].to(torch.int32)
    w = torch.rand(m, TOPK, generator=g) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return ids.to(device), w.to(device)


def make_states(n, slot_start=0):
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import \
        Sm12xW4a8DecodeState
    sts = []
    for l in range(n):
        w1, s1, w2, s2 = make_weights(seed=l)
        w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
        del w1, s1, w2, s2
        st = Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=E, slot_start=slot_start, swiglu_limit=10.0,
            enable_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
        st._keep = (w31, s31, w2v, s2s)
        sts.append(st)
    sts[0].warmup(TOPK)
    return sts


def time_graph(fn, iters, warmup=20, flush_mb=512):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    # Read-only L2 flush (pitfall 1356): a write flush leaves L2 full of dirty lines and
    # inflates the chain; reading >= 512 MB evicts clean.
    flush = torch.ones(flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    sink = torch.empty((), dtype=torch.float32, device="cuda")
    ts = []
    for i in range(warmup + iters):
        torch.sum(flush, dim=0, out=sink)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warmup:
            ts.append(a.elapsed_time(b))
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[len(ts) // 10], ts[(9 * len(ts)) // 10]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise": (p90 - p10) / med}, g, out


def inputs_for(m, n):
    xs, idss, ws = [], [], []
    for l in range(n):
        torch.manual_seed(100 * m + l)
        xs.append(torch.randn(m, H, dtype=torch.bfloat16, device="cuda"))
        ids, w = routing(m, N_LOCAL.get(m, 2), 0, seed=100 * m + l)
        idss.append(ids)
        ws.append(w)
    return xs, idss, ws


PAIR_LAYERS = int(os.environ.get("BENCH_PAIR_LAYERS", "40"))
PAIR_WSETS = int(os.environ.get("BENCH_PAIR_WSETS", "8"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))


def mode_pair():
    """40-layer chain (layer l uses weight set l % 8: >= 4 sets apart is > 128 MB of expert
    weights, so every layer reads cold), REPEATS independent timing blocks per M."""
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as mod
    fused = hasattr(mod, "_route_fuses_prologue") and mod._route_fuses_prologue()
    arm = "CAND" if fused else "BASE"
    kernels = ("route_assign_decode_kernel[+remap+zero-fill],_mxfp8_q0_route_direct_kernel,"
               "fc1_act_q1,fc2_finalize" if fused else
               "add,compare_scalar(ge),compare_scalar(lt),bitwise_and,FillFunctor<int>,where,"
               "FillFunctor<uchar>,route_assign_decode_kernel,_mxfp8_q0_route_direct_kernel,"
               "fc1_act_q1,fc2_finalize")
    n_l = PAIR_LAYERS
    sets = make_states(min(PAIR_WSETS, n_l))
    sts = [sets[l % len(sets)] for l in range(n_l)]
    log(arm, "states built: layers", n_l, "weight sets", len(sets), "kernels/layer:", kernels)
    batch = {}
    for m in MS:
        xs, idss, ws = inputs_for(m, n_l)

        def fn():
            return [sts[l].run(xs[l], idss[l], ws[l]) for l in range(n_l)]

        meds = []
        for r in range(REPEATS):
            stats, g, outs = time_graph(fn, ITERS)
            del g
            meds.append(stats["median_ms"] / n_l)
            log(arm, "M", m, "repeat", r, "per_layer_us", round(1000 * meds[-1], 3),
                "within-block noise", round(stats["noise"], 4), "kernels:", kernels)
        meds.sort()
        med = meds[len(meds) // 2]
        noise = max((meds[-1] - meds[0]) / med, stats["noise"])
        batch[str(m)] = {"kernel_ms": med, "noise_floor": noise,
                         "repeat_ms": meds, "graph_ms": med * n_l}
        log(arm, "M", m, json.dumps(batch[str(m)]), "kernels:", kernels)
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "arm": arm,
                      "kernels_per_layer": kernels,
                      "units": f"ms per layer: median over {REPEATS} timing blocks of "
                               f"({ITERS}-replay median of a {n_l}-layer CUDA-graph chain) / "
                               f"{n_l}; routed-MoE FI SM12x decode leaf, one EP4 rank, L2 "
                               "flushed by a 512 MB read before every replay"}))


def mode_stage0():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as mod
    sts = make_states(L)
    route, _, fc1, out_sf_shape, fc2, fi_act = mod._load_ops()
    res = {}
    for m in MS:
        xs, idss, ws = inputs_for(m, L)

        def full():
            return [sts[l].run(xs[l], idss[l], ws[l]) for l in range(L)]

        locs = [sts[l].local_ids(idss[l]) for l in range(L)]
        pairs = m * TOPK
        bufs = []
        for l in range(L):
            q1 = torch.zeros(pairs, I, dtype=torch.float8_e4m3fn, device="cuda")
            sf1 = torch.zeros(out_sf_shape(pairs, I, E), dtype=torch.int32, device="cuda")
            out = torch.zeros(m, H, dtype=torch.bfloat16, device="cuda")
            bufs.append((q1, sf1, out))
        wsp = mod._workspace(xs[0], locs[0], E)

        def essential():
            outs = []
            for l in range(L):
                st = sts[l]
                q1, sf1, out = bufs[l]
                offsets, token_map, rw, a_q, a_sf = route(xs[l], locs[l], ws[l], E,
                                                          workspace=wsp,
                                                          enable_pdl=st.enable_pdl)
                fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=st.enable_pdl,
                    out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=10.0)
                fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m, tune=False,
                    enable_pdl=st.enable_pdl, out=out)
                outs.append(out)
            return outs

        nbuf = pairs * I + out_sf_shape(pairs, I, E)[0] * out_sf_shape(pairs, I, E)[1] * 4 \
            + m * H * 2 + 512

        def prologue():
            o = []
            for l in range(L):
                o.append(sts[l].local_ids(idss[l]))
                o.append(torch.zeros(nbuf, dtype=torch.uint8, device="cuda"))
            return o

        row = {}
        for name, fn in (("full", full), ("essential", essential), ("prologue", prologue),
                         ("full2", full)):
            stats, g, _ = time_graph(fn, ITERS)
            row[name] = {"per_layer_us": 1000 * stats["median_ms"] / L,
                         "noise": stats["noise"]}
            del g
        row["prologue_cost_us_per_layer"] = (
            0.5 * (row["full"]["per_layer_us"] + row["full2"]["per_layer_us"])
            - row["essential"]["per_layer_us"])
        res[str(m)] = row
        log("M", m, json.dumps(row))
    print(json.dumps({"stage0": res, "layers": L}))


if __name__ == "__main__":
    {"pair": mode_pair, "stage0": mode_stage0}[sys.argv[1]]()
