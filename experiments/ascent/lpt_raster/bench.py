"""Idea 32: L2-aware tile order (raster / swizzle) for the SM120 NVFP4 LPT pair
(flux3_fp4_lpt_pair_sm120.cu: down K=9216 + attn_out K=3072, N=3072, one grouped launch).

Shapes (FLUX.3 klein 2048^2): single blocks at M=16896 (t2i) and M=33280 (i2i). Synthetic NVFP4
operands (fixed seeds), CASES distinct weight/activation sets per M, cycled. No model load.

  pair  : run_gpu_pair unit (both trees). CUDA graph of CALLS lpt.run calls (hoisted quant, outputs
          preallocated) per M, interleaved replays; kernel_ms per call per case label.
  sweep : branch only. Per M, every (raster, max_swizzle) variant + an A/A main arm, interleaved
          graph replays; plus QKV3 (gemm_qkv3) and SwiGLU-FP4 timing at both M (ratio check).
  prof  : branch only. One pair launch per (M, variant) for ncu: main, AlongN, AlongM swizzle 8.
  check : branch only, untimed. Per M: references = FlashInfer stock tactic 17 per GEMM and the
          kernel at main's order (raster 0, swizzle 1). For the default policy and each variant:
          WARMUP+ITERS calls, outputs poisoned before each call, every call bitwise vs refs and vs
          call 0, inputs unchanged, tile counter; CUDA-graph capture + 60 replays bitwise vs eager;
          TRTLLM_FLUX3_LPT_RASTER=0 -> raster_for == main's (0, 1) at both M, outputs bitwise.
"""

import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

H, FF = 3072, 9216
MS = {"t2i-2048": 16896, "i2i-2048": 33280}
CASES = 2
CALLS = 20
WARMUP = 3
ITERS = 20
VARIANTS = {
    "main_M1": (0, 1), "N1": (1, 1), "N2": (1, 2), "N4": (1, 4), "N8": (1, 8),
    "M2": (0, 2), "M4": (0, 4), "M8": (0, 8),
}


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make_case(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_attn = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    x_down = torch.randn(M, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    s = list(map(_gs, (x_attn, x_down, w_out, w_down)))
    c = {"M": M}
    c["a_attn"], c["sf_attn"] = fp.nvfp4_quantize(x_attn, _inv(s[0]))
    c["a_down"], c["sf_down"] = fp.nvfp4_quantize(x_down, _inv(s[1]))
    c["w_out"], c["sfw_out"] = fp.nvfp4_quantize(w_out, _inv(s[2]))
    c["w_down"], c["sfw_down"] = fp.nvfp4_quantize(w_down, _inv(s[3]))
    c["al_down"] = _scalar(x_attn.device, s[1] * s[3])
    c["al_attn"] = _scalar(x_attn.device, s[0] * s[2])
    c["o_down"] = torch.empty((M, H), device=dev, dtype=torch.bfloat16)
    c["o_attn"] = torch.empty((M, H), device=dev, dtype=torch.bfloat16)
    del x_attn, x_down, w_out, w_down
    return c


IN_KEYS = ["a_attn", "sf_attn", "a_down", "sf_down", "w_out", "sfw_out", "w_down", "sfw_down",
           "al_down", "al_attn"]


def run_pair(c, raster=None):
    kw = {} if raster is None else {"raster": raster}
    lpt.run(c["a_down"], c["w_down"], c["sf_down"], c["sfw_down"], c["al_down"], c["o_down"],
            c["a_attn"], c["w_out"], c["sf_attn"], c["sfw_out"], c["al_attn"], c["o_attn"], **kw)


def fi_linear(a, w, sfa, sfw, alpha, tactic):
    out = torch.empty((a.shape[0], w.shape[0]), device="cuda", dtype=torch.bfloat16)
    fp._sm120_fi_fp4_module().fp4_gemm(a, w, sfa, sfw, alpha, out,
                                       fp._sm120_fi_workspace(a.device), tactic)
    return out


def stat(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2], v[n // 10], v[(n * 9) // 10]


def capture(fns, stream):
    with torch.cuda.stream(stream):
        for f in fns[:2]:
            f()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for f in fns:
                f()
    torch.cuda.synchronize()
    return graph


def time_graphs(graphs, samples, warm, calls):
    names = list(graphs)
    for _ in range(warm):
        for n in names:
            graphs[n].replay()
    torch.cuda.synchronize()
    out = {n: [] for n in names}
    for i in range(samples):
        order = names[i % len(names):] + names[:i % len(names)]
        evs = []
        for n in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            evs.append((n, a, b))
        torch.cuda.synchronize()
        for n, a, b in evs:
            out[n].append(a.elapsed_time(b) * 1e3 / calls)
        if i % 10 == 0:
            log(f"  sample {i}: " + " ".join(f"{n}={out[n][-1]:.1f}" for n in names))
    return out


def preload():
    fp.sm120_linear_preload(torch.device("cuda", torch.cuda.current_device()))
    lpt.preload(torch.device("cuda", torch.cuda.current_device()))
    log(f"lpt_pair_info={lpt.module().lpt_pair_info()} has_raster={hasattr(lpt, 'raster_for')}")


def bench():
    preload()
    res = {}
    for label, M in MS.items():
        cs = [make_case(M, 1000 * M + i) for i in range(CASES)]
        pristine = [[c[k].clone() for k in IN_KEYS] for c in cs]
        s = torch.cuda.Stream()
        g = capture([lambda c=cs[i % CASES]: run_pair(c) for i in range(CALLS)], s)
        samples = 60 if M < 20000 else 40
        t = time_graphs({"pair": g}, samples=samples, warm=5, calls=CALLS)["pair"]
        for p, c in zip(pristine, cs):
            assert all(torch.equal(x, c[k]) for x, k in zip(p, IN_KEYS)), "inputs changed"
        med, p10, p90 = stat(t)
        log(f"{label} M={M}: median {med:.1f} us/call p10 {p10:.1f} p90 {p90:.1f}")
        res[label] = {"kernel_ms": med / 1e3, "noise_floor": (p90 - p10) / med, "M": M,
                      "samples": len(t)}
        del g, cs, pristine
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"batch": res, "unit": f"one lpt.run pair call (down K=9216 + attn_out "
                      f"K=3072, N=3072, pre-quantized), graph of {CALLS} calls, {CASES} sets",
                      "output": {"class": "exact"}}))


def qkv_swiglu_times():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as st
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as sw

    st.preload(torch.device("cuda", torch.cuda.current_device()))
    out = {}
    for label, M in MS.items():
        g = torch.Generator(device="cuda").manual_seed(M)
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
        wq = torch.randn(3 * H, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
        wg = torch.randn(2 * FF, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
        a, sfa = fp.nvfp4_quantize(x, _inv(_gs(x)))
        q_w, q_sf = fp.nvfp4_quantize(wq, _inv(_gs(wq)))
        g_w, g_sf = fp.nvfp4_quantize(wg, _inv(_gs(wg)))
        alpha3 = torch.full((3,), 1e-3, device="cuda", dtype=torch.float32)
        alpha1 = _scalar(x.device, 1e-3)
        inv_o = torch.ones(1, device="cuda", dtype=torch.float32)
        qkv_out = torch.empty((M, 3 * H), device="cuda", dtype=torch.bfloat16)
        sw_o, sw_s = sw.allocate_outputs(M, 2 * FF, x.device)
        del x, wq, wg
        s = torch.cuda.Stream()
        graphs = {
            "qkv3": capture([lambda: st.gemm_qkv3(a, q_w, sfa, q_sf, alpha3, qkv_out)] * CALLS, s),
            "swiglu": capture([lambda: sw.swiglu_packed(a, g_w, sfa, g_sf, alpha1, inv_o,
                                                        output=sw_o, output_scale=sw_s)] * CALLS,
                              s),
        }
        t = time_graphs(graphs, samples=30, warm=3, calls=CALLS)
        out[label] = {n: round(stat(v)[0], 1) for n, v in t.items()}
        log(f"{label} M={M}: {out[label]}")
        del graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    r = {n: out["i2i-2048"][n] / out["t2i-2048"][n] for n in ("qkv3", "swiglu")}
    log(f"ratio 33280/16896 (rows x{33280 / 16896:.3f}): {r}")
    return out, r


def sweep():
    preload()
    rows = {}
    for label, M in MS.items():
        cs = [make_case(M, 1000 * M + i) for i in range(CASES)]
        s = torch.cuda.Stream()
        graphs = {}
        for name, rv in VARIANTS.items():
            graphs[name] = capture([lambda c=cs[i % CASES], rv=rv: run_pair(c, rv)
                                    for i in range(CALLS)], s)
        graphs["main_M1_AA"] = capture([lambda c=cs[i % CASES]: run_pair(c, (0, 1))
                                        for i in range(CALLS)], s)
        log(f"{label}: default policy = {lpt.raster_for(M)}")
        t = time_graphs(graphs, samples=30 if M < 20000 else 20, warm=3, calls=CALLS)
        row = {}
        for n, v in t.items():
            med, p10, p90 = stat(v)
            row[n] = {"us": round(med, 1), "p10": round(p10, 1), "p90": round(p90, 1)}
        base = row["main_M1"]["us"]
        for n in row:
            row[n]["vs_main"] = round(row[n]["us"] / base, 4)
        rows[label] = row
        log(f"{label} M={M}: {json.dumps(row)}")
        del graphs, cs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    qs, ratio = qkv_swiglu_times()
    print(json.dumps({"pair_us_per_call": rows, "qkv3_swiglu_us": qs, "ratio_33280_16896": ratio}))


def prof():
    preload()
    for label, M in MS.items():
        c = make_case(M, 1000 * M)
        torch.cuda.synchronize()
        for name in ("main_M1", "N1", "M8"):
            log(f"prof {label} M={M} {name} {VARIANTS[name]}")
            run_pair(c, VARIANTS[name])
            torch.cuda.synchronize()
        del c
        torch.cuda.empty_cache()
    print(json.dumps({"order": [f"{l}:{n}" for l in MS for n in ("main_M1", "N1", "M8")]}))


def check():
    preload()
    ws = lpt.workspace(torch.device("cuda", torch.cuda.current_device()))
    results = {}
    passed = True
    arms = {"default": None, **VARIANTS}
    for label, M in MS.items():
        c = make_case(M, 77 + M)
        pristine = [c[k].clone() for k in IN_KEYS]
        tiles = 2 * ((M + 127) // 128) * (H // 128)
        ref17 = {
            "down": fi_linear(c["a_down"], c["w_down"], c["sf_down"], c["sfw_down"], c["al_down"],
                              17),
            "attn": fi_linear(c["a_attn"], c["w_out"], c["sf_attn"], c["sfw_out"], c["al_attn"],
                              17),
        }
        run_pair(c, (0, 1))
        torch.cuda.synchronize()
        ref_main = {"down": c["o_down"].clone(), "attn": c["o_attn"].clone()}
        main_vs_17 = (torch.equal(ref_main["down"], ref17["down"])
                      and torch.equal(ref_main["attn"], ref17["attn"]))
        results[f"{label}_main_order_vs_fi_t17"] = main_vs_17
        passed &= main_vs_17
        log(f"{label}: policy {lpt.raster_for(M)}, main order vs t17 bitwise {main_vs_17}")
        for arm, rv in arms.items():
            for k, p in zip(IN_KEYS, pristine):
                c[k].copy_(p)
            for _ in range(WARMUP):
                run_pair(c, rv)
            torch.cuda.synchronize()
            n_ref = n_first = n_17 = 0
            counters = set()
            first = None
            for i in range(ITERS):
                c["o_down"].fill_(float("nan"))
                c["o_attn"].fill_(float("nan"))
                run_pair(c, rv)
                torch.cuda.synchronize()
                counters.add(int(ws[:4].view(torch.int32).item()))
                if first is None:
                    first = (c["o_attn"].clone(), c["o_down"].clone())
                n_ref += bool(torch.equal(c["o_attn"], ref_main["attn"])
                              and torch.equal(c["o_down"], ref_main["down"]))
                n_17 += bool(torch.equal(c["o_attn"], ref17["attn"])
                             and torch.equal(c["o_down"], ref17["down"]))
                n_first += bool(torch.equal(c["o_attn"], first[0])
                                and torch.equal(c["o_down"], first[1]))
            inputs_ok = all(torch.equal(p, c[k]) for k, p in zip(IN_KEYS, pristine))
            # CUDA graph capture + 60 back-to-back replays (counter reset by the args kernel).
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                run_pair(c, rv)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    run_pair(c, rv)
            torch.cuda.synchronize()
            c["o_down"].fill_(float("nan"))
            c["o_attn"].fill_(float("nan"))
            for _ in range(60):
                graph.replay()
            torch.cuda.synchronize()
            g_back = (torch.equal(c["o_attn"], first[0]) and torch.equal(c["o_down"], first[1]))
            g_each = 0
            for _ in range(10):
                c["o_down"].fill_(float("nan"))
                c["o_attn"].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                g_each += bool(torch.equal(c["o_attn"], first[0])
                               and torch.equal(c["o_down"], first[1]))
                counters.add(int(ws[:4].view(torch.int32).item()))
            del graph
            # dynamic mode: the counter ends at total claimed = grid + tiles beyond; >= tiles
            ok = (n_ref == ITERS and n_17 == ITERS and n_first == ITERS and inputs_ok and g_back
                  and g_each == 10 and min(counters) >= tiles - 188)
            passed &= ok
            key = f"{label}_{arm}"
            results[key] = {"raster": rv if rv is not None else list(lpt.raster_for(M)),
                            "bitwise_vs_main_order": f"{n_ref}/{ITERS}",
                            "bitwise_vs_fi_t17": f"{n_17}/{ITERS}",
                            "bitwise_vs_call0": f"{n_first}/{ITERS}", "inputs_unchanged": inputs_ok,
                            "graph_60_back_to_back_bitwise": g_back,
                            "graph_replay_poisoned_x10": f"{g_each}/10",
                            "counter_values": sorted(counters), "tiles": tiles, "ok": ok}
            log(key, json.dumps(results[key]))
        # Kill switch: TRTLLM_FLUX3_LPT_RASTER=0 -> main's order (0, 1) at every M.
        os.environ["TRTLLM_FLUX3_LPT_RASTER"] = "0"
        lpt.raster_enabled.cache_clear()
        ks_order = tuple(lpt.raster_for(M))
        c["o_down"].fill_(float("nan"))
        c["o_attn"].fill_(float("nan"))
        run_pair(c, None)
        torch.cuda.synchronize()
        ks_ok = (ks_order == (0, 1) and torch.equal(c["o_attn"], ref_main["attn"])
                 and torch.equal(c["o_down"], ref_main["down"]))
        os.environ.pop("TRTLLM_FLUX3_LPT_RASTER")
        lpt.raster_enabled.cache_clear()
        results[f"{label}_killswitch"] = {"raster_for": list(ks_order), "ok": ks_ok}
        passed &= ks_ok
        log(f"{label} killswitch: {results[f'{label}_killswitch']}")
        del c, pristine, ref17, ref_main
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(passed), "compared": "LPT pair outputs vs FlashInfer stock "
                      "tactic 17 and vs the same kernel at main's order (AlongM, swizzle 1), "
                      "per call, graph replays, kill switch", "results": results}))


ALT_ROUNDS = 50
SMALL_M = 4176


def _tiles(M):
    return 2 * ((M + 127) // 128) * (H // 128)


def _counter(ws):
    return int(ws[:4].view(torch.int32).item())


def _alt_set(name, ms, seed0):
    """One graph set: cases at row counts ms (cycled), with per-case refs: main order (0, 1) eager,
    FlashInfer tactic 17, pristine inputs."""
    cs = []
    for i, M in enumerate(ms):
        c = make_case(M, seed0 + i)
        c["pristine"] = [c[k].clone() for k in IN_KEYS]
        c["ref17"] = (fi_linear(c["a_attn"], c["w_out"], c["sf_attn"], c["sfw_out"], c["al_attn"],
                                17),
                      fi_linear(c["a_down"], c["w_down"], c["sf_down"], c["sfw_down"], c["al_down"],
                                17))
        run_pair(c, (0, 1))
        torch.cuda.synchronize()
        c["ref_main"] = (c["o_attn"].clone(), c["o_down"].clone())
        c["main_vs_17"] = (torch.equal(c["ref_main"][0], c["ref17"][0])
                           and torch.equal(c["ref_main"][1], c["ref17"][1]))
        c["round0"] = None
        cs.append(c)
    s = torch.cuda.Stream()
    # The same graph the timed run (bench) uses: CALLS lpt.run calls, cases cycled, default policy.
    graph = capture([lambda c=cs[i % len(cs)]: run_pair(c) for i in range(CALLS)], s)
    last_M = cs[(CALLS - 1) % len(cs)]["M"]
    log(f"set {name}: Ms {ms}, policy {[tuple(lpt.raster_for(M)) for M in ms]}, "
        f"main==t17 {[c['main_vs_17'] for c in cs]}, counter after graph = tiles({last_M})")
    return {"name": name, "cases": cs, "graph": graph, "last_M": last_M,
            "stats": {"replays": 0, "bitwise_main": 0, "bitwise_t17": 0, "bitwise_round0": 0,
                      "counter_ok": 0, "inputs_ok": 0, "max_abs_diff_vs_main": 0.0}}


def _poison(c):
    c["o_attn"].fill_(float("nan"))
    c["o_down"].fill_(float("nan"))


def _eq(c, ref):
    return torch.equal(c["o_attn"], ref[0]) and torch.equal(c["o_down"], ref[1])


def _maxdiff(c, ref):
    d = 0.0
    for o, r in zip((c["o_attn"], c["o_down"]), ref):
        x = (o.float() - r.float()).abs().nan_to_num(float("inf")).max().item()
        d = max(d, x)
    return d


def _replay_check(st, ws):
    for c in st["cases"]:
        _poison(c)
    st["graph"].replay()
    torch.cuda.synchronize()
    cnt = _counter(ws)
    ok_all = True
    s = st["stats"]
    s["replays"] += 1
    s["counter_ok"] += cnt == _tiles(st["last_M"])
    ok_all &= cnt == _tiles(st["last_M"])
    bm = bt = br = inp = True
    for c in st["cases"]:
        if c["round0"] is None:
            c["round0"] = (c["o_attn"].clone(), c["o_down"].clone())
        m = _eq(c, c["ref_main"])
        if not m:
            s["max_abs_diff_vs_main"] = max(s["max_abs_diff_vs_main"], _maxdiff(c, c["ref_main"]))
        bm &= m
        bt &= _eq(c, c["ref17"])
        br &= _eq(c, c["round0"])
        inp &= all(torch.equal(p, c[k]) for k, p in zip(IN_KEYS, c["pristine"]))
    s["bitwise_main"] += bm
    s["bitwise_t17"] += bt
    s["bitwise_round0"] += br
    s["inputs_ok"] += inp
    return ok_all and bm and bt and br and inp


def _eager_check(c, rv, ws, stats):
    """Eager call at another M between replays (rv=(0, 1): main's order; None: default policy)."""
    _poison(c)
    run_pair(c, rv)
    torch.cuda.synchronize()
    ok = (_eq(c, c["ref_main"]) and _eq(c, c["ref17"]) and _counter(ws) == _tiles(c["M"])
          and all(torch.equal(p, c[k]) for k, p in zip(IN_KEYS, c["pristine"])))
    stats["calls"] += 1
    stats["ok"] += ok
    return ok


def alt():
    """Untimed alternation over the ONE shared lpt.workspace (args + tile counter).
    Phase A (evaluator recipe): the timed run's graphs (CALLS calls, 2 cycled cases) at M=16896
    (t2i) and M=33280 (i2i), default policy; ALT_ROUNDS rounds of t2i->i2i->t2i replays, an
    eager main-order (0, 1) call at the other M after every replay, and an eager default-policy
    small-M (4176) call per round. Phase B (plan 407 at the served Ms): step-shaped graphs
    t2i = (16464, 16384) and i2i = (32848, 16384), same alternation. Outputs NaN-poisoned before
    every replay/call; every replay compared bitwise with main's order (0, 1), FI tactic 17 and
    round 0; tile counter == tile count after every replay; inputs vs pristine every replay."""
    preload()
    ws = lpt.workspace(torch.device("cuda", torch.cuda.current_device()))
    small = _alt_set("small", [SMALL_M], 5000)  # its own graph unused; eager calls only
    phases = {
        "A_evaluator": (("t2i", [16896, 16896]), ("i2i", [33280, 33280])),
        "B_served": (("t2i", [16464, 16384]), ("i2i", [32848, 16384])),
    }
    out = {}
    passed = small["cases"][0]["main_vs_17"]
    for pname, ((tn, tms), (iname, ims)) in phases.items():
        t2i = _alt_set(f"{pname}_{tn}", tms, 77 + tms[0])
        i2i = _alt_set(f"{pname}_{iname}", ims, 77 + ims[0])
        passed &= all(c["main_vs_17"] for c in t2i["cases"] + i2i["cases"])
        eager = {"main_order_other_M": {"calls": 0, "ok": 0},
                 "default_small_M": {"calls": 0, "ok": 0}}
        rounds_ok = 0
        for r in range(ALT_ROUNDS):
            ok = True
            for st, other in ((t2i, i2i), (i2i, t2i), (t2i, i2i)):
                ok &= _replay_check(st, ws)
                ok &= _eager_check(other["cases"][0], (0, 1), ws, eager["main_order_other_M"])
            ok &= _eager_check(small["cases"][0], None, ws, eager["default_small_M"])
            rounds_ok += ok
            if r % 10 == 0 or not ok:
                log(f"{pname} round {r}: ok={ok} t2i={t2i['stats']} i2i={i2i['stats']} "
                    f"eager={eager}")
        res = {"rounds": ALT_ROUNDS, "rounds_passed": rounds_ok,
               "t2i": {"Ms": tms, "tiles_after_graph": _tiles(t2i["last_M"]), **t2i["stats"]},
               "i2i": {"Ms": ims, "tiles_after_graph": _tiles(i2i["last_M"]), **i2i["stats"]},
               "eager": eager,
               "policy": {str(M): list(lpt.raster_for(M)) for M in sorted(set(tms + ims))}}
        passed &= rounds_ok == ALT_ROUNDS
        out[pname] = res
        log(f"{pname}: {json.dumps(res)}")
        del t2i, i2i
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(passed),
                      "compared": "one shared lpt.workspace; per-round alternation t2i->i2i->t2i "
                      "of the timed graphs (CALLS calls, cycled cases, default policy (1,1)) with "
                      "eager main-order (0,1) calls at the other M; every replay NaN-poisoned, "
                      "bitwise vs main order (0,1), FI tactic 17 and round 0; counter == tiles; "
                      "inputs vs pristine", "results": out}))


if __name__ == "__main__":
    {"pair": bench, "sweep": sweep, "prof": prof, "check": check, "alt": alt}[sys.argv[1]]()
