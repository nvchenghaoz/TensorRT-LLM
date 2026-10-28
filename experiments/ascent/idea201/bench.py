"""Idea 201: CLC dynamic tile scheduler for the FLUX.3 SM120 SwiGLU->NVFP4 gate_up (CUTLASS).

Synthetic tensors at the served shapes (H=3072, 24 heads x 128, FF=9216 -> gate_up N=18432, K=3072;
single blocks S=4176, image mode blocks S=4096), built with experiments/ascent/idea161/step0.py's
make_case. No model load.

  info  : HEAD, MAIN_AT_START, ancestry probe, tensorrt_llm.__file__, selected scheduler.
  table : (candidate tree only) graph replay, per S: K served mode-3 op calls
          trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up (FMHA || gate_up), and K
          standalone gate_up calls; arms static / dynamic, each captured twice (A, B = A/A).
  pair  : evidence unit = one graph of 28 x S4176 + 4 x S4096 served mode-3 op calls with the
          tree's default module (main: static; branch: dynamic). Prints the evidence JSON.
  check : (candidate tree only) untimed bitwise check of gate_up out/out_sf dynamic vs static
          (and vs the CuTe run('packed') reference) at M=4096/4176/4256, 0xFF prefill, warmup +
          iterations vs call 0, inputs unchanged, graph capture + 3 replays; served op dynamic vs
          the sequential static ops; env static restores main's module; counters.
  grid  : a few launches of each module (for nsys launch shapes).
"""

import json
import math
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "idea161"))

import step0  # noqa: E402
import tensorrt_llm  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc  # noqa: E402

MAIN_AT_START = "cec13c57"
HEADS, D = step0.HEADS, step0.D
K_CALLS = int(os.environ.get("IDEA201_K", "10"))
ROUNDS = int(os.environ.get("IDEA201_ROUNDS", "48"))
SPIN = 6_000_000
PAIR_SAMPLES = 200
PAIR_SOAK_S = 20.0
PAIR_REPLAYS = 3


def log(*a):
    print(*a, flush=True)


def info():
    wt = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

    def git(*args):
        r = subprocess.run(["git", "-C", wt, *args], capture_output=True, text=True)
        return r.returncode, r.stdout.strip()

    _, head = git("rev-parse", "HEAD")
    rc, _ = git("merge-base", "--is-ancestor", MAIN_AT_START, "HEAD")
    sched = swc.tile_sched() if hasattr(swc, "tile_sched") else "static(main: no selector)"
    log(f"HEAD={head} MAIN_AT_START={MAIN_AT_START} ancestry rc={rc} "
        f"tensorrt_llm.__file__={tensorrt_llm.__file__} sched={sched}")
    return {"head": head, "ancestry_rc": rc, "sched": sched}


def modules():
    """{arm: module}; main's tree has only the static module (get_module)."""
    if hasattr(swc, "get_module_dynamic"):
        return {"static": swc.get_module_static(), "dynamic": swc.get_module_dynamic()}
    return {"static": swc.get_module()}


_ORIG_GET_MODULE = swc.get_module


def use(module):
    swc.get_module = (lambda: module) if module is not None else _ORIG_GET_MODULE


def prequant(c):
    qkv8, scales, _ = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope_fp8(
        c["a_x"], c["qkv"].weight, c["sf_x"], c["qkv"].scale, c["q_norm"], c["k_norm"],
        c["cos_sin"], c["qkv"].qkv_scale, c["x"], c["qkv"].weight, 1.0 / math.sqrt(D),
    )
    torch.cuda.synchronize()
    c["qkv8"], c["scales"] = qkv8, scales
    return c


def served_op(c, mode=3):
    up, down = c["up"], c["down"]
    return torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_side_gate_up(
        c["qkv8"], c["scales"], 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale,
        down.inverse_scale, c["x"], c["qkv"].weight, mode,
    )


def sequential(c):
    up, down = c["up"], c["down"]
    attn = torch.ops.trtllm.visual_gen_fp8_attention_sm120_prequant(c["qkv8"], c["scales"], 1, HEADS)
    packed, sf = torch.ops.trtllm.flux3_nvfp4_swiglu_prepared(
        c["a_x"], up.weight, c["sf_x"], up.scale, down.inverse_scale, c["x"], c["qkv"].weight,
    )
    return attn, packed, sf


def gate_up_only(c):
    up, down = c["up"], c["down"]
    alpha = c.setdefault("alpha", torch.tensor([up.input_global_scale * up.weight_global_scale],
                                               device="cuda", dtype=torch.float32))
    return swc.swiglu_packed(c["a_x"], up.weight, c["sf_x"], up.scale, alpha, down.inverse_scale)


def table():
    mods = modules()
    assert "dynamic" in mods, "table runs in the candidate tree"
    res = {}
    only = os.environ.get("IDEA201_S")
    for S, seed in ((4176, 1), (4096, 2)):
        if only and int(only) != S:
            continue
        c = prequant(step0.make_case(S, seed))
        graphs = {}
        for unit, fn in (("op", served_op), ("gu", gate_up_only)):
            for arm, mod in mods.items():
                use(mod)
                for rep in ("A", "B"):
                    graphs[f"{unit}_{arm}_{rep}"], _ = step0.capture(
                        lambda: [fn(c)[0] for _ in range(K_CALLS)])
        use(None)
        log(f"S={S}: captured {len(graphs)} graphs; soak")
        t0 = time.time()
        while time.time() - t0 < 5.0:
            for g in graphs.values():
                g.replay()
            torch.cuda.synchronize()
        keys = list(graphs)
        times = {k: [] for k in keys}
        for r in range(ROUNDS):
            order = keys if r % 2 == 0 else keys[::-1]
            evs = []
            for k in order:
                torch.cuda.synchronize()
                evs.append((k, *step0.time_graph(graphs[k])))
            torch.cuda.synchronize()
            for k, a, b in evs:
                times[k].append(a.elapsed_time(b) * 1e3 / K_CALLS)
            if r % 12 == 0:
                log(f"  S={S} round {r}: " + " ".join(f"{k}={times[k][-1]:.1f}" for k in keys))
        row = {}
        for k in keys:
            med, noise = step0.stat(times[k])
            row[k] = {"us_per_call": round(med, 2), "noise": round(noise, 4)}
        for unit in ("op", "gu"):
            for arm in mods:
                a, b = row[f"{unit}_{arm}_A"]["us_per_call"], row[f"{unit}_{arm}_B"]["us_per_call"]
                row[f"{unit}_{arm}"] = {"mean": round((a + b) / 2, 2), "aa_A_minus_B": round(a - b, 2)}
            row[f"{unit}_delta_dynamic_minus_static"] = round(
                row[f"{unit}_dynamic"]["mean"] - row[f"{unit}_static"]["mean"], 2)
        res[S] = row
        log(f"S={S}: " + json.dumps({k: v for k, v in row.items() if not k.endswith(("_A", "_B"))}))
        del graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"table": {str(k): v for k, v in res.items()}, "rounds": ROUNDS,
                      "calls_per_graph": K_CALLS, "counters": dict(fp.SIDE_GATE_UP_COUNTERS)}))


def pair():
    meta = info()
    swc.get_module()  # build/load the tree's selected module outside timing
    cases = [(prequant(step0.make_case(4176, 1)), 28), (prequant(step0.make_case(4096, 2)), 4)]
    pristine = [[t.clone() for t in step0.inputs(c)] for c, _ in cases]

    def step():
        outs = []
        for c, n in cases:
            for _ in range(n):
                outs.append(served_op(c)[1])
        return outs[-1]

    g, _ = step0.capture(step)
    log(f"captured; soak {PAIR_SOAK_S}s")
    t0 = time.time()
    while time.time() - t0 < PAIR_SOAK_S:
        g.replay()
        torch.cuda.synchronize()
    for (c, _), p in zip(cases, pristine):
        assert all(torch.equal(a, b) for a, b in zip(p, step0.inputs(c)))
    times = []
    for i in range(PAIR_SAMPLES):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SPIN)
        a.record()
        for _ in range(PAIR_REPLAYS):
            g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / PAIR_REPLAYS)
        if i % 40 == 0:
            log(f"  sample {i}: {times[-1]:.3f} ms")
    med, noise = step0.stat(times)
    log(f"step ops: {med:.4f} ms noise {noise:.4f}; counters {fp.SIDE_GATE_UP_COUNTERS}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "unit": "one graph replay = 28 x S4176 + 4 x S4096 calls of the served mode-3 op "
                "trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up (fmha_v2 || gate_up)",
        "sched": meta["sched"], "head": meta["head"], "ancestry_rc": meta["ancestry_rc"],
        "samples": PAIR_SAMPLES, "replays_per_sample": PAIR_REPLAYS, "soak_s": PAIR_SOAK_S,
        "output": {"class": "exact"},
    }))


def _gate_up_case(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    H, FF = step0.H, step0.FF
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(2 * FF, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    s_x, s_w = step0._gs(x), step0._gs(w)
    a, sfa = fp.nvfp4_quantize(x, step0._inv(s_x))
    b, sfb = fp.nvfp4_quantize(w, step0._inv(s_w))
    alpha = torch.tensor([s_x * s_w], device="cuda", dtype=torch.float32)
    inv = torch.tensor([1.0 / 0.05], device="cuda", dtype=torch.float32)
    return {"M": M, "a": a, "b": b, "sfa": sfa, "sfb": sfb, "alpha": alpha, "inv": inv,
            "s_x": s_x, "s_w": s_w}


def _gu(c, module, out=None):
    o, s = out if out is not None else swc.allocate_outputs(c["M"], c["b"].shape[0], c["a"].device)
    o.fill_(0xFF)
    s.fill_(0xFF)
    swc.swiglu_packed(c["a"], c["b"], c["sfa"], c["sfb"], c["alpha"], c["inv"], module=module,
                      output=o, output_scale=s)
    return o, s


def check():
    meta = info()
    mods = modules()
    assert "dynamic" in mods, "check runs in the candidate tree"
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4 as cute

    results, passed = {}, True
    WARM, ITERS, REPLAYS = 2, 5, 3
    for M, seed in ((4096, 21), (4176, 22), (4256, 23)):
        c = _gate_up_case(M, seed)
        keys = ("a", "b", "sfa", "sfb", "alpha", "inv")
        pristine = {k: c[k].clone() for k in keys}
        ref = [t.clone() for t in _gu(c, mods["static"])]
        rows = M
        # CuTe run('packed') reference (unmodified original kernel), compared on the valid rows.
        try:
            cp, cs = cute.run("packed", c["a"], c["b"], c["sfa"], c["sfb"], c["s_x"], c["s_w"],
                              output_inverse_scale=c["inv"])
            torch.cuda.synchronize()
            cute_out = torch.equal(cp[:rows].reshape(-1), ref[0][:rows].reshape(-1))
            cute_sf = torch.equal(cs.reshape(-1)[: ref[1].numel()], ref[1].reshape(-1)[: cs.numel()])
        except Exception as e:  # noqa: BLE001
            cute_out = cute_sf = f"n/a: {type(e).__name__}: {str(e)[:120]}"
        row = {"static_vs_cute_out": cute_out, "static_vs_cute_sf": cute_sf}
        for arm, mod in mods.items():
            for k in keys:
                c[k].copy_(pristine[k])
            for _ in range(WARM):
                _gu(c, mod)
            first, n_ref, n_first = None, 0, 0
            for i in range(ITERS):
                out = [t.clone() for t in _gu(c, mod)]
                torch.cuda.synchronize()
                if first is None:
                    first = out
                n_ref += all(torch.equal(x, y) for x, y in zip(out, ref))
                n_first += all(torch.equal(x, y) for x, y in zip(out, first))
            inputs_ok = all(torch.equal(pristine[k], c[k]) for k in keys)
            bufs = swc.allocate_outputs(M, c["b"].shape[0], c["a"].device)
            _gu(c, mod, bufs)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            if not step0._CAPTURE_STREAM:
                step0._CAPTURE_STREAM.append(torch.cuda.Stream())
            with torch.cuda.graph(g, stream=step0._CAPTURE_STREAM[0], capture_error_mode="thread_local"):
                swc.swiglu_packed(c["a"], c["b"], c["sfa"], c["sfb"], c["alpha"], c["inv"],
                                  module=mod, output=bufs[0], output_scale=bufs[1])
            g_ok = 0
            for _ in range(REPLAYS):
                bufs[0].fill_(0xFF)
                bufs[1].fill_(0xFF)
                g.replay()
                torch.cuda.synchronize()
                g_ok += torch.equal(bufs[0], ref[0]) and torch.equal(bufs[1], ref[1])
            del g
            inputs_ok &= all(torch.equal(pristine[k], c[k]) for k in keys)
            ok = n_ref == ITERS and n_first == ITERS and g_ok == REPLAYS and inputs_ok
            passed &= ok
            row[arm] = {"bitwise_vs_static_call0": f"{n_ref}/{ITERS}",
                        "bitwise_vs_own_call0": f"{n_first}/{ITERS}",
                        "graph_replays_bitwise": f"{g_ok}/{REPLAYS}", "inputs_unchanged": inputs_ok,
                        "ok": ok}
        results[f"gate_up_M{M}"] = row
        log(f"M={M}: {json.dumps(row)}")

    # Served op (mode 3, side stream) with the dynamic module vs main's sequential static ops.
    for S, seed in ((4176, 11), (4096, 12)):
        c = prequant(step0.make_case(S, seed))
        pristine = [t.clone() for t in step0.inputs(c)]
        use(mods["static"])
        ref = [t.clone() for t in sequential(c)]
        torch.cuda.synchronize()
        use(mods["dynamic"])
        before = dict(fp.SIDE_GATE_UP_COUNTERS)
        for _ in range(2):
            served_op(c)
        n_ref = 0
        for _ in range(5):
            out = served_op(c)
            torch.cuda.synchronize()
            n_ref += all(torch.equal(x, y) for x, y in zip(out, ref))
        g, gout = step0.capture(lambda: served_op(c))
        g_ok = 0
        for _ in range(3):
            for t in gout:
                t.zero_()
            g.replay()
            torch.cuda.synchronize()
            g_ok += all(torch.equal(x, y) for x, y in zip(gout, ref))
        del g
        use(None)
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, step0.inputs(c)))
        after = dict(fp.SIDE_GATE_UP_COUNTERS)
        d = {k: after[k] - before[k] for k in after}
        ok = n_ref == 5 and g_ok == 3 and inputs_ok and d["issued"] == d["joined"] > 0 \
            and d["sync_fallback"] == 0
        passed &= ok
        results[f"served_op_S{S}"] = {"eager_bitwise_vs_sequential_static": f"{n_ref}/5",
                                      "graph_replays_bitwise": f"{g_ok}/3",
                                      "inputs_unchanged": inputs_ok, "counters_delta": d, "ok": ok}
        log(f"served S={S}: {json.dumps(results[f'served_op_S{S}'])}")

    # Env selection: static -> main's module name/source; bogus raises; default dynamic.
    env_rows = {}
    for val in ("static", "dynamic", None, "bogus"):
        env = dict(os.environ)
        env.pop("TRTLLM_FLUX3_SM120_SWIGLU_SCHED", None)
        if val is not None:
            env["TRTLLM_FLUX3_SM120_SWIGLU_SCHED"] = val
        code = ("import json,tensorrt_llm\n"
                "from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as s\n"
                "try:\n m=s.get_module(); print(json.dumps({'sched': s.tile_sched(), "
                "'is_static': m is s.get_module_static(), 'name': getattr(m,'__name__',None) or str(m)[:120]}))\n"
                "except ValueError as e:\n print(json.dumps({'error': str(e)[:120]}))\n")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or ["{}"])[-1]
        env_rows[str(val)] = json.loads(line) if line.startswith("{") else {"raw": r.stderr[-300:]}
        log(f"env {val}: {env_rows[str(val)]}")
    env_ok = (env_rows["static"].get("is_static") is True
              and env_rows["dynamic"].get("is_static") is False
              and env_rows["None"].get("sched") == "dynamic"
              and "error" in env_rows["bogus"])
    passed &= env_ok
    results["env_selection"] = {"rows": env_rows, "ok": env_ok}
    print(json.dumps({
        "passed": bool(passed),
        "reference": "gate_up: main's static-scheduler module (same .cu at FLUX3_TILESCHED=0) on "
                     "pristine inputs, whole padded out/out_sf buffers after 0xFF prefill; also the "
                     "CuTe run('packed') kernel on valid rows. served op: main's sequential "
                     "visual_gen_fp8_attention_sm120_prequant + flux3_nvfp4_swiglu_prepared (static)",
        "head": meta["head"], "ancestry_rc": meta["ancestry_rc"],
        "results": results, "counters": dict(fp.SIDE_GATE_UP_COUNTERS),
    }))


def grid():
    info()
    mods = modules()
    for S, seed in ((4176, 1), (4096, 2)):
        c = prequant(step0.make_case(S, seed))
        for arm, mod in mods.items():
            use(mod)
            for _ in range(3):
                gate_up_only(c)
                served_op(c)
            torch.cuda.synchronize()
            log(f"S={S} {arm}: done")
    use(None)
    print(json.dumps({"done": True}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    {"info": info, "table": table, "pair": pair, "check": check, "grid": grid}[cmd]()
