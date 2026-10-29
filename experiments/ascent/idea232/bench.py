"""Idea 232: gate_up SwiGLU->NVFP4 epilogue with one 16-B FP4 store + one u16 SF store per lane row.

FLUX3_EPI_VEC=1 (flux3_fp4_gemm_swiglu_fp4_sm120.cu) changes the TiledMma N permutation to
(8,2,8):(1,64,8) so each warp owns two ADJACENT scale blocks, and the epilogue merges the two
8-B stores (and two SF bytes) of a lane. Env TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC=0 = main's modules.

Synthetic tensors at served shapes (gate_up N=18432, K=3072; S=4176 / 4096), no model load.
  info      : HEAD, main, ancestry, tensorrt_llm.__file__, selector state, grep hits.
  resources : SASS STL/LDL counts and cuobjdump resource usage of the swiglu modules (main
              static/dynamic and EPI_VEC static/dynamic), ncu --import of report 1654 local sectors.
  check     : bitwise EPI_VEC vs main per scheduler (0xFF prefill, M=4096/4176/4256, warmup +
              iterations vs call 0, inputs unchanged, graph capture + 3 replays); served op and the
              whole block front with the flag on vs off; env selection.
  table     : (candidate tree) graph replay A/B with A/A: isolated gate_up and served mode-3 op,
              main dynamic vs EPI_VEC dynamic, interleaved rounds.
  pair      : evidence unit = one graph of 28 x S4176 + 4 x S4096 served mode-3 op calls with the
              tree's default module (main: dynamic; branch: dynamic EPI_VEC).
  ncu       : one call of the tree's default gate_up module inside a profiler region.
"""

import glob
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

MAIN = "1185e8f9"
HEADS, D = step0.HEADS, step0.D
K_CALLS = int(os.environ.get("IDEA232_K", "10"))
ROUNDS = int(os.environ.get("IDEA232_ROUNDS", "100"))
SPIN = 6_000_000
PAIR_SAMPLES = 200
PAIR_SOAK_S = 20.0
PAIR_REPLAYS = 3
REPORT_1654 = ("/home/scratch.chengzhang/trtllm-dev-flux3-klein-20261027-125409/runner/slots/2/jobs/"
               "f4d88e022ffa4a78903d55630aa7acd4/artifacts/kernels.ncu-repz")


def log(*a):
    print(*a, flush=True)


def wt():
    return os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def git(*args):
    r = subprocess.run(["git", "-C", wt(), *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def info():
    _, head = git("rev-parse", "HEAD")
    _, main_full = git("rev-parse", MAIN)
    rc, _ = git("merge-base", "--is-ancestor", MAIN, "HEAD")
    has_vec = hasattr(swc, "epi_vec")
    vec = swc.epi_vec() if has_vec else "n/a(main)"
    _, grep = git("grep", "-n", "-E", "FLUX3_TILESCHED|TRTLLM_FLUX3_SM120_SWIGLU_SCHED|FLUX3_EPI_VEC",
                  "--", "tensorrt_llm", "experiments/ascent-flashinfer")
    log(f"HEAD={head} main={MAIN}->{main_full} is_ancestor rc={rc} "
        f"tensorrt_llm.__file__={tensorrt_llm.__file__} sched={swc.tile_sched()} epi_vec={vec}")
    log("grep:\n" + grep)
    return {"head": head, "ancestry_rc": rc, "sched": swc.tile_sched(), "epi_vec": vec}


def modules():
    """{arm: module}: main_static/main_dynamic (EPI_VEC=0) and vec_static/vec_dynamic."""
    if hasattr(swc, "epi_vec"):
        return {"main_static": swc.get_module_static(False),
                "main_dynamic": swc.get_module_dynamic(False),
                "vec_static": swc.get_module_static(True),
                "vec_dynamic": swc.get_module_dynamic(True)}
    return {"main_static": swc.get_module_static(), "main_dynamic": swc.get_module_dynamic()}


_ORIG_GET_MODULE = swc.get_module


def use(module):
    swc.get_module = (lambda: module) if module is not None else _ORIG_GET_MODULE


def prequant(c):
    qkv8, scales, _ = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope_fp8(
        c["a_x"], c["qkv"].weight, c["sf_x"], c["qkv"].scale, c["q_norm"], c["k_norm"],
        c["cos_sin"], c["qkv"].qkv_scale, c["x"], c["qkv"].weight, 1.0 / math.sqrt(D),
    )
    up = c["up"]
    c["alpha"] = torch.tensor([up.input_global_scale * up.weight_global_scale], device="cuda",
                              dtype=torch.float32)
    torch.cuda.synchronize()
    c["qkv8"], c["scales"] = qkv8, scales
    return c


def served_op(c, mode=3):
    up, down = c["up"], c["down"]
    return torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_side_gate_up(
        c["qkv8"], c["scales"], 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale,
        down.inverse_scale, c["x"], c["qkv"].weight, mode,
    )


def gate_up_only(c):
    up, down = c["up"], c["down"]
    return swc.swiglu_packed(c["a_x"], up.weight, c["sf_x"], up.scale, c["alpha"],
                             down.inverse_scale)


# ------------------------------------------------------------------------------------- resources
def _lib_path(spec):
    for attr in ("jit_library_path", "library_path"):
        p = getattr(spec, attr, None)
        if p is not None:
            return str(p)
    from flashinfer.jit import env as jit_env
    hits = glob.glob(str(jit_env.FLASHINFER_JIT_DIR / spec.name / "*.so"))
    return hits[0] if hits else None


def resources():
    info()
    modules()  # build all
    variants = {"main_static": ("", ()), "main_dynamic": ("_dyn", ("-DFLUX3_TILESCHED=1",))}
    if hasattr(swc, "epi_vec"):
        variants["vec_static"] = ("_ev", ("-DFLUX3_EPI_VEC=1",))
        variants["vec_dynamic"] = ("_dyn_ev", ("-DFLUX3_TILESCHED=1", "-DFLUX3_EPI_VEC=1"))
    cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    rows = {}
    for arm, (suffix, flags) in variants.items():
        spec = swc.gen_spec(suffix, flags)
        so = _lib_path(spec)
        row = {"so": so}
        if so and os.path.exists(so):
            r = subprocess.run([cuobjdump, "--dump-resource-usage", so], capture_output=True,
                               text=True)
            lines = [ln.strip() for ln in r.stdout.splitlines()]
            res = []
            for i, ln in enumerate(lines):
                if "device_kernel" in ln and i + 1 < len(lines):
                    res.append(lines[i + 1][:200])
            row["resource_usage_device_kernel"] = res
            s = subprocess.run([cuobjdump, "-sass", so], capture_output=True, text=True).stdout
            # Count local ld/st only inside the GEMM kernel function.
            cur, counts = None, {}
            for ln in s.splitlines():
                if "Function :" in ln:
                    cur = ln.split("Function :")[1].strip()
                    counts[cur] = {"STL": 0, "LDL": 0, "STG": 0, "STG.E.128": 0, "STG.E.64": 0,
                                   "STG.E.U16": 0, "STG.E.U8": 0}
                elif cur is not None:
                    for op in ("STL", "LDL"):
                        if f" {op}" in ln:
                            counts[cur][op] += 1
                    if " STG" in ln:
                        counts[cur]["STG"] += 1
                        for w in ("STG.E.128", "STG.E.64", "STG.E.U16", "STG.E.U8"):
                            if w in ln:
                                counts[cur][w] += 1
            row["sass"] = {k[:60]: v for k, v in counts.items() if "device_kernel" in k}
        rows[arm] = row
        log(f"{arm}: {json.dumps(row)}")
    ncu_rows = {}
    if os.path.exists(REPORT_1654):
        metrics = ("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,"
                   "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sectors_srcunit_tex.sum,"
                   "lts__t_sectors_srcunit_tex_op_write.sum,launch__registers_per_thread,"
                   "gpu__time_duration.sum")
        r = subprocess.run(["/usr/local/bin/ncu", "--import", REPORT_1654, "--page", "raw", "--csv",
                            "--metrics", metrics], capture_output=True, text=True)
        ncu_rows = {"rc": r.returncode, "csv": r.stdout[-6000:], "err": r.stderr[-800:]}
        log("ncu import 1654:\n" + r.stdout[-6000:] + r.stderr[-800:])
    else:
        ncu_rows = {"missing": REPORT_1654}
        log(f"report 1654 not reachable: {REPORT_1654}")
    print(json.dumps({"resources": rows, "ncu_1654": ncu_rows}))


# ----------------------------------------------------------------------------------------- check
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
    assert "vec_dynamic" in mods, "check runs in the candidate tree"
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4 as cute

    results, passed = {}, True
    WARM, ITERS, REPLAYS = 2, 5, 3
    keys = ("a", "b", "sfa", "sfb", "alpha", "inv")
    for M, seed in ((4096, 21), (4176, 22), (4256, 23)):
        c = _gate_up_case(M, seed)
        pristine = {k: c[k].clone() for k in keys}
        refs = {s: [t.clone() for t in _gu(c, mods[f"main_{s}"])] for s in ("static", "dynamic")}
        torch.cuda.synchronize()
        row = {"main_static_vs_main_dynamic": all(torch.equal(x, y) for x, y in
                                                   zip(refs["static"], refs["dynamic"]))}
        ref = refs["static"]
        try:
            cp, cs = cute.run("packed", c["a"], c["b"], c["sfa"], c["sfb"], c["s_x"], c["s_w"],
                              output_inverse_scale=c["inv"])
            torch.cuda.synchronize()
            row["main_vs_cute_out"] = torch.equal(cp[:M].reshape(-1), ref[0][:M].reshape(-1))
        except Exception as e:  # noqa: BLE001
            row["main_vs_cute_out"] = f"n/a: {type(e).__name__}: {str(e)[:120]}"
        # Padding rows (>= M, < rows_padded) are written by the kernel as on main; compare whole
        # buffers incl. the 0xFF-prefilled region that neither writes.
        for arm in ("vec_static", "vec_dynamic", "main_static", "main_dynamic"):
            mod = mods[arm]
            sref = refs[arm.split("_")[1]]
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
                n_ref += all(torch.equal(x, y) for x, y in zip(out, sref))
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
                g_ok += torch.equal(bufs[0], sref[0]) and torch.equal(bufs[1], sref[1])
            del g
            inputs_ok &= all(torch.equal(pristine[k], c[k]) for k in keys)
            ok = n_ref == ITERS and n_first == ITERS and g_ok == REPLAYS and inputs_ok
            passed &= ok
            row[arm] = {"bitwise_vs_main_same_sched": f"{n_ref}/{ITERS}",
                        "bitwise_vs_own_call0": f"{n_first}/{ITERS}",
                        "graph_replays_bitwise": f"{g_ok}/{REPLAYS}", "inputs_unchanged": inputs_ok,
                        "ok": ok}
        passed &= row["main_static_vs_main_dynamic"] is True
        results[f"gate_up_M{M}"] = row
        log(f"M={M}: {json.dumps(row)}")

    # Served op (mode 3) and the whole block front (QKV, FMHA || gate_up, out pair, residual):
    # EPI_VEC dynamic vs main dynamic in one process.
    for S, seed in ((4176, 11), (4096, 12)):
        c = prequant(step0.make_case(S, seed))
        pristine = [t.clone() for t in step0.inputs(c)]
        use(mods["main_dynamic"])
        ref = [t.clone() for t in served_op(c)]
        ref_bf = [t.clone() for t in step0.block_front(c, "m3")]
        torch.cuda.synchronize()
        use(mods["vec_dynamic"])
        before = dict(fp.SIDE_GATE_UP_COUNTERS)
        for _ in range(2):
            served_op(c)
        n_ref = n_first = 0
        first = None
        for _ in range(5):
            out = [t.clone() for t in served_op(c)]
            torch.cuda.synchronize()
            if first is None:
                first = out
            n_ref += all(torch.equal(x, y) for x, y in zip(out, ref))
            n_first += all(torch.equal(x, y) for x, y in zip(out, first))
        g, gout = step0.capture(lambda: served_op(c))
        g_ok = 0
        for _ in range(3):
            for t in gout:
                t.zero_()
            g.replay()
            torch.cuda.synchronize()
            g_ok += all(torch.equal(x, y) for x, y in zip(gout, ref))
        del g
        bf_ok = 0
        for _ in range(3):
            out = step0.block_front(c, "m3")
            torch.cuda.synchronize()
            bf_ok += all(torch.equal(x, y) for x, y in zip(out, ref_bf))
        use(None)
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, step0.inputs(c)))
        after = dict(fp.SIDE_GATE_UP_COUNTERS)
        d = {k: after[k] - before[k] for k in after}
        ok = (n_ref == 5 and n_first == 5 and g_ok == 3 and bf_ok == 3 and inputs_ok
              and d["issued"] == d["joined"] > 0 and d["sync_fallback"] == 0)
        passed &= ok
        results[f"served_op_S{S}"] = {"eager_bitwise_vs_main": f"{n_ref}/5",
                                      "eager_vs_call0": f"{n_first}/5",
                                      "graph_replays_bitwise": f"{g_ok}/3",
                                      "block_front_vec_vs_main_bitwise": f"{bf_ok}/3",
                                      "inputs_unchanged": inputs_ok, "counters_delta": d, "ok": ok}
        log(f"served S={S}: {json.dumps(results[f'served_op_S{S}'])}")

    # Env selection: EPI_VEC=0 -> main's module names (the same .cu preprocesses to main's code).
    env_rows = {}
    for vec in ("0", "1", None, "bogus"):
        for sched in ("dynamic", "static"):
            env = dict(os.environ)
            env.pop("TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC", None)
            if vec is not None:
                env["TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC"] = vec
            env["TRTLLM_FLUX3_SM120_SWIGLU_SCHED"] = sched
            code = ("import json,tensorrt_llm\n"
                    "from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as s\n"
                    "try:\n m=s.get_module()\n names={n: s._build(*a) is m for n,a in "
                    "{'main_static':('',()),'main_dynamic':('_dyn',('-DFLUX3_TILESCHED=1',)),"
                    "'vec_static':('_ev',('-DFLUX3_EPI_VEC=1',)),'vec_dynamic':('_dyn_ev',"
                    "('-DFLUX3_TILESCHED=1','-DFLUX3_EPI_VEC=1'))}.items()}\n"
                    " print(json.dumps({'epi_vec': s.epi_vec(), 'sched': s.tile_sched(), "
                    "'module': [n for n,v in names.items() if v]}))\n"
                    "except ValueError as e:\n print(json.dumps({'error': str(e)[:120]}))\n")
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or ["{}"])[-1]
            key = f"vec={vec},sched={sched}"
            env_rows[key] = json.loads(line) if line.startswith("{") else {"raw": r.stderr[-300:]}
            log(f"env {key}: {env_rows[key]}")
    env_ok = (env_rows["vec=0,sched=dynamic"].get("module") == ["main_dynamic"]
              and env_rows["vec=0,sched=static"].get("module") == ["main_static"]
              and env_rows["vec=1,sched=dynamic"].get("module") == ["vec_dynamic"]
              and env_rows["vec=1,sched=static"].get("module") == ["vec_static"]
              and env_rows["vec=None,sched=dynamic"].get("module") == ["vec_dynamic"]
              and "error" in env_rows["vec=bogus,sched=dynamic"])
    passed &= env_ok
    results["env_selection"] = {"rows": env_rows, "ok": env_ok}
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's modules (EPI_VEC=0, same .cu with the macro off, module names "
                     "fp4_gemm_swiglu_fp4_sm120 / _dyn) on pristine inputs, whole padded "
                     "out/out_sf after 0xFF prefill, per scheduler; main also vs CuTe "
                     "run('packed'); served op + block front (m3) vs main dynamic in-process",
        "head": meta["head"], "ancestry_rc": meta["ancestry_rc"],
        "results": results, "counters": dict(fp.SIDE_GATE_UP_COUNTERS),
    }))


# ------------------------------------------------------------------------------------- timing
def table():
    info()
    mods = modules()
    assert "vec_dynamic" in mods, "table runs in the candidate tree"
    arms = {"main": mods["main_dynamic"], "vec": mods["vec_dynamic"]}
    res = {}
    only = os.environ.get("IDEA232_S")
    for S, seed in ((4176, 1), (4096, 2)):
        if only and int(only) != S:
            continue
        c = prequant(step0.make_case(S, seed))
        pristine = [t.clone() for t in step0.inputs(c)]
        graphs = {}
        for unit, fn in (("op", served_op), ("gu", gate_up_only)):
            for arm, mod in arms.items():
                use(mod)
                for rep in ("A", "B"):
                    graphs[f"{unit}_{arm}_{rep}"], _ = step0.capture(
                        lambda: [fn(c)[0] for _ in range(K_CALLS)])
        use(None)
        for p, t in zip(pristine, step0.inputs(c)):
            t.copy_(p)
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
            if r % 25 == 0:
                log(f"  S={S} round {r}: " + " ".join(f"{k}={times[k][-1]:.1f}" for k in keys))
        assert all(torch.equal(p, t) for p, t in zip(pristine, step0.inputs(c)))
        row = {}
        for k in keys:
            v = sorted(times[k])
            med, noise = step0.stat(v)
            row[k] = {"median": round(med, 2), "p10": round(v[len(v) // 10], 2),
                      "p90": round(v[(len(v) * 9) // 10], 2), "noise": round(noise, 4)}
        for unit in ("op", "gu"):
            for arm in arms:
                a, b = row[f"{unit}_{arm}_A"]["median"], row[f"{unit}_{arm}_B"]["median"]
                row[f"{unit}_{arm}"] = {"mean": round((a + b) / 2, 2), "aa_A_minus_B": round(a - b, 2)}
            row[f"{unit}_delta_vec_minus_main"] = round(
                row[f"{unit}_vec"]["mean"] - row[f"{unit}_main"]["mean"], 2)
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
    before = dict(fp.SIDE_GATE_UP_COUNTERS)

    def step():
        outs = []
        for c, n in cases:
            for _ in range(n):
                outs.append(served_op(c)[1])
        return outs[-1]

    g, _ = step0.capture(step)
    for (c, _), p in zip(cases, pristine):
        for a, b in zip(p, step0.inputs(c)):
            b.copy_(a)
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
    after = dict(fp.SIDE_GATE_UP_COUNTERS)
    d = {k: after[k] - before[k] for k in after}
    log(f"step ops: {med:.4f} ms noise {noise:.4f}; counters delta {d}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "unit": "one graph replay = 28 x S4176 + 4 x S4096 calls of the served mode-3 op "
                "trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up (fmha_v2 || gate_up)",
        "sched": meta["sched"], "epi_vec": meta["epi_vec"], "head": meta["head"],
        "ancestry_rc": meta["ancestry_rc"], "counters_delta_capture": d,
        "samples": PAIR_SAMPLES, "replays_per_sample": PAIR_REPLAYS, "soak_s": PAIR_SOAK_S,
        "output": {"class": "exact"},
    }))


def ncu():
    info()
    mod = swc.get_module()
    c = prequant(step0.make_case(4176, 1))
    for _ in range(3):
        gate_up_only(c)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    gate_up_only(c)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({"done": True, "module": str(mod)[:100]}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    {"info": info, "resources": resources, "check": check, "table": table, "pair": pair,
     "ncu": ncu}[cmd]()
