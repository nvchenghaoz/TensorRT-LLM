"""Idea 233 acceptance item 4, 1-GPU form (evaluator request on commit fa3db136): Kineto route audit of the production
WO projection through the module stand-in, plus the csa2 module tests with the route on and off.

1. NL module stand-ins (experiments/ascent/b12x_wo/harness.py: Hn.make_module, Hn.project = the real
   DeepseekV41Attention._project_output) at TP4 rank shapes. Per M in (1, 4): one eager call per layer (builds the
   projection and its prewarm), then _project_output for all NL layers captured in ONE CUDA graph. During capture,
   every Triton compile, CuTe DSL compile and torch cpp_extension load is counted (wrapped entry points, plus Triton
   JIT cache growth) and must be 0.
2. WARMUP graph replays, then PROF replays under torch.profiler (Kineto, CUDA activities). Reset point before every
   replay: the attention-output buffers are restored from a pristine copy. From the chrome trace, per replay and per
   layer: the route kernels must be exactly [csa2_lowm_wo_a_kernel, csa2_mxfp8_low_m_gemm_kernel] per layer with
   nothing between them inside a replay (no _inv_rope_quant128_kernel, no b12x / CuTe / Triton kernel anywhere), and
   K3' start < K2' end (PDL overlap); overlap_us = K2'.end - K3'.start per layer (median over replays) and its min.
   Every replay's outputs are compared bitwise with the eager outputs.
3. tests/unittest/_torch/attention/sparse/csa2/test_module.py with TRTLLM_CSA2_LOWM_WO unset and =0: the only allowed
   failures are the 2 pre-existing test_module_sends_whole_context_and_generation_phases[trtllm|flashinfer]
   (knowledge 2200, fail on unmodified main).
Usage: kineto_audit.py <ckpt> [--layers 4] [--warmup 5] [--prof 5] [--no-tests]. Last stdout line: JSON. Not timing.
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b12x_wo"))
import common as C  # noqa: E402  (sets TRTLLM_ENABLE_PDL=1)
import harness as Hn  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--prof", type=int, default=5)
ap.add_argument("--batches", default="1,4")
ap.add_argument("--no-tests", action="store_true")
args = ap.parse_args()
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/lowm_wo_audit"))
ART.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[3]

K2 = "csa2_lowm_wo_a_kernel"
K3 = "csa2_mxfp8_low_m_gemm_kernel"
BAD = re.compile(r"inv_rope_quant128|b12x|Sm120B12x|cute|triton|wo_a_dense|wo_b_dense", re.I)
failures = []


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


# ---------------------------------------------------------------- compile / JIT counters
COMPILES = {"count": 0, "names": []}
_capturing = {"on": False}


def _wrap(fn, label):
    if getattr(fn, "_audit_wrapped", False):
        return fn

    def w(*a, **k):
        COMPILES["count"] += 1
        COMPILES["names"].append(label + (" (in capture)" if _capturing["on"] else ""))
        return fn(*a, **k)

    w._audit_wrapped = True
    return w


def install_compile_hooks():
    targets = []
    try:
        import triton.compiler.compiler as tcc
        targets.append(("triton.compile", tcc.compile))
    except Exception as e:  # noqa: BLE001
        log("triton hook unavailable", repr(e))
    try:
        import cutlass.cute as cute
        targets.append(("cute.compile", cute.compile))
    except Exception as e:  # noqa: BLE001
        log("cute hook unavailable", repr(e))
    try:
        import torch.utils.cpp_extension as ce
        targets.append(("cpp_extension.load", ce.load))
        targets.append(("cpp_extension.load_inline", ce.load_inline))
    except Exception as e:  # noqa: BLE001
        log("cpp_extension hook unavailable", repr(e))
    origs = {id(f): (lbl, f) for lbl, f in targets}
    hooked = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(("triton", "cutlass", "torch.utils.cpp_extension", "b12x",
                                                "flashinfer", "tensorrt_llm")):
            continue
        for attr in ("compile", "load", "load_inline"):
            try:
                f = getattr(mod, attr, None)
            except Exception:  # noqa: BLE001
                continue
            if f is not None and id(f) in origs:
                try:
                    setattr(mod, attr, _wrap(f, origs[id(f)][0]))
                    hooked.append(f"{name}.{attr}")
                except Exception:  # noqa: BLE001
                    pass
    return hooked


def triton_cache_size():
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001
        return None
    import gc
    n = 0
    for o in gc.get_objects():
        try:
            if isinstance(o, JITFunction):
                for c in getattr(o, "device_caches", {}).values():
                    n += len(c[0])
        except Exception:  # noqa: BLE001
            pass
    return n


# ---------------------------------------------------------------- build
dims = C.dims_from_ckpt(args.ckpt)
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(args.layers - 1)]
mods = [Hn.make_module(dims, w, cs) for w in ws]
log(f"layers={args.layers} dims={dims} TRTLLM_CSA2_LOWM_WO={os.environ.get('TRTLLM_CSA2_LOWM_WO')} "
    f"TRTLLM_ENABLE_PDL={os.environ.get('TRTLLM_ENABLE_PDL')}")

summary = {"layers": args.layers, "warmup": args.warmup, "prof_replays": args.prof, "batches": {}}
for M in [int(b) for b in args.batches.split(",")]:
    o3, pos = C.make_inputs(dims, M, seed=200 + M)
    pristine = o3.view(M, -1).clone()
    bufs = [pristine.clone() for _ in range(args.layers)]
    eager = [Hn.project(mods[i], bufs[i], pos).clone() for i in range(args.layers)]
    torch.cuda.synchronize()
    impl = type(mods[0]._b12x_wo).__name__
    path = Hn.path_name(mods[0])
    log(f"M={M} impl={impl} path={path}")
    if impl != "LowMWOProjection":
        fail(f"M={M}: projection impl is {impl}, not LowMWOProjection")
    hooked = install_compile_hooks()
    for b in bufs:
        b.copy_(pristine)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        Hn.project(mods[0], bufs[0], pos)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for b in bufs:
        b.copy_(pristine)
    tc0 = triton_cache_size()
    c0 = COMPILES["count"]
    g = torch.cuda.CUDAGraph()
    t0 = time.time()
    _capturing["on"] = True
    try:
        with torch.cuda.graph(g):
            outs = [Hn.project(mods[i], bufs[i], pos) for i in range(args.layers)]
    finally:
        _capturing["on"] = False
    cap_s = time.time() - t0
    tc1 = triton_cache_size()
    cap = {"capture_s": round(cap_s, 3), "compile_calls_in_capture": COMPILES["count"] - c0,
           "compile_names": COMPILES["names"][-10:], "triton_cache_before_after": [tc0, tc1],
           "hooked": hooked[:20]}
    if cap["compile_calls_in_capture"] or (tc0 is not None and tc1 != tc0):
        fail(f"M={M}: compile/JIT during capture {cap}")
    log(f"M={M} capture {cap}")

    def replay():
        for b in bufs:  # reset point
            b.copy_(pristine)
        g.replay()

    bad_replays = 0
    for _ in range(args.warmup):
        replay()
        torch.cuda.synchronize()
        bad_replays += sum(int(not torch.equal(o, e)) for o, e in zip(outs, eager))
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for r in range(args.prof):
            torch.cuda.synchronize()
            with torch.profiler.record_function(f"audit_replay_{r}"):
                replay()
                torch.cuda.synchronize()
    for o, e in zip(outs, eager):
        bad_replays += int(not torch.equal(o, e))
    trace = ART / f"kineto_M{M}.json"
    prof.export_chrome_trace(str(trace))
    with open(trace) as fh:
        data = json.load(fh)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    kern = sorted([(float(e["ts"]), float(e.get("dur", 0)), e.get("name", "")) for e in evs
                   if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"])
    route = [k for k in kern if K2 in k[2] or K3 in k[2]]
    other = [k for k in kern if not (K2 in k[2] or K3 in k[2])]
    other_names = {}
    for _, _, n in other:
        other_names[n[:100]] = other_names.get(n[:100], 0) + 1
    bad_kernels = {n: c for n, c in other_names.items() if BAD.search(n)}
    if bad_kernels:
        fail(f"M={M}: forbidden kernels in trace {bad_kernels}")
    L = args.layers
    expect = 2 * L * args.prof
    seq_ok = len(route) == expect and all(
        (K2 in route[i][2]) == (i % 2 == 0) and (K3 in route[i][2]) == (i % 2 == 1) for i in range(len(route)))
    interleaved = 0
    for ts, dur, n in other:
        for j in range(len(route) - 1):
            if j % (2 * L) == 2 * L - 1:
                continue  # replay boundary: the reset copies live here
            if route[j][0] < ts < route[j + 1][0]:
                interleaved += 1
                break
    per_layer = [[] for _ in range(L)]
    per_layer_gap = [[] for _ in range(L)]
    if seq_ok:
        for r in range(args.prof):
            for li in range(L):
                k2 = route[(r * L + li) * 2]
                k3 = route[(r * L + li) * 2 + 1]
                per_layer[li].append(round(k2[0] + k2[1] - k3[0], 3))  # > 0: K3' started before K2' ended
                per_layer_gap[li].append(round(k3[0] - k2[0], 3))
    else:
        fail(f"M={M}: route kernel sequence wrong: {len(route)} route kernels (expected {expect}); "
             f"first names {[k[2][:40] for k in route[:6]]}")
    if interleaved:
        fail(f"M={M}: {interleaved} other kernels inside the K2'/K3' layer chain")
    ov = {f"layer{li}": {"overlap_us_median": st.median(v), "overlap_us_min": min(v),
                         "k3_start_minus_k2_start_us_median": st.median(per_layer_gap[li])}
          for li, v in enumerate(per_layer) if v}
    all_ov = [x for v in per_layer for x in v]
    n_overlap = sum(int(x > 0) for x in all_ov)
    if not all_ov or n_overlap != len(all_ov):
        fail(f"M={M}: K3' start before K2' end in only {n_overlap}/{len(all_ov)} layer instances")
    if bad_replays:
        fail(f"M={M}: {bad_replays} graph outputs differ bitwise from eager")
    durs = {}
    for ts, dur, n in route:
        durs.setdefault("K2'" if K2 in n else "K3'", []).append(dur)
    res = {"impl": impl, "path": path, "capture": cap, "route_kernels": len(route), "expected_route_kernels": expect,
           "sequence_exact": seq_ok, "other_kernels_in_trace": other_names, "forbidden_kernels": bad_kernels,
           "kernels_interleaved_in_chain": interleaved, "k3_before_k2_end": f"{n_overlap}/{len(all_ov)}",
           "overlap_us_min_all": min(all_ov) if all_ov else None, "per_layer": ov,
           "dur_us_median": {k: st.median(v) for k, v in durs.items()},
           "route_kernel_names": sorted({k[2][:120] for k in route}),
           "graph_vs_eager_bitwise_bad": bad_replays}
    summary["batches"][str(M)] = res
    log(f"M={M} " + json.dumps(res)[:3000])
    del g, outs
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- tests, route on and off
SUITE = "tests/unittest/_torch/attention/sparse/csa2/test_module.py"
KNOWN = {f"{SUITE}::test_module_sends_whole_context_and_generation_phases[trtllm]",
         f"{SUITE}::test_module_sends_whole_context_and_generation_phases[flashinfer]"}


def run_tests(extra):
    env = dict(os.environ)
    env.pop("TRTLLM_CSA2_LOWM_WO", None)
    env.update(extra)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", SUITE]
    log(" ".join(cmd), extra)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    summ = next((ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|error)", ln)), "")
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|deselected)", summ)}
    failed = {ln.split()[1] for ln in lines if ln.startswith(("FAILED", "ERROR")) and len(ln.split()) > 1}
    print(p.stdout[-3000:], flush=True)
    unexpected = sorted(failed - KNOWN)
    ok = counts.get("passed", 0) > 0 and not unexpected and not counts.get("error") and not counts.get("errors")
    return {"ok": ok, "summary": summ.strip("= "), "counts": counts, "failed": sorted(failed),
            "unexpected_failures": unexpected}


tests = {}
if not args.no_tests:
    tests["lowm_on"] = run_tests({})
    tests["lowm_off"] = run_tests({"TRTLLM_CSA2_LOWM_WO": "0"})
    for k, v in tests.items():
        if not v["ok"]:
            fail(f"tests {k}: {v['unexpected_failures'] or v['summary']}")
summary["tests"] = tests
summary["failures"] = failures
(ART / "kineto_audit_summary.json").write_text(json.dumps(summary, indent=1))
brief = {M: {k: r[k] for k in ("impl", "sequence_exact", "forbidden_kernels", "k3_before_k2_end",
                                "overlap_us_min_all", "graph_vs_eager_bitwise_bad", "dur_us_median")}
         | {"compile_calls_in_capture": r["capture"]["compile_calls_in_capture"],
            "overlap_us_median_per_layer": [v["overlap_us_median"] for v in r["per_layer"].values()]}
         for M, r in summary["batches"].items()}
print(json.dumps({"passed": not failures,
                  "compared": "Kineto trace of graph-replayed _project_output (route kernels, PDL overlap, no compile "
                              "in capture), graph outputs bitwise vs eager; csa2 test_module with route on/off",
                  "batches": brief,
                  "tests": {k: {kk: v[kk] for kk in ("summary", "failed", "unexpected_failures")}
                            for k, v in tests.items()},
                  "failures": failures[:20]}))
