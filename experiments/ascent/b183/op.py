"""Branch 183 (idea 213): v2 persistent prefetching LN+adaLN+NVFP4 (launch_pf) vs served warp kernel.

  op.py check -> HEAD probe info, per-variant regs/local/smem/CTAs-per-SM + ptxas -v lines;
                 fp4 + full poisoned SF + forced-store x_mod BITWISE v2 vs served warp kernel at
                 M=80/4096/4176/4256, select and mode forms, normal and 4x-saturating gs, store
                 on/off, every (RPW, X_POLICY, FIRST_DIRECT) variant + MINB=3; untimed 23-iteration
                 repeat (resets every 8) vs iter 0 and reference, inputs vs pristine copies;
                 public-op dispatch with _V2 on/off bitwise.
  op.py sweep -> in-process interleaved timing: served warp vs v2 variants, one CUDA graph of 28
                 select M=4176 calls each preceded by a stand-in gated-residual addcmul (x L2-hot),
                 rotating 4 input sets; per-round medians, per-call us.
  op.py bench -> served per-step mix through the public op (28 select M=4176 + 4 mode M=4096, each
                 after a residual stand-in), same as experiments/ascent/b125/op.py bench; the arm
                 is picked by env (TRTLLM_FLUX3_ADALN_V2 ...). Same script both arms.
"""

import json
import os
import statistics
import subprocess
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk

D = 3072
EPS = 1e-6
dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
g = torch.Generator(device=dev).manual_seed(1234)


def heavy(*shape):
    base = torch.randn(*shape, device=dev, generator=g)
    mag = torch.exp(torch.randn(*shape[:-1], 1, device=dev, generator=g) * 2.0)
    chan = torch.exp(torch.randn(shape[-1], device=dev, generator=g) * 1.5)
    return (base * mag * chan).to(torch.bfloat16)


W = (1.0 + 0.3 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
B = (0.2 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)


def make_cases():
    cases = []
    for s in (80, 4096, 4176, 4256):
        x = heavy(1, s, D)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        shift_t, scale_t = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        select = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        select[:, : (40 if s == 80 else 80)] = True
        cases.append((f"select_M{s}", (x, select, scale_t, shift_t)))
        table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        row = table[:, 5:6]
        cases.append((f"mode_M{s}", (x, None, row[..., D: 2 * D], row[..., :D])))
    return cases


def gs_for(x):
    return float(torch.tensor(2688.0 / x.float().abs().max().item(), dtype=torch.float32))


def poisoned(x, byte=0xCD):
    m = x.numel() // D
    xm = torch.empty_like(x)
    xm.view(torch.uint8).fill_(byte)
    return (
        xm,
        torch.full((m, D // 2), byte, dtype=torch.uint8, device=dev),
        torch.full((faq._scale_size(m, D),), byte, dtype=torch.uint8, device=dev),
    )


def same(a, b):
    return (torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
            and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))


VARIANTS = [
    dict(rpw=r, x_policy=xp, first_direct=fd, extra_defines=(mb,))
    for mb in ("MINB=3", "MINB=4") for r in (1, 2, 3, 0) for xp in (0, 2) for fd in (1, 0)
    if mb == "MINB=3" or r == 2
]
if os.environ.get("B183_VARIANTS"):
    _want = set(os.environ["B183_VARIANTS"].split(","))
    VARIANTS = [v for v in VARIANTS if
                f"rpw{v['rpw']}_xp{v['x_policy']}_fd{v['first_direct']}_{v['extra_defines'][0]}" in _want]


def vname(v):
    return f"rpw{v['rpw']}_xp{v['x_policy']}_fd{v['first_direct']}_{v['extra_defines'][0]}"


def served(args, gs_t, store, outs=None):
    x, sel, sc, sh = args
    return wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=store,
                     outs=outs if outs is not None else poisoned(x))


def v2(args, gs_t, store, v, outs=None):
    x, sel, sc, sh = args
    return wk.launch_pf(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=store,
                        outs=outs if outs is not None else poisoned(x), **v)


def head_info():
    wt = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(wt, "..", "..", ".."))
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:  # noqa
            return f"ERR {e}"
    info = {
        "head": sh("git rev-parse HEAD"),
        "log": sh("git log --oneline -5"),
        "anc_0761b30b": sh("git merge-base --is-ancestor 0761b30b HEAD; echo rc=$?"),
        "gpu": sh("nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv,noheader"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": sh("hostname"),
        "grep": sh("grep -n '_IMPL =\\|_AUTO_WARP_MIN_ROWS =\\|CACHE_X:' "
                   "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_adaln_quant.py"),
    }
    for k, val in info.items():
        print("INFO", k, val, flush=True)
    return info


def run_check():
    info = head_info()
    ok_all = True
    report = {"variants": {}}
    dev0 = torch.device("cuda", 0)
    wk._compile(dev0, 4, False, wk._DEFAULT_DEFINES)
    report["served_regs"] = {str(k): v for k, v in wk.NUM_REGS.items()}
    for v in VARIANTS:
        defs = tuple(v["extra_defines"]) + (f"PF_X_POLICY={v['x_policy']}",
                                            f"PF_FIRST_DIRECT={v['first_direct']}")
        wk._compile_pf(dev0, 4, defs)
        inf = dict(wk.PF_INFO[(0, 4, defs)])
        lines = [ln for ln in inf.pop("ptxas").splitlines() if "flux3_adaln_nvfp4_pf" in ln
                 or "registers" in ln or "stack" in ln or "spill" in ln]
        inf["ptxas_v"] = lines
        report["variants"][vname(v)] = inf
        print("PTXAS", vname(v), json.dumps(inf), flush=True)
    # served kernel ptxas for comparison (same options + -v), compiled out of band
    try:
        from cuda.bindings import nvrtc
        prog = wk._check(nvrtc.nvrtcCreateProgram(wk._SRC.encode(), b"s.cu", 0, [], []))
        opts = [b"--gpu-architecture=sm_120a", b"--fmad=false", b"--prec-div=true",
                b"--prec-sqrt=true", b"--ftz=false", b"-std=c++17", b"--ptxas-options=-v",
                b"-DWARPS=4", b"-DUSE_PDL=0", b"-DMINB=4"]
        nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        n = wk._check(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        report["served_ptxas_v"] = [ln for ln in log.decode(errors="replace").splitlines()
                                    if "registers" in ln or "stack" in ln or "spill" in ln]
        print("PTXAS served", report["served_ptxas_v"], flush=True)
    except Exception as e:  # noqa
        print("served ptxas probe failed", e, flush=True)

    bad = []
    with torch.inference_mode():
        cases = make_cases()
        for name, args in cases:
            x = args[0]
            gs = gs_for(x)
            pristine = [a.clone() if a is not None else None for a in args]
            for gname, gsv in (("gs", gs), ("gs_sat4x", float(torch.tensor(gs * 4.0, dtype=torch.float32)))):
                gs_t = torch.tensor([gsv], dtype=torch.float32, device=dev)
                for store in (True, False):
                    ref = served(args, gs_t, store)
                    for v in VARIANTS:
                        out = v2(args, gs_t, store, v)
                        ok = same(out, ref)  # store False: both x_mod buffers still the poison
                        if not ok:
                            bad.append(f"{name}|{gname}|store{int(store)}|{vname(v)}")
                            ok_all = False
                    print(name, gname, "store", int(store), "all variants bitwise:",
                          not any(b.startswith(f"{name}|{gname}|store{int(store)}|") for b in bad),
                          flush=True)
            inp_ok = all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
            ok_all &= inp_ok
            report[f"{name}|inputs_unchanged"] = inp_ok
        report["bitwise_failures"] = bad[:50]
        report["n_bitwise_failures"] = len(bad)

        # Untimed repeat sequence per variant at M4176 select (served shape) and M4256 mode:
        # warmup 3 + 20 iterations like the timed loop, resets (re-poison) every 8 iterations.
        rep_ok = True
        for name, args in cases:
            if name not in ("select_M4176", "mode_M4256", "mode_M4096"):
                continue
            x = args[0]
            gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
            ref = served(args, gs_t, True)
            pristine = [a.clone() if a is not None else None for a in args]
            for v in VARIANTS:
                outs = poisoned(x)
                first = None
                eq0 = eqref = inp = True
                for it in range(23):
                    if it % 8 == 0:
                        inp &= all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
                        for o in outs:
                            o.view(torch.uint8).fill_(0xCD)
                    o = v2(args, gs_t, True, v, outs=outs)
                    cur = tuple(t.clone() for t in o)
                    if first is None:
                        first = cur
                    eq0 &= same(cur, first)
                    eqref &= same(cur, ref)
                inp &= all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
                r = eq0 and eqref and inp
                rep_ok &= r
                if not r:
                    print("REPEAT FAIL", name, vname(v), eq0, eqref, inp, flush=True)
        report["repeat_passed"] = rep_ok
        ok_all &= rep_ok
        print("repeat all variants", rep_ok, flush=True)

        # Public-op dispatch: _V2=warp (rpw 2, xp 0 and 2) bitwise to _V2=0 at every case.
        disp_ok = True
        saved = (faq._V2, faq._V2_RPW, faq._V2_X_POLICY)
        for name, args in cases:
            x = args[0]
            gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
            faq._V2 = "0"
            a = torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, args[1], args[2], args[3], W, B, gs_t, EPS, True)
            a = tuple(t.clone() for t in a)
            for rpw, xp in ((2, 0), (2, 2), (0, 2)):
                faq._V2, faq._V2_RPW, faq._V2_X_POLICY = "warp", rpw, xp
                b = torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, args[1], args[2], args[3], W, B, gs_t, EPS, True)
                disp_ok &= same(a, b)
        faq._V2, faq._V2_RPW, faq._V2_X_POLICY = saved
        report["dispatch_v2_on_eq_off"] = disp_ok
        ok_all &= disp_ok
        print("dispatch v2 on == off", disp_ok, flush=True)
    print(json.dumps({"passed": bool(ok_all), "info": info, **report}))


def make_sets(s, kind, nsets=4):
    sets = []
    for _ in range(nsets):
        x = heavy(1, s, D)
        res = heavy(1, s, D)
        y = heavy(1, s, D)
        gate = (torch.randn(1, 1, D, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        if kind == "select":
            sel = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
            sel[:, :80] = True
            sc, sh = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        else:
            sel = None
            sc, sh = mod[:, 0:1, D: 2 * D], mod[:, 0:1, :D]
        gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
        outs = poisoned(x)
        sets.append((x, res, y, gate, sel, sc, sh, gs_t, outs))
    return sets


def run_sweep():
    info = head_info()
    calls = 28
    sets = make_sets(4176, "select")
    dev0 = torch.device("cuda", 0)
    arms = {"served": None}
    for v in VARIANTS:
        arms[vname(v)] = v
    sel_arms = [a for a in sys.argv[2:]] if len(sys.argv) > 2 else list(arms)
    arms = {k: arms[k] for k in (["served"] + [a for a in sel_arms if a != "served"])}
    arms["served_MINB3"] = "S3"

    def body(v, op=True):
        for i in range(calls):
            x, res, y, gate, sel, sc, sh, gs_t, outs = sets[i % len(sets)]
            torch.addcmul(res, gate, y, out=x)
            if not op:
                continue
            if v is None:
                wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=False, outs=outs)
            elif v == "S3":
                wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=False, outs=outs,
                          extra_defines=("MINB=3",))
            else:
                wk.launch_pf(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=False, outs=outs, **v)

    graphs = {}
    with torch.inference_mode():
        for name, v in list(arms.items()) + [("residual_only", "RO")]:
            vv = None if v == "RO" else v
            op = v != "RO"
            for _ in range(2):
                body(vv, op)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body(vv, op)
            graphs[name] = gr
        for name, v in arms.items():
            if isinstance(v, dict):
                defs = tuple(v["extra_defines"]) + (f"PF_X_POLICY={v['x_policy']}",
                                                    f"PF_FIRST_DIRECT={v['first_direct']}")
                i = wk.PF_INFO[(0, 4, defs)]
                print("ARM", name, "regs", i["regs"], "local", i["local_bytes"], "smem",
                      i["smem_bytes"], "ctas/sm", i["ctas_per_sm"], flush=True)
        t_end = time.time() + 8.0
        while time.time() < t_end:
            for gr in graphs.values():
                gr.replay()
            torch.cuda.synchronize()
        samples = {k: [] for k in graphs}
        rounds = 60
        for r in range(rounds):
            for name, gr in graphs.items():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(3):
                    gr.replay()
                e.record()
                e.synchronize()
                samples[name].append(s.elapsed_time(e) / 3)
            if r % 20 == 0:
                print("round", r, flush=True)
    ro = statistics.median(samples["residual_only"])
    res = {}
    for name, ts in samples.items():
        st = sorted(ts)
        med = statistics.median(st)
        p10, p90 = st[len(st) // 10], st[(9 * len(st)) // 10]
        res[name] = {"ms": med, "noise": (p90 - p10) / med,
                     "op_us": (med - ro) / calls * 1000.0 if name != "residual_only" else None}
    base = res["served"]["ms"]
    for name in res:
        if name != "residual_only":
            # paired per-round delta vs served
            d = sorted(a - b for a, b in zip(samples[name], samples["served"]))
            res[name]["delta_vs_served_ms_med"] = statistics.median(d)
            res[name]["neg_frac"] = sum(1 for z in d if z < 0) / len(d)
        print("SWEEP", name, json.dumps(res[name]), flush=True)
    print(json.dumps({"passed": True, "info": info, "residual_only_ms": ro, "res": res,
                      "served_ms": base}))


SERVED_CFGS = [(4, "MINB=4"), (4, "MINB=3"), (2, "MINB=7"), (2, "MINB=6"), (2, "MINB=5"),
               (4, "MINB=2"), (8, "MINB=1"), (1, "MINB=12"), (1, "MINB=14")]
if os.environ.get("B183_SCFG"):  # e.g. "4:MINB=4,1:MINB=12" (first entry = reference)
    SERVED_CFGS = [(int(a), b) for a, b in (c.split(":") for c in os.environ["B183_SCFG"].split(","))]


def run_served_sweep():
    """Served kernel at other launch bounds (warps, MINB): ptxas info, bitwise vs the default
    (4, MINB=4) build at every check case, then interleaved timing like run_sweep."""
    info = head_info()
    dev0 = torch.device("cuda", 0)
    cfg_info = {}
    for wp, mb in SERVED_CFGS:
        wk._compile(dev0, wp, False, (mb,))
    from cuda.bindings import driver
    for (dvi, wp, pdl, defs), fn in wk._MODULES.items():
        attr = driver.CUfunction_attribute
        loc = wk._check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn))
        occ = wk._check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, wp * 32, 0))
        cfg_info[f"w{wp}_{defs[0]}"] = {"regs": wk.NUM_REGS[(dvi, wp, pdl, defs)],
                                        "local_bytes": loc, "ctas_per_sm": occ,
                                        "warps_per_sm": occ * wp}
        print("CFG", f"w{wp}_{defs[0]}", json.dumps(cfg_info[f"w{wp}_{defs[0]}"]), flush=True)
    bitwise = {}
    with torch.inference_mode():
        for name, args in make_cases():
            x, sel, sc, sh = args
            gs = gs_for(x)
            for gsv in (gs, float(torch.tensor(gs * 4.0, dtype=torch.float32))):
                gs_t = torch.tensor([gsv], dtype=torch.float32, device=dev)
                for store in (True, False):
                    ref = served(args, gs_t, store)
                    for wp, mb in SERVED_CFGS:
                        out = wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=store,
                                        outs=poisoned(x), warps=wp, extra_defines=(mb,))
                        k = f"w{wp}_{mb}"
                        bitwise[k] = bitwise.get(k, True) and same(out, ref)
        print("BITWISE", json.dumps(bitwise), flush=True)
        calls = 28
        sets = make_sets(4176, "select")

        def body(cfg, op=True):
            for i in range(calls):
                x, res, y, gate, sel, sc, sh, gs_t, outs = sets[i % len(sets)]
                torch.addcmul(res, gate, y, out=x)
                if op:
                    wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=False, outs=outs,
                              warps=cfg[0], extra_defines=(cfg[1],))

        graphs = {}
        for cfg in SERVED_CFGS + [None]:
            nm = "residual_only" if cfg is None else f"w{cfg[0]}_{cfg[1]}"
            for _ in range(2):
                body(cfg or SERVED_CFGS[0], cfg is not None)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body(cfg or SERVED_CFGS[0], cfg is not None)
            graphs[nm] = gr
        t_end = time.time() + 8.0
        while time.time() < t_end:
            for gr in graphs.values():
                gr.replay()
            torch.cuda.synchronize()
        samples = {k: [] for k in graphs}
        for r in range(60):
            for name, gr in graphs.items():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(3):
                    gr.replay()
                e.record()
                e.synchronize()
                samples[name].append(s.elapsed_time(e) / 3)
    ro = statistics.median(samples["residual_only"])
    res = {}
    base_k = f"w{SERVED_CFGS[0][0]}_{SERVED_CFGS[0][1]}"
    for name, ts in samples.items():
        st = sorted(ts)
        med = statistics.median(st)
        res[name] = {"ms": med, "noise": (st[(9 * len(st)) // 10] - st[len(st) // 10]) / med,
                     "op_us": (med - ro) / calls * 1000.0}
        d = sorted(a - b for a, b in zip(samples[name], samples[base_k]))
        res[name]["delta_vs_default_ms_med"] = statistics.median(d)
        res[name]["neg_frac"] = sum(1 for z in d if z < 0) / len(d)
        print("SSWEEP", name, json.dumps(res[name]), flush=True)
    print(json.dumps({"passed": all(bitwise.values()), "info": info, "cfg": cfg_info,
                      "bitwise_vs_default": bitwise, "res": res}))


def run_bench():
    nsets = 4
    single_calls, mode_calls = 28, 4
    arm = f"v2={getattr(faq, '_V2', 'absent')}"
    sel_sets = make_sets(4176, "select", nsets)
    mode_sets = make_sets(4096, "mode", nsets)

    def one(st, residual=True, op=True):
        x, res, y, gate, sel, sc, sh, gs_t, _ = st
        if residual:
            torch.addcmul(res, gate, y, out=x)
        if op:
            torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, sel, sc, sh, W, B, gs_t, EPS, False)

    def step(residual=True, op=True):
        for i in range(single_calls):
            one(sel_sets[i % nsets], residual, op)
        for i in range(mode_calls):
            one(mode_sets[i % nsets], residual, op)

    results = {}
    with torch.inference_mode():
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        graphs = {}
        for name, kw in (("mix", {}), ("residual_only", {"op": False})):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                step(**kw)
            graphs[name] = gr
        t_end = time.time() + 10.0
        n = 0
        while time.time() < t_end:
            for gr in graphs.values():
                gr.replay()
            n += 1
            if n % 200 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        print("soak replays", n, flush=True)
        samples = {k: [] for k in graphs}
        for _ in range(200):
            for name, gr in graphs.items():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                gr.replay()
                e.record()
                e.synchronize()
                samples[name].append(s.elapsed_time(e))
    for name, ts in samples.items():
        ts = sorted(ts)
        med = statistics.median(ts)
        p10, p90 = ts[len(ts) // 10], ts[(9 * len(ts)) // 10]
        results[name] = {"ms": med, "p10": p10, "p90": p90, "noise": (p90 - p10) / med, "n": len(ts)}
        print(arm, name, json.dumps(results[name]), flush=True)
    mix = results["mix"]
    op_est = (mix["ms"] - results["residual_only"]["ms"]) / (single_calls + mode_calls) * 1000.0
    print(json.dumps({
        "batch": {"1": {"kernel_ms": mix["ms"], "noise_floor": mix["noise"]}},
        "output": {"class": "exact"},
        "arm": arm,
        "detail": results,
        "op_us_per_call_est": op_est,
    }))


if MODE == "check":
    run_check()
elif MODE == "sweep":
    run_sweep()
elif MODE == "served_sweep":
    run_served_sweep()
else:
    run_bench()
