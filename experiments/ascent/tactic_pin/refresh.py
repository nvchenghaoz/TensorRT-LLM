"""Idea 116 (branch 50): refresh pair after the hunk sync onto main 87621e99 (one run_gpu_pair).

usage (same command in both arms): refresh.py --checkpoint CK --captures DIR --shared DIR [--reps 3]

Arm is detected from the tree: main has no trtllm pins (torch_custom_ops.nvfp4_pinned_tactic).
Each arm runs REPS fresh processes (lifetimes). Main rep r tunes as the engine warmup does (one
M=8192 trtllm::nvfp4_gemm call per shape under autotune()) and saves its autotuner cache to
<shared>/rep<r>/; candidate rep r LOADS that cache (no tuning), so both arms of rep r run the same
tuner picks and differ only by the pins. Per process: 10 s GPU warm, clock/temp logged, weights
make_weight(n, k, 1000 + i) (bench_pair.py), activations make_act(m, k, 11); warmed: one call at
every served M before timing (the pins' host query runs once per M); then per key bench.timed:
256 MiB flush before each of WARMUP=5 + ITERS=30 calls, CUDA events, median/p10/p90. Per key and M
it prints the tuner key that serves the call (gemm_dense_dp / gemm_dense / gemm), the cache tactic,
the pinned tactic (candidate) and the kernel names (torch profiler).

Main arm, after timing: int16 sha256 of the op's output per key (same inputs) -> <shared>/rep<r>.
Candidate rep 0, after timing (checks, all untimed):
  K: both kill switches 0 (module flags): output sha256 per key == main rep 0's sha256;
  T: down pins (M 1200/1500/1800/2000) vs main's pick (kill switches 0, same process): real
     captures L0/30/59 (manifest sha256 verified) and the timing inputs: n_diff, max abs, max ULP;
     FP32 reference (dequantized NVFP4 operands, torch.matmul FP32): pinned max-abs error <= 1.10x
     main's + 1 BF16 ulp;
  R: pitfall 1208: the timed run's sequence call for call, untimed (the warm pass over every key,
     then per key flush + WARMUP+ITERS calls, no reset between), every call int16-equal to call 1,
     inputs unchanged, call 1 vs main rep 0's sha256 (must be equal where no down pin applies).
Last line: {"batch": {key: {kernel_ms, noise_floor, ...}}, "output": {...}, "passed": ...}.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "nvfp4_dense"))
import bench  # noqa: E402
import protocol  # noqa: E402  (imports bench_pair: WARMUP=5, ITERS=30, 256 MiB flush)

log = bench.log
GU_M = [300, 1000, 1200, 2000, 4000]
DN_M = [300, 1000, 1200, 1500, 1800, 2000, 2300, 4000]
DOWN_PIN_M = [1200, 1500, 1800, 2000]
LAYERS = (0, 30, 59)
KEYS = [("gate_up", m) for m in GU_M] + [("down", m) for m in DN_M]


def sha16(o):
    return hashlib.sha256(o.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def ordered(x):
    i = x.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)


def diff(a, b):
    nd = int((a.view(torch.int16) != b.view(torch.int16)).sum().item())
    if nd == 0:
        return {"n_diff": 0, "max_abs": 0.0, "max_ulp": 0}
    return {"n_diff": nd, "max_abs": float((a.float() - b.float()).abs().max().item()),
            "max_ulp": int((ordered(a) - ordered(b)).abs().max().item())}


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    p = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(p)
    tr = json.load(open(p))
    os.remove(p)
    return [(e["name"][:100], e.get("args", {}).get("grid")) for e in tr["traceEvents"] if e.get("cat") == "kernel"]


def resolve(t, inputs, n, k, m):
    """Which tuner key serves this call in trtllm::nvfp4_gemm, its cache tactic, and (branch) the
    tactic the pins make it run."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    tuner = AutoTuner.get()
    bl = bench.BACKENDS.split(",")
    shapes = tuple(x.size() for x in inputs)
    pin = getattr(t, "nvfp4_pinned_tactic", None)
    dpr = t._nvfp4_dense_dp_range(n, k) if hasattr(t, "_nvfp4_dense_dp_range") else None
    if dpr is not None and dpr[0] <= m <= dpr[1]:
        r = t.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl, dense_min_m=dpr[0], dense_cutlass_only=True)
        hit, _, tac, _ = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm_dense_dp", [r], shapes,
                                                            t._nvfp4_dense_tuning_config(*dpr))
        if hit and tac != -1:
            return {"key": "gemm_dense_dp", "tuned": str(tac), "runs": str(tac)}
    rng = t._nvfp4_dense_range(n, k)
    if rng is not None and rng[0] <= m <= rng[1]:
        r = t.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl, dense_min_m=rng[0])
        hit, _, tac, _ = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm_dense", [r], shapes,
                                                            t._nvfp4_dense_tuning_config(*rng))
        tac = tac if hit else -1
        runs = pin("gemm_dense", inputs, torch.bfloat16, tac, bl, rng[0]) if pin else tac
        if runs != -1:
            return {"key": "gemm_dense", "tuned": str(tac), "runs": str(runs)}
    r = t.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl)
    hit, _, tac, _ = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm", [r], shapes,
                                                        t.NVFP4GemmUnifiedRunner.tuning_config)
    runs = pin("gemm", inputs, torch.bfloat16, tac, bl) if pin else tac
    return {"key": "gemm", "tuned": str(tac), "runs": str(runs)}


def set_pins(t, on):
    if hasattr(t, "NVFP4_GATEUP_PIN"):
        t.NVFP4_GATEUP_PIN = on
        t.NVFP4_DOWN_PIN = on


def one(args):
    t0 = time.time()
    dims = bench.read_dims(args.checkpoint)
    t = bench.tco()
    arm = "candidate" if hasattr(t, "nvfp4_pinned_tactic") else "main"
    rep_dir = os.path.join(args.shared, f"rep{args.rep}")
    log(f"[arm] {arm} rep={args.rep} torch_custom_ops={t.__file__} sha256="
        f"{hashlib.sha256(open(t.__file__, 'rb').read()).hexdigest()[:16]}")
    log(f"[gpu] {torch.cuda.get_device_name()} start {bench.gpu_state()}")
    log(protocol.protocol_line())
    weights = {}
    for i, name in enumerate(("gate_up", "down")):
        n, k = dims[name]
        weights[name] = bench.make_weight(n, k, 1000 + i)
    bench.warm_gpu()
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    if arm == "main":
        shutil.rmtree(rep_dir, ignore_errors=True)
        os.makedirs(rep_dir)
        cache = os.path.join(rep_dir, "autotuner_cache.json")
        ts = time.time()
        with autotune(tune_mode=True, cache_path=cache):
            for name in ("gate_up", "down"):
                wq, wsf, wgs = weights[name]
                n, k = dims[name]
                xq, xsf, xgs = bench.make_act(8192, k, 7)
                bench.gemm(xq, xsf, wq, wsf, (1.0 / (xgs * wgs)).float())
                torch.cuda.synchronize()
        log(f"[tune] {time.time() - ts:.1f}s saved {glob.glob(cache + '*')}")
    else:
        src = os.path.join(rep_dir, "autotuner_cache.json")
        if not os.path.exists(src):
            print(json.dumps({"passed": False, "error": f"main cache missing: {src}"}), flush=True)
            return
        tmp = tempfile.mkdtemp()
        cache = os.path.join(tmp, "autotuner_cache.json")
        shutil.copy(src, cache)
        with autotune(tune_mode=False, cache_path=cache):
            pass
        log(f"[cache] loaded main rep{args.rep} cache sha256="
            f"{hashlib.sha256(open(src, 'rb').read()).hexdigest()[:16]}")
    log(f"[cache] entries={len(AutoTuner.get().profiling_cache.cache)}")
    flush = protocol.make_flush()
    acts = {}
    for name, m in KEYS:
        wq, wsf, wgs = weights[name]
        n, k = dims[name]
        xq, xsf, xgs = bench.make_act(m, k, 11)
        acts[(name, m)] = (xq, xsf, (1.0 / (xgs * wgs)).float())

    def fn_of(name, m):
        wq, wsf, _ = weights[name]
        xq, xsf, alpha = acts[(name, m)]
        return lambda: bench.gemm(xq, xsf, wq, wsf, alpha)

    # warm pass: every served M once before timing
    for name, m in KEYS:
        fn_of(name, m)()
    torch.cuda.synchronize()
    log(f"[gpu] before timing {bench.gpu_state()}")
    batch, picks = {}, {}
    for name, m in KEYS:
        med, p10, p90 = bench.timed(fn_of(name, m), flush)
        key = f"{name}@{m}"
        batch[key] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med}
        log(f"[time] {arm} rep{args.rep} {key} {med * 1e3:.1f}us p10={p10 * 1e3:.1f} p90={p90 * 1e3:.1f}")
    log(f"[gpu] after timing {bench.gpu_state()}")
    for name, m in KEYS:
        wq, wsf, _ = weights[name]
        xq, xsf, alpha = acts[(name, m)]
        n, k = dims[name]
        rs = resolve(t, [xq, wq, xsf, wsf, alpha], n, k, m)
        rs["kernels"] = kernels_of(fn_of(name, m))
        picks[f"{name}@{m}"] = rs
        log(f"[pick] {arm} rep{args.rep} {name}@{m}: key={rs['key']} tuned={rs['tuned']} runs={rs['runs']} "
            f"kernels={rs['kernels']}")
    res = {"arm": arm, "rep": args.rep, "batch": batch, "picks": picks}
    outs = {f"{name}@{m}": sha16(fn_of(name, m)()) for name, m in KEYS}
    if arm == "main":
        with open(os.path.join(rep_dir, "main_sha.json"), "w") as f:
            json.dump(outs, f)
        res["sha"] = outs
        res["passed"] = True
    else:
        res["sha_pins_on"] = outs
        if args.checks:
            res["checks"] = checks(args, t, dims, weights, acts, fn_of, flush, rep_dir)
            res["passed"] = res["checks"]["passed"]
        else:
            res["passed"] = True
    log(f"[done] {time.time() - t0:.0f}s")
    print(json.dumps(res), flush=True)


def checks(args, t, dims, weights, acts, fn_of, flush, rep_dir):
    import census
    import check
    main_sha = json.load(open(os.path.join(rep_dir, "main_sha.json")))
    out = {}
    # K: kill switches 0 vs main rep 0, every key
    set_pins(t, False)
    kres = {f"{name}@{m}": sha16(fn_of(name, m)()) == main_sha[f"{name}@{m}"] for name, m in KEYS}
    set_pins(t, True)
    log(f"[K] kill switches 0 sha256 == main 87621e99 (arm A rep0): {kres}")
    out["killswitch0_sha_equal_main"] = kres
    # T: down-pin tolerance vs main's pick and vs the FP32 reference
    man, bad = census.load_manifest(args.captures)
    trows = []
    n, k = dims["down"]
    one_ = torch.ones(1, device="cuda")

    def tol_row(src, m, xq, xsf, wq, wsf, alpha, wdq):
        inputs = [xq, wq, xsf, wsf, alpha]
        pinned = bench.gemm(xq, xsf, wq, wsf, alpha)
        set_pins(t, False)
        mainp = bench.gemm(xq, xsf, wq, wsf, alpha)
        main_res = resolve(t, inputs, n, k, m)
        set_pins(t, True)
        pin_res = resolve(t, inputs, n, k, m)
        # GEMM math: (q_x*sf_x) @ (q_w*sf_w)^T * alpha, FP32
        ref = (bench.dequant(xq, xsf, one_, m, k) @ wdq.t()) * alpha.float()
        e_p = (pinned.float() - ref).abs().max().item()
        e_m = (mainp.float() - ref).abs().max().item()
        ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
        r = {"src": src, "M": m, "pin_runs": pin_res["runs"], "main_runs": main_res["runs"],
             "vs_main_pick": diff(pinned, mainp), "pin_err_vs_fp32": e_p, "main_err_vs_fp32": e_m,
             "ok": e_p <= 1.10 * e_m + ulp and bool(torch.isfinite(pinned.float()).all())}
        log(f"[T] {'OK ' if r['ok'] else 'BAD'} {src} down@{m}: pin {r['pin_runs']} vs main {r['main_runs']}: "
            f"{r['vs_main_pick']}; fp32 err pin={e_p:.5g} main={e_m:.5g} ulp={ulp:.3g}")
        trows.append(r)
        del pinned, mainp, ref

    wq, wsf, _ = weights["down"]
    wdq = bench.dequant(wq, wsf, one_, n, k)
    for m in DOWN_PIN_M:
        xq, xsf, alpha = acts[("down", m)]
        tol_row("synthetic-timing-inputs", m, xq, xsf, wq, wsf, alpha, wdq)
    del wdq
    by = {}
    for e in man["entries"]:
        by.setdefault((e["layer"], e["shape"]), []).append(e)
    for layer in LAYERS:
        w = census.load_weights(args.captures, man, layer)
        wdq = bench.dequant(w["down"][0], w["down"][1], one_, n, k)
        loaded = {}
        for m in DOWN_PIN_M:
            ent = next((e for e in sorted(by.get((layer, "down"), []), key=lambda e: e["M"]) if e["M"] >= m), None)
            if ent is None:
                log(f"[T] skip L{layer} down@{m}: no capture with >= {m} rows")
                continue
            if ent["file"] not in loaded:
                tt = torch.load(os.path.join(args.captures, ent["file"]))
                loaded[ent["file"]] = (tt["act_fp4"].cuda(), tt["act_sf"].cuda(),
                                       tt["alpha"].cuda().float().reshape(-1)[:1])
            ax, asf, aal = loaded[ent["file"]]
            xq, xsf = check.slice_rows(ax, asf, m, k)
            tol_row(f"real-L{layer}", m, xq, xsf, w["down"][0], w["down"][1], aal, wdq)
        del w, wdq, loaded
        torch.cuda.empty_cache()
    out["down_pin_tolerance"] = trows
    out["captures_sha_mismatch"] = bad
    # R: pitfall 1208 repeated sequence, mirrors the timed run call for call
    snaps = {(name, m): protocol.snapshot(list(acts[(name, m)])) for name, m in KEYS}
    wsnap = {nm: protocol.snapshot(list(weights[nm])) for nm in weights}
    for name, m in KEYS:  # the warm pass
        fn_of(name, m)()
    rrows = {}
    for name, m in KEYS:
        first, calls, nd, _ = protocol.seq(fn_of(name, m), flush)
        key = f"{name}@{m}"
        same_main = sha16(first) == main_sha[key]
        nn, kk = dims[name]
        pinned = t._nvfp4_pin_entry("gemm", nn, kk, m) or t._nvfp4_pin_entry("gemm_dense", nn, kk, m)
        r = {"calls": calls, "differ_from_call1": sum(1 for v in nd if v), "call1_equal_main": same_main,
             "pinned": str(pinned), "inputs_unchanged": protocol.unchanged(snaps[(name, m)], list(acts[(name, m)]))}
        r["ok"] = calls == protocol.CALLS and r["differ_from_call1"] == 0 and r["inputs_unchanged"]
        rrows[key] = r
        log(f"[R] {'OK ' if r['ok'] else 'BAD'} {key}: calls {calls}/{protocol.CALLS}, differ from call 1: "
            f"{r['differ_from_call1']}, call1 == main sha: {same_main} (pin {pinned}), "
            f"inputs unchanged {r['inputs_unchanged']}")
        del first
    w_ok = all(protocol.unchanged(wsnap[nm], list(weights[nm])) for nm in weights)
    log(f"[R] weights unchanged {w_ok}")
    out["repeat"] = rrows
    out["weights_unchanged"] = w_ok
    # call 1 must equal main wherever no down pin applies (the gate_up pins are exact)
    exact_ok = all(r["call1_equal_main"] for kk, r in rrows.items()
                   if not (kk.startswith("down@") and int(kk.split("@")[1]) in DOWN_PIN_M))
    out["unpinned_and_gateup_call1_equal_main"] = exact_ok
    out["passed"] = bool(all(kres.values()) and all(r["ok"] for r in trows) and not bad
                         and all(r["ok"] for r in rrows.values()) and w_ok and exact_ok)
    log(f"[checks] passed={out['passed']}")
    return out


def provenance():
    for cmd in (["git", "rev-parse", "HEAD"], ["git", "diff", "--stat", "87621e993960a2ded006d11cf9d226b0182e0ddc",
                                                "HEAD"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            log(f"[git] {' '.join(cmd)} rc={r.returncode}\n{r.stdout}{r.stderr}")
        except Exception as e:  # noqa: BLE001
            log(f"[git] {' '.join(cmd)} failed: {e!r}")


def drive(args):
    provenance()
    reps = []
    for r in range(args.reps):
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", "one", "--checkpoint", args.checkpoint,
               "--captures", args.captures, "--shared", args.shared, "--rep", str(r)] + (["--checks"] if r == 0 else [])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = None
        for line in p.stdout:
            sys.stdout.write(f"[rep{r}] {line}")
            sys.stdout.flush()
            if line.startswith("{"):
                last = line
        p.wait()
        if last is None:
            print(json.dumps({"passed": False, "error": f"rep {r} gave no JSON (rc {p.returncode})"}))
            return
        reps.append(json.loads(last))
        if "batch" not in reps[-1]:
            print(json.dumps({"passed": False, "error": reps[-1]}))
            return
    batch = {}
    for key in reps[0]["batch"]:
        meds = sorted(rp["batch"][key]["kernel_ms"] for rp in reps)
        within = sorted(rp["batch"][key]["noise_floor"] for rp in reps)[len(reps) // 2]
        med = meds[len(meds) // 2]
        across = (meds[-1] - meds[0]) / med
        batch[key] = {"kernel_ms": med, "noise_floor": max(within, across), "within_process_noise": within,
                      "across_process_spread": across, "process_medians_ms": meds,
                      "p10_ms": sorted(rp["batch"][key]["p10_ms"] for rp in reps)[len(reps) // 2],
                      "p90_ms": sorted(rp["batch"][key]["p90_ms"] for rp in reps)[len(reps) // 2],
                      "picks_per_process": [{kk: rp["picks"][key][kk] for kk in ("key", "tuned", "runs")}
                                            for rp in reps],
                      "kernels_rep0": reps[0]["picks"][key]["kernels"]}
        log(f"[agg] {reps[0]['arm']} {key} med={med * 1e3:.1f}us procs={[round(x * 1e3, 1) for x in meds]} "
            f"noise={batch[key]['noise_floor']:.4f} picks={batch[key]['picks_per_process']}")
    res = {"batch": batch, "output": {"class": "reordered"}, "arm": reps[0]["arm"], "processes": args.reps,
           "passed": all(rp.get("passed") for rp in reps)}
    if reps[0]["arm"] == "candidate":
        c = reps[0].get("checks", {})
        res["checks"] = {"passed": c.get("passed"),
                         "killswitch0_sha_equal_main": c.get("killswitch0_sha_equal_main"),
                         "down_pin_tolerance": c.get("down_pin_tolerance"),
                         "repeat": c.get("repeat"), "weights_unchanged": c.get("weights_unchanged"),
                         "unpinned_and_gateup_call1_equal_main": c.get("unpinned_and_gateup_call1_equal_main"),
                         "captures_sha_mismatch": c.get("captures_sha_mismatch"),
                         "protocol": protocol.protocol_line()}
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pair")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--checks", action="store_true")
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    if args.mode == "one":
        return one(args)
    return drive(args)


if __name__ == "__main__":
    main()
