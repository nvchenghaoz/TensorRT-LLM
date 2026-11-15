"""Idea 38 worker: FA2 paged prefill, Gemma4 sliding d256, BF16 Q + FP8 E4M3 KV (HND, page 32).

One arm per process. Overlay arms (v1, v1b, pin) patch a temp copy of FlashInfer's include tree
and build under their own fresh FLASHINFER_WORKSPACE_BASE; --fresh-base gives the plain arm its
own fresh base too. Env is set BEFORE flashinfer is imported.

Per case: plan once; untimed repeated-execution sequence (warmup+iters calls, every call
bitwise vs the first; Q and KV pool bitwise unchanged afterwards); CUDA-event timing
(warmup, iters); first-call output saved to --out-dir for the driver's cross-arm compare.
Last stdout line: {"batch": {case: {"kernel_ms", "noise_floor", ...}}, "output": {...}}.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import patch  # noqa: E402

CASES = "1x300,16x300,27x300,1x1000,8x1000,1x2000,4x2000,16x512,1x4000,2x4000,1x8000"
# Workload case label -> the sliding-attention call one prefill forward makes for it under
# max_num_tokens 8192 (requests packed per forward, each prompt fresh: kv_len == qo_len).
WORKLOAD_CASES = ",".join([
    "isl300@1:1x300", "isl300@4:4x300", "isl300@16:16x300", "isl300@64:27x300",
    "isl1k@1:1x1000", "isl1k@4:4x1000", "isl1k@16:8x1000", "isl1k@64:8x1000",
    "isl2k@1:1x2000", "isl2k@4:4x2000", "isl2k@16:4x2000", "isl2k@64:4x2000",
    "isl4k@1:1x4000", "isl4k@4:2x4000", "isl4k@16:2x4000", "isl4k@64:2x4000",
    "isl8k@1:1x8000", "isl8k@4:1x8000", "isl8k@16:1x8000", "isl8k@64:1x8000"])


def log(*a):
    print(*a, flush=True)


def heartbeat(stop, tag):
    t0 = time.time()
    while not stop.wait(60):
        log(f"[hb {tag}] {time.time() - t0:.0f}s load={os.getloadavg()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tree", help="tree | v1 | v1b | pin")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--base", default="", help="FLASHINFER_WORKSPACE_BASE to use (reuse)")
    ap.add_argument("--fresh-base", action="store_true")
    ap.add_argument("--only-run", action="store_true", help="1 call per case (ncu)")
    ap.add_argument("--no-time", action="store_true", help="check sequence only")
    ap.add_argument("--ptxas-v", action="store_true", help="build with -Xptxas=-v (spills)")
    a = ap.parse_args()

    log("ARM", a.arm, "load", os.getloadavg())
    tree_inc = patch.include_dir()
    log("FLASHINFER_INCLUDE_DIR", tree_inc)
    with open(os.path.join(tree_inc, "flashinfer", "attention", "prefill.cuh")) as f:
        tree_txt = f.read()
    log("TREE_PREFILL_VARIANT", patch.tree_variant(tree_txt))
    base = a.base
    if a.arm in patch.VARIANTS or a.arm == "pin":
        tmp = tempfile.mkdtemp(prefix=f"idea38_{a.arm}_inc_")
        inc, _ = patch.make_overlay(a.arm, tmp)
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = f"-I{inc}"
        log("OVERLAY", a.arm, inc)
        if not base:
            base = tempfile.mkdtemp(prefix=f"idea38_{a.arm}_ws_")
    elif a.fresh_base and not base:
        base = tempfile.mkdtemp(prefix=f"idea38_{a.arm}_ws_")
    if a.ptxas_v:
        # spill report for the build (needs a fresh base so the module is really compiled)
        if not base:
            base = tempfile.mkdtemp(prefix=f"idea38_{a.arm}_ptxas_ws_")
        flags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (flags + " -Xptxas=-v").strip()
        os.environ["FLASHINFER_JIT_VERBOSE"] = "1"
    if base:
        os.environ["FLASHINFER_WORKSPACE_BASE"] = base
    log("FLASHINFER_WORKSPACE_BASE", os.environ.get("FLASHINFER_WORKSPACE_BASE", "<default>"))
    log("FLASHINFER_EXTRA_CUDAFLAGS", os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "<none>"))
    os.environ.setdefault("FLASHINFER_LOGGING_LEVEL", "INFO")

    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, a.arm), daemon=True).start()

    import torch
    import flashinfer
    from flashinfer.jit import env as jit_env
    import fa_census as fc

    log("flashinfer", getattr(flashinfer, "__version__", "?"), flashinfer.__file__)
    log("JIT_DIR", jit_env.FLASHINFER_JIT_DIR)
    dims = fc.load_dims(a.ckpt)
    log("DIMS", json.dumps(dims))
    p = torch.cuda.get_device_properties(0)
    log("GPU", p.name, "SMs", p.multi_processor_count)

    batch = {}
    all_ok = True
    import cases as cz
    case_list = WORKLOAD_CASES if a.cases == "workload" else a.cases
    for cs in case_list.split(","):
        label, _, spec = cs.rpartition(":")
        label = label or spec
        c = cz.make_case(spec, dims)
        rec = dict(case=c.label(), Hq=c.Hq, Hkv=c.Hkv, D=c.D, window_left=c.window_left)
        t0 = time.time()
        fn, out, info = fc.build_variant("fa2", c)
        rec["plan_info"] = info.get("plan_info")
        rec["plan_ms_first"] = round(info["plan_ms_first"], 1)
        if a.only_run:
            fn()
            torch.cuda.synchronize()
            log("RAN", label)
            continue
        q0, kv0 = c.q.clone(), c.kv_fp8.clone()
        # untimed repeated-call sequence: warmup+iters calls, each byte-compared to the first
        fn()
        torch.cuda.synchronize()
        first = out.clone()
        bad = 0
        for _ in range(a.warmup + a.iters - 1):
            fn()
            torch.cuda.synchronize()
            bad += not cz.bytes_equal(out, first)
        rc = dict(calls=a.warmup + a.iters, mismatches=bad, compare="raw bytes")
        rc["q_unchanged"] = cz.bytes_equal(q0, c.q)
        rc["kv_unchanged"] = cz.bytes_equal(kv0, c.kv_fp8)
        rc["finite"] = bool(torch.isfinite(first.float()).all())
        ok = rc["mismatches"] == 0 and rc["q_unchanged"] and rc["kv_unchanged"] and rc["finite"]
        all_ok &= ok
        rec["repeat"] = rc
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
            torch.save(first.cpu(), os.path.join(a.out_dir, f"{a.arm}_{label}.pt"))
        if not a.no_time:
            t = fc.time_fn(fn, a.warmup, a.iters)
            rec.update({k: round(v, 5) for k, v in t.items()})
            rec["tflops"] = round(c.flops / (t["median_ms"] * 1e-3) / 1e12, 1)
            batch[label] = dict(kernel_ms=round(t["median_ms"], 5),
                                noise_floor=round(t["noise"], 4),
                                p10_ms=round(t["p10_ms"], 5), p90_ms=round(t["p90_ms"], 5))
        rec["wall_s"] = round(time.time() - t0, 1)
        log("RESULT", json.dumps(rec))
        del c, fn, out, first, q0, kv0
        torch.cuda.empty_cache()

    # Proof the module was built in this workspace (not an AOT/shared cache).
    try:
        jd = jit_env.FLASHINFER_JIT_DIR
        mods = [d for d in os.listdir(jd) if "prefill" in d] if os.path.isdir(jd) else []
        log("JIT_MODULES", mods)
        for d in mods:
            so = os.path.join(jd, d, f"{d}.so")
            if os.path.exists(so):
                log("JIT_SO", so, "mtime_age_s", round(time.time() - os.path.getmtime(so)))
    except Exception as ex:  # noqa: BLE001
        log("JIT_DIR listing failed", ex)
    stop.set()
    log("END load", os.getloadavg())
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}, "passed": all_ok,
                      "arm": a.arm, "repeat_ok": all_ok}))


if __name__ == "__main__":
    main()
