"""Idea 112 / branch 51: same-slot op pair for down_proj NVFP4 GEMM, one shared tuner cache.

Runs unchanged in both arms of run_gpu_pair (base = main, cand = branch).
  --role base : delete the shared cache file, tune (gate_up + down, one M=8192 call each under
                autotune(), as the engine warmup), save it to the shared file.
  --role cand : load the base arm's shared file (main's keys), tune (only keys missing from it:
                the branch's new gemm_dense_dp entries), check every loaded entry is unchanged,
                save to a candidate copy.
Then REPS fresh processes each warm the GPU (8 s BF16 GEMMs), load the arm's cache with tuning
off, and time trtllm::nvfp4_gemm on down (N, K from config.json) per M with a 256 MiB L2 flush
before every call (5 warmup + 20 timed, CUDA events). kernel_ms = median of process medians;
noise_floor = max(within-process (p90-p10)/median, across-process spread).
Picks and the kernel name (torch.profiler) per M are logged in the tune process.

usage: pair.py --role base|cand --checkpoint CK --tag TAG
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nvfp4_dense"))
import bench  # noqa: E402

log = bench.log
M_IN = [513, 600, 640, 768, 896, 900, 960, 1000, 1023]
M_OUT = [300, 512, 1024, 1200, 2000, 4000, 8000]
REPS = 3
SHARED = "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/shared/b51_tuner_cache"
OPS = ("trtllm::nvfp4_gemm::gemm", "trtllm::nvfp4_gemm::gemm_dense", "trtllm::nvfp4_gemm::gemm_dense_dp")


def nvfp4_entries():
    from tensorrt_llm._torch.autotuner import AutoTuner
    out = {}
    for key, val in AutoTuner.get().profiling_cache.cache.items():
        if key[0] in OPS:
            shapes = key[3]
            out[f"{key[0].split('::')[-1]}|N{shapes[1][0]}|K{shapes[0][1] * 2}|M{shapes[0][0]}|{key}"] = str(val[1])
    return out


def all_picks(n, k, m):
    t = bench.tco()
    res = bench.picks_for("down", m, n, k)
    rng = getattr(t, "_NVFP4_DENSE_DP_SHAPES", {}).get((n, k))
    if rng is not None and rng[0] <= m <= rng[1]:
        from tensorrt_llm._torch.autotuner import AutoTuner
        b = t.nvfp4_dense_bucket(m, rng[0])
        for key, val in AutoTuner.get().profiling_cache.cache.items():
            sh = key[3]
            if key[0] == OPS[2] and sh[0][0] == b and sh[1][0] == n:
                res[f"gemm_dense_dp@{b}"] = str(val[1])
    return res


def kernel_of(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name[:110] for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")][:2]


def child_tune(args, dims):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    os.makedirs(os.path.dirname(args.cache_in), exist_ok=True)
    loaded = {}
    if args.role == "base" and os.path.exists(args.cache_in):
        os.remove(args.cache_in)
    if args.role == "cand" and os.path.exists(args.cache_out):
        os.remove(args.cache_out)
    if args.role == "cand":
        if os.path.exists(args.cache_in):
            AutoTuner.get().profiling_cache.load_cache(args.cache_in, 0)
            loaded = nvfp4_entries()
            log(f"[tune] loaded {len(loaded)} nvfp4 entries from {args.cache_in} sha={bench_sha(args.cache_in)}")
        else:
            log(f"[tune] WARNING shared cache {args.cache_in} missing; tuning from scratch")
    weights = {name: bench.make_weight(*dims[name], 100 + i) for i, name in enumerate(("gate_up", "down"))}
    t0 = time.time()
    with autotune(cache_path=None):
        for name, (wq, wsf, wgs) in weights.items():
            xq, xsf, xgs = bench.make_act(8192, dims[name][1], 7)
            bench.gemm(xq, xsf, wq, wsf, (1.0 / (xgs * wgs)).float())
            torch.cuda.synchronize()
            log(f"[tune] {name} done {time.time() - t0:.1f}s")
    after = nvfp4_entries()
    changed = [k for k, v in loaded.items() if after.get(k) != v]
    new = sorted(k for k in after if k not in loaded)
    AutoTuner.get().profiling_cache.save_cache(args.cache_out, 0)
    log(f"[tune] saved {args.cache_out} sha={bench_sha(args.cache_out)} entries={len(after)} "
        f"loaded_changed={changed} new_keys={len(new)}")
    for k in new if args.role == "cand" else []:
        log(f"[tune] new {k.split('|(')[0]} -> {after[k]}")
    n, k = dims["down"]
    wq, wsf, wgs = weights["down"]
    picks = {}
    for m in M_IN + M_OUT:
        xq, xsf, xgs = bench.make_act(m, k, 11)
        alpha = (1.0 / (xgs * wgs)).float()
        picks[m] = {"picks": all_picks(n, k, m),
                    "kernel": kernel_of(lambda: bench.gemm(xq, xsf, wq, wsf, alpha))}
        log(f"[pick] down M={m} {picks[m]}")
    print(json.dumps({"loaded_changed": changed, "n_loaded": len(loaded), "n_new": len(new), "picks": picks,
                      "tune_s": time.time() - t0}))


def bench_sha(p):
    import hashlib
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except OSError:
        return None


def child_time(args, dims):
    from tensorrt_llm._torch.autotuner import autotune
    log(f"[gpu] start {bench.gpu_state()}")
    bench.warm_gpu(8.0)
    n, k = dims["down"]
    wq, wsf, wgs = bench.make_weight(n, k, 101)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    batch = {}
    with autotune(tune_mode=False, cache_path=args.cache_out):
        pass
    log(f"[time] cache {args.cache_out} sha={bench_sha(args.cache_out)} entries={len(nvfp4_entries())}")
    for m in M_IN + M_OUT:
        xq, xsf, xgs = bench.make_act(m, k, 11)
        alpha = (1.0 / (xgs * wgs)).float()
        bench.WARMUP, bench.ITERS = 5, 20
        med, p10, p90 = bench.timed(lambda: bench.gemm(xq, xsf, wq, wsf, alpha), flush)
        pk = all_picks(n, k, m)
        batch[f"down@{m}"] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med,
                              "picks": pk}
        log(f"[time] down M={m} med={med * 1e3:.1f}us p10={p10 * 1e3:.1f} p90={p90 * 1e3:.1f} picks={pk}")
    log(f"[gpu] end {bench.gpu_state()}")
    print(json.dumps({"batch": batch}))


def run_child(cmd, tag):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        sys.stdout.write(f"[{tag}] {line}")
        sys.stdout.flush()
        if line.startswith("{"):
            last = line
    p.wait()
    return json.loads(last)


def driver(args):
    base = [sys.executable, os.path.abspath(__file__), "--role", args.role, "--checkpoint", args.checkpoint,
            "--cache-in", args.cache_in, "--cache-out", args.cache_out]
    tune = run_child(base + ["--child", "tune"], "tune")
    reps = [run_child(base + ["--child", "time"], f"rep{r}") for r in range(REPS)]
    batch = {}
    for key in reps[0]["batch"]:
        meds = sorted(rp["batch"][key]["kernel_ms"] for rp in reps)
        within = sorted(rp["batch"][key]["noise_floor"] for rp in reps)[REPS // 2]
        med = meds[REPS // 2]
        across = (meds[-1] - meds[0]) / med
        batch[key] = {"kernel_ms": med, "noise_floor": max(within, across), "within_process_noise": within,
                      "across_process_spread": across, "process_medians_ms": meds,
                      "picks_per_process": [rp["batch"][key]["picks"] for rp in reps]}
        log(f"[agg] {key} med={med * 1e3:.1f}us procs={[round(x * 1e3, 1) for x in meds]} "
            f"noise={batch[key]['noise_floor']:.4f}")
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "role": args.role,
                      "tune": {k: tune[k] for k in ("loaded_changed", "n_loaded", "n_new", "picks")},
                      "processes": REPS}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--child", default=None)
    ap.add_argument("--cache-in", default=None)
    ap.add_argument("--cache-out", default=None)
    args = ap.parse_args()
    if args.cache_in is None:
        args.cache_in = os.path.join(SHARED, f"{args.tag}_base.json")
        args.cache_out = args.cache_in if args.role == "base" else os.path.join(SHARED, f"{args.tag}_cand.json")
    import tensorrt_llm  # noqa: F401
    dims = bench.read_dims(args.checkpoint)
    log(f"[env] role={args.role} cache_in={args.cache_in} cache_out={args.cache_out} "
        f"TLLM_AUTOTUNER_CACHE_PATH={os.environ.get('TLLM_AUTOTUNER_CACHE_PATH')} "
        f"TRTLLM_CACHE_DIR={os.environ.get('TRTLLM_CACHE_DIR')} "
        f"dp_shapes={getattr(bench.tco(), '_NVFP4_DENSE_DP_SHAPES', None)}")
    if args.child == "tune":
        return child_tune(args, dims)
    if args.child == "time":
        return child_time(args, dims)
    return driver(args)


if __name__ == "__main__":
    main()
