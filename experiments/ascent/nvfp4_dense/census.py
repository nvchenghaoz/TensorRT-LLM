"""Idea 9 / branch 30, gate 0: per-(shape, M) NVFP4 tactic census on real captures.

No model load. Reads the shared capture set written by llm_ab.py (verifies
every file's sha256 against manifest.json first), and for every captured
(prompt, step, layer, shape, M) inside the dense range of the shape, plus
synthetic inputs (seeds 1, 2 and heavy-tailed randn*2^U(-8,8) rows, layer-0
real weights) at the protocol M:
  - main's pick and the dense pick in this process (tuned like the engine
    warmup: one M=8192 call per shape under autotune()), and in two separate
    subprocesses (mode picks);
  - every tactic main can run at that M (TRT CUTLASS 0-2, cuBLASLt i of the
    actual-M heuristic list) and every tactic the dense mode can run (CUTLASS
    0-2, cuBLASLt i of the bucket list resolved at M), each classed by its
    kernel names (torch profiler) and its cuBLASLt algo config, offered or not;
  - int16 mismatch count of each against main's pick (and against the
    captured in-model output when stored), first mismatch;
  - cold-L2 time (256 MiB flush before every iteration; 3 warmup + 15 timed).

usage: census.py --checkpoint CK --captures DIR [--mode census|picks]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

log = bench.log
GU_M = [300, 301, 320, 1000, 1023, 1200, 1856, 2000, 2048]
DN_M = [4096, 4097, 4800, 8000, 8100, 8192]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(d):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    bad = []
    for fn, h in man["files_sha256"].items():
        p = os.path.join(d, fn)
        if not os.path.exists(p) or sha256(p) != h:
            bad.append(fn)
    log(f"[manifest] commit={man['commit']} entries={len(man['entries'])} files={len(man['files_sha256'])} "
        f"sha_mismatch={bad}")
    return man, bad


def tco():
    return bench.tco()


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA"):
            if e.name not in names:
                names.append(e.name)
    return names


def classify(names):
    j = " ".join(names).lower()
    if "red_ws" in j or "splitk" in j or "split_k" in j or "reduce" in j:
        return "splitK/red_ws"
    if "stream_k" in j or "streamk" in j:
        return "stream-K"
    if "nvjet" in j:
        return "nvjet DP"
    if "cutlass" in j or "gemm" in j:
        return "cutlass DP"
    return "unknown"


def cold_us(fn, flush, warm=3, iters=15):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warm):
        flush.zero_()
        fn()
    for i in range(iters):
        flush.zero_()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return ts[len(ts) // 2] * 1e3


def tactic_fns(xq, xsf, wq, wsf, alpha, m, min_m):
    """(label, fn, meta) for every tactic main or the dense mode can run at M."""
    t = tco()
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
    fp4 = torch.classes.trtllm.FP4GemmRunner(torch.bfloat16, int(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4))
    lt = torch.classes.trtllm.CublasLtFP4GemmRunner(torch.bfloat16)
    out = []
    for c in range(fp4.get_num_configs()):
        out.append((f"cutlass{c}", (lambda c=c: fp4.run_gemm(xq, wq, xsf, wsf, alpha, 0, c, None, None)),
                    {"backend": "cutlass", "main": True, "dense_offered": True}))
    n_m = lt.get_num_heuristic_algos(xq, wq, xsf, wsf)
    for i in range(n_m):
        cfg = lt.get_algo_config(xq, wq, xsf, wsf, i, -1)
        out.append((f"lt_main{i}", (lambda i=i: lt.run_gemm(xq, wq, xsf, wsf, alpha, 0, i, None, None)),
                    {"backend": "cublaslt", "main": True, "algo": cfg}))
    b = t.nvfp4_dense_bucket(m, min_m)
    bq, bsf, _ = bench.make_act(b, xq.shape[1] * 2, 99)
    n_b = lt.get_num_heuristic_algos(bq, wq, bsf, wsf)
    for i in range(n_b):
        prof_cfg = lt.get_algo_config(bq, wq, bsf, wsf, i, -1)  # the algo profiled at the bucket
        cfg = lt.get_algo_config(xq, wq, xsf, wsf, i, b)  # what runs at M
        offered = bool(prof_cfg[0]) and not prof_cfg[1] and prof_cfg[5] <= 1 and prof_cfg[6] == 0
        out.append((f"lt_dense{i}@{b}",
                    (lambda i=i: lt.run_gemm_bucketed(xq, wq, xsf, wsf, alpha, 0, i, b, None, None)),
                    {"backend": "cublaslt", "main": False, "bucket": b, "algo_profiled": prof_cfg,
                     "algo_run": cfg, "fallback_at_M": bool(cfg[1]), "dense_offered": offered}))
    return out


def census_case(tag, xq, xsf, wq, wsf, alpha, m, n, min_m, flush, captured_out=None, timing=True,
                names_cache=None):
    t = tco()
    t.NVFP4_DENSE_TUNING = False
    r_main = bench.gemm(xq, xsf, wq, wsf, alpha)
    t.NVFP4_DENSE_TUNING = True
    r_dense = bench.gemm(xq, xsf, wq, wsf, alpha)
    torch.cuda.synchronize()
    row = {"case": tag, "M": m, "N": n, "picks": bench.picks_for("gate_up" if n > 10000 else "down", m, n,
                                                                  xq.shape[1] * 2)}
    row["dense_vs_main_int16"] = bench.int16_mismatch(r_dense, r_main)[0]
    if captured_out is not None:
        c, f = bench.int16_mismatch(r_main, captured_out.to(r_main.device))
        row["main_vs_inmodel_int16"] = c
        row["main_vs_inmodel_first"] = f
    tacs = []
    for label, fn, meta in tactic_fns(xq, xsf, wq, wsf, alpha, m, min_m):
        try:
            o = fn()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            tacs.append({"tactic": label, "error": str(e)[:200], **meta})
            continue
        cnt, first = bench.int16_mismatch(o, r_main)
        ent = {"tactic": label, "mismatch_vs_main": cnt, "first": first, **meta}
        key = (n, m, label)
        if names_cache is not None and key in names_cache:
            ent["kernels"] = names_cache[key]
        else:
            ent["kernels"] = kernel_names(fn)
            if names_cache is not None:
                names_cache[key] = ent["kernels"]
        ent["class"] = classify(ent["kernels"])
        if timing:
            ent["cold_us"] = cold_us(fn, flush)
        del o
        tacs.append(ent)
    row["tactics"] = tacs
    offered = [x for x in tacs if x.get("dense_offered")]
    row["offered_all_bitwise"] = all(x.get("mismatch_vs_main") == 0 for x in offered)
    row["offered_nonbitwise"] = [x["tactic"] for x in offered if x.get("mismatch_vs_main") != 0]
    row["main_set_nonbitwise"] = [x["tactic"] for x in tacs if x.get("main") and x.get("mismatch_vs_main") != 0]
    del r_main, r_dense
    return row


def tune_real(wts, dims):
    from tensorrt_llm._torch.autotuner import autotune
    t0 = time.time()
    with autotune():
        for name, (wq, wsf, alpha) in wts.items():
            n, k = dims[name]
            xq, xsf, _ = bench.make_act(8192, k, 7)
            bench.gemm(xq, xsf, wq, wsf, alpha)
            torch.cuda.synchronize()
    return time.time() - t0


def load_weights(d, man, layer, dev="cuda"):
    out = {}
    for key, w in man["weights"].items():
        if w["layer"] == layer:
            t = torch.load(os.path.join(d, w["file"]))
            out[w["shape"]] = (t["weight"].to(dev), t["weight_scale"].to(dev))
    return out


def mode_picks(args, dims, man, d):
    w0 = load_weights(d, man, 0)
    alphas = {e["shape"]: e["alpha"] for e in man["entries"] if e["layer"] == 0}
    wts = {s: (w0[s][0], w0[s][1], torch.tensor([alphas[s]], dtype=torch.float32, device="cuda")) for s in w0}
    ts = tune_real(wts, dims)
    res = {"tune_seconds": ts}
    for name, ms in (("gate_up", GU_M + [64, 256]), ("down", [300, 1000, 2000, 4000] + DN_M)):
        n, k = dims[name]
        for m in ms:
            res[f"{name}@{m}"] = bench.picks_for(name, m, n, k)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--mode", default="census")
    ap.add_argument("--no-subprocess-picks", action="store_true")
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    dims = bench.read_dims(args.checkpoint)
    d = args.captures
    man, bad = load_manifest(d)
    if args.mode == "picks":
        return mode_picks(args, dims, man, d)
    t = tco()
    ranges = {name: t._NVFP4_DENSE_SHAPES[dims[name]] for name in dims}
    min_ms = {name: r[0] for name, r in ranges.items()}
    max_ms = {name: r[1] for name, r in ranges.items()}
    log(f"[policy] ranges={ranges} sm_ok={t._nvfp4_dense_sm_ok()}")
    result = {"capture_dir": d, "capture_commit": man["commit"], "sha_mismatch": bad,
              "producer_sources_sha256": man.get("producer_sources_sha256")}
    # main/dense picks in two separate processes
    if not args.no_subprocess_picks:
        procs = []
        for i in range(2):
            cmd = [sys.executable, os.path.abspath(__file__), "--checkpoint", args.checkpoint, "--captures", d,
                   "--mode", "picks"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            try:
                procs.append(json.loads(p.stdout.strip().splitlines()[-1]))
            except Exception:  # noqa: BLE001
                procs.append({"error": p.stderr[-2000:]})
            log(f"[picks proc{i}] {json.dumps(procs[-1])}")
        result["picks_two_processes"] = procs
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    entries = man["entries"]
    by_layer = {}
    for e in entries:
        by_layer.setdefault(e["layer"], []).append(e)
    rows = []
    names_cache = {}
    tuned = False
    for layer in sorted(by_layer):
        w = load_weights(d, man, layer)
        if not tuned:
            alphas = {e["shape"]: e["alpha"] for e in by_layer[layer]}
            wts = {s: (w[s][0], w[s][1], torch.tensor([alphas[s]], dtype=torch.float32, device="cuda"))
                   for s in w}
            result["tune_seconds"] = tune_real(wts, dims)
            tuned = True
        seen = set()
        for e in by_layer[layer]:
            name, m = e["shape"], e["M"]
            n, k = dims[name]
            min_m = min_ms[name]
            in_range = min_m <= m <= max_ms[name]
            tag = f"{e['prompt']}|s{e['step']}|L{layer}|{name}|M{m}|{e.get('path')}"
            if not in_range:
                rows.append({"case": tag, "M": m, "N": n, "in_dense_range": False})
                log(f"[census] {tag} outside dense range (main path)")
                continue
            # Identical-M duplicates at one layer (4x8k steps) are all compared, timed once.
            timing = (name, m) not in seen
            seen.add((name, m))
            tt = torch.load(os.path.join(d, e["file"]))
            xq = tt["act_fp4"].cuda()
            xsf = tt["act_sf"].cuda()
            alpha = tt["alpha"].cuda().float().reshape(-1)[:1]
            wq, wsf = w[name]
            row = census_case(tag, xq, xsf, wq, wsf, alpha, m, n, min_m, flush, tt.get("out"), timing=timing,
                              names_cache=names_cache)
            row["in_dense_range"] = True
            row["real"] = True
            rows.append(row)
            log(f"[census] {tag} dense_vs_main={row['dense_vs_main_int16']} "
                f"inmodel={row.get('main_vs_inmodel_int16')} offered_all_bitwise={row['offered_all_bitwise']} "
                f"offered_nonbitwise={row['offered_nonbitwise']} main_set_nonbitwise={row['main_set_nonbitwise']} "
                f"picks={row['picks']}")
            for x in row["tactics"]:
                log(f"    {x['tactic']:<16} mm={x.get('mismatch_vs_main')} offered={x.get('dense_offered')} "
                    f"class={x.get('class')} us={x.get('cold_us', float('nan')):.1f} "
                    f"algo={x.get('algo_run', x.get('algo'))} k={x.get('kernels', [])[:3]}")
            del xq, xsf, tt
    # synthetic gate at protocol M, layer-0 weights
    w = load_weights(d, man, 0)
    alphas = {e["shape"]: e["alpha"] for e in man["entries"] if e["layer"] == 0}
    for name, ms in (("gate_up", GU_M), ("down", DN_M)):
        n, k = dims[name]
        wq, wsf = w[name]
        alpha = torch.tensor([alphas[name]], dtype=torch.float32, device="cuda")
        for variant, seed, heavy in (("seed1", 1, False), ("seed2", 2, False), ("heavy", 3, True)):
            for m in ms:
                xq, xsf, _ = bench.make_act(m, k, seed * 7919 + m, heavy)
                tag = f"synthetic-{variant}|{name}|M{m}"
                row = census_case(tag, xq, xsf, wq, wsf, alpha, m, n, min_ms[name], flush,
                                  timing=(variant == "seed1"), names_cache=names_cache)
                row["in_dense_range"] = True
                row["real"] = False
                rows.append(row)
                log(f"[census] {tag} dense_vs_main={row['dense_vs_main_int16']} "
                    f"offered_all_bitwise={row['offered_all_bitwise']} offered_nonbitwise={row['offered_nonbitwise']} "
                    f"main_set_nonbitwise={row['main_set_nonbitwise']} picks={row['picks']}")
                if variant == "seed1":
                    for x in row["tactics"]:
                        log(f"    {x['tactic']:<16} mm={x.get('mismatch_vs_main')} offered={x.get('dense_offered')} "
                            f"class={x.get('class')} us={x.get('cold_us', float('nan')):.1f} "
                            f"algo={x.get('algo_run', x.get('algo'))} k={x.get('kernels', [])[:3]}")
    dense_rows = [r for r in rows if r.get("in_dense_range")]
    fails = [r["case"] for r in dense_rows if not r["offered_all_bitwise"] or r["dense_vs_main_int16"] != 0]
    fallbacks = [(r["case"], x["tactic"]) for r in dense_rows for x in r["tactics"] if x.get("fallback_at_M")]
    result.update({"passed": not fails and not bad, "rows": rows, "fails": fails, "fallbacks": fallbacks,
                   "compared": "every offered dense tactic and the dense pick vs main's pick, int16, real captures "
                               "(layers 0/30/59) + synthetic seeds 1/2/heavy"})
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, "census.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"passed": result["passed"], "fails": fails, "fallbacks": fallbacks[:20],
                      "n_rows": len(rows), "n_dense_rows": len(dense_rows), "sha_mismatch": bad}))


if __name__ == "__main__":
    main()
