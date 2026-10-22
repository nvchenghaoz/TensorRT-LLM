# Idea 13 checks (model-free, synthetic NVFP4 operands of the real W4A4 prefill shapes).
#   python check.py            : (a) acceptance 1 -- NPROC fresh processes each tune nvfp4_gemm the
#                                way warmup does and report picks; mixed rows must agree in all;
#                                (b) numerics + untimed repeated execution of this tree's picks;
#                                (c) a cuBLASLt pick runs the SAME kernel at the bucket rows and at the
#                                actual mixed rows (shape-stable identity).
#   python check.py picks      : one fresh process: tune, print picks JSON (used by (a)).
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nvfp4_coldtune as ct  # noqa: E402

SH = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o_out": (5120, 6144), "down": (5120, 17408)}
MIXED = (2055, 2063, 2079, 2080, 4103, 4111, 4127, 4128, 8168, 8176, 8191)
PURE = (8, 16, 32, 2048, 4096, 6175, 6176, 6208, 8192)
NUM_MS = (8, 16, 32, 2048, 2055, 2063, 2079, 4096, 4111, 4127, 6175, 8168, 8191, 8192)
BUCKETS = (2080, 4128, 8191)
ALL_MS = tuple(sorted(set(MIXED) | set(PURE) | set(NUM_MS) | set(BUCKETS)))
NPROC = 3
log = ct.log


def tune_and_picks(shapes):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner as U
    tuner = AutoTuner.get()
    tuner.clear_cache()
    for s in shapes:
        with autotune():
            torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16, allowed_backends=ct.BACKENDS)
    torch.cuda.synchronize()
    r = ct.runner()
    picks = {}
    for s in shapes:
        picks[s.name] = {}
        for m in sorted(set(MIXED) | set(PURE) | set(NUM_MS)):
            hit, _, tac, t = tuner.profiling_cache.search_cache(
                ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))), U.tuning_config,
                apply_map_to_tuning_buckets=True)
            picks[s.name][str(m)] = {"tactic": list(tac) if isinstance(tac, tuple) else tac,
                                     "hit": bool(hit), "tuner_ms": t}
    return picks


def main_picks():
    import tensorrt_llm  # noqa: F401
    ct.MS = ALL_MS
    shapes = [ct.Shape(k, n, kk) for k, (n, kk) in SH.items()]
    picks = tune_and_picks(shapes)
    print("PICKS " + json.dumps(picks), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "picks":
        return main_picks()
    res = {"acceptance1": {}, "numerics": {}, "identity": {}}
    ok = True
    # (a) fresh processes
    runs = []
    for i in range(NPROC):
        log("== fresh process %d" % i)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "picks"],
                           capture_output=True, text=True, timeout=900)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("PICKS ")]
        if p.returncode != 0 or not line:
            log(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit("picks subprocess failed")
        runs.append(json.loads(line[-1][6:]))
        log("process %d mixed picks: %s" % (i, json.dumps(
            {s: {m: runs[-1][s][str(m)]["tactic"] for m in MIXED} for s in SH})))
        log("process %d pure picks: %s" % (i, json.dumps(
            {s: {m: runs[-1][s][str(m)]["tactic"] for m in PURE} for s in SH})))
    mixed_agree, pure_agree = True, {}
    for s in SH:
        for m in MIXED:
            ts = [json.dumps(r[s][str(m)]["tactic"]) for r in runs]
            hits = all(r[s][str(m)]["hit"] for r in runs)
            same = len(set(ts)) == 1 and hits
            mixed_agree = mixed_agree and same
            res["acceptance1"]["%s/M%d" % (s, m)] = {"tactics": ts, "all_hit": hits, "identical": same}
        for m in PURE:
            pure_agree["%s/M%d" % (s, m)] = len({json.dumps(r[s][str(m)]["tactic"]) for r in runs}) == 1
    res["acceptance1_mixed_identical"] = mixed_agree
    res["pure_identical_info"] = pure_agree
    ok = ok and mixed_agree
    log("acceptance 1 (mixed rows identical over %d fresh processes): %s" % (NPROC, mixed_agree))

    # (b) numerics + repeated execution, this process's own tune
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import OptimizationProfile
    ct.MS = ALL_MS
    shapes = [ct.Shape(k, n, kk) for k, (n, kk) in SH.items()]
    picks = tune_and_picks(shapes)
    r = ct.runner()
    for s in shapes:
        for m in NUM_MS:
            ref = s.ref(m)
            ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
            tac = ct.as_tac(picks[s.name][str(m)]["tactic"])
            errs = []
            for t in r.get_valid_tactics(s.inputs(m), OptimizationProfile()):
                o = r(s.inputs(m), tactic=t).float()
                errs.append((o - ref).abs().max().item())
            o = r(s.inputs(m), tactic=tac).float()
            e = (o - ref).abs()
            pick_max = e.max().item()
            med_all = sorted(errs)[len(errs) // 2]
            tol_ok = bool(torch.isfinite(o).all()) and pick_max <= 1.25 * med_all + ulp
            g, outs = ct.build_graph(s, m, tac, r)
            first, nmis, per_it = None, 0, []
            for it in range(ct.WARM + ct.ITERS):
                g.replay()
                torch.cuda.synchronize()
                cur = torch.stack([x.clone() for x in outs])
                if first is None:
                    first = cur[0].clone()
                nmis += int(sum(not torch.equal(cur[j], first) for j in range(s.nb)))
                per_it.append((cur.float() - ref).abs().max().item())
            del g, outs
            ok = ok and tol_ok and nmis == 0
            res["numerics"]["%s/M%d" % (s.name, m)] = {
                "tactic": picks[s.name][str(m)]["tactic"], "pick_max_abs": pick_max,
                "pick_mean_abs": e.mean().item(), "all_tactics_max_abs": errs, "bf16_ulp": ulp,
                "tol_ok": tol_ok, "repeat_iters": ct.WARM + ct.ITERS, "calls_per_iter": s.nb,
                "repeat_mismatch_vs_iter1": nmis, "per_iter_max_err_min": min(per_it),
                "per_iter_max_err_max": max(per_it)}
            log("  %s M=%d pick %s max_abs %.4g (median over tactics %.4g, ulp %.3g) tol_ok %s "
                "repeat mismatches %d" % (s.name, m, tac, pick_max, med_all, ulp, tol_ok, nmis))
            del ref
            torch.cuda.empty_cache()
    # (c) identity: cuBLASLt ids at the bucket vs at the actual mixed rows -> same kernel
    id_ok = True
    for s in shapes:
        for m in MIXED:
            from tensorrt_llm._torch.custom_ops.torch_custom_ops import _nvfp4_map_m
            b = _nvfp4_map_m(m)
            ids_b = [t for t in r.get_valid_tactics(s.inputs(b), OptimizationProfile())
                     if t[0] == "cublaslt"]
            ids_m = [t for t in r.get_valid_tactics(s.inputs(m), OptimizationProfile())
                     if t[0] == "cublaslt"]
            rows = []
            for t in ids_b:
                kb = ct.kernel_names(s, b, t)
                km = ct.kernel_names(s, m, t)
                same = kb == km
                id_ok = id_ok and same
                rows.append({"id": t[1], "kernel": [k for k in km if "Memset" not in k][:1],
                             "memset": any("Memset" in k for k in km), "same_kernel": same})
            res["identity"]["%s/M%d" % (s.name, m)] = {
                "bucket": b, "n_ids_bucket": len(ids_b), "n_ids_actual": len(ids_m),
                "same_id_set": sorted(x[1] for x in ids_b) == sorted(x[1] for x in ids_m),
                "same_order": [x[1] for x in ids_b] == [x[1] for x in ids_m], "ids": rows}
            log("  identity %s M=%d vs bucket %d: %d/%d cuBLASLt ids, same set %s, same order %s, "
                "all same kernel: %s" % (s.name, m, b, len(ids_b), len(ids_m),
                                         res["identity"]["%s/M%d" % (s.name, m)]["same_id_set"],
                                         res["identity"]["%s/M%d" % (s.name, m)]["same_order"],
                                         all(x["same_kernel"] for x in rows)))
            tac = picks[s.name][str(m)]["tactic"]
            if tac[0] == "cublaslt":
                res["identity"]["%s/M%d/pick" % (s.name, m)] = {
                    "tactic": tac, "kernels_at_m": ct.kernel_names(s, m, ct.as_tac(tac))}
    res["identity_same_kernel"] = id_ok
    ok = ok and id_ok
    print(json.dumps({"passed": ok, "compared": (
        "acceptance 1: %d fresh processes' nvfp4_gemm picks at mixed rows %s for qkvz/qkv/o_out/down; "
        "numerics: pick vs FP32 reference from the same FP4 operands (<= 1.25x median error over all "
        "valid tactics + 1 bf16 ulp); repeated execution: nb rotated copies x %d graph replays bitwise vs "
        "iteration 1" % (NPROC, list(MIXED), ct.WARM + ct.ITERS)), "results": res}))


if __name__ == "__main__":
    main()
