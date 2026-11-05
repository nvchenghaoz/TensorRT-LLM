"""Idea 226 job A steps 3-4: grid-width sweep of lowPrecisionTwoShotAllReduceKernel (LOWPRECISION FP8 two-shot AR).

4 MPI ranks (mpi4py.futures, as bench.py), one process per rank for the whole sweep. The two-shot grid is chosen per
call from TRTLLM_LP_AR_TWOSHOT_CTAS (customLowPrecisionAllReduceKernels.cu lowPrecisionTwoShotCtas); every rank sets
the same value between 4-rank barriers. Order (timing before any CUPTI, pitfall 1464):
  1. TIMING  [M,5120] bf16, M = 2048/4096/6144: NCCL (strategy AUTO, served PCIe NCCL defaults applied) and LP at
     16/32/48/64/96/128 CTAs; plus M = 16/128 under AUTO and LOWPRECISION (routing: both must be NCCL). 20 warmup +
     100 timed, median of 3 sequences, inputs re-created from the seed before each sequence; max over ranks.
     SM clocks by UUID before and after.
  2. BITWISE untimed, same 120 calls per sequence: env=16 reference outputs for 120 seeded inputs (each also checked
     vs the fp32 sum); then 16 again (determinism), 32, 48, 64, default (env unset), 96, 128 must be bitwise equal on
     every element and every call. After each width the IPC barrier flags are inspected: entry flags and blocks
     [0, grid) hold the last call's flag, every other slot holds less.
  3. ROBUST 1000 calls/rank at the default (env unset), changing inputs, 4-rank reset barrier every 250 calls,
     IPC flags inspected before each reset with grid == LP_TWOSHOT_DEFAULT_CTAS, every call checked, re-runs bitwise.
  4. CUPTI last: kernel names at 2048 (LP) and 16/128 (LP strategy must equal AUTO bitwise, NCCL kernels).
Writes $AUTO_DEV_ARTIFACTS/lp_width_table.json; last stdout line: {"passed": ..., ...}.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

ENV = "TRTLLM_LP_AR_TWOSHOT_CTAS"
BIG_M = [2048, 4096, 6144]
SMALL_M = [16, 128]
WIDTHS = [16, 32, 48, 64, 96, 128]
DEFAULT_CTAS = 64  # LP_TWOSHOT_DEFAULT_CTAS in customLowPrecisionAllReduceKernels.h
ELTS_PER_BLOCK = 16 * 31 * 16  # fp8 elts per load * (warp - 1) * warps per block: one block pass
BIT_ITERS = bench.WARMUP + bench.ITERS
LP_BOUND = 0.05


def log(*a):
    print(*a, flush=True)


def _set_width(w):
    if w is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = str(w)


def _grid(m, w):
    rounds = (m * bench.HIDDEN // bench.WORLD - 1) // ELTS_PER_BLOCK + 1
    return min(DEFAULT_CTAS if w is None else w, 128, rounds)


def rank_width(env):
    wd = bench._watchdog("width", 1500)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    dev = torch.device("cuda", rank)
    res = {"rank": rank, "applied": applied, "timing": {}, "bitwise": {}, "flags": {}, "err16": {}, "routing": {}}
    S = AllReduceStrategy

    # ------------------------------------------------------------------ 1. timing
    for m in BIG_M + SMALL_M:
        res["timing"][str(m)] = {}
        auto = bench.Runner(torch, mapping, S.AUTO, m, graph=False)
        res["timing"][str(m)]["NCCL_AUTO"] = bench._time(torch, auto, m, rank, dev)
        del auto
        lp = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        for w in (WIDTHS if m in BIG_M else [None]):
            _set_width(w)
            torch.cuda.synchronize()
            bench._barrier()
            key = f"LP_{w}" if w is not None else "LP_strategy"
            res["timing"][str(m)][key] = bench._time(torch, lp, m, rank, dev)
            if rank == 0:
                t = res["timing"][str(m)][key]
                log(f"[time] m={m} {key} med={t['median_ms']*1e3:.1f}us nf={t['noise_floor']:.3f} "
                    f"seqs={[round(v*1e3,1) for v in t['seq_medians_ms']]}")
        _set_width(None)
        if rank == 0:
            t = res["timing"][str(m)]["NCCL_AUTO"]
            log(f"[time] m={m} NCCL_AUTO med={t['median_ms']*1e3:.1f}us nf={t['noise_floor']:.3f}")
        del lp
        torch.cuda.synchronize()
        bench._barrier()
    res["clocks_after_timing"] = bench._gpu_state() if rank == 0 else None

    # ------------------------------------------------------------------ 2. bitwise vs env=16
    for m in BIG_M:
        lp = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        _set_width(16)
        torch.cuda.synchronize()
        bench._barrier()
        refs = []
        e16 = {"e": 0.0, "rel_rms": 0.0, "nonfinite": 0}
        for it in range(BIT_ITERS):
            heavy = it % 4 == 3
            lp.x.copy_(bench._gen(torch, m, 300 + it, rank, dev, heavy))
            out = lp().clone()
            refs.append(out)
            r, ab = bench._ref(torch, m, 300 + it, dev, heavy)
            er = bench._err(torch, out, r, ab)
            e16["e"] = max(e16["e"], er["e"])
            e16["rel_rms"] = max(e16["rel_rms"], er["rel_rms"])
            e16["nonfinite"] += int(not er["finite"])
        torch.cuda.synchronize()
        res["err16"][str(m)] = e16
        bench._barrier()
        res["flags"][f"{m}_16ref"] = bench._lp_flag_state(torch, mapping, _grid(m, 16))
        res["bitwise"][str(m)] = {}
        for w in [16, 32, 48, 64, None, 96, 128]:
            _set_width(w)
            torch.cuda.synchronize()
            bench._barrier()
            mism, mism_elems, nonfinite = 0, 0, 0
            for it in range(BIT_ITERS):
                heavy = it % 4 == 3
                lp.x.copy_(bench._gen(torch, m, 300 + it, rank, dev, heavy))
                out = lp()
                ne = int((out.view(torch.int16) != refs[it].view(torch.int16)).sum().item())
                mism += int(ne != 0)
                mism_elems += ne
                nonfinite += int(not bool(torch.isfinite(out).all().item()))
            torch.cuda.synchronize()
            key = "default" if w is None else str(w)
            if key == "16" and key in res["bitwise"][str(m)]:
                key = "16_rerun"
            bench._barrier()
            fs = bench._lp_flag_state(torch, mapping, _grid(m, w))
            res["flags"][f"{m}_{key}"] = fs
            res["bitwise"][str(m)][key] = {"calls": BIT_ITERS, "mismatched_calls": mism,
                                           "mismatched_elems": mism_elems, "nonfinite_calls": nonfinite,
                                           "grid_expected": _grid(m, w), "grid_seen_in_flags": fs["blocks_used"],
                                           "flags_ok": fs["ok"]}
            if rank == 0:
                log(f"[bitwise] m={m} width={key} vs env=16: mismatched_calls={mism} elems={mism_elems} "
                    f"nonfinite={nonfinite} grid={fs['blocks_used']}/{_grid(m, w)} flags_ok={fs['ok']}")
        _set_width(None)
        del refs, lp
        torch.cuda.empty_cache()
        bench._barrier()

    # ------------------------------------------------------------------ 3. robust at the default
    _set_width(None)
    r = bench._robust(torch, mapping, S.LOWPRECISION, 2048, rank, dev, expect_blocks=_grid(2048, None))
    res["robust"] = r
    if rank == 0:
        log(f"[robust] default m=2048 calls={r['calls']} nonfinite={r['nonfinite_calls']} "
            f"max_rel_rms={r['max_rel_rms']:.4g} max_e={r['max_e']:.4g} flags_ok={r['flags_ok']} "
            f"rerun_mismatch={r['rerun_bitwise_mismatch']} {r['seconds']}s")
        for s in r["flag_states"]:
            log(f"[flags] counter={s['counter']} blocks_used={s['blocks_used']} ok={s['ok']} "
                f"adv={s.get('advanced_by')} entry={s['entry_slots']}")

    # ------------------------------------------------------------------ 4. CUPTI: kernel names, routing
    for m in SMALL_M + [2048]:
        a = bench.Runner(torch, mapping, S.AUTO, m, graph=False)
        b = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        same = True
        for it in range(5):
            x = bench._gen(torch, m, 900 + it, rank, dev, heavy=(it % 2 == 1))
            a.x.copy_(x)
            b.x.copy_(x)
            oa = a().clone()
            ob = b().clone()
            torch.cuda.synchronize()
            same = same and bool(torch.equal(oa, ob))
        entry = {"LP_bitwise_equal_AUTO": same}
        if rank == 0:
            entry["kernels_AUTO"] = bench._kernel_names(torch, a)
            entry["kernels_LP"] = bench._kernel_names(torch, b)
        else:
            a()
            b()
        torch.cuda.synchronize()
        res["routing"][str(m)] = entry
        if rank == 0:
            log(f"[routing] m={m} LP==AUTO bitwise={same} AUTO={entry['kernels_AUTO']} LP={entry['kernels_LP']}")
        del a, b
        bench._barrier()
    bench._barrier()
    wd.cancel()
    return res


def main():
    t0 = time.time()
    log(f"[width] start half(CUDA_VISIBLE_DEVICES)={os.environ.get('CUDA_VISIBLE_DEVICES')} {ENV} in env: "
        f"{os.environ.get(ENV)}")
    os.environ.pop(ENV, None)
    clocks_before = bench._gpu_state()
    log(clocks_before)
    rs = bench.run_ranks(rank_width, {"_name": "width", "_apply_nccl_tuning": "1"})
    clocks_after = bench._gpu_state()
    log(clocks_after)
    table = {"shape": "[M,5120] bf16, 4 ranks, per call, max over ranks of the median of 3 x 100 (20 warmup)",
             "clocks_before": clocks_before, "clocks_after_timing_rank0": rs[0].get("clocks_after_timing"),
             "clocks_after": clocks_after, "timing_us": {}, "bitwise": {}, "err16": {}, "passed": True,
             "reasons": []}
    for m in rs[0]["timing"]:
        rows = {}
        for k in rs[0]["timing"][m]:
            rows[k] = {"median_us": round(max(r["timing"][m][k]["median_ms"] for r in rs) * 1e3, 1),
                       "seq_medians_us": [round(max(r["timing"][m][k]["seq_medians_ms"][s] for r in rs) * 1e3, 1)
                                          for s in range(bench.SEQS)],
                       "noise_floor": round(max(r["timing"][m][k]["noise_floor"] for r in rs), 4),
                       "seq_dispersion": round(max(r["timing"][m][k]["seq_dispersion"] for r in rs), 4)}
        if "LP_16" in rows:
            b = rows["LP_16"]["median_us"]
            for k, v in rows.items():
                v["vs_LP16"] = round(v["median_us"] / b - 1.0, 4)
        table["timing_us"][m] = rows
        log(f"[table] m={m} " + " ".join(f"{k}={v['median_us']}" for k, v in rows.items()))
    for m in map(str, BIG_M):
        table["err16"][m] = {"e": max(r["err16"][m]["e"] for r in rs),
                             "rel_rms": max(r["err16"][m]["rel_rms"] for r in rs),
                             "nonfinite": sum(r["err16"][m]["nonfinite"] for r in rs)}
        if table["err16"][m]["rel_rms"] > LP_BOUND or table["err16"][m]["nonfinite"]:
            table["passed"] = False
            table["reasons"].append(f"LP16 error m={m}")
        table["bitwise"][m] = {}
        for k in rs[0]["bitwise"][m]:
            v = {"mismatched_calls": sum(r["bitwise"][m][k]["mismatched_calls"] for r in rs),
                 "mismatched_elems": sum(r["bitwise"][m][k]["mismatched_elems"] for r in rs),
                 "nonfinite_calls": sum(r["bitwise"][m][k]["nonfinite_calls"] for r in rs),
                 "calls_per_rank": rs[0]["bitwise"][m][k]["calls"],
                 "grid_expected": rs[0]["bitwise"][m][k]["grid_expected"],
                 "grid_seen_in_flags": [r["bitwise"][m][k]["grid_seen_in_flags"] for r in rs],
                 "flags_ok_all_ranks": all(r["bitwise"][m][k]["flags_ok"] for r in rs)}
            table["bitwise"][m][k] = v
            if v["mismatched_calls"] or v["nonfinite_calls"] or not v["flags_ok_all_ranks"]:
                table["passed"] = False
                table["reasons"].append(f"bitwise/flags m={m} width={k}: {v}")
    rob = [r["robust"] for r in rs]
    table["robust_default_2048"] = {
        "calls_per_rank": bench.ROBUST_CALLS, "reset_barrier_every": bench.ROBUST_CHUNK,
        "expect_grid": _grid(2048, None),
        "nonfinite_calls": sum(x["nonfinite_calls"] for x in rob), "max_rel_rms": max(x["max_rel_rms"] for x in rob),
        "max_e": max(x["max_e"] for x in rob), "ipc_flags_ok_all_ranks": all(x["flags_ok"] for x in rob),
        "rerun_bitwise_mismatch": sum(x["rerun_bitwise_mismatch"] for x in rob),
        "blocks_used_rank0": [s["blocks_used"] for s in rob[0]["flag_states"]],
        "flag_states_bad": [s for x in rob for s in x["flag_states"] if not s["ok"]][:6]}
    rb = table["robust_default_2048"]
    if rb["nonfinite_calls"] or rb["max_rel_rms"] > LP_BOUND or not rb["ipc_flags_ok_all_ranks"] \
            or rb["rerun_bitwise_mismatch"]:
        table["passed"] = False
        table["reasons"].append("robust default")
    table["routing"] = {}
    for m in rs[0]["routing"]:
        eq = all(r["routing"][m]["LP_bitwise_equal_AUTO"] for r in rs)
        table["routing"][m] = {"LP_bitwise_equal_AUTO_all_ranks": eq,
                               "kernels_AUTO": rs[0]["routing"][m].get("kernels_AUTO"),
                               "kernels_LP": rs[0]["routing"][m].get("kernels_LP")}
        if int(m) < 2048 and not eq:
            table["passed"] = False
            table["reasons"].append(f"routing m={m}")
    t2048 = table["timing_us"]["2048"]
    best = min((k for k in t2048 if k.startswith("LP_")), key=lambda k: t2048[k]["median_us"])
    table["best_width_2048"] = best
    table["default_vs_LP16_2048"] = t2048[f"LP_{DEFAULT_CTAS}"]["vs_LP16"]
    table["noise_2048"] = {k: max(v["noise_floor"], v["seq_dispersion"]) for k, v in t2048.items()}
    table["wall_s"] = round(time.time() - t0, 1)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "lp_width_table.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(art, "lp_width_raw.json"), "w") as f:
        json.dump(rs, f, default=str)
    summary = {k: table[k] for k in ("passed", "reasons", "best_width_2048", "default_vs_LP16_2048", "err16",
                                     "wall_s")}
    summary["timing_us_median"] = {m: {k: v["median_us"] for k, v in rows.items()}
                                   for m, rows in table["timing_us"].items()}
    summary["robust"] = {k: rb[k] for k in ("nonfinite_calls", "max_rel_rms", "ipc_flags_ok_all_ranks",
                                            "rerun_bitwise_mismatch", "expect_grid")}
    print(json.dumps(summary))
    sys.exit(0 if table["passed"] else 4)


if __name__ == "__main__":
    main()
