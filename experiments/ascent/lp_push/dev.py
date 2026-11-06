"""Idea 360 / branch 215: PUSH (posted-write) FP8 two-shot for the LOWPRECISION prefill all-reduce.

The push kernel (lowPrecisionTwoShotPushAllReduceKernel, customLowPrecisionAllReduceKernels.cu) replaces the pull
path's preprocess + peer-READ two-shot with quantize + posted peer WRITES; TRTLLM_LP_AR_PUSH=0 restores the pull path.
4 MPI ranks (mpi4py.futures via experiments/ascent/lp_ar/bench.py), one process per GPU (pitfall 3336).

Modes:
  dev   one 4-GPU job, in this order (timing before any CUPTI, pitfall 1464):
        1. TIMING [M,5120] bf16, M = 1024/2048/4096/6144: PULL (env 0) and PUSH (env 1), 20 warmup + 100 timed,
           median of 3 sequences, seeded inputs re-created before each sequence; per call, max over ranks.
        2. BITWISE push vs pull on the same inputs at M = 205/257/1024/1536/2047/2048/3001/4096/6144/8192, 3 inputs
           each (one heavy-tailed), every element, every rank; plus error vs the fp32 sum (sanity only).
        3. SEQUENCE 1000 back-to-back push calls, shapes cycling 2048/257/6144 (odd cycle: both ping/pong parities at
           every shape), inputs changing from a pool of 4 per shape, every call compared bitwise with the pull
           output of the same input (computed before), mismatches counted on the GPU (no host sync inside a chunk);
           4-rank reset barrier every 250 calls; before each reset the IPC flags are inspected (entry, scatter and
           gather flags hold the last call's flag, nothing holds more, counters equal on all ranks).
        4. DISPATCH (CUPTI): LOWPRECISION autotuned at M=1/2/4/8/128 like the served warm-up, then one eager call per
           M = 1/2/4/8/128/204/205/2048 under push=1 and push=0; kernel names per rank.
  pair  kernel_time evidence (both arms run this same script; base = pull, branch = push):
        1. TIMING this tree's DEFAULT LP path (env unset) at M = 1024/2048/4096/6144/8192, same counts as dev; each
           rank reads its own SM clock through NVML while the timed sequence is still on the GPU (under load). A
           trigger (dispersion > 3% or clock < 2200 MHz on any rank) repeats that shape once in-job (both reported).
        2. INTERLEAVED in-process ABAB: TRTLLM_LP_AR_PUSH=0,1,0,1,... 4+4 blocks per M, one 20+100 sequence per
           block, same inputs. On the base tree the env is ignored: the same 4+4 blocks of the pull kernel (control).
           Rank 0 logs the node's state (all 8 GPUs' util/SM clock/memory.used, compute apps, loadavg) around
           every timed block, outside the timed region.
        3. Candidate tree only (push support present): dev's BITWISE (2) and 1000-call SEQUENCE (3) checks.
        4. RECEIPT (CUPTI, after all timing): kernel names of one eager call at M=2048 with env unset, =1 and =0,
           every rank; plus rank 0 counts the push/pull symbol and getenv strings (strings -a equivalent) in every
           .so under tensorrt_llm/libs and every tensorrt_llm .so mapped in its process (/proc/self/maps).
        Last stdout line: {"batch": {M: {"kernel_ms", "noise_floor"}}, "output": {"class": "exact"}, ...}.
"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lp_ar"))
import bench  # noqa: E402

ENV = "TRTLLM_LP_AR_PUSH"
HIDDEN = bench.HIDDEN
WORLD = bench.WORLD
# Served on main after landing 77: 1024 (design-D chunk at 4096 <= T <= 6144), 2048 (c1), 8192 (c4, unpipelined);
# 4096/6144 are secondary (served only with the pipeline kill switch off).
TIME_M = [1024, 2048, 4096, 6144, 8192]
NOISE_TRIGGER = 0.03
CLOCK_TRIGGER_MHZ = 2200
BIT_M = [205, 257, 1024, 1536, 2047, 2048, 3001, 4096, 6144, 8192]
SEQ_SHAPES = [2048, 257, 6144]
SEQ_POOL = 4
SEQ_CALLS = 1000
SEQ_CHUNK = 250
DISPATCH_M = [1, 2, 4, 8, 128, 204, 205, 2048]
TUNE_M = [1, 2, 4, 8, 128]
LP_REL_RMS_BOUND = 2.0 ** -4
AB_ROUNDS = 4  # 4 x (PUSH=0, PUSH=1) = 8 alternating blocks per shape (ABAB), same in both arms
RECEIPT_M = 2048
LIB_STRINGS = ["lowPrecisionTwoShotPushAllReduceKernel", "lowPrecisionTwoShotAllReduceKernel",
               "lowPrecisionPreprocessKernel", "TRTLLM_LP_AR_PUSH", "TRTLLM_LP_AR_TWOSHOT_CTAS"]


def log(*a):
    print(*a, flush=True)


def _set_push(v):
    if v is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = str(v)


def _has_push_support():
    from tensorrt_llm._torch.distributed.allreduce_helper import CustomAllReduceHelper as H
    return hasattr(H, "LP_PUSH_CTAS_PER_PEER")


def _lp_ipc_bytes():
    """Per-GPU bytes of the LOWPRECISION IPC allocations (ping + pong + barrier in + out), 2 MiB aligned."""
    from tensorrt_llm._torch.distributed.allreduce_helper import CustomAllReduceHelper as H
    al = lambda n: (n + (1 << 21) - 1) // (1 << 21) * (1 << 21)  # noqa: E731
    ws = H.max_workspace_size_lowprecision(WORLD)
    try:
        bb = H.lowprecision_barrier_bytes(WORLD)
    except AttributeError:
        from tensorrt_llm._ipc_utils import IpcMemory
        bb = IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * WORLD * 2
    return {"workspace_bytes": ws, "barrier_bytes": bb, "allocated_per_gpu": 2 * al(ws) + 2 * al(bb)}


def _sm_clock(torch, rank):
    try:
        return int(torch.cuda.clock_rate(rank))
    except Exception:  # noqa
        return None


def _time_seq(torch, run, m, rank, dev):
    """One timed sequence exactly as bench._time's (input re-created from the seed, 20 warmup, 100 timed).
    Also reads this GPU's SM clock while the timed calls are still queued (under load)."""
    run.x.copy_(bench._gen(torch, m, 0, rank, dev))
    torch.cuda.synchronize()
    bench._barrier()
    for _ in range(bench.WARMUP):
        run()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(bench.ITERS)]
    for i in range(bench.ITERS):
        ev[i][0].record()
        run()
        ev[i][1].record()
    clk = _sm_clock(torch, rank)
    torch.cuda.synchronize()
    raw = [a.elapsed_time(b) for a, b in ev]
    _PARITY.append(raw)
    return sorted(raw), clk


_PARITY = []  # raw per-call times (call order) of every timed sequence, for the even/odd call diagnostic


def _parity_medians(raws):
    ev = sorted(t for r in raws for t in r[0::2])
    od = sorted(t for r in raws for t in r[1::2])
    return {"even_call_median_ms": ev[len(ev) // 2], "odd_call_median_ms": od[len(od) // 2]}


def _stats(seqs):
    flat = sorted(t for s in seqs for t in s)
    meds = [s[len(s) // 2] for s in seqs]
    n = len(flat)
    med = sorted(meds)[len(meds) // 2]
    return {"median_ms": med, "seq_medians_ms": meds, "p10": flat[n // 10], "p90": flat[(9 * n) // 10],
            "noise_floor": (flat[(9 * n) // 10] - flat[n // 10]) / med,
            "seq_dispersion": (max(meds) - min(meds)) / med}


def _time(torch, run, m, rank, dev):
    seqs, clks = [], []
    _PARITY.clear()
    for _ in range(bench.SEQS):
        ts, c = _time_seq(torch, run, m, rank, dev)
        seqs.append(ts)
        clks.append(c)
    t = _stats(seqs)
    t.update(_parity_medians(_PARITY))
    t["sm_clock_under_load_mhz"] = clks
    return t


def _other_half(visible):
    """nvidia-smi view of ALL GPUs + compute apps + loadavg; 'other' = GPUs not in this job's CUDA_VISIBLE_DEVICES."""
    mine = set(v.strip() for v in (visible or "").split(",") if v.strip())
    gpus = []
    for line in bench.sh(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,clocks.sm,memory.used",
                          "--format=csv,noheader,nounits"]).strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            gpus.append({"index": p[0], "uuid": p[1], "util": p[2], "sm": p[3], "mem_mib": p[4],
                         "mine": p[1] in mine})
    apps = [line.strip() for line in bench.sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                               "--format=csv,noheader,nounits"]).strip().splitlines()
            if line.strip()]
    other = [g for g in gpus if not g["mine"]]

    def _num(v):
        try:
            return float(v)
        except ValueError:
            return 0.0
    busy = any(_num(g["util"]) > 0 or _num(g["mem_mib"]) > 1024 for g in other)
    return {"t": round(time.time(), 1), "loadavg": os.getloadavg(), "gpus": gpus,
            "other_apps": [a for a in apps if a.split(",")[0].strip() not in mine], "other_half_busy": busy}


def _flag_state(torch, mapping, expect_push):
    """IPC barrier_in flags of this rank after a chunk. Push layout (customLowPrecisionAllReduceKernels.cu): entry
    [0, tp); pull blocks [tp, tp*(1+128)); scatter tp*(1+128) + lb*tp + src (lb < 16); gather after that, b*tp + src
    (b < tp*16). The host counter (workspace[4*tp+1]) is the last call's flag."""
    from mpi4py import MPI
    from tensorrt_llm._torch.distributed import ops
    from tensorrt_llm._torch.distributed.allreduce_helper import CustomAllReduceHelper as H
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    ws = ops._thread_local.lowprecision_allreduce_workspaces
    bufs, wst = ws[mapping] if mapping in ws else next(iter(ws.values()))
    tp = mapping.tp_size
    counter = int(wst[4 * tp + 1].item())
    nbytes = H.lowprecision_barrier_bytes(tp)
    host = torch.zeros(nbytes // 8, dtype=torch.int64)
    torch.cuda.synchronize()
    err = cudart.cudaMemcpy(host.data_ptr(), bufs[2].local_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    err = err[0] if isinstance(err, tuple) else err
    s = host.tolist()
    P = H.LP_PUSH_CTAS_PER_PEER
    sb = tp * (1 + H.LP_TWOSHOT_MAX_CTAS)
    gb = sb + tp * P
    entry, scatter, gather = s[:tp], s[sb:gb], s[gb:gb + tp * tp * P]
    counters = MPI.COMM_WORLD.allgather(counter)
    ok = all(v == counter for v in entry) and max(s) <= counter and len(set(counters)) == 1
    if expect_push:
        ok = ok and all(v == counter for v in scatter) and all(v == counter for v in gather)
    return {"counter": counter, "counters_all_ranks": counters, "entry": entry,
            "scatter_eq_counter": sum(v == counter for v in scatter), "scatter_slots": len(scatter),
            "gather_eq_counter": sum(v == counter for v in gather), "gather_slots": len(gather),
            "max_slot": max(s), "ok": bool(ok), "memcpy_err": str(err)}


def _kernels(torch, fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = sorted({e.name for e in p.events() if e.device_type.name == "CUDA"})
    return [n[:100] for n in names if "Memcpy" not in n and "Memset" not in n]


def _lib_receipt():
    """Every .so under tensorrt_llm/libs plus every tensorrt_llm shared object mapped in THIS process
    (/proc/self/maps), with byte counts (strings -a equivalent) of the LP kernel symbols / getenv strings."""
    import glob

    import tensorrt_llm
    paths = sorted(os.path.realpath(p) for p in
                   glob.glob(os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "**", "*.so*"),
                             recursive=True))
    mapped = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                p = parts[-1] if len(parts) >= 6 else ""
                if ".so" in p and ("tensorrt_llm" in p or "trtllm" in p or "th_common" in p):
                    rp = os.path.realpath(p)
                    if rp not in mapped:
                        mapped.append(rp)
    except Exception as e:  # noqa
        mapped = [f"ERR {e}"]
    paths += [p for p in mapped if p not in paths and not p.startswith("ERR")]
    out = {}
    for p in paths:
        try:
            st = os.stat(p)
            with open(p, "rb") as f:
                data = f.read()
            out[p] = {"size": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                      "counts": {s: data.count(s.encode()) for s in LIB_STRINGS}}
            del data
        except Exception as e:  # noqa
            out[p] = {"error": str(e)}
    hits = {s: sorted(p for p, v in out.items() if v.get("counts", {}).get(s)) for s in LIB_STRINGS}
    return {"libs": out, "libs_with_string": hits, "mapped": mapped}


def _bitwise_section(torch, mapping, rank, dev):
    from tensorrt_llm._torch.distributed import AllReduceStrategy as S
    rows = {}
    for m in BIT_M:
        run = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        row = {"inputs": 0, "mismatched_inputs": 0, "mismatched_elems": 0, "nonfinite": 0, "e": 0.0,
               "rel_rms_push": 0.0, "rel_rms_pull": 0.0, "pull_deterministic": True}
        for it in range(3):
            heavy = it == 2
            run.x.copy_(bench._gen(torch, m, 700 + it, rank, dev, heavy))
            _set_push(0)
            ref = run().clone()
            ref2 = run().clone()
            _set_push(1)
            out = run().clone()
            torch.cuda.synchronize()
            ne = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
            row["inputs"] += 1
            row["mismatched_inputs"] += int(ne != 0)
            row["mismatched_elems"] += ne
            row["pull_deterministic"] = row["pull_deterministic"] and bool(torch.equal(ref, ref2))
            r, ab = bench._ref(torch, m, 700 + it, dev, heavy)
            e1 = bench._err(torch, out, r, ab)
            e0 = bench._err(torch, ref, r, ab)
            row["nonfinite"] += int(not e1["finite"])
            row["e"] = max(row["e"], e1["e"])
            row["rel_rms_push"] = max(row["rel_rms_push"], e1["rel_rms"])
            row["rel_rms_pull"] = max(row["rel_rms_pull"], e0["rel_rms"])
            del ref, ref2, out, r, ab
        rows[str(m)] = row
        if rank == 0:
            log(f"[bitwise] m={m} {row}")
        del run
        torch.cuda.synchronize()
        bench._barrier()
    _set_push(None)
    return rows


def _seq_section(torch, mapping, rank, dev):
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy as S
    ar = AllReduce(mapping=mapping, strategy=S.LOWPRECISION, dtype=torch.bfloat16)
    pool = {m: [bench._gen(torch, m, 2000 + k, rank, dev, heavy=(k == SEQ_POOL - 1)) for k in range(SEQ_POOL)]
            for m in SEQ_SHAPES}
    _set_push(0)
    refs = {m: [ar(x).clone() for x in pool[m]] for m in SEQ_SHAPES}
    torch.cuda.synchronize()
    _set_push(1)
    bad = torch.zeros(SEQ_CALLS, dtype=torch.int64, device=dev)
    nonfin = torch.zeros(SEQ_CALLS, dtype=torch.int64, device=dev)
    flags, done, t0 = [], 0, time.time()
    bench._barrier()
    flags.append(_flag_state(torch, mapping, expect_push=False))
    while done < SEQ_CALLS:
        n = min(SEQ_CHUNK, SEQ_CALLS - done)
        torch.cuda.synchronize()
        bench._barrier()  # reset barrier: all 4 ranks start the chunk together
        for i in range(done, done + n):  # back to back, no host sync
            m = SEQ_SHAPES[i % len(SEQ_SHAPES)]
            k = (i // len(SEQ_SHAPES)) % SEQ_POOL
            out = ar(pool[m][k])
            bad[i] = (out.view(torch.int16) != refs[m][k].view(torch.int16)).sum()
            nonfin[i] = (~torch.isfinite(out)).sum()
        torch.cuda.synchronize()
        bench._barrier()
        fs = _flag_state(torch, mapping, expect_push=True)
        fs["advanced_by"] = fs["counter"] - flags[-1]["counter"]
        fs["ok"] = fs["ok"] and fs["advanced_by"] == n
        flags.append(fs)
        bench._barrier()
        done += n
    badl = bad.tolist()
    seq = {"calls": SEQ_CALLS, "mismatched_calls": sum(1 for v in badl if v), "mismatched_elems": sum(badl),
           "first_bad": [i for i, v in enumerate(badl) if v][:10], "nonfinite_calls": int((nonfin > 0).sum()),
           "flags_ok": all(f["ok"] for f in flags[1:]), "flag_states": flags,
           "parities_per_shape": {str(m): sorted({(flags[0]["counter"] + 1 + i) % 2
                                                  for i in range(SEQ_CALLS)
                                                  if SEQ_SHAPES[i % len(SEQ_SHAPES)] == m})
                                  for m in SEQ_SHAPES},
           "seconds": round(time.time() - t0, 1)}
    if rank == 0:
        log(f"[seq] { {k: v for k, v in seq.items() if k != 'flag_states'} }")
        for f in flags:
            log(f"[flags] {f}")
    del pool, refs, bad, nonfin
    _set_push(None)
    torch.cuda.empty_cache()
    bench._barrier()
    return seq, ar


def _merge_bitwise(rs):
    out, bad = {}, []
    for m in rs[0]["bitwise"]:
        v = {"mismatched_inputs": sum(r["bitwise"][m]["mismatched_inputs"] for r in rs),
             "mismatched_elems": sum(r["bitwise"][m]["mismatched_elems"] for r in rs),
             "inputs_per_rank": rs[0]["bitwise"][m]["inputs"],
             "nonfinite": sum(r["bitwise"][m]["nonfinite"] for r in rs),
             "rel_rms_push": max(r["bitwise"][m]["rel_rms_push"] for r in rs),
             "rel_rms_pull": max(r["bitwise"][m]["rel_rms_pull"] for r in rs),
             "pull_deterministic": all(r["bitwise"][m]["pull_deterministic"] for r in rs)}
        out[m] = v
        if v["mismatched_inputs"] or v["nonfinite"] or v["rel_rms_push"] > LP_REL_RMS_BOUND:
            bad.append(f"bitwise m={m}: {v}")
    return out, bad


def _merge_seq(rs):
    sq = [r["seq"] for r in rs]
    out = {"calls_per_rank": SEQ_CALLS, "shapes": SEQ_SHAPES, "pool": SEQ_POOL, "reset_every": SEQ_CHUNK,
           "mismatched_calls": sum(s["mismatched_calls"] for s in sq),
           "mismatched_elems": sum(s["mismatched_elems"] for s in sq),
           "first_bad_rank0": sq[0]["first_bad"], "nonfinite_calls": sum(s["nonfinite_calls"] for s in sq),
           "flags_ok_all_ranks": all(s["flags_ok"] for s in sq),
           "parities_per_shape": sq[0]["parities_per_shape"],
           "flag_states_bad": [f for s in sq for f in s["flag_states"][1:] if not f["ok"]][:6],
           "seconds": max(s["seconds"] for s in sq)}
    bad = []
    if out["mismatched_calls"] or out["nonfinite_calls"] or not out["flags_ok_all_ranks"]:
        bad.append("1000-call sequence")
    return out, bad


def rank_dev(env):
    wd = bench._watchdog("dev", 1500)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    S = AllReduceStrategy
    dev = torch.device("cuda", rank)
    res = {"rank": rank, "timing": {}, "bitwise": {}, "seq": {}, "dispatch": {}, "ipc": _lp_ipc_bytes()}

    # ------------------------------------------------------------------ 1. timing
    for m in TIME_M:
        res["timing"][str(m)] = {}
        run = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        for push in (0, 1):
            _set_push(push)
            torch.cuda.synchronize()
            bench._barrier()
            t = bench._time(torch, run, m, rank, dev)
            res["timing"][str(m)][f"push{push}"] = t
            if rank == 0:
                log(f"[time] m={m} push={push} med={t['median_ms']*1e3:.1f}us nf={t['noise_floor']:.3f} "
                    f"seqs={[round(v*1e3, 1) for v in t['seq_medians_ms']]}")
        del run
        torch.cuda.synchronize()
        bench._barrier()
    res["clocks_after_timing"] = bench._gpu_state() if rank == 0 else None

    # ------------------------------------------------------------------ 2. bitwise push vs pull
    res["bitwise"] = _bitwise_section(torch, mapping, rank, dev)

    # ------------------------------------------------------------------ 3. 1000-call sequence
    res["seq"], ar = _seq_section(torch, mapping, rank, dev)

    # ------------------------------------------------------------------ 4. dispatch (CUPTI last)
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    AutoTuner.get().setup_distributed_state(mapping)
    _set_push(1)
    with autotune():
        for m in TUNE_M:
            ar(torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize()
    bench._barrier()
    for m in DISPATCH_M:
        x = bench._gen(torch, m, 5, rank, dev)
        ent = {}
        for push in (1, 0):
            _set_push(push)
            bench._barrier()
            ent[f"push{push}"] = _kernels(torch, lambda: ar(x))
            bench._barrier()
        res["dispatch"][str(m)] = ent
        if rank == 0:
            log(f"[dispatch] m={m} {ent}")
    _set_push(None)
    bench._barrier()
    wd.cancel()
    return res


def _dispatch_ok(m, ent):
    p1, p0 = " ".join(ent["push1"]), " ".join(ent["push0"])
    if m <= 8:
        return all("oneshot_lamport" in p and "lowPrecision" not in p for p in (p1, p0))
    if m * HIDDEN * 2 < (2 << 20):
        return all("nccl" in p.lower() and "lowPrecision" not in p for p in (p1, p0))
    return ("lowPrecisionTwoShotPushAllReduceKernel" in p1 and "lowPrecisionPreprocessKernel" not in p1
            and "lowPrecisionPreprocessKernel" in p0 and "lowPrecisionTwoShotAllReduceKernel" in p0
            and "Push" not in p0)


def main_dev():
    t0 = time.time()
    log(f"[dev] start {time.strftime('%Y-%m-%d %H:%M:%S %Z')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ.pop(ENV, None)
    clocks_before = bench._gpu_state()
    log(clocks_before)
    rs = bench.run_ranks(rank_dev, {"_name": "dev", "_apply_nccl_tuning": "1"})
    clocks_after = bench._gpu_state()
    log(clocks_after)
    out = {"passed": True, "reasons": [], "timing_us": {}, "bitwise": {}, "dispatch": {}, "ipc": rs[0]["ipc"]}
    for m in rs[0]["timing"]:
        row = {}
        for k in rs[0]["timing"][m]:
            per = [r["timing"][m][k]["median_ms"] * 1e3 for r in rs]
            row[k] = {"max_rank_us": round(max(per), 1), "per_rank_us": [round(v, 1) for v in per],
                      "noise_floor": round(max(r["timing"][m][k]["noise_floor"] for r in rs), 4),
                      "seq_dispersion": round(max(r["timing"][m][k]["seq_dispersion"] for r in rs), 4)}
        row["push_vs_pull"] = round(row["push1"]["max_rank_us"] / row["push0"]["max_rank_us"] - 1.0, 4)
        out["timing_us"][m] = row
        log(f"[table] m={m} pull={row['push0']['max_rank_us']} push={row['push1']['max_rank_us']} "
            f"({row['push_vs_pull']:+.1%})")
    out["bitwise"], bad = _merge_bitwise(rs)
    out["reasons"] += bad
    out["seq"], bad = _merge_seq(rs)
    out["reasons"] += bad
    for m in rs[0]["dispatch"]:
        ok = all(_dispatch_ok(int(m), r["dispatch"][m]) for r in rs)
        out["dispatch"][m] = {"ok_all_ranks": ok, "rank0": rs[0]["dispatch"][m]}
        if not ok:
            out["reasons"].append(f"dispatch m={m}")
    out["passed"] = not out["reasons"]
    out["wall_s"] = round(time.time() - t0, 1)
    out["clocks_before"] = clocks_before
    out["clocks_after_timing_rank0"] = rs[0].get("clocks_after_timing")
    out["clocks_after"] = clocks_after
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "lp_push_dev.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    with open(os.path.join(art, "lp_push_dev_raw.json"), "w") as f:
        json.dump(rs, f, default=str)
    summary = {k: out[k] for k in ("passed", "reasons", "ipc", "wall_s")}
    summary["timing_us"] = {m: {"pull": r["push0"]["max_rank_us"], "push": r["push1"]["max_rank_us"],
                                "push_vs_pull": r["push_vs_pull"],
                                "noise": max(r["push0"]["noise_floor"], r["push1"]["noise_floor"])}
                            for m, r in out["timing_us"].items()}
    summary["bitwise_mismatched_inputs"] = {m: v["mismatched_inputs"] for m, v in out["bitwise"].items()}
    summary["seq"] = {k: out["seq"][k] for k in ("mismatched_calls", "nonfinite_calls", "flags_ok_all_ranks",
                                                 "parities_per_shape")}
    summary["dispatch_ok"] = {m: v["ok_all_ranks"] for m, v in out["dispatch"].items()}
    print(json.dumps(summary), flush=True)
    sys.exit(0 if out["passed"] else 4)


def rank_pair(env):
    wd = bench._watchdog("pair", 1300)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    S = AllReduceStrategy
    dev = torch.device("cuda", rank)
    _set_push(None)
    push_tree = _has_push_support()
    res = {"rank": rank, "timing": {}, "ab": {}, "env": os.environ.get(ENV), "ipc": _lp_ipc_bytes(),
           "push_tree": push_tree, "gpu_uuid": str(getattr(torch.cuda.get_device_properties(rank), "uuid", None))}

    from mpi4py import MPI
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    res["neighbour"] = []

    def _nb(tag):  # rank 0 logs the whole node's GPU state between blocks (outside every timed region)
        if rank == 0:
            s = _other_half(visible)
            s["tag"] = tag
            res["neighbour"].append(s)
        bench._barrier()

    # ------------------------------------------------------------------ 1. timing, this tree's default path
    res["timing_repeat"] = {}
    for m in TIME_M:
        run = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        torch.cuda.synchronize()
        _nb(f"timing m={m} start")
        t = _time(torch, run, m, rank, dev)
        res["timing"][str(m)] = t
        _nb(f"timing m={m} end")
        if rank == 0:
            log(f"[pair] m={m} med={t['median_ms']*1e3:.1f}us nf={t['noise_floor']:.3f} "
                f"sd={t['seq_dispersion']:.4f} seqs={[round(v*1e3, 1) for v in t['seq_medians_ms']]} "
                f"even/odd={t['even_call_median_ms']*1e3:.1f}/{t['odd_call_median_ms']*1e3:.1f} "
                f"clk={t['sm_clock_under_load_mhz']}")
        # 13a(c): a trigger (dispersion > 3% or clock < 2200 MHz on any rank) repeats this shape's sequence once
        trig = (max(t["noise_floor"], t["seq_dispersion"]) > NOISE_TRIGGER
                or any(c is not None and c < CLOCK_TRIGGER_MHZ for c in t["sm_clock_under_load_mhz"]))
        if MPI.COMM_WORLD.allreduce(bool(trig), op=MPI.LOR):
            t2 = _time(torch, run, m, rank, dev)
            res["timing_repeat"][str(m)] = t2
            _nb(f"timing repeat m={m} end")
            if rank == 0:
                log(f"[pair-repeat] m={m} med={t2['median_ms']*1e3:.1f}us nf={t2['noise_floor']:.3f} "
                    f"sd={t2['seq_dispersion']:.4f} clk={t2['sm_clock_under_load_mhz']}")
        del run
        torch.cuda.synchronize()
        bench._barrier()

    # ------------------------------------------------------------------ 2. interleaved in-process ABAB
    for m in TIME_M:
        run = bench.Runner(torch, mapping, S.LOWPRECISION, m, graph=False)
        seqs = {"push0": [], "push1": []}
        clks = {"push0": [], "push1": []}
        raws = {"push0": [], "push1": []}
        blocks = []
        for rd in range(AB_ROUNDS):
            for v in (0, 1):  # ABAB: PUSH=0, PUSH=1, PUSH=0, PUSH=1, ...
                k = f"push{v}"
                _set_push(v)
                _PARITY.clear()
                ts, c = _time_seq(torch, run, m, rank, dev)
                seqs[k].append(ts)
                clks[k].append(c)
                raws[k] += _PARITY
                blocks.append({"block": len(blocks), "env": v, "median_ms": ts[len(ts) // 2], "clk": c})
                _nb(f"ab m={m} block={len(blocks) - 1} env={v}")
        _set_push(None)
        ab = {k: _stats(seqs[k]) for k in seqs}
        for k in ab:
            ab[k]["sm_clock_under_load_mhz"] = clks[k]
            ab[k].update(_parity_medians(raws[k]))
        ab["blocks"] = blocks
        res["ab"][str(m)] = ab
        if rank == 0:
            log(f"[ab] m={m} push0={ab['push0']['median_ms']*1e3:.1f}us push1={ab['push1']['median_ms']*1e3:.1f}us "
                f"nf0={ab['push0']['noise_floor']:.3f} nf1={ab['push1']['noise_floor']:.3f} "
                f"blocks={[round(b['median_ms']*1e3, 1) for b in blocks]}")
        del run
        torch.cuda.synchronize()
        bench._barrier()

    # ------------------------------------------------------------------ 3. correctness (push tree only)
    if push_tree:
        res["bitwise"] = _bitwise_section(torch, mapping, rank, dev)
        res["seq"], _ = _seq_section(torch, mapping, rank, dev)

    # ------------------------------------------------------------------ 4. receipt (CUPTI, after all timing)
    x = bench._gen(torch, RECEIPT_M, 5, rank, dev)
    ar = AllReduce(mapping=mapping, strategy=S.LOWPRECISION, dtype=torch.bfloat16)
    ar(x)
    torch.cuda.synchronize()
    res["receipt"] = {}
    for k, v in (("unset", None), ("push1", 1), ("push0", 0)):
        _set_push(v)
        bench._barrier()
        res["receipt"][k] = _kernels(torch, lambda: ar(x))
        bench._barrier()
    _set_push(None)
    if rank == 0:
        log(f"[receipt] M={RECEIPT_M} {res['receipt']}")
        res["lib"] = _lib_receipt()
        for s, files in res["lib"]["libs_with_string"].items():
            log(f"[lib] {s}: {files if files else 'NO HIT'}")
        log(f"[lib] mapped tensorrt_llm .so: {res['lib']['mapped']}")
        for p, v in res["lib"]["libs"].items():
            log(f"[lib]   {p} {v}")
    bench._barrier()
    wd.cancel()
    return res


def _pull_ok(s):
    return "lowPrecisionPreprocessKernel" in s and "lowPrecisionTwoShotAllReduceKernel" in s and "Push" not in s


def _push_ok(s):
    return ("lowPrecisionTwoShotPushAllReduceKernel" in s and "lowPrecisionPreprocessKernel" not in s
            and "lowPrecisionTwoShotAllReduceKernel" not in s)


def main_pair():
    os.environ.pop(ENV, None)
    log(f"[pair] start {time.strftime('%Y-%m-%d %H:%M:%S %Z')} host={bench.sh(['hostname']).strip()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} load={os.getloadavg()}")
    log(bench._gpu_state())
    log(bench.sh(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv"]))
    log(bench.sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]))
    load_start = os.getloadavg()
    stop, samples = threading.Event(), []
    th = threading.Thread(target=bench._clock_sampler, args=(stop, samples), daemon=True)
    th.start()
    rs = bench.run_ranks(rank_pair, {"_name": "pair", "_apply_nccl_tuning": "1"})
    stop.set()
    th.join(timeout=10)
    load_end = os.getloadavg()
    log(bench._gpu_state())
    clocks = {}
    for p in samples:
        try:
            c, util = int(float(p[1])), int(float(p[4]))
        except ValueError:
            continue
        d = clocks.setdefault(p[0], {"sm_min_busy": None, "sm_max": 0, "n": 0})
        d["n"] += 1
        d["sm_max"] = max(d["sm_max"], c)
        if util > 0:
            d["sm_min_busy"] = c if d["sm_min_busy"] is None else min(d["sm_min_busy"], c)
    batch = {}
    for m in rs[0]["timing"]:
        per = [r["timing"][m]["median_ms"] for r in rs]
        clk = [c for r in rs for c in r["timing"][m]["sm_clock_under_load_mhz"] if c is not None]
        batch[m] = {"kernel_ms": max(per),
                    "noise_floor": max(max(r["timing"][m]["noise_floor"], r["timing"][m]["seq_dispersion"])
                                       for r in rs),
                    "per_rank_median_us": [round(v * 1e3, 1) for v in per],
                    "rank_dispersion": round((max(per) - min(per)) / max(per), 4),
                    "seq_medians_us_max_rank": [round(max(r["timing"][m]["seq_medians_ms"][s] for r in rs) * 1e3, 1)
                                                for s in range(bench.SEQS)],
                    "sm_clock_under_load_min_mhz": min(clk) if clk else None,
                    "sm_clock_under_load_by_rank": [r["timing"][m]["sm_clock_under_load_mhz"] for r in rs],
                    "even_odd_call_median_us_max_rank": [
                        round(max(r["timing"][m]["even_call_median_ms"] for r in rs) * 1e3, 1),
                        round(max(r["timing"][m]["odd_call_median_ms"] for r in rs) * 1e3, 1)]}
        if m in rs[0]["timing_repeat"]:
            per2 = [r["timing_repeat"][m]["median_ms"] for r in rs]
            batch[m]["repeat"] = {
                "kernel_ms": max(per2),
                "noise_floor": max(max(r["timing_repeat"][m]["noise_floor"], r["timing_repeat"][m]["seq_dispersion"])
                                   for r in rs),
                "seq_dispersion": max(r["timing_repeat"][m]["seq_dispersion"] for r in rs),
                "sm_clock_under_load_min_mhz": min((c for r in rs for c in r["timing_repeat"][m]["sm_clock_under_load_mhz"]
                                                    if c is not None), default=None)}
        batch[m]["seq_dispersion"] = max(r["timing"][m]["seq_dispersion"] for r in rs)
        batch[m]["p90_p10"] = max(r["timing"][m]["noise_floor"] for r in rs)
    ab = {}
    for m in rs[0]["ab"]:
        row = {}
        for k in ("push0", "push1"):
            per = [r["ab"][m][k]["median_ms"] * 1e3 for r in rs]
            clk = [c for r in rs for c in r["ab"][m][k]["sm_clock_under_load_mhz"] if c is not None]
            row[k] = {"max_rank_us": round(max(per), 1),
                      "noise_floor": round(max(max(r["ab"][m][k]["noise_floor"], r["ab"][m][k]["seq_dispersion"])
                                               for r in rs), 4),
                      "seq_dispersion": round(max(r["ab"][m][k]["seq_dispersion"] for r in rs), 4),
                      "block_medians_us_max_rank": [round(max(r["ab"][m][k]["seq_medians_ms"][i] for r in rs) * 1e3, 1)
                                                    for i in range(AB_ROUNDS)],
                      "even_odd_call_median_us_max_rank": [
                          round(max(r["ab"][m][k]["even_call_median_ms"] for r in rs) * 1e3, 1),
                          round(max(r["ab"][m][k]["odd_call_median_ms"] for r in rs) * 1e3, 1)],
                      "sm_clock_under_load_min_mhz": min(clk) if clk else None}
        row["push1_vs_push0"] = round(row["push1"]["max_rank_us"] / row["push0"]["max_rank_us"] - 1.0, 4)
        ab[m] = row
        log(f"[ab-table] m={m} push0={row['push0']['max_rank_us']} push1={row['push1']['max_rank_us']} "
            f"({row['push1_vs_push0']:+.1%}) noise={max(row['push0']['noise_floor'], row['push1']['noise_floor'])} "
            f"blocks0={row['push0']['block_medians_us_max_rank']} blocks1={row['push1']['block_medians_us_max_rank']}")
    nb = rs[0]["neighbour"]
    neighbour = {"blocks_logged": len(nb), "other_half_busy_blocks": sum(1 for s in nb if s["other_half_busy"]),
                 "loadavg_1m_min": min(s["loadavg"][0] for s in nb), "loadavg_1m_max": max(s["loadavg"][0] for s in nb),
                 "my_sm_clock_min_mhz_logged": min((float(g["sm"]) for s in nb for g in s["gpus"]
                                                    if g["mine"] and g["sm"].replace(".", "").isdigit()),
                                                   default=None),
                 "other_apps_seen": sorted({a for s in nb for a in s["other_apps"]})}
    log(f"[neighbour] {neighbour}")
    push_tree = rs[0]["push_tree"]
    receipt = {"M": RECEIPT_M, "unset_all_ranks": [r["receipt"]["unset"] for r in rs],
               "push1_all_ranks": [r["receipt"]["push1"] for r in rs],
               "push0_all_ranks": [r["receipt"]["push0"] for r in rs]}

    def _path(key):
        ss = [" ".join(k) for k in receipt[key]]
        return "push" if all(_push_ok(s) for s in ss) else "pull" if all(_pull_ok(s) for s in ss) else "unexpected"
    receipt["default_path"] = _path("unset_all_ranks")
    receipt["push1_path"] = _path("push1_all_ranks")
    receipt["push0_path"] = _path("push0_all_ranks")
    exp = "push" if push_tree else "pull"
    receipt["expected"] = {"default": exp, "push1": exp, "push0": "pull"}
    receipt["ok"] = (receipt["default_path"] == exp and receipt["push1_path"] == exp
                     and receipt["push0_path"] == "pull")
    reasons = [] if receipt["ok"] else [f"kernel receipt: {receipt['default_path']}/{receipt['push1_path']}/"
                                        f"{receipt['push0_path']}"]
    libs = rs[0].get("lib", {}).get("libs_with_string", {})
    if not libs.get("TRTLLM_LP_AR_TWOSHOT_CTAS"):
        reasons.append("lib receipt broken: TRTLLM_LP_AR_TWOSHOT_CTAS found in no .so")
    if push_tree and not libs.get("lowPrecisionTwoShotPushAllReduceKernel"):
        reasons.append("lib receipt: push kernel symbol in no .so of the push tree")
    corr, bw, sq = None, None, None
    if push_tree:
        bw, b1 = _merge_bitwise(rs)
        sq, b2 = _merge_seq(rs)
        reasons += b1 + b2
        corr = {"bitwise_mismatched_inputs": {m: v["mismatched_inputs"] for m, v in bw.items()},
                "bitwise_rel_rms_push_max": max(v["rel_rms_push"] for v in bw.values()),
                "seq": {k: sq[k] for k in ("mismatched_calls", "mismatched_elems", "nonfinite_calls",
                                           "flags_ok_all_ranks", "parities_per_shape", "seconds")}}
    out = {"batch": batch, "output": {"class": "exact" if not reasons else "failed"}, "passed": not reasons,
           "reasons": reasons, "tree": "push" if push_tree else "pull", "kernel_receipt": receipt,
           "lib_strings": libs, "neighbour": neighbour,
           "ab_in_process": ab, "correctness": corr, "ipc": rs[0]["ipc"],
           "gpu_uuid_by_rank": [r["gpu_uuid"] for r in rs],
           "load_average": {"start": load_start, "end": load_end}, "sm_clocks_by_uuid": clocks}
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "lp_push_pair.json"), "w") as f:
        json.dump({"summary": out, "lib": rs[0].get("lib"), "bitwise": bw, "seq": sq}, f, indent=1, default=str)
    with open(os.path.join(art, "lp_push_pair_raw.json"), "w") as f:
        json.dump(rs, f, default=str)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        main_dev()
    elif mode == "pair":
        main_pair()
    else:
        raise SystemExit(f"unknown mode {mode}")
