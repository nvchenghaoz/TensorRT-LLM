"""4-GPU bench + checks for LOWPRECISION (FP8-payload two-shot) all-reduce at prefill sizes (idea 94).

Modes (each config spawns 4 MPI ranks through mpi4py.futures, so NCCL env is fresh per config):
  sweep  step 3: topology gate, LP vs NCCL default vs NCCL PHB at [2048, 6144] x 5120 bf16 (20 warmup + 100 timed,
         median of 3, inputs re-created from the seed before each timed sequence), untimed every-iteration error
         pass (same counts and reset points) on realistic and heavy-tailed inputs, 1000-call robustness pass with a
         4-rank reset barrier every 250 calls (every call checked), and the routing check: M=1..16 (CUDA graph) and
         128 rows under strategy LOWPRECISION must be BITWISE equal to strategy AUTO (same path as main), with the
         kernel names each call launches. Writes $AUTO_DEV_ARTIFACTS/lp_ar_table.json and prints it last.
  pair   what each tree serves: the strategy from its own experiments/ascent-config/engine.json (AUTO on main,
         LOWPRECISION on the branch), PHB as BaseWorker applies it; times 2048/6144 rows eager and decode M=1,2,4
         in a CUDA graph. Last stdout line: evidence JSON.
"""
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback

HIDDEN = 5120
WORLD = 4
BIG_M = [2048, 6144]
DECODE_M = [1, 2, 4, 8, 16]
WARMUP = 20
ITERS = 100
SEQS = 3
NCCL_BOUND = WORLD * 2.0 ** -8  # method 491 B: e <= 2^-6 at TP4 for the bf16 NCCL arms
LP_REL_RMS_BOUND = 2.0 ** -4    # sanity bound for the FP8 payload (3 mantissa bits, two quantizations)
ROBUST_CALLS = 1000
ROBUST_CHUNK = 250
HEAVY_CHANNELS = (7, 911, 2048, 3333, 4095, 5000)  # a few channels 50x larger, same on every rank
WATCHDOG_S = 600


def log(*a):
    print(*a, flush=True)


def _watchdog(tag, seconds=WATCHDOG_S):
    def fire():
        sys.stderr.write(f"[watchdog:{tag}] {seconds}s elapsed, dumping stacks and exiting\n")
        for tid, frame in sys._current_frames().items():
            sys.stderr.write(f"--- thread {tid}\n" + "".join(traceback.format_stack(frame)))
        sys.stderr.flush()
        os._exit(2)
    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()
    return t


# ----------------------------------------------------------------------------------------------- rank-side code
def _setup(env):
    if env.get("_drop_NCCL_P2P_LEVEL"):
        os.environ.pop("NCCL_P2P_LEVEL", None)
    os.environ.update({k: v for k, v in env.items() if not k.startswith("_")})
    import torch
    from tensorrt_llm._utils import mpi_rank
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    applied = None
    if env.get("_apply_nccl_tuning"):
        from tensorrt_llm._torch.distributed import nccl_tuning
        applied = nccl_tuning.apply_pcie_nccl_defaults(list(range(WORLD)))
    from tensorrt_llm.mapping import Mapping
    mapping = Mapping(world_size=WORLD, rank=rank, gpus_per_node=WORLD, tp_size=WORLD)
    return torch, rank, mapping, applied


def _barrier():
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()


def _gen(torch, m, seed, r, dev, heavy=False):
    g = torch.Generator(device=dev)
    g.manual_seed(1000003 * m + 7919 * seed + 104729 * r + 17)
    x = torch.randn((m, HIDDEN), generator=g, device=dev, dtype=torch.float32)
    if heavy:
        x[:, list(HEAVY_CHANNELS)] *= 50.0
    return x.to(torch.bfloat16)


def _ref(torch, m, seed, dev, heavy=False):
    ins = [_gen(torch, m, seed, r, dev, heavy).float() for r in range(WORLD)]
    ref = ins[0] + ins[1] + ins[2] + ins[3]
    absum = ins[0].abs() + ins[1].abs() + ins[2].abs() + ins[3].abs()
    return ref, absum


def _err(torch, out, ref, absum):
    o = out.float()
    finite = bool(torch.isfinite(o).all().item())
    d = (o - ref).abs()
    return {"e": (d / absum.clamp_min(1e-30)).max().item(), "max_abs": d.max().item(),
            "mean_abs": d.mean().item(),
            "rel_rms": (d.square().sum().sqrt() / ref.square().sum().sqrt().clamp_min(1e-30)).item(),
            "finite": finite}


class Runner:
    def __init__(self, torch, mapping, strategy, m, graph):
        from tensorrt_llm._torch.distributed import AllReduce
        self.torch = torch
        self.ar = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
        self.x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
        self.graph = None
        self.out = self.ar(self.x)
        torch.cuda.synchronize()
        if graph:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.out = self.ar(self.x)
                with torch.cuda.graph(g, stream=s):
                    self.out = self.ar(self.x)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = g

    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
            return self.out
        self.out = self.ar(self.x)
        return self.out


def _time(torch, run, m, rank, dev):
    seq_meds, flat = [], []
    for s in range(SEQS):
        run.x.copy_(_gen(torch, m, 0, rank, dev))  # re-created from the seed before each timed sequence
        torch.cuda.synchronize()
        _barrier()
        for _ in range(WARMUP):
            run()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
        for i in range(ITERS):
            ev[i][0].record()
            run()
            ev[i][1].record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) for a, b in ev)
        flat += ts
        seq_meds.append(ts[len(ts) // 2])
    flat.sort()
    n = len(flat)
    med = sorted(seq_meds)[SEQS // 2]
    return {"median_ms": med, "seq_medians_ms": seq_meds, "p10": flat[n // 10], "p90": flat[(9 * n) // 10],
            "noise_floor": (flat[(9 * n) // 10] - flat[n // 10]) / med,
            "seq_dispersion": (max(seq_meds) - min(seq_meds)) / med}


def _check(torch, run, m, rank, dev, heavy):
    """Untimed sequence with the timing's counts; reset (fresh input) before every call; every call checked."""
    agg = {"e": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "rel_rms": 0.0}
    nonfinite = 0
    per_iter_e = []
    for it in range(WARMUP + ITERS):
        run.x.copy_(_gen(torch, m, it + 1, rank, dev, heavy))
        x_keep = run.x.clone()
        out = run()
        torch.cuda.synchronize()
        assert torch.equal(run.x, x_keep), "all-reduce modified its input"
        ref, absum = _ref(torch, m, it + 1, dev, heavy)
        r = _err(torch, out, ref, absum)
        per_iter_e.append(r["e"])
        for k in ("e", "max_abs", "rel_rms"):
            agg[k] = max(agg[k], r[k])
        agg["mean_abs"] += r["mean_abs"] / (WARMUP + ITERS)
        nonfinite += int(not r["finite"])
    # determinism: the same input twice gives the same bits
    run.x.copy_(_gen(torch, m, 1, rank, dev, heavy))
    a = run().clone()
    b = run().clone()
    torch.cuda.synchronize()
    agg.update({"iters": WARMUP + ITERS, "nonfinite_iters": nonfinite, "deterministic": bool(torch.equal(a, b)),
                "e_first": per_iter_e[0], "e_last": per_iter_e[-1]})
    return agg


def _lp_flag_state(torch, mapping):
    """IPC flag state of the LOWPRECISION workspace (ops.allocate_low_presicion_allreduce_workspace):
    host counter workspace[4*tp+1] (incremented once per call in LowPrecisionAllReduceParams::deserialize, the
    barrier_flag of that call) and this rank's barrier_in IPC buffer (uint64): slots [0, tp) are written by
    multi_gpu_barrier, slots [tp + b*tp + r] by block b's inter-stage barrier."""
    from mpi4py import MPI
    from tensorrt_llm._ipc_utils import IpcMemory
    from tensorrt_llm._torch.distributed import ops
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    ws = ops._thread_local.lowprecision_allreduce_workspaces
    bufs, wst = ws[mapping] if mapping in ws else next(iter(ws.values()))
    tp = mapping.tp_size
    counter = int(wst[4 * tp + 1].item())
    nbytes = IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * tp * 2
    host = torch.zeros(nbytes // 8, dtype=torch.int64)
    torch.cuda.synchronize()
    err = cudart.cudaMemcpy(host.data_ptr(), bufs[2].local_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    err = err[0] if isinstance(err, tuple) else err
    slots = host.tolist()
    entry = slots[:tp]
    block = slots[tp:]
    used_blocks = [b for b in range(len(block) // tp) if any(block[b * tp + r] for r in range(tp))]
    counters = MPI.COMM_WORLD.allgather(counter)
    ok = (all(v == counter for v in entry)  # every peer signalled this rank with the last call's flag
          and all(block[b * tp + r] == counter for b in used_blocks for r in range(tp))
          and max(slots) <= counter        # nothing already holds counter+1, the next call's flag
          and len(set(counters)) == 1)     # every rank's next call uses the same flag
    return {"counter": counter, "counters_all_ranks": counters, "entry_slots": entry, "blocks_used": len(used_blocks),
            "max_slot": max(slots), "min_used_slot": min([v for v in slots if v] or [0]), "ok": bool(ok),
            "memcpy_err": str(err)}


def _robust(torch, mapping, strategy, m, rank, dev):
    run = Runner(torch, mapping, strategy, m, graph=False)
    worst = {"e": 0.0, "rel_rms": 0.0}
    nonfinite, done, chunks = 0, 0, []
    torch.cuda.synchronize()
    _barrier()
    flag0 = _lp_flag_state(torch, mapping)
    flags = [flag0]
    flags_ok = flag0["ok"]
    bitwise_mismatch = 0
    t0 = time.time()
    while done < ROBUST_CALLS:
        n = min(ROBUST_CHUNK, ROBUST_CALLS - done)
        bank = [_gen(torch, m, 5000 + done + i, rank, dev, heavy=(i % 2 == 1)) for i in range(n)]
        outs = torch.empty((n, m, HIDDEN), dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()
        _barrier()  # reset barrier: all 4 ranks start the chunk together
        for i in range(n):  # enqueued back to back, no host sync
            run.x.copy_(bank[i])
            outs[i].copy_(run())
        torch.cuda.synchronize()
        _barrier()
        # before the next reset barrier: the IPC flags must hold the values the next call expects
        fs = _lp_flag_state(torch, mapping)
        fs["advanced_by"] = fs["counter"] - flags[-1]["counter"]
        fs["ok"] = fs["ok"] and fs["advanced_by"] == n
        flags.append(fs)
        flags_ok = flags_ok and fs["ok"]
        ce = 0.0
        for i in range(n):
            ref, absum = _ref(torch, m, 5000 + done + i, dev, heavy=(i % 2 == 1))
            r = _err(torch, outs[i], ref, absum)
            worst["e"] = max(worst["e"], r["e"])
            worst["rel_rms"] = max(worst["rel_rms"], r["rel_rms"])
            ce = max(ce, r["rel_rms"])
            nonfinite += int(not r["finite"])
        # repeated-execution: every 25th input re-run alone (synced) must give the same bits as inside the chain
        mism = 0
        for i in range(0, n, 25):
            run.x.copy_(bank[i])
            o = run().clone()
            torch.cuda.synchronize()
            mism += int(not torch.equal(o, outs[i]))
        _barrier()
        fs2 = _lp_flag_state(torch, mapping)
        fs2["advanced_by"] = fs2["counter"] - fs["counter"]
        fs2["ok"] = fs2["ok"] and fs2["advanced_by"] == len(range(0, n, 25))
        flags.append(fs2)
        flags_ok = flags_ok and fs2["ok"]
        bitwise_mismatch += mism
        chunks.append({"calls": n, "max_rel_rms": ce, "rerun_bitwise_mismatch": mism})
        del bank, outs
        done += n
    return {"calls": ROBUST_CALLS, "nonfinite_calls": nonfinite, "max_e": worst["e"],
            "max_rel_rms": worst["rel_rms"], "chunks": chunks, "seconds": round(time.time() - t0, 1),
            "flags_ok": bool(flags_ok), "flag_states": flags, "rerun_bitwise_mismatch": bitwise_mismatch}


def _kernel_names(torch, run):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        run()
        torch.cuda.synchronize()
    names = sorted({e.name for e in p.events() if e.device_type.name == "CUDA"})
    return [n[:90] for n in names if "Memcpy" not in n and "Memset" not in n]


def rank_sweep(env, rows):
    wd = _watchdog(f"rank-{env.get('_name')}")
    torch, rank, mapping, applied = _setup(env)
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    dev = torch.device("cuda", rank)
    res = {"rank": rank, "applied": applied, "timing": {}, "check": {}, "robust": {}, "routing": {},
           "nccl_env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "TLLM_PCIE"))}}
    for row in rows:
        strat = getattr(AllReduceStrategy, row)
        res["timing"][row], res["check"][row] = {}, {}
        for m in BIG_M:
            run = Runner(torch, mapping, strat, m, graph=False)
            if rank == 0:
                res.setdefault("kernels", {})[f"{row}_{m}"] = _kernel_names(torch, run)
            else:
                run()  # keep the call count in step with rank 0's profiled call
            t = _time(torch, run, m, rank, dev)
            res["timing"][row][str(m)] = t
            res["check"][row][str(m)] = {"realistic": _check(torch, run, m, rank, dev, False),
                                         "heavy": _check(torch, run, m, rank, dev, True)}
            if rank == 0:
                c = res["check"][row][str(m)]
                log(f"[{env.get('_name')}] {row} m={m} med={t['median_ms']*1e3:.1f}us nf={t['noise_floor']:.3f} "
                    f"seqs={[round(v*1e3,1) for v in t['seq_medians_ms']]} "
                    f"real e={c['realistic']['e']:.3g} rr={c['realistic']['rel_rms']:.3g} "
                    f"heavy e={c['heavy']['e']:.3g} rr={c['heavy']['rel_rms']:.3g} det={c['realistic']['deterministic']}")
            del run
            torch.cuda.synchronize()
    if env.get("_robust"):
        r = _robust(torch, mapping, AllReduceStrategy.LOWPRECISION, 2048, rank, dev)
        res["robust"]["LOWPRECISION_2048"] = r
        if rank == 0:
            log(f"[robust] LP m=2048 calls={r['calls']} nonfinite={r['nonfinite_calls']} "
                f"max_rel_rms={r['max_rel_rms']:.3g} max_e={r['max_e']:.3g} {r['seconds']}s "
                f"flags_ok={r['flags_ok']} rerun_mismatch={r['rerun_bitwise_mismatch']}")
            for s in r["flag_states"]:
                log(f"[flags] {s}")
    if env.get("_routing"):
        for m in DECODE_M + [128, 2048]:
            graph = m in DECODE_M
            a = Runner(torch, mapping, AllReduceStrategy.AUTO, m, graph)
            b = Runner(torch, mapping, AllReduceStrategy.LOWPRECISION, m, graph)
            same = True
            for it in range(5):
                x = _gen(torch, m, 900 + it, rank, dev, heavy=(it % 2 == 1))
                a.x.copy_(x)
                b.x.copy_(x)
                oa = a().clone()
                ob = b().clone()
                torch.cuda.synchronize()
                same = same and bool(torch.equal(oa, ob))
            entry = {"bitwise_equal_to_AUTO": same, "graph": graph}
            # kernel names of one eager call under each strategy
            ea = Runner(torch, mapping, AllReduceStrategy.AUTO, m, False)
            eb = Runner(torch, mapping, AllReduceStrategy.LOWPRECISION, m, False)
            if rank == 0:
                entry["kernels_AUTO"] = _kernel_names(torch, ea)
                entry["kernels_LP"] = _kernel_names(torch, eb)
            else:
                ea()
                eb()
            torch.cuda.synchronize()
            res["routing"][str(m)] = entry
            if rank == 0:
                log(f"[routing] m={m} graph={graph} LP==AUTO bitwise: {same} AUTO={entry['kernels_AUTO']} "
                    f"LP={entry['kernels_LP']}")
            del a, b, ea, eb
            _barrier()
    _barrier()
    wd.cancel()
    return res


def _served_strategy(tree):
    p = os.path.join(tree, "experiments", "ascent-config", "engine.json")
    try:
        with open(p) as f:
            return str(json.load(f).get("allreduce_strategy", "AUTO")).upper()
    except Exception:
        return "AUTO"


def rank_pair(env):
    wd = _watchdog("rank-pair")
    torch, rank, mapping, applied = _setup(env)
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    dev = torch.device("cuda", rank)
    strat = getattr(AllReduceStrategy, env["_strategy"])
    # autotune like the served warm-up does
    AutoTuner.get().setup_distributed_state(mapping)
    ar = AllReduce(mapping=mapping, strategy=strat, dtype=torch.bfloat16)
    with autotune():
        for m in [1, 2, 4] + BIG_M:
            ar(torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize()
    res = {"rank": rank, "applied": applied, "strategy": env["_strategy"], "timing": {}, "err": {}}
    for m in [1, 2, 4] + BIG_M:
        run = Runner(torch, mapping, strat, m, graph=m < 128)
        res["timing"][str(m)] = _time(torch, run, m, rank, dev)
        run.x.copy_(_gen(torch, m, 1, rank, dev))
        out = run().clone()
        ref, absum = _ref(torch, m, 1, dev)
        res["err"][str(m)] = _err(torch, out, ref, absum)
        if rank == 0:
            t = res["timing"][str(m)]
            log(f"[pair:{env['_strategy']}] m={m} med={t['median_ms']*1e3:.1f}us nf={t['noise_floor']:.3f} "
                f"rel_rms={res['err'][str(m)]['rel_rms']:.3g}")
        del run
    _barrier()
    wd.cancel()
    return res


# ----------------------------------------------------------------------------------------------- driver
def run_ranks(fn, *args):
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    t0 = time.time()
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        out = list(ex.map(fn, *[[a] * WORLD for a in args]))
    log(f"[driver] {fn.__name__} {args[0].get('_name', '')} done in {time.time()-t0:.0f}s")
    return sorted(out, key=lambda r: r["rank"])


def sh(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                              timeout=60).stdout
    except Exception as e:  # noqa
        return f"ERR {e}"


def _gpu_state():
    return sh(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
               "clocks_throttle_reasons.active", "--format=csv"])


def _max_rank(results, row, m, key="median_ms"):
    return max(r["timing"][row][m][key] for r in results)


def _merge_check(results, row, m, kind):
    cs = [r["check"][row][m][kind] for r in results]
    return {"e": max(c["e"] for c in cs), "max_abs": max(c["max_abs"] for c in cs),
            "mean_abs": sum(c["mean_abs"] for c in cs) / len(cs), "rel_rms": max(c["rel_rms"] for c in cs),
            "nonfinite_iters": sum(c["nonfinite_iters"] for c in cs),
            "deterministic": all(c["deterministic"] for c in cs), "iters": cs[0]["iters"]}


def main_sweep():
    wd = _watchdog("driver", 880)
    log(_gpu_state())
    log(sh(["nvidia-smi", "topo", "-m"]))
    import torch
    peer = [[int(torch.cuda.can_device_access_peer(i, j)) for j in range(WORLD) if j != i] for i in range(WORLD)]
    gate = all(all(row) for row in peer)
    log(f"[gate] can_device_access_peer on 12 ordered pairs: {peer} -> {gate}")
    if not gate:
        print(json.dumps({"passed": False, "gate": "peer access missing", "peer": peer}))
        return
    base = sh(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"]).strip()
    cfgs = {
        "nccl_default": ({"_name": "nccl_default", "_drop_NCCL_P2P_LEVEL": "1"}, ["NCCL"]),
        "phb": ({"_name": "phb", "NCCL_P2P_LEVEL": "PHB"}, ["NCCL"]),
        "lp": ({"_name": "lp", "NCCL_P2P_LEVEL": "PHB", "_robust": "1", "_routing": "1"}, ["LOWPRECISION"]),
    }
    res = {}
    for name, (env, rows) in cfgs.items():
        res[name] = run_ranks(rank_sweep, env, rows)
        log(_gpu_state())
    table = {"idea": 94, "head_commit": base, "branch_base": "ab62f1d6fe7bff39b47d00ef3451e88178119aa0",
             "shape": "[M,5120] bf16, 4 ranks, per call, max over ranks of the median of 3 x 100", "peer_gate": gate}
    passed = True
    reasons = []
    for m in map(str, BIG_M):
        t_lp = _max_rank(res["lp"], "LOWPRECISION", m)
        t_phb = _max_rank(res["phb"], "NCCL", m)
        t_def = _max_rank(res["nccl_default"], "NCCL", m)
        disp = {k: max(r["timing"][row][m]["seq_dispersion"] for r in res[c])
                for k, c, row in (("LP", "lp", "LOWPRECISION"), ("PHB", "phb", "NCCL"),
                                  ("NCCL_default", "nccl_default", "NCCL"))}
        nf = {k: max(r["timing"][row][m]["noise_floor"] for r in res[c])
              for k, c, row in (("LP", "lp", "LOWPRECISION"), ("PHB", "phb", "NCCL"),
                                ("NCCL_default", "nccl_default", "NCCL"))}
        table[m] = {
            "t_LP_us": round(t_lp * 1e3, 1), "t_PHB_us": round(t_phb * 1e3, 1),
            "t_NCCL_default_us": round(t_def * 1e3, 1),
            "seq_medians_us": {
                "LP": [round(max(r["timing"]["LOWPRECISION"][m]["seq_medians_ms"][s] for r in res["lp"]) * 1e3, 1)
                       for s in range(SEQS)],
                "PHB": [round(max(r["timing"]["NCCL"][m]["seq_medians_ms"][s] for r in res["phb"]) * 1e3, 1)
                        for s in range(SEQS)],
                "NCCL_default": [round(max(r["timing"]["NCCL"][m]["seq_medians_ms"][s]
                                           for r in res["nccl_default"]) * 1e3, 1) for s in range(SEQS)]},
            "seq_dispersion": disp, "noise_floor_p90_p10": nf,
            "LP_saving_vs_PHB_us": round((t_phb - t_lp) * 1e3, 1),
            "err": {
                "LP": {k: _merge_check(res["lp"], "LOWPRECISION", m, k) for k in ("realistic", "heavy")},
                "NCCL_PHB": {k: _merge_check(res["phb"], "NCCL", m, k) for k in ("realistic", "heavy")},
                "NCCL_default": {k: _merge_check(res["nccl_default"], "NCCL", m, k) for k in ("realistic", "heavy")},
            },
            "kernels": {"LP": res["lp"][0].get("kernels", {}).get(f"LOWPRECISION_{m}"),
                        "PHB": res["phb"][0].get("kernels", {}).get(f"NCCL_{m}")},
        }
        for arm in ("NCCL_PHB", "NCCL_default"):
            for k, v in table[m]["err"][arm].items():
                if v["e"] > NCCL_BOUND or v["nonfinite_iters"]:
                    passed = False
                    reasons.append(f"{arm} {m} {k} e={v['e']:.3g} over 2^-6 (harness wrong)")
        for k, v in table[m]["err"]["LP"].items():
            if v["rel_rms"] > LP_REL_RMS_BOUND or v["nonfinite_iters"] or not v["deterministic"]:
                passed = False
                reasons.append(f"LP {m} {k}: rel_rms={v['rel_rms']:.3g} nonfinite={v['nonfinite_iters']} "
                               f"det={v['deterministic']}")
    rob = [r["robust"]["LOWPRECISION_2048"] for r in res["lp"]]
    table["robust_LP_2048"] = {"calls_per_rank": ROBUST_CALLS, "reset_barrier_every": ROBUST_CHUNK,
                               "nonfinite_calls": sum(x["nonfinite_calls"] for x in rob),
                               "max_rel_rms": max(x["max_rel_rms"] for x in rob),
                               "max_e": max(x["max_e"] for x in rob), "seconds": max(x["seconds"] for x in rob),
                               "no_hang": True,
                               "ipc_flags_ok_all_ranks": all(x["flags_ok"] for x in rob),
                               "rerun_bitwise_mismatch": sum(x["rerun_bitwise_mismatch"] for x in rob),
                               "flag_states_rank0": rob[0]["flag_states"],
                               "flag_states_bad": [(x.get("chunk"), s) for x in rob for s in x["flag_states"]
                                                   if not s["ok"]][:8]}
    if table["robust_LP_2048"]["nonfinite_calls"] or table["robust_LP_2048"]["max_rel_rms"] > LP_REL_RMS_BOUND:
        passed = False
        reasons.append("LP robustness")
    if not table["robust_LP_2048"]["ipc_flags_ok_all_ranks"]:
        passed = False
        reasons.append("LP IPC flag state wrong before a reset")
    if table["robust_LP_2048"]["rerun_bitwise_mismatch"]:
        passed = False
        reasons.append("LP re-run differs bitwise from the in-chain output")
    routing = {}
    for m, entry in res["lp"][0]["routing"].items():
        eq = all(r["routing"][m]["bitwise_equal_to_AUTO"] for r in res["lp"])
        routing[m] = {"LP_bitwise_equal_AUTO_all_ranks": eq, "graph": entry["graph"],
                      "kernels_AUTO": entry.get("kernels_AUTO"), "kernels_LP": entry.get("kernels_LP")}
        if int(m) < 2048 and not eq:
            passed = False
            reasons.append(f"routing m={m}: LP strategy differs from AUTO")
    table["routing"] = routing
    table["stop_rule"] = {"LP_faster_than_PHB_at_2048": table["2048"]["t_LP_us"] < table["2048"]["t_PHB_us"]}
    table["passed"] = passed
    table["reasons"] = reasons
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "lp_ar_table.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(art, "lp_ar_raw.json"), "w") as f:
        json.dump(res, f, default=str)
    wd.cancel()
    print(json.dumps(table))


def main_pair():
    wd = _watchdog("driver", 880)
    log(_gpu_state())
    tree = os.getcwd()
    strategy = _served_strategy(tree)
    log(f"[pair] tree={tree} served allreduce_strategy={strategy}")
    res = run_ranks(rank_pair, {"_name": "pair", "_apply_nccl_tuning": "1", "_strategy": strategy})
    log(_gpu_state())
    batch = {}
    for m in res[0]["timing"]:
        batch[m] = {"step_ms": max(r["timing"][m]["median_ms"] for r in res),
                    "noise_floor": max(r["timing"][m]["noise_floor"] for r in res),
                    "seq_dispersion": max(r["timing"][m]["seq_dispersion"] for r in res)}
    err = {m: {"rel_rms": max(r["err"][m]["rel_rms"] for r in res), "e": max(r["err"][m]["e"] for r in res)}
           for m in res[0]["err"]}
    cls = "mode_switch" if strategy == "LOWPRECISION" else "exact"
    wd.cancel()
    print(json.dumps({"batch": batch, "output": {"class": cls, "error_vs_fp32_sum": err}, "strategy": strategy,
                      "applied": res[0]["applied"]}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        main_sweep()
    elif mode == "pair":
        main_pair()
    else:
        raise SystemExit(f"unknown mode {mode}")
