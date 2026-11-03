"""4-GPU benchmark and checks for TRT-LLM's custom ONESHOT/TWOSHOT all-reduce over PCIe P2P (idea 9).

Modes (each spawns 4 MPI ranks per config through mpi4py.futures, so NCCL env knobs are fresh per config):
  sweep   ONE job: topology gate, timing rows (default NCCL, NCCL PHB, ONESHOT, TWOSHOT, ONESHOT with PDL off)
          at decode M=1,2,4,8 (CUDA graph) and M=16..8192 (eager), the untimed every-iteration correctness pass
          (method 491 B) and the 1000-replay robustness pass with lamport state checks.
  pair    what the served model runs: AllReduce(strategy=AUTO) after autotuning, graph M=1,2,4,8 and eager 2048.
          In the candidate tree the PHB default of nccl_tuning.py applies (as BaseWorker does); in the base tree
          the module is absent and NCCL keeps its defaults. Last stdout line: evidence JSON.
  check   correctness of that AUTO path (every iteration vs the fp32 reference) plus 1000 back-to-back captured
          replays at M=1 and M=8 with a 4-rank reset barrier every 250. Last stdout line: {"passed": ...}.
"""
import json
import os
import pickle
import sys
import time

HIDDEN = 5120
WORLD = 4
GRAPH_M = [1, 2, 4, 8]
EAGER_M = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 20
ITERS = 100
SEQS = 3
BOUND = WORLD * 2.0 ** -8  # method 491 B: e = max|out-ref| / sum_i|x_i| <= n * 2^-8 = 2^-6
ROBUST_CALLS = 1000
ROBUST_CHUNK = 250


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------- rank-side code
def _setup(env):
    for k in ("NCCL_P2P_LEVEL",):
        if env.get("_drop_" + k):
            os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if not k.startswith("_")})
    import torch
    from tensorrt_llm._utils import mpi_rank
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    applied = None
    if env.get("_apply_nccl_tuning"):
        try:
            from tensorrt_llm._torch.distributed import nccl_tuning  # candidate tree only
            applied = nccl_tuning.apply_pcie_nccl_defaults(list(range(WORLD)))
        except ImportError:
            applied = "absent"
    from tensorrt_llm.mapping import Mapping
    mapping = Mapping(world_size=WORLD, rank=rank, gpus_per_node=WORLD, tp_size=WORLD)
    return torch, rank, mapping, applied


def _barrier():
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()


def _allgather(obj):
    from mpi4py import MPI
    return MPI.COMM_WORLD.allgather(obj)


def _gen(torch, m, seed, r, dev):
    g = torch.Generator(device=dev)
    g.manual_seed(1000003 * m + 7919 * seed + 104729 * r + 17)
    return torch.randn((m, HIDDEN), generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16)


def _ref(torch, m, seed, dev):
    ins = [_gen(torch, m, seed, r, dev).float() for r in range(WORLD)]
    ref = ins[0] + ins[1] + ins[2] + ins[3]  # fixed order fp32; exact enough for the 2^-6 bound
    absum = ins[0].abs() + ins[1].abs() + ins[2].abs() + ins[3].abs()
    return ref, absum


def _err(torch, out, ref, absum):
    o = out.float()
    finite = bool(torch.isfinite(o).all().item())
    e = ((o - ref).abs() / absum.clamp_min(1e-30)).max().item()
    negz = int((out.view(torch.int16) == -32768).sum().item())  # bf16 -0.0 bit pattern 0x8000
    return e, finite, negz


class Runner:
    """One AllReduce module plus, for decode sizes, a captured CUDA graph of it on a static input."""

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
                self.out = self.ar(self.x)  # warm on the capture stream
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
    seq_meds, flat, bursts = [], [], []
    for s in range(SEQS):
        run.x.copy_(_gen(torch, m, 0, rank, dev))  # same seed before each timed sequence
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
        # back-to-back burst: mean per call without per-call events
        _barrier()
        b0, b1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        b0.record()
        for _ in range(ITERS):
            run()
        b1.record()
        torch.cuda.synchronize()
        bursts.append(b0.elapsed_time(b1) / ITERS)
    flat.sort()
    n = len(flat)
    med = sorted(seq_meds)[SEQS // 2]
    return {"median_ms": med, "seq_medians": seq_meds, "burst_mean_ms": sorted(bursts)[SEQS // 2],
            "p10": flat[n // 10], "p90": flat[(9 * n) // 10],
            "noise_floor": (flat[(9 * n) // 10] - flat[n // 10]) / med}


def _check(torch, run, m, rank, dev):
    """Untimed sequence with the timing's counts; every call compared to the fp32 reference."""
    worst, bad, nonfinite, negz = 0.0, 0, 0, 0
    for it in range(WARMUP + ITERS):
        run.x.copy_(_gen(torch, m, it + 1, rank, dev))
        out = run()
        ref, absum = _ref(torch, m, it + 1, dev)
        e, fin, nz = _err(torch, out, ref, absum)
        worst = max(worst, e)
        bad += int(e > BOUND or not fin)
        nonfinite += int(not fin)
        negz += nz
    return {"iters": WARMUP + ITERS, "bad_iters": bad, "max_e": worst, "nonfinite_iters": nonfinite,
            "neg_zero_outputs": negz}


def _lamport_state(torch, mapping):
    from tensorrt_llm._torch.distributed import ops as dops
    ws = getattr(dops._thread_local, f"allreduce_workspaces_{mapping.pp_rank}")[mapping]
    bufs = ws[0]
    lamport, flag_buffer, layout_buffer = bufs[2], bufs[3], bufs[4]
    torch.cuda.synchronize()
    return lamport, [int(v) for v in flag_buffer.tolist()], [int(v) for v in layout_buffer.tolist()]


def _next_buffer_cleared(torch, mapping, m):
    """The lamport buffer the next call writes into must hold -0.0 sentinels over this size's data area."""
    from tensorrt_llm._ipc_utils import cudart
    lamport, flags, layout = _lamport_state(torch, mapping)
    comm_size = layout[1]
    nbytes = WORLD * m * HIDDEN * 2
    host = torch.empty(nbytes // 4, dtype=torch.int32)
    err = cudart.cudaMemcpy(host.data_ptr(), lamport.local_ptr + (flags[2] % 3) * comm_size, nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)[0]
    assert err == cudart.cudaError_t.cudaSuccess, err
    sentinel = -2147483648  # 0x80000000
    return int((host != sentinel).sum().item())


def _robust(torch, mapping, strategy, m, rank, dev):
    """1000 back-to-back captured replays with changing inputs; reset barrier across the 4 ranks every chunk."""
    run = Runner(torch, mapping, strategy, m, graph=True)
    _, f_start, _ = _lamport_state(torch, mapping)
    worst, bad, negz, chunks = 0.0, 0, 0, []
    done = 0
    t0 = time.time()
    while done < ROBUST_CALLS:
        n = min(ROBUST_CHUNK, ROBUST_CALLS - done)
        bank = [_gen(torch, m, 5000 + done + i, rank, dev) for i in range(n)]
        outs = torch.empty((n, m, HIDDEN), dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()
        _barrier()  # reset barrier: all 4 ranks start the chunk together
        _, f0, _ = _lamport_state(torch, mapping)
        for i in range(n):  # enqueued back to back, no host sync
            run.x.copy_(bank[i])
            run.graph.replay()
            outs[i].copy_(run.out)
        torch.cuda.synchronize()
        _, f1, l1 = _lamport_state(torch, mapping)
        for i in range(n):
            ref, absum = _ref(torch, m, 5000 + done + i, dev)
            e, fin, nz = _err(torch, outs[i], ref, absum)
            worst = max(worst, e)
            bad += int(e > BOUND or not fin)
            negz += nz
        uncleared = _next_buffer_cleared(torch, mapping, m) if strategy_is_oneshot(strategy, m) else 0
        chunks.append({"flags_before": f0, "flags_after": f1, "layout_after": l1, "uncleared_next": uncleared})
        done += n
    _, f_end, l_end = _lamport_state(torch, mapping)
    return {"calls": ROBUST_CALLS, "bad_calls": bad, "max_e": worst, "neg_zero_outputs": negz,
            "flags_start": f_start, "flags_end": f_end, "layout_end": l_end, "chunks": chunks,
            "seconds": round(time.time() - t0, 1)}


def strategy_is_oneshot(strategy, m):
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    return strategy == AllReduceStrategy.ONESHOT


def rank_sweep(env, rows, do_check, do_robust):
    torch, rank, mapping, applied = _setup(env)
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    dev = torch.device("cuda", rank)
    res = {"rank": rank, "nccl_env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "TLLM_PCIE",
                                                                                            "TRTLLM_ENABLE_PDL"))},
           "applied": applied, "timing": {}, "check": {}, "robust": {}}
    if rank == 0:
        res["peer"] = [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else -1 for j in range(WORLD)]
                       for i in range(WORLD)]
    for row in rows:
        strat = getattr(AllReduceStrategy, row)
        res["timing"][row], res["check"][row] = {}, {}
        for m in GRAPH_M + EAGER_M:
            if row == "TWOSHOT" and m < WORLD:
                continue
            run = Runner(torch, mapping, strat, m, graph=m in GRAPH_M)
            t = _time(torch, run, m, rank, dev)
            res["timing"][row][str(m)] = t
            if do_check:
                res["check"][row][str(m)] = _check(torch, run, m, rank, dev)
            if rank == 0:
                c = res["check"][row].get(str(m), {})
                log(f"[{env.get('_name')}] {row} m={m} med={t['median_ms']*1e3:.1f}us burst={t['burst_mean_ms']*1e3:.1f}us "
                    f"nf={t['noise_floor']:.3f} check_bad={c.get('bad_iters')} max_e={c.get('max_e')}")
            del run
            torch.cuda.synchronize()
    if do_robust:
        for row, m in (("ONESHOT", 1), ("ONESHOT", 8), ("TWOSHOT", 8)):
            r = _robust(torch, mapping, getattr(AllReduceStrategy, row), m, rank, dev)
            res["robust"][f"{row}_{m}"] = r
            if rank == 0:
                log(f"[robust] {row} m={m} bad={r['bad_calls']}/{r['calls']} max_e={r['max_e']:.3g} "
                    f"flags {r['flags_start']}->{r['flags_end']} {r['seconds']}s")
    _barrier()
    return res


def _autotune_ar(torch, mapping, m_list, dev):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    AutoTuner.get().setup_distributed_state(mapping)
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
    with autotune():
        for m in m_list:
            ar(torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize()
    # chosen tactics per bucket, from the profiling cache
    chosen = {}
    cache = AutoTuner.get().profiling_cache.cache
    for key, val in cache.items():
        if "allreduce" in str(key[0]):
            chosen[str(key)] = str(val[1])
    return chosen


def rank_pair(env, mode):
    torch, rank, mapping, applied = _setup(env)
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    dev = torch.device("cuda", rank)
    sizes = GRAPH_M + [2048]
    chosen = _autotune_ar(torch, mapping, sizes, dev)
    res = {"rank": rank, "applied": applied, "chosen": chosen, "timing": {}, "check": {}, "robust": {},
           "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    for m in sizes:
        run = Runner(torch, mapping, AllReduceStrategy.AUTO, m, graph=m in GRAPH_M)
        if mode == "pair":
            res["timing"][str(m)] = _time(torch, run, m, rank, dev)
            if rank == 0:
                t = res["timing"][str(m)]
                log(f"[pair] AUTO m={m} med={t['median_ms']*1e3:.1f}us burst={t['burst_mean_ms']*1e3:.1f}us "
                    f"nf={t['noise_floor']:.3f}")
        else:
            res["check"][str(m)] = _check(torch, run, m, rank, dev)
            if rank == 0:
                log(f"[check] AUTO m={m} {res['check'][str(m)]}")
        del run
    if mode == "check":
        for m in (1, 8):
            r = _robust(torch, mapping, AllReduceStrategy.AUTO, m, rank, dev)
            res["robust"][str(m)] = r
            if rank == 0:
                log(f"[robust] AUTO m={m} bad={r['bad_calls']}/{r['calls']} max_e={r['max_e']:.3g} "
                    f"flags {r['flags_start']}->{r['flags_end']}")
    _barrier()
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
    log(f"[driver] {fn.__name__} {args[0].get('_name')} done in {time.time()-t0:.0f}s")
    return sorted(out, key=lambda r: r["rank"])


def _combine_timing(results, row=None):
    out = {}
    keys = results[0]["timing"][row] if row else results[0]["timing"]
    for m in keys:
        per = [(r["timing"][row] if row else r["timing"])[m] for r in results]
        out[m] = {"median_ms_max_rank": max(p["median_ms"] for p in per),
                  "rank_medians_us": [round(p["median_ms"] * 1e3, 2) for p in per],
                  "burst_mean_us_max_rank": round(max(p["burst_mean_ms"] for p in per) * 1e3, 2),
                  "noise_floor": max(p["noise_floor"] for p in per)}
    return out


def _robust_ok(results, key):
    """Lamport flag must advance by calls mod 3 identically on every rank; next buffer fully cleared."""
    rows = [r["robust"][key] for r in results]
    ok = all(x["bad_calls"] == 0 and x["neg_zero_outputs"] == 0 for x in rows)
    flags = [x["flags_end"] for x in rows]
    ok = ok and all(f == flags[0] for f in flags)
    ok = ok and all(c["uncleared_next"] == 0 for x in rows for c in x["chunks"])
    return ok, flags


def sh(cmd):
    import subprocess
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                              timeout=60).stdout
    except Exception as e:  # noqa
        return f"ERR {e}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    log(sh(["nvidia-smi", "--query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,"
            "clocks_throttle_reasons.active", "--format=csv"]))
    if mode == "sweep":
        log(sh(["nvidia-smi", "topo", "-m"]))
        import torch
        peer = [[int(torch.cuda.can_device_access_peer(i, j)) for j in range(WORLD) if j != i] for i in range(WORLD)]
        gate = all(all(row) for row in peer)
        log(f"[gate] can_device_access_peer on 12 ordered pairs: {peer} -> {gate}")
        if not gate:
            print(json.dumps({"passed": False, "gate": "peer access missing", "peer": peer}))
            return
        big = str(1 << 40)  # lift the PCIe cap so the custom kernels run at every size the workspace takes
        configs = [
            ({"_name": "nccl_default", "_drop_NCCL_P2P_LEVEL": "1"}, ["NCCL"], True, False),
            ({"_name": "phb", "NCCL_P2P_LEVEL": "PHB", "TLLM_PCIE_CUSTOM_AR_MAX_BYTES": big},
             ["NCCL", "ONESHOT", "TWOSHOT"], True, True),
            ({"_name": "phb_pdl0", "NCCL_P2P_LEVEL": "PHB", "TLLM_PCIE_CUSTOM_AR_MAX_BYTES": big,
              "TRTLLM_ENABLE_PDL": "0"}, ["ONESHOT"], False, False),
        ]
        table, checks, robust, allok = {}, {}, {}, True
        for env, rows, do_check, do_robust in configs:
            res = run_ranks(rank_sweep, env, rows, do_check, do_robust)
            for row in rows:
                name = f"{env['_name']}:{row}"
                table[name] = _combine_timing(res, row)
                if do_check:
                    chk = {m: {"bad_iters": sum(r["check"][row][m]["bad_iters"] for r in res),
                               "max_e": max(r["check"][row][m]["max_e"] for r in res),
                               "neg_zero_outputs": sum(r["check"][row][m]["neg_zero_outputs"] for r in res),
                               "iters": res[0]["check"][row][m]["iters"]} for m in res[0]["check"][row]}
                    checks[name] = chk
                    allok = allok and all(c["bad_iters"] == 0 for c in chk.values())
            for key in res[0]["robust"]:
                ok, flags = _robust_ok(res, key)
                robust[f"{env['_name']}:{key}"] = {"passed": ok, "flags_end_per_rank": flags,
                                                   "max_e": max(r["robust"][key]["max_e"] for r in res),
                                                   "bad_calls": sum(r["robust"][key]["bad_calls"] for r in res),
                                                   "rank0": res[0]["robust"][key]}
                allok = allok and ok
            log(f"[{env['_name']}] env={res[0]['nccl_env']}")
        # summary table, us
        log("\nrow                     " + " ".join(f"{m:>8}" for m in [str(x) for x in GRAPH_M + EAGER_M]))
        for name, t in table.items():
            log(f"{name:24}" + " ".join(f"{t[m]['median_ms_max_rank']*1e3:8.1f}" if m in t else "       -"
                                        for m in [str(x) for x in GRAPH_M + EAGER_M]))
        print(json.dumps({"passed": allok, "bound": "e=max|out-ref|/sum|x_i| <= 2^-6 every call",
                          "table_us": {n: {m: round(v["median_ms_max_rank"] * 1e3, 2) for m, v in t.items()}
                                       for n, t in table.items()},
                          "timing": table, "checks": checks, "robust": robust}))
        return
    if mode in ("pair", "check"):
        env = {"_name": mode, "_apply_nccl_tuning": "1"}
        res = run_ranks(rank_pair, env, mode)
        log(f"[{mode}] nccl_tuning applied={res[0]['applied']} env={res[0]['nccl_env']}")
        log(f"[{mode}] autotuner tactics (rank0): {json.dumps(res[0]['chosen'])[:3000]}")
        if mode == "pair":
            t = _combine_timing(res)
            out = {"batch": {m: {"step_ms": v["median_ms_max_rank"], "noise_floor": v["noise_floor"],
                                 "burst_mean_us": v["burst_mean_us_max_rank"], "rank_medians_us": v["rank_medians_us"]}
                             for m, v in t.items()},
                   "output": {"class": "reordered"}}
            print(json.dumps(out))
        else:
            ok = True
            summ = {}
            for m in res[0]["check"]:
                bad = sum(r["check"][m]["bad_iters"] for r in res)
                summ[m] = {"bad_iters": bad, "max_e": max(r["check"][m]["max_e"] for r in res),
                           "iters": res[0]["check"][m]["iters"]}
                ok = ok and bad == 0
            rob = {}
            for key in res[0]["robust"]:
                rok, flags = _robust_ok(res, key)
                rob[key] = {"passed": rok, "flags_end_per_rank": flags,
                            "bad_calls": sum(r["robust"][key]["bad_calls"] for r in res),
                            "max_e": max(r["robust"][key]["max_e"] for r in res)}
                ok = ok and rok
            print(json.dumps({"passed": ok, "compared": "AUTO all-reduce (autotuned) vs fp32 sum of the 4 ranks' "
                              "bf16 inputs, e=max|out-ref|/sum|x_i| <= 2^-6, every call",
                              "check": summ, "robust_1000_replays": rob}))
        return


if __name__ == "__main__":
    main()
