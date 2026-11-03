"""4-GPU microbench + checks for idea 70: SM120 launch refinements of the lamport ONESHOT all-reduce.

Variants (env, read once per process by allReduceFusionKernels.cu):
  TLLM_AR_SM120_PDL=0|1            PDL launch attribute on SM120 (0 = branch 22 as landed)
  TLLM_AR_SM120_CTAS_PER_TOKEN=k   CTAs per token for the plain one-shot AR (1 = branch 22 as landed)

Modes:
  sweep   topology gate; rows NCCL+PHB and ONESHOT {k1 pdl0 (reference), k1 pdl1, k2/k4/k5 pdl0/1} at
          bf16 [M,5120], M=1,2,4,8,16, graph-captured: isolated (event per replay), burst (100 replays back to
          back) and chain (8 ARs captured in one graph, per call); untimed every-iteration check whose outputs are
          compared BITWISE with the k1 pdl0 outputs on the same inputs; e vs the fp32 sum; 1000 back-to-back
          captured replays at M=1 and M=8 with lamport flag/layout/clear checks on all 4 ranks.
  pair    ONESHOT graph timing M=1,2,4,8,16 under the tree's defaults (run_gpu_pair). Last line: evidence JSON.
  check   default-env candidate vs explicit k1 pdl0 bitwise every iteration + 1000-replay robustness.
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcie_custom_ar_bench as B  # noqa: E402

HIDDEN = B.HIDDEN
WORLD = B.WORLD
MS = [1, 2, 4, 8, 16]
WARMUP, ITERS, SEQS = B.WARMUP, B.ITERS, B.SEQS
CHAIN = 8
log = B.log


class ChainRunner:
    """CHAIN all-reduces of the same static input captured in one graph (PDL can overlap consecutive launches)."""

    def __init__(self, torch, mapping, strategy, m):
        from tensorrt_llm._torch.distributed import AllReduce
        self.ar = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
        self.x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
        self.ar(self.x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(CHAIN):
                self.ar(self.x)
            with torch.cuda.graph(g, stream=s):
                self.outs = [self.ar(self.x) for _ in range(CHAIN)]
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = g

    def __call__(self):
        self.graph.replay()
        return self.outs[-1]


def _time_chain(torch, run, m, rank, dev):
    meds = []
    for s in range(SEQS):
        run.x.copy_(B._gen(torch, m, 0, rank, dev))
        torch.cuda.synchronize()
        B._barrier()
        for _ in range(WARMUP):
            run()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
        for i in range(ITERS):
            ev[i][0].record()
            run()
            ev[i][1].record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) for a, b in ev)
        meds.append(ts[len(ts) // 2] / CHAIN)
    return {"per_call_ms": sorted(meds)[SEQS // 2], "seq": meds}


def _emul(torch, m, seed, dev):
    """Bitwise emulation of the lamport sum: fp32 adds in rank order 0..3, one rounding to bf16."""
    ins = [B._gen(torch, m, seed, r, dev).float() for r in range(WORLD)]
    return (((ins[0] + ins[1]) + ins[2]) + ins[3]).to(torch.bfloat16)


def _check_save(torch, run, m, rank, dev, path):
    worst, bad, emul_mis, negz = 0.0, 0, 0, 0
    outs = torch.empty((WARMUP + ITERS, m, HIDDEN), dtype=torch.bfloat16, device=dev)
    per_iter_e = []
    for it in range(WARMUP + ITERS):
        run.x.copy_(B._gen(torch, m, it + 1, rank, dev))
        out = run()
        outs[it].copy_(out)
        ref, absum = B._ref(torch, m, it + 1, dev)
        e, fin, nz = B._err(torch, out, ref, absum)
        per_iter_e.append(e)
        worst = max(worst, e)
        bad += int(e > 2.0 ** -8 or not fin)
        negz += nz
        emul_mis += int(not torch.equal(out.view(torch.int16), _emul(torch, m, it + 1, dev).view(torch.int16)))
    torch.cuda.synchronize()
    torch.save(outs.cpu(), path)
    return {"iters": WARMUP + ITERS, "bad_iters_e_gt_2^-8": bad, "max_e": worst, "neg_zero_outputs": negz,
            "iters_not_bitwise_vs_fp32_rank_order_emulation": emul_mis,
            "e_first_last": [per_iter_e[0], per_iter_e[-1]]}


def rank_variant(env, do_time, do_check, do_robust):
    torch, rank, mapping, _ = B._setup(env)
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    dev = torch.device("cuda", rank)
    strat = getattr(AllReduceStrategy, env["_strategy"])
    name = env["_name"]
    res = {"rank": rank, "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "TLLM_", "TRTLLM_"))},
           "timing": {}, "check": {}, "robust": {}}
    if rank == 0:
        res["peer"] = [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else -1 for j in range(WORLD)]
                       for i in range(WORLD)]
    for m in MS:
        if do_time:
            run = B.Runner(torch, mapping, strat, m, graph=True)
            t = B._time(torch, run, m, rank, dev)
            del run
            ch = ChainRunner(torch, mapping, strat, m)
            t["chain"] = _time_chain(torch, ch, m, rank, dev)
            del ch
            res["timing"][str(m)] = t
        if do_check:
            run = B.Runner(torch, mapping, strat, m, graph=True)
            res["check"][str(m)] = _check_save(torch, run, m, rank, dev,
                                               os.path.join(env["_outdir"], f"{name}_m{m}_r{rank}.pt"))
            del run
        torch.cuda.synchronize()
        if rank == 0:
            t = res["timing"].get(str(m), {})
            c = res["check"].get(str(m), {})
            log(f"[{name}] m={m} iso={t.get('median_ms', 0)*1e3:.2f}us burst={t.get('burst_mean_ms', 0)*1e3:.2f}us "
                f"chain={t.get('chain', {}).get('per_call_ms', 0)*1e3:.2f}us nf={t.get('noise_floor', 0):.3f} "
                f"check={c}")
    if do_robust:
        for m in (1, 8):
            r = B._robust(torch, mapping, strat, m, rank, dev)
            res["robust"][f"ONESHOT_{m}"] = r
            if rank == 0:
                log(f"[{name}][robust] m={m} bad={r['bad_calls']}/{r['calls']} max_e={r['max_e']:.3g} "
                    f"flags {r['flags_start']}->{r['flags_end']} {r['seconds']}s")
    B._barrier()
    return res


def run_ranks(fn, *args):
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(B)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    t0 = time.time()
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        out = list(ex.map(fn, *[[a] * WORLD for a in args]))
    log(f"[driver] {args[0].get('_name')} done in {time.time()-t0:.0f}s")
    return sorted(out, key=lambda r: r["rank"])


def _bitwise(outdir, name, refname):
    import torch
    res = {}
    for m in MS:
        bad_iters, total = 0, 0
        for r in range(WORLD):
            a = torch.load(os.path.join(outdir, f"{name}_m{m}_r{r}.pt"))
            b = torch.load(os.path.join(outdir, f"{refname}_m{m}_r{r}.pt"))
            eq = (a.view(torch.int16) == b.view(torch.int16)).flatten(1).all(1)
            bad_iters += int((~eq).sum().item())
            total += eq.numel()
        res[str(m)] = {"iters_x_ranks": total, "not_bitwise": bad_iters}
    return res


def _gate():
    import torch
    peer = [[int(torch.cuda.can_device_access_peer(i, j)) for j in range(WORLD) if j != i] for i in range(WORLD)]
    gate = all(all(row) for row in peer)
    log(f"[gate] can_device_access_peer on 12 ordered pairs: {peer} -> {gate}")
    return gate, peer


def _summ_timing(res):
    out = {}
    for m in res[0]["timing"]:
        per = [r["timing"][m] for r in res]
        out[m] = {"iso_us": round(max(p["median_ms"] for p in per) * 1e3, 2),
                  "burst_us": round(max(p["burst_mean_ms"] for p in per) * 1e3, 2),
                  "chain_us": round(max(p["chain"]["per_call_ms"] for p in per) * 1e3, 2),
                  "noise_floor": round(max(p["noise_floor"] for p in per), 4)}
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    log(B.sh(["nvidia-smi", "--query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,"
              "clocks_throttle_reasons.active", "--format=csv"]))
    outdir = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), f"sm120ar_{mode}_{os.getpid()}")
    os.makedirs(outdir, exist_ok=True)
    gate, peer = _gate()
    if not gate:
        print(json.dumps({"passed": False, "gate": "peer access missing", "peer": peer}))
        return
    base = {"NCCL_P2P_LEVEL": "PHB", "_outdir": outdir}
    if mode == "pair":
        res = run_ranks(rank_variant, dict(base, _name="pair", _strategy="ONESHOT"), True, False, False)
        t = _summ_timing(res)
        log(json.dumps(t))
        print(json.dumps({"batch": {m: {"step_ms": v["iso_us"] / 1e3, "noise_floor": v["noise_floor"],
                                        "burst_us": v["burst_us"], "chain_us": v["chain_us"]}
                                    for m, v in t.items()},
                          "output": {"class": "exact"}}))
        return
    if mode == "sweep":
        variants = [("nccl_phb", "NCCL", None, None), ("k1_pdl0", "ONESHOT", 1, 0), ("k1_pdl1", "ONESHOT", 1, 1),
                    ("k2_pdl0", "ONESHOT", 2, 0), ("k2_pdl1", "ONESHOT", 2, 1), ("k4_pdl1", "ONESHOT", 4, 1),
                    ("k5_pdl0", "ONESHOT", 5, 0), ("k5_pdl1", "ONESHOT", 5, 1)]
    else:  # check: candidate defaults vs explicit branch-22 launch
        variants = [("k1_pdl0", "ONESHOT", 1, 0), ("default", "ONESHOT", None, None)]
    table, checks, robust, bitwise = {}, {}, {}, {}
    allok = True
    for name, strat, k, pdl in variants:
        env = dict(base, _name=name, _strategy=strat)
        if k is not None:
            env["TLLM_AR_SM120_CTAS_PER_TOKEN"] = str(k)
            env["TLLM_AR_SM120_PDL"] = str(pdl)
        is_custom = strat == "ONESHOT"
        res = run_ranks(rank_variant, env, mode == "sweep", True, is_custom)
        if mode == "sweep":
            table[name] = _summ_timing(res)
        checks[name] = {m: {"bad_iters_e": sum(r["check"][m]["bad_iters_e_gt_2^-8"] for r in res),
                            "max_e": max(r["check"][m]["max_e"] for r in res),
                            "neg_zero_outputs": sum(r["check"][m]["neg_zero_outputs"] for r in res),
                            "not_bitwise_vs_emulation": sum(r["check"][m][
                                "iters_not_bitwise_vs_fp32_rank_order_emulation"] for r in res)}
                        for m in res[0]["check"]}
        for m, c in checks[name].items():
            allok = allok and c["bad_iters_e"] == 0
        if is_custom:
            for key in res[0]["robust"]:
                ok, flags = B._robust_ok(res, key)
                robust[f"{name}:{key}"] = {"passed": ok, "flags_end_per_rank": flags,
                                           "layout_end_per_rank": [r["robust"][key]["layout_end"] for r in res],
                                           "bad_calls": sum(r["robust"][key]["bad_calls"] for r in res),
                                           "neg_zero": sum(r["robust"][key]["neg_zero_outputs"] for r in res),
                                           "max_e": max(r["robust"][key]["max_e"] for r in res)}
                allok = allok and ok
            if name != "k1_pdl0":
                bitwise[name] = _bitwise(outdir, name, "k1_pdl0")
                for m, b in bitwise[name].items():
                    allok = allok and b["not_bitwise"] == 0
                log(f"[bitwise] {name} vs k1_pdl0: {bitwise[name]}")
        log(f"[{name}] env={res[0]['env']}")
    if table:
        log("\nrow        " + " ".join(f"{'M='+str(m)+' iso/burst/chain':>26}" for m in MS))
        for name, t in table.items():
            log(f"{name:10} " + " ".join(f"{t[str(m)]['iso_us']:8.2f}/{t[str(m)]['burst_us']:7.2f}/"
                                         f"{t[str(m)]['chain_us']:7.2f}  " for m in MS))
    print(json.dumps({"passed": allok, "peer_12_pairs": peer,
                      "compared": "every ONESHOT variant's outputs bitwise vs the k1_pdl0 (branch 22) lamport output "
                                  "on the same inputs, every iteration, M=1..16, 4 ranks; e vs fp32 sum <= 2^-8; "
                                  "1000 replays at M=1/8 with lamport flag/clear checks",
                      "bitwise_vs_k1_pdl0": bitwise, "table_us": table, "checks": checks, "robust": robust}))


if __name__ == "__main__":
    main()
