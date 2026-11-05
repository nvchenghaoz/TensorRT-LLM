"""Kill-bounded peer-READ gate for the LOWPRECISION (FP8 E4M3 two-shot, pull) all-reduce, no model (area lead 13:0x).

4 MPI ranks (mpi4py.futures, as experiments/ascent/lp_ar/bench.py), tensorrt_llm AllReduce(strategy=LOWPRECISION) on
[2048,5120] bf16 = the served L0-19 prefill AR shape, through the same ops.py -> allreduceOp.cpp selectImplementation
path the model uses. 200 calls with fresh realistic inputs + 20 heavy-tailed (6 channels x50, as bench.py). EVERY call
is checked against the fp32 sum of the 4 rank inputs: finite, rel_rms <= 0.05 realistic / <= 0.03 heavy, output
bitwise identical on all 4 ranks (sha256 allgather), and a repeat of the same input is bitwise equal (determinism,
every 10th call). The first LP call is logged before and after. Per-rank watchdog 240 s (os._exit); run it under
`timeout -s KILL 300`. Prints `LP-CHECK half=... calls=... worst_rel_rms=... mismatched_ranks=... nondeterministic=...
rc=... wall=...`, then a JSON line with "passed". Exit 0 only when every call is right.
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lp_ar"))
import bench  # noqa: E402

M = 2048
N_REAL, N_HEAVY = 200, 20
BOUND_REAL, BOUND_HEAVY = 0.05, 0.03


def rank_check(_):
    wd = bench._watchdog("lp_check", 240)
    torch, rank, mapping, _ = bench._setup({})
    from mpi4py import MPI
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    comm = MPI.COMM_WORLD
    dev = torch.device("cuda", rank)
    run = bench.Runner(torch, mapping, AllReduceStrategy.LOWPRECISION, M, graph=False)  # one zero-input call inside
    torch.cuda.synchronize()
    bench._barrier()
    res = {"rank": rank, "calls": 0, "worst_real": 0.0, "worst_heavy": 0.0, "nonfinite": 0, "over_bound": 0,
           "rank_mismatch": 0, "nondeterministic": 0, "first_call_s": None, "kernels": None}
    t0 = time.time()
    for i in range(N_REAL + N_HEAVY):
        heavy = i >= N_REAL
        seed = 7000 + i
        run.x.copy_(bench._gen(torch, M, seed, rank, dev, heavy))
        torch.cuda.synchronize()
        if i == 0 and rank == 0:
            print(f"[lp_check] first LP call start (rank 0, [{M},{bench.HIDDEN}] bf16)", flush=True)
        tc = time.time()
        out = run().clone()
        torch.cuda.synchronize()
        if i == 0:
            res["first_call_s"] = round(time.time() - tc, 4)
            if rank == 0:
                print(f"[lp_check] first LP call done in {res['first_call_s']}s", flush=True)
        ref, absum = bench._ref(torch, M, seed, dev, heavy)
        e = bench._err(torch, out, ref, absum)
        key = "worst_heavy" if heavy else "worst_real"
        res[key] = max(res[key], e["rel_rms"])
        res["nonfinite"] += int(not e["finite"])
        res["over_bound"] += int(e["rel_rms"] > (BOUND_HEAVY if heavy else BOUND_REAL))
        h = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
        hs = comm.allgather(h)
        res["rank_mismatch"] += int(len(set(hs)) != 1)
        if i % 10 == 0:
            again = run().clone()
            torch.cuda.synchronize()
            res["nondeterministic"] += int(not torch.equal(again, out))
        res["calls"] += 1
        if rank == 0 and (i + 1) % 55 == 0:
            print(f"[lp_check] {i+1} calls, worst real {res['worst_real']:.4f} heavy {res['worst_heavy']:.4f}, "
                  f"rank_mismatch {res['rank_mismatch']}, nondet {res['nondeterministic']}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    if rank == 0:
        res["kernels"] = bench._kernel_names(torch, run)
    else:
        run()
    torch.cuda.synchronize()
    bench._barrier()
    wd.cancel()
    return res


def main():
    t0 = time.time()
    half = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[lp_check] start half(CUDA_VISIBLE_DEVICES)={half}", flush=True)
    rs = bench.run_ranks(rank_check, {"_name": "lp_check"})
    calls = min(r["calls"] for r in rs)
    worst_real = max(r["worst_real"] for r in rs)
    worst_heavy = max(r["worst_heavy"] for r in rs)
    mism = max(r["rank_mismatch"] for r in rs)
    nondet = sum(r["nondeterministic"] for r in rs)
    nonfinite = sum(r["nonfinite"] for r in rs)
    over = sum(r["over_bound"] for r in rs)
    lp_kernels = [k for k in (rs[0].get("kernels") or []) if "lowPrecision" in k]
    ok = (calls == N_REAL + N_HEAVY and nonfinite == 0 and over == 0 and mism == 0 and nondet == 0
          and len(lp_kernels) > 0)
    rc = 0 if ok else 4
    wall = round(time.time() - t0, 1)
    print(f"LP-CHECK half={half} calls={calls} worst_rel_rms={max(worst_real, worst_heavy):.4f} "
          f"(real {worst_real:.4f} heavy {worst_heavy:.4f}) mismatched_ranks={mism} nondeterministic={nondet} "
          f"nonfinite={nonfinite} over_bound={over} rc={rc} wall={wall}", flush=True)
    print(json.dumps({"passed": ok, "half": half, "calls": calls, "worst_rel_rms_real": worst_real,
                      "worst_rel_rms_heavy": worst_heavy, "mismatched_rank_calls": mism, "nondeterministic": nondet,
                      "nonfinite": nonfinite, "over_bound": over, "lp_kernels_rank0": lp_kernels,
                      "first_call_s": [r["first_call_s"] for r in rs], "wall_s": wall}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
