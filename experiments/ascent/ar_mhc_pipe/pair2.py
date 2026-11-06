"""Idea 374 / branch 226 final kernel pair (top-lead ruling 12550): one L0-19 prefill boundary (AR + fused mHC) as
each tree runs it, with the correctness receipts folded in. 4 ranks (mpi4py.futures), real LOWPRECISION AllReduce and a
real mHC at V4.1 dims, autotuned like warm-up.

Shapes: T=6145 PRIMARY (the served c4 follower step: 3 x 2048 ctx + 1 gen), 8192/6144/4096 secondary, 2048 a
no-change check. Per T the tree's own decision (ar_pipeline.plan): base main 06efaf15 = D for 4096..6144, serial
elsewhere; head = D for 4096..6144, B (role swap) for 6145..8192.
Timing: stage0._timeit = 20 warmup + 100 timed per sequence, median of 3 sequences, inputs re-created from the seed
before each sequence, max over ranks; timed before any CUPTI; SM clocks by UUID logged before/after.
Untimed repeated-execution sequence (same warmup count, iteration count and reset points as the timed run, inputs
re-seeded every 20th call): every iteration's layer_input / residual vs the serial chain (rel_rms), vs the fp32-sum
reference chain, finiteness; and wherever the tree pipelines T, BITWISE vs landing 77's run()
(ref_ar_pipeline_main.py) forced to the same chunk bounds (for T > 6144 that is D forced past its cap, MAX_ROWS=6400
at 6145 and 8192 at 8192).
ABAB at T=6145: D (77's run() forced, same bounds) vs this tree's pipelined run, 4+4 alternating blocks of 20 warmup +
100 timed, block medians (max over ranks) and spread.
Last stdout line: {"batch": {"<T>": {"kernel_ms", "noise_floor", ...}}, "output": {"class", "error"}, ...}.
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lp_ar"))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import stage0  # noqa: E402

HIDDEN = 5120
WORLD = 4
TS = [6145, 8192, 6144, 4096, 2048]
ABAB_T = 6145
ABAB_BLOCKS = 8


def log(*a):
    print(*a, flush=True)


def rank_pair(env):
    wd = bench._watchdog("pair2", 3000)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.modules.mhc import ar_pipeline as arp
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
    if env["_here"] not in sys.path:
        sys.path.insert(0, env["_here"])
    import ref_ar_pipeline_main as ref
    dev = torch.device("cuda", rank)
    cfg = stage0._cfg()
    n = int(cfg["hc_mult"])
    mhc = mHC(n, HIDDEN, int(cfg["hc_sinkhorn_iters"]), dtype=torch.float32, eps=cfg["hc_eps"],
              norm_eps=cfg["rms_norm_eps"], sinkhorn_eps=cfg["hc_eps"], post_mult_value=2.0).to(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    with torch.no_grad():
        mhc.fn.copy_(torch.randn(mhc.fn.shape, generator=g, device=dev) * (mhc.hc_dim ** -0.5))
        mhc.base.copy_(torch.randn(mhc.base.shape, generator=g, device=dev) * 0.5)
        mhc.scale.copy_(torch.tensor([0.5, 0.5, 0.5], device=dev))
    nw = (1.0 + 0.1 * torch.randn((HIDDEN,), generator=g, device=dev)).to(torch.bfloat16)
    neps = float(cfg["rms_norm_eps"])
    AutoTuner.get().setup_distributed_state(mapping)
    ar = AllReduce(mapping=mapping, strategy=getattr(AllReduceStrategy, env["_strategy"]), dtype=torch.bfloat16)
    pipe = arp.ArFusedHcPipeline()
    rpipe = ref.ArFusedHcPipeline()
    sts = {T: stage0.St(torch, T, n, rank, dev) for T in TS}
    for T in TS:
        sts[T].refill(0)
    with autotune():
        st = sts[max(TS)]
        mhc.fused_hc_lagged(st.x, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
    torch.cuda.synchronize()

    def serial(st):
        return mhc.fused_hc_lagged(ar(st.x), st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)

    res = {"rank": rank, "timing": {}, "err": {}, "plan": {}, "abab": None,
           "swap_above_rows": getattr(arp, "SWAP_ABOVE_ROWS", None), "max_rows": arp.PIPELINE_MAX_ROWS,
           "min_rows": arp.PIPELINE_MIN_ROWS}
    for T in TS:
        st = sts[T]
        bounds = arp.plan(T)
        rbounds = ref.chunk_bounds(T, ref.k_for(T))
        res["plan"][str(T)] = {"bounds": bounds, "ref_bounds_forced": rbounds,
                               "role": None if bounds is None else (
                                   "swap" if T > (res["swap_above_rows"] or 10 ** 9) else "landing77")}

        def boundary():
            if bounds is None:
                return serial(st)
            return pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
        res["timing"][str(T)] = stage0._timeit(torch, boundary, st.refill)
        bench._barrier()
        worst = {"li_rel_vs_serial": 0.0, "res_rel_vs_serial": 0.0, "li_rel_vs_ref": 0.0,
                 "li_rel_vs_ref_serial": 0.0, "nonfinite": 0, "iters": 0,
                 "bitwise_vs_77_forced_calls": 0, "bitwise_checked_calls": 0}

        def rr(a, b):
            d = a.float() - b.float()
            return (d.square().sum().sqrt() / b.float().square().sum().sqrt().clamp_min(1e-30)).item()
        for s in range(stage0.SEQS):
            st.refill(s)
            for it in range(stage0.WARMUP + stage0.ITERS):
                if it % 20 == 0:
                    st.refill(1000 * s + it)
                o = [t.clone() for t in boundary()]
                sref = serial(st)
                worst["li_rel_vs_serial"] = max(worst["li_rel_vs_serial"], rr(o[4], sref[4]))
                worst["res_rel_vs_serial"] = max(worst["res_rel_vs_serial"], rr(o[0], sref[0]))
                worst["nonfinite"] += int(not all(torch.isfinite(t.float()).all().item() for t in o))
                if bounds is not None:
                    r77 = rpipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
                    worst["bitwise_vs_77_forced_calls"] += int(all(torch.equal(a, b) for a, b in zip(o, r77)))
                    worst["bitwise_checked_calls"] += 1
                    del r77
                if it % 20 == 0:
                    sd = 1000 * s + it
                    ins = [bench._gen(torch, T, sd, r, dev).float() for r in range(WORLD)]
                    refsum = (ins[0] + ins[1] + ins[2] + ins[3]).to(torch.bfloat16)
                    del ins
                    ro = mhc.fused_hc_lagged(refsum, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
                    worst["li_rel_vs_ref"] = max(worst["li_rel_vs_ref"], rr(o[4], ro[4]))
                    worst["li_rel_vs_ref_serial"] = max(worst["li_rel_vs_ref_serial"], rr(sref[4], ro[4]))
                    del ro, refsum
                worst["iters"] += 1
                del o, sref
            torch.cuda.synchronize()
        res["err"][str(T)] = worst
        if rank == 0:
            t = res["timing"][str(T)]
            log(f"[pair2] T={T} plan={res['plan'][str(T)]} us={t['us']:.1f} nf={t['noise_floor']:.4f} "
                f"host={t['host_us']:.0f} err={worst}")
        bench._barrier()

    # ABAB at the served c4 follower shape: D (77's run() forced) vs this tree's pipelined run, same bounds
    st = sts[ABAB_T]
    rb = ref.chunk_bounds(ABAB_T, ref.k_for(ABAB_T))
    tree_bounds = arp.plan(ABAB_T) or rb
    fns = {"D_77_forced": lambda: rpipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, rb),
           "tree": lambda: pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, tree_bounds)}
    blocks = []
    for blk in range(ABAB_BLOCKS):
        name = "D_77_forced" if blk % 2 == 0 else "tree"
        fn = fns[name]
        st.refill(77 + blk)
        torch.cuda.synchronize()
        bench._barrier()
        for _ in range(stage0.WARMUP):
            fn()
        torch.cuda.synchronize()
        bench._barrier()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(stage0.ITERS)]
        for i in range(stage0.ITERS):
            ev[i][0].record()
            fn()
            ev[i][1].record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) for a, b in ev)
        blocks.append({"variant": name, "median_us": ts[len(ts) // 2] * 1e3,
                       "mean_us": sum(ts) / len(ts) * 1e3})
    res["abab"] = {"T": ABAB_T, "bounds": rb, "tree_bounds": tree_bounds, "blocks": blocks}
    if rank == 0:
        log(f"[pair2] ABAB T={ABAB_T} " + json.dumps(res["abab"]))
    bench._barrier()
    wd.cancel()
    return res


def main():
    stage0.CKPT = sys.argv[1]
    os.environ["STAGE0_CKPT"] = sys.argv[1]
    wd = bench._watchdog("driver", 3300)
    tree = os.getcwd()
    try:
        with open(os.path.join(tree, "experiments", "ascent-config", "engine.json")) as f:
            strategy = str(json.load(f).get("allreduce_strategy", "AUTO")).upper()
    except Exception:  # noqa: BLE001
        strategy = "AUTO"
    head = bench.sh(["git", "-C", tree, "rev-parse", "HEAD"]).strip()
    log(bench._gpu_state())
    log(f"[pair2] host={bench.sh(['hostname']).strip()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"tree={tree} head={head} strategy={strategy} "
        f"env={ {k: v for k, v in os.environ.items() if k.startswith('TRTLLM_DSV41_PREFILL')} }")
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(bench)
    cloudpickle.register_pickle_by_value(stage0)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        rs = sorted(ex.map(rank_pair, [{"_name": "pair2", "_apply_nccl_tuning": "1", "_strategy": strategy,
                                        "_here": HERE, "STAGE0_CKPT": sys.argv[1]}] * WORLD),
                    key=lambda r: r["rank"])
    log(bench._gpu_state())
    batch, err = {}, {}
    for T in map(str, TS):
        batch[T] = {"kernel_ms": max(r["timing"][T]["us"] for r in rs) / 1e3,
                    "noise_floor": max(r["timing"][T]["noise_floor"] for r in rs),
                    "per_rank_us": [round(r["timing"][T]["us"], 1) for r in rs],
                    "host_us_per_rank": [round(r["timing"][T]["host_us"], 1) for r in rs],
                    "seq_us_rank_max": [round(max(r["timing"][T]["seq_us"][s] for r in rs), 1) for s in range(3)],
                    "plan": rs[0]["plan"][T]}
        err[T] = {k: max(r["err"][T][k] for r in rs) for k in rs[0]["err"][T]
                  if k not in ("nonfinite", "iters", "bitwise_vs_77_forced_calls", "bitwise_checked_calls")}
        for k in ("nonfinite", "bitwise_vs_77_forced_calls", "bitwise_checked_calls"):
            err[T][k] = sum(r["err"][T][k] for r in rs)
        err[T]["iters_per_rank"] = rs[0]["err"][T]["iters"]
    # ABAB: per block, max over ranks of the block median; per variant the median of its 4 blocks and the spread
    nb = len(rs[0]["abab"]["blocks"])
    blk = [{"variant": rs[0]["abab"]["blocks"][i]["variant"],
            "median_us_max_rank": round(max(r["abab"]["blocks"][i]["median_us"] for r in rs), 2),
            "mean_us_max_rank": round(max(r["abab"]["blocks"][i]["mean_us"] for r in rs), 2)} for i in range(nb)]
    abab = {"T": ABAB_T, "bounds": rs[0]["abab"]["bounds"], "tree_bounds": rs[0]["abab"]["tree_bounds"],
            "blocks": blk}
    for v in ("D_77_forced", "tree"):
        vals = sorted(b["median_us_max_rank"] for b in blk if b["variant"] == v)
        med = (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2 if len(vals) % 2 == 0 else vals[len(vals) // 2]
        abab[v] = {"block_medians": vals, "median_of_blocks": round(med, 2),
                   "spread": round((vals[-1] - vals[0]) / med, 4)}
    abab["tree_vs_D"] = round(abab["tree"]["median_of_blocks"] / abab["D_77_forced"]["median_of_blocks"] - 1, 4)
    piped = any(b["plan"]["bounds"] for b in batch.values())
    bit_ok = all(err[T]["bitwise_vs_77_forced_calls"] == err[T]["bitwise_checked_calls"] for T in err)
    fin_ok = all(err[T]["nonfinite"] == 0 for T in err)
    wd.cancel()
    print(json.dumps({"batch": batch,
                      "output": {"class": "reordered" if piped else "exact", "error": err,
                                 "bitwise_vs_77_forced_all": bit_ok, "finite_all": fin_ok},
                      "abab": abab, "strategy": strategy, "head": head,
                      "swap_above_rows": rs[0]["swap_above_rows"], "max_rows": rs[0]["max_rows"],
                      "min_rows": rs[0]["min_rows"]}))


if __name__ == "__main__":
    main()
