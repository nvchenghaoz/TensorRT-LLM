"""Idea 374 / branch 226 STAGE 0: design B (stream-role swap) vs design D (comm role) per prefill boundary, 4 GPUs.

Per rank (mpi4py.futures), real tensorrt_llm AllReduce(LOWPRECISION) and a real mHC at V4.1 dims (autotuned like
warm-up). Per T in {4096, 6144, 8192} with the model's k(T) chunk bounds:
  serial      y = AR(x); mHC.fused_hc_lagged(y, ...)                         (main's unpipelined path)
  comm        ArFusedHcPipeline(role="comm")    chunk ARs on a priority -1 comm stream, mHC on main (design D)
  swap_p0     ArFusedHcPipeline(role="swap", priority=0)   chunk ARs on main, mHC on a priority 0 side stream
  swap_pm1    ArFusedHcPipeline(role="swap", priority=-1)  same, side stream at priority -1
Timing: stage0._timeit (20 warmup + 100 timed per sequence, median of 3), max over ranks; host us per call.
Memory (per role, untimed): after empty_cache, 3 calls, then torch.cuda.memory_snapshot(): allocator segments owned by
the pipeline's side stream (must be 0 for swap), and peak allocated above the inputs vs serial.
Numerics (untimed): swap vs comm on the same inputs bitwise (same kernels, same bounds, same tactic), swap vs serial
rel_rms of layer_input, for 30 changing inputs per T.
Last stdout line: JSON summary.
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
TS = [4096, 6144, 8192]
VARIANTS = ["serial", "comm", "swap_p0", "swap_pm1"]


def log(*a):
    print(*a, flush=True)


def rank_main(env):
    wd = bench._watchdog("swap_stage0", 2400)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.modules.mhc import ar_pipeline as arp
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
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
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.LOWPRECISION, dtype=torch.bfloat16)
    pipes = {"comm": arp.ArFusedHcPipeline(role="comm"),
             "swap_p0": arp.ArFusedHcPipeline(role="swap", priority=0),
             "swap_pm1": arp.ArFusedHcPipeline(role="swap", priority=-1)}
    sts = {T: stage0.St(torch, T, n, rank, dev) for T in TS}
    for T in TS:
        sts[T].refill(0)
    with autotune():
        st = sts[max(TS)]
        mhc.fused_hc_lagged(st.x, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
    torch.cuda.synchronize()
    res = {"rank": rank, "timing": {}, "mem": {}, "num": {}, "bounds": {},
           "default_role": arp.PIPELINE_ROLE, "default_max_rows": arp.PIPELINE_MAX_ROWS,
           "plan": {str(T): arp.plan(T) for T in [2048] + TS}}

    def rr(a, b):
        d = a.float() - b.float()
        return (d.square().sum().sqrt() / b.float().square().sum().sqrt().clamp_min(1e-30)).item()

    for T in TS:
        st = sts[T]
        bounds = arp.chunk_bounds(T, arp.k_for(T))
        res["bounds"][str(T)] = bounds

        def make(v):
            if v == "serial":
                return lambda: mhc.fused_hc_lagged(ar(st.x), st.res, st.post, st.comb, st.pre, norm_weight=nw,
                                                   norm_eps=neps)
            p = pipes[v]
            return lambda: p.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)

        res["timing"][str(T)] = {}
        for v in VARIANTS:
            res["timing"][str(T)][v] = stage0._timeit(torch, make(v), st.refill)
            bench._barrier()
        # memory census per variant
        res["mem"][str(T)] = {}
        for v in VARIANTS:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            fn = make(v)
            for _ in range(3):
                o = fn()
                del o
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base_alloc
            snap = torch.cuda.memory_snapshot()
            side = pipes[v].stream.cuda_stream if v != "serial" else None
            main_s = torch.cuda.current_stream().cuda_stream
            segs_side = [s for s in snap if side is not None and s.get("stream") == side]
            segs_main = [s for s in snap if s.get("stream") == main_s]
            res["mem"][str(T)][v] = {"peak_alloc_above_inputs_MiB": round(peak / 2**20, 2),
                                     "side_stream_segments": len(segs_side),
                                     "side_stream_segment_MiB": round(sum(s["total_size"] for s in segs_side) / 2**20,
                                                                      2),
                                     "main_stream_segment_MiB": round(sum(s["total_size"] for s in segs_main) / 2**20,
                                                                      2),
                                     "reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 1)}
            bench._barrier()
        # numerics: swap vs comm bitwise, swap vs serial rel_rms, 30 changing inputs
        num = {"calls": 0, "swap_vs_comm_bitwise_calls": 0, "swap_pm1_vs_comm_bitwise_calls": 0,
               "max_li_rel_swap_vs_serial": 0.0, "max_res_rel_swap_vs_serial": 0.0, "nonfinite": 0}
        for i in range(30):
            st.refill(7000 + i)
            oc = [t.clone() for t in pipes["comm"].run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps,
                                                       bounds)]
            os_ = [t.clone() for t in pipes["swap_p0"].run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw,
                                                           neps, bounds)]
            om = [t.clone() for t in pipes["swap_pm1"].run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw,
                                                           neps, bounds)]
            se = mhc.fused_hc_lagged(ar(st.x), st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
            num["swap_vs_comm_bitwise_calls"] += int(all(torch.equal(a, b) for a, b in zip(os_, oc)))
            num["swap_pm1_vs_comm_bitwise_calls"] += int(all(torch.equal(a, b) for a, b in zip(om, oc)))
            num["max_li_rel_swap_vs_serial"] = max(num["max_li_rel_swap_vs_serial"], rr(os_[4], se[4]))
            num["max_res_rel_swap_vs_serial"] = max(num["max_res_rel_swap_vs_serial"], rr(os_[0], se[0]))
            num["nonfinite"] += int(not all(torch.isfinite(t.float()).all().item() for t in os_))
            num["calls"] += 1
        torch.cuda.synchronize()
        res["num"][str(T)] = num
        if rank == 0:
            tt = res["timing"][str(T)]
            log(f"[s0] T={T} bounds={bounds} " + " ".join(
                f"{v}={tt[v]['us']:.1f}us(nf{tt[v]['noise_floor']:.3f},h{tt[v]['host_us']:.0f})" for v in VARIANTS))
            log(f"[s0] T={T} mem={json.dumps(res['mem'][str(T)])}")
            log(f"[s0] T={T} num={json.dumps(num)}")
        bench._barrier()
    wd.cancel()
    return res


def main():
    stage0.CKPT = sys.argv[1]
    os.environ["STAGE0_CKPT"] = sys.argv[1]
    wd = bench._watchdog("driver", 2700)
    log(bench._gpu_state())
    head = bench.sh(["git", "-C", HERE, "rev-parse", "HEAD"]).strip()
    log(f"[s0] host={bench.sh(['hostname']).strip()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"head={head} load={os.getloadavg()}")
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(bench)
    cloudpickle.register_pickle_by_value(stage0)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        rs = sorted(ex.map(rank_main, [{"_name": "swap_stage0", "_apply_nccl_tuning": "1",
                                        "STAGE0_CKPT": sys.argv[1]}] * WORLD), key=lambda r: r["rank"])
    log(bench._gpu_state())
    out = {"head": head, "default_role": rs[0]["default_role"], "default_max_rows": rs[0]["default_max_rows"],
           "plan": rs[0]["plan"], "rows": {}}
    for T in map(str, TS):
        row = {"bounds": rs[0]["bounds"][T]}
        for v in VARIANTS:
            row[v] = {"us_max_rank": round(max(r["timing"][T][v]["us"] for r in rs), 1),
                      "noise": round(max(r["timing"][T][v]["noise_floor"] for r in rs), 4),
                      "host_us_per_rank": [round(r["timing"][T][v]["host_us"], 1) for r in rs],
                      "mem_rank0": rs[0]["mem"][T][v],
                      "side_segments_all_ranks": [r["mem"][T][v]["side_stream_segments"] for r in rs],
                      "peak_MiB_all_ranks": [r["mem"][T][v]["peak_alloc_above_inputs_MiB"] for r in rs]}
        row["num"] = {k: (sum if k in ("calls", "swap_vs_comm_bitwise_calls", "swap_pm1_vs_comm_bitwise_calls",
                                       "nonfinite") else max)(r["num"][T][k] for r in rs) for k in rs[0]["num"][T]}
        out["rows"][T] = row
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "swap_stage0.json"), "w") as f:
        json.dump({"summary": out, "raw": rs}, f, indent=1, default=str)
    wd.cancel()
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
