"""Idea 96 / branch 199 kernel_time pair: one L0-19 prefill boundary (AR + mHC.fused_hc_lagged) as each tree runs it.

4 ranks (mpi4py.futures), real tensorrt_llm AllReduce with the tree's served strategy (engine.json; LOWPRECISION on
main a9ae1a75+) and a real mHC at V4.1 dims, mHC autotuned like warm-up. Per T in {2048, 4096, 6144, 8192}:
  base tree (no mhc/ar_pipeline.py) or env 0 / T below MIN_ROWS: y = AR(x); mHC.fused_hc_lagged(y, ...)  (serial)
  branch tree at T >= TRTLLM_DSV41_PREFILL_AR_PIPELINE_MIN_ROWS: ArFusedHcPipeline.run with k = PIPELINE_K   (piped)
the exact decision the model's DeepseekV41DecoderLayer._ar_pipe_bounds makes. 20 warmup + 100 timed per sequence,
median of 3, inputs re-created from the seed per sequence, max over ranks. Also prints the per-rank host us per
boundary call. An untimed check runs the same sequence counts and compares every iteration's layer_input /
residual with the serial chain on the same inputs (rel_rms) and with the fp32-sum reference chain.
Last stdout line: {"batch": {"<T>": {"kernel_ms", "noise_floor", ...}}, "output": {"class": "reordered", ...}}.
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
TS = [2048, 4096, 6144, 8192]


def log(*a):
    print(*a, flush=True)


def _served_strategy(tree):
    try:
        with open(os.path.join(tree, "experiments", "ascent-config", "engine.json")) as f:
            return str(json.load(f).get("allreduce_strategy", "AUTO")).upper()
    except Exception:  # noqa: BLE001
        return "AUTO"


def rank_pair(env):
    wd = bench._watchdog("pair", 2400)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
    try:
        from tensorrt_llm._torch.modules.mhc import ar_pipeline as arp
    except ImportError:
        arp = None
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
    pipe = arp.ArFusedHcPipeline() if arp is not None else None
    sts = {T: stage0.St(torch, T, n, rank, dev) for T in TS}
    for T in TS:
        sts[T].refill(0)
    with autotune():
        st = sts[max(TS)]
        mhc.fused_hc_lagged(st.x, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
    torch.cuda.synchronize()

    def plan(T):  # the model's decision (DeepseekV41DecoderLayer._ar_pipe_bounds, shape part)
        if arp is None or arp.PIPELINE_K <= 1 or T < max(arp.PIPELINE_MIN_ROWS, 2 * arp.MIN_CHUNK_ROWS):
            return None
        return arp.chunk_bounds(T, arp.PIPELINE_K)

    def serial(st):
        return mhc.fused_hc_lagged(ar(st.x), st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)

    res = {"rank": rank, "timing": {}, "err": {}, "plan": {},
           "k": None if arp is None else arp.PIPELINE_K, "min_rows": None if arp is None else arp.PIPELINE_MIN_ROWS}
    for T in TS:
        st = sts[T]
        bounds = plan(T)
        res["plan"][str(T)] = bounds

        def boundary():
            if bounds is None:
                return serial(st)
            return pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
        res["timing"][str(T)] = stage0._timeit(torch, boundary, st.refill)
        # untimed: same counts and reset points; every iteration vs serial and vs the fp32-sum reference chain
        worst = {"li_rel_vs_serial": 0.0, "res_rel_vs_serial": 0.0, "li_rel_vs_ref": 0.0,
                 "li_rel_vs_ref_serial": 0.0, "nonfinite": 0, "iters": 0}

        def rr(a, b):
            d = a.float() - b.float()
            return (d.square().sum().sqrt() / b.float().square().sum().sqrt().clamp_min(1e-30)).item()
        for s in range(stage0.SEQS):
            st.refill(s)
            for it in range(stage0.WARMUP + stage0.ITERS):
                if it % 20 == 0:  # changing inputs inside the sequence (every 20th call re-seeded)
                    st.refill(1000 * s + it)
                o = boundary()
                o = [t.clone() for t in o]
                sref = serial(st)
                worst["li_rel_vs_serial"] = max(worst["li_rel_vs_serial"], rr(o[4], sref[4]))
                worst["res_rel_vs_serial"] = max(worst["res_rel_vs_serial"], rr(o[0], sref[0]))
                worst["nonfinite"] += int(not all(torch.isfinite(t.float()).all().item() for t in o))
                if it % 20 == 0:
                    sd = 1000 * s + it
                    ins = [bench._gen(torch, T, sd, r, dev).float() for r in range(WORLD)]
                    ref = (ins[0] + ins[1] + ins[2] + ins[3]).to(torch.bfloat16)
                    del ins
                    ro = mhc.fused_hc_lagged(ref, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
                    worst["li_rel_vs_ref"] = max(worst["li_rel_vs_ref"], rr(o[4], ro[4]))
                    worst["li_rel_vs_ref_serial"] = max(worst["li_rel_vs_ref_serial"], rr(sref[4], ro[4]))
                    del ro, ref
                worst["iters"] += 1
            torch.cuda.synchronize()
        res["err"][str(T)] = worst
        if rank == 0:
            t = res["timing"][str(T)]
            log(f"[pair] T={T} plan={bounds} us={t['us']:.1f} nf={t['noise_floor']:.3f} host={t['host_us']:.0f} "
                f"err={worst}")
        bench._barrier()
    wd.cancel()
    return res


def main():
    stage0.CKPT = sys.argv[1]
    os.environ["STAGE0_CKPT"] = sys.argv[1]
    wd = bench._watchdog("driver", 2700)
    tree = os.getcwd()
    strategy = _served_strategy(tree)
    head = bench.sh(["git", "-C", tree, "rev-parse", "HEAD"]).strip()
    log(bench._gpu_state())
    log(f"[pair] host={bench.sh(['hostname']).strip()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"tree={tree} head={head} strategy={strategy} "
        f"has_ar_pipeline={os.path.exists(os.path.join(tree, 'tensorrt_llm/_torch/modules/mhc/ar_pipeline.py'))} "
        f"env={ {k: v for k, v in os.environ.items() if k.startswith('TRTLLM_DSV41_PREFILL')} }")
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(bench)
    cloudpickle.register_pickle_by_value(stage0)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        rs = sorted(ex.map(rank_pair, [{"_name": "pair", "_apply_nccl_tuning": "1", "_strategy": strategy,
                                        "STAGE0_CKPT": sys.argv[1]}] * WORLD), key=lambda r: r["rank"])
    log(bench._gpu_state())
    batch, err = {}, {}
    for T in map(str, TS):
        batch[T] = {"kernel_ms": max(r["timing"][T]["us"] for r in rs) / 1e3,
                    "noise_floor": max(r["timing"][T]["noise_floor"] for r in rs),
                    "per_rank_us": [round(r["timing"][T]["us"], 1) for r in rs],
                    "host_us_per_rank": [round(r["timing"][T]["host_us"], 1) for r in rs],
                    "seq_us_rank_max": [round(max(r["timing"][T]["seq_us"][s] for r in rs), 1) for s in range(3)],
                    "plan": rs[0]["plan"][T]}
        err[T] = {k: max(r["err"][T][k] for r in rs) for k in rs[0]["err"][T] if k not in ("nonfinite", "iters")}
        err[T]["nonfinite"] = sum(r["err"][T]["nonfinite"] for r in rs)
        err[T]["iters_per_rank"] = rs[0]["err"][T]["iters"]
    piped = any(b["plan"] for b in batch.values())
    wd.cancel()
    print(json.dumps({"batch": batch, "output": {"class": "reordered" if piped else "exact", "error": err},
                      "strategy": strategy, "head": head, "k": rs[0]["k"], "min_rows": rs[0]["min_rows"]}))


if __name__ == "__main__":
    main()
