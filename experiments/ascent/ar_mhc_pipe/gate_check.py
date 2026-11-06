"""Idea 374 / branch 226 gate receipts (plan 15f (2)(a)-(c)), 4 ranks, real LOWPRECISION AllReduce + real mHC.

(a) T=4096/6144: the branch's ArFusedHcPipeline.run runs landing 77's roles: every chunk AR is enqueued on the
    pipeline's comm stream (priority -1), every chunk mHC on the caller's stream; outputs bitwise equal to main's run()
    (ref_ar_pipeline_main.py, a copy of main's ar_pipeline.py) on the same inputs, 12 changing inputs per T.
(b) T=8192: the branch runs the swapped roles (chunk ARs on the caller's stream, chunk mHCs on the side stream,
    priority 0); outputs bitwise equal to main's run() forced to the same 8192-row bounds (MAX_ROWS=8192).
(c) TRTLLM_DSV41_PREFILL_AR_PIPELINE=0: in a fresh process the branch module has PIPELINE_ON False and plan(T) None
    for every T in 1..16384, so the model's _ar_pipe_bounds returns None and it runs main's unpipelined code.
Plus: plan(T) equals main's plan(T) for every T <= 6144 (T=2048 unpipelined) and main's chunk bounds for
6144 < T <= 8192; None above 8192. Stream receipts are taken host-side when each chunk op is enqueued (both ops
enqueue on at::cuda::getCurrentCUDAStream). Last stdout line: {"passed": ...}.
"""
import json
import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lp_ar"))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import stage0  # noqa: E402

HIDDEN = 5120
WORLD = 4
TS = [4096, 6144, 8192]
ITERS = 12


def log(*a):
    print(*a, flush=True)


def rank_gate(env):
    wd = bench._watchdog("gate", 1800)
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
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.LOWPRECISION, dtype=torch.bfloat16)
    pipe = arp.ArFusedHcPipeline()
    rpipe = ref.ArFusedHcPipeline()
    st_big = stage0.St(torch, max(TS), n, rank, dev)
    st_big.refill(0)
    with autotune():
        mhc.fused_hc_lagged(st_big.x, st_big.res, st_big.post, st_big.comb, st_big.pre, norm_weight=nw, norm_eps=neps)
    torch.cuda.synchronize()

    # host-side stream receipts: the stream each chunk op is enqueued on
    rec = []

    def ar_rec(rows):
        s = torch.cuda.current_stream()
        rec.append(("ar", s.cuda_stream, s.priority))
        return ar(rows)
    orig_call = arp._fused_hc_call

    def mhc_rec(*a, **k):
        s = torch.cuda.current_stream()
        rec.append(("mhc", s.cuda_stream, s.priority))
        return orig_call(*a, **k)
    arp._fused_hc_call = mhc_rec

    res = {"rank": rank, "shapes": {}, "swap_above_rows": arp.SWAP_ABOVE_ROWS, "max_rows": arp.PIPELINE_MAX_ROWS}
    # plan equality (pure host)
    bad = []
    for T in range(1, 16385):
        mine = arp.plan(T)
        if T <= 6144:
            want = ref.plan(T)
        elif T <= 8192:
            want = ref.chunk_bounds(T, ref.k_for(T)) if T >= ref.PIPELINE_MIN_ROWS else None
        else:
            want = None
        if mine != want:
            bad.append(T)
    res["plan_mismatch_T"] = bad[:20]
    res["plan_mismatch_count"] = len(bad)
    res["plan_2048"] = arp.plan(2048)

    caller = torch.cuda.current_stream()
    for T in TS:
        st = stage0.St(torch, T, n, rank, dev)
        bounds = arp.plan(T)
        rbounds = ref.chunk_bounds(T, ref.k_for(T))
        out = {"bounds": bounds, "bounds_equal_main": bounds == rbounds, "iters": 0, "bitwise_calls": 0,
               "stream_ok_calls": 0, "ar_streams": set(), "mhc_streams": set()}
        expect_swap = T > arp.SWAP_ABOVE_ROWS
        for it in range(ITERS):
            st.refill(5000 + 37 * it + T)
            rec.clear()
            ob = [t.clone() for t in pipe.run(ar_rec, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps,
                                              bounds)]
            recs = list(rec)
            orr = rpipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, rbounds)
            torch.cuda.synchronize()
            out["bitwise_calls"] += int(all(torch.equal(a, b) for a, b in zip(ob, orr)))
            ars = [(s, p) for kind, s, p in recs if kind == "ar"]
            mhs = [(s, p) for kind, s, p in recs if kind == "mhc"]
            if expect_swap:
                ok = (all(s == caller.cuda_stream for s, _ in ars)
                      and all(s == pipe.side_stream.cuda_stream and p == 0 for s, p in mhs))
            else:
                ok = (all(s == pipe.stream.cuda_stream and p == -1 for s, p in ars)
                      and all(s == caller.cuda_stream for s, _ in mhs))
            ok = ok and len(ars) == len(bounds) and len(mhs) == len(bounds)
            out["stream_ok_calls"] += int(ok)
            out["ar_streams"].update(("caller" if s == caller.cuda_stream else "comm" if s == pipe.stream.cuda_stream
                                      else "side" if pipe._side_stream is not None
                                      and s == pipe._side_stream.cuda_stream else hex(s), p) for s, p in ars)
            out["mhc_streams"].update(("caller" if s == caller.cuda_stream else "comm" if s == pipe.stream.cuda_stream
                                       else "side" if pipe._side_stream is not None
                                       and s == pipe._side_stream.cuda_stream else hex(s), p) for s, p in mhs)
            out["iters"] += 1
            del ob, orr
        out["ar_streams"] = sorted(map(list, out["ar_streams"]))
        out["mhc_streams"] = sorted(map(list, out["mhc_streams"]))
        out["role"] = "swap" if expect_swap else "landing77"
        res["shapes"][str(T)] = out
        if rank == 0:
            log(f"[gate] T={T} " + json.dumps(out, default=str))
        del st
        torch.cuda.synchronize()
        bench._barrier()
    arp._fused_hc_call = orig_call
    wd.cancel()
    return res


ENV0_CODE = r"""
import json, os
from tensorrt_llm._torch.modules.mhc import ar_pipeline as arp
nonnull = [T for T in range(1, 16385) if arp.plan(T) is not None]
print(json.dumps({"env": os.environ.get("TRTLLM_DSV41_PREFILL_AR_PIPELINE"), "PIPELINE_ON": arp.PIPELINE_ON,
                  "nonnull_plans": len(nonnull), "first": nonnull[:5]}))
"""


def main():
    stage0.CKPT = sys.argv[1]
    os.environ["STAGE0_CKPT"] = sys.argv[1]
    wd = bench._watchdog("driver", 2400)
    log(bench._gpu_state())
    head = bench.sh(["git", "-C", HERE, "rev-parse", "HEAD"]).strip()
    log(f"[gate] host={bench.sh(['hostname']).strip()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"head={head}")
    # (c) env 0 in a fresh process
    env0 = dict(os.environ)
    for k in list(env0):
        if k.startswith("TRTLLM_DSV41_PREFILL_AR_PIPELINE"):
            env0.pop(k)
    env0["TRTLLM_DSV41_PREFILL_AR_PIPELINE"] = "0"
    p = subprocess.run([sys.executable, "-c", ENV0_CODE], env=env0, capture_output=True, text=True, timeout=600)
    try:
        e0 = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        e0 = {"error": (p.stdout[-500:] + p.stderr[-1500:])}
    log(f"[gate] env0 {e0}")
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(bench)
    cloudpickle.register_pickle_by_value(stage0)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        rs = sorted(ex.map(rank_gate, [{"_name": "gate", "_apply_nccl_tuning": "1", "_here": HERE,
                                        "STAGE0_CKPT": sys.argv[1]}] * WORLD), key=lambda r: r["rank"])
    summary = {}
    passed = e0.get("PIPELINE_ON") is False and e0.get("nonnull_plans") == 0
    for T in map(str, TS):
        per = [r["shapes"][T] for r in rs]
        s = {"role": per[0]["role"], "bounds": per[0]["bounds"],
             "bounds_equal_main_all_ranks": all(p["bounds_equal_main"] for p in per),
             "iters_per_rank": per[0]["iters"],
             "bitwise_vs_main_calls": sum(p["bitwise_calls"] for p in per),
             "stream_ok_calls": sum(p["stream_ok_calls"] for p in per), "calls": sum(p["iters"] for p in per),
             "ar_streams_rank0": per[0]["ar_streams"], "mhc_streams_rank0": per[0]["mhc_streams"]}
        s["passed"] = (s["bounds_equal_main_all_ranks"] and s["bitwise_vs_main_calls"] == s["calls"]
                       and s["stream_ok_calls"] == s["calls"])
        passed = passed and s["passed"]
        summary[T] = s
    plan_ok = all(r["plan_mismatch_count"] == 0 and r["plan_2048"] is None for r in rs)
    passed = passed and plan_ok
    out = {"passed": bool(passed), "head": head, "env0": e0, "plan_equal_main_T1_16384": plan_ok,
           "plan_mismatch_T_rank0": rs[0]["plan_mismatch_T"], "swap_above_rows": rs[0]["swap_above_rows"],
           "max_rows": rs[0]["max_rows"], "shapes": summary}
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "gate_check.json"), "w") as f:
        json.dump({"summary": out, "raw": rs}, f, indent=1, default=str)
    wd.cancel()
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
