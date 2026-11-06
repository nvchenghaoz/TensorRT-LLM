"""Idea 96 / branch 199: 4-rank repeated-execution check of the pipelined prefill boundary (ArFusedHcPipeline.run).

Per rank (mpi4py.futures, 4 GPUs), real tensorrt_llm AllReduce(LOWPRECISION) and a real mHC at V4.1 dims:
  * CALLS pipelined boundary calls per shape (T = 4096 and the mixed-step T = 4099 at k = TRTLLM_DSV41_PREFILL_AR_PIPELINE
    default; T = 8192), enqueued back to back with a fresh seeded input per call, a 4-rank reset barrier every 250
    calls. EVERY call is checked against (a) the unpipelined serial chain on the same inputs (LP full-T AR +
    mHC.fused_hc_lagged): layer_input / residual rel_rms difference, and (b) the fp32-sum reference chain (mHC on the
    exact fp32 sum of the 4 rank inputs, rounded to bf16): rel_rms of pipelined vs serial.
  * Before each reset barrier the LOWPRECISION IPC flags are inspected (lp_ar/bench.py _lp_flag_state): the host
    counter advanced by exactly the calls issued, every peer signalled the last flag, nothing holds the next flag.
  * Every 25th input is re-run alone after the chain: bitwise equal to its in-chain output (determinism).
  * The chunked mHC on identical AR output is bitwise equal per row to the full-M call (identity all-reduce).
Last stdout line: {"passed": ...}.
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
SHAPES = [4096, 4099, 8192]
CALLS = int(os.environ.get("CHAIN_CALLS", "1000"))
CHUNK = 250
# Pipelined vs fp32-sum reference must be no worse than the serial (full-T LP) chain beyond this factor.
REL_FACTOR = 1.25


def log(*a):
    print(*a, flush=True)


def rank_chain(env):
    wd = bench._watchdog("chain", 2400)
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
    pipe = arp.ArFusedHcPipeline()
    st_big = stage0.St(torch, max(SHAPES), n, rank, dev)
    st_big.refill(0)
    with autotune():
        mhc.fused_hc_lagged(st_big.x, st_big.res, st_big.post, st_big.comb, st_big.pre, norm_weight=nw, norm_eps=neps)
    torch.cuda.synchronize()
    res = {"rank": rank, "k": arp.PIPELINE_K or f"auto/{arp.PIPELINE_CHUNK_ROWS}", "shapes": {}}

    def rr_dev(a, b):  # device scalar, no host sync
        d = a.float() - b.float()
        return d.square().sum().sqrt() / b.float().square().sum().sqrt().clamp_min(1e-30)

    def fp(t):  # device fingerprint of a bf16 tensor's bits
        v = t.contiguous().view(torch.int16).reshape(-1, 128).to(torch.int64)
        w = torch.arange(1, 129, device=v.device, dtype=torch.int64)
        return torch.stack([v.sum(), (v * v).sum(), (v * w).sum(), (v.sum(-1) * (torch.arange(
            v.shape[0], device=v.device, dtype=torch.int64) % 9973)).sum()])

    for T in SHAPES:
        k = arp.k_for(T)
        bounds = arp.chunk_bounds(T, k)  # the k(T) the model uses, also below MIN_ROWS
        st = stage0.St(torch, T, n, rank, dev)
        out = {"T": T, "bounds": bounds, "calls": 0, "max_li_rel_vs_serial": 0.0, "max_res_rel_vs_serial": 0.0,
               "max_li_rel_vs_ref_piped": 0.0, "max_li_rel_vs_ref_serial": 0.0, "worse_than_serial_calls": 0,
               "nonfinite_calls": 0, "flags_ok": True, "flag_states": [], "rerun_mismatch": 0,
               "mhc_identity_bitwise": None}
        # identity all-reduce: chunked mHC bitwise per row vs full-M
        st.refill(5)
        full = mhc.fused_hc_lagged(st.x, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
        chk = pipe.run(lambda rows: rows, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
        torch.cuda.synchronize()
        out["mhc_identity_bitwise"] = all(bool(torch.equal(a, b)) for a, b in zip(full, chk))
        del full, chk
        torch.cuda.synchronize()
        bench._barrier()
        f0 = bench._lp_flag_state(torch, mapping)
        out["flag_states"].append(f0)
        out["flags_ok"] = out["flags_ok"] and f0["ok"]
        last = f0["counter"]
        done = 0
        keep = {}
        t0 = time.time()
        while done < CALLS:
            nc = min(CHUNK, CALLS - done)
            seeds = [10000 * T + done + i for i in range(nc)]
            met = torch.zeros((nc, 5), dtype=torch.float32, device=dev)  # li_vs_serial, res_vs_serial, finite, ref_p, ref_s
            met[:, 3:] = -1.0
            torch.cuda.synchronize()
            bench._barrier()  # reset barrier: all 4 ranks start the chunk together
            for i, sd in enumerate(seeds):  # back to back; no host sync inside the chunk
                st.refill(sd)
                p = pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
                s = mhc.fused_hc_lagged(ar(st.x), st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
                met[i, 0] = rr_dev(p[4], s[4])
                met[i, 1] = rr_dev(p[0], s[0])
                met[i, 2] = (torch.isfinite(p[4].float()).all() & torch.isfinite(p[0].float()).all()).float()
                if i % 10 == 0:  # fp32-sum reference chain on every 10th call (cost), device side
                    ins = [bench._gen(torch, T, sd, r, dev).float() for r in range(WORLD)]
                    ref = (ins[0] + ins[1] + ins[2] + ins[3]).to(torch.bfloat16)
                    del ins
                    ro = mhc.fused_hc_lagged(ref, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
                    met[i, 3] = rr_dev(p[4], ro[4])
                    met[i, 4] = rr_dev(s[4], ro[4])
                    del ro, ref
                if i % 25 == 0:
                    keep[sd] = (fp(p[0]), p[4].clone())
                del p, s
            torch.cuda.synchronize()
            bench._barrier()
            # before the next reset barrier: the LP IPC flags must hold what the next call expects
            fs = bench._lp_flag_state(torch, mapping)
            fs["advanced_by"] = fs["counter"] - last
            fs["expected_advance"] = nc * (len(bounds) + 1)
            fs["ok"] = fs["ok"] and fs["advanced_by"] == fs["expected_advance"]
            out["flag_states"].append(fs)
            out["flags_ok"] = out["flags_ok"] and fs["ok"]
            last = fs["counter"]
            m = met.cpu()
            out["max_li_rel_vs_serial"] = max(out["max_li_rel_vs_serial"], m[:, 0].max().item())
            out["max_res_rel_vs_serial"] = max(out["max_res_rel_vs_serial"], m[:, 1].max().item())
            out["nonfinite_calls"] += int((m[:, 2] < 0.5).sum().item())
            refrows = m[m[:, 3] >= 0]
            if refrows.numel():
                out["max_li_rel_vs_ref_piped"] = max(out["max_li_rel_vs_ref_piped"], refrows[:, 3].max().item())
                out["max_li_rel_vs_ref_serial"] = max(out["max_li_rel_vs_ref_serial"], refrows[:, 4].max().item())
                out["worse_than_serial_calls"] += int((refrows[:, 3] > REL_FACTOR * refrows[:, 4]).sum().item())
                out["ref_checked_calls"] = out.get("ref_checked_calls", 0) + int(refrows.shape[0])
            out["calls"] += nc
            done += nc
            if rank == 0:
                log(f"[chain] T={T} k={k} calls={done} li_vs_serial={out['max_li_rel_vs_serial']:.4g} "
                    f"ref piped/serial={out['max_li_rel_vs_ref_piped']:.4g}/{out['max_li_rel_vs_ref_serial']:.4g} "
                    f"flags_ok={out['flags_ok']} adv={fs['advanced_by']}/{fs['expected_advance']}")
        # determinism: every 25th input re-run alone
        torch.cuda.synchronize()
        bench._barrier()
        for sd, (pr, pl) in keep.items():
            st.refill(sd)
            p = pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
            torch.cuda.synchronize()
            out["rerun_mismatch"] += int(not (torch.equal(fp(p[0]), pr) and torch.equal(p[4], pl)))
        torch.cuda.synchronize()
        bench._barrier()
        fs = bench._lp_flag_state(torch, mapping)
        fs["advanced_by"] = fs["counter"] - last
        fs["expected_advance"] = len(keep) * len(bounds)
        fs["ok"] = fs["ok"] and fs["advanced_by"] == fs["expected_advance"]
        out["flag_states"].append(fs)
        out["flags_ok"] = out["flags_ok"] and fs["ok"]
        out["seconds"] = round(time.time() - t0, 1)
        out["reruns"] = len(keep)
        res["shapes"][str(T)] = out
        if rank == 0:
            log(f"[chain] T={T} done: " + json.dumps({kk: v for kk, v in out.items() if kk != 'flag_states'}))
        del st, keep
        torch.cuda.synchronize()
        bench._barrier()
    wd.cancel()
    return res


def main():
    stage0.CKPT = sys.argv[1]
    os.environ["STAGE0_CKPT"] = sys.argv[1]
    wd = bench._watchdog("driver", 2700)
    log(bench._gpu_state())
    head = bench.sh(["git", "-C", HERE, "rev-parse", "HEAD"]).strip()
    log(f"[chain] host={bench.sh(['hostname']).strip()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"head={head}")
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(bench)
    cloudpickle.register_pickle_by_value(stage0)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        rs = sorted(ex.map(rank_chain, [{"_name": "chain", "_apply_nccl_tuning": "1",
                                         "STAGE0_CKPT": sys.argv[1]}] * WORLD), key=lambda r: r["rank"])
    summary = {}
    passed = True
    for T in map(str, SHAPES):
        per = [r["shapes"][T] for r in rs]
        s = {"bounds": per[0]["bounds"], "calls_per_rank": min(p["calls"] for p in per),
             "max_li_rel_vs_serial": max(p["max_li_rel_vs_serial"] for p in per),
             "max_res_rel_vs_serial": max(p["max_res_rel_vs_serial"] for p in per),
             "max_li_rel_vs_ref_piped": max(p["max_li_rel_vs_ref_piped"] for p in per),
             "max_li_rel_vs_ref_serial": max(p["max_li_rel_vs_ref_serial"] for p in per),
             "worse_than_serial_calls": sum(p["worse_than_serial_calls"] for p in per),
             "nonfinite_calls": sum(p["nonfinite_calls"] for p in per),
             "ipc_flags_ok_all_ranks": all(p["flags_ok"] for p in per),
             "flag_checks_per_rank": len(per[0]["flag_states"]),
             "rerun_mismatch": sum(p["rerun_mismatch"] for p in per), "reruns_per_rank": per[0]["reruns"],
             "mhc_identity_bitwise_all_ranks": all(p["mhc_identity_bitwise"] for p in per),
             "flag_states_rank0_last": per[0]["flag_states"][-1]}
        ok = (s["calls_per_rank"] == CALLS and s["nonfinite_calls"] == 0 and s["ipc_flags_ok_all_ranks"]
              and s["rerun_mismatch"] == 0 and s["mhc_identity_bitwise_all_ranks"]
              and s["worse_than_serial_calls"] == 0)
        s["passed"] = ok
        passed = passed and ok
        summary[T] = s
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "chain_check.json"), "w") as f:
        json.dump({"head": head, "summary": summary, "raw": rs}, f, indent=1, default=str)
    wd.cancel()
    print(json.dumps({"passed": passed, "head": head, "k": rs[0]["k"], "summary": summary}, default=str))


if __name__ == "__main__":
    main()
