"""Idea 96 / branch 199 STAGE 0: serial vs k-chunk pipelined (AR + mHC.fused_hc_lagged) per prefill boundary, 4 GPUs.

Each of the 4 MPI ranks (mpi4py.futures) builds the real tensorrt_llm AllReduce (LOWPRECISION = main's served
strategy, and NCCL with the PHB PCIe defaults BaseWorker applies) and a real mHC module at V4.1's dims (config.json),
autotunes mHC like warm-up does, then per T in {2048, 4096, 6144} and k in {1, 2, 3, 4}:
  (a) serial:  y = AR(x); mHC.fused_hc_lagged(y, ...)                     (main's code path)
  (b/c) k-chunk: ArFusedHcPipeline.run (tensorrt_llm/_torch/modules/mhc/ar_pipeline.py)
  (d) per-chunk AR alone, mHC chunk alone (pinned full-M tactic), AR(chunk) on the comm stream || mHC(chunk) on main
  (e) residual = pipelined - max(AR_full, mHC_full) and pipelined - (sum AR chunks + last mHC chunk)
  (g) host us per boundary call (enqueue), every rank
  (h) tactic at chunk M vs full M, us per chunk with each
  (i) LP / NCCL chunked-vs-full per-row diff (bitwise rows, max-abs-diff quantiles, rel_rms vs the fp32 sum)
  mHC chunked bitwise per row vs full-M on identical inputs; pipelined vs serial end to end.
Timing: 20 warmup + 100 timed per sequence, median of 3 sequences, inputs re-created from the seed before each
sequence, max over ranks. Writes $AUTO_DEV_ARTIFACTS/stage0_table.json; last stdout line is a JSON summary.
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lp_ar"))
import bench  # noqa: E402

HIDDEN = 5120
WORLD = 4
TS = [2048, 4096, 6144]
KS = [1, 2, 3, 4]
WARMUP, ITERS, SEQS = 20, 100, 3
CKPT = os.environ.get("STAGE0_CKPT", "")


def log(*a):
    print(*a, flush=True)


def _cfg():
    with open(os.path.join(os.environ.get("STAGE0_CKPT") or CKPT, "config.json")) as f:
        c = json.load(f)
    return {k: c.get(k) for k in ("hc_mult", "hc_sinkhorn_iters", "hc_eps", "rms_norm_eps", "hidden_size")}


class St:
    """Boundary inputs for T rows: rank-local AR input x, replicated residual / post / comb / pre."""

    def __init__(self, torch, T, n, rank, dev):
        self.torch, self.T, self.n, self.rank, self.dev = torch, T, n, rank, dev
        self.x = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
        self.res = torch.empty((T, n, HIDDEN), dtype=torch.bfloat16, device=dev)
        self.post = torch.empty((T, n, 1), dtype=torch.float32, device=dev)
        self.comb = torch.empty((T, n, n), dtype=torch.float32, device=dev)
        self.pre = torch.empty((T, n, 1), dtype=torch.float32, device=dev)

    def refill(self, seed):
        torch = self.torch
        self.x.copy_(bench._gen(torch, self.T, seed, self.rank, self.dev))
        g = torch.Generator(device=self.dev)
        g.manual_seed(31337 + 7 * seed + self.T)  # replicated tensors: same on every rank
        self.res.copy_(torch.randn(self.res.shape, generator=g, device=self.dev).to(torch.bfloat16))
        self.post.copy_(2.0 * torch.rand(self.post.shape, generator=g, device=self.dev))
        c = torch.rand(self.comb.shape, generator=g, device=self.dev) + 0.05
        self.comb.copy_(c / c.sum(-2, keepdim=True))
        self.pre.copy_(torch.rand(self.pre.shape, generator=g, device=self.dev) + 1e-6)


def _timeit(torch, fn, refill, sync_each=False):
    seq_meds, host_meds, flat = [], [], []
    for s in range(SEQS):
        refill(s)
        torch.cuda.synchronize()
        bench._barrier()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        bench._barrier()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
        hts = []
        for i in range(ITERS):
            ev[i][0].record()
            t = time.perf_counter()
            fn()
            hts.append(time.perf_counter() - t)
            ev[i][1].record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) for a, b in ev)
        flat += ts
        seq_meds.append(ts[len(ts) // 2])
        hts.sort()
        host_meds.append(hts[len(hts) // 2] * 1e6)
    flat.sort()
    n = len(flat)
    med = sorted(seq_meds)[SEQS // 2]
    return {"us": med * 1e3, "seq_us": [v * 1e3 for v in seq_meds], "p10_us": flat[n // 10] * 1e3,
            "p90_us": flat[(9 * n) // 10] * 1e3, "noise_floor": (flat[(9 * n) // 10] - flat[n // 10]) / med,
            "host_us": sorted(host_meds)[SEQS // 2]}


def rank_stage0(env):
    wd = bench._watchdog("stage0", 2400)
    torch, rank, mapping, applied = bench._setup(env)
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.modules.mhc import ar_pipeline as arp
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (_FUSED_HC_BACKEND_CODE, _alloc_fused_hc_scratch,
                                                           _fused_hc_call, mhc_fused_hc_tactic)
    dev = torch.device("cuda", rank)
    cfg = _cfg()
    n = int(cfg["hc_mult"])
    assert int(cfg["hidden_size"]) == HIDDEN
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
    ars = {"LP": AllReduce(mapping=mapping, strategy=AllReduceStrategy.LOWPRECISION, dtype=torch.bfloat16),
           "NCCL_PHB": AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=torch.bfloat16)}
    pipe = arp.ArFusedHcPipeline()
    sts = {T: St(torch, T, n, rank, dev) for T in TS}
    for T in TS:
        sts[T].refill(0)
    t0 = time.time()
    with autotune():
        st = sts[max(TS)]
        mhc.fused_hc_lagged(st.x, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
    torch.cuda.synchronize()
    if rank == 0:
        log(f"[stage0] mHC autotune done in {time.time()-t0:.0f}s cfg={cfg}")

    def tactic_of(T, m=None):
        st = sts[T]
        m = T if m is None else m
        return mhc_fused_hc_tactic(st.x[:m], st.res[:m].reshape(m, n, HIDDEN), st.post[:m].reshape(m, n),
                                   st.comb[:m].reshape(m, n, n), mhc.fn, mhc.scale, mhc.base, n, HIDDEN,
                                   mhc.norm_eps, mhc.eps, mhc.sinkhorn_eps, mhc.post_mult_value, mhc.sinkhorn_iters,
                                   norm_weight=nw, norm_eps=neps, pre_mix_ext=st.pre[:m].reshape(m, n))

    def mhc_chunk(st, y, a, b, tactic, outs):
        backend, tn, ks, bs, tm = tactic
        m = b - a
        wy, wr, wd_ = _alloc_fused_hc_scratch(backend, m, n, ks, tm, dev)
        _fused_hc_call(_FUSED_HC_BACKEND_CODE[backend], tn, ks, bs, tm, y, st.res[a:b], st.post[a:b], st.comb[a:b],
                       mhc.fn, mhc.scale, mhc.base, outs[0][a:b], outs[1][a:b], outs[2][a:b], outs[3][a:b], wy, wr,
                       wd_, m, HIDDEN, n, mhc.norm_eps, mhc.eps, mhc.sinkhorn_eps, mhc.post_mult_value,
                       mhc.sinkhorn_iters, norm_weight=nw, norm_eps=neps, pre_mix_ext=st.pre[a:b].reshape(m, n),
                       pre_mix_out=outs[4][a:b])

    def new_outs(T):
        return (torch.empty((T, n, HIDDEN), dtype=torch.bfloat16, device=dev),
                torch.empty((T, n), dtype=torch.float32, device=dev),
                torch.empty((T, n * n), dtype=torch.float32, device=dev),
                torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev),
                torch.empty((T, n), dtype=torch.float32, device=dev))

    res = {"rank": rank, "applied": applied, "timing": {}, "check": {}, "tactics": {}, "kernels": {}}
    for T in TS:
        st = sts[T]
        full_tac = tactic_of(T)
        res["tactics"][str(T)] = {"full": list(full_tac)}
        for name, ar in ars.items():
            key = f"{name}_{T}"
            row = {}
            row["serial"] = _timeit(torch, lambda: mhc.fused_hc_lagged(ar(st.x), st.res, st.post, st.comb, st.pre,
                                                                         norm_weight=nw, norm_eps=neps), st.refill)
            row["ar_full"] = _timeit(torch, lambda: ar(st.x), st.refill)
            y_full_keep = ar(st.x)
            row["mhc_full"] = _timeit(torch, lambda: mhc.fused_hc_lagged(y_full_keep, st.res, st.post, st.comb,
                                                                           st.pre, norm_weight=nw, norm_eps=neps),
                                      st.refill)
            for k in KS[1:]:
                bounds = arp.chunk_bounds(T, k)
                if bounds is None or len(bounds) != k:
                    row[f"k{k}"] = {"not_evaluated": f"bounds={bounds}"}
                    continue
                kr = {"bounds": bounds}
                kr["pipelined"] = _timeit(torch, lambda: pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre,
                                                                    nw, neps, bounds), st.refill)
                a0, b0 = bounds[0]
                m = b0 - a0
                kr["ar_chunk"] = _timeit(torch, lambda: ar(st.x[a0:b0]), st.refill)
                outs = new_outs(T)
                yc = ar(st.x[a0:b0])
                kr["mhc_chunk_pinned"] = _timeit(torch, lambda: mhc_chunk(st, yc, a0, b0, full_tac, outs), st.refill)
                own = tactic_of(T, m)
                kr["tactic_chunk_own"] = list(own)
                kr["mhc_chunk_own"] = _timeit(torch, lambda: mhc_chunk(st, yc, a0, b0, own, outs), st.refill)
                a1, b1 = bounds[1]

                def overlap():
                    main = torch.cuda.current_stream()
                    pipe.stream.wait_stream(main)
                    with torch.cuda.stream(pipe.stream):
                        y = ar(st.x[a1:b1])
                    mhc_chunk(st, yc, a0, b0, full_tac, outs)
                    main.wait_stream(pipe.stream)
                    y.record_stream(main)
                kr["overlap_ar_mhc_chunk"] = _timeit(torch, overlap, st.refill)
                ar_sum = sum(_ar_us for _ar_us in [kr["ar_chunk"]["us"]] * k)
                kr["ideal_us"] = ar_sum + kr["mhc_chunk_pinned"]["us"]
                kr["residual_vs_max_us"] = kr["pipelined"]["us"] - max(row["ar_full"]["us"], row["mhc_full"]["us"])
                kr["residual_vs_ideal_us"] = kr["pipelined"]["us"] - kr["ideal_us"]
                kr["host_delta_us"] = kr["pipelined"]["host_us"] - row["serial"]["host_us"]
                row[f"k{k}"] = kr
                if rank == 0 and k == 2:
                    res["kernels"][key] = bench._kernel_names(
                        torch, lambda: pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds))
                elif k == 2:
                    pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
                torch.cuda.synchronize()
            res["timing"][key] = row
            if rank == 0:
                s = row["serial"]
                ks_txt = " ".join(f"k{k}={row[f'k{k}']['pipelined']['us']:.1f}(h{row[f'k{k}']['host_delta_us']:+.0f})"
                                  for k in KS[1:] if "pipelined" in row[f"k{k}"])
                log(f"[stage0] {key} serial={s['us']:.1f}us host={s['host_us']:.0f} ar={row['ar_full']['us']:.1f} "
                    f"mhc={row['mhc_full']['us']:.1f} {ks_txt}")
            # ---- correctness (untimed): chunked-vs-full AR rows, mHC bitwise per row, pipelined vs serial
            st.refill(99)
            torch.cuda.synchronize()
            ins = [bench._gen(torch, T, 99, r, dev).float() for r in range(WORLD)]
            ref = ins[0] + ins[1] + ins[2] + ins[3]
            del ins

            def rel_rms(o):
                d = o.float() - ref
                return (d.square().sum().sqrt() / ref.square().sum().sqrt()).item()
            y_full = ar(st.x).clone()
            ck = {"rel_rms_full": rel_rms(y_full)}
            full_out = mhc.fused_hc_lagged(y_full, st.res, st.post, st.comb, st.pre, norm_weight=nw, norm_eps=neps)
            full_out = [t.clone() for t in full_out]
            ref_out = mhc.fused_hc_lagged(ref.to(torch.bfloat16), st.res, st.post, st.comb, st.pre, norm_weight=nw,
                                          norm_eps=neps)

            def li_err(li):
                d = li.float() - ref_out[4].float()
                return (d.square().sum().sqrt() / ref_out[4].float().square().sum().sqrt()).item()
            ck["serial_layer_input_rel_rms_vs_fp32ar"] = li_err(full_out[4])
            for k in KS[1:]:
                bounds = arp.chunk_bounds(T, k)
                if bounds is None or len(bounds) != k:
                    continue
                y_ch = torch.cat([ar(st.x[a:b]) for a, b in bounds])
                torch.cuda.synchronize()
                eq_rows = (y_full.view(torch.int16) == y_ch.view(torch.int16)).all(-1)
                d = (y_full.float() - y_ch.float()).abs().amax(-1)
                q = torch.quantile(d, torch.tensor([0.5, 0.9, 0.99, 1.0], device=dev)).tolist()
                c = {"ar_rows_bitwise_equal_frac": eq_rows.float().mean().item(),
                     "ar_row_maxabs_diff_q50_q90_q99_max": q, "rel_rms_chunked": rel_rms(y_ch)}
                # mHC chunked on the identical AR output (identity 'all-reduce') vs full-M: bitwise per row
                ch_out = pipe.run(lambda rows: rows, mhc, y_full, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
                torch.cuda.synchronize()
                names = ("residual", "pre", "post", "comb", "layer_input")
                c["mhc_chunked_bitwise"] = {nm: bool(torch.equal(a_, b_)) for nm, a_, b_ in zip(names, full_out, ch_out)}
                # end to end: pipelined (real chunk ARs) vs serial
                p_out = pipe.run(ar, mhc, st.x, st.res, st.post, st.comb, st.pre, nw, neps, bounds)
                torch.cuda.synchronize()
                c["pipelined_layer_input_rel_rms_vs_fp32ar"] = li_err(p_out[4])
                c["pipelined_vs_serial_layer_input_maxabs"] = (p_out[4].float() - full_out[4].float()).abs().max().item()
                c["pipelined_vs_serial_residual_maxabs"] = (p_out[0].float() - full_out[0].float()).abs().max().item()
                c["finite"] = bool(all(torch.isfinite(t.float()).all().item() for t in p_out))
                ck[f"k{k}"] = c
                del y_ch, ch_out, p_out
            res["check"][key] = ck
            if rank == 0:
                log(f"[check] {key} " + json.dumps(ck)[:900])
            del y_full, full_out, ref_out, ref, y_full_keep
            torch.cuda.synchronize()
            bench._barrier()
    bench._barrier()
    wd.cancel()
    return res


def run_ranks(fn, arg):
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(bench)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        out = list(ex.map(fn, [arg] * WORLD))
    return sorted(out, key=lambda r: r["rank"])


def main():
    global CKPT
    CKPT = sys.argv[1]
    os.environ["STAGE0_CKPT"] = CKPT
    wd = bench._watchdog("driver", 2700)
    t0 = time.time()
    log(bench._gpu_state())
    head = bench.sh(["git", "-C", HERE, "rev-parse", "HEAD"]).strip()
    log(f"[stage0] host={bench.sh(['hostname']).strip()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"head={head} load={os.getloadavg()}")
    rs = run_ranks(rank_stage0, {"_name": "stage0", "_apply_nccl_tuning": "1", "STAGE0_CKPT": CKPT})
    log(bench._gpu_state())
    table = {"idea": 96, "branch": 199, "head": head, "host": bench.sh(["hostname"]).strip(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "applied_nccl": rs[0]["applied"],
             "tactics_rank0": rs[0]["tactics"], "kernels_rank0_k2": rs[0]["kernels"],
             "SP_rows": "not_evaluated", "rows": {}}
    for key in rs[0]["timing"]:
        r0 = rs[0]["timing"][key]
        row = {}
        for part in ("serial", "ar_full", "mhc_full"):
            row[part + "_us"] = round(max(r["timing"][key][part]["us"] for r in rs), 1)
        row["serial_host_us_per_rank"] = [round(r["timing"][key]["serial"]["host_us"], 1) for r in rs]
        row["serial_noise"] = round(max(r["timing"][key]["serial"]["noise_floor"] for r in rs), 4)
        for k in KS[1:]:
            kk = f"k{k}"
            if "pipelined" not in r0[kk]:
                row[kk] = r0[kk]
                continue
            g = lambda part: round(max(r["timing"][key][kk][part]["us"] for r in rs), 1)  # noqa: E731
            row[kk] = {"bounds": r0[kk]["bounds"], "pipelined_us": g("pipelined"),
                       "pipelined_noise": round(max(r["timing"][key][kk]["pipelined"]["noise_floor"] for r in rs), 4),
                       "saving_us": round(row["serial_us"] - g("pipelined"), 1),
                       "ar_chunk_us": g("ar_chunk"), "mhc_chunk_pinned_us": g("mhc_chunk_pinned"),
                       "mhc_chunk_own_us": g("mhc_chunk_own"), "tactic_chunk_own_rank0": r0[kk]["tactic_chunk_own"],
                       "overlap_ar_mhc_chunk_us": g("overlap_ar_mhc_chunk"),
                       "residual_vs_max_us": round(max(r["timing"][key][kk]["residual_vs_max_us"] for r in rs), 1),
                       "residual_vs_ideal_us": round(max(r["timing"][key][kk]["residual_vs_ideal_us"] for r in rs), 1),
                       "host_delta_us_per_rank": [round(r["timing"][key][kk]["host_delta_us"], 1) for r in rs],
                       "pipelined_host_us_per_rank": [round(r["timing"][key][kk]["pipelined"]["host_us"], 1)
                                                      for r in rs]}
        row["check_rank0"] = rs[0]["check"][key]
        row["check_all_ranks_mhc_bitwise"] = all(
            all(v for v in r["check"][key][kk]["mhc_chunked_bitwise"].values())
            for r in rs for kk in r["check"][key] if kk.startswith("k"))
        row["check_max_rel_rms_chunked"] = max(
            [r["check"][key][kk]["rel_rms_chunked"] for r in rs for kk in r["check"][key] if kk.startswith("k")]
            or [0.0])
        row["check_max_rel_rms_full"] = max(r["check"][key]["rel_rms_full"] for r in rs)
        table["rows"][key] = row
    passed = all(row["check_all_ranks_mhc_bitwise"] for row in table["rows"].values())
    table["passed"] = passed
    table["wall_s"] = round(time.time() - t0, 1)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "stage0_table.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(art, "stage0_raw.json"), "w") as f:
        json.dump(rs, f, default=str)
    summ = {k: {kk: (v[kk]["pipelined_us"] if isinstance(v.get(kk), dict) and "pipelined_us" in v[kk] else None)
                for kk in ("k2", "k3", "k4")} | {"serial_us": v["serial_us"]} for k, v in table["rows"].items()}
    wd.cancel()
    print(json.dumps({"passed": passed, "head": head, "summary_us": summ}))


if __name__ == "__main__":
    main()
