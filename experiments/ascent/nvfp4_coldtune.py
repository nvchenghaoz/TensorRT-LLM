# NVFP4 W4A4 GEMM AutoTuner cold-L2 study (idea 27), model-free, synthetic operands of the
# real Qwen3.8-27B MLP shapes: gate_up (N=34816, K=5120) and down (N=5120, K=17408).
#
# usage: python nvfp4_coldtune.py <mode>
#   bench  : tune with THIS TREE's NVFP4GemmUnifiedRunner.tuning_config (main: warm L2,
#            branch: use_cold_l2_cache=True), then time the picked (backend, tactic) per M
#            cold (>= 400 MB of distinct weight copies rotated inside one CUDA graph, PDL as
#            configured by env). Last line: evidence JSON.
#   check  : tune warm (main's config) and cold (branch config) in-process; for every M,
#            error of both picks vs an FP32 reference built from the same FP4 operands, and an
#            untimed repeated-execution sequence (same warmup/iters/rotation as timing) of
#            the tree's pick, bitwise vs iteration 1.
#   study  : picks (warm x2, cold x2, with tuning time + peak memory), per-tactic cold
#            timings for every valid tactic at M=1,2,4,8, and the streaming-read ceiling.
import dataclasses
import json
import os
import sys
import time

import torch

SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}
MS = (1, 2, 3, 4, 8, 2048, 8192)
DECODE_MS = (1, 2, 4, 8)
ROT_BYTES = 400 << 20  # distinct weight copies per graph replay (>= 256 MB, ~3x L2)
WARM, ITERS = 10, 60
BACKENDS = "cutlass,cublaslt,cuda_core"
OP = "trtllm::nvfp4_gemm::gemm"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def log(*a):
    print(*a, flush=True)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def stats(ts):
    med = pct(ts, 0.5)
    return {"median_ms": med, "p10_ms": pct(ts, 0.1), "p90_ms": pct(ts, 0.9),
            "noise": (pct(ts, 0.9) - pct(ts, 0.1)) / med, "n": len(ts)}


def quant(x, gs, swizzled=True):
    return torch.ops.trtllm.fp4_quantize(x, gs, 16, False, swizzled)


class Shape:
    def __init__(self, name, n, k):
        self.name, self.n, self.k = name, n, k
        g = torch.Generator(device="cuda").manual_seed(n * 7 + k)
        w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        self.gs_w = (448.0 * 6.0 / w.float().abs().max()).reshape(1).float()
        self.w, self.w_sf = quant(w, self.gs_w)
        _, self.w_sf_lin = quant(w, self.gs_w, swizzled=False)
        del w
        self.wbytes = self.w.numel() + self.w_sf.numel()
        self.nb = max(2, -(-ROT_BYTES // self.wbytes))
        self.copies = [(self.w, self.w_sf)] + [(self.w.clone(), self.w_sf.clone())
                                               for _ in range(self.nb - 1)]
        self.acts = {}
        for m in MS:
            a = (torch.randn(m, k, generator=g, device="cuda")).to(torch.bfloat16)
            gs_a = (448.0 * 6.0 / a.float().abs().max()).reshape(1).float()
            af, asf = quant(a, gs_a)
            _, asf_lin = quant(a, gs_a, swizzled=False)
            alpha = (1.0 / (gs_a * self.gs_w)).float()
            self.acts[m] = dict(a=af, sf=asf, sf_lin=asf_lin, alpha=alpha)
        log("  shape %s N=%d K=%d weight+SF %.1f MB, %d rotated copies (%.0f MB)" %
            (name, n, k, self.wbytes / 1e6, self.nb, self.nb * self.wbytes / 1e6))

    def inputs(self, m, i=0):
        a = self.acts[m]
        w, wsf = self.copies[i % self.nb]
        return [a["a"], w, a["sf"], wsf, a["alpha"]]

    def ref(self, m):
        lut = torch.tensor(E2M1, dtype=torch.float32, device="cuda")

        def deq(q, sf_lin, rows, cols):
            q = q.view(torch.uint8)
            v = torch.empty(rows, cols, dtype=torch.float32, device="cuda")
            v[:, 0::2] = lut[(q & 0x0F).long()]
            v[:, 1::2] = lut[((q >> 4) & 0x0F).long()]
            s = sf_lin.view(torch.uint8)[:rows * (cols // 16)].view(torch.float8_e4m3fn)
            s = s.float().view(rows, cols // 16)
            return (v.view(rows, cols // 16, 16) * s.unsqueeze(-1)).view(rows, cols)

        a = self.acts[m]
        A = deq(a["a"], a["sf_lin"], m, self.k)
        out = torch.empty(m, self.n, dtype=torch.float32, device="cuda")
        ch = 8192
        for r0 in range(0, self.n, ch):
            W = deq(self.w[r0:r0 + ch], self.w_sf_lin[r0 * (self.k // 16):(r0 + ch) * (self.k // 16)],
                    min(ch, self.n - r0), self.k)
            out[:, r0:r0 + ch] = A @ W.T
            del W
        return out * a["alpha"]


def runner():
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    return NVFP4GemmUnifiedRunner(0, torch.bfloat16, BACKENDS.split(","))


def set_cold(cold):
    """None: keep this tree's config. True/False: override use_cold_l2_cache in-process."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    if not hasattr(set_cold, "orig"):
        set_cold.orig = NVFP4GemmUnifiedRunner.tuning_config
    cfg = set_cold.orig if cold is None else dataclasses.replace(set_cold.orig,
                                                                 use_cold_l2_cache=cold)
    NVFP4GemmUnifiedRunner.tuning_config = cfg
    return cfg.use_cold_l2_cache


def tune(shapes, cold=None):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    eff = set_cold(cold)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    info = {"use_cold_l2_cache": eff, "per_shape": {}}
    for s in shapes:
        t0 = time.perf_counter()
        with autotune():
            torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16, allowed_backends=BACKENDS)
        torch.cuda.synchronize()
        info["per_shape"][s.name] = {"tune_s": round(time.perf_counter() - t0, 3)}
        log("  tuned %s (use_cold_l2_cache=%s) in %.1f s" % (s.name, eff,
                                                             time.perf_counter() - t0))
    info["peak_extra_mb"] = round((torch.cuda.max_memory_allocated() - base) / 1e6, 1)
    picks = {}
    r = runner()
    for s in shapes:
        picks[s.name] = {}
        for m in MS:
            hit, rid, tac, t = tuner.profiling_cache.search_cache(
                OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))),
                NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
            picks[s.name][m] = {"tactic": list(tac) if isinstance(tac, tuple) else tac,
                                "hit": bool(hit), "tuner_ms": t}
    info["cache_entries"] = len(tuner.profiling_cache.cache)
    return picks, info


def as_tac(t):
    return tuple(t) if isinstance(t, list) else t


def build_graph(s, m, tac, r):
    """One CUDA graph with nb back-to-back calls, each over a distinct weight copy."""
    for i in range(s.nb):
        r(s.inputs(m, i), tactic=tac)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for i in range(s.nb):
            r(s.inputs(m, i), tactic=tac)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [r(s.inputs(m, i), tactic=tac) for i in range(s.nb)]
    return g, outs


def time_graph(g, calls, warm=WARM, iters=ITERS):
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warm):
        g.replay()
    for i in range(iters):
        torch.cuda._sleep(100000)  # host runs ahead: no launch gap inside the timed window
        e0[i].record()
        g.replay()
        e1[i].record()
    torch.cuda.synchronize()
    return [e0[i].elapsed_time(e1[i]) / calls for i in range(iters)]


def time_tactic(s, m, tac, warm=WARM, iters=ITERS):
    r = runner()
    g, _ = build_graph(s, m, as_tac(tac), r)
    st = stats(time_graph(g, s.nb, warm, iters))
    st["TBps"] = s.wbytes / (st["median_ms"] * 1e-3) / 1e12
    del g
    return st


def kernel_names(s, m, tac):
    from torch.profiler import ProfilerActivity, profile
    r = runner()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        r(s.inputs(m), tactic=as_tac(tac))
        torch.cuda.synchronize()
    return sorted({e.key[:90] for e in prof.key_averages() if e.device_type.name == "CUDA"})


# ---------------------------------------------------------------- streaming-read ceiling
def ceiling():
    import triton
    import triton.language as tl

    @triton.jit
    def _read(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        for off in range(pid * BLOCK, n, nprog * BLOCK):
            offs = off + tl.arange(0, BLOCK)
            acc ^= tl.load(x_ptr + offs, mask=offs < n, other=0)
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))

    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    sizes_mb = (31.5, 50.1, 73.4, 83.9, 100.3)
    nchunk = 16
    maxb = int(max(sizes_mb) * 1e6) // 64 * 64
    pool = torch.randint(0, 2**31 - 1, (nchunk * maxb // 4,), dtype=torch.int32, device="cuda")
    out = torch.zeros(nsm * 16, dtype=torch.int32, device="cuda")
    cfgs = [(2048, 4, 4), (4096, 8, 4), (4096, 4, 8), (8192, 8, 2), (2048, 4, 8)]
    res = {}
    for smb in sizes_mb:
        nb = int(smb * 1e6) // 64 * 64
        n = nb // 4
        chunks = [pool[i * (maxb // 4):i * (maxb // 4) + n] for i in range(nchunk)]
        best = None
        for blk, nw, per in cfgs:
            grid = (nsm * per,)

            def launch(c):
                _read[grid](c, out, n, BLOCK=blk, num_warps=nw)

            for c in chunks:
                launch(c)
            torch.cuda.synchronize()
            # form B: 16 back-to-back launches over distinct chunks in one graph
            gB = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gB):
                for c in chunks:
                    launch(c)
            tB = stats(time_graph(gB, nchunk, 5, 30))
            # form A: one chunk per launch (one graph per chunk), rotating chunks
            gA = []
            for c in chunks:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    launch(c)
                gA.append(g)
            for g in gA:
                g.replay()
            tsA = []
            for it in range(48):
                g = gA[it % nchunk]
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda._sleep(100000)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                tsA.append(e0.elapsed_time(e1))
            tA = stats(tsA)
            row = {"cfg": [blk, nw, per],
                   "B_TBps": nb / (tB["median_ms"] * 1e-3) / 1e12,
                   "B_p90_TBps": nb / (tB["p90_ms"] * 1e-3) / 1e12,
                   "A_TBps": nb / (tA["median_ms"] * 1e-3) / 1e12,
                   "A_p90_TBps": nb / (tA["p90_ms"] * 1e-3) / 1e12}
            log("  read %.1f MB cfg %s: back-to-back %.3f TB/s (p90 %.3f), single %.3f TB/s (p90 %.3f)"
                % (smb, row["cfg"], row["B_TBps"], row["B_p90_TBps"], row["A_TBps"], row["A_p90_TBps"]))
            if best is None or row["B_TBps"] > best["B_TBps"]:
                best = row
            del gB, gA
        res[str(smb)] = best
    del pool
    torch.cuda.empty_cache()
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    import tensorrt_llm  # noqa: F401  (loads torch.ops.trtllm)
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.cuda.get_device_properties(0)
    log("device %s sm%d%d %d SMs L2 %d MB; tree tuning_config.use_cold_l2_cache=%s; "
        "TLLM_AUTOTUNER_CACHE_PATH=%s TRTLLM_ENABLE_PDL=%s" %
        (dev.name, dev.major, dev.minor, dev.multi_processor_count, dev.L2_cache_size >> 20,
         NVFP4GemmUnifiedRunner.tuning_config.use_cold_l2_cache,
         os.environ.get("TLLM_AUTOTUNER_CACHE_PATH"), os.environ.get("TRTLLM_ENABLE_PDL")))
    shapes = [Shape(nm, n, k) for nm, (n, k) in SHAPES.items()]
    byname = {s.name: s for s in shapes}

    if mode == "bench":
        picks, info = tune(shapes, None)
        log("tune info %s" % json.dumps(info))
        log("picks %s" % json.dumps(picks))
        batch, detail = {}, {}
        for m in MS:
            tot, noise, d = 0.0, 0.0, {}
            for s in shapes:
                tac = picks[s.name][m]["tactic"]
                st = time_tactic(s, m, tac)
                d[s.name] = dict(st, tactic=tac)
                tot += st["median_ms"]
                noise = max(noise, st["noise"])
                log("  M=%d %s tactic %s: median %.4f ms p90 %.4f ms %.3f TB/s noise %.3f" %
                    (m, s.name, tac, st["median_ms"], st["p90_ms"], st["TBps"], st["noise"]))
            batch[str(m)] = {"kernel_ms": round(tot, 5), "noise_floor": round(noise, 4)}
            detail[str(m)] = d
        print(json.dumps({"batch": batch, "output": {"class": "reordered"},
                          "use_cold_l2_cache": info["use_cold_l2_cache"], "tune": info,
                          "detail": detail}))
        return

    if mode == "check":
        res, ok = {}, True
        pk = {}
        for cold in (False, True):
            p, _ = tune(shapes, cold)
            pk["cold" if cold else "warm"] = p
        tree = "cold" if set_cold(None) else "warm"
        log("picks warm %s\npicks cold %s; tree arm=%s" %
            (json.dumps(pk["warm"]), json.dumps(pk["cold"]), tree))
        r = runner()
        for s in shapes:
            for m in MS:
                ref = s.ref(m)
                ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
                errs = {}
                for arm in ("warm", "cold"):
                    tac = as_tac(pk[arm][s.name][m]["tactic"])
                    o = r(s.inputs(m), tactic=tac).float()
                    dd = (o - ref).abs()
                    errs[arm] = {"tactic": pk[arm][s.name][m]["tactic"], "max_abs": dd.max().item(),
                                 "mean_abs": dd.mean().item(),
                                 "finite": bool(torch.isfinite(o).all())}
                cand = errs[tree]
                main_e = errs["warm"]
                tol_ok = (cand["finite"]
                          and cand["max_abs"] <= 1.25 * main_e["max_abs"] + ulp
                          and cand["mean_abs"] <= 1.25 * main_e["mean_abs"] + ulp)
                # untimed repeated execution: same warmup/iters/rotation as the timed loop
                tac = as_tac(pk[tree][s.name][m]["tactic"])
                g, outs = build_graph(s, m, tac, r)
                first, nmis, per_it = None, 0, []
                for it in range(WARM + ITERS):
                    g.replay()
                    torch.cuda.synchronize()
                    cur = torch.stack([o.clone() for o in outs])
                    if first is None:
                        first = cur[0].clone()
                    nmis += int(sum(not torch.equal(cur[j], first) for j in range(s.nb)))
                    per_it.append((cur.float() - ref).abs().max().item())
                del g, outs
                rep_ok = nmis == 0
                res["%s/M%d" % (s.name, m)] = {
                    "errors": errs, "bf16_ulp": ulp, "tol_ok": tol_ok,
                    "repeat": {"iters": WARM + ITERS, "calls_per_iter": s.nb,
                               "mismatch_vs_iter1": nmis, "per_iter_max_err_min": min(per_it),
                               "per_iter_max_err_max": max(per_it)}}
                ok = ok and tol_ok and rep_ok
                log("  %s M=%d warm %s cold %s tol_ok %s repeat mismatches %d" %
                    (s.name, m, errs["warm"], errs["cold"], tol_ok, nmis))
                del ref
                torch.cuda.empty_cache()
        print(json.dumps({"passed": ok, "tree_arm": tree, "compared": (
            "tree's autotuner pick vs FP32 reference from the same FP4 operands (<= 1.25x main warm-L2 "
            "pick error + 1 bf16 ulp); graph replays (nb distinct weight copies x 70 iters) bitwise vs "
            "iteration 1"), "results": res}))
        return

    # study
    out = {"picks": {}, "tune": {}}
    for tag, cold in (("warm1", False), ("cold1", True), ("warm2", False), ("cold2", True)):
        p, info = tune(shapes, cold)
        out["picks"][tag], out["tune"][tag] = p, info
        log("%s tune %s" % (tag, json.dumps(info)))
        log("%s picks %s" % (tag, json.dumps(p)))
    # kernel names of every distinct pick
    names = {}
    for s in shapes:
        for m in MS:
            for tag in out["picks"]:
                tac = out["picks"][tag][s.name][m]["tactic"]
                key = "%s/M%d/%s" % (s.name, m, tac)
                if key not in names:
                    try:
                        names[key] = kernel_names(s, m, tac)
                    except Exception as e:  # noqa: BLE001
                        names[key] = ["err %s" % e]
                    log("  kernels %s: %s" % (key, names[key]))
    out["kernels"] = names
    # every valid tactic, cold, at decode M
    r = runner()
    scan = {}
    for s in shapes:
        for m in DECODE_MS:
            tacs = r.get_valid_tactics(s.inputs(m), None)
            rows = []
            for tac in tacs:
                try:
                    st = time_tactic(s, m, tac, 5, 30)
                    rows.append((st["median_ms"], st["p90_ms"], st["TBps"], list(tac)))
                except Exception as e:  # noqa: BLE001
                    torch.cuda.synchronize()
                    rows.append((float("inf"), float("inf"), 0.0, list(tac)))
            rows.sort()
            best = {}
            for row in rows:
                best.setdefault(row[3][0], row)
            scan["%s/M%d" % (s.name, m)] = {"best_per_backend": best, "top5": rows[:5],
                                           "n_tactics": len(rows)}
            log("  scan %s M=%d (%d tactics): best per backend %s" % (s.name, m, len(rows), best))
    out["cold_scan"] = scan
    for s in shapes:
        del s.copies
    torch.cuda.empty_cache()
    out["read_ceiling"] = ceiling()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
