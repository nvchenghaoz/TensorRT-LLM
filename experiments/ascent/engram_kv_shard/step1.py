"""Idea 224 step 1 (1 GPU): Engram WKV N-shard GEMM + b12x core fence.

Driver (no args) runs fresh subprocesses, because the b12x fence is a compile-time flag
read from FLASHINFER_B12X_CORE_FENCE at kernel construction:
  check  (fence on):  (a) swizzle proof: block_scale_interleave(raw[r*6400:(r+1)*6400]) equals
                      the byte range [r*50*48*512, (r+1)*50*48*512) of the full swizzled scale;
                      (b) graph-captured quantize + full [M,25600,6144] + 4 shard [M,6400,6144]
                      GEMMs, outputs NaN-poisoned before each of >=30 replays per M, every shard
                      compared bitwise with the full output's columns, full vs eager reference,
                      and vs a dequantized FP32 reference (tolerance); (c) per-call us of full and
                      shard (cold weights: rotated copies > L2), plus kernel names / tile_n.
  s2a    (fence 0 and 1): the kv graph (3 full calls, or 12 shard calls = same bytes) on stream A
                      beside one graph of 48 mHC fused_hc_lagged_split forks (K2 on aux) on
                      stream B; >=90 windows per M and arm; every window every output compared
                      bitwise; input checksums every window; measured overlap per window.
  cost   (ABBA: fence 1, 0, 0, 1): per-call us of wq_b [8192,1280], o_b [5120,2048],
                      index_wq_b [4096,1280] and the kv shard [6400,6144] at M=1/2/4, graph,
                      rotated cold weights; output sha256 per shape/M (fenced vs unfenced alone
                      must be bitwise).
Inputs are fixed-seed and held in module-level KEEP lists (never closures).
"""

import hashlib
import json
import os
import subprocess
import sys
import time

KEEP = []  # every tensor a captured graph reads stays alive here
N_FULL, K = 25600, 6144
TP = 4
N_SHARD = N_FULL // TP
SF_COLS = K // 32  # 192, already a multiple of 4
MS = (1, 2, 4)


def log(*a):
    print(*a, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=60).stdout
    except Exception as e:  # noqa
        return f"ERR {e}"


# --------------------------------------------------------------------------------------- common
def _setup():
    import torch

    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x as k
    fence = k.b12x_core_fence_enabled() if hasattr(k, "b12x_core_fence_enabled") else None
    log(f"[setup] FLASHINFER_B12X_CORE_FENCE={os.environ.get('FLASHINFER_B12X_CORE_FENCE')} "
        f"kernel fence flag={fence} device={torch.cuda.get_device_name()}")
    return torch, fence


def make_kv_weight(torch, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = (torch.randn((N_FULL, K), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    tile = torch.randint(118, 127, (N_FULL // 32, SF_COLS), generator=g, device=dev,
                         dtype=torch.int32).to(torch.uint8)
    raw = tile.repeat_interleave(32, dim=0).contiguous()  # expand_engram_scales form
    sw = torch.ops.trtllm.block_scale_interleave(raw).flatten().contiguous()
    return w, raw, sw


def shard_views(w, sw, r):
    rows = N_SHARD
    per = sw.numel() // N_FULL
    return w.narrow(0, r * rows, rows), sw.narrow(0, r * rows * per, rows * per)


def make_emb(torch, M, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    return torch.randn((M, K), generator=g, device="cuda").to(torch.bfloat16)


def mm(torch, a, a_sf, w, s):
    return torch.ops.trtllm.flashinfer_mm_mxfp8(a, a_sf, w, s, torch.bfloat16)


def kv_call(torch, emb, w, s):
    a, a_sf = torch.ops.trtllm.mxfp8_quantize(emb, True)
    return mm(torch, a, a_sf, w, s)


def nbad(torch, x, ref):
    return int((x.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)).sum().item())


def tile_n_for(torch, n):
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    return 64 if (n + 63) // 64 <= sm else 128


def kernel_names(torch, fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.name not in names:
            names.append(e.name)
    return [n[:160] for n in names]


def timed_graph(torch, calls, replays=30, warm=3):
    """Capture ``calls`` (list of thunks) into one graph; return per-call us samples."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [c() for c in calls]
    KEEP.append(outs)
    for _ in range(warm):
        g.replay()
    ts = []
    for _ in range(replays):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / len(calls))
    del g
    return ts


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[min(n - 1, int(0.9 * n))]
    return {"median_us": round(med, 3), "p10_us": round(p10, 3), "p90_us": round(p90, 3),
            "noise_floor": round((p90 - p10) / med, 4)}


# --------------------------------------------------------------------------------------- check
def dequant_ref(torch, a, a_sf, M, w, raw):
    from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight
    wd = dequant_mxfp8_weight(w, raw, 32).float()
    rows = ((M + 127) // 128) * 128
    a_raw = torch.ops.trtllm.block_scale_interleave_reverse(
        a_sf.view(rows, SF_COLS)).reshape(rows, SF_COLS)[:M]
    ad = dequant_mxfp8_weight(a, a_raw.contiguous(), 32).float()
    return ad @ wd.t()


def mode_check():
    torch, fence = _setup()
    torch.manual_seed(0)
    res = {"fence": fence, "swizzle": {}, "graph": {}, "timing": {}, "kernels": {}}
    ok = True
    w, raw, sw = make_kv_weight(torch, 0)
    KEEP.extend([w, raw, sw])
    # (a) the swizzled scale of a 128-aligned row shard is a contiguous chunk of the full one
    per = sw.numel() // N_FULL
    assert per == SF_COLS, per
    for r in range(TP):
        ref_r = torch.ops.trtllm.block_scale_interleave(
            raw[r * N_SHARD:(r + 1) * N_SHARD].contiguous()).flatten()
        _, s_r = shard_views(w, sw, r)
        b = int((ref_r != s_r).sum().item())
        res["swizzle"][f"rank{r}"] = {"bytes": int(s_r.numel()), "bad": b,
                                      "offset": r * N_SHARD * per}
        ok = ok and b == 0 and ref_r.numel() == s_r.numel()
    log(f"[check] swizzle slice proof: {res['swizzle']}")
    # (b) graph-captured bitwise shard vs full columns
    replays = int(os.environ.get("REPLAYS", "30"))
    for M in MS:
        emb = make_emb(torch, M, M)
        emb0 = emb.clone()
        KEEP.extend([emb, emb0])
        a, a_sf = torch.ops.trtllm.mxfp8_quantize(emb, True)
        full_ref = mm(torch, a, a_sf, w, sw)
        torch.cuda.synchronize()
        ref32 = dequant_ref(torch, a, a_sf, M, w, raw)
        err = (full_ref.float() - ref32).abs()
        tol = 1.6e-2 + 1.6e-2 * ref32.abs()
        tol_bad = int((err > tol).sum().item())
        eager_shard_bad = 0
        for r in range(TP):
            wr, sr = shard_views(w, sw, r)
            o = mm(torch, a, a_sf, wr, sr)
            eager_shard_bad += nbad(torch, o, full_ref[:, r * N_SHARD:(r + 1) * N_SHARD])
        KEEP.extend([a, a_sf, full_ref])

        def body():
            qa, qsf = torch.ops.trtllm.mxfp8_quantize(emb, True)
            outs = [mm(torch, qa, qsf, w, sw)]
            for r in range(TP):
                wr, sr = shard_views(w, sw, r)
                outs.append(mm(torch, qa, qsf, wr, sr))
            return outs

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = body()
        KEEP.append(outs)
        bad_replays, max_bad, in_bad = 0, 0, 0
        for _ in range(replays):
            for o in outs:
                o.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            b = nbad(torch, outs[0], full_ref)
            for r in range(TP):
                b += nbad(torch, outs[1 + r], full_ref[:, r * N_SHARD:(r + 1) * N_SHARD])
            in_bad += int(not torch.equal(emb, emb0))
            bad_replays += int(b > 0)
            max_bad = max(max_bad, b)
        del g
        res["graph"][str(M)] = {"replays": replays, "bad_replays": bad_replays,
                                "max_bad_elems": max_bad, "input_changed": in_bad,
                                "eager_shard_vs_full_bad": eager_shard_bad,
                                "full_vs_fp32_dequant_out_of_tol": tol_bad,
                                "full_vs_fp32_max_abs": round(err.max().item(), 5),
                                "finite": bool(torch.isfinite(full_ref).all().item())}
        ok = ok and bad_replays == 0 and eager_shard_bad == 0 and tol_bad == 0 and in_bad == 0
        log(f"[check] M={M} graph {res['graph'][str(M)]}")
        res["kernels"][str(M)] = {
            "full": kernel_names(torch, lambda: mm(torch, a, a_sf, w, sw)),
            "full_tile_n": tile_n_for(torch, N_FULL),
            "shard": kernel_names(torch, lambda: mm(torch, a, a_sf, *shard_views(w, sw, 1))),
            "shard_tile_n": tile_n_for(torch, N_SHARD)}
        log(f"[check] M={M} kernels {res['kernels'][str(M)]}")
    # (c) per-call us at both shapes, cold weights (copies rotate beyond the 128 MB L2)
    copies = [(w, sw)]
    for i in range(2):
        w2, _, sw2 = make_kv_weight(torch, 10 + i)
        copies.append((w2, sw2))
    KEEP.append(copies)
    for M in MS:
        emb = make_emb(torch, M, M)
        a, a_sf = torch.ops.trtllm.mxfp8_quantize(emb, True)
        KEEP.extend([emb, a, a_sf])
        full_calls = [(lambda ww=ww, ss=ss: mm(torch, a, a_sf, ww, ss)) for ww, ss in copies]
        shard_calls = []
        for ww, ss in copies:
            for r in range(TP):
                wr, sr = shard_views(ww, ss, r)
                shard_calls.append(lambda wr=wr, sr=sr: mm(torch, a, a_sf, wr, sr))
        fq = [(lambda ww=ww, ss=ss: kv_call(torch, emb, ww, ss)) for ww, ss in copies]
        tf = stats(timed_graph(torch, full_calls))
        ts = stats(timed_graph(torch, shard_calls))
        tq = stats(timed_graph(torch, fq))
        res["timing"][str(M)] = {"full_gemm": tf, "shard_gemm": ts, "full_quant_plus_gemm": tq}
        log(f"[check] M={M} per-call full {tf['median_us']} us shard {ts['median_us']} us "
            f"(full+quant {tq['median_us']})")
    res["passed"] = bool(ok)
    print("RESULT " + json.dumps(res), flush=True)


# --------------------------------------------------------------------------------------- s2a
def mode_s2a():
    torch, fence = _setup()
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    dev = torch.device("cuda")
    torch.manual_seed(0)
    windows = int(os.environ.get("WINDOWS", "90"))
    hidden = 5120
    mhc = mHC(4, hidden, 20, dtype=torch.float32, eps=1e-6, norm_eps=1e-6, sinkhorn_eps=1e-6,
              post_mult_value=2.0).to(dev)
    with torch.no_grad():
        mhc.fn.normal_(0, 1e-2)
        mhc.base.normal_(0, 0.1)
        mhc.scale.copy_(torch.tensor([0.5, 0.7, 0.9]))
    nw = (1 + 0.1 * torch.randn(hidden, device=dev)).to(torch.bfloat16)
    aux = torch.cuda.Stream()  # created once, warmed eagerly before capture (pitfall 2195)
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    KEEP.extend([mhc, nw, aux, ev0, ev1, sA, sB])
    w, raw, sw = make_kv_weight(torch, 0)
    w0, sw0 = w.clone(), sw.clone()
    KEEP.extend([w, raw, sw, w0, sw0])
    res = {"fence": fence, "windows": windows, "arms": {}}
    for M in MS:
        x = torch.randn(M, hidden, device=dev).to(torch.bfloat16)
        rs = torch.randn(M, 4, hidden, device=dev).to(torch.bfloat16)
        post = (2 * torch.rand(M, 4, 1, device=dev)).float()
        comb = torch.rand(M, 4, 4, device=dev)
        comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
        pre = (torch.rand(M, 4, 1, device=dev) + 1e-3).float()
        emb = make_emb(torch, M, M)
        emb0 = emb.clone()
        KEEP.extend([x, rs, post, comb, pre, emb, emb0])
        with autotune(skip_dynamic_tuning_buckets=True):
            mhc.fused_hc_lagged(x, rs, post, comb, pre, norm_weight=nw, norm_eps=1e-6)
        full_ref = kv_call(torch, emb, w, sw)
        torch.cuda.synchronize()
        KEEP.append(full_ref)

        def mhc_body():
            outs = []
            for _ in range(48):
                split = mhc.fused_hc_lagged_split(x, rs, post, comb, pre, nw, 1e-6, aux, ev0, ev1)
                assert split is not None, "split path did not engage"
                o, join = split
                join.join()
                outs.append(o[0])
            return outs

        with with_multi_stream(True):
            # eager warm on stream B (aux stream warmed too), then capture on B
            with torch.cuda.stream(sB):
                mhc_body()
            torch.cuda.synchronize()
            gB = torch.cuda.CUDAGraph()
            with torch.cuda.stream(sB):
                with torch.cuda.graph(gB, stream=sB):
                    mouts = mhc_body()
        torch.cuda.synchronize()
        KEEP.extend([gB, mouts])
        with torch.cuda.stream(sB):
            gB.replay()
        torch.cuda.synchronize()
        mref = [o.clone() for o in mouts]
        KEEP.append(mref)

        arms = {
            "full": [(lambda: kv_call(torch, emb, w, sw), None)] * 3,
            "shard": [(lambda r=r: kv_call(torch, emb, *shard_views(w, sw, r)), r)
                      for _ in range(3) for r in range(TP)],
        }
        for arm, calls in arms.items():
            with torch.cuda.stream(sA):
                for c, _ in calls:
                    c()
            torch.cuda.synchronize()
            gA = torch.cuda.CUDAGraph()
            with torch.cuda.stream(sA):
                with torch.cuda.graph(gA, stream=sA):
                    kouts = [c() for c, _ in calls]
            torch.cuda.synchronize()
            KEEP.extend([gA, kouts])
            refs = [full_ref if r is None else full_ref[:, r * N_SHARD:(r + 1) * N_SHARD]
                    for _, r in calls]
            bad_w, max_bad, mhc_bad, in_bad, max_abs = 0, 0, 0, 0, 0.0
            overlaps = []
            for _ in range(windows):
                for o in kouts:
                    o.fill_(float("nan"))
                for o in mouts:
                    o.fill_(float("nan"))
                go = torch.cuda.Event()
                go.record()
                sA.wait_event(go)
                sB.wait_event(go)
                eA0, eA1, eB0, eB1 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                with torch.cuda.stream(sB):
                    eB0.record()
                    gB.replay()
                    eB1.record()
                with torch.cuda.stream(sA):
                    eA0.record()
                    gA.replay()
                    eA1.record()
                torch.cuda.synchronize()
                b = 0
                for o, ref in zip(kouts, refs):
                    nb = nbad(torch, o, ref)
                    if nb:
                        d = (o.float() - ref.float()).abs()
                        d = torch.nan_to_num(d, nan=float("inf"))
                        max_abs = max(max_abs, float(d.max().item()))
                    b += nb
                mb = sum(nbad(torch, o, r) for o, r in zip(mouts, mref))
                in_bad += int(not (torch.equal(emb, emb0) and torch.equal(w.view(torch.uint8),
                                                                         w0.view(torch.uint8))
                                   and torch.equal(sw, sw0)))
                bad_w += int(b > 0)
                max_bad = max(max_bad, b)
                mhc_bad += int(mb > 0)
                tA1 = eA0.elapsed_time(eA1)
                tB0 = eA0.elapsed_time(eB0)
                tB1 = eA0.elapsed_time(eB1)
                overlaps.append(max(0.0, min(tA1, tB1) - max(0.0, tB0)) * 1000.0)
            del gA
            overlaps.sort()
            row = {"bad_windows": bad_w, "windows": windows, "max_bad_elems": max_bad,
                   "max_abs_err": max_abs, "mhc_bad_windows": mhc_bad,
                   "input_checksum_bad": in_bad, "kv_calls_per_window": len(calls),
                   "overlap_us_min_med_max": [round(overlaps[0], 1),
                                              round(overlaps[len(overlaps) // 2], 1),
                                              round(overlaps[-1], 1)]}
            res["arms"][f"M{M}/{arm}"] = row
            log(f"[s2a fence={fence}] M={M} {arm}: {row}")
        del gB
    print("RESULT " + json.dumps(res), flush=True)


# --------------------------------------------------------------------------------------- cost
COST_SHAPES = [("wq_b", 8192, 1280), ("o_b", 5120, 2048), ("index_wq_b", 4096, 1280),
               ("kv_shard", N_SHARD, K)]


def mode_cost():
    torch, fence = _setup()
    import math
    dev = torch.device("cuda")
    res = {"fence": fence, "rows": {}, "sha": {}}
    for name, n, k in COST_SHAPES:
        g = torch.Generator(device=dev)
        g.manual_seed(7 + n + k)
        wb = (torch.randn((n, k), generator=g, device=dev) * 0.05).to(torch.bfloat16)
        wq, wsf = torch.ops.trtllm.mxfp8_quantize(wb, True)
        L = max(2, math.ceil(300e6 / (n * k * (1 + 1 / 32))))
        copies = [(wq, wsf)] + [(wq.clone(), wsf.clone()) for _ in range(L - 1)]
        KEEP.append(copies)
        for M in MS:
            g2 = torch.Generator(device=dev)
            g2.manual_seed(100 + M + n)
            x = torch.randn((M, k), generator=g2, device=dev).to(torch.bfloat16)
            a, a_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
            KEEP.extend([x, a, a_sf])
            o = mm(torch, a, a_sf, wq, wsf)
            torch.cuda.synchronize()
            res["sha"][f"{name}/M{M}"] = hashlib.sha256(
                o.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]
            calls = [(lambda ww=ww, ss=ss: mm(torch, a, a_sf, ww, ss)) for ww, ss in copies]
            st = stats(timed_graph(torch, calls, replays=30))
            res["rows"][f"{name}/M{M}"] = st
            log(f"[cost fence={fence}] {name} [{n},{k}] M={M} L={L}: {st}")
    print("RESULT " + json.dumps(res), flush=True)


# --------------------------------------------------------------------------------------- driver
def run_child(mode, fence, extra_env=None):
    env = dict(os.environ)
    env["FLASHINFER_B12X_CORE_FENCE"] = str(fence)
    env.update(extra_env or {})
    t0 = time.time()
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), mode], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace")
    result = None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        else:
            print(f"  [{mode} F{fence}] {line[:400]}", flush=True)
    p.wait()
    log(f"[driver] {mode} fence={fence} rc={p.returncode} {time.time()-t0:.0f}s")
    return result


def clocks():
    return sh(["nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.max.sm,temperature.gpu,"
               "clocks_throttle_reasons.active", "--format=csv,noheader"]).strip()


def driver():
    parts = os.environ.get("PARTS", "check,s2a,cost").split(",")
    out = {"clock_start": clocks()}
    log(f"[driver] clocks {out['clock_start']}")
    passed = True
    if "check" in parts:
        r = run_child("check", 1)
        out["check"] = r
        passed = passed and bool(r and r.get("passed"))
    if "s2a" in parts:
        out["s2a"] = {}
        for f in (0, 1):
            r = run_child("s2a", f)
            out["s2a"][f"fence{f}"] = r
            if f == 1:
                passed = passed and bool(r) and all(
                    v["bad_windows"] == 0 and v["input_checksum_bad"] == 0
                    for v in r["arms"].values())
    if "cost" in parts:
        runs = []
        for f in (1, 0, 0, 1):  # ABBA
            runs.append((f, run_child("cost", f)))
        agg = {}
        sha_ok = all(r is not None for _, r in runs)
        if sha_ok:
            for key in runs[0][1]["rows"]:
                on = [r["rows"][key]["median_us"] for f, r in runs if f == 1]
                off = [r["rows"][key]["median_us"] for f, r in runs if f == 0]
                shas = {r["sha"][key] for _, r in runs}
                sha_ok = sha_ok and len(shas) == 1
                agg[key] = {"fenced_us": round(sum(on) / 2, 3),
                            "unfenced_us": round(sum(off) / 2, 3),
                            "delta_us": round(sum(on) / 2 - sum(off) / 2, 3),
                            "runs_fenced": on, "runs_unfenced": off,
                            "noise_floor_max": max(r["rows"][key]["noise_floor"] for _, r in runs),
                            "fenced_bitwise_unfenced": len(shas) == 1}
                log(f"[cost] {key}: {agg[key]}")
        out["cost"] = agg
        out["cost_bitwise_fenced_vs_unfenced"] = sha_ok
        passed = passed and sha_ok
    out["clock_end"] = clocks()
    log(f"[driver] clocks {out['clock_end']}")
    out["passed"] = bool(passed)
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        {"check": mode_check, "s2a": mode_s2a, "cost": mode_cost}[sys.argv[1]]()
    else:
        driver()
