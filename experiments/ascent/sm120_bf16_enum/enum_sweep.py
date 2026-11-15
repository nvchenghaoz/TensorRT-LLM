"""Stage A step 2: exhaustive cuBLASLt config sweep for the Gemma-4-31B BF16 attention projections on SM120.

For every (shape, M) with M in the BCG buckets (300/600/900/1000/1200/1500/1800/2000):
  main     = what this tree dispatches (trtllm::cublas_mm with the SM120 LUT where routed, else F.linear);
             main's cuBLASLt config = LUT entry, or the heuristic top-1 (checked bitwise against F.linear).
  pool     = every AlgoCheck-valid (algo, tile, stages, custom) from AlgoGetIds x CAP_TILE_IDS x CAP_STAGES_IDS x
             [0..CAP_CUSTOM_OPTION_MAX], cluster AUTO, swizzle 0, with main's split-K count and reduction scheme
             (qkv: no split; o_proj@300: splitK(2) r2), plus the heuristic top-32 list. Geometry prune: tile area
             >= 128x64, padding efficiency >= 0.70, >= 94 CTAs.
  screen   = cold-L2 (4 rotated weight copies) CUDA-event time of 4 calls after 1 warmup, every pool config.
  refine   = top 16 re-timed (warm 3 / 12 timed, median); then bitwise gate in time order: full-shape raw int16 compare
             against main into a 0xA5-prefilled output, 2 weight copies, randn and heavy-tailed (randn*2^U(-8,8) per
             row) inputs.
  cluster  = best 3 bitwise configs re-tried at every AlgoCheck-valid cluster id; bitwise-gated again.
  final    = main and the best bitwise config interleaved (main right before candidate), 5 rounds x (warm 5, 20
             timed) per-call events: median, p10, p90; SM clock sampled by NVML during the final loop;
             FLOP / (clk x SM) and CTA count reported.
Also: which tiles the sweep3 heuristic pool (union of top-32 lists over sweep3's M grid) never contained.

usage: python enum_sweep.py <checkpoint> [shape,...] [M,...]
"""
import json
import math
import os
import re
import sys
import threading
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltx import LtX, kernel_names, main_dispatch, read_shapes  # noqa: E402

BUCKETS = [300, 600, 900, 1000, 1200, 1500, 1800, 2000]
NCOPY = 4
NSM = 188
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Clock:
    """NVML SM-clock sampler (MHz) in a background thread."""

    def __init__(self):
        self.samples, self._stop, self.h, self.nv = [], None, None, None
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = torch.cuda.current_device()
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            if vis and vis.startswith("GPU-"):
                self.h = pynvml.nvmlDeviceGetHandleByUUID(vis.split(",")[idx])
            else:
                self.h = pynvml.nvmlDeviceGetHandleByIndex(int(vis.split(",")[idx]) if vis else idx)
            self.nv = pynvml
        except Exception as e:  # noqa: BLE001
            log("NVML unavailable:", e)

    def start(self):
        self.samples = []
        if self.h is None:
            return
        self._stop = threading.Event()

        def run():
            while not self._stop.is_set():
                try:
                    self.samples.append((self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM),
                                         self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.003)

        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def stop(self):
        if self._stop is None:
            return None
        self._stop.set()
        self.t.join()
        if not self.samples:
            return None
        c = sorted(s[0] for s in self.samples)
        p = sorted(s[1] for s in self.samples)
        return dict(n=len(c), sm_mhz_med=c[len(c) // 2], sm_mhz_min=c[0], sm_mhz_max=c[-1], power_w_med=p[len(p) // 2])


def tile_dims(lt, t):
    m = re.match(r"^(\d+)x(\d+)", lt.tiles.get(t, ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def ctas(lt, cfg, N, M):
    d = tile_dims(lt, cfg[1])
    if d is None:
        return None
    return math.ceil(N / d[0]) * math.ceil(M / d[1]) * max(cfg[3], 1)


def geometry_ok(lt, t, N, M):
    d = tile_dims(lt, t)
    if d is None:
        return True
    tn, tm = d
    if tn * tm < 128 * 64:
        return False
    nt = math.ceil(N / tn) * math.ceil(M / tm)
    if nt < 94:
        return False
    eff = (N * M) / (math.ceil(N / tn) * tn * math.ceil(M / tm) * tm)
    return eff >= 0.70


def stats(ts):
    t = sorted(ts)
    n = len(t)
    return dict(med=t[n // 2], p10=t[int(0.1 * n)], p90=t[min(n - 1, int(0.9 * n))], n=n)


def main():
    ckpt = sys.argv[1]
    shapes, info = read_shapes(ckpt)
    want = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else ["sl_qkv", "gl_qkv_dedup", "sl_o", "gl_o"]
    Ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else BUCKETS
    log("config", json.dumps(info), "shapes", json.dumps(shapes), "want", want, "M", Ms)
    op_main, routed, mg = main_dispatch()
    lt = LtX()
    clock = Clock()
    log("cublasLt", lt.path, lt.version, torch.cuda.get_device_name(), "slot env",
        os.environ.get("CUDA_VISIBLE_DEVICES"))
    ids = lt.algo_ids()
    caps = {a: lt.caps_of(a) for a in ids}
    # LUT entries (main's config where routed)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm120_bf16_lut"))
    from check import lookup, parse_table
    wt = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    tables = parse_table(os.path.join(wt, "cpp/tensorrt_llm/thop/cublasScaledMMLut.h"))
    # sweep3's heuristic pool (union of top-32 lists over its grid), for the "never in the pool" report
    s3 = set()
    for isl in (300, 1000, 2000, 4000, 8000):
        k = 1
        while isl * k <= 8192:
            s3.add(isl * k)
            k += 1
    grid = sorted(set(list(range(1280, 8193, 64)) + list(s3)))
    pool3 = set()
    for sname, (N, K) in shapes.items():
        for M in grid:
            for (c8, inner), ws, wv, _ in lt.heuristics(M, N, K, 32):
                if c8[0] in (66, 67) and c8[4] != 1:
                    pool3.add(c8)
    pool3_tiles = {c[1] for c in pool3}
    log(f"sweep3 heuristic pool: {len(pool3)} configs, {len(pool3_tiles)} tiles: "
        f"{sorted(lt.tile_name(t) for t in pool3_tiles)}")
    # pre-heat
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t_end = time.time() + 20
    while time.time() < t_end:
        for _ in range(20):
            torch.matmul(a, a)
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    log("pre-heat done")
    results = {}
    for sname in want:
        N, K = shapes[sname]
        torch.manual_seed(0)
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NCOPY)]
        xr = torch.randn(max(Ms), K, device="cuda", dtype=torch.bfloat16)
        g = torch.Generator(device="cuda").manual_seed(7)
        xh = (torch.randn(max(Ms), K, device="cuda", dtype=torch.float32, generator=g)
              * torch.exp2(torch.empty(max(Ms), 1, device="cuda").uniform_(-8, 8, generator=g))).to(torch.bfloat16)
        obuf = torch.empty(max(Ms), N, device="cuda", dtype=torch.bfloat16)
        for M in Ms:
            x = xr[:M]
            out = obuf[:M]
            is_routed = routed(N, K, M)
            lut_cfg = lookup(tables.get((K, N), []), M) if is_routed else None
            heur = lt.heuristics(M, N, K, 32)
            top1 = heur[0][0][0] if heur else None
            main_cfg = tuple(lut_cfg) if lut_cfg is not None else top1
            # confirm main's config reproduces main's op bitwise
            ref = op_main(x, Ws[0])
            lt.matmul(x, Ws[0], out, lt.make_algo(main_cfg))
            main_cfg_bitwise = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            main_kn = kernel_names(lambda: op_main(x, Ws[0]))
            sk, red = main_cfg[3], main_cfg[4]
            log(f"== {sname} N={N} K={K} M={M}: routed={is_routed} main cfg {list(main_cfg)} {lt.name(main_cfg)} "
                f"(reproduces main bitwise: {main_cfg_bitwise}) kernels {main_kn}")
            # ---- pool
            t1 = time.time()
            pool = {}
            n_valid_all = 0
            for aid in ids:
                c = caps.get(aid)
                if not c:
                    continue
                if sk > 1 and not c.get("SPLITK_SUPPORT"):
                    continue
                if sk > 1 and not ((c.get("REDUCTION_SCHEME_MASK") or 0) & red):
                    continue
                tiles = c.get("TILE_IDS") or [0]
                stg = c.get("STAGES_IDS") or [0]
                for t in tiles:
                    geo = geometry_ok(lt, t, N, M)
                    for s in stg:
                        for cu in range((c.get("CUSTOM_OPTION_MAX") or 0) + 1):
                            cfg = (aid, t, s, sk, red if sk > 1 else 0, 0, cu, 0)
                            al = lt.make_algo_raw(cfg)
                            if al is None:
                                continue
                            ok, ws, wv = lt.check_ws(M, N, K, al)
                            if not ok:
                                continue
                            n_valid_all += 1
                            if geo:
                                pool[cfg] = al
            for (c8, inner), ws, wv, _ in heur:
                if c8[3] == sk and c8[4] == (red if sk > 1 else 0) and c8 not in pool:
                    al = lt.make_algo_raw(c8)
                    if al is not None and lt.check_ws(M, N, K, al)[0]:
                        pool[c8] = al
            log(f"   pool: {n_valid_all} AlgoCheck-valid (sk={sk}), {len(pool)} after geometry prune, "
                f"{time.time() - t1:.1f}s")
            # ---- screen
            t1 = time.time()

            def main_fn(i):
                return op_main(x, Ws[i % NCOPY])

            def ev_time(fn, warm, iters):
                for i in range(warm):
                    fn(i)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for i in range(iters):
                    fn(warm + i)
                e.record()
                e.synchronize()
                return s.elapsed_time(e) / iters

            main_screen = [ev_time(main_fn, 2, 8)]
            scr = {}
            for i, (cfg, al) in enumerate(pool.items()):
                t = ev_time(lambda j: lt.matmul(x, Ws[j % NCOPY], out, al), 1, 4)
                scr[cfg] = t
                if i % 400 == 399:
                    main_screen.append(ev_time(main_fn, 2, 8))
                    torch.cuda.synchronize()
            main_screen.append(ev_time(main_fn, 2, 8))
            ms = sorted(main_screen)[len(main_screen) // 2]
            order = sorted(scr, key=scr.get)
            log(f"   screen {len(scr)} configs in {time.time() - t1:.1f}s; main {ms * 1e3:.1f} us "
                f"(drift {min(main_screen) * 1e3:.1f}-{max(main_screen) * 1e3:.1f}); best 8: " + " | ".join(
                    f"{lt.name(c)} {scr[c] * 1e3:.1f}" for c in order[:8]))
            # ---- refine top 16
            def ev_med(fn, warm, iters):
                for i in range(warm):
                    fn(i)
                evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(iters)]
                for i, (s, e) in enumerate(evs):
                    s.record()
                    fn(warm + i)
                    e.record()
                torch.cuda.synchronize()
                return [s.elapsed_time(e) for s, e in evs]

            ref_t = stats(ev_med(main_fn, 3, 12))["med"]
            refined = {}
            for cfg in order[:16]:
                al = pool[cfg]
                refined[cfg] = stats(ev_med(lambda j: lt.matmul(x, Ws[j % NCOPY], out, al), 3, 12))["med"]
            ref_t = min(ref_t, stats(ev_med(main_fn, 3, 12))["med"])

            # ---- bitwise gate
            refs = {}

            def bitwise(cfg, al):
                for j, xx in ((0, xr), (1, xr), (0, xh)):
                    key = (j, xx is xh)
                    if key not in refs:
                        refs[key] = op_main(xx[:M], Ws[j]).clone()
                    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                    o.view(torch.int16).fill_(-23131)  # 0xA5A5
                    lt.matmul(xx[:M], Ws[j], o, al)
                    mm = (o.view(torch.int16) != refs[key].view(torch.int16))
                    nmm = int(mm.sum().item())
                    if nmm:
                        idx = mm.nonzero()[0].tolist()
                        return False, dict(mismatches=nmm, weights=j, heavy=xx is xh, row=idx[0], col=idx[1],
                                           cand=float(o[idx[0], idx[1]]), main=float(refs[key][idx[0], idx[1]]))
                return True, None

            gated = []
            rejects = []
            for cfg in sorted(refined, key=refined.get):
                ok, why = bitwise(cfg, pool[cfg])
                if ok:
                    gated.append(cfg)
                else:
                    rejects.append((cfg, why))
            log(f"   refine main {ref_t * 1e3:.1f} us; " + " | ".join(
                f"{lt.name(c)} {refined[c] * 1e3:.1f}{'' if c in gated else ' NOTBITWISE'}"
                for c in sorted(refined, key=refined.get)))
            for cfg, why in rejects:
                log(f"   not bitwise: {lt.name(cfg)} first mismatch {json.dumps(why)}")
            # ---- cluster variants of the best 3 bitwise
            cl_tried = 0
            for cfg in list(gated[:3]):
                for cga in sorted(lt.clusters):
                    if cga == cfg[7]:
                        continue
                    c2 = cfg[:7] + (cga,)
                    al = lt.make_algo_raw(c2)
                    if al is None or not lt.check_ws(M, N, K, al)[0]:
                        continue
                    cl_tried += 1
                    pool[c2] = al
                    t = stats(ev_med(lambda j: lt.matmul(x, Ws[j % NCOPY], out, al), 3, 12))["med"]
                    if t < refined[cfg] * 0.99:
                        ok, why = bitwise(c2, al)
                        refined[c2] = t
                        if ok:
                            gated.append(c2)
                        log(f"   cluster {lt.name(c2)} {t * 1e3:.1f} us bitwise={ok}")
            gated.sort(key=lambda c: refined[c])
            best = gated[0] if gated else None
            rec = dict(shape=sname, N=N, K=K, M=M, routed=is_routed, main_cfg=list(main_cfg),
                       main_name=lt.name(main_cfg), main_cfg_bitwise=main_cfg_bitwise, main_kernels=main_kn,
                       n_valid=n_valid_all, n_pool=len(pool), n_screened=len(scr), cluster_tried=cl_tried,
                       screen_top8=[[lt.name(c), round(scr[c] * 1e3, 1)] for c in order[:8]],
                       n_rejected_nonbitwise=len(rejects), main_split_k=sk)
            if best is not None:
                al = pool[best]
                tm_, tc_ = [], []
                clock.start()
                for r in range(5):
                    tm_ += ev_med(main_fn, 5, 20)
                    tc_ += ev_med(lambda j: lt.matmul(x, Ws[j % NCOPY], out, al), 5, 20)
                ck = clock.stop()
                ok_final, why = bitwise(best, al)
                sm_, sc_ = stats(tm_), stats(tc_)
                flops = 2.0 * M * N * K
                mhz = ck["sm_mhz_med"] if ck else None
                rate = (lambda t: flops / (t * 1e-3) / (mhz * 1e6) / NSM) if mhz else (lambda t: None)
                rec.update(best_cfg=list(best), best_name=lt.name(best), best_kernels=kernel_names(
                    lambda: lt.matmul(x, Ws[0], out, al)), best_in_sweep3_pool=best[:8] in pool3,
                    best_tile_in_sweep3_pool=best[1] in pool3_tiles, best_ctas=ctas(lt, best, N, M),
                    main_ctas=ctas(lt, main_cfg, N, M), best_split_k=best[3], bitwise_final=ok_final,
                    main_us=dict((k, round(v * 1e3, 2)) for k, v in sm_.items() if k != "n"),
                    best_us=dict((k, round(v * 1e3, 2)) for k, v in sc_.items() if k != "n"),
                    ratio=round(sc_["med"] / sm_["med"], 4), clock=ck,
                    main_flop_per_clk_sm=rate(sm_["med"]) and round(rate(sm_["med"]), 1),
                    best_flop_per_clk_sm=rate(sc_["med"]) and round(rate(sc_["med"]), 1))
                log(f"   FINAL {sname} M={M}: main {lt.name(main_cfg)} {sm_['med'] * 1e3:.1f} us "
                    f"[{sm_['p10'] * 1e3:.1f},{sm_['p90'] * 1e3:.1f}] vs best {lt.name(best)} "
                    f"{sc_['med'] * 1e3:.1f} us [{sc_['p10'] * 1e3:.1f},{sc_['p90'] * 1e3:.1f}] "
                    f"ratio {rec['ratio']} bitwise {ok_final} ctas {rec['main_ctas']}->{rec['best_ctas']} "
                    f"clock {ck} FLOP/clk/SM {rec['main_flop_per_clk_sm']}->{rec['best_flop_per_clk_sm']} "
                    f"in sweep3 pool: cfg {rec['best_in_sweep3_pool']} tile {rec['best_tile_in_sweep3_pool']}")
            else:
                log(f"   FINAL {sname} M={M}: no bitwise config among the refined top")
            results[f"{sname}_M{M}"] = rec
            print("@@REC " + json.dumps(rec), flush=True)
        del Ws, xr, xh, obuf
        refs = None
        torch.cuda.empty_cache()
    print(json.dumps({"done": True, "results": {k: {kk: v.get(kk) for kk in (
        "main_name", "best_name", "ratio", "bitwise_final", "main_us", "best_us")} for k, v in results.items()}}))


if __name__ == "__main__":
    main()
