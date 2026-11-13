"""Idea 68 acceptance 3: real MiniMax-H3 TP2 block (compiled, chunked TP overlap, NCCL co-run),
SM120 cublas_mm tile pin on vs off, arms INTERLEAVED per iteration in one 2-GPU process pair.

Two identical blocks are built with experiments/ascent/tp_overlap/block_pair.py's build()
(checkpoint config, same synthetic per-rank weights, M = 37,966, VANILLA attention, 4 chunks):
  block M: use_custom_cublas_mm False on qkv_proj / to_out / gate_up_proj / down_proj (F.linear,
           the path main runs),
  block P: the flags as the model sets them on SM120 (trtllm::cublas_mm + sm120_bf16_algo_list).
Each is compiled once (torch.compile mode=default). Every iteration runs the arms in ABBA order
main, pin, pin_aa, main_aa (main/main_aa = block M, pin/pin_aa = block P), each call fenced by a
barrier + synchronize and timed with CUDA events. A/A noise = main_aa vs main and pin_aa vs pin.
Then:
  - untimed bitwise pass, same warmup/iteration counts and order: every arm's output vs the first
    main output (torch.equal, both ranks), inputs checked read-only, no recompiles after warmup;
  - profiled pass (torch.profiler, CUDA only, PROF iterations of the same interleaving): per-chunk
    to_out / gate_up / down GEMM times and the span from the first to_out chunk start to the end
    of the last bf16 all-reduce kernel (method 1153), per arm and rank.
Refuses (wrong_pair) unless the two visible GPUs are a PIX pair at x16 without GPU 5.
usage: python block_pin_inter.py <checkpoint>     Last stdout line: JSON.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tp_overlap"))

WARMUP = 5
ITERS = int(os.environ.get("BLOCK_PIN_ITERS", "40"))
PROF = int(os.environ.get("BLOCK_PIN_PROF", "10"))
ORDER = ["main", "pin", "pin_aa", "main_aa"]
GAP_SLEEP_S = 0.03
GAP_SPLIT_US = 15000.0


def log(*a):
    print(*a, flush=True)


def pair_link():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,pcie.link.width.current,"
                        "pcie.link.width.max", "--format=csv,noheader"], capture_output=True, text=True).stdout
    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    log("nvidia-smi gpus:\n" + q)
    log("topo -m:\n" + topo)
    rows = [[c.strip() for c in l.split(",")] for l in q.strip().splitlines()]
    by_uuid = {r[1]: r for r in rows}
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    sel = []
    for v in vis[:2]:
        r = by_uuid.get(v) or next((x for x in rows if x[0] == v), None)
        sel.append(r)
    if len(sel) != 2 or any(r is None for r in sel):
        raise RuntimeError(f"parser error: cannot map CUDA_VISIBLE_DEVICES={vis} to nvidia-smi rows")
    i0, i1 = int(sel[0][0]), int(sel[1][0])
    link, topo_line = None, None
    for l in topo.splitlines():
        parts = l.split()
        if parts and parts[0] == f"GPU{i0}" and len(parts) > 1 + i1:
            link = parts[1 + i1]
            topo_line = l
            break
    if link is None:
        raise RuntimeError("parser error: topo row not found")
    widths = [(int(r[0]), r[2], r[3], r[4]) for r in sel]
    return {"gpus": [i0, i1], "link": link, "widths": widths, "uuids": [r[1] for r in sel],
            "topo_line": topo_line}


def clocks(uuids):
    q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout
    out = []
    for l in q.strip().splitlines():
        c = [x.strip() for x in l.split(",")]
        if not uuids or c[0] in uuids:
            out.append({"uuid": c[0], "sm_mhz": c[1], "max_sm_mhz": c[2], "temp_c": c[3], "power_w": c[4]})
    return out


def stats(v):
    v = sorted(v)
    med = statistics.median(v)
    lo, hi = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1) + 0.5)]
    return {"median": med, "p10": lo, "p90": hi, "noise": (hi - lo) / abs(med) if med else None, "n": len(v)}


def _is_gemm(n):
    n = n.lower()
    return ("nvjet" in n or "gemm" in n or "cutlass" in n) and "flash" not in n and "fmha" not in n \
        and "sdpa" not in n and "attn" not in n


def analyze(path, n_calls):
    with open(path) as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = []
    for e in evs:
        if e.get("cat", "") == "kernel":
            kern.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", "")))
    kern.sort()
    calls, cur, last_end = [], [], None
    for k in kern:
        if cur and k[0] - last_end > GAP_SPLIT_US:
            calls.append(cur)
            cur = []
        cur.append(k)
        last_end = k[1] if last_end is None else max(last_end, k[1])
    if cur:
        calls.append(cur)
    # drop the f32 barrier all-reduce kernels (dist.barrier before each call)
    calls = [[k for k in c if not ("nccl" in k[2].lower() and "f32" in k[2].lower())] for c in calls]
    calls = [c for c in calls if c]
    res = {"n_calls_found": len(calls), "n_calls_expected": n_calls, "per_call": []}
    for c in calls:
        g = [k for k in c if _is_gemm(k[2]) and (k[1] - k[0]) > 200.0]
        nc = [k for k in c if "nccl" in k[2].lower()]
        d = {"n_gemm_raw": len(g), "raw_names": [k[2][:60] for k in g]}
        if len(g) == 14:  # the AdaLN table GEMV (M = 1-2, 520 MB weight read) precedes qkv
            d["dropped_first"] = g[0][2][:90]
            g = g[1:]
        d["n_gemm"] = len(g)
        d["gemm_names"] = sorted({k[2][:90] for k in g})
        if len(g) == 13:
            dur = [(k[1] - k[0]) / 1000.0 for k in g]
            d["qkv"] = dur[0]
            d["to_out"] = dur[1:5]
            d["gate_up"] = dur[5::2]
            d["down"] = dur[6::2]
            d["gemm_sum"] = sum(dur)
            if nc:
                d["span_ms"] = (max(k[1] for k in nc) - g[1][0]) / 1000.0
        d["call_ms"] = (c[-1][1] - c[0][0]) / 1000.0
        res["per_call"].append(d)
    return res


def worker(rank, ckpt, q, uuids, port):
    import datetime
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    log(f"rank {rank}: worker start, port {port}")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank),
                            timeout=datetime.timedelta(seconds=600))
    log(f"rank {rank}: process group up")
    for k in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
              "accumulated_recompile_limit"):
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, max(64, getattr(torch._dynamo.config, k)))
    import block_pair as bp
    torch.manual_seed(0)
    block_m, (x, temb, idx, rot), cfg = bp.build(ckpt, rank)
    log(f"rank {rank}: block M built")
    torch.manual_seed(0)
    block_p, _inputs2, _ = bp.build(ckpt, rank)
    del _inputs2
    log(f"rank {rank}: block P built")

    def lins(b):
        return [getattr(b.attn, "qkv_proj", None), b.attn.to_out[0],
                getattr(b.ff, "gate_up_proj", None), b.ff.down_proj]

    model_flags = [bool(getattr(m, "use_custom_cublas_mm", False)) for m in lins(block_p)]
    same_w = all(torch.equal(a, b) for (_, a), (_, b) in
                 zip(sorted(block_m.named_parameters()), sorted(block_p.named_parameters())))
    for m in lins(block_m):
        if m is not None:
            m.use_custom_cublas_mm = False
    chunks = [getattr(block_m, "_tp_overlap_chunks", 0), getattr(block_p, "_tp_overlap_chunks", 0)]
    if rank == 0:
        log(f"model-set use_custom_cublas_mm (pin block) {model_flags}; main block "
            f"{[bool(getattr(m, 'use_custom_cublas_mm', False)) for m in lins(block_m)]}; "
            f"overlap chunks {chunks}; identical weights {same_w}")
    pristine = [x.clone(), temb.clone(), idx.clone(), rot[0].clone(), rot[1].clone()]

    torch._dynamo.reset()
    fns = {"M": torch.compile(block_m, mode="default", dynamic=None),
           "P": torch.compile(block_p, mode="default", dynamic=None)}
    arm_fn = {"main": "M", "main_aa": "M", "pin": "P", "pin_aa": "P"}

    def call(arm):
        with torch.inference_mode():
            return fns[arm_fn[arm]](x, temb, idx, rot, None, None, valid_seq_len=None)

    from torch._dynamo.utils import counters
    counters.clear()
    res = {"rank": rank, "model_flags": model_flags, "chunks": chunks, "identical_weights": same_w}

    # ---- timed, interleaved ABBA per iteration ----
    for w in range(WARMUP):
        for arm in ORDER:
            call(arm)
        torch.cuda.synchronize()
        log(f"rank {rank}: warmup {w + 1}/{WARMUP} done")
    stats_warm = dict(counters["stats"])
    gb_warm = len(counters["graph_break"])
    res["clocks_before"] = clocks(uuids) if rank == 0 else None
    ts = {a: [] for a in ORDER}
    t0 = time.time()
    for it in range(ITERS):
        for arm in ORDER:
            dist.barrier()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            call(arm)
            e.record()
            e.synchronize()
            ts[arm].append(s.elapsed_time(e))
        if rank == 0 and (it + 1) % 10 == 0:
            log(f"timed iter {it + 1}/{ITERS} ({time.time() - t0:.1f} s): "
                + ", ".join(f"{a} {ts[a][-1]:.2f}" for a in ORDER))
    res["clocks_after"] = clocks(uuids) if rank == 0 else None
    res["timing_raw"] = ts
    res["timing"] = {a: stats(v) for a, v in ts.items()}
    paired = [(ts["pin"][i] + ts["pin_aa"][i]) / 2 - (ts["main"][i] + ts["main_aa"][i]) / 2 for i in range(ITERS)]
    res["paired_pin_minus_main"] = stats(paired)
    res["paired_main_aa_minus_main"] = stats([ts["main_aa"][i] - ts["main"][i] for i in range(ITERS)])
    res["paired_pin_aa_minus_pin"] = stats([ts["pin_aa"][i] - ts["pin"][i] for i in range(ITERS)])
    if rank == 0:
        log(f"timing: { {a: round(v['median'], 3) for a, v in res['timing'].items()} }")
        log(f"paired pin-main {res['paired_pin_minus_main']}")

    # ---- untimed bitwise pass, same counts and order ----
    ref = None
    bad = []
    for it in range(WARMUP + ITERS):
        for arm in ORDER:
            y = call(arm)
            torch.cuda.synchronize()
            if ref is None:
                ref = y.clone()
            if not torch.equal(y, ref):
                bad.append(f"{arm}@{it}")
            for t, p in zip((x, temb, idx, rot[0], rot[1]), pristine):
                if not torch.equal(t, p):
                    bad.append(f"input_changed@{arm}{it}")
            del y
    stats_end = dict(counters["stats"])
    res["bitwise"] = {"all_arms_all_iters_equal": not bad, "bad": bad[:20],
                      "recompiled_after_warmup": stats_warm != stats_end,
                      "graph_breaks_warm": gb_warm, "graph_breaks_end": len(counters["graph_break"]),
                      "dynamo_stats": stats_end}
    if rank == 0:
        log(f"bitwise: {res['bitwise']}")

    # ---- profiled pass: per-chunk GEMMs and span ----
    try:
        from torch.profiler import ProfilerActivity, profile
        tmpd = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
        path = os.path.join(tmpd, f"pin_inter_rank{rank}.json")
        dist.barrier()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for it in range(PROF):
                for arm in ORDER:
                    dist.barrier()
                    torch.cuda.synchronize()
                    time.sleep(GAP_SLEEP_S)
                    call(arm)
                    torch.cuda.synchronize()
                    time.sleep(GAP_SLEEP_S)
        prof.export_chrome_trace(path)
        an = analyze(path, PROF * len(ORDER))
        per = an.pop("per_call")
        # barrier kernels removed; the profiled calls may include leading/trailing fragments, so
        # keep the calls that hold the block's 13 GEMMs and assign arms cyclically.
        full = [d for d in per if d["n_gemm"] == 13]
        an["n_full_calls"] = len(full)
        prof_res = {}
        if len(full) == PROF * len(ORDER):
            for j, arm in enumerate(ORDER):
                ds = full[j::len(ORDER)]
                pa = {"gemm_names": sorted({n for d in ds for n in d["gemm_names"]})}
                pa["qkv_ms"] = stats([d["qkv"] for d in ds])["median"]
                for key in ("to_out", "gate_up", "down"):
                    pa[f"{key}_chunk_ms"] = [stats([d[key][c] for d in ds])["median"] for c in range(4)]
                    pa[f"{key}_sum_ms"] = sum(pa[f"{key}_chunk_ms"])
                pa["gemm_sum_ms"] = stats([d["gemm_sum"] for d in ds])
                pa["span_ms"] = stats([d["span_ms"] for d in ds if "span_ms" in d])
                pa["span_raw"] = [d.get("span_ms") for d in ds]
                pa["call_ms"] = stats([d["call_ms"] for d in ds])
                prof_res[arm] = pa
        else:
            an["sample"] = per[:6]
        res["profile"] = {"meta": an, "arms": prof_res}
        if rank == 0:
            log(f"profile meta: {an}")
            for arm, pa in prof_res.items():
                log(f"profile {arm}: qkv {pa['qkv_ms']:.3f} to_out {pa['to_out_sum_ms']:.3f} "
                    f"gate_up {pa['gate_up_sum_ms']:.3f} down {pa['down_sum_ms']:.3f} "
                    f"span {pa['span_ms']['median']:.3f} names {pa['gemm_names']}")
    except Exception as ex:  # profiling is diagnostic; never lose the timing
        res["profile"] = {"error": repr(ex)}
        log(f"rank {rank} profile error: {ex!r}")
    q.put(res)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ckpt = sys.argv[1]
    try:
        pl = pair_link()
    except RuntimeError as e:
        log(f"FAIL LOUD: {e}")
        print(json.dumps({"passed": False, "error": str(e)}))
        return
    log(f"pair: {pl}")
    x8 = any(w[2] != "16" for w in pl["widths"]) or 5 in pl["gpus"]
    if pl["link"] != "PIX" or x8:
        print(json.dumps({"wrong_pair": True, "pair": pl}))
        return
    import socket
    import threading
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    done = threading.Event()

    def heartbeat():
        t0 = time.time()
        while not done.wait(60):
            log(f"[heartbeat] {time.time() - t0:.0f} s")

    threading.Thread(target=heartbeat, daemon=True).start()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, ckpt, q, pl["uuids"], port)) for r in range(2)]
    for p in ps:
        p.start()
    outs = []
    while len(outs) < 2:
        try:
            outs.append(q.get(timeout=30))
        except Exception:
            dead = [p.exitcode for p in ps if not p.is_alive()]
            if any(c not in (None, 0) for c in dead):
                done.set()
                log(f"FAIL LOUD: worker exit codes {[p.exitcode for p in ps]}")
                for p in ps:
                    if p.is_alive():
                        p.terminate()
                print(json.dumps({"passed": False, "error": "worker died",
                                  "exitcodes": [p.exitcode for p in ps]}))
                return
    done.set()
    for p in ps:
        p.join()
    outs.sort(key=lambda o: o["rank"])
    per_rank = {}
    for o in outs:
        t = o["timing"]
        main_ms = (t["main"]["median"] + t["main_aa"]["median"]) / 2
        pin_ms = (t["pin"]["median"] + t["pin_aa"]["median"]) / 2
        pr = {"timing": t, "main_ms": main_ms, "pin_ms": pin_ms, "pin_minus_main_ms": pin_ms - main_ms,
              "aa_main_ms": t["main_aa"]["median"] - t["main"]["median"],
              "aa_pin_ms": t["pin_aa"]["median"] - t["pin"]["median"],
              "paired_pin_minus_main": o["paired_pin_minus_main"],
              "paired_main_aa_minus_main": o["paired_main_aa_minus_main"],
              "paired_pin_aa_minus_pin": o["paired_pin_aa_minus_pin"]}
        prof = o.get("profile", {}).get("arms", {})
        if prof:
            pr["profile"] = {a: {k: v for k, v in pa.items() if k != "span_raw"} for a, pa in prof.items()}
            pr["span_pin_minus_main_ms"] = ((prof["pin"]["span_ms"]["median"] + prof["pin_aa"]["span_ms"]["median"]) / 2
                                            - (prof["main"]["span_ms"]["median"] + prof["main_aa"]["span_ms"]["median"]) / 2)
            pr["span_aa_main_ms"] = prof["main_aa"]["span_ms"]["median"] - prof["main"]["span_ms"]["median"]
            pr["span_aa_pin_ms"] = prof["pin_aa"]["span_ms"]["median"] - prof["pin"]["span_ms"]["median"]
        else:
            pr["profile_error"] = o.get("profile")
        per_rank[o["rank"]] = pr
    # min over ranks of the per-iteration span (method 1153)
    span_min = None
    try:
        sp = {a: [min(s0, s1) for s0, s1 in zip(outs[0]["profile"]["arms"][a]["span_raw"],
                                                outs[1]["profile"]["arms"][a]["span_raw"])] for a in ORDER}
        span_min = {a: stats(v) for a, v in sp.items()}
        span_min["pin_minus_main_ms"] = ((span_min["pin"]["median"] + span_min["pin_aa"]["median"]) / 2
                                         - (span_min["main"]["median"] + span_min["main_aa"]["median"]) / 2)
        span_min["aa_main_ms"] = span_min["main_aa"]["median"] - span_min["main"]["median"]
        span_min["aa_pin_ms"] = span_min["pin_aa"]["median"] - span_min["pin"]["median"]
    except Exception as ex:
        span_min = {"error": repr(ex)}
    bit = all(o["bitwise"]["all_arms_all_iters_equal"] for o in outs)
    no_rc = not any(o["bitwise"]["recompiled_after_warmup"] for o in outs)
    pin_on = any(outs[0]["model_flags"])
    r0 = per_rank[0]
    win_vs_aa = {r: (-p["pin_minus_main_ms"] > max(abs(p["aa_main_ms"]), abs(p["aa_pin_ms"])))
                 for r, p in per_rank.items()}
    summary = {"pair": pl, "clocks_before": outs[0]["clocks_before"], "clocks_after": outs[0]["clocks_after"],
               "model_flags": outs[0]["model_flags"], "chunks": outs[0]["chunks"],
               "identical_weights": [o["identical_weights"] for o in outs],
               "per_rank": per_rank, "span_min_over_ranks": span_min,
               "pin_beats_main_beyond_aa": win_vs_aa,
               "bitwise_all_arms_both_ranks": bit, "no_recompile_after_warmup": no_rc,
               "bitwise_detail": [o["bitwise"] for o in outs],
               "batch": {"1": {"kernel_ms": r0["pin_ms"],
                               "noise_floor": max(r0["timing"][a]["noise"] for a in ORDER)}},
               "output": {"class": "exact"},
               "passed": bool(bit and no_rc and pin_on and all(o["identical_weights"] for o in outs))}
    for r, p in per_rank.items():
        log(f"rank {r}: main {p['main_ms']:.3f} pin {p['pin_ms']:.3f} ms/block, pin-main "
            f"{p['pin_minus_main_ms']:+.3f} ms, A/A main {p['aa_main_ms']:+.3f} pin {p['aa_pin_ms']:+.3f}; "
            f"paired median {p['paired_pin_minus_main']['median']:+.3f} ms; span pin-main "
            f"{p.get('span_pin_minus_main_ms')}")
    log(f"span min over ranks: {span_min}")
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
