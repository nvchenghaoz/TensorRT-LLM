"""Idea 127 Stage 2 gain evidence: real MiniMax-H3 TP2 block (compiled, chunked TP overlap with the
NCCL all-reduces co-running), main's FFN chain vs the fused SM120 gate_up+SwiGLU, arms INTERLEAVED.

Two identical blocks from experiments/ascent/tp_overlap/block_pair.py build() (checkpoint config,
same synthetic per-rank weights, M = 37,966, VANILLA = BF16 attention, 4 chunks, the model's
SM120 cublas pins on):
  block M ("main"):  ff.fuse_bf16_gate_up_swiglu_sm120 = False -> pinned cublas_mm gate_up ->
                     silu_and_mul (main's path, unchanged code);
  block F ("fused"): trtllm::gate_up_swiglu_bf16_sm120.
Every iteration runs main, fused, fused_aa, main_aa (ABBA), each fenced by barrier + synchronize,
CUDA-event timed. Then an untimed bitwise pass (same counts and order; every arm's output vs the
first main output on both ranks; inputs read-only; recompiles after warmup), and a profiled pass
(torch.profiler, CUDA only): per arm and rank the post-attention span (first to_out start -> last
all-reduce end), the down_3-end -> last-AR-start gap, per-chunk gate_up (+ silu_and_mul on main)
kernel times; min over ranks of the per-iteration span.
Refuses (wrong_pair) unless the two visible GPUs are a PIX pair at x16 without GPU 5.
usage: python block_fused_inter.py <checkpoint>     Last stdout line: JSON.
"""
import json
import os
import statistics
import sys
import tempfile
import time

import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tp_overlap"))
sys.path.insert(0, os.path.join(HERE, "..", "gemm_pin"))

WARMUP = 5
ITERS = int(os.environ.get("BLOCK_FUSED_ITERS", "40"))
PROF = int(os.environ.get("BLOCK_FUSED_PROF", "10"))
ORDER = ["main", "fused", "fused_aa", "main_aa"]
GAP_SLEEP_S = 0.03
GAP_SPLIT_US = 15000.0


def log(*a):
    print(*a, flush=True)


def stats(v):
    v = sorted(v)
    med = statistics.median(v)
    lo, hi = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1) + 0.5)]
    return {"median": med, "p10": lo, "p90": hi, "noise": (hi - lo) / abs(med) if med else None,
            "n": len(v)}


def _is_gemm(n):
    n = n.lower()
    return (("nvjet" in n or "gemm" in n or "cutlass" in n) and "flash" not in n and "fmha" not in n
            and "sdpa" not in n and "attn" not in n)


def analyze(path):
    with open(path) as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", ""))
                  for e in evs if e.get("cat", "") == "kernel")
    calls, cur, last_end = [], [], None
    for k in kern:
        if cur and k[0] - last_end > GAP_SPLIT_US:
            calls.append(cur)
            cur = []
        cur.append(k)
        last_end = k[1] if last_end is None else max(last_end, k[1])
    if cur:
        calls.append(cur)
    calls = [[k for k in c if not ("nccl" in k[2].lower() and "f32" in k[2].lower())] for c in calls]
    calls = [c for c in calls if c]
    out = []
    for c in calls:
        g = [k for k in c if _is_gemm(k[2]) and (k[1] - k[0]) > 200.0]
        nc = sorted(k for k in c if "nccl" in k[2].lower())
        silu = [k for k in c if "silu_and_mul" in k[2]]
        d = {"n_gemm_raw": len(g)}
        if len(g) == 14:  # AdaLN table GEMV precedes qkv
            g = g[1:]
        d["n_gemm"] = len(g)
        d["gemm_names"] = sorted({k[2][:90] for k in g})
        if len(g) == 13:
            dur = [(k[1] - k[0]) / 1000.0 for k in g]
            d["gate_up"] = dur[5::2]
            d["down"] = dur[6::2]
            d["silu"] = [(k[1] - k[0]) / 1000.0 for k in silu]
            d["gate_up_plus_silu"] = sum(d["gate_up"]) + sum(d["silu"])
            if nc:
                d["span_ms"] = (max(k[1] for k in nc) - g[1][0]) / 1000.0
                d["down3_to_lastAR_ms"] = (nc[-1][0] - g[12][1]) / 1000.0
        out.append(d)
    return out


def worker(rank, ckpt, q, port):
    import datetime
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank),
                            timeout=datetime.timedelta(seconds=900))
    log(f"rank {rank}: process group up")
    for k in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
              "accumulated_recompile_limit"):
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, max(64, getattr(torch._dynamo.config, k)))
    import block_pair as bp
    from tensorrt_llm._torch.modules import gate_up_swiglu_sm120 as G
    torch.manual_seed(0)
    block_m, (x, temb, idx, rot), cfg = bp.build(ckpt, rank)
    torch.manual_seed(0)
    block_f, _i2, _ = bp.build(ckpt, rank)
    del _i2
    block_m.ff.fuse_bf16_gate_up_swiglu_sm120 = False
    flags = {"main": bool(block_m.ff.fuse_bf16_gate_up_swiglu_sm120),
             "fused": bool(block_f.ff.fuse_bf16_gate_up_swiglu_sm120),
             "pin_gate_up": bool(getattr(block_f.ff.gate_up_proj, "use_custom_cublas_mm", False)),
             "chunks": [getattr(block_m, "_tp_overlap_chunks", 0),
                        getattr(block_f, "_tp_overlap_chunks", 0)],
             "chunk_residual_env": os.environ.get("TLLM_H3_TP_CHUNK_RESIDUAL")}
    same_w = all(torch.equal(a, b) for (_, a), (_, b) in
                 zip(sorted(block_m.named_parameters()), sorted(block_f.named_parameters())))
    log(f"rank {rank}: flags {flags}; identical weights {same_w}")
    pristine = [x.clone(), temb.clone(), idx.clone(), rot[0].clone(), rot[1].clone()]
    torch._dynamo.reset()
    fns = {"M": torch.compile(block_m, mode="default", dynamic=None),
           "F": torch.compile(block_f, mode="default", dynamic=None)}
    arm_fn = {"main": "M", "main_aa": "M", "fused": "F", "fused_aa": "F"}

    def call(arm):
        with torch.inference_mode():
            return fns[arm_fn[arm]](x, temb, idx, rot, None, None, valid_seq_len=None)

    from torch._dynamo.utils import counters
    counters.clear()
    res = {"rank": rank, "flags": flags, "identical_weights": same_w}
    for w in range(WARMUP):
        for arm in ORDER:
            call(arm)
        torch.cuda.synchronize()
    stats_warm = dict(counters["stats"])
    gb_warm = len(counters["graph_break"])
    calls_warm = G.FUSED_CALLS
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
    res["fused_calls_timed"] = G.FUSED_CALLS - calls_warm
    res["timing"] = {a: stats(v) for a, v in ts.items()}
    res["paired_fused_minus_main"] = stats(
        [(ts["fused"][i] + ts["fused_aa"][i]) / 2 - (ts["main"][i] + ts["main_aa"][i]) / 2
         for i in range(ITERS)])
    res["paired_main_aa_minus_main"] = stats([ts["main_aa"][i] - ts["main"][i] for i in range(ITERS)])
    res["paired_fused_aa_minus_fused"] = stats([ts["fused_aa"][i] - ts["fused"][i]
                                                for i in range(ITERS)])
    if rank == 0:
        log(f"timing: { {a: round(v['median'], 3) for a, v in res['timing'].items()} }; "
            f"paired fused-main {res['paired_fused_minus_main']}")
    ref, bad = None, []
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
                      "compiles": G.COMPILE_COUNT}
    if rank == 0:
        log(f"bitwise: {res['bitwise']}")
    try:
        from torch.profiler import ProfilerActivity, profile
        tmpd = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
        path = os.path.join(tmpd, f"fused_inter_rank{rank}.json")
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
        per = analyze(path)
        full = [d for d in per if d["n_gemm"] == 13]
        pr = {"n_full_calls": len(full), "n_calls": len(per)}
        if len(full) == PROF * len(ORDER):
            for j, arm in enumerate(ORDER):
                ds = full[j::len(ORDER)]
                pa = {"gemm_names": sorted({n for d in ds for n in d["gemm_names"]})}
                pa["gate_up_chunk_ms"] = [stats([d["gate_up"][c] for d in ds])["median"]
                                          for c in range(4)]
                pa["silu_chunk_ms"] = ([stats([d["silu"][c] for d in ds])["median"] for c in range(4)]
                                       if all(len(d["silu"]) == 4 for d in ds) else
                                       [len(d["silu"]) for d in ds][:3])
                pa["gate_up_plus_silu_ms"] = stats([d["gate_up_plus_silu"] for d in ds])["median"]
                pa["span_ms"] = stats([d["span_ms"] for d in ds if "span_ms" in d])
                pa["span_raw"] = [d.get("span_ms") for d in ds]
                pa["down3_to_lastAR_ms"] = stats([d["down3_to_lastAR_ms"] for d in ds
                                                  if "down3_to_lastAR_ms" in d])["median"]
                pr[arm] = pa
        else:
            pr["sample"] = per[:6]
        res["profile"] = pr
        if rank == 0:
            log(f"profile: {json.dumps({a: {k: v for k, v in pa.items() if k != 'span_raw'} for a, pa in pr.items() if isinstance(pa, dict) and 'span_ms' in pa}, default=str)[:3000]}")
    except Exception as ex:
        res["profile"] = {"error": repr(ex)}
        log(f"rank {rank} profile error: {ex!r}")
    q.put(res)
    dist.barrier()
    dist.destroy_process_group()


def main():
    import block_pin_inter as bpi
    ckpt = sys.argv[1]
    try:
        pl = bpi.pair_link()
    except RuntimeError as e:
        print(json.dumps({"passed": False, "error": str(e)}))
        return
    log(f"pair: {pl}")
    x8 = any(w[2] != "16" for w in pl["widths"]) or 5 in pl["gpus"]
    if pl["link"] != "PIX" or x8:
        print(json.dumps({"wrong_pair": True, "pair": pl}))
        return
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, ckpt, q, port)) for r in range(2)]
    for p in ps:
        p.start()
    outs = []
    t0 = time.time()
    while len(outs) < 2:
        try:
            outs.append(q.get(timeout=60))
        except Exception:
            log(f"[heartbeat] {time.time() - t0:.0f} s")
            if any(p.exitcode not in (None, 0) for p in ps):
                for p in ps:
                    if p.is_alive():
                        p.terminate()
                print(json.dumps({"passed": False, "error": "worker died",
                                  "exitcodes": [p.exitcode for p in ps]}))
                return
    for p in ps:
        p.join()
    outs.sort(key=lambda o: o["rank"])
    per_rank = {}
    for o in outs:
        t = o["timing"]
        pr = {"main_ms": (t["main"]["median"] + t["main_aa"]["median"]) / 2,
              "fused_ms": (t["fused"]["median"] + t["fused_aa"]["median"]) / 2,
              "paired_fused_minus_main": o["paired_fused_minus_main"],
              "paired_main_aa_minus_main": o["paired_main_aa_minus_main"],
              "paired_fused_aa_minus_fused": o["paired_fused_aa_minus_fused"],
              "fused_calls_timed": o["fused_calls_timed"], "flags": o["flags"]}
        prof = o.get("profile", {})
        if all(a in prof for a in ORDER):
            pr["profile"] = {a: {k: v for k, v in prof[a].items() if k != "span_raw"} for a in ORDER}
            pr["span_fused_minus_main_ms"] = (
                (prof["fused"]["span_ms"]["median"] + prof["fused_aa"]["span_ms"]["median"]) / 2
                - (prof["main"]["span_ms"]["median"] + prof["main_aa"]["span_ms"]["median"]) / 2)
            pr["span_aa_main_ms"] = prof["main_aa"]["span_ms"]["median"] - prof["main"]["span_ms"]["median"]
            pr["span_aa_fused_ms"] = (prof["fused_aa"]["span_ms"]["median"]
                                      - prof["fused"]["span_ms"]["median"])
        else:
            pr["profile_error"] = prof
        per_rank[o["rank"]] = pr
    span_min = None
    try:
        sp = {a: [min(s0, s1) for s0, s1 in zip(outs[0]["profile"][a]["span_raw"],
                                                outs[1]["profile"][a]["span_raw"])] for a in ORDER}
        span_min = {a: stats(v) for a, v in sp.items()}
        span_min["fused_minus_main_ms"] = ((span_min["fused"]["median"] + span_min["fused_aa"]["median"]) / 2
                                           - (span_min["main"]["median"] + span_min["main_aa"]["median"]) / 2)
        span_min["aa_main_ms"] = span_min["main_aa"]["median"] - span_min["main"]["median"]
        span_min["aa_fused_ms"] = span_min["fused_aa"]["median"] - span_min["fused"]["median"]
    except Exception as ex:
        span_min = {"error": repr(ex)}
    bit = all(o["bitwise"]["all_arms_all_iters_equal"] for o in outs)
    no_rc = not any(o["bitwise"]["recompiled_after_warmup"] for o in outs)
    fused_on = all(o["flags"]["fused"] and not o["flags"]["main"] for o in outs)
    print(json.dumps({"passed": bool(bit and no_rc and fused_on), "pair": pl, "bitwise_all": bit,
                      "no_recompile_after_warmup": no_rc,
                      "bitwise_detail": [o["bitwise"] for o in outs],
                      "per_rank": per_rank, "span_min_over_ranks": span_min}, default=str), flush=True)


if __name__ == "__main__":
    main()
