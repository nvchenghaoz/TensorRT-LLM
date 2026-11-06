"""W_crit for the idea-208 stand-in chain (plan addendum 66 (i), pitfall 2985).

CPU part (no GPU): from a main 4-rank decode profile (kineto traces, rank-{0..3}), per (step, MoE-AR call) in
decode_c1 / decode_c4 (/ decode_c2 if present), the MoE lamport all-reduce kernel duration on the rank that STARTS
the AR last (clocks aligned on AR end), p10/p50/p90 = W_last. MoE vs attention AR by the kernel that ran just before
the AR on the same stream (predecessor names reported), with a parity fallback when the counts are off.

GPU part: the stand-in producer's own overhead (kineto duration - W) at W = 2/3/5 us and grid 4*M x 160.
W_crit(M, q) = W_last_q(window of M) - overhead(M). M=1 <- c1, M=4 <- c4, M=2 <- c2 if present else the mean of
c1 and c4 (interpolated), M=8 <- c4 (extrapolated). Without traces: the plan fallback 3.0 / 3.5 / 4.0 / 4.0 us.
"""

import gzip
import json
import os
import re
import statistics
import time

P = "/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots"
ROOTS = [  # (label, artifacts root). Same main 8e291e60 first (1581 primary, 1622 contrast), then ba74f05c.
    ("job1581_profile1284_main8e291e60", f"{P}/0/jobs/ae975462541b4d89b17a1226cf5dba4a/artifacts"),
    ("job1622_profile1320_main8e291e60", f"{P}/0/jobs/42824ea9dcec4b8b8c42263740ec97c2/artifacts"),
    ("job1542_profile1251_mainba74f05c", f"{P}/0/jobs/f14624fcfa98439c959a02cabc268324/artifacts"),
    ("job1532_profile1242_mainba74f05c", f"{P}/4/jobs/1a166ee0f17d4b4aa0568204fdda7e77/artifacts"),
]
FALLBACK = {1: 3.0, 2: 3.5, 4: 4.0, 8: 4.0}
AR_RE = re.compile(r"allreduce_fusion_kernel|oneshot_lamport|twoshot|ncclDevKernel_AllReduce|customAllReduce", re.I)
MOE_PRED_RE = re.compile(r"moe|expert|finalize|swiglu|grouped_gemm|permute|routing|mxfp4|w4a|_v2_down|"
                         r"add|elementwise", re.I)


def log(*a):
    print(f"[wcrit {time.strftime('%H:%M:%S')}]", *a, flush=True)


def pct(v, q):
    v = sorted(v)
    return v[int(round(q * (len(v) - 1)))] if v else None


def load_kernels(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in evs if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"
            and e.get("dur") is not None]


def ar_calls(evs):
    by_stream = {}
    for e in evs:
        by_stream.setdefault((e.get("pid"), e.get("tid")), []).append(e)
    calls = []
    for lst in by_stream.values():
        lst.sort(key=lambda e: e["ts"])
        prev = None
        for e in lst:
            if AR_RE.search(e.get("name", "")):
                calls.append({"ts": float(e["ts"]), "dur": float(e["dur"]), "name": e["name"][:80],
                              "pred": prev["name"][:80] if prev else ""})
            else:
                prev = e
    calls.sort(key=lambda c: c["ts"])
    return calls


def window(files):
    per = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(per)
        t0 = time.time()
        per[r] = ar_calls(load_kernels(f))
        log(f"  {os.path.basename(f)}: {len(per[r])} AR calls ({time.time() - t0:.1f}s)")
    ranks = sorted(per)
    n = min(len(per[r]) for r in ranks)
    if len(ranks) < 4 or n == 0:
        return {"error": f"ranks {ranks} calls {[len(per[r]) for r in ranks]}"}
    end = {r: [c["ts"] + c["dur"] for c in per[r][:n]] for r in ranks}
    off = {r: statistics.median(a - b for a, b in zip(end[r], end[ranks[0]])) for r in ranks}
    lab_pred = ["moe" if MOE_PRED_RE.search(per[ranks[0]][i]["pred"]) else "attn" for i in range(n)]
    n_moe = lab_pred.count("moe")
    method = "predecessor"
    labels = lab_pred
    if not (0.4 * n <= n_moe <= 0.6 * n):
        # Parity fallback: pick the phase whose calls the predecessor rule calls MoE most often.
        ph = max((0, 1), key=lambda p: sum(lab_pred[i] == "moe" for i in range(p, n, 2)))
        labels = ["moe" if i % 2 == ph else "attn" for i in range(n)]
        method = f"parity(phase {ph})"
    preds = {}
    w_last, w_min, skew = [], [], []
    for i in range(n):
        p0 = per[ranks[0]][i]["pred"][:60]
        preds.setdefault(labels[i], {}).setdefault(p0, 0)
        preds[labels[i]][p0] += 1
        if labels[i] != "moe":
            continue
        starts = {r: per[r][i]["ts"] - off[r] for r in ranks}
        last = max(ranks, key=lambda r: starts[r])
        w_last.append(per[last][i]["dur"])
        w_min.append(min(per[r][i]["dur"] for r in ranks))
        skew.append(max(starts.values()) - min(starts.values()))
    q = lambda v: {"p10": pct(v, 0.1), "p50": pct(v, 0.5), "p90": pct(v, 0.9)}  # noqa: E731
    return {"ranks": ranks, "matched_calls": n, "moe_calls": len(w_last), "label_method": method,
            "clock_offsets_us": {r: round(off[r], 2) for r in ranks},
            "W_last_us": q(w_last), "min_rank_dur_us": q(w_min), "arrival_skew_us": q(skew),
            "top_predecessors": {k: sorted(v.items(), key=lambda kv: -kv[1])[:4] for k, v in preds.items()}}


def trace_part():
    """Returns (label, {window: result}) of the first root that has 4-rank decode_c1 and decode_c4 traces."""
    import glob

    for label, root in ROOTS:
        tr = os.path.join(root, "profile", "traces")
        if not os.path.isdir(tr):
            log(f"{label}: {tr} missing")
            continue
        wins = {}
        for w in ("decode_c1", "decode_c2", "decode_c4"):
            files = sorted(glob.glob(os.path.join(tr, w, "*rank-*.json*")))
            if not files:
                continue
            log(f"{label} {w}: {len(files)} files")
            try:
                wins[w] = window(files)
            except Exception as exc:  # noqa: BLE001
                wins[w] = {"error": repr(exc)[:300]}
            log(f"{label} {w}: {json.dumps(wins[w])[:900]}")
        if all("W_last_us" in wins.get(w, {}) for w in ("decode_c1", "decode_c4")):
            return label, wins
        log(f"{label}: unusable ({ {k: v.get('error') for k, v in wins.items()} })")
    return None, {}


def overhead_part(batches, ws=(2.0, 3.0, 5.0), n=60):
    """Stand-in producer kineto duration - W, per M (median over W of the per-W median)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    import pdl_ext

    ext = pdl_ext.ext()
    res = {}
    for M in batches:
        src = torch.randn(M, 5120, device="cuda").to(torch.bfloat16)
        dst = torch.empty_like(src)
        for w in ws:
            for _ in range(10):
                ext.producer(src, dst, int(w * 1000), 4 * M, 160, pdl_ext.empty(), False)
        torch.cuda.synchronize()
        per_w = {}
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for w in ws:
                for _ in range(n):
                    ext.producer(src, dst, int(w * 1000), 4 * M, 160, pdl_ext.empty(), False)
                torch.cuda.synchronize()
        evs = [e for e in prof.events() if "producer_kernel" in e.name and e.device_type.name == "CUDA"]
        evs.sort(key=lambda e: e.time_range.start)
        # Kernels launch in order of ws, n each.
        durs = [e.time_range.elapsed_us() for e in evs]
        for k, w in enumerate(ws):
            d = durs[k * n:(k + 1) * n]
            per_w[f"W{w:g}"] = round(statistics.median(d) - w, 3) if d else None
        vals = [v for v in per_w.values() if v is not None]
        res[M] = {"overhead_us": round(statistics.median(vals), 3) if vals else 0.4, "per_W": per_w,
                  "kernels_seen": len(durs)}
        log(f"stand-in overhead M={M}: {res[M]}")
    return res


def compute(batches):
    label, wins = trace_part()
    ov = overhead_part(batches)
    table = {}
    for M in batches:
        if not label:
            w50 = w10 = FALLBACK.get(M, 4.0)
            src = "fallback (plan 2987: branch 61 lamport 3.5/4.0 us minus stand-in ~0.5)"
            table[M] = {"W_crit_p10": w10, "W_crit_p50": w50, "source": src}
            continue
        c1, c4 = wins["decode_c1"]["W_last_us"], wins["decode_c4"]["W_last_us"]
        c2 = wins.get("decode_c2", {}).get("W_last_us")
        if M == 1:
            wl, src = c1, "decode_c1"
        elif M == 2:
            wl, src = (c2, "decode_c2") if c2 else ({k: (c1[k] + c4[k]) / 2 for k in c1}, "mean(c1,c4) interpolated")
        elif M == 4:
            wl, src = c4, "decode_c4"
        else:
            wl, src = c4, "decode_c4 (extrapolated to M=8)"
        o = ov[M]["overhead_us"]
        table[M] = {"W_last_p10": round(wl["p10"], 3), "W_last_p50": round(wl["p50"], 3),
                    "W_last_p90": round(wl["p90"], 3), "standin_overhead_us": o,
                    "W_crit_p10": round(max(1.0, wl["p10"] - o), 2), "W_crit_p50": round(max(1.0, wl["p50"] - o), 2),
                    "source": f"{label} {src}"}
    return {"trace_root": label, "windows": wins, "overhead": ov, "W_crit": table}
