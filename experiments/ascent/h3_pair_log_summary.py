"""Summarise a run_serving_pair's server logs (no model run, no GPU work).

Per arm (label=job_dir): per timed request, mean of 'Step k/49 | x s' forward times split into
dense/cached and BF16 phase (forward < bf16_forwards) / FP8 phase, plus Cache-DiT enable/refresh/
decision lines, Recompiling lines and VAE 'failed with OOM' lines per rank log.

Usage: python h3_pair_log_summary.py --cached 8,9,... --bf16-forwards 14 base=<artifacts dir> cand=<dir> ...
Each arm may give its own mask with label:mask=dir (mask string of 49 chars, 1 = computed).
"""

import argparse
import glob
import json
import os
import re
import statistics

STEP = re.compile(r"Step (\d+)/(\d+) \| ([\d.]+)s")
ENABLE = re.compile(r"(Cache-DiT: MiniMax-H3 \S* ?\d+ blocks, Fn=.*)$")
REFRESH = re.compile(r"(Cache-DiT refresh rank=\d+: .*)$")
DEC = re.compile(r"(Cache-DiT decisions rank=\d+: .*)$")
STATS = re.compile(r"(Cache-DiT stats: .*)$")
RECOMP = re.compile(r"Recompiling")
OOM = re.compile(r"failed with OOM", re.IGNORECASE)
TAYLOR = re.compile(r"[Tt]aylor[Ss]eer")


def stats(xs):
    return {"n": len(xs), "mean_s": round(statistics.fmean(xs), 4) if xs else None,
            "median_s": round(statistics.median(xs), 4) if xs else None}


def find_dir(d):
    if os.path.isdir(d):
        return d
    hits = glob.glob(f"/home/scratch.chengzhang/trtllm-dev-minimax-h3-2g-20261112-143634/runner/slots/*/jobs/{d}/artifacts")
    return hits[0] if hits else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-forwards", type=int, default=14)
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    out = {}
    for item in a.arms:
        label, rest = item.split("=", 1)
        mask, d = rest.split("=", 1)
        d = find_dir(d)
        print(f"[arm] {label} dir={d} exists={os.path.isdir(d)}", flush=True)
        cached = {i for i, c in enumerate(mask) if c == "0"}
        seg = os.path.join(d, "bench/timed/server_segment.log")
        reqs, cur = [], None
        decs, refreshes, stat_lines = [], [], []
        recomp_seg = 0
        if os.path.exists(seg):
            with open(seg, errors="replace") as fh:
                for ln in fh:
                    m = STEP.search(ln)
                    if m and int(m.group(2)) == 49:
                        k, t = int(m.group(1)), float(m.group(3))
                        if k == 1:
                            cur = []
                            reqs.append(cur)
                        if cur is not None:
                            cur.append((k - 1, t, m.group(3)))
                    for rx, acc in ((DEC, decs), (REFRESH, refreshes), (STATS, stat_lines)):
                        mm = rx.search(ln)
                        if mm:
                            acc.append(mm.group(1)[:400])
                    if RECOMP.search(ln):
                        recomp_seg += 1
        rows = []
        decimals = set()
        for r in reqs:
            for _, _, s in r:
                decimals.add(len(s.split(".")[1]) if "." in s else 0)
            g = lambda pred: [t for i, t, _ in r if pred(i)]
            rows.append({
                "forwards": len(r),
                "denoise_sum_s": round(sum(t for _, t, _ in r), 2),
                "dense_bf16": stats(g(lambda i: i not in cached and i < a.bf16_forwards)),
                "cached_bf16": stats(g(lambda i: i in cached and i < a.bf16_forwards)),
                "dense_fp8": stats(g(lambda i: i not in cached and i >= a.bf16_forwards)),
                "cached_fp8": stats(g(lambda i: i in cached and i >= a.bf16_forwards)),
                "cached_all": stats(g(lambda i: i in cached)),
                "step1_s": r[0][1] if r else None, "step2_s": r[1][1] if len(r) > 1 else None,
            })
        ranks = {}
        for name in ("logs/server-proxy.log", "logs/server-proxy-rank1.log"):
            p = os.path.join(d, name)
            if not os.path.exists(p):
                continue
            en, oom, rec, tay = [], [], [], 0
            with open(p, errors="replace") as fh:
                for ln in fh:
                    mm = ENABLE.search(ln)
                    if mm:
                        en.append(mm.group(1)[:300])
                    if OOM.search(ln):
                        oom.append(ln.strip()[:200])
                    if RECOMP.search(ln):
                        rec.append(ln.strip()[:200])
                    if TAYLOR.search(ln):
                        tay += 1
            ranks[name] = {"enable": en, "oom_lines": len(oom), "oom_sample": oom[:3],
                           "recompiling_lines": len(rec), "recompiling_sample": rec[:6], "taylorseer_mentions": tay}
        arm = {"requests": rows, "step_decimals": sorted(decimals), "decisions": decs,
               "refreshes": refreshes, "stats_lines": stat_lines[:6], "recompiling_in_timed_segment": recomp_seg,
               "ranks": ranks}
        print(f"[arm] {label}: {json.dumps(arm)}", flush=True)
        out[label] = arm
    print(json.dumps({"passed": True, "summary": out}), flush=True)


if __name__ == "__main__":
    main()
