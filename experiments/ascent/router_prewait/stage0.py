"""Stage 0 reducer (CPU only) for idea 223 + idea 216 STEP -1.

(a) Job 1021 (post, main bf2855d6) decode_c1/decode_c4 traces, ranks 0-3: for every router_gemm_mma kernel, the
    kernel directly before it on the same stream (name, grid, dur) and the overlap window
    pred_end - router_start (us; > 0 means the router's CTAs were resident while the predecessor still ran).
(b) gate_forward_kernel per-call median/mean by template args, per rank, pre (job 5aa50e9a) vs post (feb81e31),
    c1 and c4; router_gemm kernels on every rank at c4 with template args and medians.
"""
import glob
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

ROOT = "/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots/0/jobs"
POST = ROOT + "/feb81e31fc004bbeb1da92660d738364/artifacts/profile/traces"
PRE = ROOT + "/5aa50e9a1d794996a1e7da554060edf6/artifacts/profile/traces"


def log(*a):
    print(*a, flush=True)


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    return ev


def kernels(ev):
    out = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if cat not in ("kernel", "cuda_kernel", "gpu_kernel") and "kernel" not in cat:
            continue
        args = e.get("args", {}) or {}
        out.append({"name": e.get("name", ""), "ts": float(e["ts"]), "dur": float(e.get("dur", 0.0)),
                    "stream": args.get("stream", e.get("tid")), "pid": e.get("pid"),
                    "grid": args.get("grid"), "block": args.get("block")})
    out.sort(key=lambda k: k["ts"])
    return out


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def short(n, w=110):
    return n if len(n) <= w else n[:w]


def tmpl(n):
    m = re.search(r"(gate_forward_kernel|router_gemm\w*)\s*<([^()]*)>", n)
    return (m.group(1) + "<" + m.group(2) + ">") if m else short(n, 80)


def files(base, case):
    fs = sorted(glob.glob(os.path.join(base, case, "*")))
    return [f for f in fs if f.endswith((".json", ".json.gz"))]


def rank_of(f):
    m = re.search(r"rank-(\d+)", f)
    return int(m.group(1)) if m else -1


def main():
    res = {"router_pred": {}, "gate_forward": {}, "router_by_rank": {}}
    for base, tag in ((POST, "post"), (PRE, "pre")):
        for case in ("decode_c1", "decode_c4"):
            fs = files(base, case)
            log(tag, case, "files:", [os.path.basename(f) for f in fs])
            for f in fs:
                r = rank_of(f)
                try:
                    ks = kernels(load(f))
                except Exception as e:  # noqa: BLE001
                    log("load failed", f, type(e).__name__, e)
                    continue
                log(tag, case, "rank", r, "kernels", len(ks))
                # gate_forward and router medians per template
                byname = defaultdict(list)
                for k in ks:
                    if "gate_forward" in k["name"] or "router_gemm" in k["name"]:
                        byname[tmpl(k["name"])].append(k["dur"])
                for n, ds in sorted(byname.items()):
                    rec = {"calls": len(ds), "median_us": round(statistics.median(ds), 3),
                           "mean_us": round(statistics.fmean(ds), 3), "p10": pct(ds, 0.1), "p90": pct(ds, 0.9)}
                    key = "gate_forward" if "gate_forward" in n else "router_by_rank"
                    res[key][f"{tag}|{case}|rank{r}|{n}"] = rec
                    log("  ", key, n, rec)
                if tag != "post":
                    continue
                # predecessor on the same stream
                by_stream = defaultdict(list)
                for k in ks:
                    by_stream[(k["pid"], k["stream"])].append(k)
                preds = Counter()
                win, pdur, rdur, gap_after_pred_start = [], [], [], []
                grids = Counter()
                ex = []
                for s, lst in by_stream.items():
                    for i, k in enumerate(lst):
                        if "router_gemm" not in k["name"] or i == 0:
                            continue
                        p = lst[i - 1]
                        pn = short(p["name"], 90)
                        preds[pn] += 1
                        grids[(pn, str(p["grid"]), str(p["block"]))] += 1
                        pend = p["ts"] + p["dur"]
                        win.append(pend - k["ts"])
                        pdur.append(p["dur"])
                        rdur.append(k["dur"])
                        gap_after_pred_start.append(k["ts"] - p["ts"])
                        if len(ex) < 3:
                            ex.append({"pred": pn, "pred_ts": p["ts"], "pred_dur": p["dur"], "router_ts": k["ts"],
                                       "router_dur": k["dur"], "next": short(lst[i + 1]["name"], 60)
                                       if i + 1 < len(lst) else None})
                if not win:
                    continue
                rec = {"n": len(win), "pred_names": preds.most_common(5),
                       "pred_grid_block": [(list(g), c) for g, c in grids.most_common(4)],
                       "window_pred_end_minus_router_start_us": {"p10": pct(win, 0.1), "p50": pct(win, 0.5),
                                                                  "p90": pct(win, 0.9)},
                       "router_start_minus_pred_start_us": {"p10": pct(gap_after_pred_start, 0.1),
                                                            "p50": pct(gap_after_pred_start, 0.5),
                                                            "p90": pct(gap_after_pred_start, 0.9)},
                       "pred_dur_us": {"p10": pct(pdur, 0.1), "p50": pct(pdur, 0.5), "p90": pct(pdur, 0.9)},
                       "router_dur_us": {"p10": pct(rdur, 0.1), "p50": pct(rdur, 0.5), "p90": pct(rdur, 0.9)},
                       "examples": ex}
                res["router_pred"][f"{case}|rank{r}"] = rec
                log("  router_pred", case, r, json.dumps(rec, default=str))
    print(json.dumps({"passed": True, "stage0": res}, default=str))


if __name__ == "__main__":
    main()
