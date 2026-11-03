"""In-server decode all-reduce trace, all 4 ranks (idea 9, deliverable of plan 357).

Runs the frozen benchmark's profile adapter (profile_run.py: prime, warm up, decode_c1 and decode_c4 windows of
40 steps, kineto traces from every rank) with the harness config given as argv[1] (a JSON string or a path to one),
then reduces the per-rank decode traces:

  * all-reduce kernels (ncclDevKernel_AllReduce*, allreduce_fusion_kernel_oneshot_lamport/twoshot) in launch order
    per rank, matched by index across the 4 ranks;
  * per call: transfer = min duration over ranks, wait_r = duration_r - transfer;
  * attn vs MoE AR: by the kernel that ran just before the AR on the same stream (MoE finalize / expert kernels
    -> MoE AR, otherwise the o_b_proj GEMM -> attn AR); the predecessor names are reported so labels can be audited;
  * per rank mean us per call, transfer, wait, and per-step totals (ARs per step = calls / steps).
Usage: python ar_trace_job.py '<harness json or path>' [only_analyze_dir]
Last stdout line: JSON summary.
"""
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AR_RE = re.compile(r"ncclDevKernel_AllReduce|ncclKernel_AllReduce|allreduce_fusion_kernel|AllReduce.*lamport|"
                   r"oneshot_lamport|twoshot_sync|customAllReduce|lowprecision", re.I)
MOE_PRED_RE = re.compile(r"moe|expert|finalize|swiglu|fc2|grouped|permute|routing|megamoe|mxfp4|w4a", re.I)
DECODE_STEPS = 40


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_profile_run():
    cands = sorted(glob.glob("/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots/*/"
                             "workspace/trtllm/experiments/ascent-benchmark/profile_run.py"))
    here = Path(__file__).resolve().parents[1] / "ascent-benchmark" / "profile_run.py"
    if here.exists():
        cands.insert(0, str(here))
    if not cands:
        raise SystemExit("frozen profile_run.py not found")
    return cands[0]


def load_trace(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in evs if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"
            and e.get("dur") is not None]


def ar_calls(evs):
    """AR kernels in time order with the name of the preceding kernel on the same stream."""
    by_stream = {}
    for e in evs:
        by_stream.setdefault((e.get("pid"), e.get("tid")), []).append(e)
    calls = []
    for key, lst in by_stream.items():
        lst.sort(key=lambda e: e["ts"])
        prev = None
        for e in lst:
            if AR_RE.search(e.get("name", "")):
                calls.append({"ts": e["ts"], "dur": e["dur"], "name": e["name"][:120],
                              "pred": (prev["name"][:120] if prev else ""), "stream": str(key)})
            else:
                prev = e
    calls.sort(key=lambda c: c["ts"])
    return calls


def analyze_window(files):
    per_rank = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(per_rank)
        per_rank[r] = ar_calls(load_trace(f))
    ranks = sorted(per_rank)
    n = min(len(per_rank[r]) for r in ranks) if ranks else 0
    counts = {r: len(per_rank[r]) for r in ranks}
    kernels = {}
    for r in ranks:
        for c in per_rank[r]:
            kernels[c["name"][:60]] = kernels.get(c["name"][:60], 0) + 1
    out = {"ranks": ranks, "calls_per_rank": counts, "matched_calls": n, "ar_kernel_names": kernels}
    if not n:
        return out
    classes = {}
    preds = {"attn": {}, "moe": {}}
    for i in range(n):
        p0 = per_rank[ranks[0]][i]["pred"]
        cls = "moe" if MOE_PRED_RE.search(p0) else "attn"
        preds[cls][p0[:80]] = preds[cls].get(p0[:80], 0) + 1
        durs = [per_rank[r][i]["dur"] for r in ranks]
        t = min(durs)
        d = classes.setdefault(cls, {"calls": 0, "sum_dur": {r: 0.0 for r in ranks}, "sum_transfer": 0.0,
                                     "sum_wait": {r: 0.0 for r in ranks}, "transfers": []})
        d["calls"] += 1
        d["sum_transfer"] += t
        d["transfers"].append(t)
        for r, x in zip(ranks, durs):
            d["sum_dur"][r] += x
            d["sum_wait"][r] += x - t
    steps = DECODE_STEPS
    res = {}
    for cls, d in classes.items():
        c = d["calls"]
        tr = sorted(d["transfers"])
        res[cls] = {
            "calls": c, "calls_per_step": round(c / steps, 2),
            "mean_us_per_call_per_rank": {r: round(d["sum_dur"][r] / c, 2) for r in ranks},
            "transfer_mean_us": round(d["sum_transfer"] / c, 2), "transfer_median_us": round(tr[len(tr) // 2], 2),
            "wait_mean_us_per_rank": {r: round(d["sum_wait"][r] / c, 2) for r in ranks},
            "per_step_total_us_per_rank": {r: round(d["sum_dur"][r] / steps, 1) for r in ranks},
            "per_step_transfer_us": round(d["sum_transfer"] / steps, 1),
            "per_step_wait_us_per_rank": {r: round(d["sum_wait"][r] / steps, 1) for r in ranks},
            "top_predecessors": sorted(preds[cls].items(), key=lambda kv: -kv[1])[:6],
        }
    out["by_class"] = res
    out["note"] = (f"steps assumed {steps} (profile_run DECODE_STEPS); per-call numbers do not depend on it. "
                   "Durations are kernel durations; for NCCL/lamport kernels they include waiting for peers.")
    return out


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[2] if len(sys.argv) > 2 else None
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts"))
    if only is None:
        harness = arg
        if arg and os.path.exists(arg):
            harness = Path(arg).read_text()
        pr = find_profile_run()
        log(f"profile adapter: {pr}")
        cmd = [sys.executable, pr, harness, os.getcwd()]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        last = ""
        for line in p.stdout:
            line = line.rstrip()
            if line:
                last = line
            if len(line) < 2000:
                print(line, flush=True)
        p.wait()
        log(f"profile adapter rc={p.returncode}")
        try:
            prof = json.loads(last)
            log("profile windows:", json.dumps({k: v.get("rank0") for k, v in (prof.get("windows") or {}).items()}))
            if prof.get("errors"):
                log("profile errors:", prof["errors"][:10])
        except Exception:  # noqa: BLE001
            pass
        root = art / "profile" / "traces"
    else:
        root = Path(only)
    if not (root / "decode_c1").exists():
        hits = sorted(glob.glob(str(art / "**" / "decode_c1"), recursive=True))
        log(f"traces not under {root}; found {hits[:4]}")
        if hits:
            root = Path(hits[0]).parent
    summary = {}
    for w in ("decode_c1", "decode_c4"):
        files = sorted(glob.glob(str(root / w / "*.json*")))
        log(f"{w}: {len(files)} trace files")
        try:
            summary[w] = analyze_window(files)
        except Exception as exc:  # noqa: BLE001
            summary[w] = {"error": repr(exc)[:400]}
        log(w, json.dumps(summary[w])[:4000])
    (art / "ar_trace_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"passed": all("by_class" in v for v in summary.values()), "windows": summary}))


if __name__ == "__main__":
    main()
