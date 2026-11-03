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


# Server settings of the frozen protocol (base proxy job 230's server_settings, engine overlay none).
SERVER_SETTINGS = {
    "context_parallel_size": 1, "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8], "enable_padding": True},
    "enable_attention_dp": False, "enable_chunked_prefill": True,
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5, "tokens_per_block": 128},
    "max_batch_size": 8, "max_num_tokens": 8192, "max_seq_len": 36864, "moe_expert_parallel_size": 4,
    "pipeline_parallel_size": 1, "tensor_parallel_size": 4, "trust_remote_code": True}
WORKLOAD = {"cached_prefix_tokens": 32768, "unique_input_tokens": 2048, "output_tokens": 256,
            "concurrency": [1, 2, 4]}


def run_profile_adapter(checkpoint):
    """Run the frozen profile_run.main() in-process with the protocol's server settings."""
    pr = find_profile_run()
    bench = str(Path(pr).parent)
    log(f"profile adapter: {pr}")
    sys.path.insert(0, bench)
    import common  # noqa: E402  (frozen harness)
    import profile_run  # noqa: E402
    harness = {"server": dict(SERVER_SETTINGS), "server_start_timeout_s": 2700}
    orig_wl = common.workload

    def workload(h):
        try:
            wl = orig_wl(h)
        except Exception as exc:  # noqa: BLE001
            log(f"common.workload failed ({exc!r}); using the protocol workload")
            wl = {}
        wl = dict(wl or {})
        for k, v in WORKLOAD.items():
            wl.setdefault(k, v)
        wl.setdefault("checkpoint", checkpoint)
        log(f"workload: { {k: v for k, v in wl.items() if not isinstance(v, (list, dict)) or k == 'concurrency'} }")
        return wl

    common.workload = workload
    common.load_config = lambda argv: (harness, Path(os.getcwd()).resolve())
    try:
        profile_run.main()
    except SystemExit as e:  # common.final may exit
        log(f"profile adapter exit {e.code}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[2] if len(sys.argv) > 2 else None
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts"))
    if only is None:
        run_profile_adapter(arg)
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
