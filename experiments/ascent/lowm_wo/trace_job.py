"""Idea 233 S4: in-server Kineto decode replay (frozen profile adapter, 4 GPUs) + route audit of the WO projection.

Runs experiments/ascent-benchmark/profile_run.py in-process with the protocol's server settings (as
experiments/ascent/ar_trace_job.py does; its analyse/gzip pool is skipped) and reduces the decode_c1 / decode_c4
windows (batch 1 and 4, 40 steps) of every rank in one process:
  * per-step counts of K2' (csa2_lowm_wo_a), K3' (csa2_mxfp8_low_m_gemm), Triton K1 (_inv_rope_quant128) and every
    b12x kernel name; the same counts from main's job-1063 traces (2fa8e034) when readable, for the delta;
  * on rank 0: the kernels before/after each K2' on its stream, the share of K2'->K3' pairs where K3' starts before
    K2' ends (PDL overlap), and the kernel after K3' (the attention all-reduce);
  * server log lines about the low-M / b12x WO projection builds and CUDA graph capture.
Usage: trace_job.py <checkpoint> [only_analyze_dir]. Last stdout line: JSON summary. Not timing evidence.
"""
import glob
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

K2_RE = re.compile(r"csa2_lowm_wo_a")
K3_RE = re.compile(r"csa2_mxfp8_low_m_gemm")
K1_RE = re.compile(r"inv_rope_quant128")
B12X_RE = re.compile(r"b12x", re.I)
AR_RE = re.compile(r"allreduce|all_reduce|lamport|oneshot|twoshot|nccl", re.I)
STEPS = 40
BASE_TRACES = ("/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots/0/jobs/"
               "90327323a0d447eda2bff999accf58c9/artifacts/profile/traces")

SERVER_SETTINGS = {
    "context_parallel_size": 1, "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8], "enable_padding": True},
    "enable_attention_dp": False, "enable_chunked_prefill": True,
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5, "tokens_per_block": 128},
    "max_batch_size": 8, "max_num_tokens": 8192, "max_seq_len": 36864, "moe_expert_parallel_size": 4,
    "pipeline_parallel_size": 1, "tensor_parallel_size": 4, "trust_remote_code": True}
WORKLOAD = {"cached_prefix_tokens": 32768, "unique_input_tokens": 2048, "output_tokens": 256,
            "concurrency": [1, 2, 4], "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "model_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170"}

_EV = re.compile(rb'"ph":\s*"X",\s*"cat":\s*"kernel",\s*"name":\s*"((?:[^"\\]|\\.)*)",\s*"pid":\s*"?([^,"]*)"?,'
                 rb'\s*"tid":\s*"?([^,"]*)"?,\s*"ts":\s*([0-9.eE+-]+),\s*"dur":\s*([0-9.eE+-]+)')


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_bench():
    cands = [Path.cwd() / "experiments" / "ascent-benchmark", Path(__file__).resolve().parents[2] / "ascent-benchmark"]
    for c in cands:
        if (c / "profile_run.py").exists():
            return c
    raise SystemExit(f"frozen profile_run.py not found in {cands}")


def run_profile_adapter(checkpoint):
    bench = find_bench()
    log(f"profile adapter dir: {bench}")
    sys.path.insert(0, str(bench))
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
        return wl

    common.workload = workload
    common.load_config = lambda argv: (harness, Path(os.getcwd()).resolve())
    profile_run.analyze_files = lambda paths, hb=None: {p: (p, {"kernels": 0, "parser": "skipped"}) for p in paths}
    try:
        profile_run.main()
    except SystemExit as e:
        log(f"profile adapter exit {e.code}")
    except Exception as exc:  # noqa: BLE001
        log(f"profile adapter raised {exc!r}; reducing the traces it wrote")


def load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as f:
        buf = f.read()
    evs = []
    for m in _EV.finditer(buf):
        try:
            name = json.loads(b'"' + m.group(1) + b'"')
        except Exception:  # noqa: BLE001
            name = m.group(1).decode(errors="replace")
        evs.append((float(m.group(4)), float(m.group(5)), name, m.group(2).decode() + "/" + m.group(3).decode()))
    if not evs and b'"kernel"' in buf:
        data = json.loads(buf)
        raw = data.get("traceEvents", data) if isinstance(data, dict) else data
        evs = [(float(e["ts"]), float(e["dur"]), e.get("name", ""), f"{e.get('pid')}/{e.get('tid')}") for e in raw
               if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
    evs.sort()
    return evs


def counts(evs):
    c = {"k2": 0, "k3": 0, "k1": 0, "b12x": 0}
    names = {}
    for _, _, n, _ in evs:
        if K2_RE.search(n):
            c["k2"] += 1
        if K3_RE.search(n):
            c["k3"] += 1
        if K1_RE.search(n):
            c["k1"] += 1
        if B12X_RE.search(n):
            c["b12x"] += 1
            names[n[:110]] = names.get(n[:110], 0) + 1
    return {k: round(v / STEPS, 2) for k, v in c.items()}, {k: round(v / STEPS, 2) for k, v in
                                                            sorted(names.items(), key=lambda kv: -kv[1])[:12]}


def sequence(evs):
    by = {}
    for e in evs:
        by.setdefault(e[3], []).append(e)
    pairs = overlap = 0
    before, after_k2, after_k3 = {}, {}, {}
    gaps = []
    for lst in by.values():
        for i, (ts, dur, n, _) in enumerate(lst):
            if not K2_RE.search(n):
                continue
            if i > 0:
                p = lst[i - 1][2][:80]
                before[p] = before.get(p, 0) + 1
            if i + 1 < len(lst):
                nx = lst[i + 1]
                after_k2[nx[2][:80]] = after_k2.get(nx[2][:80], 0) + 1
                if K3_RE.search(nx[2]):
                    pairs += 1
                    overlap += int(nx[0] < ts + dur)
                    gaps.append(round(nx[0] - (ts + dur), 2))
                    if i + 2 < len(lst):
                        a = lst[i + 2][2][:80]
                        after_k3[a] = after_k3.get(a, 0) + 1
    top = lambda d: sorted(d.items(), key=lambda kv: -kv[1])[:6]  # noqa: E731
    gaps.sort()
    return {"k2_k3_pairs": pairs, "k3_starts_before_k2_ends": overlap,
            "k3_start_minus_k2_end_us_p10_p50_p90": [gaps[int(q * (len(gaps) - 1))] for q in (0.1, 0.5, 0.9)]
            if gaps else None,
            "before_k2": top(before), "after_k2": top(after_k2), "after_k3": top(after_k3)}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[2] if len(sys.argv) > 2 else None
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "artifacts"))
    if only is None:
        run_profile_adapter(ckpt)
        root = art / "profile" / "traces"
    else:
        root = Path(only)
    if not (root / "decode_c1").exists():
        hits = sorted(glob.glob(str(art / "**" / "decode_c1"), recursive=True))
        if hits:
            root = Path(hits[0]).parent
    summary = {"trace_root": str(root)}
    ok = True
    for w in ("decode_c1", "decode_c4"):
        files = sorted(glob.glob(str(root / w / "*.json*")))
        res = {"files": len(files), "ranks": {}}
        for f in files:
            m = re.search(r"rank-(\d+)", os.path.basename(f))
            r = int(m.group(1)) if m else -1
            evs = load(f)
            c, bn = counts(evs)
            res["ranks"][r] = {"kernels": len(evs), "per_step": c}
            if r == 0:
                res["rank0_b12x_names_per_step"] = bn
                res["rank0_sequence"] = sequence(evs)
            log(w, "rank", r, len(evs), "kernels", c)
            del evs
        base_files = sorted(glob.glob(os.path.join(BASE_TRACES, w, "*rank-0*.json*")))
        if base_files:
            try:
                bevs = load(base_files[0])
                bc, bbn = counts(bevs)
                res["main_1063_rank0_per_step"] = bc
                res["main_1063_rank0_b12x_names_per_step"] = bbn
                del bevs
            except Exception as exc:  # noqa: BLE001
                res["main_1063_error"] = repr(exc)[:300]
        else:
            res["main_1063_error"] = f"no base trace under {BASE_TRACES}/{w}"
        r0 = res["ranks"].get(0, {}).get("per_step", {})
        seq = res.get("rank0_sequence", {})
        res["route_ok"] = bool(files) and r0.get("k1", 1) == 0 and r0.get("k2", 0) == 40 and r0.get("k3", 0) >= 40 \
            and seq.get("k2_k3_pairs", 0) > 0
        ok &= res["route_ok"]
        summary[w] = res
        log(w, json.dumps(res)[:3000])
    logs = sorted(glob.glob(str(art / "**" / "*.log"), recursive=True))
    hits = []
    pat = re.compile(r"low-M WO|b12x WO projection|csa2_lowm|capture.*(error|fail)|Warm up the", re.I)
    for lf in logs:
        try:
            with open(lf, errors="replace") as fh:
                for line in fh:
                    if pat.search(line):
                        hits.append(f"{os.path.basename(lf)}: {line.strip()[:240]}")
        except OSError:
            pass
    summary["server_log_hits"] = hits[:40]
    (art / "lowm_wo_trace_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"passed": ok, "summary": {w: {k: summary[w].get(k) for k in
                                                     ("route_ok", "main_1063_rank0_per_step")}
                                                 | {"rank0": summary[w]["ranks"].get(0, {}).get("per_step"),
                                                    "seq": {k: v for k, v in summary[w].get("rank0_sequence",
                                                                                            {}).items()
                                                            if k.startswith("k")}}
                                                 for w in ("decode_c1", "decode_c4")},
                      "server_log_hits": hits[:10]}, default=str))


if __name__ == "__main__":
    main()
