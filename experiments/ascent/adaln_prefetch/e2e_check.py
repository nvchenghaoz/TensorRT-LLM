"""2-GPU end-to-end check for idea 105 (run in both trees by run_gpu_pair: main, then the branch).

Runs the frozen check adapter (3 brief prompts + the park_portrait repeat, committed engine.json)
through experiments/ascent/run_check_wrapper.py, then summarizes from its artifacts:
  - sha256 of every output MP4 (pass1 + repeat), so the branch's outputs can be compared bitwise
    with main's own outputs from the same pair;
  - the adapter's per-prompt LPIPS/PSNR/pass and repeat verdict;
  - per rank: '[adaln_prefetch]' request lines (hits/misses/prefetches/dropped), 'failed with OOM'
    lines split into warmup (before the first bench request) and timed, the peak_memory lines,
    recompile / graph-break warnings, attention backend / step_precision lines.
Last stdout line: {"passed": <adapter passed>, ...}
Usage: python e2e_check.py <checkpoint>
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "ascent-benchmark").is_dir())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ckpt = sys.argv[1]
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    cmd = [sys.executable, str(ROOT / "experiments/ascent/run_check_wrapper.py"),
           str(ROOT / "experiments/ascent-benchmark/run_check.py"), ckpt]
    print(f"[e2e] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("{"):
            last = line
    proc.wait()
    try:
        adapter = json.loads(last) if last else {}
    except json.JSONDecodeError:
        adapter = {}
    out = {"adapter_valid": adapter.get("valid"), "adapter_errors": adapter.get("errors")}
    per = (adapter.get("comparisons") or {}).get("1", {}).get("per_prompt", [])
    out["per_prompt"] = [{k: p.get(k) for k in ("key", "passed", "lpips", "psnr", "audio_snr",
                                                 "lpips_tol", "bitwise_equal")} for p in per]
    out["repeated_execution_passed"] = adapter.get("repeated_execution_passed")
    out["repeat"] = [{"key": r.get("key"), "passed": r.get("passed"),
                      "bitwise_vs_first": (r.get("vs_first_pass") or {}).get("bitwise_equal")}
                     for r in adapter.get("repeated_execution") or []]
    hashes = {}
    for rec in adapter.get("requests") or []:
        p = rec.get("path")
        if p and Path(p).is_file():
            hashes.setdefault(rec.get("key"), []).append(sha(p))
    out["mp4_sha256"] = hashes
    out["server_e2e_s"] = [(r.get("key"), r.get("server_e2e_s")) for r in adapter.get("requests") or []]
    logs = {}
    for lp in sorted((art / "logs").glob("server-check*.log")):
        text = lp.read_text(errors="replace")
        first_bench = text.find("Generating video")
        lines = text.splitlines()
        oom = [i for i, l in enumerate(lines) if "failed with OOM" in l]
        bench_line = text[:first_bench].count("\n") if first_bench >= 0 else len(lines)
        logs[lp.name] = {
            "adaln_prefetch": [l.strip()[-200:] for l in lines if "[adaln_prefetch]" in l],
            "oom_warmup": sum(1 for i in oom if i < bench_line),
            "oom_timed": sum(1 for i in oom if i >= bench_line),
            "peak_memory": [l.strip()[-200:] for l in lines if "peak_memory" in l][-6:],
            "recompile_or_graph_break": [l.strip()[-200:] for l in lines
                                         if re.search(r"[Rr]ecompil|graph.break|cache_size_limit", l)][:10],
            "step_precision": [l.strip()[-240:] for l in lines if "[step_precision]" in l
                               and "forward_summary" not in l][:6],
            "cached_forwards": sum(1 for l in lines if "forward_summary" in l and "cache=cached" in l),
            "computed_forwards": sum(1 for l in lines if "forward_summary" in l and "cache=computed" in l),
        }
    out["logs"] = logs
    out["passed"] = bool(adapter.get("valid")) and all(p.get("passed") for p in per) and bool(per) \
        and bool(adapter.get("repeated_execution_passed"))
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
