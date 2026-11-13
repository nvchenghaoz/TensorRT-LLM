"""2-GPU E2E check for fa2_sm120 (non-timing).

Runs experiments/ascent/h3_quality_job.py <checkpoint> engine (frozen check adapter with the
committed engine.json: per-prompt LPIPS/PSNR/audio SNR vs the frozen references, same-server
park_portrait repeat, Cache-DiT decisions on both ranks, Qwen3-VL judge), streaming its output,
then scans both ranks' server logs for the fa2_sm120 lines (extension build, call counter),
'failed with OOM' / out-of-memory hits and recompile / graph-break messages.
Usage: python e2e_check.py <checkpoint>
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ckpt = sys.argv[1]
    root = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    proc = subprocess.Popen([sys.executable, str(HERE.parent / "h3_quality_job.py"), ckpt, "engine"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in proc.stdout:
        if line.startswith("{") and '"configs"' in line:
            last = line
        sys.stdout.write(line if len(line) < 4000 else line[:4000] + "\n")
        sys.stdout.flush()
    proc.wait()
    q = json.loads(last) if last else {"passed": False, "error": "no quality result"}
    logs = {}
    for d in ("engine", "engine_retry"):
        for name in ("server-check.log", "server-check-rank1.log"):
            p = root / d / "logs" / name
            if p.is_file():
                logs[f"{d}/{name}"] = p.read_text(errors="replace")
    scan = {}
    for key, text in logs.items():
        lines = text.splitlines()
        fa2 = [ln[-200:] for ln in lines if "[fa2_sm120]" in ln]
        calls = [int(m.group(1)) for ln in fa2 for m in [re.search(r"calls=(\d+)", ln)] if m]
        scan[key] = {
            "fa2_lines": fa2[:3] + (["..."] + fa2[-3:] if len(fa2) > 6 else fa2[3:]),
            "max_calls_logged": max(calls) if calls else 0,
            "disabled": any("disabled, falling back" in ln for ln in fa2),
            "oom_hits": [ln[-240:] for ln in lines if "failed with OOM" in ln or "out of memory" in ln.lower()][:5],
            "recompile_or_graph_break": [ln[-240:] for ln in lines
                                         if re.search(r"[Rr]ecompil|graph break|Graph break", ln)][:5],
        }
    ranks = [v for k, v in scan.items() if k.endswith(".log")]
    op_ran_both = len(ranks) >= 2 and all(v["max_calls_logged"] > 0 and not v["disabled"] for v in ranks[-2:])
    cfg = (q.get("configs") or {}).get("engine", {})
    out = {
        "passed": bool(q.get("passed")) and op_ran_both,
        "quality_passed": q.get("passed"),
        "op_ran_both_ranks": op_ran_both,
        "check_valid": cfg.get("check_valid"),
        "repeated_execution_passed": cfg.get("repeated_execution_passed"),
        "repeat": cfg.get("repeat"),
        "per_prompt": cfg.get("per_prompt"),
        "ranks_identical": cfg.get("ranks_identical"),
        "accuracy": cfg.get("accuracy"),
        "same_subject": cfg.get("same_subject"),
        "fidelity_pass": cfg.get("fidelity_pass"),
        "requests": cfg.get("requests"),
        "log_scan": scan,
    }
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
