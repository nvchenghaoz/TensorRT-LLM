"""Idea 108 2-GPU check: frozen check adapter under this tree's engine.json, then log audit.

usage: python experiments/ascent/fa2_server_check.py <checkpoint>

1. experiments/ascent/run_check_wrapper.py runs experiments/ascent-benchmark/run_check.py
   (3 check prompts + same-server park_portrait repeat).
2. Audits server-check*.log on both ranks: every [step_precision] forward_summary line carries
   attention_calls (blocks whose gated attention ran) and fa2_sm120_calls (fa2 op calls in that
   forward). BF16 forwards must have fa2_sm120_calls == attention_calls (zero SDPA fallbacks),
   FP8 forwards 0; the first op calls must come in warmup forwards (JIT build before timing).
3. Counts 'failed with OOM' lines per rank and phase, and sha256 of every output.
Prints one JSON object as the last line.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
BENCH = WT / "experiments" / "ascent-benchmark"
FS = re.compile(
    r"rank=(\d+) forward_summary (\d+)/(\d+) warmup=(\w+) attention=([\w-]+) .*"
    r"cache=(\w+) attention_calls=(\d+)/(\d+) fa2_sm120_calls=(\d+)"
)


def log(*a):
    print("[fa2-check]", *a, flush=True)


def last_json(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ckpt = sys.argv[1]
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    proc = subprocess.Popen(
        [sys.executable, str(WT / "experiments/ascent/run_check_wrapper.py"), str(BENCH / "run_check.py"), ckpt],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(WT))
    out = []
    for line in proc.stdout:
        out.append(line)
        if len(line) < 4000:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    check = last_json("".join(out)) or {}
    pp = ((check.get("comparisons") or {}).get("1") or {}).get("per_prompt", [])
    summary = {
        "valid": check.get("valid"), "errors": check.get("errors"),
        "per_prompt": [{k: m.get(k) for k in ("key", "lpips", "psnr", "audio_snr_db", "bitwise_equal", "passed")}
                       for m in pp],
        "repeated_execution_passed": check.get("repeated_execution_passed"),
        "repeat_vs_first": [{"key": r.get("key"), "passed": r.get("passed"),
                             "bitwise_equal": (r.get("vs_first_pass") or {}).get("bitwise_equal"),
                             "lpips": (r.get("vs_first_pass") or {}).get("lpips")}
                            for r in check.get("repeated_execution") or []],
        "requests": [],
    }
    for r in check.get("requests") or []:
        p = r.get("path")
        summary["requests"].append({"key": r.get("key"), "server_e2e_s": r.get("server_e2e_s"),
                                    "server_denoise_s": r.get("server_denoise_s"),
                                    "sha256": sha(p) if p and Path(p).is_file() else None})
    logs_dir = art / "logs"
    audit, all_ok = {}, True
    for p in sorted(logs_dir.glob("server-check*.log")):
        lines = p.read_text(errors="replace").splitlines()
        fwd = []
        first_timed_line = None
        for i, l in enumerate(lines):
            m = FS.search(l)
            if m:
                fwd.append({"line": i, "rank": int(m.group(1)), "idx": int(m.group(2)), "warmup": m.group(4) == "True",
                            "attention": m.group(5), "cache": m.group(6), "calls": int(m.group(7)),
                            "blocks": int(m.group(8)), "fa2": int(m.group(9))})
                if first_timed_line is None and m.group(4) != "True":
                    first_timed_line = i
        timed = [f for f in fwd if not f["warmup"]]
        bf16 = [f for f in timed if f["attention"].startswith("bf16")]
        fp8 = [f for f in timed if f["attention"] == "fp8"]
        bad_bf16 = [f for f in bf16 if f["fa2"] != f["calls"] or f["calls"] == 0]
        bad_fp8 = [f for f in fp8 if f["fa2"] != 0]
        warm_fa2 = sum(f["fa2"] for f in fwd if f["warmup"])
        oom_warm = oom_timed = 0
        for i, l in enumerate(lines):
            if "failed with OOM" in l:
                if first_timed_line is None or i < first_timed_line:
                    oom_warm += 1
                else:
                    oom_timed += 1
        fa2_lines = [l for l in lines if "[fa2_sm120]" in l][:6]
        ok = bool(bf16) and not bad_bf16 and not bad_fp8 and warm_fa2 > 0 and oom_timed == 0
        all_ok &= ok
        audit[p.name] = {
            "timed_forwards": len(timed), "bf16_forwards": len(bf16), "fp8_forwards": len(fp8),
            "bf16_op_calls_total": sum(f["fa2"] for f in bf16),
            "bf16_attention_calls_total": sum(f["calls"] for f in bf16),
            "bf16_per_forward_head": [(f["idx"], f["cache"], f["calls"], f["fa2"]) for f in bf16[:16]],
            "bad_bf16": bad_bf16[:5], "bad_fp8": bad_fp8[:5], "warmup_op_calls": warm_fa2,
            "oom_warmup": oom_warm, "oom_timed": oom_timed, "fa2_log_lines": fa2_lines, "ok": ok,
        }
        log(p.name, json.dumps({k: v for k, v in audit[p.name].items() if k != "bf16_per_forward_head"}))
    summary["log_audit"] = audit
    summary["audit_ok"] = bool(audit) and all_ok
    summary["passed"] = bool(summary["audit_ok"] and check.get("valid") and check.get("repeated_execution_passed")
                             and pp and all(m.get("passed") for m in pp))
    summary["compared"] = ("frozen check adapter (LPIPS/PSNR vs frozen references, park repeat) + per-forward "
                           "fa2_sm120 op count == gated attention calls in BF16 forwards on both ranks")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
