"""Idea 81 re-land check: frozen check adapter at K=14, then fidelity_judge, then log quotes.

usage: python experiments/ascent/reland_check.py <checkpoint>

1. experiments/ascent/run_check_wrapper.py runs the frozen experiments/ascent-benchmark/run_check.py
   (3 check prompts + same-server park_portrait repeat) with this worktree's engine.json overlay.
2. experiments/ascent/fidelity_judge.py on the pass-1 outputs: LPIPS (alex, all frames), PSNR,
   audio SNR vs the frozen BF16 refs, stills at frames 25/62/98 with the same-subject verdict.
3. Quotes: the server's effective visual_gen_args.yaml, TLLM_H3_TP_OVERLAP_CHUNKS, every
   [step_precision] line on both ranks (gate/overlap log + per-forward recipe lines), and any
   recompile / graph-break lines inside the timed segments.
Prints one JSON object as the last line.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
BENCH = WT / "experiments" / "ascent-benchmark"
BARS = {"park_portrait": 0.347, "perfume_bottle": 0.166, "golden_robot": 0.508}


def log(*a):
    print("[reland]", *a, flush=True)


def last_json(text: str):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_streamed(cmd):
    log("running", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(WT))
    out = []
    for line in proc.stdout:
        out.append(line)
        # Keep progress visible but do not echo the giant final JSON line.
        if len(line) < 4000:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    return proc.returncode, "".join(out)


def main():
    ckpt = sys.argv[1]
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    summary = {"env_TLLM_H3_TP_OVERLAP_CHUNKS": os.environ.get("TLLM_H3_TP_OVERLAP_CHUNKS")}
    rc, out = run_streamed([sys.executable, str(WT / "experiments/ascent/run_check_wrapper.py"),
                            str(BENCH / "run_check.py"), ckpt])
    check = last_json(out)
    if check is None:
        p = art / "adapter_result.json"
        check = json.loads(p.read_text()) if p.is_file() else {}
    pp = (check.get("comparisons", {}).get("1", {}) or {}).get("per_prompt", [])
    summary["check"] = {
        "valid": check.get("valid"),
        "errors": check.get("errors"),
        "output_class": (check.get("output") or {}).get("class"),
        "engine_overlay": check.get("engine_overlay"),
        "tolerance": check.get("tolerance"),
        "per_prompt": [{k: m.get(k) for k in ("key", "lpips", "psnr", "audio_snr_db", "bitwise_equal",
                                               "passed")} for m in pp],
        "repeated_execution_passed": check.get("repeated_execution_passed"),
        "repeated_execution": [
            {"key": r.get("key"), "passed": r.get("passed"),
             "vs_first_pass": {k: (r.get("vs_first_pass") or {}).get(k)
                               for k in ("bitwise_equal", "lpips", "psnr", "audio_snr_db")},
             "vs_reference": {k: (r.get("vs_reference") or {}).get(k) for k in ("lpips", "psnr")}}
            for r in check.get("repeated_execution", []) or []],
        "requests": [{k: r.get(k) for k in ("key", "path", "server_e2e_s", "server_denoise_s")}
                     for r in check.get("requests", []) or []],
    }
    server_log = (check.get("artifacts") or {}).get("server_log")
    # Effective server settings.
    logs_dir = art / "logs"
    yaml_path = logs_dir / "server-check.visual_gen_args.yaml"
    summary["visual_gen_args_yaml"] = yaml_path.read_text() if yaml_path.is_file() else None
    # [step_precision] lines and compile diagnostics on both ranks.
    quotes = {}
    recompiles = {}
    for p in sorted(logs_dir.glob("server-check*.log")):
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        sp = [l for l in lines if "[step_precision]" in l or "[FlashInfer FP8 attention]" in l]
        gate = [l for l in sp if "attention_bf16_forwards=" in l and "forward " not in l]
        fwd = [l for l in sp if " forward " in l]
        quotes[p.name] = {"gate_lines": gate[:4], "n_forward_lines": len(fwd),
                          "forward_lines_head": fwd[:3], "forward_lines_sample": fwd[-52:],
                          "tactic_lines": [l for l in sp if "FlashInfer FP8 attention" in l][:4]}
        rc_lines = [l for l in lines if re.search(r"[Rr]ecompil|graph break|Graph break", l)]
        recompiles[p.name] = rc_lines[:20]
    summary["log_quotes"] = quotes
    summary["recompile_or_graph_break_lines"] = recompiles
    # Per-request recipe summary on each rank: which forwards ran bf16/fp8 (non-warmup).
    recipe = {}
    for p in sorted(logs_dir.glob("server-check*.log")):
        reqs = []
        for l in p.read_text(errors="replace").splitlines():
            m = re.search(r"forward (\d+)/(\d+) warmup=(\w+) attention=(bf16|fp8)", l)
            if not m:
                continue
            idx, n, warm, path = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            if warm == "True":
                continue
            if idx == 0 or not reqs:
                reqs.append({})
            reqs[-1][idx] = path
        recipe[p.name] = [{"n": len(r),
                           "bf16": sorted(k for k, v in r.items() if v == "bf16"),
                           "fp8": [min((k for k, v in r.items() if v == "fp8"), default=None),
                                   max((k for k, v in r.items() if v == "fp8"), default=None)]}
                          for r in reqs]
    summary["recipe_per_request"] = recipe
    # Fidelity judge on pass-1 outputs.
    pass1 = {}
    for r in (check.get("requests") or [])[:3]:
        if r.get("key") and r.get("path"):
            pass1.setdefault(r["key"], r["path"])
    judge = None
    if pass1:
        cmd = [sys.executable, str(WT / "experiments/ascent/fidelity_judge.py"), "--checkpoint", ckpt,
               "--label", "fp8_k14", "--bench", str(BENCH / "ab_metrics.py"), "--first-gated-step", "15"]
        if server_log:
            cmd += ["--server-log", server_log]
        cmd += [f"{k}={v}" for k, v in pass1.items()]
        _, jout = run_streamed(cmd)
        judge = last_json(jout)
    summary["fidelity"] = judge
    ok_lpips = None
    if judge:
        ok_lpips = {}
        for k, m in (judge.get("metrics") or {}).items():
            lp = (m.get("candidate") or {}).get("lpips")
            ok_lpips[k] = {"lpips": lp, "psnr": (m.get("candidate") or {}).get("psnr"),
                           "audio_snr_db": (m.get("candidate") or {}).get("audio_snr_db"),
                           "floor": (m.get("bf16_reorder_floor") or {}).get("lpips"),
                           "bar": BARS.get(k),
                           "within_bar_minus_margin": lp is not None and BARS.get(k) is not None
                           and lp <= BARS[k] - 0.005}
    summary["lpips_vs_bar"] = ok_lpips
    passed = bool(
        check.get("repeated_execution_passed")
        and ok_lpips and all(v["within_bar_minus_margin"] for v in ok_lpips.values())
        and judge and judge.get("judge_controls_ok") and all((judge.get("same_subject") or {}).values())
    )
    summary["passed"] = passed
    summary["compared"] = ("FP8 K=14 server outputs vs frozen BF16 references (LPIPS alex 124 frames, PSNR, "
                           "audio SNR) against bar floor+0.03 with 0.005 margin; same-server park_portrait "
                           "repeat vs first pass; Qwen3-VL same-subject verdict at frames 25/62/98")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
