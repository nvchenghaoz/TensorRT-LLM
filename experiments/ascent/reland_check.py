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
import threading
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


def source_grep():
    """Quote the single config class, controller class, set_step and reset sites."""
    pats = {
        "class_StepPrecisionConfig": re.compile(r"^class StepPrecisionConfig\b"),
        "controller_class": re.compile(r"^class \w*(StepController|PrecisionController)\b"),
        "set_step_call": re.compile(r"\.set_step\("),
        "step_controller_reset_call": re.compile(r"step_controller\.reset\(\)"),
        "step_precision_controller_assign": re.compile(r"\.step_precision_controller\s*="),
    }
    hits = {k: [] for k in pats}
    for p in sorted((WT / "tensorrt_llm").rglob("*.py")):
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for k, pat in pats.items():
                if pat.search(line):
                    hits[k].append(f"{p.relative_to(WT)}:{i}:{line.strip()}")
    return hits


def config_parses():
    """No-GPU parse of the merged engine.json and of linear-only / VANILLA configs."""
    out = {}
    try:
        from tensorrt_llm.visual_gen.args import StepPrecisionConfig, VisualGenArgs
    except Exception as exc:  # noqa: BLE001
        return {"import_error": repr(exc)}
    out["step_precision_fields"] = sorted(StepPrecisionConfig.model_fields)
    eng = json.loads((WT / "experiments/ascent-config/engine.json").read_text())["visual_gen_args"]
    cases = {
        "merged_engine_json": (eng, True),
        "main_cache_only_vanilla": ({"attention_config": {"backend": "VANILLA"},
                                     "cache_config": eng["cache_config"]}, True),
        "vanilla_step_precision_without_attention_gate": (
            {"attention_config": {"backend": "VANILLA"}, "cache_config": eng["cache_config"],
             "step_precision": {}}, True),
        "vanilla_with_attention_bf16_forwards_must_raise": (
            {"attention_config": {"backend": "VANILLA"}, "step_precision": {"attention_bf16_forwards": 14}},
            False),
    }
    if "linear_bf16_forwards" in StepPrecisionConfig.model_fields:
        cases["linear_only_vanilla"] = ({"attention_config": {"backend": "VANILLA"},
                                         "step_precision": {"linear_bf16_forwards": 10}}, True)
    for name, (cfg, should_parse) in cases.items():
        try:
            args = VisualGenArgs(**cfg)
            parsed, detail = True, {"step_precision": None if args.step_precision is None
                                    else args.step_precision.model_dump(),
                                    "backend": args.attention_config.backend,
                                    "cache_backend": args.cache_backend}
        except Exception as exc:  # noqa: BLE001
            parsed, detail = False, repr(exc)[:300]
        out[name] = {"parsed": parsed, "expected": should_parse, "ok": parsed == should_parse,
                     "detail": detail}
    return out


class MemPoller(threading.Thread):
    """Peak nvidia-smi memory.used per GPU over the whole check (MiB)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = {}
        self.total = {}
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                txt = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
                for line in txt.strip().splitlines():
                    idx, used, total = [x.strip() for x in line.split(",")]
                    self.peak[idx] = max(self.peak.get(idx, 0), int(used))
                    self.total[idx] = int(total)
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(2.0)


def forward_summaries(logs_dir: Path):
    """Per rank, per request: attention recipe, cache state per forward (non-warmup)."""
    res = {}
    pat = re.compile(r"rank=(\d+) forward_summary (\d+)/(\d+) warmup=(\w+) attention=(\S+) "
                     r"linear=(\S+).* cache=(\w+) attention_calls=(\d+)/(\d+)")
    per_rank = {}
    warm = {}
    for p in sorted(logs_dir.glob("server-check*.log")):
        for l in p.read_text(errors="replace").splitlines():
            m = pat.search(l)
            if not m:
                continue
            rank, idx, n, w, att, lin, cache, calls, nb = m.groups()
            if w == "True":
                warm.setdefault(rank, []).append((int(idx), att, cache, int(calls)))
                continue
            reqs = per_rank.setdefault(rank, [])
            if int(idx) == 0 or not reqs:
                reqs.append({})
            reqs[-1][int(idx)] = (att, lin, cache, int(calls))
    for rank, reqs in per_rank.items():
        out = []
        for r in reqs:
            out.append({
                "n_forwards": len(r),
                "bf16_fa2": sorted(k for k, v in r.items() if v[0] == "bf16-fa2"),
                "fp8": sorted(k for k, v in r.items() if v[0] == "fp8"),
                "cached": sorted(k for k, v in r.items() if v[2] == "cached"),
                "fp8_and_cached": sorted(k for k, v in r.items() if v[0] == "fp8" and v[2] == "cached"),
                "linear": sorted({v[1] for v in r.values()}),
                "attention_calls": {k: v[3] for k, v in sorted(r.items())},
            })
        res[f"rank{rank}"] = out
    res["warmup"] = {f"rank{k}": v for k, v in warm.items()}
    return res


def main():
    ckpt = sys.argv[1]
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    summary = {"env_TLLM_H3_TP_OVERLAP_CHUNKS": os.environ.get("TLLM_H3_TP_OVERLAP_CHUNKS"),
               "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    summary["source_grep"] = source_grep()
    log("source grep:", json.dumps(summary["source_grep"]))
    summary["config_parses"] = config_parses()
    log("config parses:", json.dumps(summary["config_parses"], default=str))
    summary["engine_json"] = json.loads((WT / "experiments/ascent-config/engine.json").read_text())
    poller = MemPoller()
    poller.start()
    rc, out = run_streamed([sys.executable, str(WT / "experiments/ascent/run_check_wrapper.py"),
                            str(BENCH / "run_check.py"), ckpt])
    poller.stop.set()
    summary["peak_memory_used_mib_per_gpu"] = {"peak": poller.peak, "total": poller.total}
    log("peak memory MiB:", json.dumps(summary["peak_memory_used_mib_per_gpu"]))
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
    fs = forward_summaries(logs_dir)
    summary["forward_summaries"] = fs
    recipe_ok = True
    for rank in ("rank0", "rank1"):
        reqs = fs.get(rank) or []
        if not reqs:
            recipe_ok = False
        for r in reqs:
            if not (r["n_forwards"] == 49 and r["bf16_fa2"] == list(range(14))
                    and r["fp8"] == list(range(14, 49)) and len(r["cached"]) == 20
                    and all(c >= 14 for c in r["cached"])):
                recipe_ok = False
    summary["recipe_ok_both_ranks"] = recipe_ok
    g = summary["source_grep"]
    summary["grep_single_sites_ok"] = (len(g["class_StepPrecisionConfig"]) == 1
                                       and len(g["controller_class"]) == 1
                                       and len(g["set_step_call"]) == 1)
    cp = summary["config_parses"]
    summary["config_parses_ok"] = all(v.get("ok") for k, v in cp.items() if isinstance(v, dict))
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
        and recipe_ok and summary["grep_single_sites_ok"] and summary["config_parses_ok"]
    )
    summary["passed"] = passed
    summary["compared"] = ("FP8 K=14 server outputs vs frozen BF16 references (LPIPS alex 124 frames, PSNR, "
                           "audio SNR) against bar floor+0.03 with 0.005 margin; same-server park_portrait "
                           "repeat vs first pass; Qwen3-VL same-subject verdict at frames 25/62/98")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
