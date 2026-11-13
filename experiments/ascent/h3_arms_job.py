#!/usr/bin/env python3
"""Idea 56 arms job: frozen check adapter + judge per Cache-DiT static-mask arm, then the arms table.

Usage: python h3_arms_job.py <checkpoint> <arm> [<arm> ...]     (arms: E D C fallback W w14_c2)

Each arm = one trtllm-serve server (2 GPUs) running the frozen run_check.py (park, perfume, robot,
then park again in the same server = per-request reset check A,B,C,A), through
h3_quality_job.main (harness built from the protocol server settings, engine.json written per arm and
restored, streamed output, judge after all servers stop: 21 items + same-subject at frames 25/62/98).
Every arm is Fn=8, Bn=0, static; max_warmup_steps = the arm's dense head, mcc = -1. Masks come from
h3_cache_arms_table.MASKS and are converted to scm bins here; length 49 and cached count asserted.
Afterwards: quotes each server's visual_gen_args.yaml (fails on CUDNN), per-rank refresh/decision
lines, warmup decision (steps=2 cached=1 on both ranks), and the arms table (LPIPS/PSNR/SNR, margins,
increments over L2 = job 371, Step k/49 timing, measured vs predicted). Last line: JSON.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h3_quality_job as q  # noqa: E402
from h3_cache_arms_table import EXPECTED_CACHED, MASKS, check_masks  # noqa: E402

L2 = os.environ.get("H3_ARMS_L2", "park=0.0390,perfume=0.0198,robot=0.0887")  # default: job 371 (BF16)
SNR_FLAG = {"park_portrait": 1.74, "perfume_bottle": 4.60, "golden_robot": 7.47}
SNR_REF = json.loads(os.environ.get("H3_ARMS_SNR_REF", "{}"))  # e.g. job 455's per-prompt audio SNR
K_BF16 = int(os.environ.get("H3_ARMS_K_BF16", "0"))  # attention_bf16_forwards on the base, 0 = none
STEP_RE = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+)s")
FWD_RE = re.compile(r"\[step_precision\] rank=\d+ forward_summary (\d+)/(\d+) warmup=(\w+) attention=(\S+) "
                    r".*?cache=(\w+) attention_calls=(\d+)/(\d+)")
OOM_RE = re.compile(r"(failed with OOM|out of memory|CUDA error)[^\n]*", re.IGNORECASE)


def step_requests(path: Path):
    """Rank-0 Step lines split into requests: list of [(k, n, seconds)]."""
    reqs, cur = [], []
    if not path.is_file():
        return reqs
    for ln in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(ln)
        if not m:
            continue
        k, n, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if k == 1 and cur:
            reqs.append(cur)
            cur = []
        cur.append((k, n, t))
    if cur:
        reqs.append(cur)
    return reqs


def phase_timing(reqs, mask, k_bf16):
    """Dense/cached mean per precision phase over 49-forward requests (forward i = Step i+1)."""
    out = []
    for r in reqs:
        if not r or r[0][1] != len(mask):
            continue
        rec = {}
        for (k, n, t) in r:
            i = k - 1
            ph = "bf16" if (k_bf16 and i < k_bf16) else ("fp8" if k_bf16 else "all")
            key = f"{ph}_{'dense' if mask[i] else 'cached'}"
            rec.setdefault(key, []).append(t)
        rec = {k: (round(sum(v) / len(v), 3), len(v)) for k, v in rec.items()}
        rec["step1_3"] = [t for (k, n, t) in r[:3]]
        rec["total"] = round(sum(t for (_, _, t) in r), 2)
        out.append(rec)
    return out


def fwd_summary(path: Path):
    """Per-forward (index, warmup, attention phase, cache state) lines of one rank."""
    if not path.is_file():
        return []
    return [(int(m.group(1)), m.group(3), m.group(4), m.group(5))
            for ln in path.read_text(errors="replace").splitlines() for m in [FWD_RE.search(ln)] if m]
HEAD = {"E": 8, "D": 8, "C": 8, "fallback": 10, "W": 14, "w14_c2": 14, "F": 8, "Es": 8, "Ds": 8, "U31": 8,
        "F_F1": 8, "U34_F1": 8}
MCC = {"w14_c2": 2}
FN = {"F_F1": 1, "U34_F1": 1}  # idea 113: Fn_compute_blocks 1 (default 8)
# Idea 113: the FIRST arm's server also runs park seed 43 then park seed 42 after the frozen check's
# repeat (per-request reset), through run_check_reset.py (imports the frozen run_check unchanged).
RESET_FIRST_ARM = os.environ.get("H3_ARMS_RESET_FIRST", "1") == "1"


def bins_of(mask):
    comp, cache = [], []
    i = 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        (comp if mask[i] == 1 else cache).append(j - i)
        i = j
    return comp, cache


def arm_config(arm):
    mask = MASKS[arm]
    assert len(mask) == 49 and mask.count(0) == EXPECTED_CACHED[arm], arm
    comp, cache = bins_of(mask)
    rebuilt = []
    for i, c in enumerate(comp):
        rebuilt += [1] * c
        if i < len(cache):
            rebuilt += [0] * cache[i]
    assert rebuilt == mask, arm
    return {"cache_backend": "cache_dit", "Fn_compute_blocks": FN.get(arm, 8), "Bn_compute_blocks": 0,
            "max_warmup_steps": HEAD[arm], "max_continuous_cached_steps": MCC.get(arm, -1),
            "residual_diff_threshold": 0.08, "scm_compute_bins": comp, "scm_cache_bins": cache,
            "scm_steps_policy": "static"}


REFRESH_RE = re.compile(r"Cache-DiT refresh rank=\d+: (.*)$")
DEC_RE = re.compile(r"Cache-DiT decisions rank=\d+: context=\S+ (steps=\d+ cached=\d+ cached_steps=\[[^\]]*\])")


def grep(path: Path, rx):
    if not path.is_file():
        return []
    return [m.group(1) for ln in path.read_text(errors="replace").splitlines() for m in [rx.search(ln)] if m]


def _run_check_script(h: dict, art: Path, script: Path) -> dict:
    """h3_quality_job.run_check with another adapter script (same harness JSON, same streaming)."""
    env = dict(os.environ, AUTO_DEV_ARTIFACTS=str(art))
    print(f"[arms] check via {script.name} -> {art}", flush=True)
    proc = subprocess.Popen([sys.executable, str(script), json.dumps(h)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = None
    for line in proc.stdout:
        if line.startswith("{") and '"adapter"' in line:
            last = line
            sys.stdout.write(line[:2000] + "\n")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return json.loads(last) if last else {"valid": False, "errors": ["no check result"]}


RECOMP_RE = re.compile(r"(Recompiling[^\n]*)")
ENABLE_RE = re.compile(r"(Cache-DiT: MiniMax-H3 \S+ \d+ blocks, Fn=.*)$")
STATS_LINE_RE = re.compile(r"accumulated_cached_steps=(\d+)")


def recompile_scan(path: Path):
    """Recompiling lines with (request ordinal by 'Step 1/' lines, last Step k/n seen before them)."""
    out, req, last = [], 0, None
    if not path.is_file():
        return out
    for ln in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(ln)
        if m:
            if m.group(1) == "1":
                req += 1
            last = f"{m.group(1)}/{m.group(2)}"
            continue
        r = RECOMP_RE.search(ln)
        if r:
            out.append({"request": req, "after_step": last, "line": r.group(1)[:260]})
    return out


def main():
    ckpt, arms = sys.argv[1], sys.argv[2:]
    check_masks()
    for arm in arms:
        q.CONFIGS[arm] = arm_config(arm)
        print(f"[arms] {arm}: {json.dumps(q.CONFIGS[arm])}", flush=True)
    root = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))

    captured = {}
    _print = print

    def tee(*a, **k):
        s = " ".join(str(x) for x in a)
        if s.startswith("{\"passed\""):
            captured["quality"] = json.loads(s)
            _print(s[:3000], **k)
        else:
            _print(*a, **k)

    q.print = tee
    if RESET_FIRST_ARM and arms:
        _orig_rc = q.run_check
        first = arms[0]

        def _rc(h, art):
            if art.name in (first, f"{first}_retry"):
                return _run_check_script(h, art, HERE / "run_check_reset.py")
            return _orig_rc(h, art)

        q.run_check = _rc
    sys.argv = [sys.argv[0], ckpt] + arms
    q.main()
    (root / "quality.json").write_text(json.dumps(captured.get("quality", {}), default=str))
    post(root, arms, captured.get("quality", {}).get("configs", {}))


def post(root: Path, arms, quality):
    out = {"arms": {}}
    specs = []
    for arm in arms:
        art = root / arm
        if not (art / "adapter_result.json").is_file() and (root / f"{arm}_retry").is_dir():
            art = root / f"{arm}_retry"
        logs = art / "logs"
        yaml_p = logs / "server-check.visual_gen_args.yaml"
        ytxt = yaml_p.read_text() if yaml_p.is_file() else ""
        print(f"[arms] {arm} visual_gen_args.yaml:\n{ytxt}", flush=True)
        r0 = grep(logs / "server-check.log", REFRESH_RE)
        r1 = grep(logs / "server-check-rank1.log", REFRESH_RE)
        d0 = grep(logs / "server-check.log", DEC_RE)
        d1 = grep(logs / "server-check-rank1.log", DEC_RE)
        for ln in r0[:3] + r1[:3]:
            print(f"[arms] {arm} refresh: {ln}", flush=True)
        qa = quality.get(arm, {})
        snr_flags = [p["key"] for p in qa.get("per_prompt", [])
                     if p.get("audio_snr_db") is not None and (p["audio_snr_db"] < SNR_FLAG.get(p["key"], 0)
                                                                 or p["audio_snr_db"] < 0)]
        mask = MASKS[arm]
        f0 = fwd_summary(logs / "server-check.log")
        f1 = fwd_summary(logs / "server-check-rank1.log")
        fwd_ok = None
        if K_BF16:
            # Timed forwards: phase must follow the gate (i < K -> bf16-fa2) and cache state the mask.
            timed = [x for x in f0 if x[1] == "False"]
            fwd_ok = bool(timed) and all(
                (a == ("bf16-fa2" if i < K_BF16 else "fp8")) and (c == ("computed" if mask[i] else "cached"))
                for (i, w, a, c) in timed if len(mask) == 49)
        import yaml as _yaml
        yv = _yaml.safe_load(ytxt) if ytxt else {}
        attn = (yv or {}).get("attention_config") or {}
        oom = []
        for p in (logs / "server-check.log", logs / "server-check-rank1.log"):
            if p.is_file():
                oom += [f"{p.name}: {m.group(0)[:200]}" for m in OOM_RE.finditer(p.read_text(errors="replace"))]
        steps = step_requests(logs / "server-check.log")
        ares = {}
        if (art / "adapter_result.json").is_file():
            try:
                ares = json.loads((art / "adapter_result.json").read_text())
            except ValueError:
                ares = {}
        want_fn = FN.get(arm, 8)
        mask_str = "".join(map(str, mask))
        full0 = [ln for ln in r0 if ln.startswith("n=49 ")]
        full1 = [ln for ln in r1 if ln.startswith("n=49 ")]
        texts = {p.name: p.read_text(errors="replace") for p in (logs / "server-check.log",
                                                                logs / "server-check-rank1.log") if p.is_file()}
        enable = {n: [m.group(1) for m in map(ENABLE_RE.search, t.splitlines()) if m] for n, t in texts.items()}
        stats_counts = [int(x) for x in STATS_LINE_RE.findall(texts.get("server-check.log", ""))]
        out.setdefault("_extra", {})[arm] = {
            "fn_ok": bool(r0) and bool(r1) and all(f"Fn={want_fn} Bn=0" in ln for ln in r0 + r1),
            "mask_ok": bool(full0) and bool(full1) and all(f"mask={mask_str}" in ln for ln in full0 + full1),
            "refresh_count": [len(r0), len(r1)],
            "enable_lines": enable,
            "taylorseer": {n: ("TaylorSeer enabled" in t) for n, t in texts.items()},
            "stats_accumulated_cached_rank0": stats_counts,
            "recompiles_rank0": recompile_scan(logs / "server-check.log"),
            "recompiles_rank1": recompile_scan(logs / "server-check-rank1.log"),
            "reset_sequence": ares.get("reset_sequence"),
            "post_denoise_s": [r.get("server_post_denoise_s") for r in (ares.get("requests") or [])],
            "denoise_s": [r.get("server_denoise_s") for r in (ares.get("requests") or [])],
            "e2e_s": [r.get("client_e2e_s") for r in (ares.get("requests") or [])],
        }
        out["arms"][arm] = {
            "yaml_keys": {"attention_backend": attn.get("backend"),
                          "quant_attention_config": attn.get("quant_attention_config"),
                          "step_precision": (yv or {}).get("step_precision"),
                          "parallel_config": (yv or {}).get("parallel_config")},
            "warmup_forward_summary_rank0": [x for x in f0 if x[1] == "True"],
            "warmup_forward_summary_rank1": [x for x in f1 if x[1] == "True"],
            "forward_summary_ranks_identical": bool(f0) and f0 == f1 if K_BF16 else None,
            "forward_phase_cache_ok": fwd_ok,
            "phase_timing": phase_timing(steps, mask, K_BF16),
            "warmup_step_lines": [r for r in steps if r and r[0][1] != 49],
            "oom_hits": oom[:20],
            "audio_snr_delta_vs_ref": {p["key"]: round(p["audio_snr_db"] - SNR_REF[p["key"]], 2)
                                       for p in qa.get("per_prompt", [])
                                       if p.get("audio_snr_db") is not None and p["key"] in SNR_REF},
            "cudnn_in_yaml": "CUDNN" in ytxt.upper(),
            "refresh_rank0": r0, "refresh_rank1": r1, "refresh_ranks_identical": bool(r0) and r0 == r1,
            "decisions_rank0": d0, "decisions_rank1": d1, "decisions_ranks_identical": bool(d0) and d0 == d1,
            "warmup_decision_rank0": d0[0] if d0 else None, "warmup_decision_rank1": d1[0] if d1 else None,
            "warmup_ok": bool(d0) and bool(d1) and d0[0].startswith("steps=2 cached=1")
            and d1[0].startswith("steps=2 cached=1"),
            "audio_snr_flags": snr_flags,
            "quality": {k: qa.get(k) for k in ("check_valid", "check_errors", "output_class",
                                               "repeated_execution_passed", "per_prompt", "repeat",
                                               "requests", "accuracy", "same_subject", "fidelity_pass",
                                               "server_cache_config", "floor_lpips")},
        }
        specs.append(f"{arm}:{arm}:{art / 'adapter_result.json'}:{logs / 'server-check.log'}")
    tbl_json = root / "arms_table.json"
    cmd = [sys.executable, str(HERE / "h3_cache_arms_table.py"), "--l2", L2, "--json-out", str(tbl_json)]
    for s in specs:
        cmd += ["--arm", s]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-12000:], proc.stderr[-3000:], flush=True)
    out["table_pick"] = None
    if tbl_json.is_file():
        tbl = json.loads(tbl_json.read_text())
        out["table_pick"] = tbl.get("pick")
        out["timing"] = {a["name"]: (a.get("timing") or {}).get("mean") for a in tbl["arms"]}
        out["table_quality"] = {a["name"]: {k: {kk: v.get(kk) for kk in ("lpips", "psnr", "audio_snr_db",
                                                                         "margin", "increment_over_L2", "passed")}
                                            for k, v in ((a.get("quality") or {}).get("prompts") or {}).items()}
                                for a in tbl["arms"]}
    def min_margin(a):
        rows = out["arms"][a]["quality"].get("per_prompt") or []
        vals = [r.get("margin_to_floor_plus_0.03") for r in rows]
        return min(vals) if rows and None not in vals else -1.0

    def passes(a):
        return (a in out["arms"] and bool(out["arms"][a]["quality"].get("fidelity_pass"))
                and len(out["arms"][a]["quality"].get("per_prompt") or []) == 3 and min_margin(a) >= 0.005)

    # Plan 736 order: F; then whichever of E*/U31 passes with the larger min margin; then D*; then periodic.
    mid = sorted([a for a in ("Es", "U31") if passes(a)], key=lambda a: -min_margin(a))
    order = ["F_F1", "U34_F1", "F"] + mid + ["Ds", "E", "D", "C", "fallback", "W"]
    passing = [a for a in order if a in arms and passes(a)]
    out["min_margin"] = {a: min_margin(a) for a in out["arms"]}
    from h3_cache_arms_table import LPIPS_EST
    keys = ("park_portrait", "perfume_bottle", "golden_robot")
    out["measured_over_est"] = {}
    for a in out["arms"]:
        est = LPIPS_EST.get(a)
        rows = {r["key"]: r.get("lpips") for r in out["arms"][a]["quality"].get("per_prompt") or []}
        if est:
            out["measured_over_est"][a] = {k: (round(rows[k] / e, 3) if rows.get(k) is not None else None)
                                           for k, e in zip(keys, est)}
    out["pick_most_aggressive_passing"] = passing[0] if passing else None
    mech_ok = all(v["warmup_ok"] and v["decisions_ranks_identical"] and v["refresh_ranks_identical"]
                  and not v["cudnn_in_yaml"] and v["quality"].get("repeated_execution_passed")
                  and (not K_BF16 or (v["forward_phase_cache_ok"] and v["forward_summary_ranks_identical"]
                                      and v["yaml_keys"]["attention_backend"] == "FLASHINFER"
                                      and (v["yaml_keys"]["step_precision"] or {}).get("attention_bf16_forwards")
                                      == K_BF16))
                  for v in out["arms"].values())
    out["mechanics_ok"] = mech_ok
    (root / "arms_summary.json").write_text(json.dumps(out, indent=1, default=str))
    small = {"passed": bool(mech_ok and passing), "pick": out["pick_most_aggressive_passing"],
             "mechanics_ok": mech_ok, "table_pick": out["table_pick"],
             "per_arm": {a: {"fidelity_pass": v["quality"].get("fidelity_pass"),
                             "repeat": v["quality"].get("repeated_execution_passed"),
                             "warmup_ok": v["warmup_ok"],
                             "ranks_identical": v["decisions_ranks_identical"] and v["refresh_ranks_identical"],
                             "lpips": {p["key"]: p.get("lpips") for p in v["quality"].get("per_prompt") or []},
                             "margin": {p["key"]: p.get("margin_to_floor_plus_0.03")
                                        for p in v["quality"].get("per_prompt") or []},
                             "snr_flags": v["audio_snr_flags"],
                             "snr_delta_vs_ref": v["audio_snr_delta_vs_ref"],
                             "fwd_phase_cache_ok": v["forward_phase_cache_ok"],
                             "yaml_keys": {k: v["yaml_keys"][k] for k in ("attention_backend",
                                                                           "quant_attention_config",
                                                                           "step_precision")},
                             "oom_hits": len(v["oom_hits"]),
                             "phase_timing": v["phase_timing"],
                             "warmup_fwd_rank0": v["warmup_forward_summary_rank0"],
                             "accuracy": (v["quality"].get("accuracy") or {}).get("correct")}
                         for a, v in out["arms"].items()},
             "min_margin": out["min_margin"], "measured_over_est": out["measured_over_est"],
             "timing": out.get("timing"),
             "extra": {a: {k: v for k, v in e.items() if k not in ("recompiles_rank0", "recompiles_rank1")}
                       | {"recompiles_rank0_n": len(e["recompiles_rank0"]),
                          "recompiles_rank1_n": len(e["recompiles_rank1"])}
                       for a, e in out.get("_extra", {}).items()}}
    print(json.dumps(small, default=str), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--post-only":  # --post-only ROOT ARM...
        _root = Path(sys.argv[2])
        _q = json.loads((_root / "quality.json").read_text()) if (_root / "quality.json").is_file() else {}
        post(_root, sys.argv[3:], _q.get("configs", {}))
    else:
        main()
