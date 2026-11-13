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

L2 = "park=0.0390,perfume=0.0198,robot=0.0887"  # landing 10 check, job 371
SNR_FLAG = {"park_portrait": 1.74, "perfume_bottle": 4.60, "golden_robot": 7.47}
HEAD = {"E": 8, "D": 8, "C": 8, "fallback": 10, "W": 14, "w14_c2": 14, "F": 8, "Es": 8, "Ds": 8, "U31": 8}
MCC = {"w14_c2": 2}


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
    return {"cache_backend": "cache_dit", "Fn_compute_blocks": 8, "Bn_compute_blocks": 0,
            "max_warmup_steps": HEAD[arm], "max_continuous_cached_steps": MCC.get(arm, -1),
            "residual_diff_threshold": 0.08, "scm_compute_bins": comp, "scm_cache_bins": cache,
            "scm_steps_policy": "static"}


REFRESH_RE = re.compile(r"Cache-DiT refresh rank=\d+: (.*)$")
DEC_RE = re.compile(r"Cache-DiT decisions rank=\d+: context=\S+ (steps=\d+ cached=\d+ cached_steps=\[[^\]]*\])")


def grep(path: Path, rx):
    if not path.is_file():
        return []
    return [m.group(1) for ln in path.read_text(errors="replace").splitlines() for m in [rx.search(ln)] if m]


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
    sys.argv = [sys.argv[0], ckpt] + arms
    q.main()

    quality = captured.get("quality", {}).get("configs", {})
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
        out["arms"][arm] = {
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
    order = ["F"] + mid + ["Ds", "E", "D", "C", "fallback", "W"]
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
                             "accuracy": (v["quality"].get("accuracy") or {}).get("correct")}
                         for a, v in out["arms"].items()},
             "min_margin": out["min_margin"], "measured_over_est": out["measured_over_est"],
             "timing": out.get("timing")}
    print(json.dumps(small, default=str), flush=True)


if __name__ == "__main__":
    main()
