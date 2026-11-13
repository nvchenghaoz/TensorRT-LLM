#!/usr/bin/env python3
"""Per-arm LPIPS/PSNR/SNR + Step k/49 timing table for MiniMax-H3 Cache-DiT static-mask arms.

Pure Python (no torch). Stage for step_cache idea 56 (branch 27 / idea 70).

Inputs per arm (``--arm NAME:MASK:CHECK:LOG``):
  NAME   free label (E, D, C, fallback, W, w14_c2, main, ...)
  MASK   a name in MASKS below, or an explicit 0/1 string of 49 chars (1 = compute, index 0..48)
  CHECK  the check adapter's result (run_check.py JSON, or a stdout log whose last JSON line is it);
         '-' when absent. Read: comparisons.<case>.per_prompt[].{key,lpips,psnr,audio_snr_db}
  LOG    the rank-0 trtllm-serve log (formats below); '-' when absent.

Real log formats parsed (tensorrt_llm/_torch/visual_gen/pipeline.py):
  :1545  "Step {i+1}/{total_steps} | {step_time:.2f}s (trans=... cfg=... sched=...) | Avg=... ETA=..."
         1-based, so "Step k/49" is forward k-1 and is cached iff mask[k-1] == 0.
  :1553  "Denoising done: {total_time:.2f}s ({per_step:.2f}s/step)"   (ends one request)
  :1560  "Cache-DiT stats: {stats}"  stats = {"transformer_0": [CacheStats(..., cached_steps=[...],
         ..., accumulated_cached_steps=N, ...)]} (cache_dit 1.5.3 summary.CacheStats repr); comes after
         the request's "Denoising done" line.
Timed requests = requests whose Step lines say total == len(mask) (49); the 2-step warmup is skipped.

Quality columns: margin = bar - LPIPS (bars park 0.347 / perfume 0.166 / robot 0.508); increment over
L2 = LPIPS - L2's LPIPS (default job 292: 0.039 / 0.0198 / 0.0887); pass iff margin >= 0.005;
perfume headroom (ADDITIVE assumption) = bar_perfume - (perfume_other + increment_perfume) - 0.005
with perfume_other = 0.059.

Timing columns (timed requests only): dense/cached mean and median of the Step times; denoise total;
stats-line cached count vs the mask's zero count; predicted denoise = (49-k) x dense_mean + k x
cached_mean; seconds saved vs main (main's mean denoise minus the arm's) and per 0.01 LPIPS on the
binding prompt (the prompt with the smallest margin). Pick = first arm in order E, D, C, fallback, W
that passes every prompt.

  --self-test           validate on synthetic logs/JSON in the real formats (uses a real cache-dit
                        stats line from --stats-line-file when given, e.g. the toy job's output).
  --find-artifacts DIR  scan DIR recursively for server logs with Step lines and print a summary of
                        each (used to look for branch 2's jobs 260/261/302/292 artifacts).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from typing import Dict, List, Optional

N_FORWARDS = 49


def _mask(spec: List[int]) -> List[int]:
    return list(spec)


# 1 = compute, 0 = cached; index = forward 0..48 (Step k/49 -> index k-1).
MASKS: Dict[str, List[int]] = {
    "w14_c2": _mask([1] * 14 + [0, 0, 1] * 10 + [1] * 5),
    "C": _mask([1] * 8 + [1, 0, 0] * 12 + [1] * 5),
    "D": _mask([1] * 8 + [1, 0, 0, 0] * 9 + [1, 0] + [1] * 3),
    "E": _mask([1] * 8 + [1, 0, 0, 0, 0] * 7 + [1, 0, 0, 0] + [1] * 2),
    "fallback": _mask([1] * 10 + [1, 0, 0] * 11 + [1] * 6),
    "W": _mask([1] * 14 + [1, 0, 0, 0] * 8 + [1] * 3),
    "main": _mask([1] * N_FORWARDS),
}


def _from_bins(comp: List[int], cache: List[int]) -> List[int]:
    m: List[int] = []
    for i, c in enumerate(comp):
        m += [1] * c
        if i < len(cache):
            m += [0] * cache[i]
    return m


# Plan 736 (idea 83 DP masks, Fn8): string and bins must agree.
_DP = {
    "F": ("1111111100000000010000000001000000000100001001011", [8, 1, 1, 1, 1, 1, 2], [9, 9, 9, 4, 2, 1]),
    "Es": ("1111111100000000010000000010000001000010010101111", [8, 1, 1, 1, 1, 1, 1, 4],
           [9, 8, 6, 4, 2, 1, 1]),
    "Ds": ("1111111100000001000000100000100001001001010111111", [8, 1, 1, 1, 1, 1, 1, 1, 6],
           [7, 6, 5, 4, 2, 2, 1, 1]),
    "U31": ("1111111100100010001000100001000010000100001000011", [8, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 3, 3, 3, 4, 4, 4, 4, 4]),
    # Idea 113 (Fn=1 arms): F verbatim, and U34 (dense 0-7, 12, 17, 23, 30, 37, 43, 48).
    "F_F1": ("1111111100000000010000000001000000000100001001011", [8, 1, 1, 1, 1, 1, 2], [9, 9, 9, 4, 2, 1]),
    "U34_F1": ("1111111100001000010000010000001000000100000100001", [8, 1, 1, 1, 1, 1, 1, 1],
               [4, 4, 5, 6, 6, 5, 4]),
    # Idea 109 (Fn=1, uniform short-run masks over U34-F1): S2 U36, S3 W6-U36, S4 U38-W6, S1 W6-U40.
    "U36_F1": ("1111111100000010000000010000000010000000010000001", [8, 1, 1, 1, 1, 1], [6, 8, 8, 8, 6]),
    "W6U36_F1": ("1111110001000001000000100000010000001000000100001", [6, 1, 1, 1, 1, 1, 1, 1],
                 [3, 5, 6, 6, 6, 6, 4]),
    "U38W6_F1": ("1111110000001000000001000000000100000000010000001", [6, 1, 1, 1, 1, 1], [6, 8, 9, 9, 6]),
    "W6U40_F1": ("1111110000000000000100000000000000100000000000001", [6, 1, 1, 1], [13, 14, 13]),
}
for _n, (_s, _c, _k) in _DP.items():
    _m = [int(ch) for ch in _s]
    assert _m == _from_bins(_c, _k), _n
    MASKS[_n] = _m
EXPECTED_CACHED = {"w14_c2": 20, "C": 24, "D": 28, "E": 31, "fallback": 22, "W": 24, "main": 0,
                   "F": 34, "Es": 31, "Ds": 28, "U31": 31, "F_F1": 34, "U34_F1": 34,
                   "U36_F1": 36, "W6U36_F1": 36, "U38W6_F1": 38, "W6U40_F1": 40}
# max_warmup_steps per the brief (the explicit [1]*n prefix). The first compute RUN can be longer
# (C, D, E start their periods with a 1, so their first run is 9; fallback 11; W 15).
DENSE_HEAD = {"w14_c2": 14, "W": 14, "fallback": 10, "C": 8, "D": 8, "E": 8, "main": 49,
              "F": 8, "Es": 8, "Ds": 8, "U31": 8, "F_F1": 8, "U34_F1": 8,
              "U36_F1": 8, "W6U36_F1": 6, "U38W6_F1": 6, "W6U40_F1": 6}
FIRST_RUN = {"w14_c2": 14, "W": 15, "fallback": 11, "C": 9, "D": 9, "E": 9, "main": 49,
             "F": 8, "Es": 8, "Ds": 8, "U31": 8, "F_F1": 8, "U34_F1": 8,
             "U36_F1": 8, "W6U36_F1": 6, "U38W6_F1": 6, "W6U40_F1": 6}
# idea 83 LPIPS_est (park, perfume, robot) = LPIPS_292 x S/S(w14_c2); U31 from its uniform-weight S ratio.
LPIPS_EST = {"C": (0.048, 0.024, 0.107), "D": (0.076, 0.037, 0.162), "E": (0.127, 0.060, 0.257),
             "fallback": (0.041, 0.021, 0.092), "W": (0.086, 0.041, 0.181),
             "F": (0.126, 0.060, 0.254), "Es": (0.080, 0.039, 0.166), "Ds": (0.055, 0.027, 0.117),
             "U31": (0.078, 0.039, 0.177)}
PICK_ORDER = ["F", "Es", "U31", "Ds", "E", "D", "C", "fallback", "W"]

PROMPTS = {"park": "park_portrait", "perfume": "perfume_bottle", "robot": "golden_robot"}
BARS = {"park": 0.347, "perfume": 0.166, "robot": 0.508}
L2_JOB292 = {"park": 0.039, "perfume": 0.0198, "robot": 0.0887}
PASS_MARGIN = 0.005
PERFUME_OTHER = 0.059

STEP_RE = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+)s")
DONE_RE = re.compile(r"Denoising done: ([0-9.]+)s")
STATS_RE = re.compile(r"Cache-DiT stats: (.*)$")
CACHED_STEPS_RE = re.compile(r"(?<![A-Za-z_])cached_steps=\[([0-9,\s]*)\]")
ACC_CACHED_RE = re.compile(r"(?<![A-Za-z_])accumulated_cached_steps=(\d+)")


def check_masks() -> None:
    for name, m in MASKS.items():
        assert len(m) == N_FORWARDS, (name, len(m))
        assert m.count(0) == EXPECTED_CACHED[name], (name, m.count(0))
        head = 0
        for v in m:
            if v != 1:
                break
            head += 1
        assert head == FIRST_RUN[name] and head >= DENSE_HEAD[name], (name, head)
        assert m[-1] == 1, name


def resolve_mask(spec: str) -> List[int]:
    if spec in MASKS:
        return list(MASKS[spec])
    s = spec.strip()
    if set(s) <= {"0", "1"} and len(s) == N_FORWARDS:
        return [int(c) for c in s]
    raise ValueError(f"unknown mask {spec!r}: use one of {sorted(MASKS)} or a 49-char 0/1 string")


# ----------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------
def parse_server_log(text: str) -> List[dict]:
    """Split the rank-0 log into requests. Each: steps [(k, total, seconds)], denoise_s, stats."""
    reqs: List[dict] = []
    cur = {"steps": [], "denoise_s": None, "stats_cached_steps": None, "stats_count": None,
           "stats_raw": None}
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            if cur["denoise_s"] is not None:  # a new request begins after a finished one
                reqs.append(cur)
                cur = {"steps": [], "denoise_s": None, "stats_cached_steps": None,
                       "stats_count": None, "stats_raw": None}
            cur["steps"].append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
            continue
        m = DONE_RE.search(line)
        if m:
            cur["denoise_s"] = float(m.group(1))
            continue
        m = STATS_RE.search(line)
        if m:
            target = cur if cur["denoise_s"] is not None else (reqs[-1] if reqs else cur)
            raw = m.group(1)
            target["stats_raw"] = raw[:400]
            cs = CACHED_STEPS_RE.search(raw)
            if cs is not None:
                body = cs.group(1).strip()
                target["stats_cached_steps"] = [int(x) for x in body.split(",") if x.strip()]
            ac = ACC_CACHED_RE.search(raw)
            if ac is not None:
                target["stats_count"] = int(ac.group(1))
            elif target["stats_cached_steps"] is not None:
                target["stats_count"] = len(target["stats_cached_steps"])
    if cur["steps"] or cur["denoise_s"] is not None:
        reqs.append(cur)
    return reqs


def load_check(path: str) -> Optional[dict]:
    """Return {prompt_key: {lpips, psnr, audio_snr_db, passed}} from a check JSON or stdout log."""
    if not path or path == "-":
        return None
    with open(path) as f:
        text = f.read()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    if obj is None:
        raise ValueError(f"no JSON object in {path}")
    if "result" in obj and isinstance(obj["result"], dict) and "comparisons" not in obj:
        obj = obj["result"]
    out = {}
    for case in (obj.get("comparisons") or {}).values():
        for rec in case.get("per_prompt", []):
            out[rec["key"]] = {"lpips": rec.get("lpips"), "psnr": rec.get("psnr"),
                               "audio_snr_db": rec.get("audio_snr_db"), "passed": rec.get("passed"),
                               "bitwise_equal": rec.get("bitwise_equal")}
    rep = obj.get("repeated_execution")
    if rep is not None:
        out["_repeat_passed"] = obj.get("repeated_execution_passed")
    return out


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
def _mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def timing_for_arm(reqs: List[dict], mask: List[int], timed: Optional[List[int]] = None) -> dict:
    total = len(mask)
    k_mask = mask.count(0)
    timed_reqs = [r for r in reqs if r["steps"] and r["steps"][0][1] == total]
    if timed is not None:
        timed_reqs = [timed_reqs[i] for i in timed if i < len(timed_reqs)]
    per_req = []
    for r in timed_reqs:
        dense = [t for (k, n, t) in r["steps"] if mask[k - 1] == 1]
        cached = [t for (k, n, t) in r["steps"] if mask[k - 1] == 0]
        n_steps = len(r["steps"])
        rec = {
            "n_steps": n_steps,
            "dense_n": len(dense), "dense_mean": _mean(dense), "dense_median": _median(dense),
            "cached_n": len(cached), "cached_mean": _mean(cached), "cached_median": _median(cached),
            "denoise_s": r["denoise_s"],
            "stats_count": r["stats_count"],
            "stats_cached_steps": r["stats_cached_steps"],
            "stats_matches_mask": (r["stats_count"] == k_mask) if r["stats_count"] is not None else None,
            "stats_set_matches_mask": (sorted(r["stats_cached_steps"]) == [i for i, v in enumerate(mask) if v == 0])
            if r["stats_cached_steps"] is not None else None,
        }
        dm = rec["dense_mean"]
        cm = rec["cached_mean"] if cached else 0.0
        rec["predicted_denoise_s"] = (total - k_mask) * dm + k_mask * cm
        per_req.append(rec)
    agg = {}
    if per_req:
        for key in ("dense_mean", "cached_mean", "denoise_s", "predicted_denoise_s"):
            vals = [p[key] for p in per_req if p[key] is not None and not math.isnan(p[key])]
            agg[key] = _mean(vals) if vals else None
    return {"k_mask": k_mask, "n_timed": len(per_req), "per_request": per_req, "mean": agg}


def quality_for_arm(check: Optional[dict], bars=BARS, l2=L2_JOB292, perfume_other=PERFUME_OTHER,
                    pass_margin=PASS_MARGIN) -> Optional[dict]:
    if check is None:
        return None
    rows = {}
    for short, key in PROMPTS.items():
        rec = check.get(key)
        if rec is None or rec.get("lpips") is None:
            rows[short] = {"missing": True, "passed": False}
            continue
        lp = float(rec["lpips"])
        margin = bars[short] - lp
        rows[short] = {"lpips": lp, "psnr": rec.get("psnr"), "audio_snr_db": rec.get("audio_snr_db"),
                       "bar": bars[short], "margin": margin, "increment_over_L2": lp - l2[short],
                       "passed": margin >= pass_margin, "check_passed": rec.get("passed")}
    if "perfume" in rows and not rows["perfume"].get("missing"):
        inc = rows["perfume"]["increment_over_L2"]
        rows["perfume"]["headroom_additive"] = bars["perfume"] - (perfume_other + inc) - pass_margin
    present = {k: v for k, v in rows.items() if not v.get("missing")}
    binding = min(present, key=lambda k: present[k]["margin"]) if present else None
    return {"prompts": rows, "binding": binding,
            "all_pass": bool(present) and len(present) == len(PROMPTS) and all(v["passed"] for v in present.values()),
            "repeat_passed": check.get("_repeat_passed")}


def build_table(arms: List[dict], main_name: Optional[str] = None, main_lpips: Optional[dict] = None) -> dict:
    by_name = {a["name"]: a for a in arms}
    main = by_name.get(main_name) if main_name else None
    main_denoise = main["timing"]["mean"].get("denoise_s") if main and main.get("timing") else None
    for a in arms:
        q, t = a.get("quality"), a.get("timing")
        a["saved_vs_main_s"] = None
        a["saved_per_0.01_lpips_binding"] = None
        if t and t["mean"].get("denoise_s") is not None and main_denoise is not None:
            a["saved_vs_main_s"] = main_denoise - t["mean"]["denoise_s"]
        if q and q["binding"] and a["saved_vs_main_s"] is not None:
            b = q["binding"]
            base = (main_lpips or {}).get(b, 0.0)
            d = q["prompts"][b]["lpips"] - base
            a["saved_per_0.01_lpips_binding"] = a["saved_vs_main_s"] / (d / 0.01) if d > 0 else None
    pick = None
    for name in PICK_ORDER:
        a = by_name.get(name)
        if a and a.get("quality") and a["quality"]["all_pass"]:
            pick = name
            break
    return {"arms": arms, "pick": pick, "pick_order": PICK_ORDER}


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


def render(table: dict) -> str:
    lines = []
    lines.append("## Quality (per arm, per prompt)")
    lines.append("| arm | prompt | LPIPS | PSNR | audio SNR | bar | margin | +over L2 | pass(>=0.005) | perfume headroom (additive) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in table["arms"]:
        q = a.get("quality")
        if not q:
            lines.append(f"| {a['name']} | (no check) | | | | | | | | |")
            continue
        for short in PROMPTS:
            r = q["prompts"][short]
            if r.get("missing"):
                lines.append(f"| {a['name']} | {short} | missing | | | | | | NO | |")
                continue
            lines.append(
                f"| {a['name']} | {short}{' *' if q['binding'] == short else ''} | {fmt(r['lpips'], 4)} | "
                f"{fmt(r['psnr'], 2)} | {fmt(r['audio_snr_db'], 2)} | {r['bar']} | {fmt(r['margin'], 4)} | "
                f"{fmt(r['increment_over_L2'], 4)} | {fmt(r['passed'])} | {fmt(r.get('headroom_additive'), 4)} |")
    lines.append("(* = binding prompt, smallest margin)")
    lines.append("")
    lines.append("## Timing (timed requests only, Step k/49 -> mask[k-1])")
    lines.append("| arm | req | dense n | dense mean | dense med | cached n | cached mean | cached med | denoise s | stats count / mask | predicted s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in table["arms"]:
        t = a.get("timing")
        if not t:
            lines.append(f"| {a['name']} | (no log) | | | | | | | | | |")
            continue
        for i, p in enumerate(t["per_request"]):
            lines.append(
                f"| {a['name']} | {i} | {p['dense_n']} | {fmt(p['dense_mean'])} | {fmt(p['dense_median'])} | "
                f"{p['cached_n']} | {fmt(p['cached_mean'])} | {fmt(p['cached_median'])} | {fmt(p['denoise_s'], 1)} | "
                f"{fmt(p['stats_count'])} / {t['k_mask']} | {fmt(p['predicted_denoise_s'], 1)} |")
    lines.append("")
    lines.append("## Summary")
    lines.append("| arm | cached k | mean denoise s | predicted s | saved vs main s | s saved per 0.01 LPIPS (binding) | all pass |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in table["arms"]:
        t = a.get("timing") or {"mean": {}, "k_mask": a["mask"].count(0)}
        q = a.get("quality")
        lines.append(f"| {a['name']} | {t['k_mask']} | {fmt(t['mean'].get('denoise_s'), 1)} | "
                     f"{fmt(t['mean'].get('predicted_denoise_s'), 1)} | {fmt(a.get('saved_vs_main_s'), 1)} | "
                     f"{fmt(a.get('saved_per_0.01_lpips_binding'), 1)} | {fmt(q['all_pass']) if q else '-'} |")
    lines.append(f"\nPick (order {', '.join(table['pick_order'])}): {table['pick']}")
    return "\n".join(lines)


def run_arms(arm_specs: List[str], main_name: Optional[str], timed: Optional[List[int]] = None,
             bars=BARS, l2=L2_JOB292, perfume_other=PERFUME_OTHER) -> dict:
    arms = []
    for spec in arm_specs:
        name, mask_spec, check_path, log_path = spec.split(":", 3)
        mask = resolve_mask(mask_spec)
        a = {"name": name, "mask": mask}
        a["quality"] = quality_for_arm(load_check(check_path), bars=bars, l2=l2, perfume_other=perfume_other)
        if log_path and log_path != "-":
            with open(log_path, errors="replace") as f:
                a["timing"] = timing_for_arm(parse_server_log(f.read()), mask, timed)
        else:
            a["timing"] = None
        arms.append(a)
    return build_table(arms, main_name)


# ----------------------------------------------------------------------------------------------
# Synthetic validation in the real formats
# ----------------------------------------------------------------------------------------------
FALLBACK_STATS_TMPL = (
    "{{'transformer_0': [CacheStats(cache_options={{'name': 'transformer_blocks_1', 'cache_config': "
    "DBCacheConfig(cache_type=<CacheType.DBCache: 'DBCache'>, Fn_compute_blocks=8, Bn_compute_blocks=0, "
    "residual_diff_threshold=0.24, max_accumulated_residual_diff_threshold=None, max_warmup_steps=8, "
    "warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=-1, enable_separate_cfg=False, "
    "cfg_compute_first=False, cfg_diff_compute_separate=True, num_inference_steps=49, "
    "steps_computation_mask=[...], steps_computation_policy='static', force_refresh_step_hint=None, "
    "force_refresh_step_policy='once')}}, cached_steps={cached}, residual_diffs={{}}, "
    "cfg_cached_steps=[], cfg_residual_diffs={{}}, accumulated_cached_steps={acc}, "
    "cfg_accumulated_cached_steps=0, accumulated_executed_steps=49, "
    "accumulated_transformer_executed_steps=49, pruned_steps=[], pruned_blocks=[], actual_blocks=[], "
    "pruned_ratio=None, cfg_pruned_steps=[], cfg_pruned_blocks=[], cfg_actual_blocks=[], "
    "cfg_pruned_ratio=None, parallelism_config=None, quantize_config=None)]}}")

PFX = "[11/12/2026-15:00:00] [TRT-LLM] [I] "


def synth_request(mask: List[int], dense: List[float], cached: List[float], stats_line: Optional[str]) -> List[str]:
    """Emit one request's rank-0 lines exactly as pipeline.py:1545/1553/1560 format them."""
    lines = []
    total = len(mask)
    di = ci = 0
    times = []
    for i in range(total):
        if mask[i] == 1:
            t = dense[di % len(dense)]
            di += 1
        else:
            t = cached[ci % len(cached)]
            ci += 1
        times.append(t)
    acc = 0.0
    for i, t in enumerate(times):
        acc += t
        avg = acc / (i + 1)
        eta = avg * (total - i - 1)
        lines.append(PFX + f"Step {i + 1}/{total} | {t:.2f}s (trans={t - 0.02:.2f}s cfg=0.000s sched=0.001s) | "
                           f"Avg={avg:.2f}s/step ETA={eta:.1f}s")
    lines.append(PFX + "=" * 80)
    lines.append(PFX + f"Denoising done: {acc:.2f}s ({acc / total:.2f}s/step)")
    if stats_line is not None:
        lines.append(PFX + f"Cache-DiT stats: {stats_line}")
    return lines


def synth_check(lp: Dict[str, float]) -> dict:
    """A run_check.py result in its real shape (finite_json'd)."""
    per = []
    for short, key in PROMPTS.items():
        per.append({"ref_frames": 124, "cand_frames": 124, "sheet": f"{key}.jpg", "bitwise_equal": False,
                    "psnr": 30.0 - 10 * lp[short], "lpips": lp[short], "audio_snr_db": 20.0 - lp[short],
                    "key": key, "passed": False, "lpips_tol": 0.01, "psnr_tol": 45.0})
    return {"adapter": "check", "engine_overlay": {}, "reference_independent": True,
            "output": {"class": "precision"}, "ffmpeg": {}, "startup_s": 300.0,
            "comparisons": {"1": {"per_prompt": per}}, "repeated_execution_passed": True,
            "repeated_execution": [{"key": "park_portrait", "passed": True}], "tolerance": {},
            "variation_pairs": [], "request_observations": {}, "repeat_observations": {}, "requests": [],
            "execution": {}, "artifacts": {"server_log": "x"}, "errors": [], "valid": True}


def self_test(outdir: str, stats_lines: Dict[str, str]) -> dict:
    os.makedirs(outdir, exist_ok=True)
    check_masks()
    results = {}

    def stats_for(name: str, mask: List[int]) -> str:
        if name in stats_lines:
            return stats_lines[name]
        cached = [i for i, v in enumerate(mask) if v == 0]
        return FALLBACK_STATS_TMPL.format(cached=cached, acc=len(cached))

    # 1) Reproduce job 302's published numbers from synthetic logs built with those per-request means.
    # base_260 (no cache): dense means 6.369 / 6.349 / 6.205 -> constant step times reproduce the mean
    # only to 2 decimals, so use a two-value pattern whose 2-decimal mean is exact.
    def two_val(mean: float, n: int) -> List[float]:
        # choose a, b at 0.01 resolution and counts so that (na*a + nb*b)/n == mean (to 1e-9 if feasible)
        target = round(mean * n, 2)
        lo = math.floor(mean * 100) / 100
        hi = lo + 0.01
        nb = round((target - lo * n) / 0.01)
        nb = max(0, min(n, nb))
        return [hi] * nb + [lo] * (n - nb)

    main_lines = [PFX + "Step 1/2 | 0.90s (trans=0.88s cfg=0.000s sched=0.001s) | Avg=0.90s/step ETA=0.9s",
                  PFX + "Step 2/2 | 0.90s (trans=0.88s cfg=0.000s sched=0.001s) | Avg=0.90s/step ETA=0.0s",
                  PFX + "Denoising done: 1.80s (0.90s/step)"]  # 2-step warmup, must be skipped
    base_targets = [6.369, 6.349, 6.205]
    for m in base_targets:
        main_lines += synth_request(MASKS["main"], two_val(m, 49), [1.0], None)
    cache_dense = [6.223, 6.223, 6.313]
    cache_cached = [1.000, 1.006, 1.028]
    c_lines = list(main_lines[:3])
    mw = MASKS["w14_c2"]
    for d, c in zip(cache_dense, cache_cached):
        c_lines += synth_request(mw, two_val(d, 29), two_val(c, 20), stats_for("w14_c2", mw))
    p_main = os.path.join(outdir, "synthetic_base_260.log")
    p_cache = os.path.join(outdir, "synthetic_cache_261.log")
    with open(p_main, "w") as f:
        f.write("\n".join(main_lines) + "\n")
    with open(p_cache, "w") as f:
        f.write("\n".join(c_lines) + "\n")
    tm = timing_for_arm(parse_server_log(open(p_main).read()), MASKS["main"])
    tc = timing_for_arm(parse_server_log(open(p_cache).read()), mw)
    got = {
        "base_dense_means": [round(p["dense_mean"], 3) for p in tm["per_request"]],
        "cache_dense_means": [round(p["dense_mean"], 3) for p in tc["per_request"]],
        "cache_dense_n": [p["dense_n"] for p in tc["per_request"]],
        "cache_cached_means": [round(p["cached_mean"], 3) for p in tc["per_request"]],
        "cache_cached_n": [p["cached_n"] for p in tc["per_request"]],
        "cache_stats_counts": [p["stats_count"] for p in tc["per_request"]],
        "cache_stats_set_ok": [p["stats_set_matches_mask"] for p in tc["per_request"]],
        "cache_denoise": [p["denoise_s"] for p in tc["per_request"]],
        "base_n_timed": tm["n_timed"], "cache_n_timed": tc["n_timed"],
    }
    exp_denoise = [round(29 * d + 20 * c, 1) for d, c in zip(cache_dense, cache_cached)]
    ok302 = (got["base_dense_means"] == base_targets and got["cache_dense_means"] == cache_dense
             and got["cache_cached_means"] == cache_cached and got["cache_dense_n"] == [29] * 3
             and got["cache_cached_n"] == [20] * 3 and got["cache_stats_counts"] == [20] * 3
             and all(got["cache_stats_set_ok"]) and got["base_n_timed"] == 3 and got["cache_n_timed"] == 3
             and [round(x, 1) for x in got["cache_denoise"]] == exp_denoise)
    results["job302_shape_reproduction"] = {"ok": ok302, "got": got,
                                            "expected": {"base_dense": base_targets, "cache_dense": cache_dense,
                                                         "cache_cached": cache_cached,
                                                         "denoise(29d+20c)": exp_denoise,
                                                         "published_denoise_302": [200.5, 200.6, 203.6]}}

    # 2) Full table on synthetic arms (E, D, C, fallback, W, w14_c2 vs main) with known LPIPS.
    lp = {
        "main": {"park": 0.0, "perfume": 0.0, "robot": 0.0},
        "w14_c2": {"park": 0.039, "perfume": 0.0198, "robot": 0.0887},
        "E": {"park": 0.30, "perfume": 0.163, "robot": 0.40},     # perfume margin 0.003 -> fails
        "D": {"park": 0.25, "perfume": 0.150, "robot": 0.35},     # passes -> pick D
        "C": {"park": 0.20, "perfume": 0.120, "robot": 0.30},
        "fallback": {"park": 0.15, "perfume": 0.100, "robot": 0.25},
        "W": {"park": 0.10, "perfume": 0.080, "robot": 0.20},
    }
    specs = []
    for name, l in lp.items():
        mask = MASKS[name]
        cp = os.path.join(outdir, f"synthetic_check_{name}.json")
        with open(cp, "w") as f:
            # emulate the adapter's stdout: progress lines, then the JSON on the last line
            f.write("progress line\n" + json.dumps(synth_check(l)) + "\n")
        lines = list(main_lines[:3])
        for _ in range(3):
            lines += synth_request(mask, [6.20, 6.30], [1.00, 1.02],
                                   None if name == "main" else stats_for(name, mask))
        lp_path = os.path.join(outdir, f"synthetic_server_{name}.log")
        with open(lp_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        specs.append(f"{name}:{name}:{cp}:{lp_path}")
    table = run_arms(specs, "main")
    by = {a["name"]: a for a in table["arms"]}
    # hand-computed expectations
    e = by["E"]
    exp_margin_E_perfume = 0.166 - 0.163
    exp_head_D = 0.166 - (0.059 + (0.150 - 0.0198)) - 0.005
    k_E = 31
    exp_dense = statistics.fmean([6.20, 6.30] * 9)  # 18 dense in E, alternating from 6.20
    checks = {
        "E_fails_on_perfume": (not e["quality"]["all_pass"]) and e["quality"]["binding"] == "perfume"
        and abs(e["quality"]["prompts"]["perfume"]["margin"] - exp_margin_E_perfume) < 1e-12,
        "pick_is_D": table["pick"] == "D",
        "D_perfume_headroom_additive": abs(by["D"]["quality"]["prompts"]["perfume"]["headroom_additive"] - exp_head_D) < 1e-12,
        "E_cached_n": all(p["cached_n"] == k_E and p["dense_n"] == 49 - k_E for p in e["timing"]["per_request"]),
        "E_stats_ok": all(p["stats_matches_mask"] and p["stats_set_matches_mask"] for p in e["timing"]["per_request"]),
        "E_dense_mean": abs(e["timing"]["per_request"][0]["dense_mean"] - exp_dense) < 1e-9,
        "E_predicted": abs(e["timing"]["per_request"][0]["predicted_denoise_s"]
                           - ((49 - k_E) * e["timing"]["per_request"][0]["dense_mean"]
                              + k_E * e["timing"]["per_request"][0]["cached_mean"])) < 1e-9,
        "all_arms_stats_ok": all(all(p["stats_matches_mask"] for p in a["timing"]["per_request"])
                                 for a in table["arms"] if a["name"] != "main"),
        "main_timed_3": by["main"]["timing"]["n_timed"] == 3,
        "saved_positive_for_E": by["E"]["saved_vs_main_s"] > 0,
        "w14_c2_increment_zero": all(abs(by["w14_c2"]["quality"]["prompts"][s]["increment_over_L2"]) < 1e-12 for s in PROMPTS),
    }
    results["synthetic_table_checks"] = checks
    results["synthetic_table_markdown"] = render(table)
    # a mismatching stats line must be flagged
    bad = synth_request(MASKS["E"], [6.2], [1.0], FALLBACK_STATS_TMPL.format(cached=list(range(30)), acc=30))
    tb = timing_for_arm(parse_server_log("\n".join(bad)), MASKS["E"])
    results["mismatch_flagged"] = tb["per_request"][0]["stats_matches_mask"] is False
    results["used_real_stats_lines"] = sorted(stats_lines)
    results["passed"] = bool(ok302 and all(checks.values()) and results["mismatch_flagged"])
    return results


def find_jobs(roots: List[str], ids: List[int]) -> dict:
    """Map runner job directories to campaign job ids via their manifest.json (schema unknown: regex)."""
    out = {"manifests_seen": 0, "matches": [], "sample_manifest": None}
    pats = {jid: re.compile(r'"[A-Za-z_]*(?:job_id|jobid|job)"\s*:\s*"?%d"?\b' % jid) for jid in ids}
    for root in roots:
        cands = glob.glob(os.path.join(root, "slots", "*", "jobs", "*", "manifest.json")) + \
            glob.glob(os.path.join(root, "jobs", "*", "manifest.json"))
        for mp in cands:
            try:
                text = open(mp, errors="replace").read()
            except Exception:
                continue
            out["manifests_seen"] += 1
            if out["sample_manifest"] is None:
                out["sample_manifest"] = {"path": mp, "head": text[:1500]}
            for jid, pat in pats.items():
                if pat.search(text):
                    d = os.path.dirname(mp)
                    logs = [p for p in glob.glob(os.path.join(d, "**", "*.log"), recursive=True)]
                    jsons = [p for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)][:50]
                    out["matches"].append({"job": jid, "dir": d, "logs": logs[:50], "jsons": jsons})
    return out


def find_artifacts(root: str, limit: int = 4000) -> List[dict]:
    """Look for trtllm-serve logs with Step/Cache-DiT lines under root (bounded)."""
    found = []
    n = 0
    for path in glob.iglob(os.path.join(root, "**", "*.log"), recursive=True):
        n += 1
        if n > limit:
            break
        try:
            if os.path.getsize(path) > 200 * 1024 * 1024:
                continue
            with open(path, errors="replace") as f:
                text = f.read()
        except Exception:
            continue
        if "Denoising done" not in text:
            continue
        reqs = parse_server_log(text)
        found.append({"path": path, "n_requests": len(reqs),
                      "totals": [r["steps"][0][1] if r["steps"] else None for r in reqs],
                      "denoise": [r["denoise_s"] for r in reqs],
                      "stats_counts": [r["stats_count"] for r in reqs]})
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="NAME:MASK:CHECK_JSON:SERVER_LOG")
    ap.add_argument("--main", default="main", help="arm name used as 'main' for seconds saved")
    ap.add_argument("--timed", default=None, help="comma list of timed-request indices to keep (default all 49-step)")
    ap.add_argument("--l2", default=None, help="park=,perfume=,robot= LPIPS of L2 (default job 292)")
    ap.add_argument("--bars", default=None, help="park=,perfume=,robot= LPIPS bars")
    ap.add_argument("--perfume-other", type=float, default=PERFUME_OTHER)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--outdir", default=os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp") + "/arms_table")
    ap.add_argument("--stats-line-file", action="append", default=[], help="NAME=path with a real 'Cache-DiT stats:' payload")
    ap.add_argument("--find-artifacts", default=None)
    args = ap.parse_args(argv)

    def kv(s, default):
        if not s:
            return dict(default)
        out = dict(default)
        for part in s.split(","):
            k, v = part.split("=")
            out[k.strip()] = float(v)
        return out

    if args.self_test:
        stats_lines = {}
        for spec in args.stats_line_file:
            name, path = spec.split("=", 1)
            if os.path.exists(path):
                stats_lines[name] = open(path).read().strip()
        res = self_test(args.outdir, stats_lines)
        print(res["synthetic_table_markdown"])
        res2 = {k: v for k, v in res.items() if k != "synthetic_table_markdown"}
        print(json.dumps(res2, indent=1, default=str))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(res, f, indent=1, default=str)
        print(json.dumps({"passed": res["passed"], "self_test": True}))
        return 0 if res["passed"] else 1
    if args.find_artifacts:
        roots = [r for r in args.find_artifacts.split(",") if r]
        jobs = find_jobs(roots, [260, 261, 302, 292])
        print(json.dumps(jobs, indent=1)[:20000])
        for m in jobs["matches"]:
            for lp in m["logs"]:
                try:
                    text = open(lp, errors="replace").read()
                except Exception:
                    continue
                if "Denoising done" in text:
                    reqs = parse_server_log(text)
                    print(json.dumps({"job": m["job"], "log": lp, "n_requests": len(reqs),
                                      "totals": [r["steps"][0][1] if r["steps"] else None for r in reqs],
                                      "denoise": [r["denoise_s"] for r in reqs],
                                      "stats_counts": [r["stats_count"] for r in reqs]}))
        return 0
    timed = [int(x) for x in args.timed.split(",")] if args.timed else None
    table = run_arms(args.arm, args.main, timed, bars=kv(args.bars, BARS), l2=kv(args.l2, L2_JOB292),
                     perfume_other=args.perfume_other)
    print(render(table))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(table, f, indent=1, default=str)
    print(json.dumps({"pick": table["pick"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
