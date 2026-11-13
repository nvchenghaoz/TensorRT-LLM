#!/usr/bin/env python3
"""Idea 141 (Stage 3 of idea 63): 2-GPU fidelity arms for the Sage2 middle tier.

Usage: python stage3_arms.py <checkpoint> <arm> [<arm> ...]      arms: ks4 ks6 sq main

Each arm = main's committed engine.json VERBATIM plus a step_precision delta, one trtllm-serve server
on the job's 2 GPUs running the frozen experiments/ascent-benchmark/run_check.py (park, perfume, robot,
then park again in the same server). Deltas:
  ks4  attention_sage_onset=4                      (BF16 0-3, Sage2 4-13, main's quantized recipe 14-48)
  ks6  attention_sage_onset=6                      (BF16 0-5, Sage2 6-13, quantized 14-48)
  sq   attention_bf16_forwards=49, sage_onset=4    (BF16 0-3, Sage2 on every forward >= 4; never submitted)
  main no delta
After all servers stop: the package's Qwen3-VL judge (21 items, same-subject at frames 25/62/98 -> 9
stills per arm), per-rank forward_summary routing check against the tier table and main's Cache-DiT mask,
Step-time per tier (request 1 vs park repeat), OOM / Recompiling scan, and new Triton cache entries
created after the first timed request started (TRITON_CACHE_DIR is private per arm).
PASS line (mode_switch, coach note 311): per prompt margin_to_floor_plus_0.03 >= .005.
Last stdout line: JSON.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASC = HERE.parent
sys.path.insert(0, str(ASC))
import h3_quality_job as q  # noqa: E402

BENCH = ASC.parent / "ascent-benchmark"
ENGINE = ASC.parent / "ascent-config" / "engine.json"
NUM = 49
# Main's U34-F1 FP8 LPIPS (jobs 629/686) for the net arithmetic.
MAIN_LPIPS = {"park_portrait": 0.2444, "perfume_bottle": 0.0700, "golden_robot": 0.2775}
MS_PER_CALL = 22.0          # BF16 55.39 -> Sage2 33.36 ms (pair 790/791)
SQ_EXTRA_MS = 33.36 - 30.5  # Sage2 vs main's FP8 recipe on the quantized phase
STEP_RE = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+)s")
FWD_RE = re.compile(r"\[step_precision\] rank=(\d+) forward_summary (\d+)/(\d+) warmup=(\w+) attention=(\S+) "
                    r".*?cache=(\w+) attention_calls=(\d+)/(\d+) fa2_sm120_calls=(\d+)")
OOM_RE = re.compile(r"(failed with OOM|out of memory|CUDA error)[^\n]*", re.IGNORECASE)
RECOMP_RE = re.compile(r"Recompiling[^\n]*")
TIER_LINE_RE = re.compile(r"\[step_precision\] rank=\d+ attention_sage_onset=[^\n]*")


def arm_engine(original: dict, arm: str) -> dict:
    eng = copy.deepcopy(original)
    sp = eng.setdefault("visual_gen_args", {}).setdefault("step_precision", {})
    if arm == "ks4":
        sp["attention_sage_onset"] = 4
    elif arm == "ks6":
        sp["attention_sage_onset"] = 6
    elif arm == "sq":
        sp["attention_bf16_forwards"] = NUM
        sp["attention_sage_onset"] = 4
    elif arm != "main":
        raise SystemExit(f"unknown arm {arm}")
    return eng


def mask_of(cache_cfg: dict) -> list[int]:
    comp, cache = cache_cfg["scm_compute_bins"], cache_cfg["scm_cache_bins"]
    m = []
    for i, c in enumerate(comp):
        m += [1] * c
        if i < len(cache):
            m += [0] * cache[i]
    assert len(m) == NUM, m
    return m


def tiers(sp: dict) -> list[str]:
    k = sp.get("attention_bf16_forwards") or 0
    ks = sp.get("attention_sage_onset")
    out = []
    for i in range(NUM):
        if i < k and ks is not None and i >= ks:
            out.append("sage2-int8qk-fp8pv")
        elif i < k:
            out.append("bf16-fa2")
        else:
            out.append("fp8")
    return out


def direct_seconds(sp: dict, mask: list[int]) -> dict:
    t = tiers(sp)
    calls = lambda i: 50 if mask[i] else 1  # noqa: E731  Fn=1: a cached forward runs block 0 only
    bf16_moved = sum(calls(i) for i in range(NUM) if t[i] == "sage2-int8qk-fp8pv" and i < 14)
    quant_moved = sum(calls(i) for i in range(NUM) if t[i] == "sage2-int8qk-fp8pv" and i >= 14)
    s = bf16_moved * MS_PER_CALL / 1000 - quant_moved * SQ_EXTRA_MS / 1000
    return {"bf16_calls_moved": bf16_moved, "quant_calls_moved": quant_moved, "direct_s": round(s, 2)}


def run_check(h: dict, art: Path, triton_dir: Path) -> tuple[dict, dict]:
    """Frozen run_check.py with a private Triton cache; snapshot the cache when pass 1 starts."""
    env = dict(os.environ, AUTO_DEV_ARTIFACTS=str(art), TRITON_CACHE_DIR=str(triton_dir))
    triton_dir.mkdir(parents=True, exist_ok=True)
    snap = {"at_pass1": None, "at_end": None}

    def listing():
        return sorted(str(p.relative_to(triton_dir)) for p in triton_dir.rglob("*") if p.is_file())

    proc = subprocess.Popen([sys.executable, str(BENCH / "run_check.py"), json.dumps(h)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = None
    for line in proc.stdout:
        if "bench pass1:" in line and snap["at_pass1"] is None:
            snap["at_pass1"] = listing()
            sys.stdout.write(f"[stage3] triton cache files at pass1 start: {len(snap['at_pass1'])}\n")
        if line.startswith("{") and '"adapter"' in line:
            last = line
            sys.stdout.write(line[:2000] + "\n")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    snap["at_end"] = listing()
    new = sorted(set(snap["at_end"]) - set(snap["at_pass1"] or []))
    jit = {"files_at_pass1": len(snap["at_pass1"] or []), "files_at_end": len(snap["at_end"]),
           "new_after_pass1_start": [n for n in new if n.endswith(".json") and "__grp__" not in n][:40],
           "new_count": len(new)}
    return (json.loads(last) if last else {"valid": False, "errors": ["no check result"]}), jit


def rank_logs(art: Path):
    logs = art / "logs"
    return [logs / "server-check.log", logs / "server-check-rank1.log"]


def fwd_lines(path: Path):
    if not path.is_file():
        return []
    return [{"i": int(m.group(2)), "warmup": m.group(4) == "True", "attention": m.group(5),
             "cache": m.group(6), "calls": int(m.group(7)), "fa2": int(m.group(9))}
            for ln in path.read_text(errors="replace").splitlines() for m in [FWD_RE.search(ln)] if m]


def requests_of(lines):
    reqs, cur = [], []
    for x in lines:
        if x["warmup"]:
            continue
        if x["i"] == 0 and cur:
            reqs.append(cur)
            cur = []
        cur.append(x)
    if cur:
        reqs.append(cur)
    return reqs


def step_requests(path: Path):
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
    return [r for r in reqs if r and r[0][1] == NUM]


def mechanics(art: Path, sp: dict, mask: list[int]) -> dict:
    want = tiers(sp)
    out = {}
    per_rank = []
    for rank, p in enumerate(rank_logs(art)):
        reqs = requests_of(fwd_lines(p))
        bad = []
        for r_i, r in enumerate(reqs):
            if [x["i"] for x in r] != list(range(NUM)):
                bad.append({"req": r_i, "error": "forward indices", "n": len(r)})
                continue
            for x in r:
                i = x["i"]
                exp_cache = "computed" if mask[i] else "cached"
                ok = x["attention"] == want[i] and x["cache"] == exp_cache
                if want[i] != "bf16-fa2" and x["fa2"] != 0 and i > 0:
                    ok = False
                if not ok:
                    bad.append({"req": r_i, **x, "want": want[i], "want_cache": exp_cache})
        sage_fwds = [x for r in reqs for x in r if x["attention"].startswith("sage2")]
        per_rank.append({"requests": len(reqs), "bad": bad[:12], "n_bad": len(bad),
                         "sage_forwards_computed": sum(1 for x in sage_fwds if x["cache"] == "computed"),
                         "sage_forwards_cached": sum(1 for x in sage_fwds if x["cache"] == "cached"),
                         "fa2_calls_f0": [r[0]["fa2"] for r in reqs if r],
                         "fa2_calls_per_tier_req0": {t: sum(x["fa2"] for x in reqs[0] if x["attention"] == t)
                                                     for t in set(want)} if reqs else None,
                         "summary": [(x["i"], x["attention"][:5], x["cache"][:4], x["calls"], x["fa2"])
                                     for x in (reqs[0] if reqs else [])]})
    out["ranks"] = per_rank
    r0 = requests_of(fwd_lines(rank_logs(art)[0]))
    r1 = requests_of(fwd_lines(rank_logs(art)[1]))
    out["ranks_identical"] = bool(r0) and [[(x["i"], x["attention"], x["cache"]) for x in r] for r in r0] == \
        [[(x["i"], x["attention"], x["cache"]) for x in r] for r in r1]
    out["routing_ok"] = all(r["requests"] == 4 and r["n_bad"] == 0 for r in per_rank) and out["ranks_identical"]
    texts = [p.read_text(errors="replace") if p.is_file() else "" for p in rank_logs(art)]
    out["tier_lines"] = [m.group(0)[:300] for t in texts for m in TIER_LINE_RE.finditer(t)][:4]
    out["oom"] = [m.group(0)[:200] for t in texts for m in OOM_RE.finditer(t)][:10]
    # Recompiling lines after the first timed request started (rank-0 Step 1/49 line).
    rec = []
    for t in texts:
        idx = t.find("Step 1/49")
        if idx >= 0:
            rec += [m.group(0)[:240] for m in RECOMP_RE.finditer(t[idx:])]
    out["recompiles_in_timed"] = rec[:10]
    # Step time per tier: request 1 (park), 2, 3, 4 (park repeat).
    steps = step_requests(rank_logs(art)[0])
    tt = []
    for r in steps:
        rec_t = {}
        for (k, n, t) in r:
            i = k - 1
            key = f"{want[i][:4]}_{'dense' if mask[i] else 'cached'}"
            rec_t.setdefault(key, []).append(t)
        tt.append({**{k: [round(sum(v) / len(v), 3), len(v), round(max(v), 3)] for k, v in rec_t.items()},
                   "total": round(sum(t for (_, _, t) in r), 2)})
    out["step_times"] = tt
    return out


def judge_all(ckpt: str, root: Path, results: dict) -> None:
    sys.path.insert(0, str(BENCH))
    from ab_common import decode_video, ensure_ffmpeg
    from ab_judge import Judge, sample_frames
    from PIL import Image

    manifest = json.loads((BENCH / "reference_manifest.json").read_text())
    store = Path(manifest["store"])
    ensure_ffmpeg(str(store))
    prompts = json.loads((BENCH / "prompts.json").read_text())
    judge = Judge(ckpt)
    refs = {k: decode_video(store / manifest["reference"][k])[0] for k in manifest["check_keys"]}
    for tag, res in results.items():
        first = {}
        for r in res["requests"]:
            first.setdefault(r["key"], r["path"])
        correct = total = 0
        wrong = []
        for key, questions in prompts["accuracy"].items():
            path = first.get(key)
            frames = sample_frames(Path(path))[0] if path else None
            for question, expected in questions:
                total += 1
                if frames is None:
                    wrong.append({"key": key, "q": question, "error": "no output"})
                    continue
                ans, margin = judge.ask(frames, question)
                if ans == expected:
                    correct += 1
                else:
                    wrong.append({"key": key, "q": question, "expected": expected, "answer": ans,
                                  "margin": round(margin, 2)})
        res["accuracy"] = {"correct": correct, "total": total, "wrong": wrong}
        stills = root / tag / "stills"
        stills.mkdir(parents=True, exist_ok=True)
        same = []
        for key in manifest["check_keys"]:
            path = first.get(key)
            if not path:
                same.append({"key": key, "error": "no output"})
                continue
            cand = decode_video(Path(path))[0]
            for f in q.FRAMES:
                a, b = Image.fromarray(refs[key][f]), Image.fromarray(cand[f])
                pair = Image.new("RGB", (a.width * 2, a.height))
                pair.paste(a, (0, 0))
                pair.paste(b, (a.width, 0))
                pair.reduce(2).save(stills / f"{key}_f{f}_ref_left_cand_right.jpg", quality=88)
                ans, margin = judge.ask([a, b], q.SAME_Q)
                same.append({"key": key, "frame": f, "same_subject": ans, "margin": round(margin, 2)})
        res["same_subject"] = same
        res["stills_yes"] = sum(1 for s in same if s.get("same_subject") == "yes")
        for r in res["requests"]:
            r.pop("path", None)
        print(f"[stage3] {tag} judged accuracy={correct}/{total} stills_yes={res['stills_yes']}/9", flush=True)
    judge.close()


def main():
    ckpt, arms = sys.argv[1], sys.argv[2:]
    root = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    original_text = ENGINE.read_text()
    original = json.loads(original_text)
    print(f"[stage3] main engine.json (verbatim): {original_text}", flush=True)
    mask = mask_of(original["visual_gen_args"]["cache_config"])
    print(f"[stage3] mask {''.join(map(str, mask))}", flush=True)
    manifest = json.loads((BENCH / "reference_manifest.json").read_text())
    h = q.harness(ckpt, manifest["store"])
    results, meta = {}, {}
    try:
        for arm in arms:
            eng = arm_engine(original, arm)
            ENGINE.write_text(json.dumps(eng, indent=2))
            sp = eng["visual_gen_args"]["step_precision"]
            print(f"[stage3] === {arm}: {ENGINE.read_text()}", flush=True)
            print(f"[stage3] {arm} tiers {''.join(t[0] for t in tiers(sp))} direct {direct_seconds(sp, mask)}",
                  flush=True)
            art = root / arm
            t0 = time.time()
            check, jit = run_check(h, art, root / f"triton-{arm}")
            errs = " ".join(map(str, check.get("errors") or []))
            if not check.get("valid") and ("EADDRINUSE" in errs or "died during initialization" in errs):
                print(f"[stage3] {arm}: server init failed (port race); retrying once", flush=True)
                art = root / f"{arm}_retry"
                check, jit = run_check(h, art, root / f"triton-{arm}-retry")
            results[arm] = q.summarize(arm, check, art)
            meta[arm] = {"engine": eng, "tiers": "".join(t[0] for t in tiers(sp)),
                         "direct": direct_seconds(sp, mask), "jit": jit, "wall_s": round(time.time() - t0),
                         "mechanics": mechanics(art, sp, mask)}
            print(f"[stage3] {arm} summary " + json.dumps({"q": results[arm], "meta": meta[arm]}, default=str)[:6000],
                  flush=True)
    finally:
        ENGINE.write_text(original_text)
    judge_all(ckpt, root, results)

    table = {}
    for arm, res in results.items():
        rows = {r["key"]: r for r in res["per_prompt"]}
        margins = {k: r.get("margin_to_floor_plus_0.03") for k, r in rows.items()}
        lp = {k: r.get("lpips") for k, r in rows.items()}
        d = {k: round(lp[k] - MAIN_LPIPS[k], 4) for k in MAIN_LPIPS if lp.get(k) is not None}
        mean_d = sum(d.values()) / len(d) if d else None
        direct = meta[arm]["direct"]["direct_s"]
        mech = meta[arm]["mechanics"]
        fid_ok = len(margins) == 3 and all(m is not None and m >= 0.005 for m in margins.values())
        table[arm] = {
            "lpips": lp, "psnr": {k: r.get("psnr") for k, r in rows.items()},
            "audio_snr_db": {k: r.get("audio_snr_db") for k, r in rows.items()},
            "margin": margins, "fidelity_margin_ok": fid_ok,
            "delta_vs_main_lpips": d, "mean_delta": None if mean_d is None else round(mean_d, 4),
            "direct_s": direct,
            "net_s_mean_delta": None if mean_d is None else round(direct - 150.0 * max(mean_d, 0.0), 2),
            "net_s_max_delta": None if not d else round(direct - 150.0 * max(max(d.values()), 0.0), 2),
            "stills_yes": res.get("stills_yes"), "accuracy": (res.get("accuracy") or {}).get("correct"),
            "repeat_passed": res.get("repeated_execution_passed"), "repeat": res.get("repeat"),
            "check_valid": res.get("check_valid"), "check_errors": res.get("check_errors"),
            "routing_ok": mech["routing_ok"], "ranks_identical_cache": res.get("ranks_identical"),
            "oom": len(mech["oom"]), "recompiles_in_timed": len(mech["recompiles_in_timed"]),
            "triton_new_after_pass1": meta[arm]["jit"]["new_after_pass1_start"],
            "step_times": mech["step_times"],
            "pass": bool(fid_ok and res.get("stills_yes") == 9 and res.get("repeated_execution_passed")
                         and mech["routing_ok"] and not mech["oom"] and res.get("check_valid")),
        }
    (root / "stage3_table.json").write_text(json.dumps({"table": table, "meta": meta, "results": results},
                                                       indent=1, default=str))
    passing = [a for a in ("ks4", "ks6") if a in table and table[a]["pass"]
               and (table[a]["net_s_mean_delta"] or -1) > 0]
    print(json.dumps({"passed": bool(passing), "pick": passing[0] if passing else None,
                      "main_lpips_ref": MAIN_LPIPS,
                      "table": {a: {k: v for k, v in t.items() if k not in ("step_times", "repeat")}
                                for a, t in table.items()}}, default=str), flush=True)


if __name__ == "__main__":
    main()
