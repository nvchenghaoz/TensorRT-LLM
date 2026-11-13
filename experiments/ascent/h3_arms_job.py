#!/usr/bin/env python3
"""MiniMax-H3 Cache-DiT static-mask arms job: frozen check adapter + judge per arm, then the arms table.

Usage: python h3_arms_job.py <checkpoint> <arm> [<arm> ...]
       python h3_arms_job.py --post-only <artifact_root> <arm> [<arm> ...]
Arms: names in h3_cache_arms_table.MASKS with an HEAD entry here (idea 109: U34_F1 = S5, U36_F1 = S2,
W6U36_F1 = S3, U38W6_F1 = S4, W6U40_F1 = S1), plus "S7": the decision arm (passing arm with the largest
measured denoise saving over U34_F1, else U34_F1) re-run with step_precision.attention_bf16_forwards =
its max_warmup_steps (skipped when it has no dense forward in [K, 14)).

GPU guard first (before any model load): nvidia-smi index / uuid / bus id / link width of the visible
GPUs; GPU 5 or bus AB:00.0 -> prints {"wrong_pair": true, ...} and exits 0. A parse failure is loud.

Each arm = one trtllm-serve server (2 GPUs) running the frozen run_check.py through run_check_reset.py
(park42, perfume42, robot42, park42 repeat, park43, park42 reset), via h3_quality_job.main (engine.json
written per arm and restored; the judge runs after all servers stop: 21 items + same subject at frames
25/62/98). Every arm: static, Bn0, Fn from FN, no TaylorSeer, mcc -1, max_warmup_steps = the leading 1s;
length 49 and the cached count asserted. W6U40_F1 is skipped when U38W6_F1 ran first and failed.
Afterwards: per-arm visual_gen_args.yaml, refresh lines on both ranks (Fn/Bn/max_warmup_steps/mask),
stats-line cached count per timed request, LPIPS/PSNR/SNR with margins to .347/.166/.508 and increments
over same-job U34_F1, per-phase forward seconds start-to-start (4 decimals, from the Step k/49 lines'
start=), phase sum vs Denoising done and server_denoise_s, savings vs U34_F1. Last line: JSON.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h3_quality_job as q  # noqa: E402
from h3_cache_arms_table import EXPECTED_CACHED, MASKS, check_masks, parse_server_log  # noqa: E402

L2 = os.environ.get("H3_ARMS_L2", "park=0.0390,perfume=0.0198,robot=0.0887")  # table's legacy column
SNR_FLAG = {"park_portrait": 1.74, "perfume_bottle": 4.60, "golden_robot": 7.47}
SNR_REF = json.loads(os.environ.get("H3_ARMS_SNR_REF", "{}"))
BARS = {"park_portrait": 0.347, "perfume_bottle": 0.166, "golden_robot": 0.508}
KEYS = ("park_portrait", "perfume_bottle", "golden_robot")
REF_686 = {"park_portrait": 0.2444, "perfume_bottle": 0.0700, "golden_robot": 0.2775}  # job 686 / 629
PASS_MARGIN = 0.005
S5 = "U34_F1"
PARK_PRICE_S = 1.5  # seconds per .01 park LPIPS (break-even pricing)
STEP_RE = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+)s")
STEP4_RE = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+)s start=([0-9.]+)")
DONE_RE = re.compile(r"Denoising done: ([0-9.]+)s")
FWD_RE = re.compile(r"\[step_precision\] rank=\d+ forward_summary (\d+)/(\d+) warmup=(\w+) attention=(\S+) "
                    r".*?cache=(\w+) attention_calls=(\d+)/(\d+)")
OOM_RE = re.compile(r"(failed with OOM|out of memory|CUDA error)[^\n]*", re.IGNORECASE)


def _engine_k() -> int:
    try:
        eng = json.loads(q.ENGINE.read_text())
        return int(((eng.get("visual_gen_args") or {}).get("step_precision") or {})
                   .get("attention_bf16_forwards") or 0)
    except Exception:
        return 0


# attention_bf16_forwards the servers get from engine.json (not set by this script, pitfall 1145);
# H3_ARMS_K_BF16 only overrides what post-processing expects when engine.json has none.
DEFAULT_K = _engine_k() or int(os.environ.get("H3_ARMS_K_BF16", "0"))
ARM_K: dict = {}  # per-arm expected K (S7: the decision arm's max_warmup_steps)

HEAD = {"E": 8, "D": 8, "C": 8, "fallback": 10, "W": 14, "w14_c2": 14, "F": 8, "Es": 8, "Ds": 8, "U31": 8,
        "F_F1": 8, "U34_F1": 8, "U36_F1": 8, "W6U36_F1": 6, "U38W6_F1": 6, "W6U40_F1": 6}
MCC = {"w14_c2": 2}
FN = {"F_F1": 1, "U34_F1": 1, "U36_F1": 1, "W6U36_F1": 1, "U38W6_F1": 1, "W6U40_F1": 1}
# Every arm's server also runs park seed 43 then park seed 42 after the frozen check's repeat.
RESET_ALL = os.environ.get("H3_ARMS_RESET_ALL", "1") == "1"
SKIP_IF_FAILED = {"W6U40_F1": "U38W6_F1"}  # S1 runs only if S4 passed (when S4 ran first)


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


def lead_ones(mask):
    n = 0
    for v in mask:
        if v != 1:
            break
        n += 1
    return n


def arm_config(arm):
    mask = MASKS[arm]
    assert len(mask) == 49 and mask.count(0) == EXPECTED_CACHED[arm], arm
    assert lead_ones(mask) == HEAD[arm], (arm, lead_ones(mask), HEAD[arm])
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


# ------------------------------------------------------------------------------------------ GPU guard
def _smi(args):
    try:
        p = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=60)
        return p.stdout
    except Exception as ex:  # noqa: BLE001
        return f"ERROR {ex!r}"


def gpu_guard() -> dict:
    """Visible GPUs by physical index / uuid / bus id / link width, without a CUDA context."""
    rows = _smi(["--query-gpu=index,uuid,pci.bus_id,pcie.link.width.current,pcie.link.width.max,name",
                 "--format=csv,noheader"])
    gpus = []
    for ln in rows.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 5 and p[0].isdigit():
            gpus.append({"index": int(p[0]), "uuid": p[1], "bus": p[2], "width_cur": p[3], "width_max": p[4]})
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    toks = [t.strip() for t in cvd.split(",") if t.strip()]
    visible, unmapped = [], []
    for t in toks:
        hit = [g for g in gpus if (t.startswith("GPU-") and g["uuid"] == t)
               or (t.isdigit() and g["index"] == int(t))]
        (visible.extend(hit) if hit else unmapped.append(t))
    if not toks:
        visible = list(gpus)
    info = {"CUDA_VISIBLE_DEVICES": cvd, "smi_rows": len(gpus), "visible": visible, "unmapped": unmapped}
    if not gpus or unmapped or len(visible) != 2:
        info["parser_error"] = True
    bad = [g for g in visible if g["index"] == 5 or g["bus"].upper().endswith("AB:00.0")]
    # In a container nvidia-smi may renumber; the bus id still names GPU 5.
    bad += [g for g in gpus if g["bus"].upper().endswith("AB:00.0") and g not in bad and len(gpus) <= 2]
    info["wrong_pair"] = bool(bad)
    info["gpu5_present"] = bool(bad)
    print("[arms] GPU_GUARD " + json.dumps(info), flush=True)
    print(_smi(["topo", "-m"]), flush=True)
    return info


# ------------------------------------------------------------------------------------------ helpers
def _art(root: Path, arm: str) -> Path:
    art = root / arm
    if not (art / "adapter_result.json").is_file() and (root / f"{arm}_retry").is_dir():
        art = root / f"{arm}_retry"
    return art


def _adapter(root: Path, arm: str) -> dict:
    p = _art(root, arm) / "adapter_result.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except ValueError:
        return {}


def _lpips_rows(summary: dict) -> dict:
    return {r["key"]: r for r in (summary.get("per_prompt") or []) if r.get("key")}


def _pass1_denoise(summary_or_adapter: dict):
    reqs = summary_or_adapter.get("requests") or []
    vals = [r.get("server_denoise_s") for r in reqs[:3] if r.get("server_denoise_s") is not None]
    return statistics.fmean(vals) if len(vals) == 3 else None


def pre_pass(tag: str, results: dict, root: Path):
    """Pre-judge verdict from the check alone: bars margins, repeat, reset. (ok, min_margin, why)."""
    s = results.get(tag) or {}
    if s.get("skipped"):
        return False, None, "skipped"
    rows = _lpips_rows(s)
    if not s.get("check_valid") or any(k not in rows or rows[k].get("lpips") is None for k in KEYS):
        return False, None, "check invalid or missing prompt"
    mm = min(BARS[k] - rows[k]["lpips"] for k in KEYS)
    reset = (_adapter(root, tag).get("reset_sequence") or {}).get("passed")
    ok = mm >= PASS_MARGIN and bool(s.get("repeated_execution_passed")) and (reset is not False)
    return ok, round(mm, 4), f"min_margin={mm:.4f} repeat={s.get('repeated_execution_passed')} reset={reset}"


def has_dense_in(mask, k, hi=14):
    return any(mask[i] == 1 for i in range(k, min(hi, len(mask))))


def decide_s7(results: dict, root: Path, candidates) -> dict:
    s5_den = _pass1_denoise(results.get(S5) or {})
    rows = []
    for arm in candidates:
        ok, mm, why = pre_pass(arm, results, root)
        den = _pass1_denoise(results.get(arm) or {})
        saving = (s5_den - den) if (s5_den is not None and den is not None) else None
        rows.append({"arm": arm, "pre_pass": ok, "min_margin": mm, "why": why, "pass1_denoise_s": den,
                     "saving_vs_S5_s": None if saving is None else round(saving, 3)})
    passing = [r for r in rows if r["pre_pass"] and r["saving_vs_S5_s"] is not None]
    pick = max(passing, key=lambda r: r["saving_vs_S5_s"])["arm"] if passing else S5
    k = HEAD[pick]
    dec = {"decision_arm": pick, "K": k, "rows": rows, "S5_pass1_denoise_s": s5_den,
           "dense_in_K_14": [i for i in range(k, 14) if MASKS[pick][i] == 1],
           "run": has_dense_in(MASKS[pick], k)}
    if not dec["run"]:
        dec["skip_reason"] = f"{pick} has no dense forward in [{k},14): S7 would equal it at K=14"
    return dec


# ------------------------------------------------------------------------------------------ timing
def step_requests4(path: Path):
    """Rank-0 Step lines split into requests: [{steps: [(k, n, t, start)], done: s}]."""
    reqs, cur = [], None
    if not path.is_file():
        return reqs
    for ln in path.read_text(errors="replace").splitlines():
        m = STEP4_RE.search(ln)
        m3 = m or STEP_RE.search(ln)
        if m3:
            k, n, t = int(m3.group(1)), int(m3.group(2)), float(m3.group(3))
            st = float(m.group(4)) if m else None
            if k == 1 or cur is None:
                cur = {"steps": [], "done": None}
                reqs.append(cur)
            cur["steps"].append((k, n, t, st))
            continue
        d = DONE_RE.search(ln)
        if d and cur is not None and cur["done"] is None:
            cur["done"] = float(d.group(1))
    return reqs


def _stat(v):
    return {"count": len(v), "mean": round(statistics.fmean(v), 4) if v else None,
            "median": round(statistics.median(v), 4) if v else None, "sum": round(sum(v), 4)}


def phase_timing(reqs, mask, k_bf16, server_denoise=None):
    """Per timed request: forward seconds start-to-start (last forward: its own step time) per phase
    bf16/quant x dense/cached; phase sum vs Denoising done and the adapter's server_denoise_s."""
    out = []
    timed = [r for r in reqs if r["steps"] and r["steps"][0][1] == len(mask) and len(r["steps"]) == len(mask)]
    for j, r in enumerate(timed):
        st = r["steps"]
        s2s = all(x[3] is not None for x in st)
        durs = []
        for i, (k, n, t, s) in enumerate(st):
            durs.append(round(st[i + 1][3] - s, 4) if (s2s and i + 1 < len(st)) else t)
        ph = {}
        for i, d in enumerate(durs):
            p = "bf16" if i < k_bf16 else "quant"
            ph.setdefault(f"{p}_{'dense' if mask[i] else 'cached'}", []).append(d)
        rec = {key: _stat(v) for key, v in sorted(ph.items())}
        rec["start_to_start"] = s2s
        rec["phase_sum_s"] = round(sum(durs), 4)
        rec["denoising_done_s"] = r["done"]
        sd = server_denoise[j] if server_denoise and j < len(server_denoise) else None
        rec["server_denoise_s"] = sd
        rec["phase_sum_minus_server_denoise_s"] = round(rec["phase_sum_s"] - sd, 4) if sd else None
        rec["forward_s"] = durs
        out.append(rec)
    return out


def phase_means(pt):
    """Mean over timed requests of each phase's per-request mean."""
    acc = {}
    for rec in pt:
        for key, v in rec.items():
            if isinstance(v, dict) and v.get("mean") is not None:
                acc.setdefault(key, []).append(v["mean"])
    return {k: round(statistics.fmean(v), 4) for k, v in acc.items()}


def fwd_summary(path: Path):
    if not path.is_file():
        return []
    return [(int(m.group(1)), m.group(3), m.group(4), m.group(5))
            for ln in path.read_text(errors="replace").splitlines() for m in [FWD_RE.search(ln)] if m]


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


def recompile_scan(path: Path):
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


def _register_s7(dec: dict):
    arm = dec["decision_arm"]
    MASKS["S7"] = list(MASKS[arm])
    EXPECTED_CACHED["S7"] = EXPECTED_CACHED[arm]
    HEAD["S7"] = HEAD[arm]
    FN["S7"] = FN.get(arm, 8)
    ARM_K["S7"] = dec["K"]


def main():
    ckpt, arms = sys.argv[1], sys.argv[2:]
    guard = gpu_guard()
    if guard.get("wrong_pair"):
        print(json.dumps({"wrong_pair": True, "passed": False, "guard": guard}), flush=True)
        sys.exit(0)
    if guard.get("parser_error"):
        # Loud, never treated as wrong_pair (top-lead ruling); no GPU 5 / AB:00.0 was seen.
        print("[arms] GPU guard PARSER ERROR: could not map CUDA_VISIBLE_DEVICES to nvidia-smi rows",
              flush=True)
    check_masks()
    print(f"[arms] DEFAULT_K (engine.json attention_bf16_forwards) = {DEFAULT_K}", flush=True)
    eng = json.loads(q.ENGINE.read_text())
    for arm in arms:
        if arm == "S7":
            q.CONFIGS[arm] = None  # filled by the pre-tag hook once the decision arm is known
            continue
        q.CONFIGS[arm] = arm_config(arm)
        print(f"[arms] {arm}: {json.dumps(q.CONFIGS[arm])}", flush=True)
    if S5 in arms:
        verbatim = eng["visual_gen_args"]["cache_config"] == q.CONFIGS[S5]
        print(f"[arms] S5 {S5} config == branch engine.json cache_config: {verbatim}", flush=True)
        assert verbatim, "S5 must be the branch's engine.json verbatim"
    root = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "gpu_guard.json").write_text(json.dumps(guard, indent=1))

    def pre_tag(tag, results, root_):
        dep = SKIP_IF_FAILED.get(tag)
        if dep and dep in results:
            ok, mm, why = pre_pass(dep, results, root_)
            if not ok:
                return f"{dep} failed first ({why})"
        if tag == "S7":
            cands = [a for a in arms if a not in ("S7", S5) and a in results]
            dec = decide_s7(results, root_, cands)
            (root_ / "S7_decision.json").write_text(json.dumps(dec, indent=1))
            print("[arms] S7 decision " + json.dumps(dec), flush=True)
            if not dec["run"]:
                return dec["skip_reason"]
            _register_s7(dec)
            q.CONFIGS["S7"] = arm_config(dec["decision_arm"])
            q.OVERRIDES["S7"] = {"step_precision": {"attention_bf16_forwards": dec["K"]}}
        return None

    q.PRE_TAG = pre_tag

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
    if RESET_ALL:
        def _rc(h, art):
            return _run_check_script(h, art, HERE / "run_check_reset.py")

        q.run_check = _rc
    sys.argv = [sys.argv[0], ckpt] + arms
    q.main()
    (root / "quality.json").write_text(json.dumps(captured.get("quality", {}), default=str))
    post(root, arms, captured.get("quality", {}).get("configs", {}))


def post(root: Path, arms, quality):
    dec_p = root / "S7_decision.json"
    s7_dec = json.loads(dec_p.read_text()) if dec_p.is_file() else None
    if "S7" in arms and s7_dec and s7_dec.get("run"):
        _register_s7(s7_dec)
    out = {"arms": {}, "_extra": {}, "S7_decision": s7_dec}
    specs = []
    ran = []
    for arm in arms:
        qa = quality.get(arm, {})
        if qa.get("skipped") or arm not in MASKS:
            out["arms"][arm] = {"skipped": qa.get("skipped") or "not run"}
            continue
        ran.append(arm)
        art = _art(root, arm)
        logs = art / "logs"
        yaml_p = logs / "server-check.visual_gen_args.yaml"
        ytxt = yaml_p.read_text() if yaml_p.is_file() else ""
        print(f"[arms] {arm} visual_gen_args.yaml:\n{ytxt}", flush=True)
        import yaml as _yaml
        yv = (_yaml.safe_load(ytxt) if ytxt else {}) or {}
        attn = yv.get("attention_config") or {}
        k_yaml = (yv.get("step_precision") or {}).get("attention_bf16_forwards")
        k_want = ARM_K.get(arm, DEFAULT_K)
        k_use = int(k_yaml) if k_yaml is not None else k_want
        r0 = grep(logs / "server-check.log", REFRESH_RE)
        r1 = grep(logs / "server-check-rank1.log", REFRESH_RE)
        d0 = grep(logs / "server-check.log", DEC_RE)
        d1 = grep(logs / "server-check-rank1.log", DEC_RE)
        for ln in r0[:2] + r1[:2]:
            print(f"[arms] {arm} refresh: {ln}", flush=True)
        mask = MASKS[arm]
        mask_str = "".join(map(str, mask))
        f0 = fwd_summary(logs / "server-check.log")
        f1 = fwd_summary(logs / "server-check-rank1.log")
        timed_f = [x for x in f0 if x[1] == "False"]
        fwd_ok = bool(timed_f) and all(
            (a == ("bf16-fa2" if i < k_use else "fp8")) and (c == ("computed" if mask[i] else "cached"))
            for (i, w, a, c) in timed_f)
        oom = []
        texts = {}
        for p in (logs / "server-check.log", logs / "server-check-rank1.log"):
            if p.is_file():
                texts[p.name] = p.read_text(errors="replace")
                oom += [f"{p.name}: {m.group(0)[:200]}" for m in OOM_RE.finditer(texts[p.name])]
        ares = _adapter(root, arm)
        sreqs = ares.get("requests") or []
        reset = ares.get("reset_sequence") or {}
        # All 49-forward requests in log order: pass1 x3, repeat, park43, park42.
        server_denoise = [r.get("server_denoise_s") for r in sreqs] + [
            (reset.get(t) or {}).get("server_denoise_s") for t in ("reset43", "reset42")]
        pt = phase_timing(step_requests4(logs / "server-check.log"), mask, k_use, server_denoise)
        tbl_reqs = [r for r in parse_server_log(texts.get("server-check.log", ""))
                    if r["steps"] and r["steps"][0][1] == 49]
        stats_counts = [r["stats_count"] for r in tbl_reqs]
        stats_sets_ok = [sorted(r["stats_cached_steps"] or []) == [i for i, v in enumerate(mask) if v == 0]
                         for r in tbl_reqs]
        want_fn = FN.get(arm, 8)
        full0 = [ln for ln in r0 if ln.startswith("n=49 ")]
        full1 = [ln for ln in r1 if ln.startswith("n=49 ")]
        sig = f"Fn={want_fn} Bn=0 max_warmup_steps={HEAD[arm]} "
        enable = {n: [m.group(1) for m in map(ENABLE_RE.search, t.splitlines()) if m] for n, t in texts.items()}
        rows = _lpips_rows(qa)
        s5rows = _lpips_rows(quality.get(S5, {}))
        per_prompt = {}
        for k in KEYS:
            r = rows.get(k, {})
            lp = r.get("lpips")
            per_prompt[k] = {"lpips": lp, "psnr": r.get("psnr"), "audio_snr_db": r.get("audio_snr_db"),
                             "margin": None if lp is None else round(BARS[k] - lp, 4),
                             "margin_floor_plus_0.03": r.get("margin_to_floor_plus_0.03"),
                             "increment_over_S5": (round(lp - s5rows[k]["lpips"], 4)
                                                   if lp is not None and s5rows.get(k, {}).get("lpips") is not None
                                                   else None)}
        same = [s for s in (qa.get("same_subject") or [])]
        out["arms"][arm] = {
            "cache_config_yaml": yv.get("cache_config"),
            "yaml_keys": {"attention_backend": attn.get("backend"),
                          "quant_attention_config": attn.get("quant_attention_config"),
                          "step_precision": yv.get("step_precision"),
                          "parallel_config": yv.get("parallel_config")},
            "K_used": k_use, "K_expected": k_want,
            "per_prompt": per_prompt,
            "min_margin": (min(v["margin"] for v in per_prompt.values())
                           if all(v["margin"] is not None for v in per_prompt.values()) else None),
            "same_subject": same,
            "same_subject_ok": bool(same) and all(s.get("same_subject") == "yes" for s in same),
            "accuracy": (qa.get("accuracy") or {}).get("correct"),
            "accuracy_wrong": (qa.get("accuracy") or {}).get("wrong"),
            "fidelity_pass": qa.get("fidelity_pass"),
            "repeat": qa.get("repeat"),
            "repeated_execution_passed": qa.get("repeated_execution_passed"),
            "reset_sequence": reset,
            "reset_passed": reset.get("passed"),
            "check_valid": qa.get("check_valid"), "check_errors": qa.get("check_errors"),
            "phase_timing": pt,
            "phase_means": phase_means(pt),
            "pass1_denoise_s": _pass1_denoise(ares),
            "client_e2e_s": [r.get("client_e2e_s") for r in sreqs],
            "fn_ok": bool(r0) and bool(r1) and all(f"Fn={want_fn} Bn=0" in ln for ln in r0 + r1),
            "refresh_sig_ok": bool(full0) and bool(full1) and all(sig in ln for ln in full0 + full1),
            "mask_ok": bool(full0) and bool(full1) and all(f"mask={mask_str}" in ln for ln in full0 + full1),
            "refresh_count": [len(r0), len(r1)], "refresh_n49_count": [len(full0), len(full1)],
            "refresh_rank0_sample": r0[:2], "refresh_rank1_sample": r1[:2],
            "refresh_ranks_identical": bool(r0) and r0 == r1,
            "stats_cached_count_per_timed_request": stats_counts,
            "stats_ok": bool(stats_counts) and all(c == mask.count(0) for c in stats_counts) and all(stats_sets_ok),
            "decisions_ranks_identical": bool(d0) and d0 == d1,
            "warmup_decision_rank0": d0[0] if d0 else None, "warmup_decision_rank1": d1[0] if d1 else None,
            "warmup_ok": bool(d0) and bool(d1) and d0[0].startswith("steps=2 cached=1")
            and d1[0].startswith("steps=2 cached=1"),
            "forward_phase_cache_ok": fwd_ok,
            "forward_summary_ranks_identical": bool(f0) and f0 == f1,
            "taylorseer": {n: ("TaylorSeer enabled" in t) for n, t in texts.items()},
            "enable_lines": enable,
            "cudnn_in_yaml": "CUDNN" in ytxt.upper(),
            "oom_hits": oom[:10],
            "audio_snr_flags": [k for k in KEYS if per_prompt[k]["audio_snr_db"] is not None
                                and per_prompt[k]["audio_snr_db"] < SNR_FLAG[k]],
            "recompiles_rank0_n": len(recompile_scan(logs / "server-check.log")),
        }
        specs.append(f"{arm}:{mask_str}:{art / 'adapter_result.json'}:{logs / 'server-check.log'}")
    # S5 reference match (job 686 / 629) and savings vs same-job S5.
    s5 = out["arms"].get(S5) or {}
    s5_den = s5.get("pass1_denoise_s")
    if s5.get("per_prompt"):
        out["S5_matches_686"] = {k: (None if s5["per_prompt"][k]["lpips"] is None
                                     else round(s5["per_prompt"][k]["lpips"], 4) == REF_686[k]) for k in KEYS}
        out["S5_lpips_4dp"] = {k: (None if s5["per_prompt"][k]["lpips"] is None
                                   else round(s5["per_prompt"][k]["lpips"], 4)) for k in KEYS}
    for arm in ran:
        a = out["arms"][arm]
        den = a.get("pass1_denoise_s")
        a["saving_vs_S5_s"] = round(s5_den - den, 3) if (s5_den and den) else None
        inc_park = a["per_prompt"]["park_portrait"]["increment_over_S5"]
        a["net_s_at_1.5s_per_.01_park"] = (round(a["saving_vs_S5_s"] - PARK_PRICE_S * inc_park / 0.01, 3)
                                          if a["saving_vs_S5_s"] is not None and inc_park is not None else None)
        a["mechanics_ok"] = bool(
            a["fn_ok"] and a["refresh_sig_ok"] and a["mask_ok"] and a["stats_ok"] and a["warmup_ok"]
            and a["decisions_ranks_identical"] and a["refresh_ranks_identical"] and not a["cudnn_in_yaml"]
            and a["forward_phase_cache_ok"] and a["forward_summary_ranks_identical"]
            and a["K_used"] == a["K_expected"] and not any(a["taylorseer"].values()))
        a["passes"] = bool(a["mechanics_ok"] and a["check_valid"] and a["fidelity_pass"]
                           and a["same_subject_ok"] and a["min_margin"] is not None
                           and a["min_margin"] >= PASS_MARGIN and a["repeated_execution_passed"]
                           and a["reset_passed"])
    # S7 vs the same arm at K = DEFAULT_K.
    if "S7" in ran and s7_dec:
        base = out["arms"].get(s7_dec["decision_arm"]) or {}
        s7 = out["arms"]["S7"]
        s7["vs_same_arm_K14"] = {
            "arm": s7_dec["decision_arm"], "K": s7_dec["K"],
            "lpips_increment": {k: (round(s7["per_prompt"][k]["lpips"] - base["per_prompt"][k]["lpips"], 4)
                                    if s7["per_prompt"][k]["lpips"] is not None and base.get("per_prompt")
                                    and base["per_prompt"][k]["lpips"] is not None else None) for k in KEYS},
            "saving_s": (round(base["pass1_denoise_s"] - s7["pass1_denoise_s"], 3)
                         if base.get("pass1_denoise_s") and s7.get("pass1_denoise_s") else None),
        }
        inc = s7["vs_same_arm_K14"]["lpips_increment"]["park_portrait"]
        sv = s7["vs_same_arm_K14"]["saving_s"]
        s7["vs_same_arm_K14"]["net_s_at_1.5s_per_.01_park"] = (round(sv - PARK_PRICE_S * inc / 0.01, 3)
                                                              if sv is not None and inc is not None else None)
    submit = [a for a in ran if a != "S7" and out["arms"][a].get("passes")]
    out["passing_arms"] = submit
    out["decision_arm"] = (max(submit, key=lambda a: out["arms"][a]["saving_vs_S5_s"] or -1e9)
                           if submit else None)
    out["S7_pre_decision_agrees"] = (s7_dec or {}).get("decision_arm") == (out["decision_arm"] or S5)
    tbl_json = root / "arms_table.json"
    cmd = [sys.executable, str(HERE / "h3_cache_arms_table.py"), "--l2", L2, "--json-out", str(tbl_json)]
    for s in specs:
        cmd += ["--arm", s]
    if specs:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout[-12000:], proc.stderr[-3000:], flush=True)
    (root / "arms_summary.json").write_text(json.dumps(out, indent=1, default=str))
    small = {"passed": bool(out["decision_arm"]) and all(out["arms"][a].get("mechanics_ok") for a in ran),
             "decision_arm": out["decision_arm"], "passing_arms": submit,
             "S5_lpips_4dp": out.get("S5_lpips_4dp"), "S5_matches_686": out.get("S5_matches_686"),
             "S7_decision": {k: v for k, v in (s7_dec or {}).items() if k != "rows"} or None,
             "per_arm": {}}
    for a, v in out["arms"].items():
        if v.get("skipped"):
            small["per_arm"][a] = {"skipped": v["skipped"]}
            continue
        small["per_arm"][a] = {
            k: v.get(k) for k in ("passes", "mechanics_ok", "min_margin", "saving_vs_S5_s",
                                  "net_s_at_1.5s_per_.01_park", "pass1_denoise_s", "phase_means",
                                  "stats_cached_count_per_timed_request", "refresh_n49_count", "K_used",
                                  "same_subject_ok", "accuracy", "repeated_execution_passed", "reset_passed",
                                  "fn_ok", "refresh_sig_ok", "mask_ok", "stats_ok", "warmup_ok",
                                  "forward_phase_cache_ok", "vs_same_arm_K14")}
        small["per_arm"][a]["prompts"] = {k: {kk: (round(x, 4) if isinstance(x, float) else x)
                                              for kk, x in p.items()} for k, p in v["per_prompt"].items()}
        small["per_arm"][a]["phase_sum_vs_denoise"] = [
            (r["phase_sum_s"], r["denoising_done_s"], r["server_denoise_s"]) for r in v["phase_timing"]]
    print(json.dumps(small, default=str), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--post-only":  # --post-only ROOT ARM...
        _root = Path(sys.argv[2])
        _q = json.loads((_root / "quality.json").read_text()) if (_root / "quality.json").is_file() else {}
        post(_root, sys.argv[3:], _q.get("configs", {}))
    else:
        main()
