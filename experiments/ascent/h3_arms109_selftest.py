#!/usr/bin/env python3
"""Self-test (no model, no CUDA) of idea 109's h3_arms_job: post(), decide_s7(), pre_pass(), gpu_guard().

Synthetic logs in the real formats (Step k/49 | t s start=..., refresh lines, forward_summary, stats,
decisions) for U34_F1 (S5), U38W6_F1 (S4) and S7 (= U38W6_F1 at K=6). Runs `h3_arms_job.py --post-only`
and checks the summary. Last stdout line: {"passed": ...}.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h3_arms_job as J  # noqa: E402
from h3_cache_arms_table import MASKS  # noqa: E402

P = "[11/13/2026-04:00:00] [TRT-LLM] [I] "
LP = {"U34_F1": (0.2444, 0.0700, 0.2775), "U38W6_F1": (0.3000, 0.1000, 0.3500),
      "S7": (0.3100, 0.1050, 0.3600)}
DT = {"U34_F1": (5.04, 3.76, 0.085), "U38W6_F1": (5.04, 3.76, 0.085), "S7": (5.04, 3.76, 0.085)}
KS = {"U34_F1": 14, "U38W6_F1": 14, "S7": 6}
HEADS = {"U34_F1": 8, "U38W6_F1": 6, "S7": 6}
KEYS = ("park_portrait", "perfume_bottle", "golden_robot")


def rank_log(arm, mask, rank, k):
    fn, head = 1, HEADS[arm]
    ms = "".join(map(str, mask))
    cached = [i for i, v in enumerate(mask) if v == 0]
    L = [P + f"Cache-DiT: MiniMax-H3 — 64 blocks, Fn={fn}, Bn=0, W={head}, MC=-1, R=0.08, scm=x",
         P + f"Cache-DiT refresh rank={rank}: n=2 Fn={fn} Bn=0 max_warmup_steps={head} resolved_warmup=1 "
             "mcc=-1 max_cached=-1 R=0.08 policy=static mask=10"]
    for i in range(2):
        L.append(f"[step_precision] rank={rank} forward_summary {i}/2 warmup=True attention=bf16-fa2 linear=x "
                 f"cache={'computed' if i == 0 else 'cached'} attention_calls={64 if i == 0 else 1}/64")
        if rank == 0:
            L.append(P + f"Step {i + 1}/2 | 3.0000s start={10.0 + 3 * i:.4f} (trans=2.9s cfg=0.000s sched=0.001s)")
    if rank == 0:
        L.append(P + "Denoising done: 6.00s (3.00s/step)")
    L.append(P + f"Cache-DiT decisions rank={rank}: context=transformer_blocks_1{rank} steps=2 cached=1 "
             "cached_steps=[1] residual_diffs={}")
    t0 = 100.0
    for _ in range(6):
        L.append(P + f"Cache-DiT refresh rank={rank}: n=49 Fn={fn} Bn=0 max_warmup_steps={head} "
                 f"resolved_warmup={head} mcc=-1 max_cached=-1 R=0.08 policy=static mask={ms}")
        tot = 0.0
        for i in range(49):
            ph = "bf16-fa2" if i < k else "fp8"
            c = "computed" if mask[i] else "cached"
            L.append(f"[step_precision] rank={rank} forward_summary {i}/49 warmup=False attention={ph} "
                     f"linear=x cache={c} attention_calls={64 if mask[i] else 1}/64")
            d = (DT[arm][0] if i < k else DT[arm][1]) if mask[i] else DT[arm][2]
            if rank == 0:
                L.append(P + f"Step {i + 1}/49 | {d - 0.001:.4f}s start={t0:.4f} (trans=1s cfg=0.000s "
                         f"sched=0.001s) | Avg=1.00s/step ETA=1.0s")
            t0 += d
            tot += d
        t0 += 20.0
        if rank == 0:
            L.append(P + f"Denoising done: {tot:.2f}s (1.00s/step)")
            L.append(P + f"Cache-DiT stats: {{'transformer_0': [CacheStats(cached_steps={cached}, "
                     f"accumulated_cached_steps={len(cached)})]}}")
        L.append(P + f"Cache-DiT decisions rank={rank}: context=transformer_blocks_9{rank} steps=49 "
                 f"cached={len(cached)} cached_steps={cached} residual_diffs={{}}")
    return "\n".join(L) + "\n", tot


def build(root, arm):
    mask = MASKS["U38W6_F1" if arm == "S7" else arm]
    logs = root / arm / "logs"
    logs.mkdir(parents=True)
    k = KS[arm]
    t0, tot = rank_log(arm, mask, 0, k)
    (logs / "server-check.log").write_text(t0)
    (logs / "server-check-rank1.log").write_text(rank_log(arm, mask, 1, k)[0])
    (logs / "server-check.visual_gen_args.yaml").write_text(
        "attention_config:\n  backend: FLASHINFER\n  quant_attention_config:\n    qk_dtype: fp8\n    v_dtype: fp8\n"
        f"cache_config:\n  cache_backend: cache_dit\n  Fn_compute_blocks: 1\nparallel_config:\n  tp_size: 2\n"
        f"step_precision:\n  attention_bf16_forwards: {k}\n")
    reqs = [{"key": key, "server_denoise_s": round(tot + 0.05, 3), "client_e2e_s": tot + 4.5}
            for key in KEYS + ("park_portrait",)]
    (root / arm / "adapter_result.json").write_text(json.dumps({
        "adapter": "check", "valid": True, "errors": [], "repeated_execution_passed": True, "requests": reqs,
        "reset_sequence": {"passed": True, "reset43": {"server_denoise_s": tot + 0.05},
                           "reset42": {"server_denoise_s": tot + 0.05}}}))
    per = [{"key": key, "lpips": LP[arm][j], "psnr": 20.0, "audio_snr_db": 9.0, "passed": True,
            "margin_to_floor_plus_0.03": round(J.BARS[key] - LP[arm][j], 4)} for j, key in enumerate(KEYS)]
    same = [{"key": key, "frame": f, "same_subject": "yes", "margin": 3.0} for key in KEYS for f in (25, 62, 98)]
    return {"check_valid": True, "repeated_execution_passed": True, "per_prompt": per, "repeat": [],
            "requests": reqs, "accuracy": {"correct": 21, "total": 21, "wrong": []}, "same_subject": same,
            "fidelity_pass": True}, tot


def main():
    J.check_masks()
    cfgs = {a: J.arm_config(a) for a in ("U34_F1", "U36_F1", "W6U36_F1", "U38W6_F1", "W6U40_F1")}
    print("configs", json.dumps(cfgs), flush=True)
    eng = json.loads(J.q.ENGINE.read_text())["visual_gen_args"]["cache_config"]
    verbatim = eng == cfgs["U34_F1"]
    root = Path(tempfile.mkdtemp())
    configs, tots = {}, {}
    for arm in ("U34_F1", "U38W6_F1", "S7"):
        configs[arm], tots[arm] = build(root, arm)
    # Pre-judge decision for S7 from the summaries, as the hook sees them.
    dec = J.decide_s7({a: configs[a] for a in ("U34_F1", "U38W6_F1")}, root, ["U38W6_F1"])
    (root / "S7_decision.json").write_text(json.dumps(dec))
    fail = dict(configs["U38W6_F1"], per_prompt=[dict(p, lpips=0.345) if p["key"] == "park_portrait" else p
                                                for p in configs["U38W6_F1"]["per_prompt"]])
    pp_fail = J.pre_pass("U38W6_F1", {"U38W6_F1": fail}, root)
    configs["W6U40_F1"] = {"tag": "W6U40_F1", "skipped": "U38W6_F1 failed first", "per_prompt": [], "requests": []}
    (root / "quality.json").write_text(json.dumps({"passed": True, "configs": configs}))
    proc = subprocess.run([sys.executable, str(HERE / "h3_arms_job.py"), "--post-only", str(root),
                           "U34_F1", "U38W6_F1", "W6U40_F1", "S7"], capture_output=True, text=True,
                          env=dict(os.environ))
    print(proc.stdout[-9000:], proc.stderr[-4000:], flush=True)
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    summ = json.loads((root / "arms_summary.json").read_text())
    a4 = summ["arms"]["U38W6_F1"]
    s7 = summ["arms"]["S7"]
    pt0 = a4["phase_timing"][0]
    exp_qc = DT["U38W6_F1"][2]
    guard = J.gpu_guard()
    checks = {
        "s5_verbatim_engine": verbatim,
        "dec_arm": dec["decision_arm"] == "U38W6_F1" and dec["K"] == 6 and dec["run"]
        and dec["dense_in_K_14"] == [12],
        "pre_pass_fail_detected": pp_fail[0] is False,
        "s5_match": all((last["S5_matches_686"] or {}).values()),
        "decision": last["decision_arm"] == "U38W6_F1",
        "mech_s5": summ["arms"]["U34_F1"]["mechanics_ok"], "mech_s4": a4["mechanics_ok"],
        "mech_s7": s7["mechanics_ok"],
        "stats": a4["stats_cached_count_per_timed_request"] == [38] * 6,
        "refresh_n49": a4["refresh_n49_count"] == [6, 6],
        # U38-W6 at K=14: dense {0-5, 12} bf16, {21, 31, 41, 48} quant; cached 7 bf16 + 31 quant.
        "phase_counts": pt0["quant_cached"]["count"] == 31 and pt0["bf16_cached"]["count"] == 7
        and pt0["bf16_dense"]["count"] == 7 and pt0["quant_dense"]["count"] == 4 and pt0["start_to_start"],
        "phase_mean": abs(pt0["quant_cached"]["mean"] - exp_qc) < 1e-3,
        "phase_sum": abs(pt0["phase_sum_s"] - tots["U38W6_F1"]) < 0.01,
        "s7_phases": s7["phase_timing"][0]["bf16_dense"]["count"] == 6
        and s7["phase_timing"][0]["quant_dense"]["count"] == 5 and s7["K_used"] == 6,
        "s7_vs_k14": s7["vs_same_arm_K14"]["lpips_increment"]["park_portrait"] == 0.01,
        "inc_over_s5": a4["per_prompt"]["park_portrait"]["increment_over_S5"] == 0.0556,
        "skip_kept": summ["arms"]["W6U40_F1"].get("skipped") is not None,
        "guard_ran": "wrong_pair" in guard,
    }
    print(json.dumps(checks), flush=True)
    print(json.dumps({"passed": all(checks.values()), "checks": checks, "guard": guard}, default=str))


if __name__ == "__main__":
    main()
