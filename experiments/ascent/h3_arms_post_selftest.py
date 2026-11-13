#!/usr/bin/env python3
"""CPU self-test of h3_arms_job.post() on synthetic logs in the real formats (K=14 stacked main).

Builds <tmp>/F/{logs,adapter_result.json} + quality.json, runs `h3_arms_job.py --post-only`, and checks
the summary: warmup steps=2 cached=1 both ranks, forward phase/cache per mask, yaml keys, phase timing.
Last stdout line: {"passed": ...}.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h3_cache_arms_table import MASKS  # noqa: E402

ARM = "F"
K = 14
mask = MASKS[ARM]
root = Path(tempfile.mkdtemp())
logs = root / ARM / "logs"
logs.mkdir(parents=True)
P = "[11/13/2026-04:00:00] [TRT-LLM] [I] "
cached = [i for i, v in enumerate(mask) if v == 0]


def rank_log(rank):
    L = [P + f"Cache-DiT refresh rank={rank}: n=2 Fn=8 Bn=0 max_warmup_steps=8 resolved_warmup=1 mcc=-1 "
         "max_cached=-1 R=0.08 policy=static mask=10"]
    for i in range(2):
        L.append(f"[step_precision] rank={rank} forward_summary {i}/2 warmup=True attention=fp8 "
                 f"linear=bf16(no linear gate on main) cache={'computed' if i == 0 else 'cached'} "
                 f"attention_calls={50 if i == 0 else 8}/50")
        if rank == 0:
            L.append(P + f"Step {i + 1}/2 | 3.00s (trans=2.9s cfg=0.000s sched=0.001s) | Avg=3.0s/step ETA=0.0s")
    if rank == 0:
        L.append(P + "Denoising done: 6.00s (3.00s/step)")
    L.append(P + f"Cache-DiT decisions rank={rank}: context=transformer_blocks_123{rank} steps=2 cached=1 "
             "cached_steps=[1] residual_diffs={}")
    for _ in range(4):
        L.append(P + f"Cache-DiT refresh rank={rank}: n=49 Fn=8 Bn=0 max_warmup_steps=8 resolved_warmup=8 "
                 f"mcc=-1 max_cached=-1 R=0.08 policy=static mask={''.join(map(str, mask))}")
        for i in range(49):
            ph = "bf16-fa2" if i < K else "fp8"
            c = "computed" if mask[i] else "cached"
            L.append(f"[step_precision] rank={rank} forward_summary {i}/49 warmup=False attention={ph} "
                     f"linear=bf16(no linear gate on main) cache={c} attention_calls={50 if mask[i] else 8}/50")
            if rank == 0:
                t = (5.5 if i < K else 4.2) if mask[i] else 0.7
                L.append(P + f"Step {i + 1}/49 | {t:.2f}s (trans={t:.2f}s cfg=0.000s sched=0.001s) | "
                         f"Avg={t:.2f}s/step ETA=1.0s")
        if rank == 0:
            L.append(P + "Denoising done: 100.00s (2.04s/step)")
            L.append(P + f"Cache-DiT stats: {{'transformer_0': [CacheStats(cached_steps={cached}, "
                     f"accumulated_cached_steps={len(cached)})]}}")
        L.append(P + f"Cache-DiT decisions rank={rank}: context=transformer_blocks_9{rank} steps=49 "
                 f"cached={len(cached)} cached_steps={cached} residual_diffs={{}}")
    return "\n".join(L) + "\n"


(logs / "server-check.log").write_text(rank_log(0))
(logs / "server-check-rank1.log").write_text(rank_log(1))
(logs / "server-check.visual_gen_args.yaml").write_text(
    "attention_config:\n  backend: FLASHINFER\n  quant_attention_config:\n    qk_dtype: fp8\n    v_dtype: fp8\n"
    "cache_config:\n  cache_backend: cache_dit\nparallel_config:\n  tp_size: 2\n"
    "step_precision:\n  attention_bf16_forwards: 14\n")
per = [{"key": "park_portrait", "lpips": 0.30, "psnr": 20.0, "audio_snr_db": 1.0, "passed": True},
       {"key": "perfume_bottle", "lpips": 0.09, "psnr": 27.0, "audio_snr_db": 3.0, "passed": True},
       {"key": "golden_robot", "lpips": 0.33, "psnr": 19.0, "audio_snr_db": 5.0, "passed": True}]
(root / ARM / "adapter_result.json").write_text(json.dumps(
    {"adapter": "check", "comparisons": {"1": {"per_prompt": per}}, "repeated_execution_passed": True,
     "repeated_execution": [{"key": "park_portrait", "passed": True}], "valid": True, "errors": []}))
bars = {"park_portrait": 0.347, "perfume_bottle": 0.166, "golden_robot": 0.508}
qper = [dict(p, floor=bars[p["key"]] - 0.03, **{"margin_to_floor_plus_0.03": round(bars[p["key"]] - p["lpips"], 4)})
        for p in per]
(root / "quality.json").write_text(json.dumps({"passed": True, "configs": {ARM: {
    "check_valid": True, "check_errors": [], "output_class": "mode_switch", "repeated_execution_passed": True,
    "per_prompt": qper, "repeat": [], "requests": [], "accuracy": {"correct": 21, "total": 21},
    "same_subject": [], "fidelity_pass": True, "server_cache_config": {}, "floor_lpips": {}}}}))
env = dict(os.environ, H3_ARMS_K_BF16=str(K), H3_ARMS_L2="park=0.1289,perfume=0.0561,robot=0.1739",
           H3_ARMS_SNR_REF=json.dumps({"park_portrait": 1.13, "perfume_bottle": 3.74, "golden_robot": 5.88}))
proc = subprocess.run([sys.executable, str(HERE / "h3_arms_job.py"), "--post-only", str(root), ARM],
                      capture_output=True, text=True, env=env)
print(proc.stdout[-6000:], proc.stderr[-4000:], flush=True)
last = json.loads(proc.stdout.strip().splitlines()[-1])
a = last["per_arm"][ARM]
ok = (proc.returncode == 0 and last["mechanics_ok"] and last["pick"] == ARM and a["warmup_ok"]
      and a["fwd_phase_cache_ok"] and a["yaml_keys"]["attention_backend"] == "FLASHINFER"
      and len(a["phase_timing"]) == 4 and a["phase_timing"][0]["bf16_dense"][1] == 8
      and abs(a["snr_delta_vs_ref"]["park_portrait"] + 0.13) < 1e-6)
print(json.dumps({"passed": bool(ok), "phase_timing0": a["phase_timing"][0]}))
