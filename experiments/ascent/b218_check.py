"""Idea 218: frozen check.py (gate v2 + repeated execution) on this tree, plus a compile audit.

usage: TORCH_LOGS=dynamo TLLM_LOG_LEVEL=INFO python experiments/ascent/b218_check.py <checkpoint>

Runs experiments/ascent/run_adapter.py check <checkpoint> unchanged (the adapter starts its own
server from this worktree with experiments/ascent-config/engine.json, here vae_decode_fp8_convs=true).
Then reads the server log in order: every Dynamo frame compile ("torchdynamo start tracing") and every
cuDNN FE FP8 plan build ("flux3_vae::fp8_conv3x3 built") is located relative to "Warmup completed"
and to the request lines ("Generating image:"). Compiles or plan builds after warmup = inside requests.
Last stdout line: check.py's result JSON with "passed" (valid, every c1 prompt within gate v2,
repeated_execution_passed) and "compile_audit" added; check.py's own fields are unchanged.
"""

import glob
import json
import os
import re
import subprocess
import sys

ckpt = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
proc = subprocess.Popen([sys.executable, os.path.join(here, "run_adapter.py"), "check", ckpt],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
last = ""
for line in proc.stdout:
    line = line.rstrip("\n")
    if line.strip():
        last = line
    print(line, flush=True)
rc = proc.wait()
try:
    res = json.loads(last)
except Exception:  # noqa: BLE001
    res = {"adapter_last_line": last[-3000:]}

art = os.environ.get("AUTO_DEV_ARTIFACTS", "")
logs = sorted(glob.glob(os.path.join(art, "**", "check_server.log"), recursive=True))
if not logs and isinstance(res.get("artifacts"), dict) and res["artifacts"].get("server_log"):
    logs = [res["artifacts"]["server_log"]]
audit = {"server_log": logs[0] if logs else None}
if logs:
    lines = open(logs[0], errors="replace").read().splitlines()
    warm = [i for i, ln in enumerate(lines) if "Warmup completed" in ln]
    gen = [i for i, ln in enumerate(lines) if "Generating image:" in ln]
    dyn = [i for i, ln in enumerate(lines) if "torchdynamo start tracing" in ln]
    plans = [i for i, ln in enumerate(lines) if "fp8_conv3x3 built" in ln]
    prep = [ln[-200:] for ln in lines if "prepare_fp8_convs" in ln or "vae_decode_fp8_convs" in ln]
    recompile = [ln[-300:] for ln in lines if "Recompiling function" in ln or "recompil" in ln.lower()]
    w_end = warm[-1] if warm else None
    after = (lambda idx: [i for i in idx if w_end is not None and i > w_end])
    audit.update({
        "warmup_completed_line": w_end, "n_request_lines": len(gen),
        "first_request_line": gen[0] if gen else None,
        "dynamo_frame_compiles_total": len(dyn), "dynamo_frame_compiles_after_warmup": len(after(dyn)),
        "fp8_plan_builds_total": len(plans), "fp8_plan_builds_after_warmup": len(after(plans)),
        "fp8_plan_build_lines": [lines[i][-220:] for i in plans][:16],
        "fp8_prepare_lines": prep[:6], "recompile_lines": recompile[:10],
        "dynamo_lines_after_warmup": [lines[i][-300:] for i in after(dyn)][:10],
        "none_inside_requests": w_end is not None and not after(dyn) and not after(plans),
    })
c1 = (res.get("comparisons") or {}).get("c1") or {}
per = c1.get("per_prompt") or []
res["compile_audit"] = audit
res["gate_v2_c1"] = {"passed": c1.get("passed"), "total": c1.get("total"),
                     "per_prompt": [{k: r.get(k) for k in ("key", "lpips", "psnr_db", "passed")} for r in per]}
res["passed"] = bool(res.get("valid") and per and all(r.get("passed") for r in per)
                     and res.get("repeated_execution_passed"))
res["adapter_rc"] = rc
print(json.dumps(res, default=str), flush=True)
