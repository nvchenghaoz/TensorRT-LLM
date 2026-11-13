"""Run the frozen check adapter with the controller's harness_config recovered from serving job 158.

The check adapter (experiments/ascent-benchmark/run_check.py) is run unmodified; this wrapper only
supplies its argv[1]. It overlays this worktree's experiments/ascent-config/engine.json itself.
After it finishes, greps the server logs for evidence of the CUDNN backend and for recompiles /
NotImplementedError, and appends a summary line.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_harness import find  # noqa: E402

WT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    harness = find(["4285f767d2ff4900a5b55c2c9abf8847"])
    print("harness.server:", json.dumps(harness.get("server")), flush=True)
    adapter = WT / "experiments" / "ascent-benchmark" / "run_check.py"
    proc = subprocess.run([sys.executable, str(adapter), json.dumps(harness)], stdout=subprocess.PIPE,
                          text=True)
    out = proc.stdout
    sys.stdout.write(out[-20000:])
    last = out.strip().splitlines()[-1] if out.strip() else "{}"
    try:
        res = json.loads(last)
    except Exception:
        res = {"valid": False, "errors": ["unparseable adapter output"]}
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    logs = sorted((art / "logs").glob("server-check*.log"))
    grep = {"cudnn_mentions": 0, "vanilla_mentions": 0, "recompile": 0, "not_implemented": 0,
            "cudnn_lines": [], "bad_lines": []}
    for p in logs:
        for line in p.read_text(errors="replace").splitlines():
            low = line.lower()
            if "cudnn" in low:
                grep["cudnn_mentions"] += 1
                if len(grep["cudnn_lines"]) < 8:
                    grep["cudnn_lines"].append(f"{p.name}: {line[:300]}")
            if "vanilla" in low:
                grep["vanilla_mentions"] += 1
            if re.search(r"recompil", low):
                grep["recompile"] += 1
                if len(grep["bad_lines"]) < 8:
                    grep["bad_lines"].append(f"{p.name}: {line[:300]}")
            if "notimplementederror" in low:
                grep["not_implemented"] += 1
                if len(grep["bad_lines"]) < 8:
                    grep["bad_lines"].append(f"{p.name}: {line[:300]}")
    yamls = {p.name: p.read_text() for p in (art / "logs").glob("*.visual_gen_args.yaml")}
    per = (res.get("comparisons", {}).get("1", {}) or {}).get("per_prompt", [])
    summary = {
        "passed": bool(res.get("valid")) and all(x.get("passed") for x in per) and bool(per)
        and bool(res.get("repeated_execution_passed")) and grep["not_implemented"] == 0,
        "adapter_valid": res.get("valid"),
        "output_class": (res.get("output") or {}).get("class"),
        "engine_overlay": (res.get("engine_overlay") or {}).get("content"),
        "server_visual_gen_args_yaml": yamls,
        "per_prompt": [{k: x.get(k) for k in ("key", "passed", "bitwise_equal", "lpips", "psnr", "audio_snr_db",
                                              "lpips_tol", "psnr_tol")} for x in per],
        "repeated_execution_passed": res.get("repeated_execution_passed"),
        "repeat": [{"key": r.get("key"), "passed": r.get("passed"),
                    "vs_first_pass": {k: (r.get("vs_first_pass") or {}).get(k)
                                      for k in ("bitwise_equal", "lpips", "psnr", "audio_snr_db")}}
                   for r in res.get("repeated_execution", [])],
        "server_log_grep": grep,
        "errors": res.get("errors"),
        "requests": [{k: r.get(k) for k in ("key", "client_e2e_s", "server_denoise_s")}
                     for r in res.get("requests", [])],
    }
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
