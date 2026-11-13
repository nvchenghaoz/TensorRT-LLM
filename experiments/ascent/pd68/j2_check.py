"""J2 for idea 133 / branch 68: the frozen check adapter on this branch, plus response-path readout.

Runs experiments/ascent/run_check_wrapper.py -> experiments/ascent-benchmark/run_check.py (frozen refs,
same-server park repeat) with TRTLLM_VISUAL_GEN_RESPONSE_TIMING=1 in the environment (the adapter's
Server inherits os.environ; engine.json is untouched). Afterwards it reports per request
server_e2e - server_gen (park first, perfume, robot, park repeat), the startup reservation log line,
and the [vg-timing] encode stamps. Non-timing check: any GPU pair.

usage: python experiments/ascent/pd68/j2_check.py <checkpoint>
"""

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    checkpoint = sys.argv[1]
    adapter = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "experiments/ascent-benchmark/run_check.py"
    print(f"[j2] adapter {adapter} exists={adapter.is_file()}", flush=True)
    env = dict(os.environ, TRTLLM_VISUAL_GEN_RESPONSE_TIMING="1")
    rc = subprocess.run([sys.executable, str(ROOT / "experiments/ascent/run_check_wrapper.py"),
                         str(adapter), checkpoint], env=env).returncode
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    res_path = art / "adapter_result.json"
    res = json.loads(res_path.read_text()) if res_path.is_file() else {}
    rows = []
    for i, r in enumerate(res.get("requests", [])):
        e2e, gen = r.get("server_e2e_s"), r.get("server_gen_s")
        rows.append({"order": i, "key": r.get("key"), "server_gen_s": gen, "server_e2e_s": e2e,
                     "e2e_minus_gen_s": round(e2e - gen, 3) if e2e is not None and gen is not None else None,
                     "client_e2e_s": r.get("client_e2e_s"), "sha256": r.get("sha256")})
    reserve_lines, encode_lines = [], []
    for f in sorted(glob.glob(str(art / "logs" / "server-check*.log"))):
        for line in open(f, errors="replace"):
            if "pinned video buffer" in line:
                reserve_lines.append(f"{os.path.basename(f)}: {line.strip()[:300]}")
            if "[vg-timing] event=encode" in line or "event=route_save" in line:
                encode_lines.append(line.strip()[-300:])
    print("== reservation lines\n" + "\n".join(reserve_lines), flush=True)
    print("== encode stamps\n" + "\n".join(encode_lines), flush=True)
    per_prompt = res.get("comparisons", {}).get("1", {}).get("per_prompt", [])
    prompts_ok = bool(per_prompt) and all(p.get("passed") for p in per_prompt)
    repeat_ok = bool(res.get("repeated_execution_passed"))
    repeat_bitwise = [r.get("vs_first_pass", {}).get("bitwise_equal") for r in res.get("repeated_execution", [])]
    park = [r for r in rows if r["key"] == "park_portrait"]
    park_gap = (round(park[0]["e2e_minus_gen_s"] - park[1]["e2e_minus_gen_s"], 3)
                if len(park) == 2 and None not in (park[0]["e2e_minus_gen_s"], park[1]["e2e_minus_gen_s"]) else None)
    passed = rc == 0 and bool(res.get("valid")) and prompts_ok and repeat_ok
    print(json.dumps({
        "passed": passed,
        "compared": "frozen check adapter (LPIPS/PSNR/audio vs original-source references, same-server park repeat)",
        "adapter_valid": res.get("valid"), "adapter_errors": res.get("errors"),
        "per_prompt": [{k: p.get(k) for k in ("key", "passed", "bitwise_equal", "lpips", "psnr", "audio_snr")}
                       for p in per_prompt],
        "repeat_passed": repeat_ok, "repeat_bitwise": repeat_bitwise,
        "e2e_minus_gen": rows, "park_first_minus_repeat_s": park_gap,
        "reservation_log": reserve_lines, "startup_s": res.get("startup_s"),
    }, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
