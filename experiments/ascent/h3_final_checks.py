"""One 2-GPU correctness job for the H3 Cache-DiT branch on the committed engine.json.

1. Per-request reset (h3_cache_run.sh, VisualGen API, TP2): server "abA" runs seeds 42, 7, 42 back
   to back after its 2-step warmup; server "fresh" runs seed 42 alone. Passes when abA's third
   output is bitwise equal to its first and to fresh's, and both ranks log identical decisions.
2. Frozen check adapter + Qwen3-VL judge (h3_quality_job.py engine): per-prompt LPIPS/PSNR/audio
   SNR vs the frozen BF16 references with floor+0.03 margins, the same-server park_portrait repeat,
   both ranks' Cache-DiT decisions, 21 accuracy items and same-subject at frames 25/62/98.
Prints one JSON line combining both.

Usage: python h3_final_checks.py <checkpoint>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, env=None) -> dict:
    print(f"[final] $ {' '.join(cmd)[:300]}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=env)
    last = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("{") and '"passed"' in line:
            last = line
    proc.wait()
    try:
        return json.loads(last) if last else {"passed": False, "error": f"rc={proc.returncode}"}
    except json.JSONDecodeError:
        return {"passed": False, "error": "unparsable result"}


def main() -> None:
    ckpt = sys.argv[1]
    doc = json.loads((HERE.parent / "ascent-config" / "engine.json").read_text())
    cache = json.dumps(doc["visual_gen_args"]["cache_config"])
    env = dict(os.environ, H3_CROSS="abA_2_seed42:fresh_0_seed42 abA_0_seed42:fresh_0_seed42")
    reset = run(["bash", str(HERE / "h3_cache_run.sh"), sys.executable, ckpt, "768", "1344", "124",
                 f"abA|{cache}|42 7 42", f"fresh|{cache}|42"], env=env)
    quality = run([sys.executable, str(HERE / "h3_quality_job.py"), ckpt, "engine"])
    out = {"passed": bool(reset.get("passed")) and bool(quality.get("passed")),
           "cache_config": json.loads(cache), "reset": reset, "quality": quality}
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
