"""Idea 101 J2 arm 3 (diagnostic, non-timing): the branch tree under Cache-DiT mask F.

Writes the job's runtime copy of experiments/ascent-config/engine.json with main's engine.json where ONLY
visual_gen_args.cache_config is replaced by mask F (fact 889, branch 31 ca50b622); attention_config and
step_precision are kept byte for byte. Runs the frozen check adapter via run_check_wrapper.py, restores the
original engine.json, then greps both ranks' server logs for the allocator counters and OOM lines.
The branch's committed engine.json is never changed.

usage: python run_check_maskF.py <checkpoint>
"""

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / "experiments/ascent-config/engine.json"
MASK_F = {
    "cache_backend": "cache_dit",
    "Fn_compute_blocks": 8,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 8,
    "max_continuous_cached_steps": -1,
    "residual_diff_threshold": 0.08,
    "scm_compute_bins": [8, 1, 1, 1, 1, 1, 2],
    "scm_cache_bins": [9, 9, 9, 4, 2, 1],
    "scm_steps_policy": "static",
}
PATTERN = re.compile(
    r"peak_memory|failed with OOM|OutOfMemory|out of memory|Processing request|step mask|Cache-DiT: MiniMax|"
    r"Warmup completed|\[step_precision\] rank=\d attention_bf16|Attention backend")


def main() -> int:
    checkpoint = sys.argv[1]
    original = ENGINE.read_text()
    cfg = json.loads(original)
    cfg["visual_gen_args"]["cache_config"] = MASK_F
    ENGINE.write_text(json.dumps(cfg, indent=2))
    print(f"[maskF] engine.json for this run: {json.dumps(cfg)}", flush=True)
    try:
        rc = subprocess.run([sys.executable, str(ROOT / "experiments/ascent/run_check_wrapper.py"),
                             str(ROOT / "experiments/ascent-benchmark/run_check.py"), checkpoint]).returncode
    finally:
        ENGINE.write_text(original)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "")
    counts = {}
    for f in sorted(glob.glob(os.path.join(art, "**", "*.log"), recursive=True)):
        if "server" not in os.path.basename(f) or "segment" in f:
            continue
        hits = [f"{i}: {line.rstrip()[:400]}" for i, line in enumerate(open(f, errors="replace"))
                if PATTERN.search(line)]
        counts[os.path.basename(f)] = len(hits)
        print(f"== {f} ({len(hits)} hits)", flush=True)
        print("\n".join(hits), flush=True)
    for f in glob.glob(os.path.join(art, "**", "*visual_gen_args.yaml"), recursive=True):
        print(f"== {f}\n{open(f).read()}", flush=True)
    print(json.dumps({"passed": rc == 0, "adapter_rc": rc, "grep_hits": counts, "diagnostic": "mask F arm 3"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
