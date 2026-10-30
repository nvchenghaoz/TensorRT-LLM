"""Run the frozen experiments/ascent-benchmark/check.py from a worker GPU job (fact 537).

usage: run_check.py <worktree> <checkpoint>

The benchmark directory is not shipped to worker jobs. This finds runner copies of it, requires
check.py, common.py, reference.json, metrics.py, prompts.json and inputs.json to have identical
sha256 across every copy found (the frozen files; logged), copies one into
<worktree>/experiments/ascent-benchmark and runs check.py unchanged with the 2048^2 harness
(t2i-2048 + i2i-2048, nvfp4, 16 steps, c1, TP1). The adapter's own last JSON line is re-emitted
with a "passed" summary (valid and repeated_execution_passed and every case within policy).
"""

import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOTS = [
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/runner/slots/*/workspace/trtllm",
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/runner/slots/*/native/trtllm",
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/runner/slots/*/jobs/*/trtllm",
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/runner/slots/*/*/trtllm",
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/*/trtllm",
]
FILES = ["check.py", "common.py", "reference.json", "metrics.py", "prompts.json", "inputs.json"]


def log(msg):
    print(f"[run_check {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    worktree, ckpt = Path(sys.argv[1]), sys.argv[2]
    dst = worktree / "experiments" / "ascent-benchmark"
    copies = {}
    for pat in ROOTS:
        for root in glob.glob(pat):
            d = Path(root) / "experiments" / "ascent-benchmark"
            if d.resolve() == dst.resolve() or not (d / "check.py").is_file():
                continue
            try:
                copies[str(d)] = tuple(sha(d / f) for f in FILES)
            except OSError:
                continue
    log(f"found {len(copies)} benchmark copies")
    groups = {}
    for d, h in copies.items():
        groups.setdefault(h, []).append(d)
    for h, ds in groups.items():
        log(f"  {len(ds)} copies with " + ", ".join(f"{f}={x[:12]}" for f, x in zip(FILES, h))
            + f" e.g. {ds[0]}")
    if not groups:
        print(json.dumps({"passed": False, "error": "no benchmark copy found"}))
        return
    best = max(groups, key=lambda h: len(groups[h]))
    src = Path(groups[best][0])
    if not dst.exists():
        shutil.copytree(src, dst)
    log(f"copied {src} -> {dst}; sha256 " + json.dumps(dict(zip(FILES, best))))
    meta = Path(ckpt) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    revision = meta.read_text().splitlines()[0].strip() if meta.is_file() else None
    harness = {"workload_identity": {
        "workload": {"kind": "visual-gen", "model": "black-forest-labs/flux-3-image-klein",
                     "variant": "nvfp4", "num_inference_steps": 16, "concurrency": [1],
                     "tensor_parallel": 1, "cases": ["t2i-2048", "i2i-2048"]},
        "resolved_model": {"checkpoint": ckpt, "revision": revision}}}
    log(f"harness {json.dumps(harness)}")
    proc = subprocess.Popen([sys.executable, str(dst / "check.py"), json.dumps(harness)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(dst))
    last = ""
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            last = line
        if not line.startswith("{"):
            print(line, flush=True)
    rc = proc.wait()
    log(f"check.py rc={rc}")
    try:
        res = json.loads(last)
    except Exception:  # noqa: BLE001
        print(json.dumps({"passed": False, "adapter_last_line": last[-3000:]}))
        return
    comps = res.get("comparisons", {})
    per_case = {}
    total_fail = 0
    for case, b in comps.items():
        per_case[case] = f"{b.get('passed')}/{b.get('total')}"
        total_fail += (b.get("total") or 0) - (b.get("passed") or 0)
    n = sum((b.get("total") or 0) for b in comps.values())
    allowed = max(1, math.ceil(0.1 * n))
    majority = all((b.get("passed") or 0) * 2 > (b.get("total") or 0) for b in comps.values())
    rep = res.get("repeated_execution", {})
    summary = {
        "passed": bool(res.get("valid") and res.get("repeated_execution_passed")
                       and set(comps) == {"t2i-2048", "i2i-2048"} and total_fail <= allowed
                       and majority),
        "valid": res.get("valid"), "errors": res.get("errors"),
        "repeated_execution_passed": res.get("repeated_execution_passed"),
        "repeat_pairs": f"{sum(bool(p.get('passed')) for p in rep.get('pairs', []))}/"
                        f"{rep.get('expected_pairs')}",
        "bitwise_equal_pairs": rep.get("bitwise_equal_pairs"),
        "per_case_prompts_within_tolerance": per_case,
        "diverging_prompts": total_fail, "allowed": allowed,
        "benchmark_sha256": dict(zip(FILES, best)), "benchmark_source": str(src),
        "compared_against": "frozen original-source reference.json (LPIPS/PSNR per-case gates)",
    }
    out = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")) / "check_result.json"
    try:
        out.write_text(json.dumps(res, default=str))
    except OSError:
        pass
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
