"""Idea 122: fresh-process A/B/A of the image.0 FP8R unit (unit.py bench), env fixed at load.
A = TRTLLM_FLUX3_FP8R_GEMM=0 (main's torch._scaled_mm path), B = 1 (routed). Same GPU, one job."""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(flag):
    env = dict(os.environ, TRTLLM_FLUX3_FP8R_GEMM=flag)
    r = subprocess.run([sys.executable, os.path.join(HERE, "unit.py"), "bench"], env=env,
                       capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{\"batch\"")]
    for ln in r.stdout.splitlines():
        if ln.startswith("[bench]") or ln.startswith("[install]"):
            print(f"  [{flag}] {ln}", flush=True)
    if not lines:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"arm {flag} failed")
    return json.loads(lines[-1])


def main():
    rows = []
    for label, flag in (("A1", "0"), ("B", "1"), ("A2", "0")):
        print(f"[aba] {label} TRTLLM_FLUX3_FP8R_GEMM={flag}", flush=True)
        res = run(flag)
        rows.append((label, res))
        print(f"[aba] {label} unit_us {res['unit_us']} routed={res['routed']}", flush=True)
    a1, b, a2 = (r[1]["batch"]["1"]["kernel_ms"] * 1000 for r in rows)
    ok = b < a1 and b < a2 and not rows[0][1]["routed"] and rows[1][1]["routed"]
    print(json.dumps({"passed": ok, "A1_us": a1, "B_us": b, "A2_us": a2,
                      "B_minus_meanA_us": b - (a1 + a2) / 2,
                      "per_request_ms": 16 * (b - (a1 + a2) / 2) / 1000,
                      "rows": {k: v for k, v in rows}}), flush=True)


if __name__ == "__main__":
    main()
