"""Clean-tree evidence: byte-exact check + per-layer decode kernel counts (fused vs reference)."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = str(HERE / "csa2_glue.py")


def last_json(cmd):
    print("==", " ".join(cmd[1:]), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-6000:])
    sys.stderr.write(proc.stderr[-3000:])
    sys.stdout.flush()
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return None
    return json.loads(lines[-1])


def main():
    py = sys.executable
    check = last_json([py, SCRIPT, "check"])
    counts = {}
    for kind in ("full", "reindex", "swa"):
        for ref in (False, True):
            cmd = [py, SCRIPT, "bench", "--kind", kind, "--cases", "1,2,4,8"]
            if ref:
                cmd.append("--force-ref")
            out = last_json(cmd)
            label = f"{kind}/{'reference' if ref else 'fused'}"
            counts[label] = (
                None
                if out is None
                else {
                    b: {"kernels_per_layer_step": v.get("kernels_per_layer_step"), "kernel_ms": v["kernel_ms"]}
                    for b, v in out["batch"].items()
                }
            )
    passed = bool(check and check.get("passed")) and all(v is not None for v in counts.values())
    print(
        json.dumps(
            {
                "passed": passed,
                "compared": (check or {}).get("compared"),
                "cases": (check or {}).get("cases"),
                "mismatches": (check or {}).get("mismatches"),
                "kernel_counts_per_layer_decode_step": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
