"""Idea 348 S0 in one job: S0.P probe (probe348.py) then S0.S standalone sweep (sweep.py + proto.cu + v3x.cuh).

Last stdout line: {"passed": ..., "probe": {...}, "sweep": {...}} JSON.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KSET = ";".join([
    "v3<8,8,1>()", "v3<16,8,1>()", "v3<8,8,2>()", "v2<24,4,true,1,true>()",
    "v3x<8,8,1>()", "v3x<9,8,1>()", "v3x<10,8,1>()", "v3x<11,8,1>()", "v3x<12,8,1>()", "v3x<13,8,1>()",
    "v3x<14,8,1>()", "v3x<15,8,1>()", "v3x<16,8,1>()", "v3x<4,8,1>()",
])
MS = os.environ.get("SWEEP_MS", "512,1024,2048,4096,8192,2047,129")
ITERS = os.environ.get("SWEEP_ITERS", "20")


def run(cmd, timeout):
    print("[s0] run:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        line = line.rstrip("\n")
        print(line[:3000], flush=True)
        if line.startswith("{"):
            try:
                last = json.loads(line)
            except Exception:  # noqa: BLE001
                pass
    p.wait(timeout=timeout)
    return last or {"passed": False, "error": f"rc={p.returncode}, no JSON"}


skip_probe = os.environ.get("SKIP_PROBE") == "1"
probe = {"skipped": True} if skip_probe else run([sys.executable, os.path.join(HERE, "probe348.py")], 2400)
sweep = run([sys.executable, os.path.join(HERE, "sweep.py"), MS, ITERS, KSET], 2400)
print(json.dumps({"passed": bool(sweep.get("passed")), "probe_passed": probe.get("passed"),
                  "probe": probe.get("report", probe), "sweep": sweep}))
