"""WAIT-ONLY dirty-slot guard (top lead rulings 20:5x/21:0x PST 11-05, pitfalls 3067/3072).

Call before any GPU allocation. Logs hostname, memory.used/util/power and compute-apps of the visible
GPUs. Polls up to 180 s; if any visible GPU still has > 4 GiB used, prints 'DIRTY SLOT <GiB>' and
exits 3. Never kills any process.
"""
import json
import os
import socket
import subprocess
import sys
import time


def dirty_guard(limit_gib=4.0, wait_s=180):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    t0 = time.time()
    print("[guard] host", socket.gethostname(), "CUDA_VISIBLE_DEVICES", vis, flush=True)
    while True:
        q = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.mem",
             "--format=csv,noheader,nounits"]
        if vis:
            q[1:1] = ["-i", vis]
        try:
            out = subprocess.run(q, capture_output=True, text=True, timeout=30).stdout.strip()
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                   "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            print("[guard] nvidia-smi failed:", e, flush=True)
            return
        used = []
        for ln in out.splitlines():
            parts = [x.strip() for x in ln.split(",")]
            try:
                used.append(float(parts[2]) / 1024.0)
            except Exception:  # noqa: BLE001
                pass
        print(f"[guard] t={time.time() - t0:.0f}s {out.replace(chr(10), ' | ')}", flush=True)
        print("[guard] compute-apps:", apps.replace("\n", " | ") or "none", flush=True)
        if not used or max(used) <= limit_gib:
            return
        if time.time() - t0 > wait_s:
            print("DIRTY SLOT " + " ".join(f"{u:.1f}GiB" for u in used), flush=True)
            print(json.dumps({"passed": False, "error": "DIRTY SLOT"}))
            sys.exit(3)
        time.sleep(15)
