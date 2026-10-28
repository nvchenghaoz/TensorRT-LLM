"""Branch 68: run experiments/ascent/idea16_e2e_check.py unchanged (one trtllm-serve server,
client test request + the 12 frozen prompts twice, b64_json) and sample, while it runs,
the GPU memory used (nvidia-smi) and the RSS of the server process tree (/proc).
Memory is read at each request boundary from the check's log lines; flat means the
level after pass 2 equals the level after pass 1 (GPU within 4 MiB, RSS within 64 MiB) and
no growth across the 12 pass-2 requests. Also counts recompile/JIT lines in the server
log after the test request. Last line: idea16's JSON fields + memory fields.
Usage: python b68_e2e_mem.py <checkpoint>
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

CKPT = sys.argv[1]
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b68_e2e")
HERE = os.path.dirname(os.path.abspath(__file__))


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def tree_rss_mib(root_pid):
    kids = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            kids.setdefault(ppid, []).append(int(p))
        except Exception:  # noqa: BLE001
            pass
    tot, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        stack += kids.get(pid, [])
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1])
        except Exception:  # noqa: BLE001
            pass
    return tot / 1024


proc = subprocess.Popen([sys.executable, os.path.join(HERE, "idea16_e2e_check.py"), CKPT],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
marks = []  # (label, gpu MiB, rss MiB)
last = None
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    m = re.search(r"(test request done|pass (\d) prompt\s+(\d+))", line)
    if m:
        marks.append((m.group(1), gpu_used_mib(), round(tree_rss_mib(proc.pid), 1)))
        print(f"MEM {marks[-1]}", flush=True)
    if line.startswith("{"):
        last = line
proc.wait()
res = json.loads(last) if last else {"passed": False}
p1 = [m for m in marks if m[0].startswith("pass 1")]
p2 = [m for m in marks if m[0].startswith("pass 2")]
flat = False
if p1 and p2 and p1[-1][1] is not None:
    gpu_d = p2[-1][1] - p1[-1][1]
    rss_d = p2[-1][2] - p1[-1][2]
    gpu_span2 = max(m[1] for m in p2) - min(m[1] for m in p2)
    flat = abs(gpu_d) <= 4 and abs(rss_d) <= 64 and gpu_span2 <= 4
    res.update(gpu_mib_after_test=marks[0][1], gpu_mib_after_pass1=p1[-1][1], gpu_mib_after_pass2=p2[-1][1],
               rss_mib_after_test=marks[0][2], rss_mib_after_pass1=p1[-1][2], rss_mib_after_pass2=p2[-1][2],
               gpu_mib_span_pass2=gpu_span2)
recomp = None
try:
    log = open(os.path.join(ART, "server.log"), errors="replace").read()
    recomp = len(re.findall(r"[Rr]ecompil", log))
except Exception:  # noqa: BLE001
    pass
res.update(memory_flat=flat, recompile_lines_in_server_log=recomp,
           passed=bool(res.get("passed")) and flat)
print(json.dumps(res), flush=True)
