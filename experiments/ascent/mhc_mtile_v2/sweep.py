"""Build proto.cu with nvcc (sm_120), print ptxas regs/stack/spills per mtile kernel, then run it.

usage: sweep.py [Ms] [iters] [kset]
  kset: ';'-separated kernel constructors, e.g. "v1<24,4>();v2<24,4,false,1>()"
Last stdout line: {"passed": ..., "ptxas": {...}, "rows": [...]} JSON.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HDR_DIR = os.path.abspath(os.path.join(HERE, "../../../cpp/tensorrt_llm/kernels/mhcKernels"))
nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

DEFAULT_KSET = ";".join([
    "v1<24,4>()", "v1<24,3>()", "v1<12,2>()",
    "v2<24,4,false,1>()", "v2<24,3,false,1>()", "v2<12,4,false,1>()", "v2<8,4,false,1>()",
    "v2<12,6,false,1>()", "v2<8,6,false,1>()", "v2<6,4,false,1>()", "v2<4,4,false,1>()",
    "v2<12,2,false,1>()", "v2<12,2,false,2>()",
    "v2<24,4,false,1,true>()", "v2<24,3,false,1,true>()", "v2<12,4,false,1,true>()", "v2<8,4,false,1,true>()",
    "v2<12,6,false,1,true>()", "v2<8,6,false,1,true>()", "v2<4,4,false,1,true>()", "v2<24,4,true,1,true>()",
    "v2<24,4,true,1>()", "v2<12,4,true,1>()", "v2<24,3,true,1>()", "v2<8,4,true,1>()",
    "v2<24,2,true,2>()", "v2<12,2,true,2>()",
])

Ms = sys.argv[1] if len(sys.argv) > 1 else "128,2048,4096,8192,33,127,2047"
iters = sys.argv[2] if len(sys.argv) > 2 else "20"
kset = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_KSET

td = tempfile.mkdtemp()
exe = os.path.join(td, "proto")
with open(os.path.join(td, "kset.h"), "w") as f:
    f.write("#define KSET " + ", ".join(kset.split(";")) + "\n")
cmd = [nvcc, "-std=c++17", "-O3", "-gencode", "arch=compute_120,code=sm_120", "-I", HDR_DIR,
       "-Xptxas", "-v", "-include", os.path.join(td, "kset.h"), os.path.join(HERE, "proto.cu"), "-o", exe]
print("nvcc:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
log = r.stderr + r.stdout
ptxas = {}
fn = None


def demangle(n):
    return subprocess.run(["c++filt"], input=n, capture_output=True, text=True).stdout.strip()


for line in log.splitlines():
    m = re.search(r"Compiling entry function '(\w+)'", line)
    if m:
        fn = re.sub(r"void fused_fma_kernels::", "", demangle(m.group(1))).split("(")[0]
        continue
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
    if m and fn:
        ptxas.setdefault(fn, {}).update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
    m = re.search(r"Used (\d+) registers", line)
    if m and fn:
        ptxas.setdefault(fn, {})["regs"] = int(m.group(1))
for k, v in ptxas.items():
    if "mtile" in k or "ksplit" in k:
        print(f"[ptxas] {k}: {v}", flush=True)
if r.returncode != 0:
    errs = [l for l in log.splitlines() if "error" in l]
    print("\n".join(errs[:15]), flush=True)
    print(log[:5000], flush=True)
    print(json.dumps({"passed": False, "error": "nvcc failed"}))
    sys.exit(0)

try:
    sm = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,clocks.max.sm,temperature.gpu",
                         "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    print("[gpu]", os.uname().nodename, sm, flush=True)
except Exception as e:  # noqa: BLE001
    print("[gpu] nvidia-smi failed:", e, flush=True)
p = subprocess.Popen([exe, Ms, iters], stdout=subprocess.PIPE, text=True)
rows = []
passed = False
for line in p.stdout:
    line = line.strip()
    print(line, flush=True)
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "kernel" in d:
        rows.append(d)
    elif "passed" in d:
        passed = d["passed"]
p.wait()
print(json.dumps({"passed": bool(passed) and p.returncode == 0,
                  "ptxas": {k: v for k, v in ptxas.items() if "mtile" in k},
                  "rows": rows}))
