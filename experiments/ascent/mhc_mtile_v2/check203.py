"""Correctness job for idea 316 (branch 203): build receipt (top lead ruling 02:0x PST 11-06), then s1_check.py.

Receipt: libtensorrt_llm.so / libth_common.so path + mtime, `strings <lib> | grep -c fused_pmap_gemm_fma_mtile_v3`
(6 on a post-landing-61 build) and the count of this branch's own symbol fused_pmap_gemm_fma_mtile_v2ILi12ELi2E
(> 0 on the branch). Then runs s1_check.py in-process (its last stdout line is the {"passed": ...} JSON).
"""
import os
import runpy
import socket
import subprocess
import sys

import tensorrt_llm

HERE = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs")
print(f"[env] host={socket.gethostname()} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} cwd={os.getcwd()}", flush=True)
print("[env]", subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,clocks.sm,memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
for name in ("libtensorrt_llm.so", "libth_common.so"):
    p = os.path.realpath(os.path.join(libdir, name))
    if not os.path.exists(p):
        print(f"[receipt] {p} missing", flush=True)
        continue
    for sym in ("fused_pmap_gemm_fma_mtile_v3", "fused_pmap_gemm_fma_mtile_v2ILi12ELi2E",
                "fused_pmap_gemm_fma_mtile_v2ILi24ELi4E"):
        c = subprocess.run(f"strings '{p}' | grep -c {sym}", shell=True, capture_output=True, text=True).stdout.strip()
        print(f"[receipt] {p} mtime={os.path.getmtime(p):.0f} strings|grep -c {sym} = {c}", flush=True)
sys.argv = [os.path.join(HERE, "s1_check.py")]
runpy.run_path(os.path.join(HERE, "s1_check.py"), run_name="__main__")
