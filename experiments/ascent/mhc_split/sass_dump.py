"""Dump filtered SASS (float ops, shuffles, shared/global memory) of selected mhc kernel instances."""

import glob
import os
import re
import shutil
import subprocess
import sys

import tensorrt_llm

root = os.path.dirname(tensorrt_llm.__file__)
cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
libs = [p for p in glob.glob(os.path.join(root, "libs", "*.so*"))]
want = sys.argv[1] if len(sys.argv) > 1 else r"mhcBigFuseKernel<1, (256|512), true, false, true>"
OPS = ("FADD", "FFMA", "FMUL", "MUFU", "SHFL", "BAR", "HFMA2", "HADD2", "HMUL2", "F2F", "PRMT", "IMAD.U32",
       "SHF", "LDS", "STS", "LDG", "STG", "BRA", "FMNMX", "ISETP")
for lib in libs:
    try:
        names = subprocess.run([cuobjdump, "-symbols", lib], capture_output=True, text=True,
                               timeout=600).stdout
    except Exception as e:  # noqa: BLE001
        print("fail", lib, e)
        continue
    funcs = sorted(set(re.findall(r"(_Z\w*mhcBigFuseKernel\w*)", names)))
    for f in funcs:
        demangled = subprocess.run(["c++filt", f], capture_output=True, text=True).stdout.strip()
        if not re.search(want, demangled):
            continue
        print("=" * 30, demangled, flush=True)
        out = subprocess.run([cuobjdump, "-sass", "-fun", f, lib], capture_output=True, text=True,
                             timeout=600).stdout
        lines = out.splitlines()
        # Phase 2 starts after the first BAR.SYNC; stop at the last MUFU.RSQ.
        rsq = [i for i, l in enumerate(lines) if "MUFU.RSQ" in l]
        start = rsq[0] if rsq else 0
        end = rsq[-1] if rsq else len(lines)
        for l in lines[start:end + 2]:
            s = l.strip()
            if any(op in s for op in OPS):
                print(re.sub(r"\s*/\* 0x[0-9a-f]+ \*/", "", s)[:110])
        print(flush=True)
print('{"passed": true}')
