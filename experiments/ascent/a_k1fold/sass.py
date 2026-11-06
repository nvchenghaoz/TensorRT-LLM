"""Idea 386: SASS of every csa2_mxfp8_low_m_gemm_kernel instance in this tree's build, hashed per instance.

Run in both arms of a run_gpu_pair (base = main, candidate = branch): the K1FOLD=false instances must hash equal.
Finds the library holding the kernel (byte search of tensorrt_llm/libs/*.so for the mangled name), then dumps only
those functions (cuobjdump -sass -fun <mangled names>). Key: the instance's template arguments (WARPS, CLUSTER_K, RT,
STAGES, NORM, PF), with the branch's trailing K1FOLD=false dropped; the hash covers the function's SASS text minus the
lines that name the function (its mangled name changes with the added template argument and kernel parameter).
"""

import glob
import hashlib
import json
import mmap
import os
import re
import shutil
import subprocess
import sys

PAT = re.compile(rb"_Z[A-Za-z0-9_$.]*csa2_mxfp8_low_m_gemm_kernel[A-Za-z0-9_$.]*")


def log(*a):
    print(*a, flush=True)


def tool(name):
    p = shutil.which(name)
    if p:
        return p
    for d in ("/usr/local/cuda/bin",):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def main():
    import tensorrt_llm

    libdir = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs")
    cuobjdump, cufilt = tool("cuobjdump"), tool("cu++filt")
    found = {}
    for lib in sorted(glob.glob(os.path.join(libdir, "*.so*"))):
        try:
            with open(lib, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                names = sorted(set(m.group(0).decode() for m in PAT.finditer(mm)))
        except Exception as e:  # noqa: BLE001
            log("skip", lib, e)
            continue
        if names:
            found[lib] = names
            log(f"{lib}: {len(names)} mangled names")
    if not found or cuobjdump is None:
        print(json.dumps({"passed": False, "error": "kernel not found or no cuobjdump", "libdir": libdir}))
        return
    hashes, dumped = {}, {}
    for lib, names in found.items():
        # Device names only (host stubs share the name); cuobjdump ignores names it does not find.
        r = subprocess.run([cuobjdump, "-sass", "-fun", ",".join(names), lib], capture_output=True, text=True,
                           timeout=3000)
        funcs, cur, body = {}, None, []
        for line in r.stdout.splitlines():
            mm = re.match(r"\s*Function : (\S+)", line)
            if mm:
                if cur is not None:
                    funcs.setdefault(cur, []).append(body)
                cur, body = mm.group(1), []
                continue
            if cur is not None:
                body.append(line)
        if cur is not None:
            funcs.setdefault(cur, []).append(body)
        dumped[lib] = {"functions": len(funcs), "rc": r.returncode, "stderr": r.stderr[-400:]}
        log(f"{lib}: dumped {len(funcs)} functions rc {r.returncode}")
        dem = {}
        if cufilt and funcs:
            rr = subprocess.run([cufilt], input="\n".join(funcs), capture_output=True, text=True)
            dem = dict(zip(funcs, rr.stdout.splitlines()))
        for n, bodies in funcs.items():
            d = dem.get(n, n)
            mt = re.search(r"csa2_mxfp8_low_m_gemm_kernel<([^>]*)>", d)
            args = [a.strip() for a in mt.group(1).split(",")] if mt else [n]
            if len(args) == 7 and args[6] == "false":
                args = args[:6]
            key = ",".join(args)
            for bi, body in enumerate(bodies):
                text = "\n".join(line for line in body if n not in line)
                hashes[f"{key}#{bi}"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    print(json.dumps({"passed": bool(hashes), "instances": len(hashes), "libs": list(found), "dumped": dumped,
                      "hashes": dict(sorted(hashes.items()))}))


if __name__ == "__main__":
    main()
