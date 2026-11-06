"""Idea 386: SASS of every csa2_mxfp8_low_m_gemm_kernel instance in this tree's build, hashed per instance.

Run in both arms of a run_gpu_pair (base = main, candidate = branch): the K1FOLD=false instances must hash equal.
Key: the instance's template arguments (WARPS, CLUSTER_K, RT, STAGES, NORM, PF), with the branch's trailing
K1FOLD=false dropped; the hash covers the function's SASS text minus the lines that name the function (its mangled
name changes with the added template argument and kernel parameter). --discover prints the candidate files only.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys


def log(*a):
    print(*a, flush=True, file=sys.stderr)


def tool(name):
    p = shutil.which(name)
    if p:
        return p
    for d in ("/usr/local/cuda/bin", "/usr/local/cuda-13/bin"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def candidates():
    roots = [os.getcwd()]
    try:
        import tensorrt_llm

        roots.append(os.path.dirname(os.path.dirname(tensorrt_llm.__file__)))
    except Exception as e:  # noqa: BLE001
        log("import tensorrt_llm failed", e)
    objs, libs = [], []
    for r in roots:
        objs += glob.glob(os.path.join(r, "cpp", "build*", "**", "mxfp8LowMGemm.cu.o"), recursive=True)
        objs += glob.glob(os.path.join(r, "build*", "**", "mxfp8LowMGemm.cu.o"), recursive=True)
        libs += glob.glob(os.path.join(r, "tensorrt_llm", "libs", "libtensorrt_llm.so"))
    return sorted(set(objs)), sorted(set(libs)), roots


def main():
    objs, libs, roots = candidates()
    cuobjdump, cufilt = tool("cuobjdump"), tool("cu++filt")
    info = {"roots": roots, "objs": objs, "libs": libs, "cuobjdump": cuobjdump, "cu++filt": cufilt}
    log(json.dumps(info))
    if "--discover" in sys.argv:
        print(json.dumps({"passed": True, **info}))
        return
    src = objs[0] if objs else (libs[0] if libs else None)
    if src is None or cuobjdump is None:
        print(json.dumps({"passed": False, "error": "no object/library or cuobjdump", **info}))
        return
    log(f"cuobjdump -sass {src}")
    out = subprocess.run([cuobjdump, "-sass", src], capture_output=True, text=True, timeout=3000).stdout
    funcs = {}
    cur, body = None, []
    for line in out.splitlines():
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
    names = [n for n in funcs if "csa2_mxfp8_low_m_gemm_kernel" in n]
    dem = {}
    if cufilt and names:
        r = subprocess.run([cufilt], input="\n".join(names), capture_output=True, text=True)
        for n, d in zip(names, r.stdout.splitlines()):
            dem[n] = d
    hashes = {}
    for n in names:
        d = dem.get(n, n)
        mt = re.search(r"csa2_mxfp8_low_m_gemm_kernel<([^>]*)>", d)
        args = [a.strip() for a in mt.group(1).split(",")] if mt else [n]
        if len(args) == 7 and args[6] == "false":
            args = args[:6]
        key = ",".join(args)
        for bi, body in enumerate(funcs[n]):
            text = "\n".join(line for line in body if n not in line)
            hashes[f"{key}#{bi}"] = {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                                     "lines": len(body)}
    print(json.dumps({"passed": bool(hashes), "source": src, "instances": len(hashes), "hashes": hashes}))


if __name__ == "__main__":
    main()
