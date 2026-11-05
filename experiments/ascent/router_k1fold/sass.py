"""SASS identity of the existing dsv4FusedRouterKernel instances (idea 313 ruling (i)).

Extracts the dsv4FusedRouter cubin(s) from the built tensorrt_llm library, dumps SASS per function, and prints a
sha256 of every dsv4FusedRouterKernel<M, kRoute[, false]> body (instruction text only, the function name and
symbol references to it normalized: main's <M, kRoute> and the branch's <M, kRoute, false> are the same kernel).
Run in both trees (run_gpu_pair); the per-instance hashes must match. The branch's fold instances
(<M, kRoute, true>) are listed separately.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile


def log(*a):
    print(*a, flush=True)


def main():
    import tensorrt_llm
    root = os.path.dirname(tensorrt_llm.__file__)
    libs = sorted(glob.glob(os.path.join(root, "libs", "*.so")))
    cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    found = {}
    for lib in libs:
        try:
            lst = subprocess.run([cuobjdump, "-lelf", lib], capture_output=True, text=True, timeout=600).stdout
        except Exception as e:  # noqa: BLE001
            log("skip", lib, e)
            continue
        names = [l.split(":", 1)[1].strip() for l in lst.splitlines() if "dsv4FusedRouter" in l and ":" in l]
        if not names:
            continue
        log("lib", lib, "elfs", names)
        with tempfile.TemporaryDirectory() as td:
            for n in names:
                subprocess.run([cuobjdump, "-xelf", n, lib], cwd=td, check=True, capture_output=True, timeout=600)
                cubin = os.path.join(td, n)
                sass = subprocess.run([cuobjdump, "-sass", cubin], capture_output=True, text=True,
                                      timeout=600).stdout
                funcs = re.split(r"\n\s*Function : ", sass)
                for f in funcs[1:]:
                    fname = f.split("\n", 1)[0].strip()
                    dem = subprocess.run(["c++filt", fname], capture_output=True, text=True).stdout.strip()
                    if "dsv4FusedRouterKernel" not in dem:
                        continue
                    mm = re.search(r"dsv4FusedRouterKernel<(\d+), (true|false)(?:, (true|false))?>", dem)
                    key = f"{mm.group(1)},{mm.group(2)}" + (",fold" if mm.group(3) == "true" else "")
                    body = f.split("\n", 1)[1]
                    lines = []
                    for line in body.splitlines():
                        s = re.sub(r"/\*[0-9a-f]{4,}\*/", "", line)  # addresses
                        s = s.replace(fname, "FN")
                        s = s.strip()
                        if s and not s.startswith(".") and "Function :" not in s:
                            lines.append(s)
                    found[f"{n}:{key}"] = {"sha": hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16],
                                           "instructions": len(lines)}
    for k in sorted(found):
        log(k, found[k])
    base = {k: v for k, v in found.items() if not k.endswith(",fold")}
    print(json.dumps({"batch": {"1": {"kernel_ms": 1.0, "noise_floor": 0.0}}, "output": {"class": "exact"},
                      "instances": found, "n_existing": len(base),
                      "existing_digest": hashlib.sha256(json.dumps(
                          {k.split(":", 1)[1]: v["sha"] for k, v in sorted(base.items())}, sort_keys=True
                      ).encode()).hexdigest()[:16]}))


if __name__ == "__main__":
    sys.exit(main())
