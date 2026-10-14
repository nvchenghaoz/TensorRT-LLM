#!/usr/bin/env python3
# Exploration only: dump the SASS of the multi-block merge section (from the semaphore atomic on) of a variant.
import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xqa_bench as xb  # noqa: E402
import xqa_explore as xe  # noqa: E402

INST = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="base")
    p.add_argument("--base-commit", default=xb.BASE_COMMIT)
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--skip", type=int, default=0)
    a = p.parse_args()
    base = xb.extract_base_tree(a.base_commit)
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    for spec in a.variants.split(","):
        cubin = xe.make_variant(spec, tree, base)
        path = os.path.join(tempfile.mkdtemp(), f"{spec}.cubin")
        with open(path, "wb") as f:
            f.write(cubin)
        sass = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout
        insts = [(m.group(1), m.group(2)) for m in (INST.match(l) for l in sass.splitlines()) if m]
        idx = [i for i, (_, s) in enumerate(insts) if "ATOM" in s and "INC" in s]
        start = (idx[-1] if idx else 0) + a.skip
        xb.log(f"===== {spec}: {len(insts)} instructions; merge section from #{start} =====")
        for addr, s in insts[start:start + a.count]:
            xb.log(f"{addr} {s}")


if __name__ == "__main__":
    main()
