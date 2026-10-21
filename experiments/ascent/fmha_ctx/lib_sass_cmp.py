#!/usr/bin/env python3
"""SASS identity of the BUILT TensorRT-LLM libraries (idea 244), base tree vs candidate tree on one host.

  lib_sass_cmp.py save     (baseline arm: the main sha's build)  -> /tmp/fmha244_sass/base/
  lib_sass_cmp.py compare  (candidate arm)                       -> prints the diff
  lib_sass_cmp.py compare <base_libs_dir>  (base libs copied there by the baseline arm)

Every .so under the tensorrt_llm package's libs/ is unpacked with cuobjdump -xelf all. Each cubin's bytes are hashed;
the base and candidate multisets of hashes are matched. Byte-identical cubins have identical SASS. For each cubin
without a byte-identical partner, cuobjdump -sass gives per-function hashes of the instruction text, and the functions
are matched by name between the unmatched base and candidate cubins (the base keeps the cubins that name fmha /
attention kernels). Prints the functions whose SASS differs. The last stdout line is a JSON summary.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = "/tmp/fmha244_sass"
CUOBJDUMP = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
KEEP = (b"fmha", b"attention", b"last_tile")


def log(*a):
    print(*a, flush=True)


def lib_dir():
    spec = importlib.util.find_spec("tensorrt_llm")
    pkg = list(spec.submodule_search_locations)[0]
    return os.path.join(pkg, "libs")


def unpack(dst, src=None):
    src = src or lib_dir()
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    libs = sorted(f for f in os.listdir(src) if f.endswith(".so") or ".so." in f)
    entries = []
    for lib in libs:
        path = os.path.join(src, lib)
        if os.path.islink(path):
            continue
        d = os.path.join(dst, lib)
        os.makedirs(d)
        t0 = time.time()
        r = subprocess.run([CUOBJDUMP, "-xelf", "all", path], cwd=d, capture_output=True, text=True)
        n = 0
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            b = open(fp, "rb").read()
            n += 1
            keep = any(k in b for k in KEEP)
            entries.append(dict(lib=lib, elf=f, sha=hashlib.sha256(b).hexdigest(), size=len(b), keep=keep))
            if not keep:
                os.remove(fp)
        log(f"[unpack] {lib}: {n} cubins in {time.time() - t0:.1f}s (rc {r.returncode}, "
            f"lib sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]})")
    return entries


def func_hashes(cubin):
    out = subprocess.run([CUOBJDUMP, "-sass", cubin], capture_output=True, text=True).stdout
    funcs, cur, h = {}, None, None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if cur is not None:
                funcs[cur] = h.hexdigest()
            cur, h = m.group(1), hashlib.sha256()
            continue
        m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if m and cur is not None:
            h.update(m.group(1).encode() + b"\n")
    if cur is not None:
        funcs[cur] = h.hexdigest()
    return funcs


def short(name):
    return name if len(name) < 160 else name[:150] + "..."


def main():
    mode = sys.argv[1]
    log(f"[sass] mode {mode} libs {lib_dir()} commit "
        + subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    if mode == "save":
        entries = unpack(f"{ROOT}/base")
        json.dump(entries, open(f"{ROOT}/base_index.json", "w"))
        log(json.dumps({"saved": len(entries), "kept": sum(e["keep"] for e in entries)}))
        return
    if len(sys.argv) > 2:
        # compare <dir>: the base libs were copied there by the baseline arm (main has no copy of this script).
        base = unpack(f"{ROOT}/base", sys.argv[2])
    else:
        base = json.load(open(f"{ROOT}/base_index.json"))
    cand = unpack(f"{ROOT}/cand")
    pool = {}
    for e in base:
        pool.setdefault(e["sha"], []).append(e)
    un_c = []
    for e in cand:
        if pool.get(e["sha"]):
            pool[e["sha"]].pop()
        else:
            un_c.append(e)
    un_b = [e for v in pool.values() for e in v]
    log(f"[sass] cubins base {len(base)} cand {len(cand)}; byte-identical {len(cand) - len(un_c)}; "
        f"unmatched base {len(un_b)} cand {len(un_c)}")
    fb, fc, lost = {}, {}, []
    for e in un_b:
        p = f"{ROOT}/base/{e['lib']}/{e['elf']}"
        if not os.path.exists(p):
            lost.append(e["elf"])
            continue
        for k, v in func_hashes(p).items():
            fb.setdefault(k, []).append(v)
    for e in un_c:
        p = f"{ROOT}/cand/{e['lib']}/{e['elf']}"
        if not os.path.exists(p):
            lost.append(e["elf"])
            continue
        for k, v in func_hashes(p).items():
            fc.setdefault(k, []).append(v)
    differ = sorted(k for k in set(fb) | set(fc) if sorted(fb.get(k, [])) != sorted(fc.get(k, [])))
    same_in_changed = sorted(k for k in set(fb) & set(fc) if sorted(fb[k]) == sorted(fc[k]))
    for k in differ:
        log(f"[sass] DIFFERS: {short(k)} base {len(fb.get(k, []))} cand {len(fc.get(k, []))}")
    log(f"[sass] functions identical inside the changed cubins: {len(same_in_changed)}")

    def kind(k):
        if "last_tile_piece" in k:
            return "piece"
        if "last_tile_merge" in k:
            return "merge"
        if "fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled" in k:
            return "main"
        return "other"

    kinds = [kind(k) for k in differ]
    res = {"cubins_base": len(base), "cubins_cand": len(cand), "byte_identical": len(cand) - len(un_c),
           "unmatched_base": len(un_b), "unmatched_cand": len(un_c), "unreadable_unmatched": lost,
           "differing_functions": [short(k) for k in differ], "differing_kinds": kinds,
           "main_piece_merge_differ": all(x in kinds for x in ("main", "piece", "merge")),
           "only_main_piece_merge_differ": all(x != "other" for x in kinds) and not lost,
           "functions_identical_in_changed_cubins": len(same_in_changed)}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
