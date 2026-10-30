"""Branch 33 (idea 39): G3 V B-fragments via ldmatrix.m16n16.x2.trans.b8 (lever v_b8t).

Shared helpers for check.py / bench.py. Arms (all built in one process from this tree):
  base   load_trim_levers(ROUTED_LEVERS) without v_b8t   (main's served G3 module)
  cand   base + v_b8t
  base2  base + aa_copy (a no-op macro, own URI / .so: the in-process A/A copy of base)
Each arm has a BF16-out module (levers) and an nvfp4_out module (levers + nvfp4_out).
In the baseline tree of run_gpu_pair, the G3 header has no FMHA_FLUX3_V_B8T code, so "cand"
compiles to base there (an extra A/A).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import b186_nfo as b  # noqa: E402

log = b.log


def lib_mod():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lib._LEVER_FLAGS.setdefault("v_b8t", ("vb8", "-DFMHA_FLUX3_V_B8T=1"))
    lib._LEVER_FLAGS.setdefault("aa_copy", ("aa", "-DFMHA_FLUX3_AA_COPY=1"))
    return lib


def arm_levers(lib, sample):
    routed = tuple(lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]))
    assert "g3" in routed, routed
    base = tuple(x for x in routed if x != "v_b8t")
    return routed, {"base": base, "cand": (*base, "v_b8t"), "base2": (*base, "aa_copy")}


def build(lib, levs, arms):
    mods = {}
    for a in arms:
        t0 = time.time()
        mods[a] = {"bf16": lib.get_flux3_d128_module(tuple(levs[a])),
                   "nfo": lib.get_flux3_d128_module((*levs[a], "nvfp4_out"))}
        log(f"[build] {a} {lib._tag(levs[a], 0)} {time.time() - t0:.1f}s")
    return mods


def so_files(lib, levers):
    import glob

    from flashinfer.jit import env as jit_env

    uri = f"trtllm_fmha_v2_flux3_d128_{lib._tag(levers, 0)}"
    sos = glob.glob(str(jit_env.FLASHINFER_JIT_DIR / uri / "**" / "*.so"), recursive=True)
    sos += glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / f"{uri}*.so"), recursive=True)
    sos += glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / uri / "*.o"), recursive=True)
    return uri, sos


OPS = ("PRMT", "LDSM", "QMMA", "SHFL", "STL", "LDL", "SEL", "MOV", "BAR", "SYNCS", "LDS", "STS")


def sass_census(lib, levers):
    """REG/STACK line and opcode counts of the G3 kernel (whole function and the BMM2 loop body
    approximated as the region between the first and last QMMA)."""
    uri, sos = so_files(lib, levers)
    out = {"uri": uri, "files": sos[:3]}
    if not sos:
        out["error"] = "no .so"
        return out
    so = sos[0]
    r = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True, timeout=120)
    cur = None
    res = {}
    for line in r.stdout.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            cur = m.group(1)
            continue
        if cur and "REG:" in line:
            res[cur] = line.strip()
            cur = None
    out["res_usage"] = {k: v for k, v in res.items() if k.endswith("_g3")}
    r = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True, timeout=300)
    funcs, name, buf = {}, None, []
    for line in r.stdout.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            if name:
                funcs[name] = buf
            name, buf = m.group(1), []
            continue
        if name:
            s = re.sub(r"/\*[0-9a-f]{4,}\*/", "", line).strip()
            s = re.sub(r"^\{|\}$", "", s).strip()
            if s and not s.startswith("/*") and not s.startswith("."):
                buf.append(s)
    if name:
        funcs[name] = buf
    for n, body in funcs.items():
        if not n.endswith("_g3"):
            continue
        ops = []
        for s in body:
            t = s.split()
            if not t:
                continue
            if t[0].startswith("@"):
                t = t[1:]
            if t:
                ops.append((t[0], s))

        def count(seq):
            c = {}
            for op, _ in seq:
                root = op.split(".")[0]
                if root in OPS:
                    c[root] = c.get(root, 0) + 1
            return c

        q = [i for i, (op, _) in enumerate(ops) if op.startswith("QMMA")]
        loop = ops[q[0]:q[-1] + 1] if q else []
        out["g3_function"] = n
        out["whole"] = count(ops)
        out["qmma_region"] = count(loop)
        out["ldsm_forms"] = sorted({op for op, _ in ops if op.startswith("LDSM")})
        out["n_instr"] = len(ops)
    return out
