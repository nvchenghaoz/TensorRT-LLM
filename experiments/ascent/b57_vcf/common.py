"""Branch 57 (idea 96): conflict-free G3 V smem layout (lever v_cf, swap23 V map).

Arms (all built in one process from this tree):
  base   load_trim_levers(ROUTED_LEVERS) without v_cf   (main's served G3 module = kill switch)
  cand   base + v_cf
  base2  base + aa_copy (a no-op macro, own URI / .so: the in-process A/A copy of base)
Each arm has a BF16-out module and an nvfp4_out module (levers + nvfp4_out).
In the baseline tree of run_gpu_pair, the G3 header has no FMHA_FLUX3_V_CF code, so "cand"
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
QR = 16384


def lib_mod():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lib._LEVER_FLAGS.setdefault("v_cf", ("vcf", "-DFMHA_FLUX3_V_CF=1"))
    lib._LEVER_FLAGS.setdefault("aa_copy", ("aa", "-DFMHA_FLUX3_AA_COPY=1"))
    return lib


def arm_levers(lib, sample):
    routed = tuple(lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]))
    assert "g3" in routed and "v_b8t" in routed, routed
    base = tuple(x for x in routed if x != "v_cf")
    return routed, {"base": base, "cand": (*base, "v_cf"), "base2": (*base, "aa_copy")}


def build(lib, levs, arms):
    mods = {}
    for a in arms:
        t0 = time.time()
        mods[a] = {"bf16": lib.get_flux3_d128_module(tuple(levs[a])),
                   "nfo": lib.get_flux3_d128_module((*levs[a], "nvfp4_out"))}
        log(f"[build] {a} {lib._tag(levs[a], 0)} {time.time() - t0:.1f}s")
    return mods


def call_nfo_q(lib, mod, inp, out, gscale, q_rows=QR):
    sc = inp["scales"]
    lib.fmha_v2_flux3_d128_nvfp4_out_q_rows(inp["q"], inp["k"], inp["v"], out.fp4, out.sf, gscale,
                                            b.H, b.D, inp["S"], q_rows, scale_bmm1_d=sc[3:4],
                                            scale_bmm2_d=sc[2:3], module=mod)


def so_files(lib, levers):
    import glob

    from flashinfer.jit import env as jit_env

    uri = f"trtllm_fmha_v2_flux3_d128_{lib._tag(levers, 0)}"
    sos = glob.glob(str(jit_env.FLASHINFER_JIT_DIR / uri / "**" / "*.so"), recursive=True)
    sos += glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / f"{uri}*.so"), recursive=True)
    sos += glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / uri / "*.o"), recursive=True)
    return uri, sos


OPS = ("PRMT", "LDSM", "QMMA", "SHFL", "STL", "LDL", "SEL", "MOV", "BAR", "SYNCS", "LDS", "STS",
       "UTMALDG")


def is_g3(name):
    return name.endswith("_g3") or name.endswith("_g3_vcf")


def sass_census(lib, levers):
    """REG/STACK line and opcode counts of every G3 kernel in the module."""
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
    out["res_usage"] = {k: v for k, v in res.items() if is_g3(k)}
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
    out["kernels"] = {}
    for n, body in funcs.items():
        if not is_g3(n):
            continue
        c = {}
        for s in body:
            t = s.split()
            if t and t[0].startswith("@"):
                t = t[1:]
            if not t:
                continue
            root = t[0].split(".")[0]
            if root in OPS:
                c[root] = c.get(root, 0) + 1
        ldsm = sorted({s.split()[0] if not s.split()[0].startswith("@") else s.split()[1]
                       for s in body if "LDSM" in s})
        out["kernels"][n[-12:]] = {"ops": c, "ldsm_forms": ldsm, "n_instr": len(body)}
    return out


def resources_ok(sc):
    """REG <= 168, STACK 0 and no STL/LDL in every G3 kernel."""
    bad = []
    for k, v in sc.get("res_usage", {}).items():
        m = re.search(r"REG:(\d+)", v)
        st = re.search(r"STACK:(\d+)", v)
        if not m or int(m.group(1)) > 168:
            bad.append(f"reg_{k[-12:]}")
        if st and int(st.group(1)) != 0:
            bad.append(f"stack_{k[-12:]}")
    for k, v in sc.get("kernels", {}).items():
        if v["ops"].get("STL", 0) or v["ops"].get("LDL", 0):
            bad.append(f"local_{k}")
    return bad
