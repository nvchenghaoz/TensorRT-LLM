"""Compile this tree's dsv3RouterGemm.cu standalone (nvcc, sm_120) and audit the router MMA kernels.

Per router_gemm_mma_kernel / router_latent_gemm_mma_kernel instance:
  - ptxas -v registers / spill stores / spill loads;
  - PTX: every ld.global before griddepcontrol.wait, with the kernel parameter its address derives from
    (backward slice over the PTX def chain). Pass: 384x5120 instances load only from param_2 (mat_b = weight)
    before the wait; every other instance (896x7168 K3, latent 896x7168) has no load before the wait;
  - SASS (cuobjdump): number of LDG before the first instruction that is not a prologue load (printed head).
Last stdout line: JSON {"passed": ..., "instances": {...}}.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC = "cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu"


def log(*a):
    print(*a, flush=True)


def nvcc():
    for c in (shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and os.path.exists(c):
            return c
    raise RuntimeError("nvcc not found")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def tmpl_of(name):
    m = re.search(r"router_gemm_mma_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", name)
    if m:
        return "router_gemm_mma_kernel<%s,%s,%s,%s,%s>" % m.groups()
    m = re.search(r"router_latent_gemm_mma_kernelILi(\d+)E", name)
    if m:
        return "router_latent_gemm_mma_kernel<%s>" % m.group(1)
    return None


def ptx_audit(ptx):
    res = {}
    for m in re.finditer(r"\.entry\s+(\S+)\s*\((.*?)\)\s*(\.maxntid[^{]*)?\{(.*?)\n\}", ptx, re.S):
        name, body = m.group(1), m.group(4)
        t = tmpl_of(name)
        if t is None:
            continue
        lines = [l.strip() for l in body.split("\n")]
        defs = {}
        wait_at = None
        for i, l in enumerate(lines):
            if "griddepcontrol.wait" in l:
                wait_at = i
                break
        loads = []
        for i, l in enumerate(lines[: wait_at if wait_at is not None else len(lines)]):
            mm = re.match(r"([a-z][\w.]*)\s+(.*);", l)
            if not mm:
                continue
            op, args = mm.group(1), mm.group(2)
            regs = re.findall(r"%[a-z]+\d+", args)
            if op.startswith("ld.param"):
                pm = re.search(r"\[(\w+)\]", args)
                if regs:
                    defs[regs[0]] = ("param", pm.group(1) if pm else "?")
                continue
            if op.startswith("ld.global") or op.startswith("ld.") and "global" in op:
                am = re.search(r"\[(%[a-z]+\d+)", args)
                loads.append((i, op, am.group(1) if am else None))
            if regs and not op.startswith(("st.", "bar", "mma", "griddep")):
                defs.setdefault(regs[0], ("op", regs[1:]))

        def params(r, seen=None):
            seen = seen or set()
            if r in seen or r not in defs:
                return set()
            seen.add(r)
            kind, v = defs[r]
            if kind == "param":
                return {v}
            out = set()
            for s in v:
                out |= params(s, seen)
            return out

        ld = []
        for i, op, a in loads:
            ps = sorted(params(a)) if a else ["?"]
            ld.append({"op": op, "params": [p.split("_")[-1] for p in ps]})
        res[t] = {"has_wait": wait_at is not None, "loads_before_wait": len(ld),
                  "load_params_before_wait": sorted({p for x in ld for p in x["params"]})}
    return res


def sass_audit(sass):
    res, heads = {}, {}
    for blk in re.split(r"\n\s*Function : ", sass)[1:]:
        name = blk.split("\n", 1)[0].strip()
        t = tmpl_of(name)
        if t is None:
            continue
        ins = re.findall(r"/\*[0-9a-f]{4,}\*/\s+([^;]*);", blk)
        res[t] = {"n_instr": len(ins)}
        heads[t] = ins[:90]
    return res, heads


def main():
    tree = os.getcwd()
    out = tempfile.mkdtemp(prefix="router_sass_")
    inc = ["-I", os.path.join(tree, "cpp/include"), "-I", os.path.join(tree, "cpp")]
    common = [nvcc(), "-std=c++17", "-O3", "-gencode=arch=compute_120,code=sm_120", "-DENABLE_BF16", "-DENABLE_FP8",
              "--expt-relaxed-constexpr"] + inc
    cub = os.path.join(out, "r.cubin")
    ptxf = os.path.join(out, "r.ptx")
    rc, so, se = run(common + ["-cubin", "-Xptxas", "-v", "-o", cub, SRC])
    if rc:
        log("nvcc cubin failed", se[-4000:])
        print(json.dumps({"passed": False, "error": "nvcc cubin failed"}))
        return
    regs = {}
    cur = None
    for l in se.split("\n"):
        m = re.search(r"Compiling entry function '(\S+)'", l)
        if m:
            cur = tmpl_of(m.group(1))
        m = re.search(r"Function properties for (\S+)", l)
        if m:
            cur = tmpl_of(m.group(1))
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", l)
        if m:
            regs.setdefault(cur, {})["spill_st"] = int(m.group(1))
            regs[cur]["spill_ld"] = int(m.group(2))
        m = re.search(r"Used (\d+) registers", l)
        if m:
            regs.setdefault(cur, {})["regs"] = int(m.group(1))
    rc, so, se = run(common + ["-ptx", "-o", ptxf, SRC])
    ptx = open(ptxf).read() if rc == 0 else ""
    pa = ptx_audit(ptx)
    rc, sass, se = run([os.path.join(os.path.dirname(nvcc()), "cuobjdump"), "-sass", cub])
    sa, heads = sass_audit(sass)
    inst = {}
    ok = True
    for t in sorted(set(regs) | set(pa)):
        rec = dict(regs.get(t, {}))
        rec.update(pa.get(t, {}))
        rec.update(sa.get(t, {}))
        v41 = ",384,5120," in t
        if v41:
            good = rec.get("load_params_before_wait") in (["param_2"], ["2"]) and rec.get("loads_before_wait", 0) > 0
        else:
            good = rec.get("loads_before_wait", 0) == 0
        good = good and rec.get("spill_st", 0) == 0 and rec.get("spill_ld", 0) == 0 and rec.get("has_wait", False)
        rec["v41"] = v41
        rec["prewait_audit_ok"] = good
        inst[t] = rec
        log(t, json.dumps(rec))
    # The prewait audit expects the candidate layout; on the base tree 384x5120 has no pre-wait loads.
    arm = sys.argv[1] if len(sys.argv) > 1 else "ARM?"
    for t in ("router_gemm_mma_kernel<1,384,5120,8,640>", "router_gemm_mma_kernel<16,384,5120,8,640>",
              "router_gemm_mma_kernel<1,896,7168,8,448>", "router_latent_gemm_mma_kernel<1>"):
        log(f"==== {arm} SASS head {t}")
        for l in heads.get(t, [])[:90]:
            log("   ", l)
    ok = all(r["prewait_audit_ok"] for r in inst.values()) and len(inst) >= 48
    summary = {t: {k: r.get(k) for k in ("regs", "spill_st", "spill_ld", "loads_before_wait",
                                         "load_params_before_wait")} for t, r in inst.items()}
    print(json.dumps({"passed": ok, "arm": arm, "n_instances": len(inst),
                      "compared": "PTX loads before griddepcontrol.wait traced to kernel params; ptxas -v regs/spills",
                      "instances": summary}))


if __name__ == "__main__":
    main()
