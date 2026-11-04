"""Step 0 gate for the spill-free fused_pmap_gemm_fma_mtile build (idea 144). One GPU, no rebuild.

Receipts:
  1. REG/STACK/LOCAL of every fused_pmap_gemm_fma_mtile<TN,TM> (and the ksplit<24,...> reference)
     instantiation in the BUILT library (cuobjdump -res-usage, sm_120 cubin), plus LDL/STL counts
     in their SASS.
  2. A standalone nvcc -Xptxas -v compile of the same header (explicit instantiations) for the
     spill-store/spill-load bytes breakdown, and whether the PTX already has a .local depot
     (local arrays from dynamic indexing exist BEFORE register allocation; spills do not).
  3. The autotuner's chosen fused_hc tactic at M=2048/4096/8192 (V4.1 lagged mode with fused
     norm, as hyper_connection.fused_hc_lagged calls it), plus every valid tactic's time.
Last stdout line: {"passed": true, ...} JSON summary.
"""
import glob
import json
import mmap
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch

import tensorrt_llm
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.mhc import mhc_cuda

HC, HID = 4, 5120
N, K = HC * (2 + HC), HC * HID
ARGS = (HC, HID, 1e-6, 1e-6, 1e-6, 2.0, 20)
MT = ((24, 2), (24, 3), (24, 4), (12, 2), (12, 4), (8, 4))
summary = {}

root = os.path.dirname(tensorrt_llm.__file__)
cuda_bin = "/usr/local/cuda/bin"
cuobjdump = shutil.which("cuobjdump") or os.path.join(cuda_bin, "cuobjdump")
nvcc = shutil.which("nvcc") or os.path.join(cuda_bin, "nvcc")


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout
    return out.splitlines()


# ---------------------------------------------------------------- 1. built library
libs = []
for p in sorted(glob.glob(os.path.join(root, "libs", "*.so*"))):
    with open(p, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            continue
        if mm.find(b"fused_pmap_gemm_fma_mtile") >= 0:
            libs.append(p)
        mm.close()
print("libs with mtile:", libs, flush=True)
built = {}
for lib in libs:
    txt = subprocess.run([cuobjdump, "-res-usage", lib], capture_output=True, text=True, timeout=900).stdout
    arch = None
    cur = None
    for line in txt.splitlines():
        m = re.search(r"arch = (sm_\d+\w*)", line)
        if m:
            arch = m.group(1)
            continue
        m = re.search(r"Function (_Z\w*fused_pmap_gemm_fma_(?:mtile|ksplit)\w*):", line)
        if m:
            cur = m.group(1)
            continue
        if cur and "REG:" in line:
            built.setdefault(cur, []).append({"arch": arch, "lib": os.path.basename(lib), "res": line.strip()})
            cur = None
names = sorted(built)
dem = dict(zip(names, demangle(names)))
built_rows = {}
for mangled in names:
    d = dem[mangled]
    short = re.sub(r"void fused_fma_kernels::", "", d).split("(")[0]
    for e in built[mangled]:
        print(f"[built] {e['arch']} {short}: {e['res']}", flush=True)
    sm120 = [e for e in built[mangled] if e["arch"] and e["arch"].startswith("sm_120")]
    if not sm120:
        continue
    e = sm120[0]
    r = dict(re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", e["res"]))
    sass = subprocess.run([cuobjdump, "-sass", "-arch", e["arch"], "-fun", mangled,
                           os.path.join(root, "libs", e["lib"])],
                          capture_output=True, text=True, timeout=900).stdout
    ins = [l for l in sass.splitlines() if re.match(r"\s*/\*[0-9a-f]{4,}\*/", l)]
    cnt = lambda op: sum(1 for l in ins if re.search(r"\b" + op + r"(\.|\s)", l))  # noqa: E731
    built_rows[short] = {"REG": int(r.get("REG", -1)), "STACK": int(r.get("STACK", -1)),
                         "LOCAL": int(r.get("LOCAL", -1)), "SHARED": int(r.get("SHARED", -1)),
                         "n_sass": len(ins), "LDL": cnt("LDL"), "STL": cnt("STL"), "FFMA": cnt("FFMA"),
                         "LDG": cnt("LDG"), "arch": e["arch"]}
    print(f"[built-sass] {short}: {built_rows[short]}", flush=True)
summary["built"] = built_rows

# ---------------------------------------------------------------- 2. standalone ptxas -v + PTX
hdr = os.path.abspath("cpp/tensorrt_llm/kernels/mhcKernels/mhc_fused_fma.cuh")
standalone = {}
if os.path.exists(hdr) and os.path.exists(nvcc):
    td = tempfile.mkdtemp()
    src = os.path.join(td, "inst.cu")
    with open(src, "w") as f:
        f.write(f'#include "{hdr}"\n')
        for tn, tm in MT:
            f.write(f"template __global__ void fused_fma_kernels::fused_pmap_gemm_fma_mtile<{tn},{tm}>("
                    "__nv_bfloat16 const*, __nv_bfloat16 const*, float const*, float const*, float const*,"
                    " float*, float*, int, int, int, int, __nv_bfloat16*);\n")
    base = [nvcc, "-std=c++17", "-O3", "-gencode", "arch=compute_120,code=sm_120", "-I", os.path.dirname(hdr)]
    r = subprocess.run(base + ["-Xptxas", "-v", "-cubin", "-o", os.path.join(td, "i.cubin"), src],
                       capture_output=True, text=True, timeout=600)
    log = r.stderr + r.stdout
    print("[nvcc rc]", r.returncode, flush=True)
    fn = None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            fn = demangle([m.group(1)])[0]
            fn = re.sub(r"void fused_fma_kernels::", "", fn).split("(")[0]
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and fn and "mtile" in fn:
            standalone.setdefault(fn, {}).update(stack_frame=int(m.group(1)), spill_st=int(m.group(2)),
                                                 spill_ld=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m and fn and "mtile" in fn:
            standalone.setdefault(fn, {})["regs"] = int(m.group(1))
    if r.returncode != 0:
        print(log[-4000:], flush=True)
    r = subprocess.run(base + ["-ptx", "-o", os.path.join(td, "i.ptx"), src], capture_output=True, text=True,
                       timeout=600)
    if r.returncode == 0:
        ptx = open(os.path.join(td, "i.ptx")).read()
        for ent in re.split(r"\.visible \.entry ", ptx)[1:]:
            mname = re.match(r"(\w+)", ent).group(1)
            dn = re.sub(r"void fused_fma_kernels::", "", demangle([mname])[0]).split("(")[0]
            body = ent.split("\n}\n")[0]
            standalone.setdefault(dn, {})["ptx_local_depot"] = bool(re.search(r"\.local\s", body))
            standalone[dn]["ptx_ld_local"] = len(re.findall(r"ld\.local", body))
            standalone[dn]["ptx_st_local"] = len(re.findall(r"st\.local", body))
    for k, v in standalone.items():
        print(f"[standalone] {k}: {v}", flush=True)
summary["standalone_ptxas"] = standalone

# ---------------------------------------------------------------- 3. autotuner choice + times
props = torch.cuda.get_device_properties(0)
print("device", props.name, props.multi_processor_count, "mma_supported", mhc_cuda._fused_hc_mma_supported(),
      flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mhc_fhc_bench import make_inputs, time_op  # noqa: E402

flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
rn = mhc_cuda.MhcFusedHcRunner(*ARGS)
chosen = {}
for M in (2048, 4096, 8192):
    inputs, nw = make_inputs(M)
    pre_ext = torch.rand((M, HC), device="cuda").float() + 0.5
    kw = dict(norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre_ext)
    with autotune(skip_dynamic_tuning_buckets=True):
        mhc_cuda.mhc_fused_hc(*inputs, *ARGS, **kw)
    torch.cuda.synchronize()
    t = mhc_cuda.mhc_fused_hc_tactic(*inputs, *ARGS, **kw)
    times = {}
    for tac in rn.get_valid_tactics(inputs, None):
        fn = lambda tac=tac: rn.forward(inputs, tactic=tac, **kw)  # noqa: E731
        for _ in range(3):
            fn()
        med, nz = time_op(fn, 20, flush)
        times[str(tac)] = round(med * 1000, 1)
    best = min(times, key=times.get)
    chosen[str(M)] = {"autotuner": list(t), "us_chosen": times.get(str(tuple(t))), "best_timed": best,
                      "us_best": times[best], "all_us": dict(sorted(times.items(), key=lambda kv: kv[1]))}
    print(f"[tactic] M={M}: {chosen[str(M)]}", flush=True)
    del inputs
    torch.cuda.empty_cache()
summary["chosen"] = chosen
print(json.dumps({"passed": True, **summary}))
