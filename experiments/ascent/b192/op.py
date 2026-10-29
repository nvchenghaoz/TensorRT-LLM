"""Branch 192 (idea 273): X_POLICY L2 hint on the served warp adaLN NVFP4 kernel's 12 x loads.

  op.py check -> HEAD/ancestry/grep probe; PTX counts (createpolicy, cache_hint loads) and
                 regs/local for X_POLICY 0/1/2 and for main's _SRC (git show) at main's defines;
                 X_POLICY=0 cubin == main's cubin; fp4 + full 0xCD-poisoned SF + forced-store
                 x_mod BITWISE X2 (and X1) vs X0 at M=80/4096/4176/4256, select and mode forms,
                 normal and 4x-saturating gs, store on/off; untimed 23-iteration repeat (re-poison
                 every 8) vs iteration 0 and the X0 reference, inputs vs pristine copies; public
                 op default (X2) vs module switch _X_POLICY=0 bitwise.
  op.py bench -> same script both arms (pair): one CUDA graph of 28 blocks, each
                 adaLN(x) (served launch defaults, select M=4176, store_x_mod False)
                 -> L2 polluter (64 MiB copy, stand-in for the GEMM/attention traffic)
                 -> gated residual x_next = x + gate * y (reads x again).
                 Rotates 4 input sets. kernel_ms = graph time / 28 (per block).
"""

import json
import os
import statistics
import subprocess
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk

D = 3072
EPS = 1e-6
dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
g = torch.Generator(device=dev).manual_seed(1234)
REL = "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_sm120_adaln_nvfp4.py"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa
        return f"ERR {e}"


def head_info():
    info = {
        "head": sh("git rev-parse HEAD"),
        "log": sh("git log --oneline -5"),
        "anc_9e499b22": sh("git merge-base --is-ancestor 9e499b22 HEAD; echo rc=$?"),
        "anc_69eac46e": sh("git merge-base --is-ancestor 69eac46e HEAD; echo rc=$?"),
        "gpu": sh("nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv,noheader"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": sh("hostname"),
        "grep_defines": sh(f"grep -n '_DEFAULT_DEFINES = (\"MINB=12\",)' {REL}"),
        "grep_warps": sh(f"grep -n '_DEFAULT_WARPS = 1' {REL}"),
        "x_policy_attr": getattr(wk, "_X_POLICY", None),
    }
    for k, val in info.items():
        print("INFO", k, val, flush=True)
    return info


def heavy(*shape):
    base = torch.randn(*shape, device=dev, generator=g)
    mag = torch.exp(torch.randn(*shape[:-1], 1, device=dev, generator=g) * 2.0)
    chan = torch.exp(torch.randn(shape[-1], device=dev, generator=g) * 1.5)
    return (base * mag * chan).to(torch.bfloat16)


W = (1.0 + 0.3 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
B = (0.2 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)


def make_cases():
    cases = []
    for s in (80, 4096, 4176, 4256):
        x = heavy(1, s, D)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        shift_t, scale_t = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        select = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        select[:, : (40 if s == 80 else 80)] = True
        cases.append((f"select_M{s}", (x, select, scale_t, shift_t)))
        table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        row = table[:, 5:6]
        cases.append((f"mode_M{s}", (x, None, row[..., D: 2 * D], row[..., :D])))
    return cases


def gs_for(x):
    return float(torch.tensor(2688.0 / x.float().abs().max().item(), dtype=torch.float32))


def poisoned(x, byte=0xCD):
    m = x.numel() // D
    xm = torch.empty_like(x)
    xm.view(torch.uint8).fill_(byte)
    return (
        xm,
        torch.full((m, D // 2), byte, dtype=torch.uint8, device=dev),
        torch.full((faq._scale_size(m, D),), byte, dtype=torch.uint8, device=dev),
    )


def same(a, b):
    return (torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
            and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))


def defs_for(xp):
    return tuple(wk._DEFAULT_DEFINES) + ((f"X_POLICY={xp}",) if xp else ())


def call(args, gs_t, store, xp, outs=None):
    x, sel, sc, sh_ = args
    return wk.launch(x, sel, sc, sh_, W, B, gs_t, EPS, store_x_mod=store,
                     outs=outs if outs is not None else poisoned(x), extra_defines=defs_for(xp))


def nvrtc_build(src, defines, warps=1):
    """PTX, cubin and ptxas -v lines with the module's exact options (+ -v)."""
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability()
    prog = wk._check(nvrtc.nvrtcCreateProgram(src.encode(), b"flux3_adaln_nvfp4.cu", 0, [], []))
    base_opts = [
        f"--gpu-architecture=sm_{major}{minor}a".encode(), b"--fmad=false", b"--prec-div=true",
        b"--prec-sqrt=true", b"--ftz=false", b"-std=c++17", f"-DWARPS={warps}".encode(),
        b"-DUSE_PDL=0",
    ] + [f"-D{d}".encode() for d in defines]
    opts = base_opts + [b"--ptxas-options=-v"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    n = wk._check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError(log.decode(errors="replace"))
    ps = wk._check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ps
    wk._check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)
    # cubin without -v (exactly the module's options)
    prog = wk._check(nvrtc.nvrtcCreateProgram(src.encode(), b"flux3_adaln_nvfp4.cu", 0, [], []))
    nvrtc.nvrtcCompileProgram(prog, len(base_opts), base_opts)
    cs = wk._check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * cs
    wk._check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    nvrtc.nvrtcDestroyProgram(prog)
    ptx = ptx.decode(errors="replace")
    lines = [ln.strip() for ln in log.decode(errors="replace").splitlines()
             if "registers" in ln or "stack" in ln or "spill" in ln]
    return {
        "createpolicy": ptx.count("createpolicy.fractional.L2::evict_last"),
        "cache_hint_loads": ptx.count("L2::cache_hint"),
        "ptxas_v": lines,
        "cubin_len": len(cubin),
    }, cubin


def module_attrs(xp):
    from cuda.bindings import driver

    key = (0, wk._DEFAULT_WARPS, False, defs_for(xp))
    wk._compile(torch.device("cuda", 0), wk._DEFAULT_WARPS, False, defs_for(xp))
    fn = wk._MODULES[key]
    attr = driver.CUfunction_attribute
    return {
        "key": str(key),
        "regs": wk.NUM_REGS[key],
        "local_bytes": wk._check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)),
        "ctas_per_sm": wk._check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            fn, wk._DEFAULT_WARPS * 32, 0)),
    }


def main_src():
    for rev in ("69eac46e", "HEAD~1", "HEAD~2"):
        txt = sh(f"git show {rev}:{REL}")
        if "_SRC = r\"\"\"" in txt and "X_POLICY" not in txt:
            ns = {}
            start = txt.index("_SRC = r\"\"\"")
            end = txt.index("\"\"\"", start + 11)
            ns["src"] = txt[start + 11: end]
            ns["defines"] = "_DEFAULT_DEFINES = (\"MINB=12\",)" in txt
            return rev, ns
    return None, None


def run_check():
    info = head_info()
    ok_all = True
    report = {}
    # ---- PTX / ptxas / cubin ----
    builds = {}
    cubins = {}
    for xp in (0, 1, 2):
        builds[f"X{xp}"], cubins[f"X{xp}"] = nvrtc_build(wk._SRC, defs_for(xp))
    rev, ms = main_src()
    if ms is not None:
        builds["main"], cubins["main"] = nvrtc_build(ms["src"], ("MINB=12",))
        builds["main"]["rev"] = rev
        report["X0_cubin_eq_main"] = cubins["X0"] == cubins["main"]
        ok_all &= report["X0_cubin_eq_main"]
    else:
        report["X0_cubin_eq_main"] = None
        print("WARN main source not found", flush=True)
    for xp in (0, 1, 2):
        builds[f"X{xp}"].update(module_attrs(xp))
    for k, v in builds.items():
        print("BUILD", k, json.dumps(v), flush=True)
    report["builds"] = builds
    ptx_ok = (builds["X2"]["createpolicy"] == 1 and builds["X2"]["cache_hint_loads"] == 12
              and builds["X0"]["createpolicy"] == 0 and builds["X0"]["cache_hint_loads"] == 0)
    report["ptx_counts_ok"] = ptx_ok
    ok_all &= ptx_ok
    report["default_extra_defines"] = list(defs_for(wk._X_POLICY))
    # ---- bitwise ----
    bad = []
    with torch.inference_mode():
        cases = make_cases()
        for name, args in cases:
            x = args[0]
            gs = gs_for(x)
            pristine = [a.clone() if a is not None else None for a in args]
            for gname, gsv in (("gs", gs), ("gs_sat4x", float(torch.tensor(gs * 4.0, dtype=torch.float32)))):
                gs_t = torch.tensor([gsv], dtype=torch.float32, device=dev)
                for store in (True, False):
                    ref = call(args, gs_t, store, 0)
                    for xp in (2, 1):
                        if not same(call(args, gs_t, store, xp), ref):
                            bad.append(f"{name}|{gname}|store{int(store)}|X{xp}")
                    print(name, gname, "store", int(store), "bitwise X2/X1 vs X0:",
                          not any(b.startswith(f"{name}|{gname}|store{int(store)}|") for b in bad),
                          flush=True)
            inp_ok = all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
            ok_all &= inp_ok
            report[f"{name}|inputs_unchanged"] = inp_ok
        report["bitwise_failures"] = bad[:50]
        report["n_bitwise_cases"] = len(cases) * 2 * 2 * 2
        report["n_bitwise_failures"] = len(bad)
        ok_all &= not bad

        # ---- untimed repeat: warmup 3 + 20 iterations, re-poison every 8 ----
        rep = {}
        for name, args in cases:
            if name not in ("select_M4176", "mode_M4096", "mode_M4256", "select_M80"):
                continue
            x = args[0]
            gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
            ref = call(args, gs_t, True, 0)
            pristine = [a.clone() if a is not None else None for a in args]
            outs = poisoned(x)
            first = None
            eq0 = eqref = inp = True
            for it in range(23):
                if it % 8 == 0:
                    inp &= all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
                    for o in outs:
                        o.view(torch.uint8).fill_(0xCD)
                o = call(args, gs_t, True, 2, outs=outs)
                cur = tuple(t.clone() for t in o)
                if first is None:
                    first = cur
                eq0 &= same(cur, first)
                eqref &= same(cur, ref)
            inp &= all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
            rep[name] = {"eq_iter0": eq0, "eq_ref_X0": eqref, "inputs_unchanged": inp}
            ok_all &= eq0 and eqref and inp
        report["repeat"] = rep
        print("REPEAT", json.dumps(rep), flush=True)

        # ---- public op: default (module _X_POLICY) vs _X_POLICY=0 ----
        disp_ok = True
        saved = wk._X_POLICY
        for name, args in cases:
            x = args[0]
            gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
            wk._X_POLICY = 0
            a = torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, args[1], args[2], args[3], W, B, gs_t, EPS, True)
            a = tuple(t.clone() for t in a)
            wk._X_POLICY = 2
            b = torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, args[1], args[2], args[3], W, B, gs_t, EPS, True)
            disp_ok &= same(a, b)
        wk._X_POLICY = saved
        report["public_op_X2_eq_X0"] = disp_ok
        report["modules_built"] = [str(k) for k in wk._MODULES]
        ok_all &= disp_ok
    print(json.dumps({"passed": bool(ok_all), "compared": "X_POLICY 2/1 vs 0 (=main build) "
                      "fp4+poisoned SF+x_mod bitwise", "info": info, **report}))


def make_sets(s, nsets=4):
    sets = []
    for _ in range(nsets):
        x = heavy(1, s, D)
        y = heavy(1, s, D)
        xn = torch.empty_like(x)
        gate = (torch.randn(1, 1, D, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        sel = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        sel[:, :80] = True
        sc, sh_ = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
        outs = poisoned(x)
        sets.append((x, y, xn, gate, sel, sc, sh_, gs_t, outs))
    return sets


def run_bench():
    info = head_info()
    calls = 28
    sets = make_sets(4176)
    pol_src = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    pol_dst = torch.empty_like(pol_src)

    def body():
        for i in range(calls):
            x, y, xn, gate, sel, sc, sh_, gs_t, outs = sets[i % len(sets)]
            wk.launch(x, sel, sc, sh_, W, B, gs_t, EPS, store_x_mod=False, outs=outs)
            pol_dst.copy_(pol_src)
            torch.addcmul(x, gate, y, out=xn)

    with torch.inference_mode():
        for _ in range(2):
            body()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            body()
        print("modules", [str(k) for k in wk._MODULES], flush=True)
        t_end = time.time() + 5.0
        while time.time() < t_end:
            gr.replay()
            torch.cuda.synchronize()
        ts = []
        for r in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e) / calls)
            if r % 25 == 0:
                print("iter", r, ts[-1], flush=True)
    st = sorted(ts)
    med = statistics.median(st)
    noise = (st[89] - st[10]) / med
    print(json.dumps({"batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
                      "output": {"class": "exact"}, "info": info,
                      "modules": [str(k) for k in wk._MODULES]}))


if __name__ == "__main__":
    {"check": run_check, "bench": run_bench}[MODE]()
