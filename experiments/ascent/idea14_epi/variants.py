"""Idea 14 diagnostics: build layout/epilogue variants of flux3_qkv_epi_fp8_sm120.cu under distinct
module names, report cuobjdump resource usage (regs, stack/local spill), and time the GEMM launch
alone (graph replay) against main's QKV3 GEMM at M (env VAR_MS). VARIANTS env: comma list of
name=defines (defines joined by '+')."""

import glob
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from op import SM, make_inputs, N  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build(name, defines):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        f"flux3_qkv_epi_v_{name}",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_qkv_epi_fp8_sm120.cu"],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP4",
                                        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"]
        + [f"-D{d}" for d in defines],
        extra_cflags=["-DFAST_BUILD"],
    )
    mod = spec.build_and_load()
    so = None
    for attr in ("jit_library_path", "library_path"):
        p = getattr(spec, attr, None)
        if p:
            so = str(p)
    if so is None or not os.path.exists(so):
        cands = glob.glob(os.path.join(str(jit_env.FLASHINFER_JIT_DIR), "**",
                                       f"flux3_qkv_epi_v_{name}*.so"), recursive=True)
        so = cands[0] if cands else None
    return mod, so


def res_usage(so):
    if not so:
        return "no .so"
    r = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True)
    lines = [l.strip() for l in r.stdout.splitlines() if "REG" in l or "Function" in l]
    keep = []
    for i, l in enumerate(lines):
        if "REG" in l and i > 0 and "device_kernel" in lines[i - 1]:
            keep.append(l)
    return keep or lines[-6:]


def main():
    dev = torch.device("cuda")
    strided.preload(dev)
    ms = [int(x) for x in os.environ.get("VAR_MS", "16896").split(",")]
    spec = os.environ.get("VARIANTS", "l0=FLUX3_EPI_LAYOUT=0,l0noop=FLUX3_EPI_LAYOUT=0+FLUX3_EPI_DIAG_NOOP,"
                          "l1noop=FLUX3_EPI_LAYOUT=1+FLUX3_EPI_DIAG_NOOP,"
                          "l2noop=FLUX3_EPI_LAYOUT=2+FLUX3_EPI_DIAG_NOOP")
    variants = {}
    for item in spec.split(","):
        name, d = item.split("=", 1)
        defines = d.split("+") if d else []
        t0 = time.time()
        mod, so = build(name, defines)
        variants[name] = mod
        log(f"built {name} {defines} in {time.time() - t0:.0f}s stages={int(mod.qkv3_epi_stages())} "
            f"smem={int(mod.qkv3_epi_smem())} res={res_usage(so)}")
    # main's module resource usage
    try:
        from flashinfer.jit import env as jit_env
        cands = glob.glob(os.path.join(str(jit_env.FLASHINFER_JIT_DIR), "**",
                                       "flux3_fp4_gemm_sm120_pingpong_dyn*.so"), recursive=True)
        log("main dyn res", res_usage(cands[0] if cands else None))
    except Exception as e:  # noqa: BLE001
        log("main res failed", e)
    R, ITERS = 4, 30
    out_all = {}
    for m in ms:
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = make_inputs(m, dev)
        qkv8 = torch.empty((3, m, N), dtype=torch.uint8, device=dev)
        gout = torch.empty((m, 3 * N), dtype=torch.bfloat16, device=dev)
        v = gout.view(3, m, N)[2]
        parts = torch.zeros(((m + 127) // 128) * (N // 128) + 8, dtype=torch.float32, device=dev)
        ws = strided.workspace(dev, 0)
        fns = {"main_gemm": lambda: strided.gemm_qkv3(a, w, sfa, sfb, alpha, gout)}
        for name, mod in variants.items():
            fns[name] = (lambda mod=mod: mod.qkv3_epi_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, qkv8, v,
                                                          parts, ws, float(_MIN_SCALING_FACTOR)))
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        graphs = {}
        for name, f in fns.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(R):
                    f()
            graphs[name] = g
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        times = {n: [] for n in graphs}
        for _ in range(ITERS):
            for name, g in graphs.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) / R)
        row = {n: round(sorted(t)[len(t) // 2], 4) for n, t in times.items()}
        log(f"M{m} ms", json.dumps(row))
        out_all[m] = row
        del graphs
    print(json.dumps(out_all))


if __name__ == "__main__":
    main()
