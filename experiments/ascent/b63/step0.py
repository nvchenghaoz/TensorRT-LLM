"""Branch 63 STEP 0: dump the CuTe PACKED kernel's PTX epilogue forms and locate CUTLASS headers."""

import glob
import os
import re
import sys
import tempfile

DUMP = tempfile.mkdtemp(prefix="b63ptx_")
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = DUMP

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4 as m  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402


def log(*a):
    print(*a, flush=True)


def main():
    import flashinfer.jit.env as jit_env

    log("FLASHINFER_CSRC_DIR", jit_env.FLASHINFER_CSRC_DIR)
    for name in dir(jit_env):
        if "CUTLASS" in name or "INCLUDE" in name:
            log("jit_env", name, getattr(jit_env, name))
    M, H, N = 256, 3072, 18432
    dev = "cuda"
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, H, device=dev, dtype=torch.bfloat16) * 0.02
    inv = torch.tensor([100.0], device=dev)
    a, sfa = fp.nvfp4_quantize(x, torch.tensor([1.0 / (float(x.abs().amax()) / 2688)], device=dev))
    q, sfq = fp.nvfp4_quantize(w, torch.tensor([1.0 / (float(w.abs().amax()) / 2688)], device=dev))
    m._COMPILED.clear()
    out = m.run("packed", a, q, sfa, sfq, 0.01, 0.01, output_inverse_scale=inv)
    torch.cuda.synchronize()
    log("ran packed", out[0].shape)
    compiled = list(m._COMPILED.values())[0]
    ptx = None
    for attr in ("__ptx__", "ptx"):
        if hasattr(compiled, attr):
            try:
                ptx = getattr(compiled, attr)
                if callable(ptx):
                    ptx = ptx()
                log("got ptx via", attr, type(ptx))
                break
            except Exception as e:  # noqa: BLE001
                log("attr fail", attr, e)
    files = glob.glob(os.path.join(DUMP, "**", "*.ptx"), recursive=True) + glob.glob("*.ptx")
    log("ptx files", files)
    if not isinstance(ptx, str) or not ptx:
        if files:
            ptx = open(max(files, key=os.path.getsize)).read()
    if not ptx:
        log("NO PTX")
        return
    lines = ptx.splitlines()
    log("ptx lines", len(lines))
    pats = ["ex2", "div", "rcp", "cvt.rn.satfinite", "e2m1", "e4m3", "fma.rn.f32", "max.f32",
            "abs.f32", "shfl", "lg2"]
    for p in pats:
        hits = [ln.strip() for ln in lines if p in ln]
        uniq = sorted(set(re.sub(r"%\w+\d*", "%r", h) for h in hits))
        log(f"== {p}: {len(hits)} hits; forms:")
        for u in uniq[:12]:
            log("   ", u)
    idx = [i for i, ln in enumerate(lines) if "ex2" in ln]
    if idx:
        i0 = idx[0]
        log("== context around first ex2")
        for ln in lines[max(0, i0 - 25): i0 + 40]:
            log("  ", ln)
    idx = [i for i, ln in enumerate(lines) if "e4m3" in ln]
    if idx:
        i0 = idx[0]
        log("== context around first e4m3")
        for ln in lines[max(0, i0 - 40): i0 + 60]:
            log("  ", ln)
    idx = [i for i, ln in enumerate(lines) if "e2m1" in ln]
    if idx:
        i0 = idx[0]
        log("== context around first e2m1")
        for ln in lines[max(0, i0 - 10): i0 + 10]:
            log("  ", ln)


def headers():
    import flashinfer

    root = os.path.dirname(flashinfer.__file__)
    cands = glob.glob(os.path.join(root, "**", "cutlass", "gemm", "collective", "sm120_blockscaled*"),
                      recursive=True)
    cands += glob.glob(os.path.join(root, "..", "**", "cutlass", "gemm", "collective",
                                    "sm120_blockscaled*"), recursive=True)
    log("sm120 collective headers", cands)


if __name__ == "__main__":
    headers()
    main()
    sys.stdout.flush()
