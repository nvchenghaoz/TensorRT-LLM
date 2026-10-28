"""Branch 63: copy selected CUTLASS headers (as FlashInfer's JIT uses them) into the artifact dir."""

import glob
import os
import shutil

import flashinfer.jit.env as jit_env

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b63art")
os.makedirs(art, exist_ok=True)
roots = []
for name in dir(jit_env):
    v = getattr(jit_env, name)
    if isinstance(v, (list, tuple)):
        vals = v
    else:
        vals = [v]
    for x in vals:
        try:
            p = str(x)
        except Exception:  # noqa: BLE001
            continue
        if os.path.isdir(os.path.join(p, "cutlass", "gemm")):
            roots.append(p)
            print("cutlass root", name, p, flush=True)
roots = sorted(set(roots))
want = [
    "cutlass/epilogue/collective/detail.hpp",
    "cutlass/epilogue/collective/default_epilogue.hpp",
    "cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl",
    "cutlass/gemm/collective/builders/sm120_common.inl",
    "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp",
    "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp",
    "cutlass/epilogue/collective/builders/sm120_builder.inl",
    "cutlass/gemm/kernel/gemm_universal.hpp",
]
for r in roots:
    for w in want:
        p = os.path.join(r, w)
        if os.path.exists(p):
            dst = os.path.join(art, w.replace("/", "__"))
            shutil.copy(p, dst)
            print("copied", p, os.path.getsize(p), flush=True)
    for p in glob.glob(os.path.join(r, "cutlass/gemm/kernel/sm120*")) + glob.glob(
        os.path.join(r, "cutlass/gemm/collective/sm120*")
    ) + glob.glob(os.path.join(r, "cutlass/gemm/collective/builders/sm120*")):
        print("exists", p, os.path.getsize(p), flush=True)
    dp = os.path.join(r, "cutlass/gemm/dispatch_policy.hpp")
    if os.path.exists(dp):
        lines = open(dp).read().splitlines()
        for i, ln in enumerate(lines):
            if "Sm120" in ln or "sm120" in ln:
                print(f"dispatch_policy:{i}: {ln}", flush=True)
    kp = os.path.join(r, "cutlass/gemm/kernel/gemm_universal.hpp")
    if os.path.exists(kp):
        for ln in open(kp).read().splitlines():
            if "#include" in ln:
                print("gemm_universal include:", ln, flush=True)
    break
print("done", flush=True)
