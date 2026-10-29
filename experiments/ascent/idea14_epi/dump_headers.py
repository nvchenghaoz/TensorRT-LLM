"""Copy the CUTLASS headers the epilogue design depends on into the job artifacts."""
import os
import shutil
import subprocess
import sys

from flashinfer.jit import env as jit_env

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/idea14_h")
os.makedirs(art, exist_ok=True)
roots = []
for name in dir(jit_env):
    v = getattr(jit_env, name)
    if "CUTLASS" in name.upper():
        print(name, v, flush=True)
        if isinstance(v, (list, tuple)):
            roots += [str(x) for x in v]
        else:
            roots.append(str(v))
want = sys.argv[1:] or [
    "cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl",
    "cutlass/gemm/collective/builders/sm120_common.inl",
    "cutlass/epilogue/collective/detail.hpp",
    "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp",
    "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp",
    "cutlass/epilogue/collective/builders/sm120_builder.inl",
    "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp",
    "cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp",
    "cutlass/arch/barrier.h",
]
for root in roots:
    for rel in want:
        for base in (root, os.path.join(root, "include")):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                dst = os.path.join(art, rel.replace("/", "__"))
                shutil.copy(p, dst)
                print("copied", p, flush=True)
print(subprocess.run(["bash", "-c", "ls " + " ".join(roots)], capture_output=True, text=True).stdout)
print('{"done": true}')
