"""JIT build of the FA2-rebuild extension (Stage 0 microbench helper)."""
import os
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
FLASH_SRC = "/opt/pytorch/pytorch/third_party/flash-attention/csrc/flash_attn/src"
CUTLASS_INC = "/opt/pytorch/pytorch/third_party/cutlass/include"
DEFS = [
    "-DFLASH_NAMESPACE=ascent_fa2",
    "-DFLASHATTENTION_DISABLE_ALIBI",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DUNFUSE_FMA",
]


def build(variants=range(8), verbose=True, name="ascent_fa2_s0"):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "16")
    srcs = [os.path.join(HERE, "csrc", "fa2_bind.cpp")]
    srcs += [os.path.join(HERE, "csrc", f"fa2_v{v}.cu") for v in variants]
    srcs += [os.path.join(HERE, "csrc", "fa2_split.cu")]
    bdir = os.environ.get("ASCENT_FA2_BUILD_DIR", os.path.expanduser(f"~/.cache/ascent_fa2/{name}"))
    os.makedirs(bdir, exist_ok=True)
    t0 = time.time()
    mod = load(
        name=name,
        sources=srcs,
        extra_include_paths=[FLASH_SRC, CUTLASS_INC, os.path.join(HERE, "csrc")],
        extra_cflags=["-O2"] + DEFS,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-Xptxas=-v"] + DEFS,
        build_directory=bdir,
        verbose=verbose,
    )
    print(f"[build] {name} done in {time.time() - t0:.1f}s dir={bdir}", flush=True)
    return mod
