"""Dump CUTLASS headers needed for the SwiGLU epilogue and the Triton PTX of
fused_silu_mul_nvfp4 into $AUTO_DEV_ARTIFACTS (no timing)."""
import glob
import os
import shutil
import subprocess
import sys

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/art")
os.makedirs(art, exist_ok=True)
cands = sorted(glob.glob(
    "/home/scratch.chengzhang/trtllm-dev-qwen38-27b-20261015-171031/runner/slots/*/native/build/candidate/cmake/_deps/cutlass-src"))
print("cutlass srcs", cands, flush=True)
root = cands[0]
print(subprocess.run(["git", "-C", root, "describe", "--tags"], capture_output=True, text=True).stdout)
files = [
    "include/cutlass/epilogue/collective/default_epilogue.hpp",
    "include/cutlass/epilogue/collective/detail.hpp",
    "include/cutlass/epilogue/collective/builders/sm120_builder.inl",
    "include/cutlass/epilogue/collective/builders/sm90_builder.inl",
    "include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl",
    "include/cutlass/gemm/collective/builders/sm120_common.inl",
    "include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp",
    "include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp",
    "include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp",
]
out = os.path.join(art, "src")
os.makedirs(out, exist_ok=True)
for f in files:
    p = os.path.join(root, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(out, f.replace("/", "__")))
    else:
        print("missing", f)
# kernel files for sm120 cooperative
r = subprocess.run(["grep", "-rln", "MainloopSm120TmaWarpSpecializedBlockScaled\\|KernelTmaWarpSpecializedCooperativeBlockScaledSm120\\|Sm120", os.path.join(root, "include/cutlass/gemm/kernel")], capture_output=True, text=True)
print("kernel files:\n", r.stdout)
for p in r.stdout.split():
    shutil.copy(p, os.path.join(out, "kernel__" + os.path.basename(p)))
r = subprocess.run(["bash", "-c", f"ls {root}/examples | grep -i 79; ls {root}/examples/79*"], capture_output=True, text=True)
print(r.stdout)
for p in glob.glob(f"{root}/examples/79*/*nvfp4*"):
    shutil.copy(p, os.path.join(out, "ex__" + os.path.basename(p)))
r = subprocess.run(["grep", "-n", "struct KernelTmaWarpSpecializedCooperative\\|Sm120\\|BlockScaled", os.path.join(root, "include/cutlass/gemm/dispatch_policy.hpp")], capture_output=True, text=True)
print(r.stdout[:6000])
r = subprocess.run(["grep", "-n", "Sm90TmaWarpSpecializedAdapter\\|NoSmemWarpSpecialized", "-r", os.path.join(root, "include/cutlass/epilogue")], capture_output=True, text=True)
print(r.stdout[:4000])

# Triton PTX of the silu kernel
sys.path.insert(0, os.getcwd())
import torch
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as F
x = torch.randn(256, 34816, device="cuda", dtype=torch.bfloat16)
s = torch.tensor([1.0], device="cuda")
fp4, sf = F._fused_silu_mul_nvfp4_impl(x, s)
torch.cuda.synchronize()
k = F.fused_silu_mul_nvfp4_kernel
done = False
for dev_cache in getattr(k, "device_caches", {}).values():
    cache = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
    for key, ck in cache.items():
        open(os.path.join(art, "silu.ptx"), "w").write(ck.asm["ptx"])
        done = True
print("ptx dumped", done)
if not done:
    r = subprocess.run(["bash", "-c", "find ~/.triton/cache -name 'fused_silu_mul_nvfp4_kernel*.ptx' | head"], capture_output=True, text=True)
    print(r.stdout)
    for p in r.stdout.split():
        shutil.copy(p, os.path.join(art, os.path.basename(p)))
print("{\"done\": true}")
