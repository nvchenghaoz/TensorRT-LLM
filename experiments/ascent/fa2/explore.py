import os, sys, glob, subprocess, importlib.util
import torch
print("torch", torch.__version__, torch.__file__, flush=True)
tdir = os.path.dirname(torch.__file__)
inc = os.path.join(tdir, "include")
for pat in ["ATen/native/transformers/cuda/flash_attn/**/*", "ATen/native/transformers/cuda/**/*.h"]:
    fs = glob.glob(os.path.join(inc, pat), recursive=True)
    print(pat, len(fs))
    for f in fs[:200]:
        print("  ", f[len(inc):], os.path.getsize(f) if os.path.isfile(f) else "dir")
for name in ["flash_attn", "flash_attn_2_cuda", "flash_attn_interface", "flash_attn.cute", "cutlass", "nvidia.cutlass_dsl"]:
    try:
        s = importlib.util.find_spec(name)
        print("spec", name, s.origin if s else None)
    except Exception as e:
        print("spec", name, "err", e)
# search the filesystem for kernel_traits.h / flash_fwd_kernel.h
roots = ["/usr", "/opt", "/home/scratch.chengzhang/trtllm-dev-minimax-h3-2g-20261112-143634/runner/slots", os.path.expanduser("~"), "/root", "/workspace", "/tmp"]
for r in roots:
    try:
        out = subprocess.run(["find", r, "-maxdepth", "12", "(", "-name", "flash_fwd_kernel.h", "-o", "-name", "kernel_traits.h", "-o", "-name", "flash_fwd_launch_template.h", "-o", "-name", "softmax.h", "-path", "*flash*", ")"], capture_output=True, text=True, timeout=120)
        print("find", r, out.stdout[:4000])
    except Exception as e:
        print("find", r, "err", e)
# cutlass include dirs
for r in ["/usr", "/opt", "/home/scratch.chengzhang/trtllm-dev-minimax-h3-2g-20261112-143634/runner/slots/0/workspace"]:
    try:
        out = subprocess.run(["find", r, "-maxdepth", "10", "-type", "d", "-name", "cute", "-path", "*include*"], capture_output=True, text=True, timeout=120)
        print("cute dirs", r, out.stdout[:3000])
    except Exception as e:
        print(e)
print("nvcc", subprocess.run(["bash", "-lc", "which nvcc; nvcc --version"], capture_output=True, text=True).stdout)
lib = os.path.join(tdir, "lib", "libtorch_cuda.so")
print("lib", lib, os.path.exists(lib), os.path.getsize(lib) if os.path.exists(lib) else 0)
print("cuobjdump", subprocess.run(["bash", "-lc", "which cuobjdump"], capture_output=True, text=True).stdout)
print("arch list", torch.cuda.get_arch_list())
print(torch.cuda.get_device_capability())
print("pip list", subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout[:6000])
