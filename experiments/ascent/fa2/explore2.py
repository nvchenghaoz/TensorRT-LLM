import os, subprocess
def sh(c, n=6000):
    r = subprocess.run(["bash", "-lc", c], capture_output=True, text=True)
    print("$", c); print(r.stdout[:n], r.stderr[:1500], flush=True)
P = "/opt/pytorch/pytorch"
sh(f"ls {P}; ls {P}/build | head -50")
sh(f"ls -la {P}/third_party/flash-attention/csrc/flash_attn/src | head -80")
sh(f"cd {P}/third_party/flash-attention && (git log -1 --format='%H %cd' 2>&1; cat .git 2>&1 | head)")
sh(f"ls {P}/aten/src/ATen/native/transformers/cuda/flash_attn/ {P}/aten/src/ATen/native/transformers/cuda/flash_attn/* | head -80")
sh(f"grep -rn -i 'flash' {P}/aten/src/ATen/CMakeLists.txt {P}/CMakeLists.txt {P}/caffe2/CMakeLists.txt {P}/cmake/*.cmake 2>/dev/null | head -80", 12000)
sh(f"grep -rn 'UNFUSE_FMA\\|FLASH_NAMESPACE' {P}/third_party/flash-attention/csrc/flash_attn/src | head -40")
sh(f"find {P}/build -name 'compile_commands.json' -o -name 'build.ninja' 2>/dev/null | head")
sh(f"grep -o 'flash_fwd_hdim128_bf16_sm80.cu[^\\n]*' {P}/build/build.ninja 2>/dev/null | head -3")
sh(f"grep -n -A3 'flash_fwd_hdim128_bf16_sm80' {P}/build/build.ninja 2>/dev/null | head -20", 12000)
sh(f"grep -n 'CUDA_FLAGS\\|NVCC_FLAGS' {P}/build/CMakeCache.txt 2>/dev/null | head -20", 8000)
sh(f"ls {P}/aten/src/ATen/native/transformers/cuda/ ")
sh(f"sed -n 1,200p {P}/third_party/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h | grep -n 'hdim128' -A40 | head -60", 8000)
