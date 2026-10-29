"""Branch 165 helper: dump the generated d128 fmha_v2 kernel .cu and device smem limits."""
import sys, os, tempfile, pathlib
import torch

print("smem optin", torch.cuda.get_device_properties(0), flush=True)
from cuda.bindings import runtime as rt  # noqa
try:
    err, v = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
    print("MaxSharedMemoryPerBlockOptin", v)
    err, v = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0)
    print("MaxSharedMemoryPerMultiprocessor", v)
    err, v = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock, 0)
    print("MaxRegistersPerBlock", v)
except Exception as e:  # noqa
    print("attr err", e)

from flashinfer.jit import env as jit_env
from flashinfer.jit.attention.fmha_v2.generate_kernels import enumerate_kernels
d = pathlib.Path(tempfile.mkdtemp())
enumerate_kernels(jit_env.FLASHINFER_CSRC_DIR / "fmha_v2", d)
p = d / "generated" / "fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16_sm120.cu"
print(p.read_text())
