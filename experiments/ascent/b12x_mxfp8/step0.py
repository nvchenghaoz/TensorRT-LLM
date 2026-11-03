"""Step 0 image check: can FlashInfer mm_mxfp8(backend='b12x') run on this image?"""
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback

import torch

res = {}
res["torch"] = torch.__version__
res["torch_cuda"] = torch.version.cuda
try:
    res["cuda_runtime"] = torch.cuda.cudart().cudaRuntimeGetVersion() if hasattr(torch.cuda, "cudart") else None
except Exception as e:  # noqa
    res["cuda_runtime"] = repr(e)
try:
    res["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:  # noqa
    res["nvidia_smi"] = repr(e)
res["gpu"] = torch.cuda.get_device_name(0)
res["sm"] = torch.cuda.get_device_capability(0)
for pkg in ("nvidia-cutlass-dsl", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
            "cuda-python", "nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime"):
    try:
        res["pkg_" + pkg] = importlib.metadata.version(pkg)
    except Exception:
        res["pkg_" + pkg] = None
import flashinfer  # noqa
res["flashinfer_version"] = getattr(flashinfer, "__version__", None)
for attr in ("__git_version__", "__commit__", "git_version"):
    if hasattr(flashinfer, attr):
        res["flashinfer_" + attr] = str(getattr(flashinfer, attr))
res["flashinfer_file"] = flashinfer.__file__
try:
    from flashinfer.jit.cpp_ext import get_cuda_version
    res["flashinfer_get_cuda_version"] = str(get_cuda_version())
except Exception as e:
    res["flashinfer_get_cuda_version"] = repr(e)
print(json.dumps(res), flush=True)

try:
    t0 = time.time()
    import flashinfer.gemm.gemm_mm_mxfp8_cute_dsl as mod  # noqa
    res["import_cute_dsl_module_s"] = time.time() - t0
    res["import_ok"] = True
except Exception:
    res["import_ok"] = False
    res["import_error"] = traceback.format_exc()[-3000:]

import tensorrt_llm  # noqa  registers torch.ops.trtllm
from flashinfer import mm_mxfp8

torch.manual_seed(0)
M, N, K = 1, 1280, 5120
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
aq, asf = torch.ops.trtllm.mxfp8_quantize(x, True)
res["wq"] = [str(wq.dtype), list(wq.shape)]
res["wsf"] = [str(wsf.dtype), list(wsf.shape)]
res["asf"] = [str(asf.dtype), list(asf.shape)]

ref_cut = mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="cutlass")
torch.cuda.synchronize()
try:
    t0 = time.time()
    out = mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="b12x")
    torch.cuda.synchronize()
    res["b12x_first_call_s"] = time.time() - t0
    t0 = time.time()
    out2 = mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="b12x")
    torch.cuda.synchronize()
    res["b12x_second_call_s"] = time.time() - t0
    res["b12x_ok"] = True
    res["bitwise_vs_cutlass"] = bool(torch.equal(out, ref_cut))
    res["max_abs_diff_vs_cutlass"] = float((out.float() - ref_cut.float()).abs().max())
    res["repeat_bitwise"] = bool(torch.equal(out, out2))
    # fp32 dequant reference
    from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight  # noqa
    res["finite"] = bool(torch.isfinite(out).all())
except Exception:
    res["b12x_ok"] = False
    res["b12x_error"] = traceback.format_exc()[-4000:]

res["passed"] = bool(res.get("b12x_ok"))
print(json.dumps(res))
