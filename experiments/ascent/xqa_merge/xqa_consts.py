#!/usr/bin/env python3
# Reports the multi-block merge constants of kernel_mha as compiled for the serving shape (same NVRTC options/macros
# as xqa_evidence.py): compiles an in-memory copy of <cwd>/cpp/kernels/xqa/mha.cu with a few appended read-only
# __device__ globals holding multiBlockMergeChunkSize, MultiBlockMergeSMem::nbTileBuffers and the smem sizes, then
# reads them back with cuModuleGetGlobal. The committed mha.cu is not modified. Also prints the source sha256.
import ctypes
import hashlib
import json
import os
import sys

import numpy as np
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/xqa_merge"))
import xqa_evidence as ev  # noqa: E402

SNIPPET = r"""
extern "C" __device__ uint32_t xqaDbgMergeChunkSize = multiBlockMergeChunkSize;
extern "C" __device__ uint32_t xqaDbgNbTileBuffers = MultiBlockMergeSMem::nbTileBuffers;
extern "C" __device__ uint32_t xqaDbgMergeSMemBytes = sizeof(MultiBlockMergeSMem);
extern "C" __device__ uint32_t xqaDbgSharedMemBytes = sizeof(SharedMem);
extern "C" __device__ uint32_t xqaDbgCtaWarpsX = ctaShapeInWarps.x;
extern "C" __device__ uint32_t xqaDbgGemm1NbWarpGrps = gemm1NbWarpGrps;
"""
NAMES = ["xqaDbgMergeChunkSize", "xqaDbgNbTileBuffers", "xqaDbgMergeSMemBytes", "xqaDbgSharedMemBytes",
         "xqaDbgCtaWarpsX", "xqaDbgGemm1NbWarpGrps"]


def main():
    ev.ck(cu.cuInit(0), "cuInit")
    dev = ev.ck(cu.cuDeviceGet(0), "cuDeviceGet")
    ctx = ev.ck(cu.cuDevicePrimaryCtxRetain(dev), "ctx")
    ev.ck(cu.cuCtxSetCurrent(ctx), "setctx")
    xdir = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    src = open(os.path.join(xdir, "mha.cu")).read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    names = [h.encode() for h in ev.XQA_HEADERS + ev.CUDA_HEADERS]
    contents = [open(os.path.join(xdir, h), "rb").read() for h in ev.XQA_HEADERS]
    contents += [open(os.path.join(ev.CUDA_INC, h), "rb").read() for h in ev.CUDA_HEADERS]
    prog = ev.ck(nvrtc.nvrtcCreateProgram((src + SNIPPET).encode(), b"default_program", len(names), contents, names),
                 "create")
    opts = [b"-dw", b"--use_fast_math", b"-default-device", b"-arch=sm_120f"]
    opts += [f"-D{k}={v}".encode() for k, v in ev.MACROS.items()]
    if int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) != 0:
        size = ev.ck(nvrtc.nvrtcGetProgramLogSize(prog), "logsize")
        buf = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("NVRTC compile failed:\n" + buf.decode(errors="replace")[-6000:])
    size = ev.ck(nvrtc.nvrtcGetCUBINSize(prog), "cubinsize")
    cubin = b" " * size
    ev.ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "getcubin")
    img = np.frombuffer(cubin, dtype=np.uint8).copy()
    mod = ev.ck(cu.cuModuleLoadData(img.ctypes.data), "load")
    vals = {}
    for n in NAMES:
        dptr, _ = ev.ck(cu.cuModuleGetGlobal(mod, n.encode()), n)
        v = ctypes.c_uint32(0)
        ev.ck(cu.cuMemcpyDtoH(ctypes.addressof(v), dptr, 4), "dtoh")
        vals[n] = int(v.value)
    sv_ptr, _ = ev.ck(cu.cuModuleGetGlobal(mod, b"smemSize"), "smemSize")
    v = ctypes.c_uint32(0)
    ev.ck(cu.cuMemcpyDtoH(ctypes.addressof(v), sv_ptr, 4), "dtoh")
    vals["smemSize"] = int(v.value)
    print(json.dumps({"mha_cu_sha256": sha, "compiled_constants": vals}), flush=True)


if __name__ == "__main__":
    main()
