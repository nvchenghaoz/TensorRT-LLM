"""cuBLASLt enumeration helpers (AlgoGetIds + AlgoCapGetAttribute) on top of sm120_bf16_lut/lt.py (unchanged).

Main's dispatch for a Gemma4 BF16 attention projection at token count M (what the bitwise gate compares to):
  trtllm::cublas_mm (SM120 range table) when the tree's _SM120_BF16_CUBLAS_MM_ROUTES gate covers M, else F.linear.
"""
import ctypes
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm120_bf16_lut"))
from lt import (CUBLAS_COMPUTE_32F, CUDA_R_16BF, CUDA_R_32F, Algo, HeurResult, Lt,  # noqa: E402,F401
                read_shapes)

# cublasLtMatmulAlgoCapAttributes_t (fallback values; parsed from cublasLt.h when present)
CAP_DEFAULT = dict(SPLITK_SUPPORT=0, REDUCTION_SCHEME_MASK=1, CTA_SWIZZLING_SUPPORT=2, STRIDED_BATCH_SUPPORT=3,
                   OUT_OF_PLACE_RESULT_SUPPORT=4, UPLO_SUPPORT=5, TILE_IDS=6, CUSTOM_OPTION_MAX=7,
                   CUSTOM_MEMORY_ORDER=10, POINTER_MODE_MASK=11, EPILOGUE_MASK=12, STAGES_IDS=13, LD_NEGATIVE=14,
                   NUMERICAL_IMPL_FLAGS=15, MIN_ALIGNMENT_A_BYTES=16, MIN_ALIGNMENT_B_BYTES=17,
                   MIN_ALIGNMENT_C_BYTES=18, MIN_ALIGNMENT_D_BYTES=19, ATOMIC_SYNC=20)
ARRAY_CAPS = ("TILE_IDS", "STAGES_IDS", "CLUSTER_SHAPE_IDS")


def parse_header(hdr):
    caps, clusters = dict(CAP_DEFAULT), {}
    if hdr and os.path.exists(hdr):
        txt = open(hdr).read()
        for name, val in re.findall(r"CUBLASLT_ALGO_CAP_(\w+)\s*=\s*(\d+)", txt):
            caps[name] = int(val)
        for name, val in re.findall(r"CUBLASLT_CLUSTER_SHAPE_(\w+)\s*=\s*(\d+)", txt):
            clusters[int(val)] = name
    return caps, clusters


class LtX(Lt):

    def __init__(self):
        super().__init__()
        self.caps, self.clusters = parse_header(self.hdr)
        L = self.lib
        L.cublasLtMatmulAlgoGetIds.restype = ctypes.c_int
        L.cublasLtMatmulAlgoCapGetAttribute.restype = ctypes.c_int
        L.cublasLtMatmulAlgoCheck.restype = ctypes.c_int

    def algo_ids(self, n=1024):
        arr = (ctypes.c_int * n)()
        ret = ctypes.c_int(0)
        self._ck(self.lib.cublasLtMatmulAlgoGetIds(self.h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                                   CUDA_R_16BF, CUDA_R_16BF, n, arr, ctypes.byref(ret)), "getids")
        return [arr[i] for i in range(ret.value)]

    def init_algo(self, algo_id):
        a = Algo()
        st = self.lib.cublasLtMatmulAlgoInit(self.h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                             CUDA_R_16BF, CUDA_R_16BF, algo_id, ctypes.byref(a))
        return a if st == 0 else None

    def cap(self, algo, name):
        attr = self.caps.get(name)
        if attr is None:
            return None
        if name in ARRAY_CAPS:
            wr = ctypes.c_size_t(0)
            st = self.lib.cublasLtMatmulAlgoCapGetAttribute(ctypes.byref(algo), attr, None, 0, ctypes.byref(wr))
            if st != 0:
                return None
            n = wr.value // 4
            if n == 0:
                return []
            buf = (ctypes.c_uint32 * n)()
            st = self.lib.cublasLtMatmulAlgoCapGetAttribute(ctypes.byref(algo), attr, buf, n * 4, ctypes.byref(wr))
            if st != 0:
                return None
            return [int(buf[i]) for i in range(n)]
        buf = ctypes.c_uint64(0)
        wr = ctypes.c_size_t(0)
        for sz in (4, 8, 2):
            buf = ctypes.c_uint64(0)
            st = self.lib.cublasLtMatmulAlgoCapGetAttribute(ctypes.byref(algo), attr, ctypes.byref(buf), sz,
                                                            ctypes.byref(wr))
            if st == 0:
                return int(buf.value)
        return None

    def caps_of(self, algo_id):
        a = self.init_algo(algo_id)
        if a is None:
            return None
        out = {}
        for name in sorted(self.caps):
            if "ALIGNMENT" in name or name in ("MATHMODE_IMPL", "GAUSSIAN_IMPL"):
                continue
            out[name] = self.cap(a, name)
        return out

    def make_algo_raw(self, cfg8):
        """Like Lt.make_algo but returns None instead of raising when an attribute is refused."""
        a = self.init_algo(cfg8[0])
        if a is None:
            return None
        from lt import CFG_CGA, CFG_CUSTOM, CFG_RED, CFG_SPLITK, CFG_STAGES, CFG_SWIZZLE, CFG_TILE
        _, tile, stages, numsk, red, swz, custom, cga = cfg8
        for attr, val, ty in [(CFG_TILE, tile, ctypes.c_int), (CFG_STAGES, stages, ctypes.c_int),
                              (CFG_SPLITK, numsk, ctypes.c_int), (CFG_RED, red, ctypes.c_int),
                              (CFG_SWIZZLE, swz, ctypes.c_int), (CFG_CUSTOM, custom, ctypes.c_uint32),
                              (CFG_CGA, cga, ctypes.c_uint16)]:
            b = ty(val)
            if self.lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), attr, ctypes.byref(b),
                                                             ctypes.sizeof(b)) != 0:
                return None
        return a

    def check_ws(self, m, n, k, algo):
        """(ok, workspace, waves) from cublasLtMatmulAlgoCheck."""
        op, A, B, C, pref = self.descs(m, n, k)
        r = HeurResult()
        st = self.lib.cublasLtMatmulAlgoCheck(self.h, op, A, B, C, C, ctypes.byref(algo), ctypes.byref(r))
        ok = st == 0 and r.state == 0 and r.workspaceSize <= self.ws.numel()
        return ok, int(r.workspaceSize), float(r.wavesCount)

    def tile_name(self, t):
        return self.tiles.get(t, str(t))


def main_dispatch():
    """Callable (x, W) -> out that runs exactly what this tree's Gemma4 attention projection runs on SM120."""
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    gates = {k: mg._CublasMmTokenGate(v) for k, v in mg._SM120_BF16_CUBLAS_MM_ROUTES.items()}

    def op(x, W):
        g = gates.get((W.shape[1], W.shape[0]))
        if g is not None and x.shape[0] in g:
            return torch.ops.trtllm.cublas_mm(x, W.t(), None, None)
        return F.linear(x, W)

    def routed(N, K, M):
        g = gates.get((K, N))
        return g is not None and M in g

    return op, routed, mg


def kernel_names(fn):
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sorted({e.name[:100] for e in prof.events() if e.device_type.name == "CUDA"})
    except Exception as e:  # noqa: BLE001
        return [f"?{e}"]
