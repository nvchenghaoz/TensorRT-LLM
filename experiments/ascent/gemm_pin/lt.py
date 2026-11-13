"""Minimal ctypes cuBLASLt wrapper for BF16 x BF16 -> BF16 (FP32 compute) F.linear-layout GEMMs.

y[M, N] = x[M, K] @ w[N, K]^T, run column-major as Y^T(NxM) = op_T(W(KxN)) * X^T(KxM).
Enum values are parsed from the cublasLt.h that matches the loaded libcublasLt when found.
"""
import ctypes
import glob
import os
import re
import sys

import torch

vp = ctypes.c_void_p
COMPUTE_32F = 68
R_32F = 0
R_16BF = 14


class Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class Heur(ctypes.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def loaded_libs(name="libcublasLt.so"):
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            if name in line:
                p = line.split()[-1]
                if p not in out:
                    out.append(p)
    return out


def find_headers(lib_path):
    c = []
    if lib_path:
        d = os.path.dirname(os.path.dirname(lib_path))
        c += glob.glob(os.path.join(d, "include", "cublasLt.h"))
    for base in [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/usr"]:
        if base:
            c += glob.glob(os.path.join(base, "include", "cublasLt.h"))
            c += glob.glob(os.path.join(base, "targets", "*", "include", "cublasLt.h"))
    for p in sys.path:
        c += glob.glob(os.path.join(p, "nvidia", "cublas", "include", "cublasLt.h"))
        c += glob.glob(os.path.join(p, "nvidia", "cu*", "include", "cublasLt.h"))
    seen = []
    for x in c:
        if x not in seen:
            seen.append(x)
    return seen


def parse_enum(text, prefix):
    d = {}
    for m in re.finditer(r"\b" + prefix + r"(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)", text):
        d.setdefault(m.group(1), int(m.group(2), 0))
    return d


class Lt:
    CFG = {"ID": 0, "TILE_ID": 1, "SPLITK_NUM": 2, "REDUCTION_SCHEME": 3, "CTA_SWIZZLING": 4,
           "CUSTOM_OPTION": 5, "STAGES_ID": 6, "INNER_SHAPE_ID": 7, "CLUSTER_SHAPE_ID": 8}
    CAP = {"SPLITK_SUPPORT": 0, "REDUCTION_SCHEME_MASK": 1, "CTA_SWIZZLING_SUPPORT": 2,
           "TILE_IDS": 6, "CUSTOM_OPTION_MAX": 7, "STAGES_IDS": 13}
    CFG_SIZE = {"INNER_SHAPE_ID": 2, "CLUSTER_SHAPE_ID": 2}

    def __init__(self, ws_bytes=32 << 20):
        libs = loaded_libs()
        if not libs:
            raise RuntimeError("libcublasLt not loaded")
        self.libs = libs
        self.path = libs[0]
        L = ctypes.CDLL(self.path)
        self.L = L
        L.cublasLtGetVersion.restype = ctypes.c_size_t
        self.version = int(L.cublasLtGetVersion())
        L.cublasLtCreate.argtypes = [ctypes.POINTER(vp)]
        L.cublasLtMatmulDescCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_int]
        L.cublasLtMatmulDescSetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t]
        L.cublasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_uint64,
                                                 ctypes.c_uint64, ctypes.c_int64]
        L.cublasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(vp)]
        L.cublasLtMatmulPreferenceSetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t]
        L.cublasLtMatmulAlgoGetHeuristic.argtypes = [vp, vp, vp, vp, vp, vp, vp, ctypes.c_int,
                                                     ctypes.POINTER(Heur), ctypes.POINTER(ctypes.c_int)]
        L.cublasLtMatmul.argtypes = [vp] * 14 + [ctypes.c_size_t, vp]
        L.cublasLtMatmulAlgoConfigGetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t,
                                                           ctypes.POINTER(ctypes.c_size_t)]
        L.cublasLtMatmulAlgoConfigSetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t]
        L.cublasLtMatmulAlgoCapGetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t,
                                                        ctypes.POINTER(ctypes.c_size_t)]
        L.cublasLtMatmulAlgoGetIds.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.cublasLtMatmulAlgoInit.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Algo)]
        L.cublasLtMatmulAlgoCheck.argtypes = [vp, vp, vp, vp, vp, vp, ctypes.POINTER(Algo),
                                              ctypes.POINTER(Heur)]
        self.h = vp()
        self._chk(L.cublasLtCreate(ctypes.byref(self.h)), "create")
        self.ws_bytes = ws_bytes
        self.ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
        self.headers = find_headers(self.path)
        self.enums = {}
        self.tiles, self.stages, self.clusters = {}, {}, {}
        if self.headers:
            text = open(self.headers[0]).read()
            cfg = parse_enum(text, "CUBLASLT_ALGO_CONFIG_")
            cap = parse_enum(text, "CUBLASLT_ALGO_CAP_")
            self.enum_mismatch = {}
            for k in list(self.CFG):
                if k in cfg and cfg[k] != self.CFG[k]:
                    self.enum_mismatch["CFG_" + k] = (self.CFG[k], cfg[k])
                    self.CFG[k] = cfg[k]
            for k in list(self.CAP):
                if k in cap and cap[k] != self.CAP[k]:
                    self.enum_mismatch["CAP_" + k] = (self.CAP[k], cap[k])
                    self.CAP[k] = cap[k]
            self.tiles = {v: k for k, v in parse_enum(text, "CUBLASLT_MATMUL_TILE_").items()}
            self.stages = {v: k for k, v in parse_enum(text, "CUBLASLT_MATMUL_STAGES_").items()}
            self.clusters = {v: k for k, v in parse_enum(text, "CUBLASLT_CLUSTER_SHAPE_").items()}
        self._probs = {}

    @staticmethod
    def _chk(st, what):
        if st != 0:
            raise RuntimeError(f"cublasLt {what} status {st}")

    # ---------------------------------------------------------------- problem
    def problem(self, M, N, K):
        key = (M, N, K)
        if key in self._probs:
            return self._probs[key]
        L = self.L
        desc = vp()
        self._chk(L.cublasLtMatmulDescCreate(ctypes.byref(desc), COMPUTE_32F, R_32F), "desc")
        opT = ctypes.c_int(1)
        self._chk(L.cublasLtMatmulDescSetAttribute(desc, 3, ctypes.cast(ctypes.byref(opT), vp), 4), "transa")
        A, B, C = vp(), vp(), vp()
        self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(A), R_16BF, K, N, K), "A")
        self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(B), R_16BF, K, M, K), "B")
        self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(C), R_16BF, N, M, N), "C")
        pref = vp()
        self._chk(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
        wsb = ctypes.c_uint64(self.ws_bytes)
        self._chk(L.cublasLtMatmulPreferenceSetAttribute(pref, 1, ctypes.cast(ctypes.byref(wsb), vp), 8),
                  "prefws")
        p = (desc, A, B, C, pref)
        self._probs[key] = p
        return p

    def heuristics(self, prob, n=128):
        desc, A, B, C, pref = prob
        res = (Heur * n)()
        cnt = ctypes.c_int(0)
        self._chk(self.L.cublasLtMatmulAlgoGetHeuristic(self.h, desc, A, B, C, C, pref, n, res,
                                                        ctypes.byref(cnt)), "heur")
        return [res[i] for i in range(cnt.value)]

    def algo_ids(self):
        arr = (ctypes.c_int * 512)()
        cnt = ctypes.c_int(0)
        self._chk(self.L.cublasLtMatmulAlgoGetIds(self.h, COMPUTE_32F, R_32F, R_16BF, R_16BF, R_16BF,
                                                  R_16BF, 512, arr, ctypes.byref(cnt)), "getids")
        return [arr[i] for i in range(cnt.value)]

    def init(self, algo_id):
        a = Algo()
        st = self.L.cublasLtMatmulAlgoInit(self.h, COMPUTE_32F, R_32F, R_16BF, R_16BF, R_16BF, R_16BF,
                                           algo_id, ctypes.byref(a))
        return a if st == 0 else None

    def cap_array(self, algo, name):
        attr = self.CAP[name]
        sz = ctypes.c_size_t(0)
        self.L.cublasLtMatmulAlgoCapGetAttribute(ctypes.byref(algo), attr, None, 0, ctypes.byref(sz))
        n = sz.value // 4
        if n == 0:
            return []
        arr = (ctypes.c_uint32 * n)()
        st = self.L.cublasLtMatmulAlgoCapGetAttribute(ctypes.byref(algo), attr, arr, n * 4,
                                                      ctypes.byref(sz))
        return list(arr) if st == 0 else []

    def cap_scalar(self, algo, name):
        attr = self.CAP[name]
        for size, ct in ((4, ctypes.c_uint32), (8, ctypes.c_uint64), (2, ctypes.c_uint16)):
            v = ct(0)
            sz = ctypes.c_size_t(0)
            st = self.L.cublasLtMatmulAlgoCapGetAttribute(ctypes.byref(algo), attr, ctypes.byref(v), size,
                                                          ctypes.byref(sz))
            if st == 0:
                return int(v.value)
        return None

    def get_cfg(self, algo):
        out = {}
        for k, attr in self.CFG.items():
            size = self.CFG_SIZE.get(k, 4)
            v = (ctypes.c_uint16 if size == 2 else ctypes.c_int32)(0)
            sz = ctypes.c_size_t(0)
            st = self.L.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(algo), attr, ctypes.byref(v),
                                                             size, ctypes.byref(sz))
            out[k] = int(v.value) if st == 0 else None
        return out

    def make(self, cfg):
        """Algo from a config dict (ID, TILE_ID, STAGES_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING,
        CUSTOM_OPTION, CLUSTER_SHAPE_ID, INNER_SHAPE_ID)."""
        a = self.init(cfg["ID"])
        if a is None:
            return None
        for k in ("TILE_ID", "STAGES_ID", "SPLITK_NUM", "REDUCTION_SCHEME", "CTA_SWIZZLING",
                  "CUSTOM_OPTION", "CLUSTER_SHAPE_ID", "INNER_SHAPE_ID"):
            if cfg.get(k) is None:
                continue
            size = self.CFG_SIZE.get(k, 4)
            if size == 2:
                v = ctypes.c_uint16(cfg[k])
            elif k == "SPLITK_NUM":
                v = ctypes.c_int32(cfg[k])
            else:
                v = ctypes.c_uint32(cfg[k])
            st = self.L.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), self.CFG[k],
                                                             ctypes.byref(v), size)
            if st != 0:
                return None
        return a

    def check(self, prob, algo):
        desc, A, B, C, _ = prob
        r = Heur()
        st = self.L.cublasLtMatmulAlgoCheck(self.h, desc, A, B, C, C, ctypes.byref(algo), ctypes.byref(r))
        if st != 0 or r.state != 0 or r.workspaceSize > self.ws_bytes:
            return None
        return {"waves": float(r.wavesCount), "ws": int(r.workspaceSize)}

    def matmul_fn(self, prob, algo, x, w, y):
        desc, A, B, C, _ = prob
        L = self.L
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        pa = ctypes.cast(ctypes.byref(alpha), vp)
        pb = ctypes.cast(ctypes.byref(beta), vp)
        palgo = ctypes.cast(ctypes.byref(algo), vp)
        wp, xp, yp = vp(w.data_ptr()), vp(x.data_ptr()), vp(y.data_ptr())
        wsp = vp(self.ws.data_ptr())

        def fn():
            stream = vp(torch.cuda.current_stream().cuda_stream)
            st = L.cublasLtMatmul(self.h, desc, pa, wp, A, xp, B, pb, yp, C, yp, C, palgo, wsp,
                                  self.ws_bytes, stream)
            if st != 0:
                raise RuntimeError(f"matmul status {st}")
            return y

        fn._keep = (alpha, beta, algo)
        return fn

    def tile_dims(self, tid):
        name = self.tiles.get(tid, "")
        m = re.match(r"(\d+)x(\d+)", name)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def describe(self, cfg):
        t = self.tiles.get(cfg.get("TILE_ID"), cfg.get("TILE_ID"))
        s = self.stages.get(cfg.get("STAGES_ID"), cfg.get("STAGES_ID"))
        c = self.clusters.get(cfg.get("CLUSTER_SHAPE_ID"), cfg.get("CLUSTER_SHAPE_ID"))
        return (f"algo{cfg.get('ID')} tile{cfg.get('TILE_ID')}({t}) st{cfg.get('STAGES_ID')}({s}) "
                f"sk{cfg.get('SPLITK_NUM')} red{cfg.get('REDUCTION_SCHEME')} swz{cfg.get('CTA_SWIZZLING')} "
                f"cus{cfg.get('CUSTOM_OPTION')} cga{cfg.get('CLUSTER_SHAPE_ID')}({c}) "
                f"inner{cfg.get('INNER_SHAPE_ID')}")
