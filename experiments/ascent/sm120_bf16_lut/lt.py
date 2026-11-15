"""Minimal ctypes binding of cuBLASLt for BF16 GEMM algo sweeps (SM120 attention projections).

Layout mirrors trtllm::cublas_mm (cpp/tensorrt_llm/thop/cublasScaledMM.cpp):
  cublasLt col-major GEMM  D[n, m] = W^T(op T on [k, n]) x X([k, m]),  i.e. out[M, N] = X[M, K] @ W[N, K]^T
"""
import ctypes
import glob
import json
import os
import re

import torch

CUBLAS_COMPUTE_32F = 68
CUDA_R_32F = 0
CUDA_R_16BF = 14
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
DESC_TRANSA = 3
DESC_TRANSB = 4
PREF_MAX_WORKSPACE_BYTES = 1
WORKSPACE = 32 * 1024 * 1024

# cublasLtMatmulAlgoConfigAttributes_t
CFG_ID, CFG_TILE, CFG_SPLITK, CFG_RED, CFG_SWIZZLE, CFG_CUSTOM, CFG_STAGES, CFG_INNER, CFG_CGA = range(9)


class Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class HeurResult(ctypes.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def _find_lib():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.linear(a, a)
    torch.cuda.synchronize()
    with open("/proc/self/maps") as f:
        for line in f:
            if "libcublasLt" in line:
                return line.split()[-1]
    cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cublas", "lib",
                                   "libcublasLt.so*"))
    return cands[0]


def _enum_names():
    """Parse tile/stages enum names from cublasLt.h if present (readability only)."""
    hdrs = []
    for root in [os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cublas", "include"),
                 "/usr/local/cuda/include", "/usr/include"]:
        p = os.path.join(root, "cublasLt.h")
        if os.path.exists(p):
            hdrs.append(p)
    tiles, stages = {}, {}
    if not hdrs:
        return tiles, stages, None
    txt = open(hdrs[0]).read()
    for name, val in re.findall(r"CUBLASLT_MATMUL_TILE_(\w+)\s*=\s*(\d+)", txt):
        tiles[int(val)] = name
    for name, val in re.findall(r"CUBLASLT_MATMUL_STAGES_(\w+)\s*=\s*(\d+)", txt):
        stages[int(val)] = name
    return tiles, stages, hdrs[0]


class Lt:

    def __init__(self):
        self.path = _find_lib()
        self.lib = ctypes.CDLL(self.path)
        self.lib.cublasLtGetVersion.restype = ctypes.c_size_t
        self.version = self.lib.cublasLtGetVersion()
        self.h = ctypes.c_void_p()
        self._ck(self.lib.cublasLtCreate(ctypes.byref(self.h)), "create")
        self.tiles, self.stages, self.hdr = _enum_names()
        self.ws = torch.empty(WORKSPACE, dtype=torch.uint8, device="cuda")
        self._desc_cache = {}
        self.alpha = ctypes.c_float(1.0)
        self.beta = ctypes.c_float(0.0)

    @staticmethod
    def _ck(st, what):
        if st != 0:
            raise RuntimeError(f"cublasLt {what} failed status={st}")

    def descs(self, m, n, k):
        key = (m, n, k)
        if key in self._desc_cache:
            return self._desc_cache[key]
        L = self.lib
        op = ctypes.c_void_p()
        self._ck(L.cublasLtMatmulDescCreate(ctypes.byref(op), CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc")
        ta = ctypes.c_int(CUBLAS_OP_T)
        tb = ctypes.c_int(CUBLAS_OP_N)
        self._ck(L.cublasLtMatmulDescSetAttribute(op, DESC_TRANSA, ctypes.byref(ta), 4), "ta")
        self._ck(L.cublasLtMatmulDescSetAttribute(op, DESC_TRANSB, ctypes.byref(tb), 4), "tb")
        A, B, C = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        # cublas m=n_out(N), n=m_tok(M), k=K ; A = W stored [k x N] ld k (op T); B = X [k x M] ld k; C [N x M] ld N
        u64 = ctypes.c_uint64
        i64 = ctypes.c_int64
        self._ck(L.cublasLtMatrixLayoutCreate(ctypes.byref(A), CUDA_R_16BF, u64(k), u64(n), i64(k)), "A")
        self._ck(L.cublasLtMatrixLayoutCreate(ctypes.byref(B), CUDA_R_16BF, u64(k), u64(m), i64(k)), "B")
        self._ck(L.cublasLtMatrixLayoutCreate(ctypes.byref(C), CUDA_R_16BF, u64(n), u64(m), i64(n)), "C")
        pref = ctypes.c_void_p()
        self._ck(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
        wsb = ctypes.c_uint64(WORKSPACE)
        self._ck(L.cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE_BYTES, ctypes.byref(wsb), 8),
                 "prefws")
        d = (op, A, B, C, pref)
        self._desc_cache[key] = d
        return d

    def heuristics(self, m, n, k, count=32):
        op, A, B, C, pref = self.descs(m, n, k)
        res = (HeurResult * count)()
        ret = ctypes.c_int(0)
        self._ck(self.lib.cublasLtMatmulAlgoGetHeuristic(self.h, op, A, B, C, C, pref, count, res,
                                                          ctypes.byref(ret)), "heur")
        out = []
        for i in range(ret.value):
            if res[i].state == 0 and res[i].workspaceSize <= WORKSPACE:
                a = Algo()
                ctypes.memmove(ctypes.byref(a), ctypes.byref(res[i].algo), ctypes.sizeof(Algo))
                out.append((self.config(a), res[i].workspaceSize, res[i].wavesCount, a))
        return out

    def config(self, algo):
        vals = []
        for attr, sz in [(CFG_ID, 4), (CFG_TILE, 4), (CFG_STAGES, 4), (CFG_SPLITK, 4), (CFG_RED, 4),
                         (CFG_SWIZZLE, 4), (CFG_CUSTOM, 4), (CFG_CGA, 2), (CFG_INNER, 2)]:
            buf = ctypes.c_uint32(0) if sz == 4 else ctypes.c_uint16(0)
            wr = ctypes.c_size_t(0)
            self._ck(self.lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(algo), attr, ctypes.byref(buf), sz,
                                                                  ctypes.byref(wr)), f"get{attr}")
            v = buf.value
            if attr == CFG_SPLITK and v >= 2**31:
                v -= 2**32
            vals.append(int(v))
        # (algo, tile, stages, numsK, reduction, swizzle, custom, cga) exactly as the TRT-LLM LUT; inner kept separately
        return tuple(vals[:8]), vals[8]

    def make_algo(self, cfg8):
        """Rebuild an algo exactly like set_algo_attr in cublasScaledMM.cpp (inner shape left at init default)."""
        algo = Algo()
        L = self.lib
        self._ck(L.cublasLtMatmulAlgoInit(self.h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                          CUDA_R_16BF, CUDA_R_16BF, cfg8[0], ctypes.byref(algo)), "init")
        algo_id, tile, stages, numsk, red, swz, custom, cga = cfg8
        for attr, val, ty in [(CFG_TILE, tile, ctypes.c_int), (CFG_STAGES, stages, ctypes.c_int),
                              (CFG_SPLITK, numsk, ctypes.c_int), (CFG_RED, red, ctypes.c_int),
                              (CFG_SWIZZLE, swz, ctypes.c_int), (CFG_CUSTOM, custom, ctypes.c_uint32),
                              (CFG_CGA, cga, ctypes.c_uint16)]:
            b = ty(val)
            self._ck(L.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(algo), attr, ctypes.byref(b),
                                                            ctypes.sizeof(b)), f"set{attr}")
        return algo

    def check(self, m, n, k, algo):
        op, A, B, C, pref = self.descs(m, n, k)
        r = HeurResult()
        st = self.lib.cublasLtMatmulAlgoCheck(self.h, op, A, B, C, C, ctypes.byref(algo), ctypes.byref(r))
        return st == 0 and r.state == 0 and r.workspaceSize <= WORKSPACE

    def matmul(self, x, w, out, algo=None):
        """out[M,N] = x[M,K] @ w[N,K]^T ; algo None -> cuBLASLt internal heuristic (as cublas_mm without LUT hit)."""
        m, k = x.shape
        n = w.shape[0]
        op, A, B, C, pref = self.descs(m, n, k)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        st = self.lib.cublasLtMatmul(self.h, op, ctypes.byref(self.alpha), ctypes.c_void_p(w.data_ptr()), A,
                                     ctypes.c_void_p(x.data_ptr()), B, ctypes.byref(self.beta),
                                     ctypes.c_void_p(out.data_ptr()), C, ctypes.c_void_p(out.data_ptr()), C,
                                     ctypes.byref(algo) if algo is not None else None,
                                     ctypes.c_void_p(self.ws.data_ptr()), ctypes.c_size_t(WORKSPACE), stream)
        if st != 0:
            raise RuntimeError(f"cublasLtMatmul status {st}")

    def name(self, cfg8):
        a, t, s, nk, r, sw, c, g = cfg8
        return f"a{a}/{self.tiles.get(t, t)}/{self.stages.get(s, s)}/sk{nk}/r{r}/sw{sw}/c{c}/g{g}"


def read_shapes(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = tc["hidden_size"]
    nq = tc["num_attention_heads"]
    hd = tc["head_dim"]
    nkv = tc["num_key_value_heads"]
    ghd = tc.get("global_head_dim", hd)
    gkv = tc.get("num_global_key_value_heads", nkv) or nkv
    keqv = tc.get("attention_k_eq_v", False)
    info = dict(hidden=H, q_heads=nq, head_dim=hd, kv_heads=nkv, global_head_dim=ghd, global_kv_heads=gkv,
                k_eq_v=keqv, layers=tc.get("num_hidden_layers"),
                sliding=sum(1 for t in tc.get("layer_types", []) if t == "sliding_attention"))
    # Fused qkv as main builds it: V duplicated from K on k_eq_v layers (knowledge 255), so 2 kv blocks.
    shapes = {
        "sl_qkv": (nq * hd + 2 * nkv * hd, H),
        "gl_qkv": (nq * ghd + 2 * gkv * ghd, H),
        "sl_o": (H, nq * hd),
        "gl_o": (H, nq * ghd),
    }
    if keqv:
        # Once the k_eq_v V-dedup (landing 1) is on main, the global fused qkv drops the duplicated V block.
        shapes["gl_qkv_dedup"] = (nq * ghd + gkv * ghd, H)
    return shapes, info
