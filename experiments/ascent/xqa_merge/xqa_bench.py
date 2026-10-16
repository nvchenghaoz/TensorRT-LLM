#!/usr/bin/env python3
# XQA kernel_mha (SM120 HMMA path) microbenchmark + correctness harness for the Qwen3.6-35B-A3B decode shape.
#
# The serving path JIT-compiles cpp/kernels/xqa/mha.cu with NVRTC (decoderXQAImplJIT/nvrtcWrapper). This harness
# compiles the same source from the current tree with the same NVRTC options/macros, loads the cubin with the driver
# API and launches kernel_mha exactly like DecoderXQARunner::runImpl (same parameter list, grid {nbSubSeq, nbKHeads,
# batch}, block {128, 1, 2}, dynamic smem = smemSize). Inputs are synthetic tensors of the serving shapes:
# 16 Q heads / 2 KV heads, head_dim 256, BF16 Q/output, FP8-E4M3 paged KV cache (32 tokens per block, K/V pages
# interleaved like the TRT-LLM block pool), 32768-token shared prefix (KV block reuse) + 2048 unique suffix + 128
# decoded tokens per request, max_seq_len 36864, default multi-block count 48 (captured with max_past_kv_length 36863).
#
# Modes
#   bench    : time kernel_mha (CUDA graph replay, L2 flushed before every call, CUDA events) for each batch size;
#              last stdout line is the evidence JSON. kernel_ms is the 10-90% trimmed mean (CUDA event timestamps on
#              this GPU are quantized to ~1 us, so the median alone is too coarse); the median is reported as well.
#   check    : untimed sequence with the same warmup/iteration/flush structure; compares every invocation against the
#              first one, against the unmodified original source (git base commit) and against an FP32 reference.
#   explore  : compare variants (in-memory patches of the original source, for exploration only).
#   timeline : in-kernel globaltimer/clock64 instrumentation of the original source (exploration only).
import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time

import numpy as np
import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

BASE_COMMIT = "2647f76a1b89cefd08846ea4167b0fa582c8136d"
CUDA_INC = "/usr/local/cuda/include"
XQA_HEADERS = [
    "cuda_hint.cuh", "defines.h", "ldgsts.cuh", "mha.h", "mhaUtils.cuh", "mma.cuh", "platform.h", "utils.cuh",
    "utils.h", "mha_stdheaders.cuh", "mha_components.cuh", "mla_sm120.cuh", "gmma.cuh", "gmma_impl.cuh",
    "barriers.cuh", "tma.h", "specDec.h"
]
CUDA_HEADERS = [
    "cuda_bf16.h", "cuda_bf16.hpp", "cuda_fp16.h", "cuda_fp16.hpp", "cuda_fp8.h", "cuda_fp8.hpp", "vector_types.h",
    "vector_functions.h", "device_types.h"
]

NB_K_HEADS = 2
HEAD_GRP = 8
NB_Q_HEADS = NB_K_HEADS * HEAD_GRP
HEAD_DIM = 256
TOKENS_PER_PAGE = 32
MAX_SEQ_LEN = 36864
MAX_PAGES = MAX_SEQ_LEN // TOKENS_PER_PAGE
PREFIX = 32768
SEQ_LEN = 32768 + 2048 + 128
DEFAULT_SUBSEQ = 48
SLIDING_WIN = MAX_SEQ_LEN
Q_SCALE = 1.0
KV_SCALE = 0.5
NB_SEM = 1024
SCRATCH_MB = 64
DBG_CTA_OFF = 40 << 20
DBG_MERGE_OFF = 48 << 20


def log(*a):
    print(*a, flush=True)


def ck(res, what):
    if isinstance(res, tuple):
        err, rest = res[0], res[1:]
    else:
        err, rest = res, ()
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


def serving_macros():
    # Same macro set as nvrtcWrapper.cpp getMacroFlags() for this model: BF16 input, FP8 KV cache, head 256,
    # 8 Q heads per KV head, 32 tokens per block, beam 1, generation (no spec-dec), no fp8 output, RoPE outside.
    return {
        "INPUT_FP16": "0",
        "DTYPE": "__nv_bfloat16",
        "GENERATE_CUBIN": "1",
        "NDEBUG": "1",
        "HEAD_ELEMS": str(HEAD_DIM),
        "BEAM_WIDTH": "1",
        "CACHE_ELEM_ENUM": "2",
        "TOKENS_PER_PAGE": str(TOKENS_PER_PAGE),
        "HEAD_GRP_SIZE": str(HEAD_GRP),
        "M_TILESIZE": str(HEAD_GRP),
        "USE_CUSTOM_BARRIER": "1",
        "SLIDING_WINDOW": "1",
        "LOW_PREC_OUTPUT": "0",
        "USE_INPUT_KV": "0",
        "ROPE_STYLE": "0",
        "ROPE_ELEMS": "64",
        "IS_SPEC_DEC_TREE": "0",
        "SKIP_SOFTMAX_ATTN": "0",
        "__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__": "1",
        "__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__": "1",
    }


def compile_xqa(src_dir, extra_defines=(), patch=None):
    t0 = time.time()
    with open(os.path.join(src_dir, "mha.cu")) as f:
        src = f.read()
    if patch is not None:
        src = patch(src)
    names, contents = [], []
    for h in XQA_HEADERS:
        names.append(h.encode())
        with open(os.path.join(src_dir, h), "rb") as f:
            contents.append(f.read())
    for h in CUDA_HEADERS:
        names.append(h.encode())
        with open(os.path.join(CUDA_INC, h), "rb") as f:
            contents.append(f.read())
    prog = ck(nvrtc.nvrtcCreateProgram(src.encode(), b"default_program", len(names), contents, names), "create")
    opts = [b"-dw", b"--use_fast_math", b"-default-device", b"-arch=sm_120f"]
    opts += [f"-D{k}={v}".encode() for k, v in serving_macros().items()]
    opts += [f"-D{d}".encode() for d in extra_defines]
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
    if int(err) != 0:
        size = ck(nvrtc.nvrtcGetProgramLogSize(prog), "logsize")
        buf = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("NVRTC compile failed:\n" + buf.decode(errors="replace")[-6000:])
    size = ck(nvrtc.nvrtcGetCUBINSize(prog), "cubinsize")
    cubin = b" " * size
    ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "getcubin")
    nvrtc.nvrtcDestroyProgram(prog)
    log(f"  compiled {src_dir} defines={list(extra_defines)} patched={patch is not None} "
        f"in {time.time() - t0:.1f}s, cubin {size} B, src sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}")
    return cubin


class KVCacheList(ctypes.Structure):
    _fields_ = [("pool", ctypes.c_uint64), ("pageList", ctypes.c_uint64), ("seqLen", ctypes.c_uint64),
                ("maxNbPagesPerSeq", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


class Kernel:

    def __init__(self, cubin):
        self._img = np.frombuffer(cubin, dtype=np.uint8).copy()
        self.mod = ck(cu.cuModuleLoadData(self._img.ctypes.data), "cuModuleLoadData")
        self.func = ck(cu.cuModuleGetFunction(self.mod, b"kernel_mha"), "cuModuleGetFunction")
        dptr, nbytes = ck(cu.cuModuleGetGlobal(self.mod, b"smemSize"), "cuModuleGetGlobal")
        v = ctypes.c_uint32(0)
        ck(cu.cuMemcpyDtoH(ctypes.addressof(v), dptr, 4), "cuMemcpyDtoH")
        self.smem = int(v.value)
        A = cu.CUfunction_attribute
        ck(cu.cuFuncSetAttribute(self.func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem), "setattr")
        self.regs = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func), "regs")
        self.lmem = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func), "lmem")

    def launch(self, args, grid):
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        ck(cu.cuLaunchKernel(self.func, grid[0], grid[1], grid[2], 128, 1, 2, self.smem, stream, args.ptr, 0),
           "cuLaunchKernel")


class Args:

    def __init__(self, inp, batch):
        self.objs = [
            ctypes.c_uint32(NB_K_HEADS),
            ctypes.c_uint32(SLIDING_WIN),
            ctypes.c_float(Q_SCALE),
            ctypes.c_uint64(inp["out"].data_ptr()),
            ctypes.c_uint64(inp["q"].data_ptr()),
            ctypes.c_uint64(0),
            KVCacheList(inp["pool"].data_ptr(), inp["pages"].data_ptr(), inp["seqlen"].data_ptr(), MAX_PAGES, 0),
            ctypes.c_uint32(batch),
            ctypes.c_uint64(inp["kvscale"].data_ptr()),
            ctypes.c_uint64(inp["sem"].data_ptr()),
            ctypes.c_uint64(inp["scratch_ptr"]),
        ]
        self.arr = (ctypes.c_void_p * len(self.objs))(*[ctypes.addressof(o) for o in self.objs])
        self.ptr = ctypes.addressof(self.arr)


def build_inputs(bmax, seed=1234):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    nbp = math.ceil(SEQ_LEN / TOKENS_PER_PAGE)
    prefix_blocks = PREFIX // TOKENS_PER_PAGE
    uniq = nbp - prefix_blocks
    nb_blocks = prefix_blocks + bmax * uniq
    pool = (torch.randn(nb_blocks * 2, NB_K_HEADS, TOKENS_PER_PAGE, HEAD_DIM, generator=g, device="cuda") *
            2.0).to(torch.float8_e4m3fn)
    pages = torch.zeros(bmax, 1, 2, MAX_PAGES, dtype=torch.int32)
    for b in range(bmax):
        blocks = list(range(prefix_blocks)) + [prefix_blocks + b * uniq + j for j in range(uniq)]
        blk = torch.tensor(blocks, dtype=torch.int32)
        pages[b, 0, 0, :nbp] = 2 * blk
        pages[b, 0, 1, :nbp] = 2 * blk + 1
    pages = pages.cuda()
    q = torch.randn(bmax, NB_Q_HEADS, HEAD_DIM, generator=g, device="cuda").to(torch.bfloat16)
    seqlen = torch.full((bmax, ), SEQ_LEN, dtype=torch.int32, device="cuda")
    kvscale = torch.tensor([KV_SCALE], dtype=torch.float32, device="cuda")
    out = torch.zeros(bmax, NB_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    sem = torch.zeros(NB_SEM, dtype=torch.int32, device="cuda")
    scratch_raw = torch.zeros(SCRATCH_MB << 20, dtype=torch.uint8, device="cuda")
    base = scratch_raw.data_ptr()
    scratch_ptr = (base + 4095) // 4096 * 4096
    return dict(pool=pool, pages=pages, q=q, seqlen=seqlen, kvscale=kvscale, out=out, sem=sem,
                scratch_raw=scratch_raw, scratch_ptr=scratch_ptr, scratch_off=scratch_ptr - base, nbp=nbp)


def tbytes(t):
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def readonly_hash(inp):
    h = hashlib.sha256()
    for k in ("pool", "pages", "q", "seqlen", "kvscale"):
        h.update(tbytes(inp[k]))
    return h.hexdigest()


def reference(inp, batch):
    torch.backends.cuda.matmul.allow_tf32 = False
    nbp = inp["nbp"]
    outs = []
    for b in range(batch):
        kp = inp["pages"][b, 0, 0, :nbp].long()
        vp = inp["pages"][b, 0, 1, :nbp].long()
        k = inp["pool"][kp].float().permute(1, 0, 2, 3).reshape(NB_K_HEADS, nbp * TOKENS_PER_PAGE,
                                                                 HEAD_DIM)[:, :SEQ_LEN] * KV_SCALE
        v = inp["pool"][vp].float().permute(1, 0, 2, 3).reshape(NB_K_HEADS, nbp * TOKENS_PER_PAGE,
                                                                 HEAD_DIM)[:, :SEQ_LEN] * KV_SCALE
        q = inp["q"][b].float().view(NB_K_HEADS, HEAD_GRP, HEAD_DIM)
        logits = torch.einsum("hgd,hsd->hgs", q, k) * (Q_SCALE / math.sqrt(HEAD_DIM))
        p = torch.softmax(logits, dim=-1)
        o = torch.einsum("hgs,hsd->hgd", p, v).reshape(NB_Q_HEADS, HEAD_DIM)
        outs.append(o)
    # FP32 accumulation, rounded once to the output dtype (BF16).
    return torch.stack(outs).to(torch.bfloat16).float()


def errors(out, ref):
    d = (out.float() - ref).abs()
    return float(d.max().item()), float(d.mean().item()), bool(torch.isfinite(out.float()).all().item())


def sha(t):
    return hashlib.sha256(tbytes(t)).hexdigest()


class Flusher:
    """Evicts the KV cache from L2 before every call. 'read' leaves clean lines in L2 (like the MoE weight reads that
    precede attention in serving); 'write' leaves dirty lines."""

    def __init__(self, kind, mb):
        self.kind = kind
        self.buf = None if kind == "none" else torch.ones((mb << 20) // 4, dtype=torch.int32, device="cuda")

    def __call__(self):
        if self.kind == "read":
            self.buf.sum()
        elif self.kind == "write":
            self.buf.zero_()


class Runner:
    """One (kernel, batch, nbSubSeq) configuration captured into a CUDA graph."""

    def __init__(self, kern, inp, batch, nsub):
        self.kern, self.inp, self.batch, self.nsub = kern, inp, batch, nsub
        self.args = Args(inp, batch)
        self.grid = (nsub, NB_K_HEADS, batch)
        kern.launch(self.args, self.grid)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            kern.launch(self.args, self.grid)
        torch.cuda.synchronize()

    def out(self):
        return self.inp["out"][:self.batch]


def time_runner(r, flush, warmup, iters):
    n = warmup + iters
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        flush()
        st[i].record()
        r.graph.replay()
        en[i].record()
    torch.cuda.synchronize()
    return np.array([st[i].elapsed_time(en[i]) for i in range(warmup, n)])


def stats(ts):
    med = float(np.median(ts))
    p10, p90 = float(np.percentile(ts, 10)), float(np.percentile(ts, 90))
    core = ts[(ts >= p10) & (ts <= p90)]
    tmean = float(core.mean()) if core.size else med
    return dict(kernel_ms=tmean, median_ms=med, p10_ms=p10, p90_ms=p90, mean_ms=float(ts.mean()),
                noise_floor=(p90 - p10) / tmean, samples=int(ts.size))


def extract_base_tree(commit):
    d = tempfile.mkdtemp(prefix="xqa_base_")
    tar = subprocess.run(["git", "archive", commit, "cpp/kernels/xqa"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
    return os.path.join(d, "cpp/kernels/xqa")


# ----------------------------------------------------------------------------------------------------------------
# Exploration-only in-memory patches of the ORIGINAL source (never used for evidence).
def _rep(src, old, new):
    if src.count(old) != 1:
        raise RuntimeError(f"patch anchor not found exactly once: {old[:80]!r}")
    return src.replace(old, new)


ROWMAX_BLK = ("                // get row max.\n"
              "                if (warpIdx.x == 0)\n"
              "                {\n"
              "                    ThrdRegRowMax const mergedRowMax = mergeRowMax<8>(warp, rowMaxBuffers + idxBufBase, "
              "nbSubSeqPerSeq);\n"
              "                    smemRowMax.storeFromReg<false>(warp, mergedRowMax);\n"
              "                }\n"
              "                __syncthreads();\n"
              "                ThrdRegRowMax const mergedRowMax = smemRowMax.loadToReg<false>(warp);\n")
LOOP_BEG = "                loadBufAsync(warpGrpIdx);\n"
LOOP_END = "\n                ThrdRegRowMax mergedRowSum{};"
NEW_LOOP = """                constexpr uint32_t nbMergeUnroll = @U@;
                constexpr uint32_t nbMergeGrps = nbTileBuffers / nbMergeUnroll;
                static_assert(nbTileBuffers % nbMergeUnroll == 0 && nbMergeGrps >= 2);
#pragma unroll
                for (uint32_t g = 0; g + 1 < nbMergeGrps; g++)
                {
#pragma unroll
                    for (uint32_t u = 0; u < nbMergeUnroll; u++)
                    {
                        uint32_t const nPre = warpGrpIdx + gemm1NbWarpGrps * (nbMergeUnroll * g + u);
                        if (nPre < nbSubSeqPerSeq)
                        {
                            loadBufAsync(nPre);
                        }
                    }
                    ldgsts::commitGroup();
                }
@ROWMAX@                WarpAcc sumAcc{};
                ThrdRegRowMax partialMergedRowSum{};
                for (uint32_t n0 = warpGrpIdx; n0 < nbSubSeqPerSeq; n0 += gemm1NbWarpGrps * nbMergeUnroll)
                {
#pragma unroll
                    for (uint32_t u = 0; u < nbMergeUnroll; u++)
                    {
                        uint32_t const nPre = n0 + gemm1NbWarpGrps * (nbMergeUnroll * (nbMergeGrps - 1) + u);
                        if (nPre < nbSubSeqPerSeq)
                        {
                            loadBufAsync(nPre);
                        }
                    }
                    ldgsts::commitGroup();
                    ldgsts::waitGroup<nbMergeGrps - 1>();
                    __syncwarp();
                    WarpAcc tiles[nbMergeUnroll];
                    ThrdRegRowMax scaledTileRowSums[nbMergeUnroll];
#pragma unroll
                    for (uint32_t u = 0; u < nbMergeUnroll; u++)
                    {
                        uint32_t const n = n0 + gemm1NbWarpGrps * u;
                        uint32_t const d = n / gemm1NbWarpGrps % nbTileBuffers;
                        tiles[u] = toWarpAcc(loadGemmOutTile(warp, mbbuf.tiles[warpGrpIdx][warpIdxInGrp][d]));
                        ThrdRegRowMax const tileRowMax = getTileBuf(mbbuf.tileRowMax, d).loadToReg<false>(warp);
                        ThrdRegRowMax const tileRowSum = getTileBuf(mbbuf.tileRowSums, d).loadToReg<false>(warp);
                        ThrdRegRowMax const tileRowScales = expf(tileRowMax - mergedRowMax);
                        scaledTileRowSums[u] = tileRowSum * tileRowScales;
                        rescaleAcc(warp, tiles[u], fullRescaleMask, scaledTileRowSums[u]);
                    }
#pragma unroll
                    for (uint32_t u = 0; u < nbMergeUnroll; u++)
                    {
                        if (n0 + gemm1NbWarpGrps * u < nbSubSeqPerSeq)
                        {
                            partialMergedRowSum = partialMergedRowSum + scaledTileRowSums[u];
                            sumAcc = sumAcc + tiles[u];
                        }
                    }
                    __syncwarp();
                }
"""


def patch_merge(src, unroll, depth, early, rm16):
    src = _rep(src, "constexpr uint32_t nbTileBuffers = 2;", f"constexpr uint32_t nbTileBuffers = {depth};")
    if early:
        src = _rep(src, ROWMAX_BLK, "")
        src = _rep(src, "SMemWarpRowMax& smemRowMax = reinterpret_cast<SMemWarpRowMax&>(smem);",
                   "SMemWarpRowMax& smemRowMax = mbbuf.rowMax;")
    if src.count(LOOP_BEG) != 1 or src.count(LOOP_END) != 1:
        raise RuntimeError("loop anchors not found")
    i0, i1 = src.index(LOOP_BEG), src.index(LOOP_END)
    new = NEW_LOOP.replace("@U@", str(unroll)).replace("@ROWMAX@", ROWMAX_BLK if early else "")
    src = src[:i0] + new + src[i1:]
    if rm16:
        src = _rep(src, "mergeRowMax<8>(", "mergeRowMax<16>(")
    return src


GT = "asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"({}));"
DBG_CTA = ("            {\n"
           "                uint64_t dbgT1;\n"
           "                " + GT.format("dbgT1") + "\n"
           "                if (warpIdx.y == 0 && warpGrpIdx == 0 && warpIdxInGrp == 0 && laneId() == 0)\n"
           "                {\n"
           f"                    uint64_t* const dbgCta = reinterpret_cast<uint64_t*>(static_cast<char*>(scratch) + {DBG_CTA_OFF}u);\n"
           "                    dbgCta[2 * idxBuf] = dbgT0;\n"
           "                    dbgCta[2 * idxBuf + 1] = dbgT1;\n"
           "                }\n"
           "            }\n")
DBG_MERGE = ("                long long const dbgC3 = clock64();\n"
             "                uint64_t dbgT2;\n"
             "                " + GT.format("dbgT2") + "\n"
             "                if (warpIdx.x == 0 && laneId() == 0)\n"
             "                {\n"
             f"                    uint64_t* const dbgMerge = reinterpret_cast<uint64_t*>(static_cast<char*>(scratch) + {DBG_MERGE_OFF}u) + 8 * idxSeq;\n"
             "                    dbgMerge[0] = dbgC1 - dbgC0;\n"
             "                    dbgMerge[1] = dbgC2 - dbgC1;\n"
             "                    dbgMerge[2] = dbgC3 - dbgC2;\n"
             "                    dbgMerge[3] = dbgT2;\n"
             "                }\n")


def patch_timing(src):
    src = _rep(src, "    uint32_t const idxReq = blockIdx.z;\n",
               "    uint32_t const idxReq = blockIdx.z;\n    uint64_t dbgT0;\n    " + GT.format("dbgT0") + "\n")
    src = _rep(src, "            // increase the semaphore by 1\n", DBG_CTA + "            // increase the semaphore by 1\n")
    a = "                MultiBlockSMem::MBBuf& mbbuf = mbsmem.storage[warpIdx.y];\n"
    src = _rep(src, a, a + "                long long const dbgC0 = clock64();\n")
    a = "                ThrdRegRowMax const mergedRowMax = smemRowMax.loadToReg<false>(warp);\n"
    src = _rep(src, a, a + "                long long const dbgC1 = clock64();\n")
    src = _rep(src, LOOP_END, "\n                long long const dbgC2 = clock64();" + LOOP_END)
    a = "                smemOutTile = mergeAndSaveOutTile(mergedOutTile, false);\n"
    src = _rep(src, a, a + DBG_MERGE)
    return src


def make_variant(spec, tree_dir, base_dir):
    # spec: base | tree | nomerge | norowmax | timing | uU_dD[_early][_rm16]
    if spec == "base":
        return compile_xqa(base_dir)
    if spec == "tree":
        return compile_xqa(tree_dir)
    if spec == "timing":
        return compile_xqa(base_dir, patch=patch_timing)
    if spec == "nomerge":  # diagnostic only: merge only the first partial (wrong output)
        return compile_xqa(base_dir, patch=lambda s: _rep(
            s, "for (uint32_t n = warpGrpIdx; n < nbSubSeqPerSeq; n += gemm1NbWarpGrps)",
            "for (uint32_t n = warpGrpIdx; n < 1u; n += gemm1NbWarpGrps)"))
    if spec == "norowmax":  # diagnostic only: skip the global row-max pass (wrong output)
        return compile_xqa(base_dir, patch=lambda s: _rep(
            s, "mergeRowMax<8>(warp, rowMaxBuffers + idxBufBase, nbSubSeqPerSeq)", "ThrdRegRowMax::filled(0.F)"))
    parts = spec.split("_")
    u = int(parts[0][1:])
    d = int(parts[1][1:])
    early = "early" in parts
    rm16 = "rm16" in parts
    return compile_xqa(base_dir, patch=lambda s: patch_merge(s, u, d, early, rm16))


def parse_configs(s):
    out = []
    for c in s.split(","):
        b, m = c.split("x")
        out.append((int(b), int(m)))
    return out


# ----------------------------------------------------------------------------------------------------------------
def mode_bench(a):
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    kern = Kernel(compile_xqa(tree))
    log(f"kernel: smem {kern.smem} B, regs {kern.regs}, local {kern.lmem} B")
    batches = [int(x) for x in a.batches.split(",")]
    inp = build_inputs(max(batches))
    flush = Flusher(a.flush, a.flush_mb)
    res = {
        "batch": {},
        "output": {
            "class": "exact",
            "sha256": {},
            "max_abs_err_vs_fp32": {},
            "mean_abs_err_vs_fp32": {}
        }
    }
    for b in batches:
        r = Runner(kern, inp, b, a.nsub)
        out0 = r.out().clone()
        ref = reference(inp, b)
        mx, mn, fin = errors(out0, ref)
        rounds = []
        for k in range(a.rounds):
            ts = time_runner(r, flush, a.warmup, a.iters)
            rounds.append(ts)
            log(f"  B={b} m={a.nsub} round {k}: trimmed mean {stats(ts)['kernel_ms'] * 1e3:.2f} us")
        ts = np.concatenate(rounds)
        st = stats(ts)
        res["batch"][str(b)] = st
        res["output"]["sha256"][str(b)] = sha(r.out())
        res["output"]["max_abs_err_vs_fp32"][str(b)] = mx
        res["output"]["mean_abs_err_vs_fp32"][str(b)] = mn
        log(f"B={b}: trimmed mean {st['kernel_ms'] * 1e3:.2f} us median {st['median_ms'] * 1e3:.2f} p10 "
            f"{st['p10_ms'] * 1e3:.2f} p90 {st['p90_ms'] * 1e3:.2f} noise {st['noise_floor']:.3f}; err max {mx:.3e} "
            f"mean {mn:.3e} finite {fin}; same-as-first {torch.equal(out0, r.out())}")
    res["config"] = dict(nb_sub_seq=a.nsub, seq_len=SEQ_LEN, prefix=PREFIX, flush=a.flush, flush_mb=a.flush_mb,
                         warmup=a.warmup, iters=a.iters, rounds=a.rounds, smem=kern.smem, regs=kern.regs,
                         local_bytes=kern.lmem, timing="cuda graph replay + cuda events; kernel_ms = 10-90% trimmed mean")
    print(json.dumps(res), flush=True)


def mode_explore(a):
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    base = extract_base_tree(a.base_commit)
    specs = a.variants.split(",")
    kerns = {}
    for s in specs:
        try:
            kerns[s] = Kernel(make_variant(s, tree, base))
            log(f"variant {s}: smem {kerns[s].smem} regs {kerns[s].regs} local {kerns[s].lmem}")
        except Exception as e:  # noqa: BLE001
            log(f"variant {s}: FAILED {str(e)[-3000:]}")
    configs = parse_configs(a.configs)
    inp = build_inputs(max(b for b, _ in configs))
    flush = Flusher(a.flush, a.flush_mb)
    summary = {}
    for (b, m) in configs:
        runners = {s: Runner(k, inp, b, m) for s, k in kerns.items()}
        outs = {}
        for s, r in runners.items():
            r.graph.replay()
            torch.cuda.synchronize()
            outs[s] = r.out().clone()
        ref_key = "base" if "base" in outs else specs[0]
        times = {s: [] for s in runners}
        for k in range(a.rounds):
            for s, r in runners.items():
                times[s].append(time_runner(r, flush, a.warmup, a.iters))
        line = []
        for s in runners:
            ts = np.concatenate(times[s])
            st = stats(ts)
            same = torch.equal(outs[s], outs[ref_key])
            summary[f"{b}x{m}:{s}"] = dict(us=st["kernel_ms"] * 1e3, med=st["median_ms"] * 1e3, p10=st["p10_ms"] * 1e3,
                                          p90=st["p90_ms"] * 1e3, bitwise_eq_base=same)
            line.append(f"{s}={st['kernel_ms'] * 1e3:.2f}us({'=' if same else 'DIFF'})")
        log(f"B={b} m={m}: " + "  ".join(line))
    print(json.dumps({"explore": summary}), flush=True)


def mode_timeline(a):
    base = extract_base_tree(a.base_commit)
    kern = Kernel(make_variant("timing", None, base))
    configs = parse_configs(a.configs)
    inp = build_inputs(max(b for b, _ in configs))
    flush = Flusher(a.flush, a.flush_mb)
    off = inp["scratch_off"]
    raw = inp["scratch_raw"]
    out = {}
    for (b, m) in configs:
        r = Runner(kern, inp, b, m)
        nseq = NB_K_HEADS * b
        recs = []
        for it in range(a.iters):
            flush()
            r.graph.replay()
            torch.cuda.synchronize()
            cta = raw[off + DBG_CTA_OFF:off + DBG_CTA_OFF + 16 * nseq * m].view(torch.int64).view(nseq, m, 2).cpu()
            mg = raw[off + DBG_MERGE_OFF:off + DBG_MERGE_OFF + 64 * nseq].view(torch.int64).view(nseq, 8).cpu()
            t0 = int(cta[:, :, 0].min())
            start = (cta[:, :, 0] - t0).double() / 1e3
            end = (cta[:, :, 1] - t0).double() / 1e3
            mend = (mg[:, 3] - t0).double() / 1e3
            recs.append(dict(start_max=float(start.max()), cta_dur_med=float((end - start).median()),
                             cta_dur_max=float((end - start).max()), last_arrival_max=float(end.max(dim=1).values.max()),
                             last_minus_median_arrival=float((end.max(dim=1).values - end.median(dim=1).values).max()),
                             merge_end_max=float(mend.max()), rowmax_cyc=float(mg[:, 0].double().mean()),
                             loop_cyc=float(mg[:, 1].double().mean()), tail_cyc=float(mg[:, 2].double().mean())))
        keys = recs[0].keys()
        med = {k: float(np.median([x[k] for x in recs])) for k in keys}
        out[f"{b}x{m}"] = med
        log(f"B={b} m={m}: " + " ".join(f"{k}={v:.2f}" for k, v in med.items()))
    print(json.dumps({"timeline_us_or_cycles": out}), flush=True)


def mode_check(a):
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    base = extract_base_tree(a.base_commit)
    cand = Kernel(compile_xqa(tree))
    orig = Kernel(compile_xqa(base))
    log(f"candidate: smem {cand.smem} regs {cand.regs} local {cand.lmem}; original: smem {orig.smem} regs {orig.regs} "
        f"local {orig.lmem}")
    configs = parse_configs(a.configs)
    inp = build_inputs(max(b for b, _ in configs))
    flush = Flusher(a.flush, a.flush_mb)
    h0 = readonly_hash(inp)
    n = a.warmup + a.iters
    report = {"configs": {}}
    passed = True
    for (b, m) in configs:
        ref = reference(inp, b)
        rec = {}
        seq_outs = {}
        for name, kern in (("original", orig), ("candidate", cand)):
            r = Runner(kern, inp, b, m)
            outs = torch.empty(n, b, NB_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            sems = torch.empty(n, NB_SEM, dtype=torch.int32, device="cuda")
            inp["out"].fill_(float("nan"))
            for i in range(n):  # same warmup/iteration count and flush points as the timed loop
                flush()
                r.graph.replay()
                outs[i].copy_(r.out())
                sems[i].copy_(inp["sem"])
            torch.cuda.synchronize()
            first = outs[0]
            rep_eq = bool((outs.view(n, -1) == first.view(1, -1)).all().item())
            per_inv_err = (outs.float() - ref.unsqueeze(0)).abs().amax(dim=(1, 2, 3))
            sem_zero = bool((sems == 0).all().item())
            mx, mn, fin = errors(first, ref)
            seq_outs[name] = outs
            rec[name] = dict(max_abs_err=mx, mean_abs_err=mn, finite=fin, repeat_bitwise_equal=rep_eq,
                             per_invocation_max_err_max=float(per_inv_err.max().item()),
                             per_invocation_max_err_min=float(per_inv_err.min().item()), semaphores_reset=sem_zero,
                             sha256=sha(first))
        cand_vs_orig = bool(torch.equal(seq_outs["candidate"], seq_outs["original"]))
        c, o = rec["candidate"], rec["original"]
        ok = (c["finite"] and c["repeat_bitwise_equal"] and c["semaphores_reset"] and cand_vs_orig
              and c["max_abs_err"] <= 1.5 * o["max_abs_err"] + 1e-12
              and c["mean_abs_err"] <= 1.5 * o["mean_abs_err"] + 1e-12)
        rec["candidate_bitwise_equal_original_all_invocations"] = cand_vs_orig
        rec["passed"] = ok
        passed = passed and ok
        report["configs"][f"B{b}_m{m}"] = rec
        log(f"B={b} m={m}: cand err max {c['max_abs_err']:.3e} mean {c['mean_abs_err']:.3e} | orig err max "
            f"{o['max_abs_err']:.3e} mean {o['mean_abs_err']:.3e} | repeat-equal {c['repeat_bitwise_equal']} | "
            f"cand==orig (all {n} calls) {cand_vs_orig} | semaphores reset {c['semaphores_reset']} -> "
            f"{'PASS' if ok else 'FAIL'}")
    h1 = readonly_hash(inp)
    report["readonly_inputs_unchanged"] = (h0 == h1)
    passed = passed and (h0 == h1)
    report["passed"] = passed
    report["compared_against"] = (f"unmodified original mha.cu from {a.base_commit} (bitwise, every invocation) and an "
                                  "independent FP32 PyTorch attention over the dequantized FP8 KV (max/mean abs err "
                                  "<= 1.5x original)")
    report["sequence"] = dict(warmup=a.warmup, iters=a.iters, flush=a.flush)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bench", "check", "explore", "timeline"], default="bench")
    p.add_argument("--batches", default="1,2,4")
    p.add_argument("--nsub", type=int, default=DEFAULT_SUBSEQ)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush", choices=["read", "write", "none"], default="read")
    p.add_argument("--flush-mb", type=int, default=384)
    p.add_argument("--base-commit", default=BASE_COMMIT)
    p.add_argument("--variants", default="base,tree")
    p.add_argument("--configs", default="1x48,2x48,4x48")
    a = p.parse_args()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    ck(cu.cuInit(0), "cuInit")
    log(f"device {torch.cuda.get_device_name()} sms {torch.cuda.get_device_properties(0).multi_processor_count} "
        f"cwd {os.getcwd()}")
    {"bench": mode_bench, "check": mode_check, "explore": mode_explore, "timeline": mode_timeline}[a.mode](a)


if __name__ == "__main__":
    main()
