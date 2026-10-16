#!/usr/bin/env python3
# Self-contained evidence harness for XQA kernel_mha (SM120 HMMA, NVRTC-JIT path) at the Qwen3.8-27B-NVFP4 decode shape
# (env XQA_KV_DTYPE / XQA_KV_HEADS / XQA_HEAD_GRP select another shape, e.g. the earlier Qwen3.6 FP8 / 2 / 8).
# No local imports, so it can run inline (python -c) in any source tree; it compiles <cwd>/cpp/kernels/xqa/mha.cu.
#
# Kernel compiled exactly like decoderXQAImplJIT/nvrtcWrapper (same NVRTC options and macros: BF16 Q/output, BF16
# paged KV cache by default, head 256, 6 Q heads per KV head, 32 tokens/block, beam 1), launched like
# DecoderXQARunner::runImpl (grid {nbSubSeq, nbKHeads, batch}, block {128,1,2}, dynamic smem = smemSize, no PDL).
# Inputs: synthetic, fixed seed, serving shapes: 24 Q / 4 KV heads, 32768-token shared prefix (block reuse) + 2048
# unique + 128 decoded tokens per request, max_seq_len 36864. Sub-sequence counts per batch are given explicitly
# (default 48 at B=1/2/4: the graph-captured count of the base heuristic; 47 is also exercised).
#
#   bench : CUDA-graph replay timed with CUDA events; before every call (outside the timed region) the output is
#           NaN-poisoned and the L2 is flushed with a 384 MiB read sweep. kernel_ms = 10-90% trimmed mean (event
#           timestamps are quantized to ~1 us on this GPU); median/p10/p90 reported too.
#   check : (1) per config, the same call sequence as the timed loop (rounds x (warmup + iters), poison + flush before
#           every call) for the unmodified original (git base commit) and the candidate: every call bitwise vs the
#           first call and vs the original's same call, semaphores back to 0 after every call, outputs finite;
#           (2) one process, fixed-seed interleaving of counts {48 at B=1,2,3,4,8; 47,31,23,11 at B=2,3,4,8} covering
#           every ordered transition, shared semaphore/scratch buffers, same checks; (3) FP32 PyTorch reference of the
#           same math from the stored FP8 KV and scale: candidate max/mean abs error <= 1.5x the original's;
#           (4) read-only inputs hashed before/after.
#   verify: the committed mha.cu equals the tested in-memory patch (experiments/ascent/xqa_merge/xqa_newmerge.py).
import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import subprocess
import sys
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
# Shape (env-overridable): default = this run's Qwen3.8-27B-NVFP4 full-attention decode, 24 Q / 4 KV heads, BF16 KV.
# XQA_KV_DTYPE=fp8 XQA_KV_HEADS=2 XQA_HEAD_GRP=8 reproduces the earlier Qwen3.6-35B-A3B shape.
KV_DTYPE = os.environ.get("XQA_KV_DTYPE", "bf16")
assert KV_DTYPE in ("bf16", "fp8"), KV_DTYPE
NB_K_HEADS = int(os.environ.get("XQA_KV_HEADS", "4"))
HEAD_GRP = int(os.environ.get("XQA_HEAD_GRP", "6"))
HEAD_DIM, TOKENS_PER_PAGE = 256, 32
NB_Q_HEADS = NB_K_HEADS * HEAD_GRP
MAX_SEQ_LEN = 36864
MAX_PAGES = MAX_SEQ_LEN // TOKENS_PER_PAGE
PREFIX = 32768
SEQ_LEN = 32768 + 2048 + 128
Q_SCALE = 1.0
KV_SCALE = 0.5 if KV_DTYPE == "fp8" else 1.0  # the kernel reads the scale only for a quantized cache
NB_SEM = 1024
MACROS = {
    "INPUT_FP16": "0", "DTYPE": "__nv_bfloat16", "GENERATE_CUBIN": "1", "NDEBUG": "1", "HEAD_ELEMS": "256",
    "BEAM_WIDTH": "1", "CACHE_ELEM_ENUM": "2" if KV_DTYPE == "fp8" else "0", "TOKENS_PER_PAGE": "32",
    "HEAD_GRP_SIZE": str(HEAD_GRP), "M_TILESIZE": str(HEAD_GRP),
    "USE_CUSTOM_BARRIER": "1", "SLIDING_WINDOW": "1", "LOW_PREC_OUTPUT": "0", "USE_INPUT_KV": "0", "ROPE_STYLE": "0",
    "ROPE_ELEMS": "64", "IS_SPEC_DEC_TREE": "0", "SKIP_SOFTMAX_ATTN": "0",
    "__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__": "1", "__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__": "1"
}


def log(*a):
    print(*a, flush=True)


def ck(res, what):
    err, rest = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return None if not rest else (rest[0] if len(rest) == 1 else rest)


def compile_xqa(src_dir):
    t0 = time.time()
    src = open(os.path.join(src_dir, "mha.cu")).read()
    names = [h.encode() for h in XQA_HEADERS + CUDA_HEADERS]
    contents = [open(os.path.join(src_dir, h), "rb").read() for h in XQA_HEADERS]
    contents += [open(os.path.join(CUDA_INC, h), "rb").read() for h in CUDA_HEADERS]
    prog = ck(nvrtc.nvrtcCreateProgram(src.encode(), b"default_program", len(names), contents, names), "create")
    opts = [b"-dw", b"--use_fast_math", b"-default-device", b"-arch=sm_120f"]
    opts += [f"-D{k}={v}".encode() for k, v in MACROS.items()]
    if int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) != 0:
        size = ck(nvrtc.nvrtcGetProgramLogSize(prog), "logsize")
        buf = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("NVRTC compile failed:\n" + buf.decode(errors="replace")[-6000:])
    size = ck(nvrtc.nvrtcGetCUBINSize(prog), "cubinsize")
    cubin = b" " * size
    ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "getcubin")
    nvrtc.nvrtcDestroyProgram(prog)
    sha = hashlib.sha256(src.encode()).hexdigest()
    log(f"compiled {src_dir}/mha.cu (sha256 {sha[:16]}) in {time.time() - t0:.1f}s, cubin {size} B")
    return cubin, sha


class KVCacheList(ctypes.Structure):
    _fields_ = [("pool", ctypes.c_uint64), ("pageList", ctypes.c_uint64), ("seqLen", ctypes.c_uint64),
                ("maxNbPagesPerSeq", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


class Kernel:

    def __init__(self, src_dir):
        cubin, self.src_sha = compile_xqa(src_dir)
        self._img = np.frombuffer(cubin, dtype=np.uint8).copy()
        self.mod = ck(cu.cuModuleLoadData(self._img.ctypes.data), "cuModuleLoadData")
        self.func = ck(cu.cuModuleGetFunction(self.mod, b"kernel_mha"), "cuModuleGetFunction")
        dptr, _ = ck(cu.cuModuleGetGlobal(self.mod, b"smemSize"), "cuModuleGetGlobal")
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
            ctypes.c_uint32(NB_K_HEADS), ctypes.c_uint32(MAX_SEQ_LEN), ctypes.c_float(Q_SCALE),
            ctypes.c_uint64(inp["out"].data_ptr()), ctypes.c_uint64(inp["q"].data_ptr()), ctypes.c_uint64(0),
            KVCacheList(inp["pool"].data_ptr(), inp["pages"].data_ptr(), inp["seqlen"].data_ptr(), MAX_PAGES, 0),
            ctypes.c_uint32(batch), ctypes.c_uint64(inp["kvscale"].data_ptr()), ctypes.c_uint64(inp["sem"].data_ptr()),
            ctypes.c_uint64(inp["scratch_ptr"])
        ]
        self.arr = (ctypes.c_void_p * len(self.objs))(*[ctypes.addressof(o) for o in self.objs])
        self.ptr = ctypes.addressof(self.arr)


def build_inputs(bmax, seed=1234):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    nbp = math.ceil(SEQ_LEN / TOKENS_PER_PAGE)
    prefix_blocks = PREFIX // TOKENS_PER_PAGE
    uniq = nbp - prefix_blocks
    pool = (torch.randn((prefix_blocks + bmax * uniq) * 2, NB_K_HEADS, TOKENS_PER_PAGE, HEAD_DIM, generator=g,
                        device="cuda") * 2.0).to(torch.float8_e4m3fn if KV_DTYPE == "fp8" else torch.bfloat16)
    pages = torch.zeros(bmax, 1, 2, MAX_PAGES, dtype=torch.int32)
    for b in range(bmax):
        blk = torch.tensor(list(range(prefix_blocks)) + [prefix_blocks + b * uniq + j for j in range(uniq)],
                           dtype=torch.int32)
        pages[b, 0, 0, :nbp] = 2 * blk
        pages[b, 0, 1, :nbp] = 2 * blk + 1
    scratch_raw = torch.zeros(64 << 20, dtype=torch.uint8, device="cuda")
    return dict(pool=pool, pages=pages.cuda(), nbp=nbp,
                q=torch.randn(bmax, NB_Q_HEADS, HEAD_DIM, generator=g, device="cuda").to(torch.bfloat16),
                seqlen=torch.full((bmax, ), SEQ_LEN, dtype=torch.int32, device="cuda"),
                kvscale=torch.tensor([KV_SCALE], dtype=torch.float32, device="cuda"),
                out=torch.zeros(bmax, NB_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
                sem=torch.zeros(NB_SEM, dtype=torch.int32, device="cuda"), scratch_raw=scratch_raw,
                scratch_ptr=(scratch_raw.data_ptr() + 4095) // 4096 * 4096)


def tbytes(t):
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def readonly_hash(inp):
    h = hashlib.sha256()
    for k in ("pool", "pages", "q", "seqlen", "kvscale"):
        h.update(tbytes(inp[k]))
    return h.hexdigest()


def reference(inp, batch):
    """Independent FP32 attention over the stored KV (FP8 dequantized with its scale, or BF16), rounded once to BF16."""
    torch.backends.cuda.matmul.allow_tf32 = False
    nbp, outs = inp["nbp"], []
    for b in range(batch):
        kv = []
        for kvi in (0, 1):
            pg = inp["pages"][b, 0, kvi, :nbp].long()
            kv.append(inp["pool"][pg].float().permute(1, 0, 2, 3).reshape(NB_K_HEADS, nbp * TOKENS_PER_PAGE,
                                                                          HEAD_DIM)[:, :SEQ_LEN] * KV_SCALE)
        q = inp["q"][b].float().view(NB_K_HEADS, HEAD_GRP, HEAD_DIM)
        p = torch.softmax(torch.einsum("hgd,hsd->hgs", q, kv[0]) * (Q_SCALE / math.sqrt(HEAD_DIM)), dim=-1)
        outs.append(torch.einsum("hgs,hsd->hgd", p, kv[1]).reshape(NB_Q_HEADS, HEAD_DIM))
    return torch.stack(outs).to(torch.bfloat16).float()


class Flusher:

    def __init__(self, mb):
        self.buf = torch.ones((mb << 20) // 4, dtype=torch.int32, device="cuda")

    def __call__(self):
        self.buf.sum()  # read sweep: evicts the KV from L2 and leaves clean lines, like the weight reads in serving


class Runner:

    def __init__(self, kern, inp, batch, nsub):
        self.inp, self.batch = inp, batch
        self.args = Args(inp, batch)
        self.grid = (nsub, NB_K_HEADS, batch)
        kern.launch(self.args, self.grid)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            kern.launch(self.args, self.grid)
        torch.cuda.synchronize()

    def call(self, flush):
        self.inp["out"].fill_(float("nan"))  # poison: a merge that never runs cannot pass on a stale output
        flush()
        self.graph.replay()

    def out(self):
        return self.inp["out"][:self.batch]


def parse_configs(s):
    return [tuple(int(x) for x in c.split("x")) for c in s.split(",") if c]


def stats(ts):
    med = float(np.median(ts))
    p10, p90 = float(np.percentile(ts, 10)), float(np.percentile(ts, 90))
    core = ts[(ts >= p10) & (ts <= p90)]
    tmean = float(core.mean()) if core.size else med
    return dict(kernel_ms=tmean, median_ms=med, p10_ms=p10, p90_ms=p90, mean_ms=float(ts.mean()),
                noise_floor=(p90 - p10) / tmean, samples=int(ts.size))


def time_config(kern, inp, flush, b, m, a):
    r = Runner(kern, inp, b, m)
    all_ts = []
    for k in range(a.rounds):
        n = a.warmup + a.iters
        st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            inp["out"].fill_(float("nan"))
            flush()
            st[i].record()
            r.graph.replay()
            en[i].record()
        torch.cuda.synchronize()
        ts = np.array([st[i].elapsed_time(en[i]) for i in range(a.warmup, n)])
        log(f"  B={b} m={m} round {k}: trimmed mean {stats(ts)['kernel_ms'] * 1e3:.2f} us")
        all_ts.append(ts)
    st = stats(np.concatenate(all_ts))
    st["nb_sub_seq"] = m
    st["output_sha256"] = hashlib.sha256(tbytes(r.out())).hexdigest()
    st["output_finite"] = bool(torch.isfinite(r.out().float()).all().item())
    return st


def mode_bench(a):
    kern = Kernel(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    log(f"kernel_mha: smem {kern.smem} B, regs {kern.regs}, local {kern.lmem} B")
    prim, extra = parse_configs(a.configs), parse_configs(a.extra_configs)
    inp = build_inputs(max(b for b, _ in prim + extra))
    flush = Flusher(a.flush_mb)
    res = {"batch": {}, "output": {"class": "exact"}, "extra": {}}
    for b, m in prim:
        res["batch"][str(b)] = time_config(kern, inp, flush, b, m, a)
        s = res["batch"][str(b)]
        log(f"B={b} m={m}: {s['kernel_ms'] * 1e3:.2f} us (median {s['median_ms'] * 1e3:.2f}, noise {s['noise_floor']:.3f})")
    for b, m in extra:
        res["extra"][f"{b}x{m}"] = time_config(kern, inp, flush, b, m, a)
        s = res["extra"][f"{b}x{m}"]
        log(f"extra B={b} m={m}: {s['kernel_ms'] * 1e3:.2f} us (median {s['median_ms'] * 1e3:.2f})")
    res["config"] = dict(source_sha256=kern.src_sha, smem=kern.smem, regs=kern.regs, local_bytes=kern.lmem,
                         seq_len=SEQ_LEN, shared_prefix=PREFIX, warmup=a.warmup, iters=a.iters, rounds=a.rounds,
                         flush=f"{a.flush_mb} MiB read sweep before every call, output NaN-poisoned before every call",
                         timing="CUDA graph replay + CUDA events; kernel_ms = 10-90% trimmed mean", pdl="off in both arms")
    print(json.dumps(res), flush=True)


INTERLEAVE = [(1, 48), (2, 48), (3, 48), (4, 48), (8, 48), (1, 47), (2, 47), (4, 47), (1, 94)]


def run_sequence(kern, inp, flush, seq):
    """Runs a list of (b, m) calls; returns per-call outputs (cpu-free GPU copies) and semaphore-zero flags."""
    runners = {}
    outs, semok = [], []
    for (b, m) in seq:
        if (b, m) not in runners:
            runners[(b, m)] = Runner(kern, inp, b, m)
        r = runners[(b, m)]
        r.call(flush)
        outs.append(r.out().clone())
        semok.append((inp["sem"] == 0).all())
    torch.cuda.synchronize()
    return outs, [bool(x.item()) for x in semok]


def mode_check(a):
    base = extract_base_tree(a.base_commit)
    orig = Kernel(base)
    cand = Kernel(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    log(f"candidate smem {cand.smem} regs {cand.regs} local {cand.lmem}; original smem {orig.smem} regs {orig.regs}")
    prim = parse_configs(a.configs)
    inp = build_inputs(8)
    flush = Flusher(a.flush_mb)
    h0 = readonly_hash(inp)
    report, passed = {"repeated": {}, "interleaved": {}, "reference": {}}, True
    n_calls = a.rounds * (a.warmup + a.iters)
    first = {}
    # (1) repeated sequence per config, mirroring the timed loop
    for (b, m) in prim:
        res = {}
        for name, kern in (("original", orig), ("candidate", cand)):
            outs, semok = run_sequence(kern, inp, flush, [(b, m)] * n_calls)
            stack = torch.stack(outs)
            res[name] = dict(stack=stack, rep=bool((stack == stack[0:1]).all().item()), sem=all(semok),
                             finite=bool(torch.isfinite(stack.float()).all().item()))
            first[(name, b, m)] = stack[0]
        eq = bool(torch.equal(res["candidate"]["stack"], res["original"]["stack"]))
        ok = all(res[k]["rep"] and res[k]["sem"] and res[k]["finite"] for k in res) and eq
        report["repeated"][f"B{b}_m{m}"] = dict(calls=n_calls, candidate_repeat_bitwise=res["candidate"]["rep"],
                                                original_repeat_bitwise=res["original"]["rep"],
                                                candidate_semaphores_zero_after_every_call=res["candidate"]["sem"],
                                                candidate_finite=res["candidate"]["finite"],
                                                candidate_equals_original_every_call=eq, passed=ok)
        passed &= ok
        log(f"repeated B={b} m={m}: {n_calls} calls, cand repeat-bitwise {res['candidate']['rep']}, semaphores 0 "
            f"{res['candidate']['sem']}, finite {res['candidate']['finite']}, cand==orig every call {eq} -> "
            f"{'PASS' if ok else 'FAIL'}")
        del res
    # (2) interleaved counts covering every ordered transition, one process, shared semaphores/scratch
    pairs = [(i, j) for i in range(len(INTERLEAVE)) for j in range(len(INTERLEAVE)) if i != j]
    random.Random(20261014).shuffle(pairs)
    seq = [INTERLEAVE[k] for p in pairs for k in p]
    trans = {(seq[t], seq[t + 1]) for t in range(len(seq) - 1)}
    covered = all((INTERLEAVE[i], INTERLEAVE[j]) in trans for i, j in pairs)
    inter = {}
    for name, kern in (("original", orig), ("candidate", cand)):
        outs, semok = run_sequence(kern, inp, flush, seq)
        firsts, same = {}, True
        for c, o in zip(seq, outs):
            if c not in firsts:
                firsts[c] = o
            same &= bool(torch.equal(o, firsts[c]))
        inter[name] = dict(outs=outs, firsts=firsts, same=same, sem=all(semok),
                           finite=all(bool(torch.isfinite(o.float()).all().item()) for o in outs))
    eq = all(torch.equal(x, y) for x, y in zip(inter["candidate"]["outs"], inter["original"]["outs"]))
    ok = covered and eq and inter["candidate"]["same"] and inter["candidate"]["sem"] and inter["candidate"]["finite"]
    report["interleaved"] = dict(calls=len(seq), configs=[f"{b}x{m}" for b, m in INTERLEAVE],
                                 every_ordered_transition_covered=covered,
                                 candidate_each_call_equals_first_of_its_count=inter["candidate"]["same"],
                                 candidate_semaphores_zero_after_every_call=inter["candidate"]["sem"],
                                 candidate_finite=inter["candidate"]["finite"],
                                 candidate_equals_original_every_call=eq, passed=ok)
    passed &= ok
    log(f"interleaved: {len(seq)} calls over {len(INTERLEAVE)} counts, transitions covered {covered}, cand==first "
        f"{inter['candidate']['same']}, semaphores 0 {inter['candidate']['sem']}, cand==orig {eq} -> "
        f"{'PASS' if ok else 'FAIL'}")
    # (3) FP32 reference
    for (b, m) in sorted(set(prim) | set(INTERLEAVE)):
        ref = reference(inp, b)
        errs = {}
        for name in ("original", "candidate"):
            o = first.get((name, b, m), inter[name]["firsts"].get((b, m)))
            d = (o.float() - ref).abs()
            errs[name] = (float(d.max().item()), float(d.mean().item()))
        ok = errs["candidate"][0] <= 1.5 * errs["original"][0] + 1e-12 and \
            errs["candidate"][1] <= 1.5 * errs["original"][1] + 1e-12
        report["reference"][f"B{b}_m{m}"] = dict(candidate_max_abs=errs["candidate"][0],
                                                 candidate_mean_abs=errs["candidate"][1],
                                                 original_max_abs=errs["original"][0],
                                                 original_mean_abs=errs["original"][1], passed=ok)
        passed &= ok
        log(f"reference B={b} m={m}: cand max {errs['candidate'][0]:.3e} mean {errs['candidate'][1]:.3e} | orig max "
            f"{errs['original'][0]:.3e} mean {errs['original'][1]:.3e} -> {'PASS' if ok else 'FAIL'}")
    h1 = readonly_hash(inp)
    report["readonly_inputs_unchanged"] = h0 == h1
    passed &= h0 == h1
    report["passed"] = bool(passed)
    report["compared_against"] = (f"unmodified original mha.cu at {a.base_commit} (bitwise, every call) and an "
                                  f"independent FP32 PyTorch attention over the stored {KV_DTYPE} KV (<=1.5x original "
                                  "max/mean abs error)")
    report["shape"] = dict(kv_dtype=KV_DTYPE, kv_heads=NB_K_HEADS, q_heads_per_kv=HEAD_GRP, head_dim=HEAD_DIM,
                           seq_len=SEQ_LEN, shared_prefix=PREFIX, tokens_per_page=TOKENS_PER_PAGE)
    report["candidate_source_sha256"] = cand.src_sha
    report["original_source_sha256"] = orig.src_sha
    print(json.dumps(report), flush=True)


def extract_base_tree(commit):
    d = tempfile.mkdtemp(prefix="xqa_base_")
    tar = subprocess.run(["git", "archive", commit, "cpp/kernels/xqa"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
    return os.path.join(d, "cpp/kernels/xqa")


def mode_verify(a):
    sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent/xqa_merge"))
    import difflib

    import xqa_newmerge as nm
    base = extract_base_tree(a.base_commit)
    want = nm.patch_final(open(os.path.join(base, "mha.cu")).read())
    have = open(os.path.join(os.getcwd(), "cpp/kernels/xqa/mha.cu")).read()
    same = want == have
    if not same:
        for line in list(difflib.unified_diff(want.splitlines(), have.splitlines(), "patch_final(base)", "tree",
                                              lineterm=""))[:300]:
            log(line)
    print(json.dumps({"tree_equals_tested_patch": same, "tree_sha256": hashlib.sha256(have.encode()).hexdigest(),
                      "passed": same}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bench", "check", "verify"], default="bench")
    p.add_argument("--configs", default="1x48,2x48,4x48")
    p.add_argument("--extra-configs", default="")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush-mb", type=int, default=384)
    p.add_argument("--base-commit", default=BASE_COMMIT)
    a = p.parse_args()
    if a.mode == "verify":
        return mode_verify(a)
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    ck(cu.cuInit(0), "cuInit")
    log(f"device {torch.cuda.get_device_name()} cwd {os.getcwd()}")
    {"bench": mode_bench, "check": mode_check}[a.mode](a)


if __name__ == "__main__":
    main()
