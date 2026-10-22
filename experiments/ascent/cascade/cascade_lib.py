#!/usr/bin/env python3
# Shared harness pieces for idea 20 (shared-prefix cascade decode): NVRTC build of cpp/kernels/xqa/mha.cu exactly as
# the XQA JIT does (via experiments/ascent/xqa_merge/xqa_evidence.py), kernel_mha + kernel_mha_prefix launchers that
# mirror DecoderXQARunner::runImpl, synthetic paged FP8 KV inputs with configurable sharing, and an FP32 reference.
import ctypes
import hashlib
import importlib.util
import math
import os
import subprocess
import tempfile

os.environ.setdefault("XQA_KV_DTYPE", "fp8")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("xe", os.path.join(HERE, "..", "xqa_merge", "xqa_evidence.py"))
xe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xe)
log = xe.log
cu = xe.cu
ck = xe.ck

MAIN_COMMIT = "d0d87a23f92239be20a7915b6cdff8470baac721"
NB_K, GRP, HD, TPP = 4, 6, 256, 32
NB_Q = NB_K * GRP
MAX_SEQ_LEN = 36864
MAX_PAGES = MAX_SEQ_LEN // TPP
KV_SCALE = 0.5
Q_SCALE = 1.0
MAX_SUB_SEQ = 6000  # getXqaMaxNumSubSeq
CASCADE_INFO_BYTES = 4096


def init_cuda():
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    ck(cu.cuInit(0), "cuInit")
    log(f"device {torch.cuda.get_device_name()}")


def extract_tree(commit):
    d = tempfile.mkdtemp(prefix="xqa_main_")
    tar = subprocess.run(["git", "archive", commit, "cpp/kernels/xqa"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
    return os.path.join(d, "cpp/kernels/xqa")


def nb_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def main_nb_sub_seq(b):
    """decoderXQARunner one-wave cap for HMMA SM120 head 256 at batch >= 2 (heuristic is 48 at max_seq_len 36864)."""
    return max(1, min(48, nb_sms() // (b * NB_K)))


def cascade_grid(b, m):
    """Prefix-kernel grid {C, G, nbKV} as the host launcher computes it."""
    g = math.ceil(b * GRP / 64)
    c = max(1, min(63, nb_sms() // (g * NB_K)))
    c = min(c, MAX_SUB_SEQ // (b * NB_K) - m)
    return c, g


class Kernels:

    def __init__(self, src_dir):
        cubin, self.src_sha = xe.compile_xqa(src_dir)
        self._img = np.frombuffer(cubin, dtype=np.uint8).copy()
        self.mod = ck(cu.cuModuleLoadData(self._img.ctypes.data), "cuModuleLoadData")
        A = cu.CUfunction_attribute
        self.mha = ck(cu.cuModuleGetFunction(self.mod, b"kernel_mha"), "get kernel_mha")
        self.smem = self._global_u32(b"smemSize")
        ck(cu.cuFuncSetAttribute(self.mha, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem), "attr")
        self.regs = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, self.mha), "regs")
        err, fn = cu.cuModuleGetFunction(self.mod, b"kernel_mha_prefix")
        self.prefix = fn if int(err) == 0 else None
        self.has_cascade_args = self.prefix is not None
        if self.prefix is not None:
            self.psmem = self._global_u32(b"prefixSmemSize")
            err = cu.cuModuleGetGlobal(self.mod, b"prefixCtaSize")[0]
            self.pcta = self._global_u32(b"prefixCtaSize") if int(err) == 0 else 128
            ck(cu.cuFuncSetAttribute(self.prefix, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.psmem),
               "attr prefix")
            self.pregs = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, self.prefix), "pregs")
            self.plmem = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.prefix), "plmem")
            log(f"kernel_mha_prefix: smem {self.psmem} B, regs {self.pregs}, local {self.plmem} B")
        log(f"kernel_mha: smem {self.smem} B, regs {self.regs}")

    def _global_u32(self, name):
        dptr, _ = ck(cu.cuModuleGetGlobal(self.mod, name), f"global {name}")
        v = ctypes.c_uint32(0)
        ck(cu.cuMemcpyDtoH(ctypes.addressof(v), dptr, 4), "dtoh")
        return int(v.value)

    @staticmethod
    def _launch(func, grid, block, smem, args):
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        ck(cu.cuLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, stream, args.ptr, 0),
           "cuLaunchKernel")


class ParamPack:

    def __init__(self, objs):
        self.objs = objs
        self.arr = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
        self.ptr = ctypes.addressof(self.arr)


def kv_list(inp):
    return xe.KVCacheList(inp["pool"].data_ptr(), inp["pages"].data_ptr(), inp["seqlen"].data_ptr(), MAX_PAGES, 0)


class Call:
    """One decode attention call: optional kernel_mha_prefix, then kernel_mha (grid {m, nbKV, B})."""

    def __init__(self, kern, inp, b, m, cascade):
        self.kern, self.inp, self.b, self.m = kern, inp, b, m
        self.cascade = cascade and kern.prefix is not None
        base = inp["scratch_ptr"]
        if self.cascade:
            self.c, self.g = cascade_grid(b, m)
            info, scratch = base, base + CASCADE_INFO_BYTES
        else:
            self.c, self.g = 0, 0
            info, scratch = 0, base
        objs = [ctypes.c_uint32(NB_K), ctypes.c_uint32(MAX_SEQ_LEN), ctypes.c_float(Q_SCALE),
                ctypes.c_uint64(inp["out"].data_ptr()), ctypes.c_uint64(inp["q"].data_ptr()), ctypes.c_uint64(0),
                kv_list(inp), ctypes.c_uint32(b), ctypes.c_uint64(inp["kvscale"].data_ptr()),
                ctypes.c_uint64(inp["sem"].data_ptr()), ctypes.c_uint64(scratch)]
        if kern.has_cascade_args:
            objs += [ctypes.c_uint64(info), ctypes.c_uint32(self.c)]
        self.mha_args = ParamPack(objs)
        if self.cascade:
            self.pfx_args = ParamPack([
                ctypes.c_uint32(NB_K), ctypes.c_float(Q_SCALE), ctypes.c_uint64(inp["q"].data_ptr()), kv_list(inp),
                ctypes.c_uint32(b), ctypes.c_uint64(inp["kvscale"].data_ptr()), ctypes.c_uint32(m),
                ctypes.c_uint64(info), ctypes.c_uint64(scratch)])

    def launch(self):
        if self.cascade:
            Kernels._launch(self.kern.prefix, (self.c, self.g, NB_K), (self.kern.pcta, 1, 1), self.kern.psmem,
                            self.pfx_args)
        Kernels._launch(self.kern.mha, (self.m, NB_K, self.b), (128, 1, 2), self.kern.smem, self.mha_args)

    def capture(self):
        self.launch()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.launch()
        torch.cuda.synchronize()
        return self

    def out(self):
        return self.inp["out"][:self.b]


def build_inputs(bmax, shared_pages=1024, seqlens=None, seed=1234, groups=1, uniq_pages=None):
    """Paged e4m3 KV pool. Request b's page list: shared_pages pages common to its group (groups > 1 gives disjoint
    prefixes), then its own pages. seqlens: per-request token counts (default 34944)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if seqlens is None:
        seqlens = [32768 + 2048 + 128] * bmax
    nbp = [math.ceil(s / TPP) for s in seqlens]
    if uniq_pages is None:
        uniq_pages = max(nbp) - shared_pages
    nblocks = groups * shared_pages + bmax * uniq_pages
    pool = (torch.randn(nblocks * 2, NB_K, TPP, HD, generator=g, device="cuda") * 2.0).to(torch.float8_e4m3fn)
    pages = torch.zeros(bmax, 1, 2, MAX_PAGES, dtype=torch.int32)
    for b in range(bmax):
        grp = b % groups
        blk = [grp * shared_pages + j for j in range(shared_pages)]
        blk += [groups * shared_pages + b * uniq_pages + j for j in range(nbp[b] - shared_pages)]
        blk = torch.tensor(blk, dtype=torch.int32)
        pages[b, 0, 0, :nbp[b]] = 2 * blk
        pages[b, 0, 1, :nbp[b]] = 2 * blk + 1
    scratch_raw = torch.zeros(96 << 20, dtype=torch.uint8, device="cuda")
    return dict(pool=pool, pages=pages.cuda(), nbp=nbp, seqlens=list(seqlens),
                q=torch.randn(bmax, NB_Q, HD, generator=g, device="cuda").to(torch.bfloat16),
                seqlen=torch.tensor(seqlens, dtype=torch.int32, device="cuda"),
                kvscale=torch.tensor([KV_SCALE], dtype=torch.float32, device="cuda"),
                out=torch.zeros(bmax, NB_Q, HD, dtype=torch.bfloat16, device="cuda"),
                sem=torch.zeros(1024, dtype=torch.int32, device="cuda"), scratch_raw=scratch_raw,
                scratch_ptr=(scratch_raw.data_ptr() + 4095) // 4096 * 4096)


def reference(inp, b):
    """Independent FP32 attention over the stored e4m3 KV (dequantized with its scale), rounded once to BF16."""
    torch.backends.cuda.matmul.allow_tf32 = False
    outs = []
    pages = inp["pages"]
    for r in range(b):
        sl = int(inp["seqlen"][r].item())
        nbp = math.ceil(sl / TPP)
        kv = []
        for kvi in (0, 1):
            pg = pages[r, 0, kvi, :nbp].long()
            kv.append(inp["pool"][pg].float().permute(1, 0, 2, 3).reshape(NB_K, nbp * TPP, HD)[:, :sl] * KV_SCALE)
        q = inp["q"][r].float().view(NB_K, GRP, HD)
        p = torch.softmax(torch.einsum("hgd,hsd->hgs", q, kv[0]) * (Q_SCALE / math.sqrt(HD)), dim=-1)
        outs.append(torch.einsum("hgs,hsd->hgd", p, kv[1]).reshape(NB_Q, HD))
    return torch.stack(outs).to(torch.bfloat16).float()


def tbytes(t):
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def readonly_hash(inp):
    h = hashlib.sha256()
    for k in ("pool", "pages", "q", "seqlen", "kvscale"):
        h.update(tbytes(inp[k]))
    return h.hexdigest()


def stats(ts):
    return xe.stats(np.asarray(ts))
