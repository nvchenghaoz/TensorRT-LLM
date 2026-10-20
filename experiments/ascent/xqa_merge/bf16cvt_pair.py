#!/usr/bin/env python3
# Branch 121 / idea 214: paired kernel bench for XQA kernel_mha, BF16 Q + FP8 e4m3 KV paired conversion
# (cpp/kernels/xqa/utils.cuh, macro XQA_BF16_E4M3_PAIRED_CVT). Free of braces, backslashes and double quotes so it
# can run inline (bash heredoc) in run_gpu_pair's base arm, which has no branch files.
# Usage: python bf16cvt_pair.py <order 0|1> <out.json>        one timing process
#        python bf16cvt_pair.py combine <r0.json> <r1.json>   pools both processes; prints the evidence JSON last
# A timing process NVRTC-compiles <cwd>/cpp/kernels/xqa/mha.cu like the JIT (-arch=sm_120f, the JIT macros, fp8 e4m3
# KV, kv scale 0.5, 24 Q / 4 KV heads, head 256, 34944 tokens, 32768-token prefix aliased across requests) and the
# same files at BASE_SHA (git archive). It loads the cwd cubin twice (own, own_aa = the A/A arm) and the base cubin
# once (ref), and times the three rotated per round at 1x47, 2x23, 3x15, 4x11 (order 1 reverses the config order and
# the rotation). Before every call: output NaN-poisoned and a 384 MiB L2 read flush (inputs are never written).
# CUDA-graph replay timed with CUDA events; ROUNDS x (WARMUP + ITERS) calls per arm per config; kernel_ms = 10-90%
# trimmed mean. Process 0 also runs the arch-macro probe, an exhaustive e4m3x2 -> bf16x2 test (paired cvt vs the
# generic per-element conversion), and the SASS census (total / F2FP / LOP3 / PRMT) of each build.
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

LB, RB, NL, DQ = chr(123), chr(125), chr(10), chr(34)
BASE_SHA = '07a287408d9a3a2313ae5f3ec22730f94f932b51'
CONFIGS = [(1, 47), (2, 23), (3, 15), (4, 11)]
PRIMARY = [(1, 47), (2, 23), (4, 11)]
WARMUP, ITERS, ROUNDS, FLUSH_MB = 10, 100, 6, 384
CUDA_INC = '/usr/local/cuda/include'
HDRS = ['cuda_hint.cuh', 'defines.h', 'ldgsts.cuh', 'mha.h', 'mhaUtils.cuh', 'mma.cuh', 'platform.h', 'utils.cuh',
        'utils.h', 'mha_stdheaders.cuh', 'mha_components.cuh', 'mla_sm120.cuh', 'gmma.cuh', 'gmma_impl.cuh',
        'barriers.cuh', 'tma.h', 'specDec.h']
CHDRS = ['cuda_bf16.h', 'cuda_bf16.hpp', 'cuda_fp16.h', 'cuda_fp16.hpp', 'cuda_fp8.h', 'cuda_fp8.hpp',
         'vector_types.h', 'vector_functions.h', 'device_types.h']
KVH, GRP, D, TPP = 4, 6, 256, 32
NQ = KVH * GRP
MAXLEN = 36864
MAXP = MAXLEN // TPP
PREFIX = 32768
SEQ = 32768 + 2048 + 128
KV_SCALE = 0.5
MACROS = dict(INPUT_FP16='0', DTYPE='__nv_bfloat16', GENERATE_CUBIN='1', NDEBUG='1', HEAD_ELEMS='256',
              BEAM_WIDTH='1', CACHE_ELEM_ENUM='2', TOKENS_PER_PAGE='32', HEAD_GRP_SIZE=str(GRP),
              M_TILESIZE=str(GRP), USE_CUSTOM_BARRIER='1', SLIDING_WINDOW='1', LOW_PREC_OUTPUT='0',
              USE_INPUT_KV='0', ROPE_STYLE='0', ROPE_ELEMS='64', IS_SPEC_DEC_TREE='0', SKIP_SOFTMAX_ATTN='0',
              __FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__='1', __FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__='1')
CUOBJDUMP = shutil.which('cuobjdump') or '/usr/local/cuda/bin/cuobjdump'
HEXCH = set('0123456789abcdef')


def log(*a):
    print(*a, flush=True)


def ck(res):
    err, rest = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
    if int(err) != 0:
        raise RuntimeError('CUDA/NVRTC error %s' % (err, ))
    return None if not rest else (rest[0] if len(rest) == 1 else rest)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run_text(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return 'unavailable: %s' % e


def gpu_state():
    return run_text(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active,'
                     'power.draw', '--format=csv,noheader'])


def extract_base_tree(commit):
    try:
        d = tempfile.mkdtemp(prefix='xqa_base_')
        tar = subprocess.run(['git', 'archive', commit, 'cpp/kernels/xqa'], check=True, capture_output=True,
                             timeout=120).stdout
        subprocess.run(['tar', '-x', '-C', d], input=tar, check=True)
        return os.path.join(d, 'cpp/kernels/xqa'), 'git archive ' + commit
    except Exception as e:  # noqa: BLE001
        log('git archive %s failed (%s); ref arm = cwd tree' % (commit, e))
        return os.path.join(os.getcwd(), 'cpp/kernels/xqa'), 'cwd (git archive failed)'


def tree_hashes(d):
    return dict(mha_cu=sha(open(os.path.join(d, 'mha.cu'), 'rb').read()),
                utils_cuh=sha(open(os.path.join(d, 'utils.cuh'), 'rb').read()))


def nvrtc_compile(src, name, headers, contents, opts):
    prog = ck(nvrtc.nvrtcCreateProgram(src.encode(), name, len(headers), contents, headers))
    if int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) != 0:
        size = ck(nvrtc.nvrtcGetProgramLogSize(prog))
        buf = b' ' * size
        nvrtc.nvrtcGetProgramLog(prog, buf)
        nvrtc.nvrtcDestroyProgram(prog)
        return None, buf.decode(errors='replace')
    size = ck(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b' ' * size
    ck(nvrtc.nvrtcGetCUBIN(prog, cubin))
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin, ''


def cuda_headers():
    return [h.encode() for h in CHDRS], [open(os.path.join(CUDA_INC, h), 'rb').read() for h in CHDRS]


def compile_mha(src_dir, arch='sm_120f', extra=()):
    t0 = time.time()
    src = open(os.path.join(src_dir, 'mha.cu')).read()
    cn, cc = cuda_headers()
    names = [h.encode() for h in HDRS] + cn
    cont = [open(os.path.join(src_dir, h), 'rb').read() for h in HDRS] + cc
    opts = [b'-dw', b'--use_fast_math', b'-default-device', ('-arch=' + arch).encode()]
    opts += [('-D%s=%s' % kv).encode() for kv in MACROS.items()]
    opts += [e.encode() for e in extra]
    cubin, err = nvrtc_compile(src, b'default_program', names, cont, opts)
    if cubin is None:
        raise RuntimeError('NVRTC compile failed (%s %s %s): %s' % (src_dir, arch, ' '.join(extra), err[-3000:]))
    log('compiled %s arch %s extra [%s] in %.1fs, cubin %d B, sha256 %s' %
        (src_dir, arch, ' '.join(extra), time.time() - t0, len(cubin), sha(cubin)[:16]))
    return cubin


def sass_insts(text):
    ins = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith('/*'):
            continue
        j = s.find('*/')
        addr = s[2:j]
        if j < 3 or not set(addr) <= HEXCH:
            continue
        rest = s[j + 2:]
        k = rest.find(';')
        if k < 0:
            continue
        ins.append(rest[:k].strip())
    return ins


def census(cubin, tag):
    d = tempfile.mkdtemp(prefix='bf16cvt_')
    p = os.path.join(d, tag + '.cubin')
    with open(p, 'wb') as f:
        f.write(cubin)
    ins = sass_insts(subprocess.run([CUOBJDUMP, '-sass', p], capture_output=True, text=True).stdout)
    ops = []
    for s in ins:
        toks = s.split()
        ops.append(toks[1] if toks[0].startswith('@') and len(toks) > 1 else toks[0])
    res = dict(total=len(ins), F2FP=sum(1 for o in ops if o.startswith('F2FP')),
               F2FP_BF16_E4M3=sum(1 for o in ops if o.startswith('F2FP.BF16') and 'E4M3' in o),
               F2FP_F16_E4M3=sum(1 for o in ops if o.startswith('F2FP.F16') and 'E4M3' in o),
               LOP3=sum(1 for o in ops if o.startswith('LOP3')), PRMT=sum(1 for o in ops if o.startswith('PRMT')),
               SHF=sum(1 for o in ops if o.startswith('SHF')), cubin_bytes=len(cubin), cubin_sha256=sha(cubin),
               sass_sha256=sha(NL.join(ins).encode()))
    log('census %s: %s' % (tag, json.dumps(res)))
    return res


PROBE_SRC = NL.join([
    '#include <cuda_fp16.h>',
    '#include <cuda_bf16.h>',
    '#include <cuda_fp8.h>',
    '#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__)',
    '__device__ int probe_fam = __CUDA_ARCH_FAMILY_SPECIFIC__;',
    '#else',
    '__device__ int probe_fam = -1;',
    '#endif',
    '#if defined(__CUDA_ARCH_SPECIFIC__)',
    '__device__ int probe_spec = __CUDA_ARCH_SPECIFIC__;',
    '#else',
    '__device__ int probe_spec = -1;',
    '#endif',
    '__device__ int probe_arch = __CUDA_ARCH__;',
    '#if PROBE_CVT',
    'extern ' + DQ + 'C' + DQ + ' __global__ void probe_exh(unsigned int* out)',
    LB,
    '    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;',
    '    unsigned short s = (unsigned short) i;',
    '    unsigned int d;',
    '    asm volatile(' + DQ + 'cvt.rn.bf16x2.e4m3x2 %0, %1;' + DQ + ' : ' + DQ + '=r' + DQ + '(d) : ' + DQ + 'h' + DQ +
    '(s));',
    '    __nv_fp8_e4m3 lo;',
    '    __nv_fp8_e4m3 hi;',
    '    lo.__x = (__nv_fp8_storage_t) (s & 0xFF);',
    '    hi.__x = (__nv_fp8_storage_t) (s >> 8);',
    '    __nv_bfloat16 blo = __nv_bfloat16(lo);',
    '    __nv_bfloat16 bhi = __nv_bfloat16(hi);',
    '    out[2 * i] = d;',
    '    out[2 * i + 1] = (unsigned int) __bfloat16_as_ushort(blo) | ((unsigned int) __bfloat16_as_ushort(bhi) << 16);',
    RB,
    '#endif',
    ''])


def probe():
    cn, cc = cuda_headers()
    res = dict()
    for arch in ('sm_120f', 'sm_120a', 'sm_120'):
        r = dict()
        opts = [b'-default-device', ('-arch=' + arch).encode(), b'-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1',
                b'-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1']
        cubin, err = nvrtc_compile(PROBE_SRC, b'probe.cu', cn, cc, opts + [b'-DPROBE_CVT=0'])
        if cubin is None:
            r['macro_probe_error'] = err[-2000:]
        else:
            img = np.frombuffer(cubin, dtype=np.uint8).copy()
            mod = ck(cu.cuModuleLoadData(img.ctypes.data))
            for nm, key in (('probe_fam', '__CUDA_ARCH_FAMILY_SPECIFIC__'), ('probe_spec', '__CUDA_ARCH_SPECIFIC__'),
                            ('probe_arch', '__CUDA_ARCH__')):
                dptr, _ = ck(cu.cuModuleGetGlobal(mod, nm.encode()))
                v = ctypes.c_int32(0)
                ck(cu.cuMemcpyDtoH(ctypes.addressof(v), dptr, 4))
                r[key] = int(v.value)
            r['note'] = '-1 = macro not defined'
        cubin, err = nvrtc_compile(PROBE_SRC, b'probe.cu', cn, cc, opts + [b'-DPROBE_CVT=1'])
        r['cvt_rn_bf16x2_e4m3x2_compiles'] = cubin is not None
        if cubin is None:
            r['cvt_error_tail'] = err[-600:]
        elif arch == 'sm_120f':
            img = np.frombuffer(cubin, dtype=np.uint8).copy()
            mod = ck(cu.cuModuleLoadData(img.ctypes.data))
            fn = ck(cu.cuModuleGetFunction(mod, b'probe_exh'))
            out = torch.zeros(2 * 65536, dtype=torch.int32, device='cuda')
            pobj = ctypes.c_uint64(out.data_ptr())
            arr = (ctypes.c_void_p * 1)(ctypes.addressof(pobj))
            stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
            ck(cu.cuLaunchKernel(fn, 256, 1, 1, 256, 1, 1, 0, stream, ctypes.addressof(arr), 0))
            torch.cuda.synchronize()
            t = out.view(65536, 2).cpu()
            mism = t[:, 0] != t[:, 1]
            idx = torch.arange(65536)
            nan_in = ((idx & 0x7F) == 0x7F) | (((idx >> 8) & 0x7F) == 0x7F)
            r['exhaustive_e4m3x2_inputs'] = 65536
            r['exhaustive_mismatches_paired_vs_generic'] = int(mism.sum().item())
            r['exhaustive_mismatches_non_nan_inputs'] = int((mism & ~nan_in).sum().item())
        res[arch] = r
        log('probe %s: %s' % (arch, json.dumps(r)))
    return res


class KVCacheList(ctypes.Structure):
    _fields_ = [('pool', ctypes.c_uint64), ('pageList', ctypes.c_uint64), ('seqLen', ctypes.c_uint64),
                ('maxNbPagesPerSeq', ctypes.c_uint32), ('pad', ctypes.c_uint32)]


class Kern(object):

    def __init__(self, cubin):
        self.cubin_sha = sha(cubin)
        self.img = np.frombuffer(cubin, dtype=np.uint8).copy()
        self.mod = ck(cu.cuModuleLoadData(self.img.ctypes.data))
        self.func = ck(cu.cuModuleGetFunction(self.mod, b'kernel_mha'))
        dptr, _ = ck(cu.cuModuleGetGlobal(self.mod, b'smemSize'))
        v = ctypes.c_uint32(0)
        ck(cu.cuMemcpyDtoH(ctypes.addressof(v), dptr, 4))
        self.smem = int(v.value)
        A = cu.CUfunction_attribute
        ck(cu.cuFuncSetAttribute(self.func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))
        self.regs = int(ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)))
        self.lmem = int(ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func)))


class Args(object):

    def __init__(self, inp, batch):
        self.objs = [
            ctypes.c_uint32(KVH), ctypes.c_uint32(MAXLEN), ctypes.c_float(1.0),
            ctypes.c_uint64(inp['out'].data_ptr()), ctypes.c_uint64(inp['q'].data_ptr()), ctypes.c_uint64(0),
            KVCacheList(inp['pool'].data_ptr(), inp['pages'].data_ptr(), inp['seqlen'].data_ptr(), MAXP, 0),
            ctypes.c_uint32(batch), ctypes.c_uint64(inp['kvscale'].data_ptr()), ctypes.c_uint64(inp['sem'].data_ptr()),
            ctypes.c_uint64(inp['scratch_ptr'])
        ]
        self.arr = (ctypes.c_void_p * len(self.objs))(*[ctypes.addressof(o) for o in self.objs])
        self.ptr = ctypes.addressof(self.arr)


def build_inputs(bmax, seed=1234):
    g = torch.Generator(device='cuda')
    g.manual_seed(seed)
    nbp = math.ceil(SEQ / TPP)
    pb = PREFIX // TPP
    uniq = nbp - pb
    pool = (torch.randn((pb + bmax * uniq) * 2, KVH, TPP, D, generator=g, device='cuda') * 2.0).to(
        torch.float8_e4m3fn)
    pages = torch.zeros(bmax, 1, 2, MAXP, dtype=torch.int32)
    for b in range(bmax):
        blk = torch.tensor(list(range(pb)) + [pb + b * uniq + j for j in range(uniq)], dtype=torch.int32)
        pages[b, 0, 0, :nbp] = 2 * blk
        pages[b, 0, 1, :nbp] = 2 * blk + 1
    scratch = torch.zeros(64 << 20, dtype=torch.uint8, device='cuda')
    return dict(pool=pool, pages=pages.cuda(), nbp=nbp,
                q=torch.randn(bmax, NQ, D, generator=g, device='cuda').to(torch.bfloat16),
                seqlen=torch.full((bmax, ), SEQ, dtype=torch.int32, device='cuda'),
                kvscale=torch.tensor([KV_SCALE], dtype=torch.float32, device='cuda'),
                out=torch.zeros(bmax, NQ, D, dtype=torch.bfloat16, device='cuda'),
                sem=torch.zeros(1024, dtype=torch.int32, device='cuda'), scratch=scratch,
                scratch_ptr=(scratch.data_ptr() + 4095) // 4096 * 4096)


def tbytes(t):
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def readonly_hash(inp):
    h = hashlib.sha256()
    for k in ('pool', 'pages', 'q', 'seqlen', 'kvscale'):
        h.update(tbytes(inp[k]))
    return h.hexdigest()


class Flusher(object):

    def __init__(self, mb):
        self.buf = torch.ones((mb << 20) // 4, dtype=torch.int32, device='cuda')

    def __call__(self):
        self.buf.sum()


class Runner(object):

    def __init__(self, kern, inp, b, m):
        self.kern = kern
        self.args = Args(inp, b)
        self.grid = (m, KVH, b)
        self.launch()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.launch()
        torch.cuda.synchronize()

    def launch(self):
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        ck(cu.cuLaunchKernel(self.kern.func, self.grid[0], self.grid[1], self.grid[2], 128, 1, 2, self.kern.smem,
                             stream, self.args.ptr, 0))


def time_round(r, inp, flush):
    n = WARMUP + ITERS
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        inp['out'].fill_(float('nan'))
        flush()
        st[i].record()
        r.graph.replay()
        en[i].record()
    torch.cuda.synchronize()
    return [st[i].elapsed_time(en[i]) for i in range(WARMUP, n)]


def stats(ts):
    ts = np.asarray(ts, dtype=np.float64)
    med = float(np.median(ts))
    p10, p90 = float(np.percentile(ts, 10)), float(np.percentile(ts, 90))
    core = ts[(ts >= p10) & (ts <= p90)]
    tm = float(core.mean()) if core.size else med
    return dict(kernel_ms=tm, median_ms=med, p10_ms=p10, p90_ms=p90, mean_ms=float(ts.mean()),
                noise_floor=(p90 - p10) / tm, samples=int(ts.size))


def run_process(order, outp):
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    ck(cu.cuInit(0))
    res = dict(order=order, git_head=run_text(['git', 'rev-parse', 'HEAD']),
               git_status_xqa=run_text(['git', 'status', '--porcelain', 'cpp/kernels/xqa']), cwd=os.getcwd(),
               device=torch.cuda.get_device_name(), gpu_state_start=gpu_state(), base_sha=BASE_SHA,
               protocol='%d rounds x (%d warmup + %d timed) per arm per config; arms own/own_aa/ref rotated; '
               'NaN-poison + %d MiB L2 read flush before every call; CUDA graph replay + CUDA events' %
               (ROUNDS, WARMUP, ITERS, FLUSH_MB))
    log('order %d git HEAD %s cwd %s state %s' % (order, res['git_head'], res['cwd'], res['gpu_state_start']))
    own_dir = os.path.join(os.getcwd(), 'cpp/kernels/xqa')
    ref_dir, ref_src = extract_base_tree(BASE_SHA)
    res['own_tree'] = tree_hashes(own_dir)
    res['ref_tree'] = tree_hashes(ref_dir)
    res['ref_source'] = ref_src
    log('own tree %s | ref tree (%s) %s' % (json.dumps(res['own_tree']), ref_src, json.dumps(res['ref_tree'])))
    cub_own = compile_mha(own_dir)
    cub_ref = compile_mha(ref_dir)
    kerns = dict(own=Kern(cub_own), own_aa=Kern(cub_own), ref=Kern(cub_ref))
    res['kernels'] = dict((k, dict(regs=v.regs, smem=v.smem, local=v.lmem, cubin_sha256=v.cubin_sha))
                          for k, v in kerns.items())
    log('kernels %s' % json.dumps(res['kernels']))
    inp = build_inputs(4)
    h0 = readonly_hash(inp)
    flush = Flusher(FLUSH_MB)
    names = ['own', 'own_aa', 'ref']
    configs = CONFIGS if order == 0 else CONFIGS[::-1]
    res['configs'] = dict()
    for (b, m) in configs:
        runners = dict((k, Runner(kerns[k], inp, b, m)) for k in names)
        samples = dict((k, []) for k in names)
        firsts = dict()
        rep_ok = dict((k, True) for k in names)
        sem_ok, fin_ok = True, True
        for r in range(ROUNDS):
            k0 = (r + order) % len(names)
            seq = names[k0:] + names[:k0]
            if order == 1:
                seq = seq[::-1]
            for k in seq:
                ts = time_round(runners[k], inp, flush)
                samples[k].extend(ts)
                o = inp['out'][:b].clone()
                fin_ok = fin_ok and bool(torch.isfinite(o.float()).all().item())
                if k not in firsts:
                    firsts[k] = o
                else:
                    rep_ok[k] = rep_ok[k] and bool(torch.equal(o, firsts[k]))
                sem_ok = sem_ok and bool((inp['sem'] == 0).all().item())
                log('  %dx%d round %d %s: trimmed mean %.2f us' % (b, m, r, k, stats(ts)['kernel_ms'] * 1e3))
        c = dict()
        for k in names:
            c[k] = stats(samples[k])
            c[k]['raw_ms'] = samples[k]
        c['own_equals_ref_bitwise'] = bool(torch.equal(firsts['own'], firsts['ref']))
        c['own_aa_equals_own_bitwise'] = bool(torch.equal(firsts['own'], firsts['own_aa']))
        c['repeat_bitwise_per_round'] = rep_ok
        c['semaphores_zero'] = sem_ok
        c['finite'] = fin_ok
        c['output_sha256'] = sha(tbytes(firsts['own']))
        c['ref_output_sha256'] = sha(tbytes(firsts['ref']))
        c['delta_own_vs_ref'] = c['own']['kernel_ms'] / c['ref']['kernel_ms'] - 1.0
        c['delta_aa_vs_own'] = c['own_aa']['kernel_ms'] / c['own']['kernel_ms'] - 1.0
        res['configs']['%dx%d' % (b, m)] = c
        log('%dx%d: own %.2f us (med %.2f p10 %.2f p90 %.2f) | own_aa %.2f | ref %.2f | own/ref-1 %+.4f aa %+.4f | '
            'own==ref %s repeat %s sem0 %s finite %s' %
            (b, m, c['own']['kernel_ms'] * 1e3, c['own']['median_ms'] * 1e3, c['own']['p10_ms'] * 1e3,
             c['own']['p90_ms'] * 1e3, c['own_aa']['kernel_ms'] * 1e3, c['ref']['kernel_ms'] * 1e3,
             c['delta_own_vs_ref'], c['delta_aa_vs_own'], c['own_equals_ref_bitwise'], json.dumps(rep_ok), sem_ok,
             fin_ok))
        del runners
    res['readonly_inputs_unchanged'] = readonly_hash(inp) == h0
    res['gpu_state_end'] = gpu_state()
    if order == 0:
        res['probe'] = probe()
        cen = dict()
        cen['own_sm_120f'] = census(cub_own, 'own_sm_120f')
        cen['ref_sm_120f'] = census(cub_ref, 'ref_sm_120f')
        if res['own_tree'] != res['ref_tree']:
            cub0 = compile_mha(own_dir, 'sm_120f', ['-DXQA_BF16_E4M3_PAIRED_CVT=0'])
            k0 = Kern(cub0)
            cen['own_sm_120f_macro0'] = census(cub0, 'own_sm_120f_macro0')
            cen['own_sm_120f_macro0']['resources'] = dict(regs=k0.regs, smem=k0.smem, local=k0.lmem)
            for arch in ('sm_120a', 'sm_120', 'sm_89'):
                for tag, dd in (('own', own_dir), ('ref', ref_dir)):
                    key = '%s_%s' % (tag, arch)
                    try:
                        cen[key] = census(compile_mha(dd, arch), key)
                    except Exception as e:  # noqa: BLE001
                        cen[key] = dict(error=str(e)[-1500:])
                        log('compile %s failed: %s' % (key, str(e)[-1500:]))

            def same(a, b):
                if 'sass_sha256' not in cen.get(a, dict()) or 'sass_sha256' not in cen.get(b, dict()):
                    return None
                return dict(sass_identical=cen[a]['sass_sha256'] == cen[b]['sass_sha256'],
                            cubin_identical=cen[a]['cubin_sha256'] == cen[b]['cubin_sha256'])

            res['sass_identity'] = dict(own_macro0_vs_ref_sm_120f=same('own_sm_120f_macro0', 'ref_sm_120f'),
                                        own_vs_ref_sm_120=same('own_sm_120', 'ref_sm_120'),
                                        own_vs_ref_sm_89=same('own_sm_89', 'ref_sm_89'),
                                        own_vs_ref_sm_120a=same('own_sm_120a', 'ref_sm_120a'),
                                        own_vs_ref_sm_120f=same('own_sm_120f', 'ref_sm_120f'))
            log('sass identity %s' % json.dumps(res['sass_identity']))
        res['census'] = cen
    with open(outp, 'w') as f:
        json.dump(res, f)
    log('wrote %s' % outp)


def combine(paths):
    rs = [json.load(open(p)) for p in paths]
    out = dict(batch=dict(), extra=dict())
    out['output'] = dict()
    out['output']['class'] = 'exact'
    out['output']['basis'] = ('same math: every e4m3 element goes through the same F2FP.BF16.E4M3 hardware op, two per '
                              'instruction; own output bitwise equal to the in-process base-sha (ref) build')
    allpass = True
    for (b, m) in CONFIGS:
        key = '%dx%d' % (b, m)
        pooled = dict()
        for arm in ('own', 'own_aa', 'ref'):
            raw = []
            for r in rs:
                raw.extend(r['configs'][key][arm]['raw_ms'])
            pooled[arm] = stats(raw)
        e = dict(pooled['own'])
        e['nb_sub_seq'] = m
        e['own_aa_kernel_ms'] = pooled['own_aa']['kernel_ms']
        e['ref_kernel_ms'] = pooled['ref']['kernel_ms']
        e['ref_median_ms'] = pooled['ref']['median_ms']
        e['ref_p10_ms'] = pooled['ref']['p10_ms']
        e['ref_p90_ms'] = pooled['ref']['p90_ms']
        e['aa_floor'] = abs(pooled['own_aa']['kernel_ms'] / pooled['own']['kernel_ms'] - 1.0)
        e['delta_own_vs_ref'] = pooled['own']['kernel_ms'] / pooled['ref']['kernel_ms'] - 1.0
        e['per_process_delta_own_vs_ref'] = [r['configs'][key]['delta_own_vs_ref'] for r in rs]
        e['per_process_delta_aa'] = [r['configs'][key]['delta_aa_vs_own'] for r in rs]
        e['own_equals_ref_bitwise'] = all(r['configs'][key]['own_equals_ref_bitwise'] for r in rs)
        e['repeat_bitwise'] = all(all(r['configs'][key]['repeat_bitwise_per_round'].values()) for r in rs)
        e['semaphores_zero'] = all(r['configs'][key]['semaphores_zero'] for r in rs)
        e['finite'] = all(r['configs'][key]['finite'] for r in rs)
        e['output_sha256'] = rs[0]['configs'][key]['output_sha256']
        allpass = (allpass and e['own_equals_ref_bitwise'] and e['repeat_bitwise'] and e['semaphores_zero'] and
                   e['finite'])
        if (b, m) in PRIMARY:
            out['batch'][str(b)] = e
        else:
            out['extra'][key] = e
        log('POOLED %s: own %.2f us (med %.2f p10 %.2f p90 %.2f) ref %.2f (p10 %.2f p90 %.2f) own/ref-1 %+.4f '
            'A/A %.4f per-process %s' %
            (key, e['kernel_ms'] * 1e3, e['median_ms'] * 1e3, e['p10_ms'] * 1e3, e['p90_ms'] * 1e3,
             e['ref_kernel_ms'] * 1e3, e['ref_p10_ms'] * 1e3, e['ref_p90_ms'] * 1e3, e['delta_own_vs_ref'],
             e['aa_floor'], json.dumps(e['per_process_delta_own_vs_ref'])))
    r0 = [r for r in rs if r['order'] == 0][0]
    out['config'] = dict(git_heads=[r['git_head'] for r in rs], git_status_xqa=[r['git_status_xqa'] for r in rs],
                         own_tree=r0['own_tree'], ref_tree=r0['ref_tree'], ref_source=r0['ref_source'],
                         base_sha=BASE_SHA, kernels=r0['kernels'], protocol=r0['protocol'],
                         gpu_states=[[r['gpu_state_start'], r['gpu_state_end']] for r in rs],
                         readonly_inputs_unchanged=all(r['readonly_inputs_unchanged'] for r in rs),
                         processes='2 per arm: order 0 (configs forward) and order 1 (configs and rotation reversed)')
    out['probe'] = r0.get('probe')
    out['census'] = r0.get('census')
    out['sass_identity'] = r0.get('sass_identity')
    out['checks_passed'] = bool(allpass and out['config']['readonly_inputs_unchanged'])
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'combine':
        combine(sys.argv[2:])
    else:
        run_process(int(sys.argv[1]), sys.argv[2])
