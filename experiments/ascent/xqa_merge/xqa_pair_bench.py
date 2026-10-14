import ctypes, hashlib, json, math, os, sys, time
import numpy as np
import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
# Paired kernel_mha benchmark (run inline with python -c in both arms; compiles <cwd>/cpp/kernels/xqa/mha.cu exactly like
# the NVRTC JIT path). Args: primary configs BxM,... and extra configs BxM,... (M = sub-sequences per sequence).
HDRS = ['cuda_hint.cuh', 'defines.h', 'ldgsts.cuh', 'mha.h', 'mhaUtils.cuh', 'mma.cuh', 'platform.h', 'utils.cuh', 'utils.h', 'mha_stdheaders.cuh', 'mha_components.cuh', 'mla_sm120.cuh', 'gmma.cuh', 'gmma_impl.cuh', 'barriers.cuh', 'tma.h', 'specDec.h']
CHDRS = ['cuda_bf16.h', 'cuda_bf16.hpp', 'cuda_fp16.h', 'cuda_fp16.hpp', 'cuda_fp8.h', 'cuda_fp8.hpp', 'vector_types.h', 'vector_functions.h', 'device_types.h']
MACROS = dict(INPUT_FP16='0', DTYPE='__nv_bfloat16', GENERATE_CUBIN='1', NDEBUG='1', HEAD_ELEMS='256', BEAM_WIDTH='1', CACHE_ELEM_ENUM='2', TOKENS_PER_PAGE='32', HEAD_GRP_SIZE='8', M_TILESIZE='8', USE_CUSTOM_BARRIER='1', SLIDING_WINDOW='1', LOW_PREC_OUTPUT='0', USE_INPUT_KV='0', ROPE_STYLE='0', ROPE_ELEMS='64', IS_SPEC_DEC_TREE='0', SKIP_SOFTMAX_ATTN='0', __FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__='1', __FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__='1')
KVH, GRP, D, TPP, MAXLEN, PREFIX, SEQ = 2, 8, 256, 32, 36864, 32768, 32768 + 2048 + 128
WARMUP, ITERS, ROUNDS, FLUSH_MB = 20, 100, 3, 384


def ck(res):
    err, rest = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
    assert int(err) == 0, err
    return None if not rest else (rest[0] if len(rest) == 1 else rest)


def log(*a):
    print(*a, flush=True)


prim = [tuple(int(x) for x in c.split('x')) for c in sys.argv[1].split(',') if c]
extra = [tuple(int(x) for x in c.split('x')) for c in (sys.argv[2] if len(sys.argv) > 2 else '').split(',') if c]
torch.cuda.init()
torch.zeros(1, device='cuda')
ck(cu.cuInit(0))
xdir = os.path.join(os.getcwd(), 'cpp/kernels/xqa')
src = open(os.path.join(xdir, 'mha.cu')).read()
src_sha = hashlib.sha256(src.encode()).hexdigest()
names = [h.encode() for h in HDRS + CHDRS]
cont = [open(os.path.join(xdir, h), 'rb').read() for h in HDRS]
cont += [open(os.path.join('/usr/local/cuda/include', h), 'rb').read() for h in CHDRS]
prog = ck(nvrtc.nvrtcCreateProgram(src.encode(), b'default_program', len(names), cont, names))
opts = [b'-dw', b'--use_fast_math', b'-default-device', b'-arch=sm_120f']
opts += [('-D%s=%s' % kv).encode() for kv in MACROS.items()]
ck(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
size = ck(nvrtc.nvrtcGetCUBINSize(prog))
cubin = b' ' * size
ck(nvrtc.nvrtcGetCUBIN(prog, cubin))
img = np.frombuffer(cubin, dtype=np.uint8).copy()
mod = ck(cu.cuModuleLoadData(img.ctypes.data))
func = ck(cu.cuModuleGetFunction(mod, b'kernel_mha'))
dptr, _ = ck(cu.cuModuleGetGlobal(mod, b'smemSize'))
sv = ctypes.c_uint32(0)
ck(cu.cuMemcpyDtoH(ctypes.addressof(sv), dptr, 4))
smem = int(sv.value)
A = cu.CUfunction_attribute
ck(cu.cuFuncSetAttribute(func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
regs = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, func))
log('compiled mha.cu sha256', src_sha[:16], 'smem', smem, 'regs', regs)

bmax = max(b for b, _ in prim + extra)
g = torch.Generator(device='cuda')
g.manual_seed(1234)
nbp = math.ceil(SEQ / TPP)
pb = PREFIX // TPP
uniq = nbp - pb
pool = (torch.randn((pb + bmax * uniq) * 2, KVH, TPP, D, generator=g, device='cuda') * 2.0).to(torch.float8_e4m3fn)
pages = torch.zeros(bmax, 1, 2, MAXLEN // TPP, dtype=torch.int32)
for b in range(bmax):
    blk = torch.tensor(list(range(pb)) + [pb + b * uniq + j for j in range(uniq)], dtype=torch.int32)
    pages[b, 0, 0, :nbp] = 2 * blk
    pages[b, 0, 1, :nbp] = 2 * blk + 1
pages = pages.cuda()
q = torch.randn(bmax, KVH * GRP, D, generator=g, device='cuda').to(torch.bfloat16)
seqlen = torch.full((bmax, ), SEQ, dtype=torch.int32, device='cuda')
kvscale = torch.tensor([0.5], dtype=torch.float32, device='cuda')
out = torch.zeros(bmax, KVH * GRP, D, dtype=torch.bfloat16, device='cuda')
sem = torch.zeros(1024, dtype=torch.int32, device='cuda')
scratch = torch.zeros(64 << 20, dtype=torch.uint8, device='cuda')
scratch_ptr = (scratch.data_ptr() + 4095) // 4096 * 4096
flushbuf = torch.ones((FLUSH_MB << 20) // 4, dtype=torch.int32, device='cuda')


class KVCacheList(ctypes.Structure):
    _fields_ = [('pool', ctypes.c_uint64), ('pageList', ctypes.c_uint64), ('seqLen', ctypes.c_uint64),
                ('maxNbPagesPerSeq', ctypes.c_uint32), ('pad', ctypes.c_uint32)]


def make_args(batch):
    objs = [ctypes.c_uint32(KVH), ctypes.c_uint32(MAXLEN), ctypes.c_float(1.0), ctypes.c_uint64(out.data_ptr()),
            ctypes.c_uint64(q.data_ptr()), ctypes.c_uint64(0),
            KVCacheList(pool.data_ptr(), pages.data_ptr(), seqlen.data_ptr(), MAXLEN // TPP, 0), ctypes.c_uint32(batch),
            ctypes.c_uint64(kvscale.data_ptr()), ctypes.c_uint64(sem.data_ptr()), ctypes.c_uint64(scratch_ptr)]
    arr = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
    return objs, arr


def launch(arr, grid):
    stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    ck(cu.cuLaunchKernel(func, grid[0], grid[1], grid[2], 128, 1, 2, smem, stream, ctypes.addressof(arr), 0))


def stats(ts):
    med = float(np.median(ts))
    p10, p90 = float(np.percentile(ts, 10)), float(np.percentile(ts, 90))
    core = ts[(ts >= p10) & (ts <= p90)]
    tm = float(core.mean()) if core.size else med
    return dict(kernel_ms=tm, median_ms=med, p10_ms=p10, p90_ms=p90, mean_ms=float(ts.mean()),
                noise_floor=(p90 - p10) / tm, samples=int(ts.size))


def bench(b, m):
    objs, arr = make_args(b)
    grid = (m, KVH, b)
    launch(arr, grid)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(arr, grid)
    torch.cuda.synchronize()
    tss = []
    for r in range(ROUNDS):
        n = WARMUP + ITERS
        st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            out.fill_(float('nan'))
            flushbuf.sum()
            st[i].record()
            graph.replay()
            en[i].record()
        torch.cuda.synchronize()
        ts = np.array([st[i].elapsed_time(en[i]) for i in range(WARMUP, n)])
        log('  B=%d m=%d round %d: trimmed mean %.2f us' % (b, m, r, stats(ts)['kernel_ms'] * 1e3))
        tss.append(ts)
    s = stats(np.concatenate(tss))
    s['nb_sub_seq'] = m
    s['output_sha256'] = hashlib.sha256(out[:b].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    s['output_finite'] = bool(torch.isfinite(out[:b].float()).all().item())
    return s


res = {'batch': {}, 'output': {'class': 'exact'}, 'extra': {}}
for b, m in prim:
    res['batch'][str(b)] = bench(b, m)
    log('B=%d m=%d: %.2f us' % (b, m, res['batch'][str(b)]['kernel_ms'] * 1e3))
for b, m in extra:
    res['extra']['%dx%d' % (b, m)] = bench(b, m)
    log('extra B=%d m=%d: %.2f us' % (b, m, res['extra']['%dx%d' % (b, m)]['kernel_ms'] * 1e3))
res['config'] = dict(mha_cu_sha256=src_sha, smem=smem, regs=regs, seq_len=SEQ, shared_prefix=PREFIX, warmup=WARMUP,
                     iters=ITERS, rounds=ROUNDS, flush='384 MiB read sweep before every call (outside timing)',
                     poison='output NaN-filled before every call', timing='CUDA graph replay + CUDA events; kernel_ms = 10-90% trimmed mean',
                     pdl='off in both arms', kv='FP8 E4M3 paged, 32 tokens/block, 32768-token shared prefix')
print(json.dumps(res), flush=True)
