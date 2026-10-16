import ctypes, glob, hashlib, json, math, os, subprocess, sys, time
import numpy as np
import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
# Paired kernel_mha benchmark (run inline with python -c in both arms; compiles <cwd>/cpp/kernels/xqa/mha.cu exactly like
# the NVRTC JIT path). Args: primary configs BxM,... ; extra configs BxM,... (M = sub-sequences per sequence) ; KV dtype
# (bf16 default = this run's serving cache, or fp8).
# Shape: Qwen3.8-27B-NVFP4 full-attention decode, 24 Q / 4 KV heads (HEAD_GRP_SIZE 6), head 256, paged KV 32 tok/block,
# 32768-token shared prefix (block reuse) + 2048 unique + 128 decoded tokens per request, max_seq_len 36864.
# Provenance: prints the arm's git HEAD and the sha256 of every file in cpp/kernels/xqa, so a pair shows that both arms
# compile identical XQA headers and differ only in mha.cu.
HDRS = ['cuda_hint.cuh', 'defines.h', 'ldgsts.cuh', 'mha.h', 'mhaUtils.cuh', 'mma.cuh', 'platform.h', 'utils.cuh', 'utils.h', 'mha_stdheaders.cuh', 'mha_components.cuh', 'mla_sm120.cuh', 'gmma.cuh', 'gmma_impl.cuh', 'barriers.cuh', 'tma.h', 'specDec.h']
CHDRS = ['cuda_bf16.h', 'cuda_bf16.hpp', 'cuda_fp16.h', 'cuda_fp16.hpp', 'cuda_fp8.h', 'cuda_fp8.hpp', 'vector_types.h', 'vector_functions.h', 'device_types.h']
KVDT = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else 'bf16'
KVH, GRP, D, TPP, MAXLEN, PREFIX, SEQ = 4, 6, 256, 32, 36864, 32768, 32768 + 2048 + 128
MACROS = dict(INPUT_FP16='0', DTYPE='__nv_bfloat16', GENERATE_CUBIN='1', NDEBUG='1', HEAD_ELEMS='256', BEAM_WIDTH='1', CACHE_ELEM_ENUM='2' if KVDT == 'fp8' else '0', TOKENS_PER_PAGE='32', HEAD_GRP_SIZE=str(GRP), M_TILESIZE=str(GRP), USE_CUSTOM_BARRIER='1', SLIDING_WINDOW='1', LOW_PREC_OUTPUT='0', USE_INPUT_KV='0', ROPE_STYLE='0', ROPE_ELEMS='64', IS_SPEC_DEC_TREE='0', SKIP_SOFTMAX_ATTN='0', __FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__='1', __FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__='1')
WARMUP, ITERS, ROUNDS, FLUSH_MB = 20, 100, 3, 384


def ck(res):
    err, rest = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
    assert int(err) == 0, err
    return None if not rest else (rest[0] if len(rest) == 1 else rest)


def log(*a):
    print(*a, flush=True)


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # provenance only
        return 'unavailable: %s' % e


prim = [tuple(int(x) for x in c.split('x')) for c in sys.argv[1].split(',') if c]
extra = [tuple(int(x) for x in c.split('x')) for c in (sys.argv[2] if len(sys.argv) > 2 else '').split(',') if c]
xdir = os.path.join(os.getcwd(), 'cpp/kernels/xqa')
head = git_head()
xqa_files = {os.path.relpath(f, xdir): hashlib.sha256(open(f, 'rb').read()).hexdigest()
             for f in sorted(glob.glob(os.path.join(xdir, '*'))) if os.path.isfile(f)}
hdr_digest = hashlib.sha256(''.join('%s:%s\n' % kv for kv in sorted(xqa_files.items()) if kv[0] != 'mha.cu').encode()).hexdigest()
log('git HEAD', head, 'kv', KVDT, 'kv heads', KVH, 'q heads per kv', GRP)
log('cpp/kernels/xqa files except mha.cu: %d, combined sha256 %s' % (len(xqa_files) - 1, hdr_digest))
for f, h in sorted(xqa_files.items()):
    log('  %s %s' % (h, f))
torch.cuda.init()
torch.zeros(1, device='cuda')
ck(cu.cuInit(0))
src = open(os.path.join(xdir, 'mha.cu')).read()
src_sha = hashlib.sha256(src.encode()).hexdigest()
names = [h.encode() for h in HDRS + CHDRS]
cont = [open(os.path.join(xdir, h), 'rb').read() for h in HDRS]
cont += [open(os.path.join('/usr/local/cuda/include', h), 'rb').read() for h in CHDRS]
t0 = time.time()
prog = ck(nvrtc.nvrtcCreateProgram(src.encode(), b'default_program', len(names), cont, names))
opts = [b'-dw', b'--use_fast_math', b'-default-device', b'-arch=sm_120f']
opts += [('-D%s=%s' % kv).encode() for kv in MACROS.items()]
if int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) != 0:
    lsz = ck(nvrtc.nvrtcGetProgramLogSize(prog))
    lbuf = b' ' * lsz
    nvrtc.nvrtcGetProgramLog(prog, lbuf)
    raise RuntimeError('NVRTC compile failed:\n' + lbuf.decode(errors='replace')[-6000:])
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
lmem = ck(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func))
nsm = torch.cuda.get_device_properties(0).multi_processor_count
log('compiled mha.cu sha256', src_sha, 'in %.1fs' % (time.time() - t0), 'smem', smem, 'regs', regs, 'local', lmem,
    'device', torch.cuda.get_device_name(), 'SMs', nsm)

bmax = max(b for b, _ in prim + extra)
g = torch.Generator(device='cuda')
g.manual_seed(1234)
nbp = math.ceil(SEQ / TPP)
pb = PREFIX // TPP
uniq = nbp - pb
pool = torch.randn((pb + bmax * uniq) * 2, KVH, TPP, D, generator=g, device='cuda') * 2.0
pool = pool.to(torch.float8_e4m3fn) if KVDT == 'fp8' else pool.to(torch.bfloat16)
pages = torch.zeros(bmax, 1, 2, MAXLEN // TPP, dtype=torch.int32)
for b in range(bmax):
    blk = torch.tensor(list(range(pb)) + [pb + b * uniq + j for j in range(uniq)], dtype=torch.int32)
    pages[b, 0, 0, :nbp] = 2 * blk
    pages[b, 0, 1, :nbp] = 2 * blk + 1
pages = pages.cuda()
q = torch.randn(bmax, KVH * GRP, D, generator=g, device='cuda').to(torch.bfloat16)
seqlen = torch.full((bmax, ), SEQ, dtype=torch.int32, device='cuda')
kvscale = torch.tensor([0.5 if KVDT == 'fp8' else 1.0], dtype=torch.float32, device='cuda')
out = torch.zeros(bmax, KVH * GRP, D, dtype=torch.bfloat16, device='cuda')
sem = torch.zeros(1024, dtype=torch.int32, device='cuda')
scratch = torch.zeros(64 << 20, dtype=torch.uint8, device='cuda')
scratch_ptr = (scratch.data_ptr() + 4095) // 4096 * 4096
flushbuf = torch.ones((FLUSH_MB << 20) // 4, dtype=torch.int32, device='cuda')
log('KV pool %.1f MB (%s), flush buffer %d MiB' % (pool.numel() * pool.element_size() / 1e6, pool.dtype, FLUSH_MB))


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
    tss, rmeans = [], []
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
        rmeans.append(stats(ts)['kernel_ms'] * 1e3)
        log('  B=%d m=%d round %d: trimmed mean %.2f us' % (b, m, r, rmeans[-1]))
        tss.append(ts)
    s = stats(np.concatenate(tss))
    s['nb_sub_seq'] = m
    s['grid'] = [m, KVH, b]
    s['round_trimmed_means_us'] = rmeans
    s['output_sha256'] = hashlib.sha256(out[:b].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    s['output_finite'] = bool(torch.isfinite(out[:b].float()).all().item())
    s['semaphores_zero'] = bool((sem == 0).all().item())
    return s


# flush bandwidth (outside every timed region): one timed 384 MiB read sweep, reported as bytes/s
fe0, fe1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(3):
    flushbuf.sum()
fe0.record()
for _ in range(10):
    flushbuf.sum()
fe1.record()
torch.cuda.synchronize()
flush_bps = 10 * flushbuf.numel() * 4 / (fe0.elapsed_time(fe1) * 1e-3)
log('flush read sweep: %d MiB at %.1f GB/s' % (FLUSH_MB, flush_bps / 1e9))

res = {'batch': {}, 'output': {'class': 'exact'}, 'extra': {}}
for b, m in prim:
    res['batch'][str(b)] = bench(b, m)
    log('B=%d m=%d: %.2f us (median %.2f, noise %.3f)' % (b, m, res['batch'][str(b)]['kernel_ms'] * 1e3,
                                                        res['batch'][str(b)]['median_ms'] * 1e3,
                                                        res['batch'][str(b)]['noise_floor']))
for b, m in extra:
    res['extra']['%dx%d' % (b, m)] = bench(b, m)
    log('extra B=%d m=%d: %.2f us' % (b, m, res['extra']['%dx%d' % (b, m)]['kernel_ms'] * 1e3))
res['config'] = dict(git_head=head, mha_cu_sha256=src_sha, xqa_other_files_sha256=hdr_digest, smem=smem, regs=regs,
                     local_bytes=lmem, sms=nsm, kv_heads=KVH, q_heads_per_kv=GRP, kv_dtype=KVDT,
                     seq_len=SEQ, shared_prefix=PREFIX, warmup=WARMUP, iters=ITERS, rounds=ROUNDS,
                     flush='%d MiB read sweep (torch.sum) before every call, outside timing; %.1f GB/s' % (FLUSH_MB, flush_bps / 1e9),
                     flush_bytes_per_s=flush_bps,
                     poison='output NaN-filled before every call', timing='CUDA graph replay + CUDA events; kernel_ms = 10-90% trimmed mean',
                     pdl='off in both arms', kv='%s paged, 32 tokens/block, 32768-token shared prefix' % KVDT)
print(json.dumps(res), flush=True)
