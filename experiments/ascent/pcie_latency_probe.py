"""Idea 244 / branch 201: PCIe probe on 4x RTX PRO 6000 (SM120), one 4-GPU job, four parts.

Parts (each runs as its own subprocess under `timeout -s KILL`, launched by pcie_probe_driver.sh):
  build    compile the load_inline CUDA helpers once (no kernel runs).
  lat      (i)  one-way posted-write latency, 12 directed pairs: 1-thread peer flag ping-pong (st/ld.volatile and
                st.release.sys/ld.acquire.sys), 10k round trips after 1k warmup; RTT/2 median/p10/p90 per pair.
  floor    (ii) one-shot primitive floor: one kernel per GPU pushes M x 5120 bf16 (16 B/thread, block 640, grid M)
                to its 3 peers and polls locally, M=1/4. Two protocols: 'lamport' (data is its own flag, -0 sentinel,
                triple buffer, as allreduce_fusion_kernel_oneshot_lamport) and 'flag' (stores, fence.sys, flag
                store, flag poll). Two harnesses: an in-kernel loop (1000 iterations after 100 warmup, per-iteration
                clock64 on CTA 0: no launch cost) and an 8-launch CUDA graph chain per GPU (1000 launches, no PDL)
                beside an 8-launch EMPTY kernel chain of the same grid. All 4 GPUs start behind a host go-flag.
  bwread   (iv-a) SM peer READ bandwidth per directed pair (2.6 / 10.5 MB, 16 and 64 CTAs x 512 thr, 16 B per
                load, unroll 1 and 4) and the 4-rank x 3-peer concurrent read pattern at 2.6 MB.
  bwrest   (iv-b/c) SM posted WRITE (16/64 CTAs) and cudaMemcpyPeerAsync push / pull, per pair and concurrent.
  served   (iii) the served lamport AR through TRT-LLM's AllReduce (strategy LOWPRECISION as main's engine.json,
                autotuned like the served warm-up), 8-AR CUDA graph chains at M=1/2/4/8, trigger_completion_at_end
                True and False; chains: AR x8, [PDL predecessor -> AR] x8 and predecessor-only x8; median of 3
                sequences, 20 warmup + 100 timed replays, seeded inputs re-created per sequence, max over ranks;
                explicit ONESHOT as a control; dispatch asserted afterwards (torch.profiler, after all timing).
Single-process parts (lat, floor, bw*) use one process with peer access enabled between all 4 devices (the same
PCIe peer path IPC handles map). Every part prints its JSON as the last stdout line.
"""
import json
import os
import pickle
import statistics
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORLD = 4
HIDDEN = 5120
SENT = 0x80008000  # two bf16 -0.0: lamport sentinel per 32-bit word


def log(*a):
    print(*a, flush=True)


def watchdog(tag, seconds):
    def fire():
        sys.stderr.write(f"[watchdog:{tag}] {seconds}s elapsed, dumping stacks and exiting\n")
        for tid, frame in sys._current_frames().items():
            sys.stderr.write(f"--- thread {tid}\n" + "".join(traceback.format_stack(frame)))
        sys.stderr.flush()
        print(json.dumps({"passed": False, "part": tag, "error": "watchdog"}), flush=True)
        os._exit(2)
    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()
    return t


_CPP = r"""
#include <torch/extension.h>
std::vector<int64_t> enable_peer(int64_t n);
void ping(int64_t remote, int64_t local, int64_t warm, int64_t n, int64_t cyc, int64_t gt, int64_t err, int64_t sem);
void pong(int64_t remote, int64_t local, int64_t total, int64_t err, int64_t sem);
void floor_launch(int64_t mode, std::vector<int64_t> recv, std::vector<int64_t> flags, int64_t rank, int64_t in,
                  int64_t out, int64_t M, int64_t niters, int64_t ctr, int64_t ts, int64_t err);
void empty_launch(int64_t M);
void copy_launch(int64_t src, int64_t dst, int64_t nbytes, int64_t ctas, int64_t unroll);
void copy3_launch(std::vector<int64_t> src, std::vector<int64_t> dst, int64_t nbytes, int64_t ctas);
void wait_go(int64_t hostflag, int64_t err);
int64_t host_alloc(int64_t nbytes);
void host_set(int64_t p, int64_t v);
void memcpy_peer(int64_t dst, int64_t dstdev, int64_t src, int64_t srcdev, int64_t nbytes);
void pdl_pred();
"""

_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <string.h>
#include <vector>

#define SENT 0x80008000u
#define TIMEOUT_CYC 6000000000LL

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void st_vol(unsigned* p, unsigned v) {
    asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned ld_vol(const unsigned* p) {
    unsigned v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
    unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_vol4(uint4* p, uint4 v) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory"); }
__device__ __forceinline__ uint4 ld_vol4(const uint4* p) {
    uint4 v; asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                          : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }

// ---------------------------------------------------------------- (i) ping-pong
__global__ void ping_k(unsigned* remote, unsigned* local, int warm, int n, long long* cyc,
                       unsigned long long* gt, int* err, int sem) {
    unsigned long long g0 = 0; long long c0 = 0;
    for (int i = 1; i <= warm + n; i++) {
        if (i == warm + 1) { g0 = gtimer(); c0 = clock64(); }
        long long t0 = clock64();
        if (sem) st_rel(remote, (unsigned)i); else st_vol(remote, (unsigned)i);
        while ((sem ? ld_acq(local) : ld_vol(local)) != (unsigned)i) {
            if (clock64() - t0 > TIMEOUT_CYC) { *err = 1; return; }
        }
        long long t1 = clock64();
        if (i > warm) cyc[i - warm - 1] = t1 - t0;
    }
    gt[0] = gtimer() - g0; gt[1] = (unsigned long long)(clock64() - c0);
}
__global__ void pong_k(unsigned* remote, unsigned* local, int total, int* err, int sem) {
    for (int i = 1; i <= total; i++) {
        long long t0 = clock64();
        while ((sem ? ld_acq(local) : ld_vol(local)) != (unsigned)i) {
            if (clock64() - t0 > TIMEOUT_CYC) { *err = 2; return; }
        }
        if (sem) st_rel(remote, (unsigned)i); else st_vol(remote, (unsigned)i);
    }
}

// ---------------------------------------------------------------- (ii) one-shot primitive
struct Ptr4 { uint4* p[4]; };
struct Flag4 { unsigned* p[4]; };

__device__ __forceinline__ void acc8(float* a, uint4 v) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    #pragma unroll
    for (int k = 0; k < 4; k++) { float2 f = __bfloat1622float2(h[k]); a[2*k] += f.x; a[2*k+1] += f.y; }
}
__device__ __forceinline__ uint4 pack8(const float* a) {
    uint4 v; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
    #pragma unroll
    for (int k = 0; k < 4; k++) h[k] = __floats2bfloat162_rn(a[2*k], a[2*k+1]);
    return v;
}

// lamport: recv layout [3 buf][4 src][M*640] uint4, sentinel-filled; data is its own flag.
__global__ void floor_lamport_k(Ptr4 R, int rank, const uint4* in, uint4* out, int M, int niters, unsigned* ctr,
                                long long* ts, int* err) {
    const int tok = blockIdx.x, t = threadIdx.x, vecs = M * 640, idx = tok * 640 + t;
    __shared__ unsigned base;
    if (t == 0) base = ctr[tok];
    __syncthreads();
    const unsigned b0 = base;
    const uint4 v = in[idx];
    if (ts && tok == 0 && t == 0) { ts[niters] = clock64(); ts[niters + 1] = (long long)gtimer(); }
    for (int it = 0; it < niters; it++) {
        const unsigned i = b0 + it; const int buf = i % 3;
        #pragma unroll
        for (int p = 0; p < 4; p++) if (p != rank) st_vol4(R.p[p] + ((size_t)(buf * 4 + rank)) * vecs + idx, v);
        float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        acc8(a, v);
        uint4* mine = R.p[rank];
        long long t0 = clock64();
        for (int p = 0; p < 4; p++) {
            if (p == rank) continue;
            uint4* q = mine + ((size_t)(buf * 4 + p)) * vecs + idx;
            uint4 r = ld_vol4(q);
            while (r.x == SENT || r.y == SENT || r.z == SENT || r.w == SENT) {
                if (clock64() - t0 > TIMEOUT_CYC) { *err = 3; return; }
                r = ld_vol4(q);
            }
            acc8(a, r);
            *q = make_uint4(SENT, SENT, SENT, SENT);
        }
        out[idx] = pack8(a);
        __threadfence();
        if (ts && tok == 0) { __syncthreads(); if (t == 0) ts[it] = clock64(); }
    }
    if (ts && tok == 0 && t == 0) { ts[niters + 2] = clock64(); ts[niters + 3] = (long long)gtimer(); }
    __syncthreads();
    if (t == 0) ctr[tok] = b0 + niters;
}

// flag: recv layout [2 buf][4 src][M*640]; flags [M][4 src] (monotonic counters).
__global__ void floor_flag_k(Ptr4 R, Flag4 F, int rank, const uint4* in, uint4* out, int M, int niters,
                             unsigned* ctr, long long* ts, int* err) {
    const int tok = blockIdx.x, t = threadIdx.x, vecs = M * 640, idx = tok * 640 + t;
    __shared__ unsigned base;
    if (t == 0) base = ctr[tok];
    __syncthreads();
    const unsigned b0 = base;
    const uint4 v = in[idx];
    if (ts && tok == 0 && t == 0) { ts[niters] = clock64(); ts[niters + 1] = (long long)gtimer(); }
    for (int it = 0; it < niters; it++) {
        const unsigned i = b0 + it + 1; const int buf = i & 1;
        #pragma unroll
        for (int p = 0; p < 4; p++) if (p != rank) R.p[p][((size_t)(buf * 4 + rank)) * vecs + idx] = v;
        __threadfence_system();
        __syncthreads();
        if (t < 4 && t != rank) {
            st_vol(F.p[t] + tok * 4 + rank, i);
            long long t0 = clock64();
            while (ld_vol(F.p[rank] + tok * 4 + t) < i) { if (clock64() - t0 > TIMEOUT_CYC) { *err = 4; return; } }
        }
        __syncthreads();
        float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        acc8(a, v);
        for (int p = 0; p < 4; p++) if (p != rank) acc8(a, ld_vol4(R.p[rank] + ((size_t)(buf * 4 + p)) * vecs + idx));
        out[idx] = pack8(a);
        if (ts && tok == 0) { __syncthreads(); if (t == 0) ts[it] = clock64(); }
    }
    if (ts && tok == 0 && t == 0) { ts[niters + 2] = clock64(); ts[niters + 3] = (long long)gtimer(); }
    __syncthreads();
    if (t == 0) ctr[tok] = b0 + niters;
}

__global__ void empty_k() {}

// ---------------------------------------------------------------- (iv) bandwidth
template <int U>
__global__ void copy_k(const uint4* __restrict__ src, uint4* __restrict__ dst, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n; i += U * stride) {
        uint4 r[U];
        #pragma unroll
        for (int u = 0; u < U; u++) r[u] = src[i + u * stride];
        #pragma unroll
        for (int u = 0; u < U; u++) dst[i + u * stride] = r[u];
    }
    for (; i < n; i += stride) dst[i] = src[i];
}
struct Seg3 { const uint4* s[3]; uint4* d[3]; };
__global__ void copy3_k(Seg3 S, long long n) {
    const uint4* __restrict__ src = S.s[blockIdx.y];
    uint4* __restrict__ dst = S.d[blockIdx.y];
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}
__global__ void wait_go_k(volatile int* f, int* err) {
    long long t0 = clock64();
    while (*f == 0) { if (clock64() - t0 > 4 * TIMEOUT_CYC) { *err = 5; return; } }
}
__global__ void pdl_pred_k() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

static cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

std::vector<int64_t> enable_peer(int64_t n) {
    std::vector<int64_t> res;
    int prev; cudaGetDevice(&prev);
    for (int i = 0; i < n; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < n; j++) {
            if (i == j) { res.push_back(-1); continue; }
            cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e = cudaSuccess; }
            res.push_back((int64_t)e);
        }
    }
    cudaSetDevice(prev);
    return res;
}
void ping(int64_t remote, int64_t local, int64_t warm, int64_t n, int64_t cyc, int64_t gt, int64_t err, int64_t sem) {
    ping_k<<<1, 1, 0, cur()>>>((unsigned*)remote, (unsigned*)local, (int)warm, (int)n, (long long*)cyc,
                               (unsigned long long*)gt, (int*)err, (int)sem);
}
void pong(int64_t remote, int64_t local, int64_t total, int64_t err, int64_t sem) {
    pong_k<<<1, 1, 0, cur()>>>((unsigned*)remote, (unsigned*)local, (int)total, (int*)err, (int)sem);
}
void floor_launch(int64_t mode, std::vector<int64_t> recv, std::vector<int64_t> flags, int64_t rank, int64_t in,
                  int64_t out, int64_t M, int64_t niters, int64_t ctr, int64_t ts, int64_t err) {
    Ptr4 R; Flag4 F;
    for (int k = 0; k < 4; k++) { R.p[k] = (uint4*)recv[k]; F.p[k] = (unsigned*)flags[k]; }
    if (mode == 0)
        floor_lamport_k<<<(unsigned)M, 640, 0, cur()>>>(R, (int)rank, (const uint4*)in, (uint4*)out, (int)M,
                                                       (int)niters, (unsigned*)ctr, (long long*)ts, (int*)err);
    else
        floor_flag_k<<<(unsigned)M, 640, 0, cur()>>>(R, F, (int)rank, (const uint4*)in, (uint4*)out, (int)M,
                                                    (int)niters, (unsigned*)ctr, (long long*)ts, (int*)err);
}
void empty_launch(int64_t M) { empty_k<<<(unsigned)M, 640, 0, cur()>>>(); }
void copy_launch(int64_t src, int64_t dst, int64_t nbytes, int64_t ctas, int64_t unroll) {
    long long n = nbytes / 16;
    if (unroll == 4) copy_k<4><<<(unsigned)ctas, 512, 0, cur()>>>((const uint4*)src, (uint4*)dst, n);
    else copy_k<1><<<(unsigned)ctas, 512, 0, cur()>>>((const uint4*)src, (uint4*)dst, n);
}
void copy3_launch(std::vector<int64_t> src, std::vector<int64_t> dst, int64_t nbytes, int64_t ctas) {
    Seg3 S;
    for (int k = 0; k < 3; k++) { S.s[k] = (const uint4*)src[k]; S.d[k] = (uint4*)dst[k]; }
    copy3_k<<<dim3((unsigned)ctas, 3), 512, 0, cur()>>>(S, nbytes / 16);
}
void wait_go(int64_t hostflag, int64_t err) { wait_go_k<<<1, 1, 0, cur()>>>((volatile int*)hostflag, (int*)err); }
int64_t host_alloc(int64_t nbytes) {
    void* p = nullptr;
    cudaHostAlloc(&p, (size_t)nbytes, cudaHostAllocMapped | cudaHostAllocPortable);
    memset(p, 0, (size_t)nbytes);
    return (int64_t)p;
}
void host_set(int64_t p, int64_t v) { *((volatile int*)p) = (int)v; }
void memcpy_peer(int64_t dst, int64_t dstdev, int64_t src, int64_t srcdev, int64_t nbytes) {
    cudaMemcpyPeerAsync((void*)dst, (int)dstdev, (const void*)src, (int)srcdev, (size_t)nbytes, cur());
}
void pdl_pred() {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = 0; cfg.stream = cur();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, pdl_pred_k);
}
"""

_FUNCS = ["enable_peer", "ping", "pong", "floor_launch", "empty_launch", "copy_launch", "copy3_launch", "wait_go",
          "host_alloc", "host_set", "memcpy_peer", "pdl_pred"]
_MOD = None


def ext():
    """Build once (part 'build'); later processes import the built .so directly (no concurrent rebuild checks)."""
    global _MOD
    if _MOD is None:
        import torch  # noqa: F401  (libtorch symbols for the extension)
        build = os.environ.get("PCIE_PROBE_EXT_DIR", "/tmp/pcie_probe_ext")
        so = os.path.join(build, "pcie_probe_ext.so")
        if os.path.exists(so) and os.environ.get("PCIE_PROBE_PREBUILT") == "1":
            import importlib.util
            spec = importlib.util.spec_from_file_location("pcie_probe_ext", so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _MOD = mod
            return _MOD
        from torch.utils.cpp_extension import load_inline
        os.makedirs(build, exist_ok=True)
        _MOD = load_inline(name="pcie_probe_ext", cpp_sources=_CPP, cuda_sources=_CU, functions=_FUNCS,
                           build_directory=build, verbose=False,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"])
    return _MOD


# ------------------------------------------------------------------------------------------- topology helpers
def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                              timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def topo_classes(torch):
    """pair class (topo -m code) per CUDA device pair, mapping CUDA ordinals to nvidia-smi indices by UUID."""
    q = sh(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"])
    smi = {}
    for line in q.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            smi[parts[1].replace("GPU-", "").lower()] = (int(parts[0]), parts[2])
    cuda2smi, uuids = {}, {}
    for d in range(torch.cuda.device_count()):
        u = str(torch.cuda.get_device_properties(d).uuid).replace("GPU-", "").lower()
        uuids[d] = u
        cuda2smi[d] = smi.get(u, (None, None))[0]
    mat = {}
    topo = sh(["nvidia-smi", "topo", "-m"])
    lines = topo.splitlines()
    hdr = [ln for ln in lines if "GPU0" in ln and not ln.strip().startswith("GPU0 ")
           and not ln.startswith("GPU")]
    cols = hdr[0].split() if hdr else []
    for ln in lines:
        if not ln.startswith("GPU"):
            continue
        f = ln.split()
        name = f[0]
        if not name[3:].isdigit():
            continue
        i = int(name[3:])
        for k, c in enumerate(cols):
            if c.startswith("GPU") and c[3:].isdigit() and k + 1 < len(f):
                mat[(i, int(c[3:]))] = f[k + 1]
    cls = {}
    for a in range(WORLD):
        for b in range(WORLD):
            if a != b:
                cls[f"{a}->{b}"] = mat.get((cuda2smi.get(a), cuda2smi.get(b)), "?")
    return {"cuda_to_smi_index": cuda2smi, "uuid": uuids, "pair_class": cls}


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def by_class(per_pair, cls, keys):
    out = {}
    for pair, row in per_pair.items():
        c = cls.get(pair, "?")
        out.setdefault(c, {k: [] for k in keys})
        for k in keys:
            if row.get(k) is not None:
                out[c][k].append(row[k])
    res = {}
    for c, d in out.items():
        res[c] = {k: (round(statistics.median(v), 3) if v else None) for k, v in d.items()}
        res[c]["min_" + keys[0]] = round(min(d[keys[0]]), 3) if d[keys[0]] else None
        res[c]["max_" + keys[0]] = round(max(d[keys[0]]), 3) if d[keys[0]] else None
        res[c]["pairs"] = len([p for p in per_pair if cls.get(p) == c])
    return res


def setup_single():
    import torch
    n = torch.cuda.device_count()
    assert n == WORLD, f"need 4 visible devices, have {n}"
    m = ext()
    pe = m.enable_peer(WORLD)
    bad = [i for i, e in enumerate(pe) if e not in (-1, 0)]
    assert not bad, f"enable_peer errors {pe}"
    return torch, m, pe


def sync_all(torch):
    for d in range(WORLD):
        torch.cuda.synchronize(d)


# ------------------------------------------------------------------------------------------------- (i)
def part_lat():
    wd = watchdog("lat", 280)
    torch, m, pe = setup_single()
    topo = topo_classes(torch)
    cls = topo["pair_class"]
    WARM, N = 1000, 10000
    flags = [torch.zeros(64, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
    errs = [torch.zeros(4, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
    res = {}
    for sem, semname in ((0, "volatile"), (1, "release_acquire")):
        per = {}
        for a in range(WORLD):
            for b in range(WORLD):
                if a == b:
                    continue
                for f in flags:
                    f.zero_()
                sync_all(torch)
                cyc = torch.zeros(N, dtype=torch.int64, device=f"cuda:{a}")
                gt = torch.zeros(2, dtype=torch.int64, device=f"cuda:{a}")
                with torch.cuda.device(b):
                    m.pong(flags[a].data_ptr(), flags[b].data_ptr(), WARM + N, errs[b].data_ptr(), sem)
                with torch.cuda.device(a):
                    m.ping(flags[b].data_ptr(), flags[a].data_ptr(), WARM, N, cyc.data_ptr(), gt.data_ptr(),
                           errs[a].data_ptr(), sem)
                sync_all(torch)
                e = int(errs[a][0].item()) | int(errs[b][0].item())
                g = gt.tolist()
                cyc_per_ns = g[1] / max(g[0], 1)
                ns = [c / cyc_per_ns for c in cyc.tolist()]
                ow = [x / 2 for x in ns]
                row = {"oneway_med_ns": round(pct(ow, 0.5), 1), "oneway_p10_ns": round(pct(ow, 0.1), 1),
                       "oneway_p90_ns": round(pct(ow, 0.9), 1), "rtt_med_ns": round(pct(ns, 0.5), 1),
                       "rtt_mean_from_globaltimer_ns": round(g[0] / N, 1),
                       "sm_ghz": round(cyc_per_ns, 3), "err": e, "class": cls[f"{a}->{b}"]}
                per[f"{a}->{b}"] = row
                log(f"[lat:{semname}] {a}->{b} ({row['class']}) one-way med {row['oneway_med_ns']} "
                    f"p10 {row['oneway_p10_ns']} p90 {row['oneway_p90_ns']} ns "
                    f"(rtt globaltimer mean {row['rtt_mean_from_globaltimer_ns']} ns, {row['sm_ghz']} GHz) err={e}")
        res[semname] = {"per_pair": per,
                        "by_class": by_class(per, cls, ["oneway_med_ns", "oneway_p10_ns", "oneway_p90_ns"])}
    wd.cancel()
    ok = all(r["err"] == 0 for v in res.values() for r in v["per_pair"].values())
    return {"passed": ok, "part": "lat", "topo": topo, "peer_enable": pe, "warmup": WARM, "round_trips": N, **res}


# ------------------------------------------------------------------------------------------------- (ii)
class FloorBufs:
    def __init__(self, torch, M):
        self.M = M
        vec = M * 640  # uint4 per source slot
        self.recv_l = [torch.full((3 * 4 * vec * 4,), SENT - (1 << 32), dtype=torch.int32, device=f"cuda:{d}")
                       for d in range(WORLD)]
        self.recv_f = [torch.zeros((2 * 4 * vec * 4,), dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
        self.flags = [torch.zeros(M * 4 + 64, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
        self.ctr = {k: [torch.zeros(64, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
                    for k in (0, 1)}
        g = torch.Generator(device="cpu")
        g.manual_seed(1234 + M)
        self.inp = [torch.randn((M, HIDDEN), generator=g).to(torch.bfloat16).to(f"cuda:{d}") for d in range(WORLD)]
        self.out = [torch.zeros((M, HIDDEN), dtype=torch.bfloat16, device=f"cuda:{d}") for d in range(WORLD)]
        self.err = [torch.zeros(4, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]

    def launch(self, m, mode, d, niters, ts_ptr=0):
        recv = [(self.recv_l if mode == 0 else self.recv_f)[p].data_ptr() for p in range(WORLD)]
        fl = [self.flags[p].data_ptr() for p in range(WORLD)]
        m.floor_launch(mode, recv, fl, d, self.inp[d].data_ptr(), self.out[d].data_ptr(), self.M, niters,
                       self.ctr[mode][d].data_ptr(), ts_ptr, self.err[d].data_ptr())

    def check(self, torch):
        sync_all(torch)
        ref = sum(x.float().cpu() for x in self.inp)
        e = 0.0
        for d in range(WORLD):
            e = max(e, (self.out[d].float().cpu() - ref).abs().max().item())
        return e


def _time_graphs(torch, m, go, goerr, graphs, CH, N):
    """all graphs enqueued per device behind a host go flag (no host pacing), one event pair per replay."""
    reps = N // CH
    for _ in range(3):  # warm replays
        for d in range(WORLD):
            with torch.cuda.device(d):
                graphs[d].replay()
    sync_all(torch)
    m.host_set(go, 0)
    evs = []
    for d in range(WORLD):
        with torch.cuda.device(d):
            m.wait_go(go, goerr[d].data_ptr())
            e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for i in range(reps):
                e[i][0].record()
                graphs[d].replay()
                e[i][1].record()
            evs.append(e)
    time.sleep(0.002)
    m.host_set(go, 1)
    sync_all(torch)
    per_rank = []
    for d in range(WORLD):
        ts = [a.elapsed_time(b) * 1e3 / CH for a, b in evs[d]]  # us per launch
        tot = evs[d][0][0].elapsed_time(evs[d][-1][1]) * 1e3 / (reps * CH)
        per_rank.append({"med_us": round(pct(ts, 0.5), 3), "p90_us": round(pct(ts, 0.9), 3),
                         "p10_us": round(pct(ts, 0.1), 3), "mean_total_us": round(tot, 3)})
    return {"med_us_max_rank": max(r["med_us"] for r in per_rank),
            "p90_us_max_rank": max(r["p90_us"] for r in per_rank), "per_rank": per_rank}


def part_floor():
    wd = watchdog("floor", 280)
    torch, m, pe = setup_single()
    go = m.host_alloc(64)
    goerr = [torch.zeros(4, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
    WARM, N, CH = 100, 1000, 8
    res = {"loop": {}, "graph": {}}
    for M in (1, 4):
        bufs = FloorBufs(torch, M)
        for mode, mname in ((0, "lamport"), (1, "flag")):
            for d in range(WORLD):  # eager warm, all 4 together
                with torch.cuda.device(d):
                    bufs.launch(m, mode, d, 10)
            sync_all(torch)
            # ---- in-kernel loop, behind the go flag
            ts = [torch.zeros(WARM + N + 8, dtype=torch.int64, device=f"cuda:{d}") for d in range(WORLD)]
            m.host_set(go, 0)
            for d in range(WORLD):
                with torch.cuda.device(d):
                    m.wait_go(go, goerr[d].data_ptr())
                    bufs.launch(m, mode, d, WARM + N, ts[d].data_ptr())
            time.sleep(0.002)
            m.host_set(go, 1)
            sync_all(torch)
            per_rank = []
            for d in range(WORLD):
                t = ts[d].tolist()
                nit = WARM + N
                c0, g0, c1, g1 = t[nit], t[nit + 1], t[nit + 2], t[nit + 3]
                cpn = (c1 - c0) / max(g1 - g0, 1)
                stamps = [c0] + t[:nit]
                dts = [(stamps[i + 1] - stamps[i]) / cpn / 1e3 for i in range(WARM, nit)]  # us
                per_rank.append({"med_us": round(pct(dts, 0.5), 3), "p10_us": round(pct(dts, 0.1), 3),
                                 "p90_us": round(pct(dts, 0.9), 3), "mean_us": round(sum(dts) / len(dts), 3),
                                 "sm_ghz": round(cpn, 3)})
            e_loop = bufs.check(torch)
            errs = [int(x[0].item()) for x in bufs.err]
            key = f"{mname}_M{M}"
            res["loop"][key] = {"med_us_max_rank": max(r["med_us"] for r in per_rank),
                                "p90_us_max_rank": max(r["p90_us"] for r in per_rank),
                                "per_rank": per_rank, "max_abs_err_vs_fp32_sum": e_loop, "kernel_err": errs}
            log(f"[floor:loop] {key} med(max rank) {res['loop'][key]['med_us_max_rank']} us "
                f"p90 {res['loop'][key]['p90_us_max_rank']} us err={errs} e={e_loop:.3g} "
                f"per-rank med {[r['med_us'] for r in per_rank]}")
            # ---- 8-launch graph chain per GPU, 125 replays = 1000 launches, no PDL
            graphs = []
            for d in range(WORLD):
                with torch.cuda.device(d):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(CH):
                            bufs.launch(m, mode, d, 1)
                    graphs.append(g)
            r = _time_graphs(torch, m, go, goerr, graphs, CH, N)
            r["max_abs_err_vs_fp32_sum"] = bufs.check(torch)
            r["kernel_err"] = [int(x[0].item()) for x in bufs.err]
            res["graph"][key] = r
            log(f"[floor:graph] {key} {json.dumps(r)}")
            del graphs
        graphs = []  # empty-kernel chain of the same grid
        for d in range(WORLD):
            with torch.cuda.device(d):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(CH):
                        m.empty_launch(M)
                graphs.append(g)
        res["graph"][f"empty_M{M}"] = _time_graphs(torch, m, go, goerr, graphs, CH, N)
        log(f"[floor:graph] empty_M{M} {json.dumps(res['graph'][f'empty_M{M}'])}")
        del graphs, bufs
    wd.cancel()
    ok = all(v.get("max_abs_err_vs_fp32_sum", 0) < 0.07 and not any(v.get("kernel_err", []))
             for k in ("loop", "graph") for v in res[k].values())
    return {"passed": ok, "part": "floor", "warmup_iters": WARM, "iters": N, "chain": CH,
            "goerr": [int(x[0].item()) for x in goerr], **res}


# ------------------------------------------------------------------------------------------------- (iv)
S_LP = 2048 * HIDDEN // 4  # 2,621,440 B: LP's FP8 per-peer chunk at 2048 rows
S_BIG = 10485760


class BwBufs:
    def __init__(self, torch):
        self.A = [torch.ones(S_BIG, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]  # 4 slots
        self.B = [torch.zeros(S_BIG, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]

    def a(self, d, slot):
        return self.A[d].data_ptr() + slot * S_BIG

    def b(self, d, slot):
        return self.B[d].data_ptr() + slot * S_BIG


def _time_one(torch, d, fn, warm=3, reps=5):
    with torch.cuda.device(d):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts), ts


def _alone_rows(torch, m, bufs, cls, kinds):
    per = {}
    for a in range(WORLD):
        for p in range(WORLD):
            if a == p:
                continue
            row = {"class": cls[f"{a}->{p}"]}
            for size, sname in ((S_LP, "2.6MB"), (S_BIG, "10.5MB")):
                for kind in kinds:
                    if kind.startswith("read"):  # SM on a pulls from p
                        _, ctas, un = kind.split("_")
                        fn = (lambda s=size, c=int(ctas[1:]), u=int(un[1:]): m.copy_launch(
                            bufs.a(p, a), bufs.b(a, p), s, c, u))
                    elif kind.startswith("write"):  # SM on a pushes to p
                        _, ctas = kind.split("_")
                        fn = (lambda s=size, c=int(ctas[1:]): m.copy_launch(bufs.a(a, p), bufs.b(p, a), s, c, 1))
                    elif kind == "ce_push":
                        fn = (lambda s=size: m.memcpy_peer(bufs.b(p, a), p, bufs.a(a, p), a, s))
                    else:  # ce_pull
                        fn = (lambda s=size: m.memcpy_peer(bufs.b(a, p), a, bufs.a(p, a), p, s))
                    us, _ = _time_one(torch, a, fn)
                    row[f"{kind}@{sname}_GBps"] = round(size / us / 1e3, 2)
            per[f"{a}->{p}"] = row
            log(f"[bw:alone] {a}->{p} ({row['class']}) " + " ".join(f"{k}={v}" for k, v in row.items() if k != "class"))
    keys = [k for k in next(iter(per.values())) if k != "class"]
    return {"per_pair": per, "by_class": by_class(per, cls, keys)}


def _concurrent(torch, m, bufs, kind, go, goerr, warm=3, reps=5):
    """all 4 ranks x 3 peers, 2.6 MB each, started together behind the host go flag; per-rank time."""
    streams = {d: [torch.cuda.Stream(device=d) for _ in range(3)] for d in range(WORLD)}
    times = {d: [] for d in range(WORLD)}
    for it in range(warm + reps):
        sync_all(torch)
        m.host_set(go, 0)
        evs = {}
        for d in range(WORLD):
            peers = [p for p in range(WORLD) if p != d]
            with torch.cuda.device(d):
                s0 = torch.cuda.current_stream(d)
                m.wait_go(go, goerr[d].data_ptr())
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(s0)
                if kind.startswith("read") or kind.startswith("write"):
                    ctas = int(kind.split("_")[1][1:])
                    if kind.startswith("read"):
                        src = [bufs.a(p, d) for p in peers]
                        dst = [bufs.b(d, p) for p in peers]
                    else:
                        src = [bufs.a(d, p) for p in peers]
                        dst = [bufs.b(p, d) for p in peers]
                    m.copy3_launch(src, dst, S_LP, ctas)
                else:
                    joins = []
                    for k, p in enumerate(peers):
                        st = streams[d][k]
                        st.wait_event(e0)
                        with torch.cuda.stream(st):
                            if kind == "ce_push":
                                m.memcpy_peer(bufs.b(p, d), p, bufs.a(d, p), d, S_LP)
                            else:
                                m.memcpy_peer(bufs.b(d, p), d, bufs.a(p, d), p, S_LP)
                        ej = torch.cuda.Event()
                        ej.record(st)
                        joins.append(ej)
                    for ej in joins:
                        s0.wait_event(ej)
                e1.record(s0)
                evs[d] = (e0, e1)
        time.sleep(0.001)
        m.host_set(go, 1)
        sync_all(torch)
        if it >= warm:
            for d in range(WORLD):
                times[d].append(evs[d][0].elapsed_time(evs[d][1]) * 1e3)
    per_rank_us = {d: statistics.median(v) for d, v in times.items()}
    worst = max(per_rank_us.values())
    return {"per_rank_us": {d: round(v, 1) for d, v in per_rank_us.items()},
            "per_rank_agg_GBps": {d: round(3 * S_LP / v / 1e3, 2) for d, v in per_rank_us.items()},
            "max_rank_us": round(worst, 1), "agg_GBps_at_max_rank": round(3 * S_LP / worst / 1e3, 2),
            "goerr": [int(x[0].item()) for x in goerr]}


def part_bw(which):
    wd = watchdog(f"bw{which}", 280)
    torch, m, pe = setup_single()
    topo = topo_classes(torch)
    cls = topo["pair_class"]
    bufs = BwBufs(torch)
    go = m.host_alloc(64)
    goerr = [torch.zeros(4, dtype=torch.int32, device=f"cuda:{d}") for d in range(WORLD)]
    if which == "read":
        kinds = ["read_c16_u1", "read_c64_u1", "read_c16_u4", "read_c64_u4"]
        ckinds = ["read_c16", "read_c64"]
    else:
        kinds = ["write_c16", "write_c64", "ce_push", "ce_pull"]
        ckinds = ["write_c16", "write_c64", "ce_push", "ce_pull"]
    alone = _alone_rows(torch, m, bufs, cls, kinds)
    conc = {}
    for k in ckinds:
        conc[k] = _concurrent(torch, m, bufs, k, go, goerr)
        log(f"[bw:concurrent] {k} (4 ranks x 3 peers x 2.6 MB; SM rows: CTAs per peer) {json.dumps(conc[k])}")
    wd.cancel()
    return {"passed": True, "part": f"bw{which}", "topo": topo, "bytes_per_peer": S_LP, "alone": alone,
            "concurrent": conc}


# ------------------------------------------------------------------------------------------------- (iii)
def rank_served(env):
    sys.path.insert(0, env["_here"])
    os.environ.update({k: v for k, v in env.items() if not k.startswith("_")})
    wdt = watchdog("served-rank", int(env.get("_wd", 700)))
    import pcie_custom_ar_bench as B
    torch, rank, mapping, _ = B._setup({})
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams, AllReduceStrategy
    dev = torch.device("cuda", rank)
    m = ext()
    MS = [1, 2, 4, 8]
    WARMUP, ITERS, SEQS, CH = 20, 100, 3, 8
    res = {"rank": rank, "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "TLLM_", "TRTLLM_"))},
           "timing": {}, "dispatch": {}, "check": {},
           "sm_clock_query": B.sh(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
                                   "--query-gpu=uuid,clocks.sm", "--format=csv,noheader"])
           if rank == 0 else None}
    AutoTuner.get().setup_distributed_state(mapping)
    ar_tune = AllReduce(mapping=mapping, strategy=AllReduceStrategy.LOWPRECISION, dtype=torch.bfloat16)
    with autotune():
        for mm in MS:
            ar_tune(torch.randn((mm, HIDDEN), dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize()
    if rank == 0:
        log("[served] autotune done")

    def build(ar, x, trig, kind):
        params = AllReduceParams(trigger_completion_at_end=trig)

        def body():
            outs = []
            for _ in range(CH):
                if kind in ("pred_ar", "pred"):
                    m.pdl_pred()
                if kind != "pred":
                    outs.append(ar(x, all_reduce_params=params))
            return outs
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
            with torch.cuda.graph(g, stream=s):
                outs = body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return g, outs

    def time_chain(g, x, mm):
        meds = []
        for _ in range(SEQS):
            x.copy_(B._gen(torch, mm, 0, rank, dev))  # seeded input re-created per timed sequence
            torch.cuda.synchronize()
            B._barrier()
            for _ in range(WARMUP):
                g.replay()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
            for i in range(ITERS):
                ev[i][0].record()
                g.replay()
                ev[i][1].record()
            torch.cuda.synchronize()
            ts = sorted(a.elapsed_time(b) * 1e3 / CH for a, b in ev)
            meds.append(ts[len(ts) // 2])
        med = sorted(meds)[SEQS // 2]
        return {"per_call_us": round(med, 3), "seq_us": [round(v, 3) for v in meds],
                "seq_spread": round((max(meds) - min(meds)) / med, 4)}

    for strat_name in ("LOWPRECISION", "ONESHOT"):
        strat = getattr(AllReduceStrategy, strat_name)
        for mm in MS:
            ar = AllReduce(mapping=mapping, strategy=strat, dtype=torch.bfloat16)
            x = torch.zeros((mm, HIDDEN), dtype=torch.bfloat16, device=dev)
            for trig in (True, False):
                for kind in ("ar", "pred_ar"):
                    key = f"{strat_name}|M{mm}|trig_end={trig}|{kind}"
                    g, outs = build(ar, x, trig, kind)
                    t = time_chain(g, x, mm)
                    sp = B._allgather(t["seq_spread"])
                    if max(sp) > 0.03:  # dispersion above the 3% floor on any rank: repeat once (all ranks)
                        t["repeat"] = time_chain(g, x, mm)
                    res["timing"][key] = t
                    # untimed check of every chain output vs the fp32 rank-order sum, fresh seed
                    x.copy_(B._gen(torch, mm, 77, rank, dev))
                    torch.cuda.synchronize()
                    B._barrier()
                    g.replay()
                    torch.cuda.synchronize()
                    ins = [B._gen(torch, mm, 77, r, dev).float() for r in range(WORLD)]
                    emul = (((ins[0] + ins[1]) + ins[2]) + ins[3]).to(torch.bfloat16)
                    res["check"][key] = {
                        "bitwise_vs_fp32_rank_order": bool(all(torch.equal(o.view(torch.int16), emul.view(torch.int16))
                                                              for o in outs)),
                        "max_abs": max((o.float() - emul.float()).abs().max().item() for o in outs)}
                    del g, outs
                    B._barrier()
            if strat_name == "LOWPRECISION":
                g, _ = build(ar, x, True, "pred")
                res["timing"][f"PRED_ONLY|M{mm}"] = time_chain(g, x, mm)
                del g
            if rank == 0:
                log(f"[served] {strat_name} M={mm} " + json.dumps(
                    {k: v["per_call_us"] for k, v in res["timing"].items()
                     if k.startswith(strat_name + f"|M{mm}|") or k == f"PRED_ONLY|M{mm}"}))
            del ar
            B._barrier()
        if env.get("_art"):  # partial per-rank receipt, in case a later step hangs
            with open(os.path.join(env["_art"], f"served_rank{rank}_partial.json"), "w") as f:
                json.dump(res, f, default=str)
    # dispatch: kernel names of eager calls per M (after all timing: first CUPTI use in this process)
    from torch.profiler import ProfilerActivity, profile
    for mm in MS:
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.LOWPRECISION, dtype=torch.bfloat16)
        x = B._gen(torch, mm, 5, rank, dev)
        torch.cuda.synchronize()
        B._barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for trig in (True, False):
                ar(x, all_reduce_params=AllReduceParams(trigger_completion_at_end=trig))
            torch.cuda.synchronize()
        names = sorted({e.name[:100] for e in p.events() if e.device_type.name == "CUDA"
                        and "Memcpy" not in e.name and "Memset" not in e.name})
        res["dispatch"][str(mm)] = {"kernels": names,
                                    "lamport": any("oneshot_lamport" in n for n in names),
                                    "nccl": any("nccl" in n.lower() for n in names)}
        B._barrier()
    wdt.cancel()
    B._barrier()
    return res


def part_served():
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    import pcie_custom_ar_bench as B
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(B)
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    env = {"_wd": "700", "_here": HERE, "_art": os.environ.get("AUTO_DEV_ARTIFACTS", "")}
    for k in ("PCIE_PROBE_EXT_DIR", "PCIE_PROBE_PREBUILT"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    t0 = time.time()
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        out = sorted(ex.map(rank_served, [env] * WORLD), key=lambda r: r["rank"])
    log(f"[served] ranks done in {time.time() - t0:.0f}s")
    table = {}
    for k in out[0]["timing"]:
        per = [r["timing"][k]["per_call_us"] for r in out]
        rep = [r["timing"][k].get("repeat", {}).get("per_call_us") for r in out]
        table[k] = {"max_rank_us": max(per), "per_rank_us": per,
                    "max_seq_spread": max(r["timing"][k]["seq_spread"] for r in out)}
        if any(v is not None for v in rep):
            table[k]["repeat_max_rank_us"] = max(v for v in rep if v is not None)
    derived = {}  # AR cost behind a PDL predecessor = (pred+AR chain) - (pred-only chain), per call, max over ranks
    for mm in (1, 2, 4, 8):
        po = table.get(f"PRED_ONLY|M{mm}", {}).get("max_rank_us")
        for trig in (True, False):
            k = f"LOWPRECISION|M{mm}|trig_end={trig}|pred_ar"
            if k in table and po is not None:
                derived[f"M{mm}|trig_end={trig}"] = {
                    "ar_chain_us": table[f"LOWPRECISION|M{mm}|trig_end={trig}|ar"]["max_rank_us"],
                    "pred_ar_chain_us": table[k]["max_rank_us"], "pred_only_us": po,
                    "ar_behind_pred_us": round(table[k]["max_rank_us"] - po, 3)}
    dispatch = {mm: {"lamport_all_ranks": all(r["dispatch"][mm]["lamport"] for r in out),
                     "nccl_any_rank": any(r["dispatch"][mm]["nccl"] for r in out),
                     "rank0_kernels": out[0]["dispatch"][mm]["kernels"]} for mm in out[0]["dispatch"]}
    checks = {k: {"bitwise_all_ranks": all(r["check"][k]["bitwise_vs_fp32_rank_order"] for r in out),
                  "max_abs": max(r["check"][k]["max_abs"] for r in out)} for k in out[0]["check"]}
    ok = all(d["lamport_all_ranks"] for d in dispatch.values())
    return {"passed": ok, "part": "served", "table_us": table, "derived": derived, "dispatch": dispatch,
            "checks": checks, "env_rank0": out[0]["env"], "sm_clock_query_rank0": out[0]["sm_clock_query"]}


def main():
    part = sys.argv[1]
    if part == "build":
        t0 = time.time()
        os.environ.pop("PCIE_PROBE_PREBUILT", None)
        ext()
        print(json.dumps({"passed": True, "part": "build", "seconds": round(time.time() - t0, 1)}), flush=True)
        return
    fn = {"lat": part_lat, "floor": part_floor, "bwread": lambda: part_bw("read"),
          "bwrest": lambda: part_bw("rest"), "served": part_served}[part]
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out = {"passed": False, "part": part, "error": repr(e)[:500]}
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        with open(os.path.join(art, f"probe_{part}.json"), "w") as f:
            json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
