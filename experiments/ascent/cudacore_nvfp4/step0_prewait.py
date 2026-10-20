# Idea 200 / branch 114 STEP 0: cudaCoreGemmFp4 TILE_M=1 with the first K step's (constant) weight +
# weight-SF loads issued BEFORE griddepcontrol.wait, timed IN-CHAIN behind the real producers.
#   D: x BF16 [M, 34816] -> fused_silu_mul_nvfp4 (triggers at start) -> down (5120, 17408)
#   Q: x BF16 [M, 5120]  -> fp4_quantize (quantize_with_block_size, triggers at end) -> gate_up (34816, 5120)
#   E: x BF16 [M, 10240] -> fused_silu_mul_nvfp4 -> gate_up (stand-in for an early-trigger producer)
# Arms: main (torch.ops.trtllm.cuda_core_nvfp4_gemm, act_sf_swizzled=True), main_aa (A/A copy),
#       plain / peeled / branch (load_inline copies of main's kernel; MODE 0 = main's TILE_M=1 loop,
#       1 = first step peeled with pre-wait loads (fact 324), 2 = pre-wait first-step loads + branch in loop),
#       prod (producer chain only), gemm_only (main's GEMM chain on precomputed activations).
# One CUDA graph per arm holding L layers (distinct weight copies, >= 600 MB per shape), PDL on,
# 512 MiB READ flush (int64 sum) before every replay outside the events, arm order rotated per iteration.
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["TRTLLM_ENABLE_PDL"] = "1"

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as _fsm  # noqa: E402,F401


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
L = int(os.environ.get("S0_LAYERS", "12"))
WARMUP = 10
ITERS = int(os.environ.get("S0_ITERS", "100"))
ROUNDS = int(os.environ.get("S0_ROUNDS", "3"))
MS = [int(x) for x in os.environ.get("S0_MS", "1,2").split(",")]
FP8_MAX, E2M1_MAX = 448.0, 6.0
FLUSH_BYTES = 512 << 20

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ int swz(int row, int col, int ntk) {
    int const tile_offset = ((row / 128) * ntk + col / 4) * 512;
    return tile_offset + (row % 32) * 16 + ((row % 128) / 32) * 4 + col % 4;
}

__device__ __forceinline__ void cvt_fp4x32(int4 const& q, float* out) {
    uint32_t const* w = reinterpret_cast<uint32_t const*>(&q);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t h[4];
        asm volatile(
            "{\n.reg .b8 b0, b1, b2, b3;\n"
            "mov.b32 {b0, b1, b2, b3}, %4;\n"
            "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
            "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
            "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
            "cvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
            : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(w[i]));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(reinterpret_cast<__half2 const&>(h[j]));
            out[i * 8 + j * 2] = f.x;
            out[i * 8 + j * 2 + 1] = f.y;
        }
    }
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
    __nv_fp8_e4m3 x;
    x.__x = static_cast<__nv_fp8_storage_t>(b & 0xff);
    return static_cast<float>(x);
}

__device__ __forceinline__ int4 ldw(uint8_t const* p) {
    int4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <int TILE_M, int TILE_N, int BLOCK>
__device__ __forceinline__ void reduce_store(float* acc, __nv_bfloat16* __restrict__ out, int n) {
    typedef cub::WarpReduce<float> WarpReduce;
    constexpr int warp_num = BLOCK / 32;
    int const tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    __shared__ float shmem[TILE_M * TILE_N * warp_num];
    __shared__ typename WarpReduce::TempStorage temp_storage[warp_num];
#pragma unroll
    for (int mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < TILE_N; ++ni) {
            float val = WarpReduce(temp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
            if (lane_id == 0) shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
        }
    }
    __syncthreads();
    for (int ii = tid; ii < TILE_M * TILE_N; ii += BLOCK) {
        int mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (int jj = 0; jj < warp_num; ++jj) val += shmem[jj * TILE_M * TILE_N + ii];
        out[mid * n + nid] = __float2bfloat16(val);
    }
}

// Copy of main's cudaCoreGemmImpl (TILE_N=2, BLOCK=128, act SF swizzled, no __launch_bounds__ like main).
// MODE 0: wait, then plain loop (main at TILE_M=1). MODE 1: first step peeled, its weight loads before the
// wait (main at TILE_M>=2). MODE 2: first step's weight loads before the wait, branch in the loop.
template <int TILE_M, int MODE>
__global__ void gemv(uint8_t const* __restrict__ act, uint8_t const* __restrict__ weight,
    uint8_t const* __restrict__ sfa, uint8_t const* __restrict__ sfw, float const* alpha_ptr,
    __nv_bfloat16* __restrict__ out, int m, int n, int k) {
    constexpr int TILE_N = 2, BLOCK = 128;
    float const alpha = alpha_ptr[0];
    constexpr int tile_k = 32 * BLOCK;
    int const tile_m = blockIdx.y * TILE_M, tile_n = blockIdx.x * TILE_N, tid = threadIdx.x;
    float acc[TILE_M * TILE_N];
#pragma unroll
    for (int i = 0; i < TILE_M * TILE_N; ++i) acc[i] = 0;
    act += tile_m * k / 2;
    weight += tile_n * k / 2;
    out += tile_m * n + tile_n;
    int const ntk = (k / 16 + 3) / 4;
    int4 wq[TILE_N];
    uint32_t wsf[TILE_N];
    float tw[TILE_N * 32], tws[TILE_N * 2], ta[32], tas[2];
    auto load_w = [&](int idx_k) {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j) wq[j] = ldw(weight + (j * k + idx_k) / 2);
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
            wsf[j] = *reinterpret_cast<uint16_t const*>(sfw + swz(tile_n + j, idx_k / 16, ntk));
    };
    auto cvt_w = [&]() {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j) cvt_fp4x32(wq[j], tw + j * 32);
#pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            tws[j * 2] = e4m3f(wsf[j]);
            tws[j * 2 + 1] = e4m3f(wsf[j] >> 8);
        }
    };
    auto fma_step = [&](int idx_k) {
#pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int4 aq = *reinterpret_cast<int4 const*>(act + (i * k + idx_k) / 2);
            cvt_fp4x32(aq, ta);
            uint32_t s = *reinterpret_cast<uint16_t const*>(sfa + swz(tile_m + i, idx_k / 16, ntk));
            tas[0] = e4m3f(s);
            tas[1] = e4m3f(s >> 8);
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
#pragma unroll
                for (int l = 0; l < 32; ++l)
                    acc[i * TILE_N + j] = fmaf(alpha * ta[l] * tas[l / 16], tw[j * 32 + l] * tws[j * 2 + l / 16],
                        acc[i * TILE_N + j]);
        }
    };
    int const first = tid * 32;
    if constexpr (MODE == 0) {
        pdl_wait();
        for (int idx_k = first; idx_k < k; idx_k += tile_k) {
            load_w(idx_k);
            cvt_w();
            fma_step(idx_k);
        }
    } else if constexpr (MODE == 1) {
        if (first < k) load_w(first);
        pdl_wait();
        if (first < k) {
            cvt_w();
            fma_step(first);
        }
        for (int idx_k = first + tile_k; idx_k < k; idx_k += tile_k) {
            load_w(idx_k);
            cvt_w();
            fma_step(idx_k);
        }
    } else {
        if (first < k) load_w(first);
        pdl_wait();
        for (int idx_k = first; idx_k < k; idx_k += tile_k) {
            if (idx_k != first) load_w(idx_k);
            cvt_w();
            fma_step(idx_k);
        }
    }
    reduce_store<TILE_M, TILE_N, BLOCK>(acc, out, n);
    pdl_trigger();
}

struct P {
    uint8_t const* a; uint8_t const* w; uint8_t const* sa; uint8_t const* sw;
    float const* alpha; __nv_bfloat16* o; int m, n, k;
};

template <typename Kern>
void pdl_launch(Kern kern, dim3 grid, P const& p) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p.a, p.w, p.sa, p.sw, p.alpha, p.o, p.m, p.n, p.k));
}

template <int TM>
bool by_mode(int mode, P const& p) {
    dim3 grid(p.n / 2, p.m / TM);
    switch (mode) {
    case 0: pdl_launch(gemv<TM, 0>, grid, p); return true;
    case 1: pdl_launch(gemv<TM, 1>, grid, p); return true;
    case 2: pdl_launch(gemv<TM, 2>, grid, p); return true;
    }
    return false;
}

template <typename K>
void attrs(K kern, std::vector<int64_t>& v) {
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kern));
    v.push_back(a.numRegs);
    v.push_back(static_cast<int64_t>(a.localSizeBytes));
}

}  // namespace

torch::Tensor run(int64_t mode, torch::Tensor aq, torch::Tensor wq, torch::Tensor asf, torch::Tensor wsf,
    torch::Tensor alpha) {
    int m = aq.size(0), n = wq.size(0), k = aq.size(1) * 2;
    auto out = torch::empty({m, n}, aq.options().dtype(torch::kBFloat16));
    P p{aq.data_ptr<uint8_t>(), wq.data_ptr<uint8_t>(), asf.data_ptr<uint8_t>(), wsf.data_ptr<uint8_t>(),
        alpha.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, n, k};
    bool ok = false;
    if (m == 1) ok = by_mode<1>(static_cast<int>(mode), p);
    else if (m == 2) ok = by_mode<2>(static_cast<int>(mode), p);
    TORCH_CHECK(ok, "variant not dispatched");
    return out;
}

std::vector<int64_t> regs() {
    std::vector<int64_t> v;
    attrs(gemv<1, 0>, v); attrs(gemv<1, 1>, v); attrs(gemv<1, 2>, v);
    attrs(gemv<2, 0>, v); attrs(gemv<2, 1>, v); attrs(gemv<2, 2>, v);
    return v;
}
"""

CPP = ("torch::Tensor run(int64_t mode, torch::Tensor aq, torch::Tensor wq, torch::Tensor asf, "
       "torch::Tensor wsf, torch::Tensor alpha);\nstd::vector<int64_t> regs();")

MODES = {"plain": 0, "peeled": 1, "branch": 2}
ARMS = ["main", "main_aa", "plain", "peeled", "branch", "prod", "gemm_only"]
CHAINS = {
    # name: (producer, x width, weight shape name)
    "D": ("silu", 34816, "down"),
    "Q": ("quant", 5120, "gate_up"),
    "E": ("silu", 10240, "gate_up"),
}
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}


def tree_info():
    try:
        wt = os.environ.get("S0_WORKTREE", os.getcwd())
        h = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD", "HEAD^{tree}"], capture_output=True,
                           text=True, timeout=20).stdout.split()
        d = subprocess.run(["git", "-C", wt, "status", "--porcelain"], capture_output=True, text=True,
                           timeout=20).stdout.strip().splitlines()
        log(f"TREE_COMMIT={h[0] if h else '?'} TREE={h[1] if len(h) > 1 else '?'} DIRTY={len(d)}")
    except Exception as e:  # noqa: BLE001
        log(f"TREE_COMMIT=? ({e!r})")


def build_ext():
    from torch.utils.cpp_extension import load_inline
    bd = "/tmp/b114_step0_build"
    os.makedirs(bd, exist_ok=True)
    log("[s0] compiling extension (ptxas -v below) ...")
    ext = load_inline(name="b114_step0", cpp_sources=CPP, cuda_sources=CU, functions=["run", "regs"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-std=c++20",
                                         "-Xptxas=-v"],
                      build_directory=bd, verbose=True)
    log("[s0] extension ready")
    return ext


def make_weight(n, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = torch.randn(n, k, device=dev, generator=g).to(torch.bfloat16)
    w.view(n, k // 16, 16)[::5, ::11, :] = 0
    w.view(n, k // 16, 16)[1::97, 2::13, :] *= 40.0
    w.view(n, k // 16, 16)[2::7, 5::9, 3] = -0.0
    gs = (FP8_MAX * E2M1_MAX / w.float().abs().amax()).reshape(1).float()
    wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return wq, wsf, gs


def make_x(m, width, seed, silu):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = (torch.randn(m, width, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    if silu:
        d = width // 2
        a, b = x[:, :d].float(), x[:, d:].float()
        h = (a * torch.sigmoid(a) * b).to(torch.bfloat16).float()
    else:
        x.view(m, width // 16, 16)[:, 3::7, :] = 0
        x.view(m, width // 16, 16)[:, 4::31, :] *= 50.0
        h = x.float()
    gs = (FP8_MAX * E2M1_MAX / h.abs().amax().clamp_min(1e-6)).reshape(1).float()
    return x, gs


def produce(kind, x, gs):
    if kind == "silu":
        return torch.ops.trtllm.fused_silu_mul_nvfp4(x, gs)
    return torch.ops.trtllm.fp4_quantize(x, gs, 16, False, True)


def main_gemm(aq, wq, asf, wsf, alpha):
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, asf, wsf, alpha, None, torch.bfloat16,
                                                 act_sf_swizzled=True)


def make_fn(ext, arm, kind, layers, pre):
    def f():
        outs = []
        for i, (x, gs, wq, wsf, alpha) in enumerate(layers):
            if arm == "gemm_only":
                aq, asf = pre[i]
                outs.append(main_gemm(aq, wq, asf, wsf, alpha))
                continue
            aq, asf = produce(kind, x, gs)
            if arm == "prod":
                outs.append(aq)
            elif arm in ("main", "main_aa"):
                outs.append(main_gemm(aq, wq, asf, wsf, alpha))
            else:
                outs.append(ext.run(MODES[arm], aq, wq, asf, wsf, alpha))
        return outs
    return f


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = fn()
    torch.cuda.synchronize()
    return g, outs


def q(s, p):
    s = sorted(s)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def summ(ts):
    return {"med": round(q(ts, 0.5), 3), "p10": round(q(ts, 0.1), 3), "p90": round(q(ts, 0.9), 3)}


def main():
    tree_info()
    log(f"[s0] L={L} WARMUP={WARMUP} ITERS={ITERS} ROUNDS={ROUNDS} MS={MS} PDL={os.environ['TRTLLM_ENABLE_PDL']}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                              "clocks_throttle_reasons.active", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        log(f"[s0] gpu: {smi}")
    except Exception:  # noqa: BLE001
        pass
    ext = build_ext()
    r = list(ext.regs())
    reg_tab = {}
    for idx, (tm, mode) in enumerate([(1, "plain"), (1, "peeled"), (1, "branch"), (2, "plain"), (2, "peeled"),
                                      (2, "branch")]):
        reg_tab[f"M{tm}/{mode}"] = {"regs": r[2 * idx], "local_bytes": r[2 * idx + 1]}
    log("[s0] regs " + json.dumps(reg_tab))

    t0 = time.time()
    weights = {}
    for name, (n, k) in SHAPES.items():
        wq, wsf, gsw = make_weight(n, k, seed=17 if name == "gate_up" else 29)
        copies = [(wq, wsf)] + [(wq.clone(), wsf.clone()) for _ in range(L - 1)]
        wbytes = wq.numel() * wq.element_size() + wsf.numel() * wsf.element_size()
        weights[name] = dict(copies=copies, gsw=gsw, wbytes=wbytes, n=n, k=k)
        log(f"[s0] {name}: {L} copies x {wbytes / 1e6:.2f} MB = {L * wbytes / 1e6:.0f} MB ({time.time() - t0:.0f}s)")
    flush_buf = torch.randint(0, 1 << 30, (FLUSH_BYTES // 8,), device=dev, dtype=torch.int64)
    torch.cuda.synchronize()

    def flush():
        return flush_buf.sum()

    results = {}
    verdict = {}
    for m in MS:
        for cname, (kind, width, wname) in CHAINS.items():
            wd = weights[wname]
            layers, pre = [], []
            for i in range(L):
                x, gs = make_x(m, width, seed=1000 * m + 37 * i + (0 if cname == "D" else 500 if cname == "Q" else 900),
                               silu=(kind == "silu"))
                alpha = (1.0 / (gs * wd["gsw"])).float()
                wq, wsf = wd["copies"][i]
                layers.append((x, gs, wq, wsf, alpha))
                pre.append(tuple(t.clone() for t in produce(kind, x, gs)))
            torch.cuda.synchronize()
            # eager reference (main's op) and eager variants
            eager_main = [main_gemm(pre[i][0], layers[i][2], pre[i][1], layers[i][3], layers[i][4]).clone()
                          for i in range(L)]
            eager_mm = {}
            for arm in MODES:
                eager_mm[arm] = sum(int(not torch.equal(ext.run(MODES[arm], pre[i][0], layers[i][2], pre[i][1],
                                                                layers[i][3], layers[i][4]), eager_main[i]))
                                    for i in range(L))
            nf = sum(int(not torch.isfinite(o.float()).all()) for o in eager_main)
            ts = {a: [] for a in ARMS}
            mism = {a: 0 for a in ARMS if a not in ("prod",)}
            prod_mism = [0]
            for rnd in range(ROUNDS):
                graphs, outs = {}, {}
                for arm in ARMS:
                    graphs[arm], outs[arm] = capture(make_fn(ext, arm, kind, layers, pre))
                for it in range(WARMUP + ITERS):
                    sh = (it + rnd) % len(ARMS)
                    order = ARMS[sh:] + ARMS[:sh]
                    evs = []
                    for arm in order:
                        flush()
                        e0 = torch.cuda.Event(enable_timing=True)
                        e1 = torch.cuda.Event(enable_timing=True)
                        e0.record()
                        graphs[arm].replay()
                        e1.record()
                        evs.append((arm, e0, e1))
                    torch.cuda.synchronize()
                    if it >= WARMUP:
                        for arm, e0, e1 in evs:
                            ts[arm].append(e0.elapsed_time(e1) * 1e3 / L)
                # bitwise vs eager main after all replays
                for arm in mism:
                    mism[arm] += sum(int(not torch.equal(outs[arm][i], eager_main[i])) for i in range(L))
                prod_mism[0] += sum(int(not torch.equal(outs["prod"][i], pre[i][0])) for i in range(L))
                log(f"[s0] M={m} chain {cname} round {rnd}: " + json.dumps(
                    {a: round(q(ts[a][-ITERS:], 0.5), 3) for a in ARMS}) +
                    f" mism={mism} prod_mism={prod_mism[0]} ({time.time() - t0:.0f}s)")
                del graphs, outs
                torch.cuda.synchronize()
            rec = {a: summ(ts[a]) for a in ARMS}
            for a in ("main", "main_aa", "plain", "peeled", "branch", "gemm_only"):
                rec[a]["TBps"] = round(wd["wbytes"] / (rec[a]["med"] * 1e-6) / 1e12, 4)
            # paired differences (same iteration): gain = plain - arm (positive = faster than plain)
            pair = {}
            for a in ("main", "main_aa", "peeled", "branch", "gemm_only"):
                d = [p - x for p, x in zip(ts["plain"], ts[a])]
                pair[f"plain_minus_{a}"] = summ(d)
            d_aa = [x - y for x, y in zip(ts["main"], ts["main_aa"])]
            pair["main_minus_main_aa"] = summ(d_aa)
            d_mp = [x - y for x, y in zip(ts["main"], ts["plain"])]
            pair["main_minus_plain"] = summ(d_mp)
            rec["paired"] = pair
            rec["bitwise_mismatch"] = {"graph": mism, "eager_variants": eager_mm, "prod_vs_eager_fp4": prod_mism[0],
                                       "main_nonfinite": nf}
            results[f"M{m}/{cname}"] = rec
            log(f"[s0] RESULT M={m} {cname}: " + json.dumps(rec))
            del layers, pre, eager_main
            torch.cuda.empty_cache()
        if m == 1:
            def best(cn):
                r_ = results[f"M1/{cn}"]["paired"]
                b = max(("peeled", "branch"), key=lambda a: r_[f"plain_minus_{a}"]["med"])
                return b, r_[f"plain_minus_{b}"], r_["main_minus_main_aa"]
            bd, gd, aad = best("D")
            bq, gq, aaq = best("Q")
            be, ge, aae = best("E")
            verdict = {
                "DOWN_GAIN_M1": {"best": bd, "gain_us": gd, "aa_us": aad,
                                 "peeled": results["M1/D"]["paired"]["plain_minus_peeled"],
                                 "branch": results["M1/D"]["paired"]["plain_minus_branch"]},
                "GATE_UP_NO_OVERLAP_COST_M1": {
                    "peeled_cost_us": results["M1/Q"]["paired"]["plain_minus_peeled"],
                    "branch_cost_us": results["M1/Q"]["paired"]["plain_minus_branch"], "aa_us": aaq,
                    "note": "values are plain - arm; negative = the pre-wait arm is slower"},
                "GATE_UP_EARLY_PRODUCER_GAIN_M1": {"best": be, "gain_us": ge, "aa_us": aae,
                                                   "peeled": results["M1/E"]["paired"]["plain_minus_peeled"],
                                                   "branch": results["M1/E"]["paired"]["plain_minus_branch"]},
            }
            go = gd["med"] >= 0.5 + abs(aad["med"]) and gd["p10"] > 0
            verdict["GO"] = bool(go)
            log("[s0] VERDICT " + json.dumps(verdict))
    allbit = all(v == 0 for r_ in results.values() for v in r_["bitwise_mismatch"]["graph"].values()) and all(
        v == 0 for r_ in results.values() for v in r_["bitwise_mismatch"]["eager_variants"].values())
    log(json.dumps({"passed": bool(allbit), "regs": reg_tab, "verdict": verdict,
                    "results": {k: {a: v[a] for a in ("main", "main_aa", "plain", "peeled", "branch", "prod",
                                                      "gemm_only")} for k, v in results.items()}}))


main()
