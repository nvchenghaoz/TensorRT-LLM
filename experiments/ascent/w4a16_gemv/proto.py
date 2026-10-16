# Branch 63 STEP 0 prototype: cuda-core W4A16 NVFP4 GEMV for the (N,K)=(5120,6144) decode
# copies (GDN out_proj x48 + attn o_proj x16), timed against the base's Marlin
# (torch.ops.trtllm.marlin_nvfp4_gemm, as FP8QDQLinearMethod.apply calls it) in the same
# process. Cold L2: C=16 distinct copies per arm (16 x 17.7 MB = 283 MB per arm) replayed
# back to back in one CUDA graph; PDL on; interleaved arms; 10 warmup + 60 timed replays.
# Weights are built exactly like linear.py _build_fp8_nvfp4_decode_copy (FP8 per-tensor
# weight -> fp4_quantize with power-of-two global scale g).
# Modes: bench [variants] | ncu <variant>
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402


def log(*a):
    print(*a, flush=True)


MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
dev = torch.device("cuda")
N, K = 5120, 6144
C = 16
WARMUP, ITERS = 10, 60
MS = [1, 2, 3, 4]

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

__device__ __forceinline__ int4 ldw(uint8_t const* p) {
    int4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// 16 packed E2M1 bytes -> 32 floats (element 2b = low nibble of byte b).
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

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// W4A16 NVFP4 GEMV. Weight [N, K/2] packed E2M1 (fp4_quantize layout), weight SF linear
// [N, K/16] E4M3, activations BF16 [M, K], out BF16 [M, N] = (x @ dequant(W)^T) * g.
// CTA = ROWS output rows x all K; thread t owns K chunks (t + s*BLOCK)*32, s < KS
// (K == KS*BLOCK*32). All of the CTA's weight + SF loads are issued before the PDL wait.
template <int M, int ROWS, int BLOCK, int KS, bool EARLY>
__global__ void __launch_bounds__(BLOCK) w4a16_gemv(uint8_t const* __restrict__ w, uint8_t const* __restrict__ sf,
    __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float g, int n, int k) {
    int const tid = threadIdx.x;
    int const n0 = blockIdx.x * ROWS;
    int const kb = k / 2;    // weight row bytes
    int const ks = k / 16;   // SF row bytes
    int4 wq[ROWS][KS];
    uint32_t ws[ROWS][KS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int s = 0; s < KS; ++s)
            wq[r][s] = ldw(w + (size_t)(n0 + r) * kb + (tid + s * BLOCK) * 16);
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int s = 0; s < KS; ++s)
            ws[r][s] = __ldg(reinterpret_cast<uint16_t const*>(sf + (size_t)(n0 + r) * ks + (tid + s * BLOCK) * 2));
    pdl_wait();
    uint32_t xa[M][KS][16];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            int4 const* p = reinterpret_cast<int4 const*>(x + (size_t)m * k + (tid + s * BLOCK) * 32);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int4 v = p[q];
                xa[m][s][q * 4 + 0] = v.x;
                xa[m][s][q * 4 + 1] = v.y;
                xa[m][s][q * 4 + 2] = v.z;
                xa[m][s][q * 4 + 3] = v.w;
            }
        }
    float acc[M][ROWS];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            float wf[32];
            cvt_fp4x32(wq[r][s], wf);
            float const s0 = e4m3f(ws[r][s]);
            float const s1 = e4m3f(ws[r][s] >> 8);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float p0 = 0.f, p1 = 0.f;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    p0 = fmaf(bf_lo(xa[m][s][l]), wf[2 * l], p0);
                    p0 = fmaf(bf_hi(xa[m][s][l]), wf[2 * l + 1], p0);
                }
#pragma unroll
                for (int l = 8; l < 16; ++l) {
                    p1 = fmaf(bf_lo(xa[m][s][l]), wf[2 * l], p1);
                    p1 = fmaf(bf_hi(xa[m][s][l]), wf[2 * l + 1], p1);
                }
                acc[m][r] = fmaf(p0, s0, acc[m][r]);
                acc[m][r] = fmaf(p1, s1, acc[m][r]);
            }
        }
    }
    if constexpr (EARLY) pdl_trigger();
    constexpr int NW = BLOCK / 32;
    __shared__ float red[NW][M * ROWS];
    int const lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = acc[m][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][m * ROWS + r] = v;
        }
    __syncthreads();
    for (int i = tid; i < M * ROWS; i += BLOCK) {
        float v = 0.f;
#pragma unroll
        for (int j = 0; j < NW; ++j) v += red[j][i];
        int const m = i / ROWS, r = i % ROWS;
        out[(size_t)m * n + n0 + r] = __float2bfloat16(v * g);
    }
    if constexpr (!EARLY) pdl_trigger();
}

template <typename Kern>
void pdl_launch(Kern kern, dim3 grid, dim3 block, uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x,
    __nv_bfloat16* o, float g, int n, int k) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w, sf, x, o, g, n, k));
}

template <int M, int ROWS, int BLOCK, int KS, bool EARLY>
bool lv(uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x, __nv_bfloat16* o, float g, int n, int k) {
    if (k != KS * BLOCK * 32 || n % ROWS != 0) return false;
    pdl_launch(w4a16_gemv<M, ROWS, BLOCK, KS, EARLY>, dim3(n / ROWS), dim3(BLOCK), w, sf, x, o, g, n, k);
    return true;
}

template <int ROWS, int BLOCK, int KS, bool EARLY>
bool bym(int m, uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x, __nv_bfloat16* o, float g, int n, int k) {
    switch (m) {
    case 1: return lv<1, ROWS, BLOCK, KS, EARLY>(w, sf, x, o, g, n, k);
    case 2: return lv<2, ROWS, BLOCK, KS, EARLY>(w, sf, x, o, g, n, k);
    case 3: return lv<3, ROWS, BLOCK, KS, EARLY>(w, sf, x, o, g, n, k);
    case 4: return lv<4, ROWS, BLOCK, KS, EARLY>(w, sf, x, o, g, n, k);
    }
    return false;
}

bool dispatch(int vid, int m, uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x, __nv_bfloat16* o,
    float g, int n, int k) {
    switch (vid) {
    case 1: return bym<4, 192, 1, false>(m, w, sf, x, o, g, n, k);
    case 2: return bym<8, 192, 1, false>(m, w, sf, x, o, g, n, k);
    case 3: return bym<16, 192, 1, false>(m, w, sf, x, o, g, n, k);
    case 4: return bym<4, 96, 2, false>(m, w, sf, x, o, g, n, k);
    case 5: return bym<8, 96, 2, false>(m, w, sf, x, o, g, n, k);
    case 6: return bym<8, 192, 1, true>(m, w, sf, x, o, g, n, k);
    case 7: return bym<4, 192, 1, true>(m, w, sf, x, o, g, n, k);
    case 8: return bym<2, 192, 1, false>(m, w, sf, x, o, g, n, k);
    case 9: return bym<16, 96, 2, false>(m, w, sf, x, o, g, n, k);
    }
    return false;
}

}  // namespace

torch::Tensor run(int64_t vid, torch::Tensor x, torch::Tensor w, torch::Tensor sf, double g) {
    int m = x.size(0), k = x.size(1), n = w.size(0);
    auto out = torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
    bool ok = dispatch(static_cast<int>(vid), m, w.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<float>(g), n, k);
    TORCH_CHECK(ok, "variant not dispatched");
    return out;
}
"""

CPP = "torch::Tensor run(int64_t vid, torch::Tensor x, torch::Tensor w, torch::Tensor sf, double g);"

VIDS = {"r4b192": 1, "r8b192": 2, "r16b192": 3, "r4b96": 4, "r8b96": 5, "r8b192e": 6, "r4b192e": 7,
        "r2b192": 8, "r16b96": 9}


def build_ext():
    from torch.utils.cpp_extension import load_inline
    bd = "/tmp/b63_w4a16_proto_build"
    os.makedirs(bd, exist_ok=True)
    log("[proto] compiling extension ...")
    ext = load_inline(name="b63_w4a16_proto", cpp_sources=CPP, cuda_sources=CU, functions=["run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-std=c++20",
                                         "-lineinfo", "-Xptxas=-v"],
                      build_directory=bd, verbose=os.environ.get("PROTO_VERBOSE", "0") == "1")
    log("[proto] extension ready")
    return ext


def make_copy(seed):
    """FP8 per-tensor weight -> NVFP4 exactly as _build_fp8_nvfp4_decode_copy, plus both layouts."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(N, K, generator=gen, device=dev) * 0.02
    w = w * (torch.rand(N, 1, generator=gen, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    w_vals = w8.to(torch.bfloat16)
    amax = torch.clamp(w_vals.abs().amax().float() * s, min=torch.finfo(torch.float32).tiny)
    g = torch.exp2(torch.ceil(torch.log2(amax / (448.0 * 6.0))))
    qg = (s / g).reshape(1).to(torch.float32)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w_vals, qg, 16, False, True)
    del w_vals, w8
    mw, msc, mg = _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, g.reshape(1).to(torch.float32), size_n=N,
                                                size_k=K, group_size=16)
    sf_lin = torch.ops.trtllm.block_scale_interleave_reverse(w_sf.view(N, -1))[:, :K // 16].contiguous()
    return dict(wq=w_fp4.view(torch.uint8).contiguous(), sf=sf_lin.view(torch.uint8).contiguous(),
                g=float(g.item()), mw=mw, msc=msc, mg=mg)


def dequant(cp):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev,
                       dtype=torch.float64)
    q = cp["wq"]
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], dim=-1).reshape(N, K)
    s = cp["sf"].view(torch.float8_e4m3fn).double()
    return v * s.repeat_interleave(16, dim=1) * cp["g"]


def marlin(x, cp):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, cp["mw"], scale_a=None, scale_b=cp["msc"], alpha=None,
                                              weight_global_scale=cp["mg"], bias=None, out_dtype=torch.bfloat16,
                                              size_n=N, size_k=K, workspace_mode=0)


def stats(ts):
    s = sorted(ts)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return q(0.5), q(0.9), q(0.1)


def bf16_ulp(v):
    # spacing of BF16 at |v|
    e = torch.floor(torch.log2(torch.tensor(abs(v)).clamp(min=1e-30)))
    return float(2.0 ** (e - 7))


def check(ext, copies, variants):
    res = {}
    cp = copies[0]
    wdq = dequant(cp)
    for m in MS:
        gen = torch.Generator(device=dev).manual_seed(77 + m)
        x = (torch.randn(m, K, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        x[:, 5::97] *= 30.0
        ref = x.double() @ wdq.t()
        mo = marlin(x, cp)
        me = (mo.double() - ref).abs()
        ulp = bf16_ulp(ref.abs().max().item())
        rec = dict(marlin_max=me.max().item(), marlin_mean=me.mean().item(), ulp_at_max=ulp)
        for vn in variants:
            o = ext.run(VIDS[vn], x, cp["wq"], cp["sf"], cp["g"])
            e = (o.double() - ref).abs()
            rec[vn] = dict(max=e.max().item(), mean=e.mean().item(),
                           ok=bool(e.max().item() <= me.max().item() + ulp and e.mean().item() <= me.mean().item() + ulp),
                           eq_marlin_frac=float((o == mo).float().mean().item()))
        torch.cuda.synchronize()
        res[f"M{m}"] = rec
        log(f"[proto] check M={m}: " + json.dumps(rec))
    del wdq
    return res


def capture(fn, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:2]:
            fn(*c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        outs = [fn(*c) for c in calls]
    torch.cuda.synchronize()
    return gr, outs


def bench(variants):
    ext = build_ext()
    log("[proto] building copies ...")
    copies = [make_copy(1000 + i) for i in range(C)]
    wbytes = N * K // 2 + N * K // 16
    log(f"[proto] {C} copies x {wbytes / 1e6:.2f} MB per arm; mem {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    chk = check(ext, copies, variants)
    out = {}
    for m in MS:
        gen = torch.Generator(device=dev).manual_seed(5 + m)
        x = (torch.randn(m, K, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        graphs = {}
        graphs["marlin"], _ = capture(lambda cp: marlin(x, cp), [(cp,) for cp in copies])
        for vn in variants:
            vid = VIDS[vn]
            graphs[vn], _ = capture(lambda cp, vid=vid: ext.run(vid, x, cp["wq"], cp["sf"], cp["g"]),
                                    [(cp,) for cp in copies])
        for _ in range(WARMUP):
            for gr in graphs.values():
                gr.replay()
        torch.cuda.synchronize()
        ts = {vn: [] for vn in graphs}
        for _ in range(ITERS):
            for vn, gr in graphs.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                gr.replay()
                e1.record()
                e1.synchronize()
                ts[vn].append(e0.elapsed_time(e1) / C * 1e3)
        rec = {}
        base = stats(ts["marlin"])[0]
        for vn, t in ts.items():
            med, p90, p10 = stats(t)
            rec[vn] = dict(us=round(med, 2), p90=round(p90, 2), nf=round((p90 - p10) / med, 4),
                           vs_marlin=round(med / base, 4), TBps=round(wbytes / (med * 1e-6) / 1e12, 3))
        out[f"M{m}"] = rec
        log(f"[proto] M={m}: " + json.dumps(rec))
        del graphs
        torch.cuda.synchronize()
    log(json.dumps(dict(timing=out, check=chk)))


def ncu_mode(vn):
    ext = build_ext()
    cp = make_copy(1000)
    for m in MS:
        x = torch.randn(m, K, device=dev).to(torch.bfloat16)
        marlin(x, cp)
        ext.run(VIDS[vn], x, cp["wq"], cp["sf"], cp["g"])
        torch.cuda.synchronize()
    log("[proto] ncu done")


if MODE == "ncu":
    ncu_mode(sys.argv[2])
else:
    vs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VIDS)
    bench(vs)
