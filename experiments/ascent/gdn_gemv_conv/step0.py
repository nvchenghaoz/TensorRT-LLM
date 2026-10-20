# Idea 198 / branch 118 STEP 0: fork-window gate for folding causal_conv1d_update into the
# in_proj_qkvz nvfp4_w4a16_gemv epilogue (GDN decode, rows <= 3, SM120).
#
# The fused kernel is a load_inline copy of the production nvfp4W4a16GemvKernel
# (cpp/tensorrt_llm/kernels/marlin/marlin_repack.cu) with a CONV template flag: each CTA owns
# output columns n0, n0+1 for all M rows; for columns < conv_dim it convolves them with
# conv_state[slot[m], n, 0..2] exactly like causal_conv1d_update_kernel_sl1 and shifts the state.
#
# Chain per layer (48 layers, every layer owns distinct cold weights), CUDA graphs, PDL on:
#   gemma fused add+RMSNorm (PDL) -> ev0 -> in_proj_qkvz on the current stream
#   || aux: ev0.wait, in_proj_ba (small-N GEMV, 96x5120), ev1 -> ev1.wait
#   -> [causal_conv1d_update] -> T1 delta rule (production fused_sigmoid_gating_delta_rule_update)
# Arms (one process, 2 captured graphs for base and fused, rotated order, 3 rounds):
#   base / base2   : production trtllm::nvfp4_w4a16_gemv + production causal_conv1d_update
#   fused / fused2 : inline GEMV with conv epilogue (no conv launch)
#   inl            : inline GEMV CONV=false + production conv (load_inline compile control)
#   ceil           : production GEMV, NO conv (negative control = conv slot ceiling; wrong math)
# Every replay: restore x/r, conv pools, ssm pools from pristine copies, then a 256 MiB L2 read
# flush, all outside the timed event pair.
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update  # noqa: E402
from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


for word in sys.argv[1:]:
    print(word, flush=True)
try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=30).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           timeout=30).stdout.strip().splitlines()
    print("TREE_COMMIT=" + (head or "unknown") + " dirty_files=" + str(len(dirty)), flush=True)
except Exception as exc:  # noqa: BLE001
    print("TREE_COMMIT=unknown " + repr(exc), flush=True)


def smi(tag):
    try:
        q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        log("nvidia-smi %s: %s" % (tag, r))
    except Exception as e:  # noqa: BLE001
        log("nvidia-smi unavailable: %r" % (e,))


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

constexpr int kGemvRows = 2;

struct ConvEpilogue
{
    __nv_bfloat16* state;
    __nv_bfloat16 const* weight;
    int const* indices;
    int64_t state_batch_stride;
    int64_t state_c_stride;
    int64_t state_l_stride;
    int64_t weight_c_stride;
    int64_t weight_l_stride;
    int conv_dim;
    int pad_slot_id;
};

template <int M, int BLOCK, bool CONV>
__global__ void __launch_bounds__(BLOCK) gemvKernel(uint8_t const* __restrict__ w,
    uint8_t const* __restrict__ sf, __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float g,
    int n, int k, ConvEpilogue ce)
{
    static_assert(M * kGemvRows <= BLOCK, "one epilogue element per thread");
    int const tid = threadIdx.x;
    int const n0 = blockIdx.x * kGemvRows;
    int const kb = k / 2;
    int const ksf = k / 16;

    int4 wq[kGemvRows];
    uint32_t ws[kGemvRows];
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        uint8_t const* p = w + static_cast<size_t>(n0 + r) * kb + tid * 16;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(wq[r].x), "=r"(wq[r].y), "=r"(wq[r].z), "=r"(wq[r].w)
                     : "l"(p));
    }
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        ws[r] = __ldg(reinterpret_cast<uint16_t const*>(sf + static_cast<size_t>(n0 + r) * ksf + tid * 2));
    }

    // Conv epilogue: thread tid < M * kGemvRows owns (row tid / 2, column n0 + tid % 2).
    bool const convThread = CONV && n0 < ce.conv_dim && tid < M * kGemvRows;
    float cw[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (CONV)
    {
        if (convThread)
        {
            // Constant conv weight: safe to load before the dependency wait.
            __nv_bfloat16 const* p = ce.weight + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.weight_c_stride;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                cw[i] = __bfloat162float(__ldg(p + i * ce.weight_l_stride));
            }
        }
    }

    asm volatile("griddepcontrol.wait;" ::: "memory");

    uint32_t xa[M][16];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
        int4 const* p = reinterpret_cast<int4 const*>(x + static_cast<size_t>(m) * k + tid * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int4 const v = p[q];
            xa[m][q * 4 + 0] = v.x;
            xa[m][q * 4 + 1] = v.y;
            xa[m][q * 4 + 2] = v.z;
            xa[m][q * 4 + 3] = v.w;
        }
    }

    // Mutable conv state and slot indices: loaded after the wait, beside the x loads.
    bool doConv = false;
    __nv_bfloat16* sp = nullptr;
    float s0f = 0.f, s1f = 0.f, s2f = 0.f;
    if constexpr (CONV)
    {
        if (convThread)
        {
            int const slot = ce.indices[tid / kGemvRows];
            if (slot != ce.pad_slot_id)
            {
                doConv = true;
                sp = ce.state + static_cast<int64_t>(slot) * ce.state_batch_stride
                    + static_cast<int64_t>(n0 + tid % kGemvRows) * ce.state_c_stride;
                s0f = __bfloat162float(sp[0]);
                s1f = __bfloat162float(sp[ce.state_l_stride]);
                s2f = __bfloat162float(sp[2 * ce.state_l_stride]);
            }
        }
    }

    float acc[M][kGemvRows];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            acc[m][r] = 0.f;
        }
    }

#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        float wf[32];
        uint32_t const* wv = reinterpret_cast<uint32_t const*>(&wq[r]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            uint32_t h[4];
            asm volatile(
                "{\n.reg .b8 b0, b1, b2, b3;\n"
                "mov.b32 {b0, b1, b2, b3}, %4;\n"
                "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
                "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
                "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
                "cvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
                : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
                : "r"(wv[i]));
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = __half22float2(reinterpret_cast<__half2 const&>(h[j]));
                wf[i * 8 + j * 2] = f.x;
                wf[i * 8 + j * 2 + 1] = f.y;
            }
        }
        __nv_fp8_e4m3 s8;
        s8.__x = static_cast<__nv_fp8_storage_t>(ws[r] & 0xff);
        float const s0 = static_cast<float>(s8);
        s8.__x = static_cast<__nv_fp8_storage_t>((ws[r] >> 8) & 0xff);
        float const s1 = static_cast<float>(s8);
#pragma unroll
        for (int m = 0; m < M; ++m)
        {
            float p0 = 0.f;
            float p1 = 0.f;
#pragma unroll
            for (int l = 0; l < 8; ++l)
            {
                p0 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p0);
                p0 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p0);
            }
#pragma unroll
            for (int l = 8; l < 16; ++l)
            {
                p1 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p1);
                p1 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p1);
            }
            acc[m][r] = fmaf(p0, s0, acc[m][r]);
            acc[m][r] = fmaf(p1, s1, acc[m][r]);
        }
    }

    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    constexpr int kWarps = BLOCK / 32;
    __shared__ float red[kWarps][M * kGemvRows];
    int const lane = tid & 31;
    int const warp = tid >> 5;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            float v = acc[m][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v += __shfl_xor_sync(0xffffffffu, v, o);
            }
            if (lane == 0)
            {
                red[warp][m * kGemvRows + r] = v;
            }
        }
    }
    __syncthreads();
    if constexpr (!CONV)
    {
        for (int i = tid; i < M * kGemvRows; i += BLOCK)
        {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < kWarps; ++j)
            {
                v += red[j][i];
            }
            int const m = i / kGemvRows;
            int const r = i % kGemvRows;
            out[static_cast<size_t>(m) * n + n0 + r] = __float2bfloat16(v * g);
        }
    }
    else
    {
        if (tid < M * kGemvRows)
        {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < kWarps; ++j)
            {
                v += red[j][tid];
            }
            int const m = tid / kGemvRows;
            int const r = tid % kGemvRows;
            __nv_bfloat16 o = __float2bfloat16(v * g);
            if (doConv)
            {
                // causal_conv1d_update_kernel_sl1 (width 4, no bias, silu): same order and intrinsics.
                float s[4];
                s[0] = s0f;
                s[1] = s1f;
                s[2] = s2f;
                s[3] = __bfloat162float(o);
                float ov = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ov = __fmaf_rn(cw[i], s[i], ov);
                }
                ov = ov * __frcp_rn(1.0f + __expf(-ov));
                o = __float2bfloat16(ov);
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    sp[i * ce.state_l_stride] = __float2bfloat16(s[i + 1]);
                }
            }
            out[static_cast<size_t>(m) * n + n0 + r] = o;
        }
    }
}

template <typename Kern>
void pdlLaunch(Kern kern, dim3 grid, dim3 block, uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x,
    __nv_bfloat16* o, float g, int n, int k, ConvEpilogue ce)
{
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
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w, sf, x, o, g, n, k, ce));
}

template <int BLOCK, bool CONV>
void byM(int m, uint8_t const* w, uint8_t const* sf, __nv_bfloat16 const* x, __nv_bfloat16* o, float g, int n,
    int k, ConvEpilogue ce)
{
    dim3 grid(n / kGemvRows);
    dim3 block(BLOCK);
    switch (m)
    {
    case 1: pdlLaunch(gemvKernel<1, BLOCK, CONV>, grid, block, w, sf, x, o, g, n, k, ce); break;
    case 2: pdlLaunch(gemvKernel<2, BLOCK, CONV>, grid, block, w, sf, x, o, g, n, k, ce); break;
    case 3: pdlLaunch(gemvKernel<3, BLOCK, CONV>, grid, block, w, sf, x, o, g, n, k, ce); break;
    case 4: pdlLaunch(gemvKernel<4, BLOCK, CONV>, grid, block, w, sf, x, o, g, n, k, ce); break;
    default: TORCH_CHECK(false, "m 1..4");
    }
}

}  // namespace

torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double g, torch::Tensor state,
    torch::Tensor cw, torch::Tensor idx, int64_t conv_dim, int64_t pad, int64_t conv)
{
    int m = x.size(0), k = x.size(1), n = w.size(0);
    TORCH_CHECK(k == 5120, "k 5120 only");
    auto out = torch::empty({m, n}, x.options());
    ConvEpilogue ce{};
    if (conv)
    {
        ce.state = reinterpret_cast<__nv_bfloat16*>(state.data_ptr());
        ce.weight = reinterpret_cast<__nv_bfloat16 const*>(cw.data_ptr());
        ce.indices = idx.data_ptr<int>();
        ce.state_batch_stride = state.stride(0);
        ce.state_c_stride = state.stride(1);
        ce.state_l_stride = state.stride(2);
        ce.weight_c_stride = cw.stride(0);
        ce.weight_l_stride = cw.stride(1);
        ce.conv_dim = static_cast<int>(conv_dim);
        ce.pad_slot_id = static_cast<int>(pad);
    }
    auto const* wp = w.data_ptr<uint8_t>();
    auto const* sp = sf.data_ptr<uint8_t>();
    auto const* xp = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    if (conv)
        byM<160, true>(m, wp, sp, xp, op, static_cast<float>(g), n, k, ce);
    else
        byM<160, false>(m, wp, sp, xp, op, static_cast<float>(g), n, k, ce);
    return out;
}
"""

CPP = ("torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double g, torch::Tensor state, "
       "torch::Tensor cw, torch::Tensor idx, int64_t conv_dim, int64_t pad, int64_t conv);")


def build_ext():
    from torch.utils.cpp_extension import load_inline
    bd = "/tmp/b118_gemv_conv_build"
    os.makedirs(bd, exist_ok=True)
    log("compiling extension ...")
    ext = load_inline(name="b118_gemv_conv", cpp_sources=CPP, cuda_sources=CU, functions=["run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-std=c++20",
                                         "-lineinfo", "-Xptxas=-v"],
                      build_directory=bd, verbose=os.environ.get("B118_VERBOSE", "0") == "1")
    log("extension ready")
    return ext


dev = torch.device("cuda")
EPS = 1e-6
K = 5120
NQKVZ, CONV, NBA = 16384, 10240, 96
H, HV, DK, DV = 16, 48, 128, 128
L = int(os.environ.get("B118_LAYERS", "48"))
SLOTS = 4
PAD = -1
G = 1.0 / 256.0
WARM = int(os.environ.get("B118_WARM", "10"))
ITERS = int(os.environ.get("B118_ITERS", "100"))
ROUNDS = int(os.environ.get("B118_ROUNDS", "3"))
MS = [int(v) for v in os.environ.get("B118_MS", "1,2,3").split(",")]
ARMS = ["base", "base2", "fused", "fused2", "inl", "ceil"]

log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"), "layers", L,
    "MS", MS, "warm", WARM, "iters", ITERS, "rounds", ROUNDS)
smi("start")
ext = build_ext()

gen = torch.Generator(device=dev).manual_seed(118)
WQ, SF, CW, WBA, CP0, SP0 = [], [], [], [], [], []
for li in range(L):
    WQ.append(torch.randint(0, 256, (NQKVZ, K // 2), generator=gen, device=dev, dtype=torch.int32)
              .to(torch.uint8))
    # E4M3 block scales in [0.25, 2): bytes 0x28..0x3f (no NaN codes).
    SF.append(torch.randint(0x28, 0x40, (NQKVZ, K // 16), generator=gen, device=dev, dtype=torch.int32)
              .to(torch.uint8))
    CW.append((torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16))
    WBA.append((torch.randn(NBA, K, generator=gen, device=dev) * 0.02).to(torch.bfloat16))
    CP0.append((torch.randn(SLOTS, CONV, 3, generator=gen, device=dev)).to(torch.bfloat16))
    SP0.append((torch.randn(SLOTS, HV, DV, DK, generator=gen, device=dev) * 0.1).to(torch.bfloat16))
CP = [t.clone() for t in CP0]
SP = [t.clone() for t in SP0]
A_LOG = torch.log(torch.rand(HV, generator=gen, device=dev) * 15.0 + 1.0).float()
DT_BIAS = (torch.randn(HV, generator=gen, device=dev) * 0.5).float()
WN = (0.1 * torch.randn(K, generator=gen, device=dev)).to(torch.bfloat16)
FLUSH = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
AUX = torch.cuda.Stream()
SIDE = torch.cuda.Stream()
torch.cuda.synchronize()
log("built %d layers; GPU memory allocated GB %.2f" % (L, torch.cuda.memory_allocated() / 1e9))


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return torch.nn.functional.linear(x, w) if y is None else y


def make_fn(arm, m, xs, rs, idx, cu):
    ev0 = [torch.cuda.Event() for _ in range(L)]
    ev1 = [torch.cuda.Event() for _ in range(L)]

    def fn():
        outs = []
        for li in range(L):
            x = xs[li]
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, rs[li], WN, EPS)
            ev0[li].record()
            if arm in ("base", "base2", "ceil"):
                y = torch.ops.trtllm.nvfp4_w4a16_gemv(x, WQ[li], SF[li], G)
            elif arm == "inl":
                y = ext.run(x, WQ[li], SF[li], G, CP[li], CW[li], idx, CONV, PAD, 0)
            else:
                y = ext.run(x, WQ[li], SF[li], G, CP[li], CW[li], idx, CONV, PAD, 1)
            with torch.cuda.stream(AUX):
                ev0[li].wait()
                ba = in_proj_ba(x, WBA[li])
                ev1[li].record()
            ev1[li].wait()
            mixed = y[:, :CONV]
            if arm in ("base", "base2", "inl"):
                mixed = causal_conv1d_update(mixed, CP[li], CW[li], None, "silu", conv_state_indices=idx)
            q = mixed[:, :H * DK].view(1, m, H, DK)
            k_ = mixed[:, H * DK:2 * H * DK].view(1, m, H, DK)
            v = mixed[:, 2 * H * DK:].view(1, m, HV, DV)
            o = fused_sigmoid_gating_delta_rule_update(
                A_log=A_LOG, dt_bias=DT_BIAS, q=q, k=k_, v=v, a=ba[:, HV:], b=ba[:, :HV],
                initial_state_source=SP[li], initial_state_indices=idx, cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0, output=None)
            outs.append((y, o))
        return outs

    return fn


def reset(st, st0):
    st.copy_(st0)
    for a, b in zip(CP, CP0):
        a.copy_(b)
    for a, b in zip(SP, SP0):
        a.copy_(b)


def capture(fn, st, st0):
    SIDE.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(SIDE):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(SIDE)
    torch.cuda.synchronize()
    reset(st, st0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    torch.cuda.synchronize()
    return gr, out


def snapshot(outs):
    return dict(y=[o[0].clone() for o in outs], o=[o[1].clone() for o in outs],
                cp=[t.clone() for t in CP], sp=[t.clone() for t in SP])


def compare(a, b):
    res = dict()
    for key in ("y", "o", "cp", "sp"):
        res[key] = all(torch.equal(u.view(torch.int16), v.view(torch.int16)) for u, v in zip(a[key], b[key]))
    res["all"] = all(res.values())
    return res


def stats(v):
    t = sorted(v)
    n = len(t)
    return dict(med=t[n // 2], p10=t[n // 10], p90=t[(9 * n) // 10])


results = dict()
checks = dict()
for m in MS:
    idx = torch.tensor([2, 0, 3][:m], dtype=torch.int32, device=dev)
    cu = torch.arange(m + 1, dtype=torch.long, device=dev)
    per = 2 * m * K
    st0 = (torch.randn(L * per, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    st = st0.clone()
    xs = [st[i * per: i * per + m * K].view(m, K) for i in range(L)]
    rs = [st[i * per + m * K: (i + 1) * per].view(m, K) for i in range(L)]
    graphs = dict()
    outs = dict()
    for arm in ARMS:
        graphs[arm], outs[arm] = capture(make_fn(arm, m, xs, rs, idx, cu), st, st0)
    log("M=%d captured %d graphs" % (m, len(graphs)))
    # Correctness: every arm replayed once from pristine state; base is the reference.
    snaps = dict()
    for arm in ARMS:
        reset(st, st0)
        graphs[arm].replay()
        torch.cuda.synchronize()
        if arm != "ceil":
            snaps[arm] = snapshot(outs[arm])
    chk = dict()
    for arm in ("base2", "fused", "fused2", "inl"):
        chk[arm + "_vs_base"] = compare(snaps[arm], snaps["base"])
    # Slot 1 is never in idx: its conv state must stay pristine.
    chk["fused_slot1_untouched"] = all(torch.equal(c[1], c0[1]) for c, c0 in zip(snaps["fused"]["cp"], CP0))
    del snaps
    # 10 sequential replays without reset (stateful): fused vs base stay bitwise.
    traj = dict()
    for arm in ("base", "fused"):
        reset(st, st0)
        first = None
        for step in range(10):
            graphs[arm].replay()
            if step == 0:
                first = snapshot(outs[arm])
        last = snapshot(outs[arm])
        torch.cuda.synchronize()
        traj[arm] = (first, last)
    chk["seq10_first"] = compare(traj["fused"][0], traj["base"][0])["all"]
    chk["seq10_last"] = compare(traj["fused"][1], traj["base"][1])["all"]
    del traj
    log("M=%d checks %s" % (m, json.dumps(chk)))
    checks[str(m)] = chk

    rounds = []
    for rnd in range(ROUNDS):
        times = dict((arm, []) for arm in ARMS)
        for it in range(WARM + ITERS):
            sh = it % len(ARMS)
            order = ARMS[sh:] + ARMS[:sh]
            if (it // len(ARMS)) % 2:
                order = order[::-1]
            for arm in order:
                reset(st, st0)
                FLUSH.sum()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm].replay()
                e1.record()
                if it >= WARM:
                    times[arm].append((e0, e1))
        torch.cuda.synchronize()
        r = dict()
        for arm in ARMS:
            r[arm] = stats([e0.elapsed_time(e1) * 1e3 / L for e0, e1 in times[arm]])
        gain = r["base"]["med"] - r["fused"]["med"]
        gain2 = r["base2"]["med"] - r["fused2"]["med"]
        floor = max(abs(r["base"]["med"] - r["base2"]["med"]), abs(r["fused"]["med"] - r["fused2"]["med"]))
        r["gain_us"] = gain
        r["gain2_us"] = gain2
        r["floor_us"] = floor
        r["ceiling_us"] = r["base"]["med"] - r["ceil"]["med"]
        r["inl_minus_base_us"] = r["inl"]["med"] - r["base"]["med"]
        r["fused_p90_lt_base_p10"] = (max(r["fused"]["p90"], r["fused2"]["p90"])
                                      < min(r["base"]["p10"], r["base2"]["p10"]))
        rounds.append(r)
        log("M=%d round %d us/layer: base %.3f [%.3f-%.3f] base2 %.3f | fused %.3f [%.3f-%.3f] fused2 %.3f | "
            "inl %.3f | ceil %.3f | gain %.3f gain2 %.3f floor %.3f ceiling %.3f | beyond p10/p90 %s"
            % (m, rnd, r["base"]["med"], r["base"]["p10"], r["base"]["p90"], r["base2"]["med"],
               r["fused"]["med"], r["fused"]["p10"], r["fused"]["p90"], r["fused2"]["med"], r["inl"]["med"],
               r["ceil"]["med"], gain, gain2, floor, r["ceiling_us"], r["fused_p90_lt_base_p10"]))
    results[str(m)] = rounds
    del graphs, outs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

smi("end")
gate = dict()
if 1 in MS:
    rr = results["1"]
    gate["M1_gain_ge_0p5_every_round"] = all(min(r["gain_us"], r["gain2_us"]) >= 0.5 for r in rr)
    gate["M1_gain_gt_2x_floor_every_round"] = all(min(r["gain_us"], r["gain2_us"]) > 2 * r["floor_us"] for r in rr)
    gate["M1_beyond_p10_p90_every_round"] = all(r["fused_p90_lt_base_p10"] for r in rr)
for m in (2, 3):
    if m in MS:
        rr = results[str(m)]
        gate["M%d_not_slower" % m] = all(min(r["gain_us"], r["gain2_us"]) >= -r["floor_us"] for r in rr)
gate["bitwise_all"] = all(c["fused_vs_base"]["all"] and c["fused2_vs_base"]["all"] and c["seq10_first"]
                          and c["seq10_last"] and c["fused_slot1_untouched"] for c in checks.values())
gate["GO"] = all(gate.values())
log("GATE", json.dumps(gate))
batch = dict()
for m in MS:
    r = results[str(m)][-1]
    f = r["fused"]
    batch[str(m)] = dict(kernel_ms=f["med"] * L / 1e3, noise_floor=(f["p90"] - f["p10"]) / f["med"])
print(json.dumps(dict(batch=batch, output={"class": "exact"}, gate=gate, checks=checks, results=results)))
