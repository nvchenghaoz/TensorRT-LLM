#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "namespace_config.h"
#include "flash.h"

namespace FLASH_NAMESPACE {
struct VariantInfo { int block_m; int nthreads; int smem; int regs; int max_blocks_per_sm; };
#define DECL(V) void ascent_run_v##V(Flash_fwd_params &, cudaStream_t); VariantInfo ascent_info_v##V();
DECL(0) DECL(1) DECL(2) DECL(3) DECL(4) DECL(5) DECL(6) DECL(7)
#undef DECL
int ascent_run_split(Flash_fwd_params &params, cudaStream_t stream, int vb, int vt, float max_rem_frac);
}  // namespace FLASH_NAMESPACE

namespace F = FLASH_NAMESPACE;

// q, k, v, out: [B, S, H, D] views (last dim contiguous), exactly what torch's
// _flash_attention_forward hands to mha_fwd. lse: [B, H, S] float32.
// Mirrors set_params_fprop() for p_dropout=0, no window, no softcap, no alibi.
static void set_params(F::Flash_fwd_params &params, at::Tensor &q, at::Tensor &k, at::Tensor &v,
                       at::Tensor &out, at::Tensor &lse, double softmax_scale_d);

void fa2_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
             double softmax_scale_d, int64_t variant) {
    F::Flash_fwd_params params;
    set_params(params, q, k, v, out, lse, softmax_scale_d);
    at::cuda::CUDAGuard device_guard{static_cast<signed char>(q.get_device())};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    switch (variant) {
        case 0: F::ascent_run_v0(params, stream); break;
        case 1: F::ascent_run_v1(params, stream); break;
        case 2: F::ascent_run_v2(params, stream); break;
        case 3: F::ascent_run_v3(params, stream); break;
        case 4: F::ascent_run_v4(params, stream); break;
        case 5: F::ascent_run_v5(params, stream); break;
        case 6: F::ascent_run_v6(params, stream); break;
        case 7: F::ascent_run_v7(params, stream); break;
        default: TORCH_CHECK(false, "bad variant");
    }
}

int64_t fa2_fwd_split(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
                      double softmax_scale_d, int64_t vb, int64_t vt, double max_rem_frac) {
    F::Flash_fwd_params params;
    set_params(params, q, k, v, out, lse, softmax_scale_d);
    at::cuda::CUDAGuard device_guard{static_cast<signed char>(q.get_device())};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    return F::ascent_run_split(params, stream, (int)vb, (int)vt, (float)max_rem_frac);
}

static void set_params(F::Flash_fwd_params &params, at::Tensor &q, at::Tensor &k, at::Tensor &v,
                       at::Tensor &out, at::Tensor &lse, double softmax_scale_d) {
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1);
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous());
    const int b = q.size(0), sq = q.size(1), h = q.size(2), d = q.size(3);
    const int sk = k.size(1), hk = k.size(2);
    TORCH_CHECK(d == 128, "d must be 128");
    TORCH_CHECK(h % hk == 0);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const float softmax_scale = static_cast<float>(softmax_scale_d);

    params = {};
    params.is_bf16 = true;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;
    params.p_ptr = nullptr;
    params.softmax_lse_ptr = lse.data_ptr();
    params.b = b;
    params.h = h;
    params.h_k = hk;
    params.h_h_k_ratio = h / hk;
    params.seqlen_q = sq;
    params.seqlen_k = sk;
    params.seqlen_q_rounded = round_multiple(sq, 128);
    params.seqlen_k_rounded = round_multiple(sk, 128);
    params.d = d;
    params.d_rounded = round_multiple(d, 32);
    params.softcap = 0.0;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    const float p_dropout = 0.f;
    params.p_dropout = 1.f - p_dropout;
    params.p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
    params.num_splits = 1;
    params.alibi_slopes_ptr = nullptr;
}

std::vector<int64_t> fa2_info(int64_t variant) {
    F::VariantInfo i;
    switch (variant) {
        case 0: i = F::ascent_info_v0(); break;
        case 1: i = F::ascent_info_v1(); break;
        case 2: i = F::ascent_info_v2(); break;
        case 3: i = F::ascent_info_v3(); break;
        case 4: i = F::ascent_info_v4(); break;
        case 5: i = F::ascent_info_v5(); break;
        case 6: i = F::ascent_info_v6(); break;
        case 7: i = F::ascent_info_v7(); break;
        default: TORCH_CHECK(false, "bad variant");
    }
    return {i.block_m, i.nthreads, i.smem, i.regs, i.max_blocks_per_sm};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fa2_fwd);
    m.def("fwd_split", &fa2_fwd_split);
    m.def("info", &fa2_info);
}
