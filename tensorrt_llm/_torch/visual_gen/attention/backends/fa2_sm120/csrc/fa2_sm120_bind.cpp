/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>

#include "flash.h"
#include "namespace_config.h"

namespace FLASH_NAMESPACE {
int run_fa2_sm120(Flash_fwd_params &params, cudaStream_t stream, float max_rem_frac);
}  // namespace FLASH_NAMESPACE

namespace F = FLASH_NAMESPACE;

// q, k, v, out: [B, S, H, D] (last dim contiguous), the views torch's
// _flash_attention_forward hands to mha_fwd. lse: [B, H, S] float32, contiguous.
// Fills Flash_fwd_params exactly as torch's set_params_fprop() does for
// p_dropout = 0, no window, no softcap, no alibi, no varlen.
int64_t fa2_sm120_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
                      double softmax_scale_d, double max_rem_frac) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda() && lse.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4);
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1,
                "last dim must be contiguous");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous());
    const int b = q.size(0), sq = q.size(1), h = q.size(2), d = q.size(3);
    const int sk = k.size(1), hk = k.size(2);
    TORCH_CHECK(d == 128, "head dim must be 128");
    TORCH_CHECK(hk == h && v.size(2) == h, "num heads of q, k, v must match");
    TORCH_CHECK(k.size(0) == b && v.size(0) == b && v.size(1) == sk && k.size(3) == d && v.size(3) == d);
    TORCH_CHECK(out.size(0) == b && out.size(1) == sq && out.size(2) == h && out.size(3) == d);
    TORCH_CHECK(lse.numel() == (int64_t)b * h * sq);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const float softmax_scale = static_cast<float>(softmax_scale_d);

    F::Flash_fwd_params params;
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
    params.p_dropout = 1.f;
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

    at::cuda::CUDAGuard device_guard{static_cast<signed char>(q.get_device())};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    return F::run_fa2_sm120(params, stream, static_cast<float>(max_rem_frac));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fa2_sm120_fwd); }
