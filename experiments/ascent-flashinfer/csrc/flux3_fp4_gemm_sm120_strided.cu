/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// FLUX.3 (add-only): SM120 CUTLASS NVFP4 GEMM writing BF16 into a caller-given strided output.
// Same kernel as FlashInfer's stock SM120 instantiation (bf16, TileShape 128x128x128 FP4 elements,
// 1x1x1 cluster, no swap-AB, StaticPersistent DP, LinearCombination epilogue), which is bitwise
// equal to TRT-LLM FP4GemmRunner tactic 0. The only difference is D's row stride (ldd) and an
// element offset into the output base, so a GEMM can write one plane / column slice of a wider
// buffer. Never allocates: the workspace must be passed in.
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace gemm {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM, false)
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace {

using StridedGemm = flashinfer::gemm::Fp4Gemm___nv_bfloat16_128_128_128false;

void fp4_gemm_strided_impl(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                           TensorView alpha, TensorView out, int64_t offset, int64_t ldd,
                           TensorView workspace) {
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_uint8) << "a must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_uint8) << "b must be packed FP4 (uint8)";
  TVM_FFI_ICHECK_EQ(sfa.dtype(), dl_uint8) << "sfa must be uint8";
  TVM_FFI_ICHECK_EQ(sfb.dtype(), dl_uint8) << "sfb must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1) << "alpha must have one element";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), bfloat16_code) << "out must be bf16";
  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be 2D";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be 2D";
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, sfa);
  CHECK_DEVICE(a, sfb);
  CHECK_DEVICE(a, alpha);
  CHECK_DEVICE(a, out);
  CHECK_DEVICE(a, workspace);
  TVM_FFI_ICHECK_EQ(b.size(1), a.size(1)) << "K mismatch";

  int64_t const m = a.size(0);
  int64_t const n = b.size(0);
  int64_t const k = a.size(1) * 2;
  TVM_FFI_ICHECK(ldd >= n) << "ldd must be >= n";
  TVM_FFI_ICHECK(ldd % 8 == 0) << "ldd must be a multiple of 8 (16-byte rows)";
  TVM_FFI_ICHECK(offset >= 0 && offset % 8 == 0) << "offset must be >= 0 and a multiple of 8";
  TVM_FFI_ICHECK(offset + (m - 1) * ldd + n <= out.numel()) << "strided output out of bounds";

  auto* out_base = static_cast<__nv_bfloat16*>(out.data_ptr());
  void* d = static_cast<void*>(out_base + offset);

  auto args = flashinfer::gemm::prepareGemmArgsImpl<StridedGemm>(
      d, a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
      static_cast<float const*>(alpha.data_ptr()), static_cast<int>(m), static_cast<int>(n),
      static_cast<int>(k), 1);
  auto d_stride = args.epilogue.dD;
  cute::get<0>(d_stride) = ldd;
  cute::get<2>(d_stride) = ldd * m;
  args.epilogue.dD = d_stride;
  args.epilogue.dC = d_stride;

  StridedGemm gemm;
  size_t const ws_bytes = static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
  size_t const ws_need = gemm.get_workspace_size(args);
  TVM_FFI_ICHECK(ws_need <= ws_bytes)
      << "fp4_gemm_strided: workspace too small, need " << ws_need << " got " << ws_bytes;
  auto status = gemm.can_implement(args);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "fp4_gemm_strided: can_implement failed: " << cutlass::cutlassGetStatusString(status);
  cudaStream_t stream = get_stream(a.device());
  status = gemm.initialize(args, workspace.data_ptr(), stream);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "fp4_gemm_strided: initialize failed: " << cutlass::cutlassGetStatusString(status);
  status = gemm.run(args, workspace.data_ptr(), stream, nullptr, /*enablePDL=*/true);
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "fp4_gemm_strided: run failed: " << cutlass::cutlassGetStatusString(status);
}

}  // namespace

void fp4_gemm_strided(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                      TensorView alpha, TensorView out, int64_t offset, int64_t ldd,
                      TensorView workspace) {
  fp4_gemm_strided_impl(a, b, sfa, sfb, alpha, out, offset, ldd, workspace);
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_gemm_strided, torch_ext::fp4_gemm_strided);
