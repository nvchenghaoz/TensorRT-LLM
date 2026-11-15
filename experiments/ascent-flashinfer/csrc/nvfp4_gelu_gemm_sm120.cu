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
// TVM-FFI binding of the SM120 NVFP4 gate_up GEMM + gelu_tanh*up + NVFP4-quant epilogue
// (include/flashinfer/gemm/nvfp4_gelu_gemm_sm120.cuh).
#include "flashinfer/gemm/nvfp4_gelu_gemm_sm120.cuh"
#include "tvm_ffi_utils.h"

namespace {

void nvfp4_gelu_gemm(TensorView mat1, TensorView mat2, TensorView mat1Scale, TensorView mat2Scale,
                     TensorView alpha, TensorView globalScale, TensorView outFp4,
                     TensorView outSf, TensorView workspace) {
  TVM_FFI_ICHECK_EQ(mat1.dtype(), dl_uint8) << "mat1 (act fp4) must be uint8";
  TVM_FFI_ICHECK_EQ(mat2.dtype(), dl_uint8) << "mat2 (weight fp4) must be uint8";
  TVM_FFI_ICHECK_EQ(mat1Scale.dtype(), dl_uint8) << "mat1Scale must be uint8";
  TVM_FFI_ICHECK_EQ(mat2Scale.dtype(), dl_uint8) << "mat2Scale must be uint8";
  TVM_FFI_ICHECK_EQ(alpha.dtype(), dl_float32) << "alpha must be float32";
  TVM_FFI_ICHECK_EQ(globalScale.dtype(), dl_float32) << "globalScale must be float32";
  TVM_FFI_ICHECK_EQ(outFp4.dtype(), dl_uint8) << "outFp4 must be uint8";
  TVM_FFI_ICHECK_EQ(outSf.dtype(), dl_uint8) << "outSf must be uint8";
  TVM_FFI_ICHECK_EQ(mat1.ndim(), 2) << "mat1 must be 2D";
  TVM_FFI_ICHECK_EQ(mat2.ndim(), 2) << "mat2 must be 2D";
  CHECK_DEVICE(mat1, mat2);
  CHECK_DEVICE(mat1, outFp4);
  CHECK_DEVICE(mat1, outSf);

  int64_t const m = mat1.size(0);
  int64_t const k = mat1.size(1) * 2;
  int64_t const n = mat2.size(0);
  TVM_FFI_ICHECK_EQ(mat2.size(1) * 2, k) << "K mismatch";
  TVM_FFI_ICHECK_EQ(n % 128, 0) << "N must be a multiple of 128";
  TVM_FFI_ICHECK_EQ(outFp4.ndim(), 2) << "outFp4 must be 2D";
  TVM_FFI_ICHECK_EQ(outFp4.size(0), m) << "outFp4 rows must equal M";
  TVM_FFI_ICHECK_EQ(outFp4.size(1), n / 4) << "outFp4 must have N/4 bytes per row";
  int64_t const nkt = ((n / 2) / 16 + 3) / 4;
  int64_t const sf_bytes = (m + 127) / 128 * 128 * nkt * 4;
  TVM_FFI_ICHECK_GE(outSf.numel(), sf_bytes) << "outSf too small";
  if (m == 0) return;

  cudaStream_t stream = get_stream(mat1.device());
  int64_t const ws_bytes = workspace.numel() * get_element_size(workspace);
  size_t const need = flashinfer::gemm::gelu_epi::runGeluGemmSm120(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), nullptr, 0, stream);
  char* ws_ptr = static_cast<char*>(workspace.data_ptr());
  Tensor fallback;
  size_t ws_avail = static_cast<size_t>(ws_bytes);
  if (need > ws_avail) {
    fallback = alloc_tensor({static_cast<int64_t>(need)}, DLDataType{kDLInt, 8, 1}, mat1.device());
    ws_ptr = static_cast<char*>(fallback.data_ptr());
    ws_avail = need;
  }
  flashinfer::gemm::gelu_epi::runGeluGemmSm120(
      mat1.data_ptr(), mat2.data_ptr(), mat1Scale.data_ptr(), mat2Scale.data_ptr(),
      static_cast<float const*>(alpha.data_ptr()), static_cast<float const*>(globalScale.data_ptr()),
      static_cast<uint8_t*>(outFp4.data_ptr()), static_cast<uint8_t*>(outSf.data_ptr()),
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), ws_ptr, ws_avail, stream);
}

int64_t nvfp4_gelu_gemm_info(int64_t what) {
  using Cfg = flashinfer::gemm::gelu_epi::GeluGemmSm120;
  if (what == 0) return Cfg::kStages;
  if (what == 1) return static_cast<int64_t>(sizeof(typename Cfg::GemmKernel::SharedStorage));
  return -1;
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_gelu_gemm, nvfp4_gelu_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_gelu_gemm_info, nvfp4_gelu_gemm_info);
