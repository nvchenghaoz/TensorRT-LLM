/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_FA2_MERGE_ROWS_CUH_
#define FLASHINFER_ATTENTION_FA2_MERGE_ROWS_CUH_

#include <cstdlib>

#include "../math.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

// FLASHINFER_FA2_MERGE_ROWS (on unless set to "0", read on every launch): the FA2 paged
// prefill split-KV merge runs GroupVariableLengthMergeStatesKernel instead of
// PersistentVariableLengthMergeStatesKernel.
inline bool FA2MergeRowsEnabled() {
  const char* env = std::getenv("FLASHINFER_FA2_MERGE_ROWS");
  return !(env != nullptr && env[0] == '0');
}

/*!
 * \brief Split-KV merge with one bdx-lane group per (position, head) and no shared memory.
 *
 * Lane tx of a group computes, for its vec_size columns, what the bdy threads (tx, 0..bdy-1) of
 * PersistentVariableLengthMergeStatesKernel (same vec_size, bdx, bdy) compute, in the same
 * order: thread ty merges index sets ty, ty + bdy, ty + 2 bdy, ... into a fresh state and
 * normalizes it; threadblock_sync_state passes each thread's o through DTypeIn (its v_smem) and
 * its lse through float, then merges them in ty order into a fresh state, which is normalized
 * and stored. Every output element therefore sees the same float operations in the same order,
 * from the same state_t code. The 0- and 1-index-set rows copy exactly as the persistent kernel
 * does.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t groups_per_block,
          typename DTypeIn, typename DTypeO, typename IdType>
__global__ void __launch_bounds__(bdx* groups_per_block) GroupVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V, float* __restrict__ S, IdType* indptr, DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged, uint32_t max_seq_len, uint32_t* __restrict__ seq_len_ptr,
    uint32_t num_heads) {
  constexpr uint32_t head_dim = vec_size * bdx;
  const uint32_t tx = threadIdx.x, gy = threadIdx.y;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const uint32_t seq_len = seq_len_ptr ? *seq_len_ptr : max_seq_len;
  const uint32_t total = seq_len * num_heads;

#pragma unroll 1
  for (uint32_t i = blockIdx.x * groups_per_block + gy; i < total;
       i += gridDim.x * groups_per_block) {
    const uint32_t pos = i / num_heads;
    const uint32_t head_idx = i % num_heads;
    const uint32_t first = indptr[pos];
    const uint32_t num_index_sets = indptr[pos + 1] - first;

    if (num_index_sets == 0) {
      vec_t<DTypeO, vec_size> v;
      v.fill(DTypeO(0.f));
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr && tx == 0) {
        s_merged[pos * num_heads + head_idx] = -math::inf;
      }
      continue;
    }

    if (num_index_sets == 1) {
      vec_t<DTypeO, vec_size> v;
      v.cast_load(V + (first * num_heads + head_idx) * head_dim + tx * vec_size);
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr && tx == 0) {
        s_merged[pos * num_heads + head_idx] = S[first * num_heads + head_idx];
      }
      continue;
    }

    const uint32_t num_iters = ceil_div(num_index_sets, bdy);
    state_t<vec_size> st;  // threadblock_sync_state's state after st.init()
#pragma unroll
    for (uint32_t ty = 0; ty < bdy; ++ty) {
      state_t<vec_size> st_ty;
#pragma unroll 1
      for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t j = iter * bdy + ty;
        if (j < num_index_sets) {
          vec_t<float, vec_size> v;
          v.cast_load(V + ((first + j) * num_heads + head_idx) * head_dim + tx * vec_size);
          float s = S[(first + j) * num_heads + head_idx];
          st_ty.merge(v, s, 1);
        }
      }
      st_ty.normalize();
      // threadblock_sync_state's round trip through DTypeIn shared memory.
      alignas(16) DTypeIn o_in[vec_size];
      st_ty.o.cast_store(o_in);
      const float s_ty = st_ty.get_lse();
      vec_t<float, vec_size> v;
      v.cast_load(o_in);
      st.merge(v, s_ty, 1);
    }
    st.normalize();

    st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr && tx == 0) {
      s_merged[pos * num_heads + head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// The FA2 paged prefill merge: GroupVariableLengthMergeStatesKernel with the persistent
// kernel's vec_size/bdx/bdy (VariableLengthMergeStates), or VariableLengthMergeStates itself.
template <typename DTypeIn, typename DTypeO, typename IdType>
cudaError_t FA2VariableLengthMergeStates(DTypeIn* v, float* s, IdType* indptr, DTypeO* v_merged,
                                         float* s_merged, uint32_t max_seq_len,
                                         uint32_t* seq_len, uint32_t num_heads,
                                         uint32_t head_dim, bool enable_pdl,
                                         cudaStream_t stream = nullptr) {
  if (!FA2MergeRowsEnabled()) {
    return VariableLengthMergeStates(v, s, indptr, v_merged, s_merged, max_seq_len, seq_len,
                                     num_heads, head_dim, enable_pdl, stream);
  }
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    constexpr uint32_t groups_per_block = num_threads / bdx;
    auto kernel = GroupVariableLengthMergeStatesKernel<vec_size, bdx, bdy, groups_per_block,
                                                       DTypeIn, DTypeO, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, 0));
    const uint32_t num_items = max_seq_len * num_heads;
    const uint32_t max_blocks = static_cast<uint32_t>(num_sms * std::max(num_blocks_per_sm, 1));
    const uint32_t num_blocks =
        std::max(1u, std::min(ceil_div(num_items, groups_per_block), max_blocks));

    dim3 nblks(num_blocks);
    dim3 nthrs(bdx, groups_per_block);
    if (enable_pdl) {
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t config;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, v, s, indptr, v_merged, s_merged,
                                              max_seq_len, seq_len, num_heads));
    } else {
      void* args[] = {&v, &s, &indptr, &v_merged, &s_merged, &max_seq_len, &seq_len, &num_heads};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_FA2_MERGE_ROWS_CUH_
