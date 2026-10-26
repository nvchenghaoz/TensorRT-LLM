/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
constexpr uint32_t copyBlockCtaSize = 128;
constexpr uint32_t copyBlocknbBufs = 2;

__device__ __host__ inline uint32_t divUp(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// dst_tensor[:, :num_seqs, 0] = src_tensor[:, copy_idx]
// dst_tensor[:, :num_seqs, 1] = dst_tensor[:, :num_seqs, 0] + 1
__global__ void copyBatchBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ srcPtr,
    SizeType32* __restrict__ dstPtr, SizeType32 const srcMaxNumSequences, SizeType32 const dstMaxNumSequences,
    SizeType32 numBlocksPerSeq, SizeType32 const* __restrict__ copyIndex, SizeType32 const* __restrict__ indexScales,
    SizeType32 const* __restrict__ kvOffset)
{
    constexpr uint32_t kvFactor = 2;
    constexpr auto elemPerAccess = sizeof(PackedInt) / sizeof(SizeType32);

    __shared__ PackedInt data[copyBlocknbBufs][copyBlockCtaSize];

    auto const iterPerSeq = divUp(numBlocksPerSeq * sizeof(SizeType32), sizeof(PackedInt) * copyBlockCtaSize);
    auto const tid = threadIdx.x;
    auto const poolIdx = blockIdx.x;
    auto const seqIdx = blockIdx.y;
    auto const seqDimStride = kvFactor * numBlocksPerSeq;
    uint32_t const srcIdxBeg = tid * elemPerAccess + (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride;
    uint32_t const dstIdxKBeg = tid * elemPerAccess + (poolIdx * dstMaxNumSequences + seqIdx) * seqDimStride;
    uint32_t const dstIdxVBeg = dstIdxKBeg + numBlocksPerSeq;

    uint32_t const srcIdxEnd = (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride + numBlocksPerSeq;

    for (uint32_t i = 0; i < iterPerSeq + copyBlocknbBufs; i++)
    {
        uint32_t const idxBuf = i % copyBlocknbBufs;
        if (i >= copyBlocknbBufs)
        {
            uint32_t const stIter = i - copyBlocknbBufs;
            assert(idxBuf == (stIter % copyBlocknbBufs));
            auto const offset = copyBlockCtaSize * stIter * elemPerAccess;
            SizeType32 const srcIdx = srcIdxBeg + offset;
            SizeType32 const dstIdxK = dstIdxKBeg + offset;
            SizeType32 const dstIdxV = dstIdxVBeg + offset;
            PackedInt const& src = data[idxBuf][tid];
            PackedInt& dstK = *reinterpret_cast<PackedInt*>(dstPtr + dstIdxK);
            PackedInt& dstV = *reinterpret_cast<PackedInt*>(dstPtr + dstIdxV);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(copyBlocknbBufs - 1) : "memory");
            if (srcIdx < srcIdxEnd)
            {
#pragma unroll
                for (uint32_t j = 0; j < elemPerAccess; j++)
                {
                    auto const val = src.unpacked[j];
                    dstK.unpacked[j] = (val == BAD_PAGE_INDEX) ? 0 : (indexScales[poolIdx] * val);
                    dstV.unpacked[j] = (val == BAD_PAGE_INDEX) ? 0 : (indexScales[poolIdx] * val + kvOffset[poolIdx]);
                }
            }
        }
        uint32_t const ldIter = i;
        PackedInt* const dst = &data[idxBuf][tid];
        uint32_t const srcIdx = srcIdxBeg + copyBlockCtaSize * ldIter * elemPerAccess;
        PackedInt const* const src = reinterpret_cast<PackedInt const*>(srcPtr + srcIdx);
        if (srcIdx < srcIdxEnd)
        {
            uint32_t const size = sizeof(PackedInt);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)),
                         "l"(src), "n"(size), "r"(size)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;\n" : : : "memory");
    }
}

// Host-side launcher
void copyBatchBlockOffsetsToDevice(ITensor const& input, ITensor& output, ITensor const& copyIndex,
    ITensor const& indexScales, ITensor const& kvOffset, CUstream stream) noexcept
{
    using namespace tensorrt_llm::runtime;

    auto const* srcPtr = input.getBufferPtr<tk::KVCacheIndex::UnderlyingType const>();
    // [numPools, maxNumSequences, kvFactor, numBlocksPerSeq]
    auto* dstPtr = output.getBufferPtr<tk::KVCacheIndex::UnderlyingType>();
    auto const* copyIndexPtr = copyIndex.getBufferPtr<SizeType32 const>();
    auto const* indexScalesPtr = indexScales.getBufferPtr<SizeType32 const>();
    auto const* kvOffsetPtr = kvOffset.getBufferPtr<SizeType32 const>();
    auto const& srcShape = input.getShape();
    auto const& dstShape = output.getShape();
    auto const& copyIndexShape = copyIndex.getShape();

    TLLM_CHECK(srcShape.nbDims == 4); // [numPools, srcMaxNumSequences, kvFactor, numBlocksPerSeq]
    TLLM_CHECK(dstShape.nbDims == 4); // [numPools, dstMaxNumSequences, kvFactor, numBlocksPerSeq]

    SizeType32 numPools = srcShape.d[0];
    SizeType32 srcMaxNumSequences = srcShape.d[1];
    SizeType32 dstMaxNumSequences = dstShape.d[1];
    SizeType32 numBlocksPerSeq = srcShape.d[3];
    SizeType32 numSeqs = copyIndexShape.d[0];

    if (numSeqs == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO((numBlocksPerSeq * sizeof(SizeType32)) % sizeof(PackedInt) == 0,
        "Not implemented case: numBlocksPerSeq * sizeof(SizeType32) = %zu must be a multiple of %zu.",
        static_cast<size_t>(numBlocksPerSeq * sizeof(SizeType32)), static_cast<size_t>(sizeof(PackedInt)));

    dim3 gridDim(numPools, numSeqs, 1);
    dim3 blockDim(copyBlockCtaSize);

    copyBatchBlockOffsetsToDeviceKernel<<<gridDim, blockDim, 0, stream>>>(srcPtr, dstPtr, srcMaxNumSequences,
        dstMaxNumSequences, numBlocksPerSeq, copyIndexPtr, indexScalesPtr, kvOffsetPtr);
}

namespace
{
constexpr uint32_t snapshotCtaSize = 128;
constexpr uint32_t snapshotVecPerThread = 4;

struct SnapshotScaleParams
{
    SizeType32 indexScale[kMaxSnapshotCopyPools];
    SizeType32 kvOffset[kMaxSnapshotCopyPools];
};

// dst_tensor[pool, seq, 0] = scale[pool] * snapshot[pool, seq] (0 for BAD_PAGE_INDEX)
// dst_tensor[pool, seq, 1] = scale[pool] * snapshot[pool, seq] + kvOffset[pool] (0 for BAD_PAGE_INDEX)
// The snapshot holds the K-plane rows already gathered by copyIndex on the host, and the per-pool scales and
// offsets arrive by value, so the only device-side reads are the snapshot rows themselves: every load of a chunk is
// issued before any of its stores, with no dependent round trips to host memory. Same arithmetic as
// copyBatchBlockOffsetsToDeviceKernel, so the destination is bitwise identical.
__global__ void copyBatchBlockOffsetsFromSnapshotKernel(SizeType32 const* __restrict__ snapshot,
    SizeType32* __restrict__ dstPtr, SizeType32 const dstMaxNumSequences, SizeType32 const numBlocksPerSeq,
    SnapshotScaleParams const __grid_constant__ params)
{
    using Vec = int4;
    constexpr uint32_t kvFactor = 2;
    uint32_t const poolIdx = blockIdx.x;
    uint32_t const seqIdx = blockIdx.y;
    uint32_t const numSeqs = gridDim.y;
    uint32_t const nbVecs = static_cast<uint32_t>(numBlocksPerSeq) / 4;
    SizeType32 const scale = params.indexScale[poolIdx];
    SizeType32 const offset = params.kvOffset[poolIdx];
    auto const* src = reinterpret_cast<Vec const*>(
        snapshot + static_cast<size_t>(poolIdx * numSeqs + seqIdx) * static_cast<size_t>(numBlocksPerSeq));
    auto* dstK = reinterpret_cast<Vec*>(dstPtr
        + static_cast<size_t>(poolIdx * static_cast<uint32_t>(dstMaxNumSequences) + seqIdx) * kvFactor
            * static_cast<size_t>(numBlocksPerSeq));
    auto* dstV = dstK + nbVecs;

    auto const convert = [scale, offset](SizeType32 val, SizeType32& k, SizeType32& v)
    {
        k = (val == BAD_PAGE_INDEX) ? 0 : (scale * val);
        v = (val == BAD_PAGE_INDEX) ? 0 : (scale * val + offset);
    };

    for (uint32_t base = 0; base < nbVecs; base += snapshotCtaSize * snapshotVecPerThread)
    {
        Vec vals[snapshotVecPerThread];
#pragma unroll
        for (uint32_t r = 0; r < snapshotVecPerThread; r++)
        {
            uint32_t const i = base + threadIdx.x + r * snapshotCtaSize;
            if (i < nbVecs)
            {
                vals[r] = src[i];
            }
        }
#pragma unroll
        for (uint32_t r = 0; r < snapshotVecPerThread; r++)
        {
            uint32_t const i = base + threadIdx.x + r * snapshotCtaSize;
            if (i < nbVecs)
            {
                Vec k;
                Vec v;
                convert(vals[r].x, k.x, v.x);
                convert(vals[r].y, k.y, v.y);
                convert(vals[r].z, k.z, v.z);
                convert(vals[r].w, k.w, v.w);
                dstK[i] = k;
                dstV[i] = v;
            }
        }
    }
}
} // namespace

void copyBatchBlockOffsetsToDeviceViaSnapshot(ITensor const& input, ITensor& snapshot, ITensor& output,
    ITensor const& copyIndex, ITensor const& indexScales, ITensor const& kvOffset, CUstream stream)
{
    using namespace tensorrt_llm::runtime;

    auto const* srcPtr = bufferCast<tk::KVCacheIndex::UnderlyingType const>(input);
    auto* snapshotPtr = bufferCast<tk::KVCacheIndex::UnderlyingType>(snapshot);
    auto* dstPtr = bufferCast<tk::KVCacheIndex::UnderlyingType>(output);
    auto const* copyIndexPtr = bufferCast<SizeType32 const>(copyIndex);
    auto const* indexScalesPtr = bufferCast<SizeType32 const>(indexScales);
    auto const* kvOffsetPtr = bufferCast<SizeType32 const>(kvOffset);
    auto const& srcShape = input.getShape();
    auto const& dstShape = output.getShape();

    TLLM_CHECK(srcShape.nbDims == 4); // [numPools, srcMaxNumSequences, kvFactor, numBlocksPerSeq]
    TLLM_CHECK(dstShape.nbDims == 4); // [numPools, dstMaxNumSequences, kvFactor, numBlocksPerSeq]
    TLLM_CHECK_WITH_INFO(input.getMemoryType() != MemoryType::kGPU && copyIndex.getMemoryType() != MemoryType::kGPU
            && indexScales.getMemoryType() != MemoryType::kGPU && kvOffset.getMemoryType() != MemoryType::kGPU,
        "The snapshot copy reads the page table, copy index, scales and offsets on the host.");
    TLLM_CHECK_WITH_INFO(
        snapshot.getMemoryType() == MemoryType::kPINNED || snapshot.getMemoryType() == MemoryType::kPINNEDPOOL,
        "The block-offset snapshot must be pinned host memory.");

    SizeType32 const numPools = srcShape.d[0];
    SizeType32 const srcMaxNumSequences = srcShape.d[1];
    SizeType32 const dstMaxNumSequences = dstShape.d[1];
    SizeType32 const numBlocksPerSeq = srcShape.d[3];
    SizeType32 const numSeqs = static_cast<SizeType32>(copyIndex.getSize());

    if (numSeqs == 0)
    {
        return;
    }
    TLLM_CHECK(srcShape.d[2] == 2 && dstShape.d[2] == 2);
    TLLM_CHECK(dstShape.d[0] >= numPools && dstShape.d[3] == numBlocksPerSeq && dstMaxNumSequences >= numSeqs);
    TLLM_CHECK_WITH_INFO(numPools <= static_cast<SizeType32>(kMaxSnapshotCopyPools),
        "numPools = %d exceeds the snapshot copy limit %u.", numPools, kMaxSnapshotCopyPools);
    TLLM_CHECK(static_cast<SizeType32>(indexScales.getSize()) >= numPools
        && static_cast<SizeType32>(kvOffset.getSize()) >= numPools);
    TLLM_CHECK_WITH_INFO((numBlocksPerSeq * sizeof(SizeType32)) % sizeof(PackedInt) == 0,
        "Not implemented case: numBlocksPerSeq * sizeof(SizeType32) = %zu must be a multiple of %zu.",
        static_cast<size_t>(numBlocksPerSeq * sizeof(SizeType32)), static_cast<size_t>(sizeof(PackedInt)));
    size_t const rowElems = static_cast<size_t>(numBlocksPerSeq);
    TLLM_CHECK_WITH_INFO(snapshot.getSize() >= static_cast<size_t>(numPools) * numSeqs * rowElems,
        "Block-offset snapshot holds %zu entries, needs %zu.", static_cast<size_t>(snapshot.getSize()),
        static_cast<size_t>(numPools) * numSeqs * rowElems);
    TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(snapshotPtr) % sizeof(PackedInt) == 0,
        "Block-offset snapshot must be %zu-byte aligned.", sizeof(PackedInt));

    // Gather the K plane of every scheduled row now, at enqueue time, so a later host rewrite of the page table or
    // of the index mapper's copy index (the overlap scheduler preparing the next batch) cannot reach this copy.
    for (SizeType32 seq = 0; seq < numSeqs; seq++)
    {
        SizeType32 const row = copyIndexPtr[seq];
        TLLM_CHECK_WITH_INFO(
            row >= 0 && row < srcMaxNumSequences, "Copy index %d out of range [0, %d).", row, srcMaxNumSequences);
    }
    for (SizeType32 pool = 0; pool < numPools; pool++)
    {
        for (SizeType32 seq = 0; seq < numSeqs; seq++)
        {
            auto const* srcRow = srcPtr
                + (static_cast<size_t>(pool) * srcMaxNumSequences + static_cast<size_t>(copyIndexPtr[seq])) * 2
                    * rowElems;
            std::memcpy(snapshotPtr + (static_cast<size_t>(pool) * numSeqs + seq) * rowElems, srcRow,
                rowElems * sizeof(SizeType32));
        }
    }

    SnapshotScaleParams params{};
    for (SizeType32 pool = 0; pool < numPools; pool++)
    {
        params.indexScale[pool] = indexScalesPtr[pool];
        params.kvOffset[pool] = kvOffsetPtr[pool];
    }

    dim3 const gridDim(numPools, numSeqs, 1);
    dim3 const blockDim(snapshotCtaSize);
    copyBatchBlockOffsetsFromSnapshotKernel<<<gridDim, blockDim, 0, stream>>>(
        snapshotPtr, dstPtr, dstMaxNumSequences, numBlocksPerSeq, params);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
