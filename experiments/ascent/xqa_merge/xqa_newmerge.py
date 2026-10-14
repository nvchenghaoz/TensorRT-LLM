# In-memory patch of the ORIGINAL mha.cu implementing the restructured multi-block merge. patch_newmerge() is the
# exploration form (ring depth / chunk size parameters); patch_final() produces exactly the committed
# cpp/kernels/xqa/mha.cu (used to verify the committed file byte for byte).

NS_ANCHOR = ("template <bool useFp32Acc, uint32_t nbWarps, uint32_t nbTiles, uint32_t rows, uint32_t cols>\n"
             "__device__ inline void smemFp16ArraySum(")
MERGE_ROW_MAX_BEG = "template <uint32_t nbBuffers>\n__device__ inline ThrdRegRowMax mergeRowMax("
MERGE_ROW_MAX_END = "__device__ inline void addAttentionSinks("

NS_CODE = r"""// Multi-block mode: the CTA that finishes last merges the partial results of all CTAs of the sequence. Only the 8-row
// slices of a warp tile that hold valid rows are merged; the other slices only pad the tile to the MMA M size and are
// never written to the output.
constexpr uint32_t nbMergeRowSlices = divUp(nbValidRows, 8U);
static_assert(nbMergeRowSlices <= WarpAcc::rows * InstAcc::rows && InstAcc::cols == 2);
// The valid row slices of a swizzled partial result tile, as copied from the multi-block scratch buffer.
using MergeTileSMemBuffer = Array2D<LdGrain, 8 * nbMergeRowSlices, SharedMem::XSmemBuffer::cols>;

__device__ inline void accumulateScaledWord(InstAcc& dst, uint32_t i, uint32_t word, float scale)
{
#if INPUT_FP16
    float2 const val = __half22float2(reinterpret_cast<InputElem2 const&>(word));
#else
    float2 const val = __bfloat1622float2(reinterpret_cast<InputElem2 const&>(word));
#endif
    dst(i, 0) += val.x * scale;
    dst(i, 1) += val.y * scale;
}

// acc += tile * rowScales[row] for the valid row slices of one partial result tile. Per element, this is the same
// arithmetic as rescaleAcc(tile) followed by acc = acc + tile.
__device__ inline void accumulateRescaledMergeTile(
    Warp const& warp, WarpAcc& acc, MergeTileSMemBuffer const& src, SMemWarpRowMax const& rowScales)
{
    uint32_t const lane = laneId();
#pragma unroll
    for (uint32_t s = 0; s < nbMergeRowSlices; s++)
    {
        uint32_t const m = s / InstAcc::rows;
        uint32_t const i = s % InstAcc::rows;
        float const scale = rowScales[8 * s + lane / 4];
#if __CUDA_ARCH__ >= 900
        constexpr uint32_t nbMatsPerLd = 4; // 8x8 b16 matrices loaded by one ldmatrix.x4, along the columns
#pragma unroll
        for (uint32_t n = 0; n < exactDiv(WarpAcc::cols, nbMatsPerLd); n++)
        {
            LdGrain const data
                = ldmatrix_4x<false>(warp, &src.template at<true>(8 * s + lane % 8, nbMatsPerLd * n + lane / 8));
#pragma unroll
            for (uint32_t j = 0; j < nbMatsPerLd; j++)
            {
                accumulateScaledWord(acc(m, nbMatsPerLd * n + j), i, data[j], scale);
            }
        }
#else
#pragma unroll
        for (uint32_t n = 0; n < WarpAcc::cols; n++)
        {
            accumulateScaledWord(acc(m, n), i, src.template at<true>(8 * s + lane / 4, n)[lane % 4], scale);
        }
#endif
    }
}

// Shared memory of the multi-block merge. It reuses the shared memory of the main loop: a ring of nbTileBuffers
// async-copy stages per warp for the partial result tiles, and the scaled row sums of a chunk of sub-sequences.
template <uint32_t nbTileBuffers_, uint32_t chunkSize_>
struct MultiBlockMergeSMemT
{
    static constexpr uint32_t nbTileBuffers = nbTileBuffers_;
    static constexpr uint32_t chunkSize = chunkSize_;

    bool isLastCta;

    struct MBBuf
    {
        MergeTileSMemBuffer tiles[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
        SMemWarpRowMax tileRowScales[chunkSize];
        SMemWarpRowMax partialRowMax[ctaShapeInWarps.x];
        SMemWarpRowMax mergedRowSum[gemm1NbWarpGrps];
    };

    MBBuf storage[ctaShapeInWarps.y];
};

template <uint32_t nbTileBuffers, uint32_t chunkSize>
HOST_DEVICE_FUNC constexpr bool multiBlockMergeSMemFits()
{
    return sizeof(MultiBlockMergeSMemT<nbTileBuffers, chunkSize>) <= sizeof(SharedMem);
}

template <uint32_t chunkSize>
HOST_DEVICE_FUNC constexpr uint32_t selectMultiBlockMergeChunkSize()
{
    if constexpr (chunkSize <= ctaShapeInWarps.x || multiBlockMergeSMemFits<2, chunkSize>())
    {
        return chunkSize;
    }
    else
    {
        return selectMultiBlockMergeChunkSize<chunkSize / 2>();
    }
}

template <uint32_t nbTileBuffers, uint32_t chunkSize>
HOST_DEVICE_FUNC constexpr uint32_t selectMultiBlockMergeNbTileBuffers()
{
    if constexpr (nbTileBuffers <= 2 || multiBlockMergeSMemFits<nbTileBuffers, chunkSize>())
    {
        return nbTileBuffers;
    }
    else
    {
        return selectMultiBlockMergeNbTileBuffers<nbTileBuffers - 1, chunkSize>();
    }
}

constexpr uint32_t multiBlockMergeChunkSize = selectMultiBlockMergeChunkSize<@CHUNK@>();
using MultiBlockMergeSMem
    = MultiBlockMergeSMemT<selectMultiBlockMergeNbTileBuffers<@DEPTH@, multiBlockMergeChunkSize>(), multiBlockMergeChunkSize>;

"""

R1_BEG = "            constexpr uint32_t nbTileBuffers = 2;\n"
R1_END = "            // increase the semaphore by 1\n"
R1_NEW = r"""            using MultiBlockSMem = MultiBlockMergeSMem;
            constexpr uint32_t nbTileBuffers = MultiBlockSMem::nbTileBuffers;
            constexpr uint32_t mergeChunkSize = MultiBlockSMem::chunkSize;
            static_assert(mergeChunkSize % gemm1NbWarpGrps == 0);

            static_assert(sizeof(MultiBlockSMem) <= smemSize);
            MultiBlockSMem& mbsmem = reinterpret_cast<MultiBlockSMem&>(smem);
"""

R2_BEG = "                SMemWarpRowMax& smemRowMax = reinterpret_cast<SMemWarpRowMax&>(smem);\n"
R2_END = "\n                ThrdRegRowMax mergedRowSum{};"
R2_NEW = r"""                // Each warp streams its partial result tiles through a ring of nbTileBuffers async-copy stages. The first
                // loads do not depend on the row statistics below, so they are issued first.
                auto loadBufAsync = [&](uint32_t n)
                {
                    uint32_t const d = n / gemm1NbWarpGrps % nbTileBuffers;
                    copyGrains<true, sizeof(ScratchBuf) / grainBytes, 1, true>(0,
                        &mbbuf.tiles[warpGrpIdx][warpIdxInGrp][d](0, 0),
                        &scratchBuffers[idxBufBase + n][warpIdxInGrp](0, 0));
                };
#pragma unroll
                for (uint32_t i = 0; i + 1 < nbTileBuffers; i++)
                {
                    uint32_t const n = warpGrpIdx + gemm1NbWarpGrps * i;
                    if (n < nbSubSeqPerSeq)
                    {
                        loadBufAsync(n);
                    }
                    ldgsts::commitGroup();
                }

                // Row max and row sum of the partial results of a chunk of sub-sequences: warp w loads sub-sequences
                // w, w + nbMergeWarps, ... of the chunk, all loads in flight together.
                constexpr uint32_t nbMergeWarps = ctaShapeInWarps.x;
                constexpr uint32_t nbStatsPerWarp = exactDiv(mergeChunkSize, nbMergeWarps);
                ThrdRegRowMax chunkRowMax[nbStatsPerWarp];
                ThrdRegRowMax chunkRowSum[nbStatsPerWarp];
                auto loadChunkRowStats = [&](uint32_t chunkBeg, uint32_t chunkEnd)
                {
#pragma unroll
                    for (uint32_t k = 0; k < nbStatsPerWarp; k++)
                    {
                        uint32_t const n = chunkBeg + nbMergeWarps * k + warpIdx.x;
                        if (n < chunkEnd)
                        {
                            chunkRowMax[k] = rowMaxBuffers[idxBufBase + n].loadToReg<false>(warp);
                            chunkRowSum[k] = rowSumBuffers[idxBufBase + n].loadToReg<false>(warp);
                        }
                    }
                };
                uint32_t const chunk0End = mha::min(mergeChunkSize, nbSubSeqPerSeq);
                loadChunkRowStats(0, chunk0End);

                // get row max. Max is exact, so splitting the sub-sequences among warps does not change the result.
                if (warpIdx.x < nbSubSeqPerSeq)
                {
                    ThrdRegRowMax warpRowMax = chunkRowMax[0];
#pragma unroll
                    for (uint32_t k = 1; k < nbStatsPerWarp; k++)
                    {
                        if (nbMergeWarps * k + warpIdx.x < chunk0End)
                        {
                            warpRowMax = fmaxf(warpRowMax, chunkRowMax[k]);
                        }
                    }
                    for (uint32_t n = mergeChunkSize + warpIdx.x; n < nbSubSeqPerSeq; n += nbMergeWarps)
                    {
                        warpRowMax = fmaxf(warpRowMax, rowMaxBuffers[idxBufBase + n].loadToReg<false>(warp));
                    }
                    mbbuf.partialRowMax[warpIdx.x].storeFromReg<false>(warp, warpRowMax);
                }
                __syncthreads();
                ThrdRegRowMax mergedRowMax = mbbuf.partialRowMax[0].loadToReg<false>(warp);
                for (uint32_t w = 1; w < mha::min(nbMergeWarps, nbSubSeqPerSeq); w++)
                {
                    mergedRowMax = fmaxf(mergedRowMax, mbbuf.partialRowMax[w].loadToReg<false>(warp));
                }

                // Scaled row sum of each partial result of a chunk: its weight in the merged output.
                auto storeChunkTileRowScales = [&](uint32_t chunkBeg, uint32_t chunkEnd)
                {
#pragma unroll
                    for (uint32_t k = 0; k < nbStatsPerWarp; k++)
                    {
                        uint32_t const n = chunkBeg + nbMergeWarps * k + warpIdx.x;
                        if (n < chunkEnd)
                        {
                            ThrdRegRowMax const tileRowScales = expf(chunkRowMax[k] - mergedRowMax);
                            ThrdRegRowMax const scaledTileRowSum = chunkRowSum[k] * tileRowScales;
                            mbbuf.tileRowScales[n - chunkBeg].storeFromReg<false>(warp, scaledTileRowSum);
                        }
                    }
                };
                storeChunkTileRowScales(0, chunk0End);
                __syncthreads();

                // rescale and accumulate, in the order of the sub-sequences
                WarpAcc sumAcc{};
                ThrdRegRowMax partialMergedRowSum{};
                for (uint32_t chunkBeg = 0; chunkBeg < nbSubSeqPerSeq; chunkBeg += mergeChunkSize)
                {
                    uint32_t const chunkEnd = mha::min(chunkBeg + mergeChunkSize, nbSubSeqPerSeq);
                    if (chunkBeg != 0)
                    {
                        loadChunkRowStats(chunkBeg, chunkEnd);
                        // All warps must be done with the row scales of the previous chunk before they are overwritten.
                        __syncthreads();
                        storeChunkTileRowScales(chunkBeg, chunkEnd);
                        __syncthreads();
                    }
                    for (uint32_t n = chunkBeg + warpGrpIdx; n < chunkEnd; n += gemm1NbWarpGrps)
                    {
                        uint32_t const nPrefetch = n + gemm1NbWarpGrps * (nbTileBuffers - 1);
                        if (nPrefetch < nbSubSeqPerSeq)
                        {
                            loadBufAsync(nPrefetch);
                        }
                        ldgsts::commitGroup();
                        ldgsts::waitGroup<nbTileBuffers - 1>();
                        // Each lane waits only for its own async-copy groups; synchronize before consuming the merge tile.
                        __syncwarp();
                        uint32_t const d = n / gemm1NbWarpGrps % nbTileBuffers;
                        SMemWarpRowMax const& scaledTileRowSum = mbbuf.tileRowScales[n - chunkBeg];
                        partialMergedRowSum = partialMergedRowSum + scaledTileRowSum.loadToReg<false>(warp);
                        assert(std::isfinite(partialMergedRowSum[0]));
                        accumulateRescaledMergeTile(
                            warp, sumAcc, mbbuf.tiles[warpGrpIdx][warpIdxInGrp][d], scaledTileRowSum);
                        // The merge tiles form a circular buffer. Make every lane finish the current ldmatrix reads
                        // before any lane advances far enough to overwrite this slot.
                        __syncwarp();
                    }
                }
"""


def _cut(src, beg, end, new):
    if src.count(beg) != 1 or src.count(end) != 1:
        raise RuntimeError(f"anchor not unique: {beg[:60]!r} / {end[:60]!r}")
    i0 = src.index(beg)
    i1 = src.index(end, i0)
    return src[:i0] + new + src[i1:]


def _merge_section(src):
    src = _cut(src, R1_BEG, R1_END, R1_NEW)
    return _cut(src, R2_BEG, R2_END, R2_NEW)


def patch_newmerge(src, depth=8, chunk=64):
    if src.count(NS_ANCHOR) != 1:
        raise RuntimeError("namespace anchor not unique")
    ns = NS_CODE.replace("@DEPTH@", str(depth)).replace("@CHUNK@", str(chunk))
    src = src.replace(NS_ANCHOR, ns + NS_ANCHOR)
    return _merge_section(src)


FINAL_DEPTH = 4
FINAL_CHUNK = 64


def patch_final(src):
    """The committed change: the new helpers replace the (now unused) mergeRowMax() in place."""
    ns = NS_CODE.replace("@DEPTH@", str(FINAL_DEPTH)).replace("@CHUNK@", str(FINAL_CHUNK))
    src = _cut(src, MERGE_ROW_MAX_BEG, MERGE_ROW_MAX_END, ns)
    return _merge_section(src)
