// V5 tail split: run the bulk of the row blocks with a big tile in whole waves,
// and the leftover rows with a smaller tile. Every row still goes through the
// unchanged compute_attn_1rowblock (same per-row math, kBlockN=64, reverse key order);
// only the (head, m_block) -> CTA mapping changes.
#include "fa2_common.cuh"

namespace FLASH_NAMESPACE {

struct RowMap { int g; int cntA; int loA; int cntB; int loB; };

// CTA i (per batch blockIdx.y): heads [0, g) own cntA row blocks starting at loA,
// heads [g, h) own cntB row blocks starting at loB. Head-major, m fastest (as torch's grid).
template <typename KT>
__global__ void flash_fwd_mapped_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params params, const RowMap map) {
#if defined(ARCH_SUPPORTS_FLASH)
    const int i = blockIdx.x;
    const int nA = map.g * map.cntA;
    int bidh, m_block;
    if (i < nA) {
        bidh = i / map.cntA;
        m_block = map.loA + (i - bidh * map.cntA);
    } else {
        const int j = i - nA;
        const int hh = j / map.cntB;
        bidh = map.g + hh;
        m_block = map.loB + (j - hh * map.cntB);
    }
    compute_attn_1rowblock<KT, false, false, false, false, false, true, false, false>(params, blockIdx.y, bidh, m_block);
#endif
}

template <typename KT>
static int mapped_slots(int num_sms) {
    static int occ = -1;
    if (occ < 0) {
        constexpr size_t smem_size = KT::kSmemSize;
        auto kernel = &flash_fwd_mapped_kernel<KT>;
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, KT::kNThreads, smem_size));
    }
    return occ * num_sms;
}

template <typename KT>
static void launch_mapped(Flash_fwd_params &params, cudaStream_t stream, const RowMap &map, int nctas) {
    constexpr size_t smem_size = KT::kSmemSize;
    auto kernel = &flash_fwd_mapped_kernel<KT>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 grid(nctas, params.b, 1);
    kernel<<<grid, KT::kNThreads, smem_size, stream>>>(params, map);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Returns number of launches (1 = no split, 2 = bulk + tail).
template <int VB, int VT>
int run_split(Flash_fwd_params &params, cudaStream_t stream, float max_rem_frac) {
    using KB = typename VariantTraits<VB>::T;
    using KTt = typename VariantTraits<VT>::T;
    static_assert(KB::kBlockM % KTt::kBlockM == 0);
    int dev, num_sms;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    const int H = params.h, S = params.seqlen_q;
    const int nb = (S + KB::kBlockM - 1) / KB::kBlockM;
    const int total = H * nb;
    const int slots = mapped_slots<KB>(num_sms) * 1;  // per batch; b == 1 in our use
    const int bulk = params.b == 1 ? (total / slots) * slots : total;
    const int rem = total - bulk;
    if (bulk == 0 || rem == 0 || float(rem) > max_rem_frac * float(slots)) {
        RowMap m{H, nb, 0, 0, 0};
        launch_mapped<KB>(params, stream, m, total);
        return 1;
    }
    const int base = bulk / H, extra = bulk % H;
    RowMap mb{extra, base + 1, 0, base, 0};
    launch_mapped<KB>(params, stream, mb, bulk);
    constexpr int r = KB::kBlockM / KTt::kBlockM;
    const int nt = (S + KTt::kBlockM - 1) / KTt::kBlockM;
    const int cA = std::max(0, nt - r * (base + 1));
    const int cB = std::max(0, nt - r * base);
    RowMap mt{extra, cA, r * (base + 1), cB, r * base};
    const int tail = extra * cA + (H - extra) * cB;
    if (tail > 0) { launch_mapped<KTt>(params, stream, mt, tail); }
    return 2;
}

int ascent_run_split(Flash_fwd_params &params, cudaStream_t stream, int vb, int vt, float max_rem_frac) {
    if (vb == 1 && vt == 2) return run_split<1, 2>(params, stream, max_rem_frac);
    if (vb == 1 && vt == 5) return run_split<1, 5>(params, stream, max_rem_frac);
    if (vb == 1 && vt == 0) return run_split<1, 0>(params, stream, max_rem_frac);
    TORCH_CHECK(false, "unsupported split pair");
    return 0;
}

}  // namespace FLASH_NAMESPACE
