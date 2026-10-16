// SM120 e4m3 tiled context FMHA harness (branch 16, idea 45).
//
// Instantiates fused_multihead_attention::device_flash_attention_nl_tiled with the same template
// arguments as the generated TU of
//   fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled
// (setup.py flash_attention_kernel_template; GENERATE_CUBIN => head_interleaved=False, flags 0x1027)
// from whatever cpp/kernels/fmha_v2/src tree is on the include path, and launches it the way the
// generated launcher does (grid (ceil(s/64), h, b), Kernel_traits_nl_tiled::BYTES_PER_SMEM).
//
// Shapes: Qwen3.8-27B full attention, 24 q / 4 kv heads, head 256, e4m3 paged KV (32-token pages,
// kv scale 1.0), e4m3 Q, BF16 O, causal over past + new tokens. Prefix pages are aliased across
// sequences (serving block reuse).
//
// Build: nvcc -O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -DUSE_DEMO_BERT_PARAMS=1
//        -DGENERATE_CUBIN=1 -gencode arch=compute_120f,code=sm_120f -I<tree>/cpp/kernels/fmha_v2/src
//        bench.cu -o bench -lcuda
//
// Usage: bench --cases c1,c4,... [--warmup 10] [--iters 50] [--dump DIR]
//   Every call (warmup and timed) is preceded by a reset point: NaN-poison of O (0xFF bytes) and a
//   384 MiB L2 read-flush (both untimed). Each call is timed with its own CUDA event pair. After every
//   call the output hash is taken (untimed) and compared with call 1; Q / KV pool hashes are checked
//   unchanged at the end. Last stdout line: CASES_JSON {...}.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));           \
            exit(2);                                                                                                   \
        }                                                                                                              \
    } while (0)

static constexpr uint32_t kFlags = 0x1027u; // LDGSTS q|k|v (0x7), !head_interleaved (0x20), tiled (0x1000)

// The kernel traits of the causal kernel and of the launcher (which takes BYTES_PER_SMEM from the padding one).
using Kt_causal = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    kFlags | 0x200, 3, true, nv_bfloat16>;
using Kt_launch = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    kFlags | 0x200, 2, true, nv_bfloat16, 0, 0, 0>;

// Reports whether the tree has the register-direct O epilogue (0 on main).
template <typename T, typename = void>
struct Store_o_from_regs
{
    static constexpr int VALUE = 0;
};

template <typename T>
struct Store_o_from_regs<T, std::void_t<decltype(T::STORE_O_FROM_REGS)>>
{
    static constexpr int VALUE = T::STORE_O_FROM_REGS ? 1 : 0;
};

extern "C" __global__ void fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled(
    bert::Fused_multihead_attention_params_v2 params)
{
    fused_multihead_attention::device_flash_attention_nl_tiled<Kt_causal>(params);
}

#define KERNEL fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ uint64_t mix64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Approximately N(0, std) values (Irwin-Hall of 4 bytes), quantized to e4m3.
__global__ void fill_e4m3(__nv_fp8_e4m3* p, size_t n, uint64_t seed, float stdev)
{
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
    {
        uint64_t a = mix64(seed * 0x9E3779B97F4A7C15ULL + i);
        float u = (float) ((a & 0xff) + ((a >> 8) & 0xff) + ((a >> 16) & 0xff) + ((a >> 24) & 0xff)) / 255.f - 2.f;
        p[i] = __nv_fp8_e4m3(u * stdev * 1.7320508f);
    }
}

// Order-independent hash of a buffer of 8-byte words.
__global__ void hash_buf(uint64_t const* p, size_t n_words, unsigned long long* out)
{
    uint64_t acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n_words; i += (size_t) gridDim.x * blockDim.x)
    {
        acc += mix64(p[i] ^ (i * 0x9E3779B97F4A7C15ULL));
    }
    for (int o = 16; o > 0; o /= 2)
    {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }
    if ((threadIdx.x & 31) == 0)
    {
        atomicAdd(out, (unsigned long long) acc);
    }
}

// Count of non-finite bf16 values.
__global__ void count_nonfinite(__nv_bfloat16 const* p, size_t n, unsigned long long* out)
{
    unsigned long long c = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
    {
        float v = __bfloat162float(p[i]);
        c += !isfinite(v);
    }
    for (int o = 16; o > 0; o /= 2)
    {
        c += __shfl_xor_sync(0xffffffff, c, o);
    }
    if ((threadIdx.x & 31) == 0 && c)
    {
        atomicAdd(out, c);
    }
}

// L2 read-flush.
__global__ void flush_read(uint4 const* p, size_t n, int* sink)
{
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
    {
        uint4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u)
    {
        *sink = acc;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Problem
{
    std::string name;
    int b = 1, h = 24, h_kv = 4, d = 256, tpb = 32;
    std::vector<int> q_lens, past_lens;
    bool alias = true;
    int s_max = 0, max_blocks = 0, shared_blocks = 0, total_blocks = 0, total_q = 0;
    size_t q_bytes = 0, o_bytes = 0, pool_bytes = 0, bytes_per_block = 0;
    void* q = nullptr;
    void* o = nullptr;
    void* pool = nullptr;
    int32_t* offs = nullptr;
    int* cu_q = nullptr;
    int* cu_kv = nullptr;
    float* scales = nullptr;
    std::vector<int32_t> h_offs;
    std::vector<int> h_cu_q, h_cu_kv;
    bert::Fused_multihead_attention_params_v2 params{};
};

static uint32_t f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static void build(Problem& p, uint64_t seed)
{
    p.b = (int) p.q_lens.size();
    p.s_max = *std::max_element(p.q_lens.begin(), p.q_lens.end());
    int min_past = *std::min_element(p.past_lens.begin(), p.past_lens.end());
    p.shared_blocks = p.alias ? min_past / p.tpb : 0;
    std::vector<int> nblk(p.b);
    p.max_blocks = 0;
    p.total_blocks = 2 * p.shared_blocks;
    p.h_cu_q.assign(p.b + 1, 0);
    p.h_cu_kv.assign(p.b + 1, 0);
    for (int i = 0; i < p.b; ++i)
    {
        int kv = p.past_lens[i] + p.q_lens[i];
        nblk[i] = (kv + p.tpb - 1) / p.tpb;
        p.max_blocks = std::max(p.max_blocks, nblk[i]);
        p.total_blocks += 2 * (nblk[i] - p.shared_blocks);
        p.h_cu_q[i + 1] = p.h_cu_q[i] + p.q_lens[i];
        p.h_cu_kv[i + 1] = p.h_cu_kv[i] + kv;
    }
    p.total_q = p.h_cu_q[p.b];
    // Block offsets [b][2][max_blocks]: shared K blocks [0, S), shared V blocks [S, 2S), then private ones.
    p.h_offs.assign((size_t) p.b * 2 * p.max_blocks, 0);
    int cursor = 2 * p.shared_blocks;
    for (int i = 0; i < p.b; ++i)
    {
        for (int j = 0; j < nblk[i]; ++j)
        {
            int32_t kb, vb;
            if (j < p.shared_blocks)
            {
                kb = j;
                vb = p.shared_blocks + j;
            }
            else
            {
                kb = cursor++;
                vb = cursor++;
            }
            p.h_offs[((size_t) i * 2 + 0) * p.max_blocks + j] = kb;
            p.h_offs[((size_t) i * 2 + 1) * p.max_blocks + j] = vb;
        }
    }
    p.bytes_per_block = (size_t) p.h_kv * p.tpb * p.d;
    p.pool_bytes = p.bytes_per_block * p.total_blocks;
    p.q_bytes = (size_t) p.total_q * p.h * p.d;
    p.o_bytes = (size_t) p.total_q * p.h * p.d * 2;

    CK(cudaMalloc(&p.q, p.q_bytes));
    CK(cudaMalloc(&p.o, p.o_bytes));
    CK(cudaMalloc(&p.pool, p.pool_bytes));
    CK(cudaMalloc(&p.offs, p.h_offs.size() * 4));
    CK(cudaMalloc(&p.cu_q, (p.b + 1) * 4));
    CK(cudaMalloc(&p.cu_kv, (p.b + 1) * 4));
    CK(cudaMalloc(&p.scales, 4 * 4));
    CK(cudaMemcpy(p.offs, p.h_offs.data(), p.h_offs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(p.cu_q, p.h_cu_q.data(), (p.b + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(p.cu_kv, p.h_cu_kv.data(), (p.b + 1) * 4, cudaMemcpyHostToDevice));
    // Serving: bmm1 = dequantQ * dequantK / sqrt(d) (kv scale 1.0), [1] = [0] * log2(e); bmm2 = quantO * dequantV.
    float h_scales[4] = {1.f / 16.f, 1.f / 16.f * 1.4426950408889634f, 1.f, 0.f};
    CK(cudaMemcpy(p.scales, h_scales, sizeof(h_scales), cudaMemcpyHostToDevice));
    fill_e4m3<<<1024, 256>>>(reinterpret_cast<__nv_fp8_e4m3*>(p.q), p.q_bytes, seed * 3 + 1, 1.0f);
    fill_e4m3<<<4096, 256>>>(reinterpret_cast<__nv_fp8_e4m3*>(p.pool), p.pool_bytes, seed * 3 + 2, 1.0f);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    auto& P = p.params;
    P.q_ptr = p.q;
    P.q_stride_in_bytes = (int64_t) p.h * p.d;
    P.k_stride_in_bytes = (int64_t) p.tpb * p.d;
    P.v_stride_in_bytes = (int64_t) p.tpb * p.d;
    P.paged_kv_cache.mMaxSeqs = p.b;
    P.paged_kv_cache.mMaxBlocksPerSeq = p.max_blocks;
    P.paged_kv_cache.mTokensPerBlock = p.tpb;
    P.paged_kv_cache.mTokensPerBlockLog2 = 5;
    P.paged_kv_cache.mBytesPerBlock = (int32_t) p.bytes_per_block;
    P.paged_kv_cache.mPoolPtr = p.pool;
    P.paged_kv_cache.mBlockOffsets = p.offs;
    P.o_ptr = p.o;
    P.o_stride_in_bytes = (int64_t) p.h * p.d * 2;
    P.softmax_stats_ptr = nullptr;
    P.attention_sinks = nullptr;
    P.packed_mask_ptr = nullptr;
    P.b = p.b;
    P.h = p.h;
    P.h_kv = p.h_kv;
    P.h_q_per_kv = p.h / p.h_kv;
    P.s = p.s_max;
    P.d = p.d;
    P.dv = p.d;
    P.num_grouped_heads = 1;
    P.sliding_window_size = INT_MAX;
    P.log2_chunked_attention_size = 0;
    P.scale_bmm1 = f2u(1.f / 16.f);
    P.softcapping_scale_bmm1 = 0;
    P.scale_softmax = f2u(1.f);
    P.scale_bmm2 = f2u(1.f);
    P.scale_bmm1_d = reinterpret_cast<uint32_t*>(p.scales);
    P.scale_bmm2_d = reinterpret_cast<uint32_t*>(p.scales + 2);
    P.cu_q_seqlens = p.cu_q;
    P.cu_kv_seqlens = p.cu_kv;
    P.cu_mask_rows = nullptr;
    P.has_alibi = false;
    P.is_s_padded = false;
}

static void release(Problem& p)
{
    cudaFree(p.q);
    cudaFree(p.o);
    cudaFree(p.pool);
    cudaFree(p.offs);
    cudaFree(p.cu_q);
    cudaFree(p.cu_kv);
    cudaFree(p.scales);
}

static dim3 grid_of(Problem const& p)
{
    // Generated launcher: ctas_per_o_row = ceil(d / Cta_tile_o::N) = 1; loop_iters = ceil(s / 64).
    using Cta_tile_o = typename Kt_launch::Cta_tile_o;
    int ctas_per_o_row = (p.params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
    int loop_iters = (p.params.s + 64 - 1) / 64;
    return dim3(loop_iters * ctas_per_o_row, p.params.h, p.params.b);
}

static void launch(Problem& p, cudaStream_t st)
{
    KERNEL<<<grid_of(p), Kt_launch::THREADS, Kt_launch::BYTES_PER_SMEM, st>>>(p.params);
}

static unsigned long long hash_dev(void const* ptr, size_t bytes, unsigned long long* d_tmp, cudaStream_t st)
{
    CK(cudaMemsetAsync(d_tmp, 0, 8, st));
    hash_buf<<<1024, 256, 0, st>>>(reinterpret_cast<uint64_t const*>(ptr), bytes / 8, d_tmp);
    unsigned long long h = 0;
    CK(cudaMemcpyAsync(&h, d_tmp, 8, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    return h;
}

static unsigned long long nonfinite_dev(void const* ptr, size_t elems, unsigned long long* d_tmp, cudaStream_t st)
{
    CK(cudaMemsetAsync(d_tmp, 0, 8, st));
    count_nonfinite<<<1024, 256, 0, st>>>(reinterpret_cast<__nv_bfloat16 const*>(ptr), elems, d_tmp);
    unsigned long long h = 0;
    CK(cudaMemcpyAsync(&h, d_tmp, 8, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    return h;
}

static double pct(std::vector<float> v, double q)
{
    std::sort(v.begin(), v.end());
    double idx = q * (v.size() - 1);
    size_t lo = (size_t) floor(idx), hi = (size_t) ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

static void write_file(std::string const& path, void const* dev, size_t bytes)
{
    std::vector<char> h(bytes);
    CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(3);
    }
    fwrite(h.data(), 1, bytes, f);
    fclose(f);
}

static bool make_case(std::string const& name, Problem& p)
{
    p.name = name;
    if (name == "c1")
    { // 1 x 2048 new tokens over 32768 cached.
        p.q_lens = {2048};
        p.past_lens = {32768};
    }
    else if (name == "c2")
    {
        p.q_lens = {2048, 2048};
        p.past_lens = {32768, 32768};
    }
    else if (name == "c4")
    {
        p.q_lens = {2048, 2048, 2048, 2048};
        p.past_lens = {32768, 32768, 32768, 32768};
    }
    else if (name == "c1_1984")
    { // 31 x 24 = 744 CTAs (area-lead tail probe).
        p.q_lens = {1984};
        p.past_lens = {32768};
    }
    else if (name == "ragged")
    { // Row guard: 2000 and 1 new tokens over the shared prefix.
        p.q_lens = {2000, 1};
        p.past_lens = {32768, 32768};
    }
    else if (name == "empty")
    { // Empty cache, s = 2048.
        p.q_lens = {2048};
        p.past_lens = {0};
        p.alias = false;
    }
    else
    {
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    std::string cases = "c1,c2,c4,c1_1984";
    std::string dump;
    int warmup = 10, iters = 50;
    uint64_t seed = 1234;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--cases" && i + 1 < argc)
            cases = argv[++i];
        else if (a == "--warmup" && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (a == "--dump" && i + 1 < argc)
            dump = argv[++i];
        else if (a == "--seed" && i + 1 < argc)
            seed = strtoull(argv[++i], nullptr, 10);
    }

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, KERNEL));
    int const smem = Kt_launch::BYTES_PER_SMEM;
    if (smem >= 48 * 1024)
    {
        CK(cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, KERNEL, Kt_launch::THREADS, smem));
    int const from_regs = Store_o_from_regs<Kt_causal>::VALUE;
    printf("[info] %s sms=%d smem_per_sm=%zu | kernel regs=%d local=%zu static_smem=%zu dyn_smem=%d threads=%d "
           "store_o_from_regs=%d occupancy=%d CTAs/SM\n",
        prop.name, prop.multiProcessorCount, prop.sharedMemPerMultiprocessor, fa.numRegs, fa.localSizeBytes,
        fa.sharedSizeBytes, smem, Kt_launch::THREADS, from_regs, occ);
    fflush(stdout);

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    unsigned long long* d_tmp;
    CK(cudaMalloc(&d_tmp, 8));
    int* d_sink;
    CK(cudaMalloc(&d_sink, 4));
    size_t const flush_bytes = 384ull << 20;
    void* flush_buf;
    CK(cudaMalloc(&flush_buf, flush_bytes));
    CK(cudaMemset(flush_buf, 1, flush_bytes));
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));

    std::string json = "{\"cases\": {";
    bool all_ok = true;
    size_t pos = 0;
    bool first_case = true;
    while (pos <= cases.size())
    {
        size_t comma = cases.find(',', pos);
        std::string cname = cases.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? cases.size() + 1 : comma + 1;
        if (cname.empty())
            continue;
        Problem p;
        if (!make_case(cname, p))
        {
            fprintf(stderr, "unknown case %s\n", cname.c_str());
            return 4;
        }
        build(p, seed);
        dim3 g = grid_of(p);
        unsigned long long q_hash0 = hash_dev(p.q, p.q_bytes, d_tmp, st);
        unsigned long long kv_hash0 = hash_dev(p.pool, p.pool_bytes, d_tmp, st);

        std::vector<float> times;
        unsigned long long out_hash1 = 0;
        int mismatches = 0;
        unsigned long long nonfinite_max = 0;
        int const total = warmup + iters;
        for (int it = 0; it < total; ++it)
        {
            // Reset point: NaN-poison O, then read-flush L2 (both untimed).
            CK(cudaMemsetAsync(p.o, 0xFF, p.o_bytes, st));
            flush_read<<<prop.multiProcessorCount * 4, 512, 0, st>>>(
                reinterpret_cast<uint4 const*>(flush_buf), flush_bytes / 16, d_sink);
            CK(cudaEventRecord(ev0, st));
            launch(p, st);
            CK(cudaEventRecord(ev1, st));
            CK(cudaGetLastError());
            CK(cudaEventSynchronize(ev1));
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (it >= warmup)
            {
                times.push_back(ms);
            }
            unsigned long long hsh = hash_dev(p.o, p.o_bytes, d_tmp, st);
            unsigned long long nf = nonfinite_dev(p.o, p.o_bytes / 2, d_tmp, st);
            nonfinite_max = std::max(nonfinite_max, nf);
            if (it == 0)
            {
                out_hash1 = hsh;
            }
            else if (hsh != out_hash1)
            {
                ++mismatches;
            }
            if (it == 0 || it == warmup || it == total - 1)
            {
                printf("[%s] call %d/%d: %.4f ms out_hash=%016llx nonfinite=%llu\n", cname.c_str(), it + 1, total, ms,
                    hsh, nf);
                fflush(stdout);
            }
        }
        unsigned long long q_hash1 = hash_dev(p.q, p.q_bytes, d_tmp, st);
        unsigned long long kv_hash1 = hash_dev(p.pool, p.pool_bytes, d_tmp, st);
        bool inputs_ok = q_hash0 == q_hash1 && kv_hash0 == kv_hash1;
        bool ok = mismatches == 0 && inputs_ok && nonfinite_max == 0;
        all_ok = all_ok && ok;
        double med = pct(times, 0.5), p10 = pct(times, 0.1), p90 = pct(times, 0.9);
        double mn = *std::min_element(times.begin(), times.end());
        printf("[%s] b=%d s_max=%d grid=(%u,%u,%u)=%u CTAs: median %.4f ms p10 %.4f p90 %.4f min %.4f | "
               "repeat mismatches %d/%d nonfinite %llu inputs_unchanged %d out_hash %016llx kv_hash %016llx\n",
            cname.c_str(), p.b, p.s_max, g.x, g.y, g.z, g.x * g.y * g.z, med, p10, p90, mn, mismatches, total - 1,
            nonfinite_max, (int) inputs_ok, out_hash1, kv_hash0);
        fflush(stdout);
        if (!dump.empty())
        {
            write_file(dump + "/" + cname + "_o.bin", p.o, p.o_bytes);
            write_file(dump + "/" + cname + "_q.bin", p.q, p.q_bytes);
            write_file(dump + "/" + cname + "_pool.bin", p.pool, p.pool_bytes);
            FILE* f = fopen((dump + "/" + cname + "_meta.json").c_str(), "w");
            fprintf(f, "{\"b\": %d, \"h\": %d, \"h_kv\": %d, \"d\": %d, \"tpb\": %d, \"max_blocks\": %d, \"q_lens\": [",
                p.b, p.h, p.h_kv, p.d, p.tpb, p.max_blocks);
            for (int i = 0; i < p.b; ++i)
                fprintf(f, "%s%d", i ? ", " : "", p.q_lens[i]);
            fprintf(f, "], \"past_lens\": [");
            for (int i = 0; i < p.b; ++i)
                fprintf(f, "%s%d", i ? ", " : "", p.past_lens[i]);
            fprintf(f, "], \"offsets\": [");
            for (size_t i = 0; i < p.h_offs.size(); ++i)
                fprintf(f, "%s%d", i ? ", " : "", p.h_offs[i]);
            fprintf(f, "], \"scale_bmm1\": %.9g, \"scale_bmm2\": 1.0, \"calls\": %d, \"repeat_mismatches\": %d, "
                       "\"inputs_unchanged\": %s, \"nonfinite\": %llu}\n",
                1.0 / 16.0, total, mismatches, inputs_ok ? "true" : "false", nonfinite_max);
            fclose(f);
        }
        char buf[1024];
        snprintf(buf, sizeof(buf),
            "%s\"%s\": {\"b\": %d, \"ctas\": %u, \"kernel_ms\": %.5f, \"p10_ms\": %.5f, \"p90_ms\": %.5f, \"min_ms\": "
            "%.5f, \"noise_floor\": %.5f, \"repeat_mismatches\": %d, \"calls\": %d, \"nonfinite\": %llu, "
            "\"inputs_unchanged\": %s, \"out_hash\": \"%016llx\"}",
            first_case ? "" : ", ", cname.c_str(), p.b, g.x * g.y * g.z, med, p10, p90, mn, (p90 - p10) / med,
            mismatches, total, nonfinite_max, inputs_ok ? "true" : "false", out_hash1);
        json += buf;
        first_case = false;
        release(p);
    }
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
        "}, \"kernel\": {\"regs\": %d, \"local_bytes\": %zu, \"dyn_smem\": %d, \"occupancy_ctas_per_sm\": %d, "
        "\"store_o_from_regs\": %d}, \"ok\": %s}",
        fa.numRegs, fa.localSizeBytes, smem, occ, from_regs, all_ok ? "true" : "false");
    json += hdr;
    printf("CASES_JSON %s\n", json.c_str());
    return all_ok ? 0 : 1;
}
