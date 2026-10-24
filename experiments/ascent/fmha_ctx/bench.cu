// SM120 e4m3 tiled context FMHA harness (branch 16, idea 45).
// Instantiates fused_multihead_attention::device_flash_attention_nl_tiled with the template arguments of the
// generated fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled
// (setup.py flash_attention_kernel_template, GENERATE_CUBIN so head_interleaved=False, flags 0x1027) from the
// fmha_v2/src tree on the include path, and launches it like the generated launcher: grid (ceil(s/64), h, b),
// 128 threads, Kernel_traits_nl_tiled::BYTES_PER_SMEM dynamic smem.
// Shapes: 24 q / 4 kv heads, head 256, e4m3 Q and paged e4m3 KV (32-token pages, kv scale 1.0), BF16 O, causal over
// past + new tokens, prefix pages aliased across sequences (serving block reuse).
// Every call (warmup and timed) has a reset point first: NaN-poison of O (0xFF bytes) and a 384 MiB L2 read-flush,
// both untimed. Each call is timed with its own CUDA event pair. After every call the output hash is compared with
// call 1 (untimed). Q and KV pool hashes are checked unchanged at the end.
// Output lines: KINFO regs local_bytes dyn_smem occupancy store_o_from_regs
//               RESULT case b ctas median p10 p90 min noise mismatches calls nonfinite inputs_ok out_hash
//               META case b max_blocks q_lens... past_lens...   (with --dump DIR, which also writes O, Q, pool, offsets)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>

// The last q-tile KV split (branch 31, idea 76), when the tree has it: --split -1 (auto, as the TRT-LLM dispatch),
// --split k (k pieces where the split applies), --split 0 (off, the default: the generated kernel alone).
#if __has_include(<fused_multihead_flash_attention_kernel_noloop_tiled_last_tile_split.h>)
#include <fused_multihead_flash_attention_kernel_noloop_tiled_last_tile_split.h>
#define HAVE_LAST_TILE_SPLIT 1
#else
#define HAVE_LAST_TILE_SPLIT 0
#endif

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

static void ck(cudaError_t e, int line)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at line %d: %s\n", line, cudaGetErrorString(e));
        exit(2);
    }
}

#define CK(x) ck((x), __LINE__)

static constexpr uint32_t kFlags = 0x1027u; // LDGSTS q|k|v (0x7), !head_interleaved (0x20), tiled (0x1000)

// The causal kernel traits, and the launcher traits (the launcher takes BYTES_PER_SMEM from the padding traits).
using Kt_causal = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    kFlags | 0x200, 3, true, nv_bfloat16>;
using Kt_launch = fmha::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    kFlags | 0x200, 2, true, nv_bfloat16, 0, 0, 0>;

// Whether the tree has the register-direct O epilogue (0 on main).
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

__device__ __forceinline__ uint64_t mix64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Approximately N(0, 1) values (Irwin-Hall of 4 bytes), quantized to e4m3.
__global__ void fill_e4m3(__nv_fp8_e4m3* p, size_t n, uint64_t seed)
{
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
    {
        uint64_t a = mix64(seed * 0x9E3779B97F4A7C15ULL + i);
        float u = (float) ((a & 0xff) + ((a >> 8) & 0xff) + ((a >> 16) & 0xff) + ((a >> 24) & 0xff)) / 255.f - 2.f;
        p[i] = __nv_fp8_e4m3(u * 1.7320508f);
    }
}

// Outlier inputs (cases with suffix _out, idea 230): Q of every 4th head x4 (peaky softmax rows, logits std ~4) and
// every 509th token row of the KV pool x8 (spike keys / values), saturated to the e4m3 range.
__global__ void outlier_e4m3(__nv_fp8_e4m3* p, size_t n, int row_elems, int period, int stride, float mult)
{
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
    {
        size_t const row = i / row_elems;
        if ((row / stride) % period == 0)
        {
            float v = float(p[i]) * mult;
            v = fminf(fmaxf(v, -448.f), 448.f);
            p[i] = __nv_fp8_e4m3(v);
        }
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
        c += !isfinite(__bfloat162float(p[i]));
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

struct Problem
{
    std::string name;
    int b = 1, h = 24, h_kv = 4, d = 256, tpb = 32;
    std::vector<int> q_lens, past_lens;
    bool alias = true;
    bool outlier = false;
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
    // Block offsets [b][2][max_blocks]: shared K blocks [0, S), shared V blocks [S, 2S), then private K/V pairs.
    p.h_offs.assign((size_t) p.b * 2 * p.max_blocks, 0);
    int cursor = 2 * p.shared_blocks;
    for (int i = 0; i < p.b; ++i)
    {
        for (int j = 0; j < nblk[i]; ++j)
        {
            int32_t kb = j, vb = p.shared_blocks + j;
            if (j >= p.shared_blocks)
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
    // As serving: bmm1 = dequantQ * dequantK / sqrt(d) with kv scale 1.0, [1] = [0] * log2(e); bmm2 = quantO * dequantV.
    float h_scales[4] = {1.f / 16.f, 1.f / 16.f * 1.4426950408889634f, 1.f, 0.f};
    CK(cudaMemcpy(p.scales, h_scales, sizeof(h_scales), cudaMemcpyHostToDevice));
    fill_e4m3<<<1024, 256>>>(reinterpret_cast<__nv_fp8_e4m3*>(p.q), p.q_bytes, seed * 3 + 1);
    fill_e4m3<<<4096, 256>>>(reinterpret_cast<__nv_fp8_e4m3*>(p.pool), p.pool_bytes, seed * 3 + 2);
    if (p.outlier)
    {
        outlier_e4m3<<<1024, 256>>>(reinterpret_cast<__nv_fp8_e4m3*>(p.q), p.q_bytes, p.d, 4, 1, 4.f);
        outlier_e4m3<<<4096, 256>>>(reinterpret_cast<__nv_fp8_e4m3*>(p.pool), p.pool_bytes, p.d, 509, 1, 8.f);
    }
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

static dim3 grid_of(bert::Fused_multihead_attention_params_v2 const& params)
{
    // As the generated launcher: ctas_per_o_row = ceil(d / Cta_tile_o::N) = 1, loop_iters = ceil(s / 64).
    using Cta_tile_o = typename Kt_launch::Cta_tile_o;
    int ctas_per_o_row = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
    int loop_iters = (params.s + 64 - 1) / 64;
    return dim3(loop_iters * ctas_per_o_row, params.h, params.b);
}

static dim3 grid_of(Problem const& p)
{
    return grid_of(p.params);
}

static unsigned long long run_reduce(int which, void const* ptr, size_t bytes, unsigned long long* d_tmp, cudaStream_t st)
{
    CK(cudaMemsetAsync(d_tmp, 0, 8, st));
    if (which == 0)
    {
        hash_buf<<<1024, 256, 0, st>>>(reinterpret_cast<uint64_t const*>(ptr), bytes / 8, d_tmp);
    }
    else
    {
        count_nonfinite<<<1024, 256, 0, st>>>(reinterpret_cast<__nv_bfloat16 const*>(ptr), bytes / 2, d_tmp);
    }
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
    if (!f || fwrite(h.data(), 1, bytes, f) != bytes)
    {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(3);
    }
    fclose(f);
}

static bool make_case(std::string const& name_in, Problem& p)
{
    p.name = name_in;
    std::string name = name_in;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, "_out") == 0)
    {
        p.outlier = true;
        name = name.substr(0, name.size() - 4);
    }
    int const P = 32768;
    if (name == "c1")
    {
        p.q_lens = {2048}; // 1 x 2048 new tokens over 32768 cached (32 x 24 = 768 CTAs)
        p.past_lens = {P};
    }
    else if (name == "c2")
    {
        p.q_lens = {2048, 2048};
        p.past_lens = {P, P};
    }
    else if (name == "c4")
    {
        p.q_lens = {2048, 2048, 2048, 2048};
        p.past_lens = {P, P, P, P};
    }
    else if (name == "c3")
    {
        p.q_lens = {2048, 2048, 2048}; // c4 serving's trio prefill: 3 x 32 x 24 = 2304 CTAs
        p.past_lens = {P, P, P};
    }
    else if (name == "K0")
    {
        p.q_lens = {2048, 2048, 2048, 1984}; // b = 4 mixed-step context part
        p.past_lens = {P, P, P, P};
    }
    else if (name == "K1")
    {
        p.q_lens = {2048, 2048, 2048, 1920, 64}; // c64 mixed step (t = 8128)
        p.past_lens = {P, P, P, P, P};
    }
    else if (name == "K2")
    {
        p.q_lens = {2048, 2048, 2048, 1856, 64}; // c128 mixed step (t = 8064)
        p.past_lens = {P, P, P, P, P};
    }
    else if (name == "c1_1984")
    {
        p.q_lens = {1984}; // 31 x 24 = 744 CTAs (tail-wave probe)
        p.past_lens = {P};
    }
    else if (name == "ragged")
    {
        p.q_lens = {2000, 1}; // row guard
        p.past_lens = {P, P};
    }
    else if (name == "empty")
    {
        p.q_lens = {2048}; // empty cache
        p.past_lens = {0};
        p.alias = false;
    }
    else if (name == "one")
    {
        p.q_lens = {1}; // 1 new token (a single q-tile: never split)
        p.past_lens = {P};
    }
    else if (name == "c1_2000")
    {
        p.q_lens = {2000}; // b = 1, partial last q-tile
        p.past_lens = {P};
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
    int split = 0;
    std::string arms_arg;
    uint64_t seed = 1234;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        std::string a = argv[i];
        if (a == "--cases")
            cases = argv[i + 1];
        else if (a == "--warmup")
            warmup = atoi(argv[i + 1]);
        else if (a == "--iters")
            iters = atoi(argv[i + 1]);
        else if (a == "--dump")
            dump = argv[i + 1];
        else if (a == "--seed")
            seed = strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--split")
            split = atoi(argv[i + 1]);
        else if (a == "--arms")
            arms_arg = argv[i + 1];
    }
    // --arms name:max_batch,...: per case, every round runs each arm once (order rotated by one arm per round), each
    // arm with the dispatch's piece count at its own max batch (--split -1). RESULT lines then end with the arm name.
    std::vector<std::string> arm_names;
    std::vector<int> arm_max_batch;
    for (size_t q = 0; q < arms_arg.size();)
    {
        size_t c = arms_arg.find(',', q);
        if (c == std::string::npos)
            c = arms_arg.size();
        std::string t = arms_arg.substr(q, c - q);
        size_t colon = t.find(':');
        if (colon == std::string::npos)
        {
            fprintf(stderr, "bad --arms item %s (want name:max_batch)\n", t.c_str());
            return 4;
        }
        arm_names.push_back(t.substr(0, colon));
        arm_max_batch.push_back(atoi(t.substr(colon + 1).c_str()));
        q = c + 1;
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
    printf("KINFO %d %zu %d %d %d\n", fa.numRegs, fa.localSizeBytes, smem, occ, from_regs);
    printf("[info] %s sms=%d smem_per_sm=%zu regs=%d local=%zu dyn_smem=%d occupancy=%d store_o_from_regs=%d\n",
        prop.name, prop.multiProcessorCount, prop.sharedMemPerMultiprocessor, fa.numRegs, fa.localSizeBytes, smem, occ,
        from_regs);
    fflush(stdout);

#if HAVE_LAST_TILE_SPLIT
    namespace lts = fused_multihead_attention::last_tile_split;
    using Ws = lts::Workspace<Kt_causal>;
    static_assert(Kt_causal::BYTES_PER_SMEM == Kt_launch::BYTES_PER_SMEM, "");
    // The constants of the TRT-LLM dispatch (last_tile_split_sm120/fmha_v2_e4m3_last_tile_split_sm120.cu). kMaxBatch
    // mirrors that file's constant: build with -DBENCH_MAX_BATCH=<its value> (check_split.py reads it per tree).
#ifndef BENCH_MAX_BATCH
#define BENCH_MAX_BATCH 2
#endif
    constexpr int kMaxPiecesTotal = 384, kMaxPieces = 16, kMaxBatch = BENCH_MAX_BATCH;
    printf("[split] bench kMaxBatch %d\n", kMaxBatch);
    float4* ws_acc = nullptr;
    float4* ws_stats = nullptr;
    int slots = 0;
    if (split != 0)
    {
        CK(cudaMalloc(&ws_acc, sizeof(float4) * Ws::ACC_FLOAT4S_PER_PIECE * kMaxPiecesTotal));
        CK(cudaMalloc(&ws_stats, sizeof(float4) * Ws::STATS_FLOAT4S_PER_PIECE * kMaxPiecesTotal));
        auto piece_fn = lts::fmha_v2_nl_tiled_last_tile_piece_kernel<Kt_causal>;
        auto merge_fn = lts::fmha_v2_nl_tiled_last_tile_merge_kernel<Kt_causal>;
        cudaFuncAttributes fb, fc;
        CK(cudaFuncGetAttributes(&fb, piece_fn));
        CK(cudaFuncGetAttributes(&fc, merge_fn));
        int occ_b = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_b, piece_fn, Kt_causal::THREADS, smem));
        // Slots of launch A: the generated kernel's own occupancy at the launcher's smem.
        slots = prop.multiProcessorCount * occ;
        printf("[split] mode %d slots %d piece_kernel regs %d local %zu occ %d merge_kernel regs %d local %zu\n", split,
            slots, fb.numRegs, fb.localSizeBytes, occ_b, fc.numRegs, fc.localSizeBytes);
        fflush(stdout);
    }
#else
    if (split != 0)
    {
        printf("[split] not available in this tree; running the generated kernel alone\n");
        split = 0;
    }
#endif

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

    bool all_ok = true;
    size_t pos = 0;
    while (pos < cases.size())
    {
        size_t comma = cases.find(',', pos);
        if (comma == std::string::npos)
            comma = cases.size();
        std::string cname = cases.substr(pos, comma - pos);
        pos = comma + 1;
        Problem p;
        if (!make_case(cname, p))
        {
            fprintf(stderr, "unknown case %s\n", cname.c_str());
            return 4;
        }
        build(p, seed);
        dim3 g = grid_of(p);
        int const q_loop_last = (p.params.s + 64 - 1) / 64 - 1;
        // One arm without --arms; with --arms, one per item (same inputs, rotated order per round).
        struct Arm
        {
            std::string name;
            int max_batch = 0;
            // The split: launch A = the generated kernel on s - 64 (q-tiles 0 .. Q-2), then B (pieces) and C (merge).
            int pieces = 0;
            bert::Fused_multihead_attention_params_v2 params_a{};
            dim3 g_a;
            std::vector<float> times;
            unsigned long long out_hash1 = 0, nonfinite = 0;
            int mismatches = 0;
        };
        std::vector<Arm> arms(arm_names.empty() ? 1 : arm_names.size());
        for (size_t ai = 0; ai < arms.size(); ++ai)
        {
            Arm& A = arms[ai];
            A.name = arm_names.empty() ? std::string() : arm_names[ai];
            A.params_a = p.params;
            A.g_a = g;
#if HAVE_LAST_TILE_SPLIT
            A.max_batch = arm_names.empty() ? kMaxBatch : arm_max_batch[ai];
            if (split != 0)
            {
                // Auto: as the TRT-LLM dispatch (last_tile_split_sm120: at most max_batch sequences). A fixed k: any
                // batch.
                int const auto_k = lts::choose_num_pieces(
                    p.b, p.h, p.params.s, 64, slots, kMaxPiecesTotal, kMaxPieces, A.max_batch);
                int const any_k
                    = lts::choose_num_pieces(p.b, p.h, p.params.s, 64, slots, kMaxPiecesTotal, kMaxPieces, INT_MAX);
                A.pieces = split < 0 ? auto_k : (any_k > 0 ? std::min(split, kMaxPiecesTotal / (p.b * p.h)) : 0);
                if (A.pieces > 0)
                {
                    A.params_a.s -= 64;
                    A.g_a = grid_of(A.params_a);
                }
                printf("SPLIT %s pieces %d q_loop_last %d ctas_a %u ctas_b %d%s%s\n", cname.c_str(), A.pieces,
                    q_loop_last, A.g_a.x * A.g_a.y * A.g_a.z, A.pieces * p.b * p.h, A.name.empty() ? "" : " ",
                    A.name.c_str());
                fflush(stdout);
            }
#endif
        }
        auto launch = [&](Arm const& A)
        {
            if (A.pieces > 0)
            {
#if HAVE_LAST_TILE_SPLIT
                KERNEL<<<A.g_a, Kt_launch::THREADS, smem, st>>>(A.params_a);
                CK(lts::launch_last_tile_split<Kt_causal>(p.params, q_loop_last, A.pieces, ws_acc, ws_stats, st));
#endif
            }
            else
            {
                KERNEL<<<g, Kt_launch::THREADS, smem, st>>>(p.params);
            }
        };
        unsigned long long q_hash0 = run_reduce(0, p.q, p.q_bytes, d_tmp, st);
        unsigned long long kv_hash0 = run_reduce(0, p.pool, p.pool_bytes, d_tmp, st);
        int const total = warmup + iters;
        int const n_arms = (int) arms.size();
        for (int it = 0; it < total; ++it)
        {
            for (int j = 0; j < n_arms; ++j)
            {
                // Round it runs the arms starting at arm (it % n): every arm takes every position equally often.
                Arm& A = arms[(it + j) % n_arms];
                // Reset point: NaN-poison O, then read-flush L2 (both untimed).
                CK(cudaMemsetAsync(p.o, 0xFF, p.o_bytes, st));
#if HAVE_LAST_TILE_SPLIT
                if (A.pieces > 0)
                {
                    // The split's workspace too (every piece must write what the merge reads).
                    CK(cudaMemsetAsync(
                        ws_acc, 0xFF, sizeof(float4) * Ws::ACC_FLOAT4S_PER_PIECE * kMaxPiecesTotal, st));
                    CK(cudaMemsetAsync(
                        ws_stats, 0xFF, sizeof(float4) * Ws::STATS_FLOAT4S_PER_PIECE * kMaxPiecesTotal, st));
                }
#endif
                flush_read<<<prop.multiProcessorCount * 4, 512, 0, st>>>(
                    reinterpret_cast<uint4 const*>(flush_buf), flush_bytes / 16, d_sink);
                CK(cudaEventRecord(ev0, st));
                launch(A);
                CK(cudaEventRecord(ev1, st));
                CK(cudaGetLastError());
                CK(cudaEventSynchronize(ev1));
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, ev0, ev1));
                if (it >= warmup)
                {
                    A.times.push_back(ms);
                }
                unsigned long long hsh = run_reduce(0, p.o, p.o_bytes, d_tmp, st);
                A.nonfinite = std::max(A.nonfinite, run_reduce(1, p.o, p.o_bytes, d_tmp, st));
                if (it == 0)
                {
                    A.out_hash1 = hsh;
                }
                A.mismatches += hsh != A.out_hash1;
                if (it == 0 || it == warmup || it == total - 1)
                {
                    printf("[%s%s%s] call %d/%d %.4f ms out_hash %016llx\n", cname.c_str(), A.name.empty() ? "" : " ",
                        A.name.c_str(), it + 1, total, ms, hsh);
                    fflush(stdout);
                }
            }
        }
        bool inputs_ok = q_hash0 == run_reduce(0, p.q, p.q_bytes, d_tmp, st)
            && kv_hash0 == run_reduce(0, p.pool, p.pool_bytes, d_tmp, st);
        for (Arm const& A : arms)
        {
            all_ok = all_ok && A.mismatches == 0 && inputs_ok && A.nonfinite == 0;
            double med = pct(A.times, 0.5), p10 = pct(A.times, 0.1), p90 = pct(A.times, 0.9);
            double mn = *std::min_element(A.times.begin(), A.times.end());
            printf("RESULT %s %d %u %.5f %.5f %.5f %.5f %.5f %d %d %llu %d %016llx%s%s\n", cname.c_str(), p.b,
                g.x * g.y * g.z, med, p10, p90, mn, (p90 - p10) / med, A.mismatches, total, A.nonfinite,
                (int) inputs_ok, A.out_hash1, A.name.empty() ? "" : " ", A.name.c_str());
        }
        fflush(stdout);
        if (!dump.empty() && n_arms > 1)
        {
            printf("[dump] skipped: --dump needs a single arm\n");
        }
        else if (!dump.empty())
        {
            write_file(dump + "/" + cname + "_o.bin", p.o, p.o_bytes);
            write_file(dump + "/" + cname + "_q.bin", p.q, p.q_bytes);
            write_file(dump + "/" + cname + "_pool.bin", p.pool, p.pool_bytes);
            write_file(dump + "/" + cname + "_offs.bin", p.offs, p.h_offs.size() * 4);
            printf("META %s %d %d", cname.c_str(), p.b, p.max_blocks);
            for (int i = 0; i < p.b; ++i)
                printf(" %d", p.q_lens[i]);
            for (int i = 0; i < p.b; ++i)
                printf(" %d", p.past_lens[i]);
            printf("\n");
        }
        release(p);
    }
    return all_ok ? 0 : 1;
}
