// Standalone sweep of fused_pmap_gemm_fma_mtile_v2 vs fused_pmap_gemm_fma_mtile (v1), both against
// fused_pmap_gemm_fma_ksplit<24,1,0,true> (the ks=1 reference). Synthetic inputs at the
// DeepSeek-V4.1-Flash shapes (hc=4, hidden=5120, N=24, K=20480). Bitwise compare of Yp, Rp and
// residual_out on every warmup and timed iteration (outputs NaN-poisoned before each call),
// cold-L2 timing (256 MB memset between calls). Prints one JSON line per (kernel, M).
// The kernel set comes from -DKSET="v2<24,4,false,1>(), ..." at compile time.
#include "mhc_fused_fma.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                           \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

using namespace fused_fma_kernels;
constexpr int HC = 4, HID = 5120, N = 24, K = HC * HID;

using Fn = void (*)(__nv_bfloat16 const*, __nv_bfloat16 const*, float const*, float const*, float const*, float*,
    float*, int, int, int, int, __nv_bfloat16*);

struct Kern
{
    std::string name;
    Fn fn;
    int tn, tm, smem;
};

static uint32_t lcg(uint64_t& s)
{
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
}

static float urand(uint64_t& s)
{
    return (lcg(s) & 0xFFFFFF) / float(0x1000000);
}

template <int TN, int TM, bool S, int MB>
Kern v2()
{
    char buf[128];
    snprintf(buf, sizeof(buf), "v2<%d,%d,%s,%d>", TN, TM, S ? "smem" : "reg", MB);
    int smem = S ? TM * 5 * 256 * 16 : 0;
    Fn f = &fused_pmap_gemm_fma_mtile_v2<TN, TM, S, MB>;
    if (smem > 48 * 1024)
        CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    return Kern{buf, f, TN, TM, smem};
}

template <int TN, int TM>
Kern v1()
{
    char buf[128];
    snprintf(buf, sizeof(buf), "v1<%d,%d>", TN, TM);
    return Kern{buf, &fused_pmap_gemm_fma_mtile<TN, TM>, TN, TM, 0};
}

int main(int argc, char** argv)
{
    std::vector<int> Ms = {128, 2048, 4096, 8192, 33, 127, 2047};
    if (argc > 1)
    {
        Ms.clear();
        char* s = strtok(argv[1], ",");
        while (s)
        {
            Ms.push_back(atoi(s));
            s = strtok(nullptr, ",");
        }
    }
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    int Mmax = *std::max_element(Ms.begin(), Ms.end());

    std::vector<Kern> ks = {KSET};

    uint64_t seed = 12345;
    std::vector<__nv_bfloat16> hx((size_t) Mmax * HID), hr((size_t) Mmax * HC * HID);
    std::vector<float> hpm(Mmax * HC), hcm(Mmax * HC * HC), hw((size_t) N * K);
    for (auto& v : hx)
        v = __float2bfloat16(urand(seed) * 4.f - 2.f);
    for (auto& v : hr)
        v = __float2bfloat16(urand(seed) * 4.f - 2.f);
    for (auto& v : hpm)
        v = urand(seed) * 2.f;
    for (auto& v : hcm)
        v = urand(seed) * 0.5f;
    for (auto& v : hw)
        v = (urand(seed) - 0.5f) * 0.04f;
    __nv_bfloat16 *dx, *dr, *dro, *dro_ref;
    float *dpm, *dcm, *dw, *dyp, *drp, *dyp_ref, *drp_ref;
    CK(cudaMalloc(&dx, hx.size() * 2));
    CK(cudaMalloc(&dr, hr.size() * 2));
    CK(cudaMalloc(&dro, hr.size() * 2));
    CK(cudaMalloc(&dro_ref, hr.size() * 2));
    CK(cudaMalloc(&dpm, hpm.size() * 4));
    CK(cudaMalloc(&dcm, hcm.size() * 4));
    CK(cudaMalloc(&dw, hw.size() * 4));
    CK(cudaMalloc(&dyp, (size_t) Mmax * N * 4));
    CK(cudaMalloc(&drp, (size_t) Mmax * 4));
    CK(cudaMalloc(&dyp_ref, (size_t) Mmax * N * 4));
    CK(cudaMalloc(&drp_ref, (size_t) Mmax * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dr, hr.data(), hr.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpm, hpm.data(), hpm.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcm, hcm.data(), hcm.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
    void* flush;
    size_t flush_bytes = 256u << 20;
    CK(cudaMalloc(&flush, flush_bytes));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    std::vector<__nv_bfloat16> h_ro((size_t) Mmax * HC * HID), h_ro_ref((size_t) Mmax * HC * HID);
    std::vector<float> h_yp(Mmax * N), h_yp_ref(Mmax * N), h_rp(Mmax), h_rp_ref(Mmax);
    bool all_ok = true;
    int HID_v = HID, N_v = N, K_v = K;

    for (int M : Ms)
    {
        CK(cudaMemset(dyp_ref, 0xFF, (size_t) M * N * 4));
        CK(cudaMemset(drp_ref, 0xFF, (size_t) M * 4));
        CK(cudaMemset(dro_ref, 0xFF, (size_t) M * HC * HID * 2));
        fused_pmap_gemm_fma_ksplit<24, 1, 0, true>
            <<<dim3(M, 1, 1), 256>>>(dr, dx, dpm, dcm, dw, dyp_ref, drp_ref, HID, N, K, dro_ref);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_yp_ref.data(), dyp_ref, (size_t) M * N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_rp_ref.data(), drp_ref, (size_t) M * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_ro_ref.data(), dro_ref, (size_t) M * HC * HID * 2, cudaMemcpyDeviceToHost));

        for (auto& kn : ks)
        {
            dim3 grid((M + kn.tm - 1) / kn.tm, N / kn.tn, 1);
            int bad_iters = 0;
            std::vector<float> ts;
            for (int it = 0; it < 3 + iters; it++)
            {
                CK(cudaMemset(dyp, 0xFF, (size_t) M * N * 4)); // NaN poison
                CK(cudaMemset(drp, 0xFF, (size_t) M * 4));
                CK(cudaMemset(dro, 0xFF, (size_t) M * HC * HID * 2));
                CK(cudaMemsetAsync(flush, it & 0xFF, flush_bytes));
                CK(cudaEventRecord(e0));
                void* args[] = {&dr, &dx, &dpm, &dcm, &dw, &dyp, &drp, (void*) &M, (void*) &HID_v, (void*) &N_v,
                    (void*) &K_v, &dro};
                CK(cudaLaunchKernel(reinterpret_cast<void const*>(kn.fn), grid, dim3(256), args, kn.smem, 0));
                CK(cudaEventRecord(e1));
                CK(cudaGetLastError());
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (it >= 3)
                    ts.push_back(ms * 1000.f);
                CK(cudaMemcpy(h_yp.data(), dyp, (size_t) M * N * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_rp.data(), drp, (size_t) M * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_ro.data(), dro, (size_t) M * HC * HID * 2, cudaMemcpyDeviceToHost));
                bool ok = !memcmp(h_yp.data(), h_yp_ref.data(), (size_t) M * N * 4)
                    && !memcmp(h_rp.data(), h_rp_ref.data(), (size_t) M * 4)
                    && !memcmp(h_ro.data(), h_ro_ref.data(), (size_t) M * HC * HID * 2);
                if (!ok)
                    bad_iters++;
            }
            std::sort(ts.begin(), ts.end());
            float med = ts[ts.size() / 2];
            float p10 = ts[(size_t) (0.1 * (ts.size() - 1))], p90 = ts[(size_t) (0.9 * (ts.size() - 1))];
            all_ok &= bad_iters == 0;
            printf("{\"kernel\": \"%s\", \"M\": %d, \"us_med\": %.1f, \"noise\": %.3f, \"bitwise_bad_iters\": %d, "
                   "\"iters\": %d}\n",
                kn.name.c_str(), M, med, (p90 - p10) / med, bad_iters, 3 + iters);
            fflush(stdout);
        }
    }
    printf("{\"passed\": %s}\n", all_ok ? "true" : "false");
    return 0;
}
