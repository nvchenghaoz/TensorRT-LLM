// Idea 348 S0.S proto-only kernel: fused_pmap_gemm_fma_mtile_v3x<TM, NG, MIN_BLOCKS>.
// Identical to the tree's fused_pmap_gemm_fma_mtile_v3 (mhc_fused_fma.cuh) except (A): TM need not
// be a multiple of 8. The pmap phase keeps v1's token-to-warp map (warp p takes tokens p, p+8, ...)
// and skips tokens t >= TM; NT_R = ceil(TM / 8). When TM % 8 == 0 every guard is compile-time true,
// so the code is v3's. Same per-token chains, so bitwise equal to v1 / ksplit ks = 1.
// Include after mhc_fused_fma.cuh (uses mtile_v3_ld_stream).
#pragma once

namespace fused_fma_kernels
{

template <int TM, int NG, int MIN_BLOCKS>
__launch_bounds__(256, MIN_BLOCKS) __global__
    void fused_pmap_gemm_fma_mtile_v3x(__nv_bfloat16 const* __restrict__ residual_in,
        __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix,
        float const* __restrict__ comb_mix, float const* __restrict__ W_T, float* __restrict__ Yp,
        float* __restrict__ Rp, int M, int hidden_size, int N, int K, __nv_bfloat16* __restrict__ residual_out)
{
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BF16_VEC = 8;
    constexpr int N_TOTAL = 24;
    constexpr int H_STEP = BLOCK_SIZE * BF16_VEC;
    constexpr int VW_SPAN = WARP_SIZE * BF16_VEC;
    static_assert(NUM_WARPS % NG == 0 && N_TOTAL % NG == 0, "NG must divide 8 and 24");
    constexpr int NT_R = (TM + NUM_WARPS - 1) / NUM_WARPS; // pmap tokens per thread (upper bound)
    constexpr bool TM_FULL = (TM % NUM_WARPS) == 0;
    constexpr int A_GROUPS = NUM_WARPS / NG;
    static_assert(TM % A_GROUPS == 0, "TM must divide into token groups");
    constexpr int TMT = TM / A_GROUPS;
    constexpr int TNT = N_TOTAL / NG;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    int const tok0 = blockIdx.x * TM;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    int const a_grp = warp_id / NG;
    int const n_grp = warp_id % NG;

    extern __shared__ float4 s_nr[];
    __shared__ float s_part[NUM_WARPS][TM][N_TOTAL];
    __shared__ float s_sqp[NUM_WARPS][TM];
    __shared__ float s_post[TM][HC_MULT];
    __shared__ float s_comb[TM][HC_MULT][HC_MULT];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    for (int i = tid; i < TM * HC_MULT; i += BLOCK_SIZE)
    {
        int const t = i / HC_MULT;
        int const tok = min(tok0 + t, M - 1);
        s_post[t][i % HC_MULT] = post_mix[tok * HC_MULT + i % HC_MULT];
    }
    for (int i = tid; i < TM * HC_MULT * HC_MULT; i += BLOCK_SIZE)
    {
        int const t = i / (HC_MULT * HC_MULT);
        int const rc = i % (HC_MULT * HC_MULT);
        int const tok = min(tok0 + t, M - 1);
        s_comb[t][rc / HC_MULT][rc % HC_MULT] = comb_mix[tok * HC_MULT * HC_MULT + rc];
    }
    for (int i = tid; i < NUM_WARPS * TM * N_TOTAL; i += BLOCK_SIZE)
        (&s_part[0][0][0])[i] = 0.0f;
    for (int i = tid; i < NUM_WARPS * TM; i += BLOCK_SIZE)
        (&s_sqp[0][0])[i] = 0.0f;

    int tok_res[NT_R];
    bool tok_ok[NT_R];
#pragma unroll
    for (int i = 0; i < NT_R; i++)
    {
        int const t = warp_id + NUM_WARPS * i;
        tok_ok[i] = (tok0 + t) < M;
        tok_res[i] = min(tok0 + t, M - 1) * HC_MULT * hidden_size;
    }
    // (A) pmap slot i of this warp exists (t = warp_id + 8 i < TM). Compile-time true when TM % 8 == 0.
    auto slot_in = [&](int i) { return TM_FULL || (warp_id + NUM_WARPS * i) < TM; };

    uint4 raw[NT_R][HC_MULT + 1];
    auto load_raw = [&](int hh)
    {
#pragma unroll
        for (int i = 0; i < NT_R; i++)
        {
            if (slot_in(i))
            {
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                    raw[i][k] = mtile_v3_ld_stream(&residual_in[tok_res[i] + k * hidden_size + hh]);
                raw[i][HC_MULT] = mtile_v3_ld_stream(&x_in[(tok_res[i] >> 2) + hh]);
            }
        }
    };

    float const* const w_row = W_T + n_grp * TNT * K;
    float wcur[TNT][BF16_VEC];
    auto load_w = [&](float (&wd)[TNT][BF16_VEC], int hh, int jj)
    {
#pragma unroll
        for (int n = 0; n < TNT; n++)
        {
            float const* wp = w_row + n * K + jj * hidden_size + hh;
            asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(wd[n][0]), "=f"(wd[n][1]), "=f"(wd[n][2]), "=f"(wd[n][3])
                         : "l"(wp));
            asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(wd[n][4]), "=f"(wd[n][5]), "=f"(wd[n][6]), "=f"(wd[n][7])
                         : "l"(wp + 4));
        }
    };

    float acc[TMT][TNT];
    float sqr[NT_R];
#pragma unroll
    for (int i = 0; i < TMT; i++)
#pragma unroll
        for (int n = 0; n < TNT; n++)
            acc[i][n] = 0.0f;
#pragma unroll
    for (int i = 0; i < NT_R; i++)
        sqr[i] = 0.0f;

    int vw = 0;
    int s = 0;
    int h = lane * BF16_VEC;
    bool valid = h < hidden_size;
    bool const any_step = hidden_size > 0;
    if (valid)
    {
        load_raw(h);
        load_w(wcur, h, 0);
    }
    __syncthreads();

    while (any_step)
    {
        if (valid)
        {
#pragma unroll
            for (int i = 0; i < NT_R; i++)
            {
                if (!slot_in(i))
                    continue;
                int const t = warp_id + NUM_WARPS * i;
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    float new_r[BF16_VEC];
                    float const pmj = s_post[t][j];
                    __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&raw[i][HC_MULT]);
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        float2 f = __bfloat1622float2(xp[v]);
                        new_r[2 * v + 0] = pmj * f.x;
                        new_r[2 * v + 1] = pmj * f.y;
                    }
#pragma unroll
                    for (int k = 0; k < HC_MULT; k++)
                    {
                        float const ckj = s_comb[t][k][j];
                        __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&raw[i][k]);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                        {
                            float2 f = __bfloat1622float2(rp[v]);
                            new_r[2 * v + 0] = fmaf(ckj, f.x, new_r[2 * v + 0]);
                            new_r[2 * v + 1] = fmaf(ckj, f.y, new_r[2 * v + 1]);
                        }
                    }
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        sqr[i] = fmaf(new_r[v], new_r[v], sqr[i]);
                    if (tok_ok[i])
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[2 * v], new_r[2 * v + 1]));
                        *reinterpret_cast<uint4*>(residual_out + tok_res[i] + j * hidden_size + h) = packed;
                    }
                    s_nr[((j * TM + t) * 2 + 0) * WARP_SIZE + lane]
                        = make_float4(new_r[0], new_r[1], new_r[2], new_r[3]);
                    s_nr[((j * TM + t) * 2 + 1) * WARP_SIZE + lane]
                        = make_float4(new_r[4], new_r[5], new_r[6], new_r[7]);
                }
            }
        }

        int nvw = vw;
        int ns = s + 1;
        if (vw * VW_SPAN + ns * H_STEP >= hidden_size)
        {
            nvw = vw + 1;
            ns = 0;
        }
        bool const has_next = nvw < NUM_WARPS && nvw * VW_SPAN < hidden_size;
        int const nh = nvw * VW_SPAN + lane * BF16_VEC + ns * H_STEP;
        bool const nvalid = has_next && nh < hidden_size;
        if (nvalid)
            load_raw(nh);
        __syncthreads();

#pragma unroll 1
        for (int j = 0; j < HC_MULT; j++)
        {
            float wnxt[TNT][BF16_VEC];
            if (j + 1 < HC_MULT)
            {
                if (valid)
                    load_w(wnxt, h, j + 1);
            }
            else if (nvalid)
            {
                load_w(wnxt, nh, 0);
            }
            if (valid)
            {
#pragma unroll
                for (int i = 0; i < TMT; i++)
                {
                    int const t = a_grp * TMT + i;
                    float4 const lo = s_nr[((j * TM + t) * 2 + 0) * WARP_SIZE + lane];
                    float4 const hi = s_nr[((j * TM + t) * 2 + 1) * WARP_SIZE + lane];
                    float const nr[BF16_VEC] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
#pragma unroll
                        for (int n = 0; n < TNT; n++)
                            acc[i][n] = fmaf(nr[v], wcur[n][v], acc[i][n]);
                }
            }
#pragma unroll
            for (int n = 0; n < TNT; n++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    wcur[n][v] = wnxt[n][v];
        }

        if (nvw != vw || !has_next)
        {
#pragma unroll
            for (int i = 0; i < TMT; i++)
            {
#pragma unroll
                for (int n = 0; n < TNT; n++)
                {
#pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        acc[i][n] += __shfl_xor_sync(FULL_WARP_MASK, acc[i][n], off);
                    if (lane == 0)
                        s_part[vw][a_grp * TMT + i][n_grp * TNT + n] = acc[i][n];
                    acc[i][n] = 0.0f;
                }
            }
#pragma unroll
            for (int i = 0; i < NT_R; i++)
            {
                if (!slot_in(i))
                    continue;
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    sqr[i] += __shfl_xor_sync(FULL_WARP_MASK, sqr[i], off);
                if (lane == 0)
                    s_sqp[vw][warp_id + NUM_WARPS * i] = sqr[i];
                sqr[i] = 0.0f;
            }
        }
        __syncthreads();
        if (!has_next)
            break;
        vw = nvw;
        s = ns;
        h = nh;
        valid = nvalid;
    }

    for (int i = tid; i < TM * N_TOTAL; i += BLOCK_SIZE)
    {
        int const t = i / N_TOTAL;
        int const n = i % N_TOTAL;
        if (tok0 + t < M)
        {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                val += s_part[w][t][n];
            Yp[(tok0 + t) * N + n] = val;
        }
    }
    if (tid < TM && tok0 + tid < M)
    {
        float sq = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++)
            sq += s_sqp[w][tid];
        Rp[tok0 + tid] = sq;
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace fused_fma_kernels
