# Idea 144 Step 0: fused_pmap_gemm_fma_mtile spill gate (branch 110)

STATUS: the receipt script is committed but the GPU receipt has NOT run. Job 1038 was cancelled
before it got a slot: it is a non-focus branch under the top lead's 12:58 ruling. Next session:
run one 1-GPU job, about 5 min:

    run_gpu_job kind=microbench  ["{python}", "experiments/ascent/mhc_mtile_step0/step0.py"]

then one ncu run on the chosen tactic, for example (24,4) bs=256:

    profile_kernels command=["{python}", "experiments/ascent/mhc_fhc_bench.py", "ncu", "2048,4096,8192",
                             "[\"fused_half_fma\", 24, 1, 256, 4]"]
                    kernel_regex="fused_pmap_gemm_fma_mtile" launch_count=3 sections=full

From the ncu run, read the share of L1TEX sectors that are local (fact 1343 had 22.6% at M=2048) and
the lmem/long_scoreboard stall share.

## What step0.py prints
1. `[built]` / `[built-sass]`: REG/STACK/LOCAL/SHARED for every mtile<TN,TM> and ksplit
   instantiation in the built lib (cuobjdump -res-usage, sm_120 cubin), with SASS LDL/STL/FFMA/LDG counts.
2. `[standalone]`: nvcc -O3 sm_120 -Xptxas -v on explicit instantiations of the same header: stack
   frame, spill store and spill load bytes, and regs. It also reports whether the PTX has a `.local`
   depot with ld/st.local. PTX local = dynamic indexing (fix first, 3821). ptxas-only spill = the live set.
3. `[tactic]`: the autotuner's chosen tactic at M=2048/4096/8192, in V4.1 lagged mode (pre_mix_ext +
   fused norm, as fused_hc_lagged calls it), plus the time of every valid tactic.

## Counted live set per instantiation (code read of mhc_fused_fma.cuh:1019-1170; fp32/u32 regs per thread in the h-loop)
acc TM*TN + sqr TM + x_raw/r_raw 20*TM (packed bf16, live across all 4 j) + new_r 8*TM (per j)
+ w 8 (per n; each W_T load hoisted for ILP adds 8) + 64-bit tok_res/tok_x 4*TM + ~12 misc.
The register cap comes from `__launch_bounds__(256, TM*TN<=48 ? 2 : 1)`: 128 regs when minBlocks=2, else 255.

| (TN,TM) | acc | raw | new_r | w | addr+sqr | misc | total | cap | expected |
|---|---|---|---|---|---|---|---|---|---|
| (24,2) | 48 | 40 | 16 | 8 | 10 | ~12 | ~134 | 128 | small spill (live set) |
| (24,3) | 72 | 60 | 24 | 8 | 15 | ~12 | ~191 | 255 | no spill |
| (24,4) | 96 | 80 | 32 | 8 | 20 | ~12 | ~248 | 255 | at the edge; measured 255 regs + 22.6% local sectors (fact 1343) |
| (12,2) | 24 | 40 | 16 | 8 | 10 | ~12 | ~110 | 128 | fits |
| (12,4) | 48 | 80 | 32 | 8 | 20 | ~12 | ~200 | 128 | heavy spill (live set + launch bound) |
| (8,4)  | 32 | 80 | 32 | 8 | 20 | ~12 | ~184 | 128 | heavy spill (live set + launch bound) |

Dynamic indexing, from the code read: every register array (acc, sqr, tok_res, tok_x, tok_ok, x_raw,
r_raw, new_r, w) is indexed only by `#pragma unroll` loops with constexpr bounds. That includes the
epilogue `acc[t][g*COLS_PER_GROUP+n]` under `if (n < n_cols)`. The h-loop is the only loop left rolled,
and nothing is indexed by h. So I expect no PTX `.local` depot, and the <24,4> spill should come from
the live set (L0-L2). step0.py's `ptx_local_depot` confirms or refutes this.

## Build levers (L0-L3), cheapest first, all exact
- L0: minBlocks=1 for the 128-capped instantiations ((12,4), (8,4), (24,2)). This changes register
  allocation only.
- L1: int32 token offsets (M*4*5120 < 2^31 up to M~100k), saving ~4*TM regs (16 at TM=4).
- L2: stage the thread-private r_raw/x_raw in shared memory ([k][t][tid] uint4; 80 KB at TM=4,
  256 thr), freeing ~80 regs at TM=4. Alternatively, run a two-phase N inside the CTA (n 0..11 then
  12..23, re-deriving new_r from L2-resident inputs) to halve acc.
- L3: 16-byte vector loads for the half-used W_T sectors.
Keep the thread->h mapping and the shuffle/8-warp epilogue: changing either changes the reduction order.
