# S0 TABLE (branch 163 / idea 285), job 1489, node smc521ge-0109 austin:7, commit 0642890c

Standalone nvcc -O3 sm_120 -Xptxas -v build of proto.cu (experiments/ascent/mhc_mtile_v2/sweep.py),
kernel-only cold-L2 (256 MB memset between calls) median of 10 timed after 3 warmup, us.
Bitwise vs fused_pmap_gemm_fma_ksplit<24,1,0,true> on Yp, Rp and residual_out, outputs NaN-poisoned
before every one of the 13 calls: 0 bad iterations for EVERY variant at EVERY M (33,128,2047,2048,4096,8192).
v2 args = <TN,TM,RAW_SMEM,MIN_BLOCKS,J_ROLLED>; all v2 have volatile W_T asm loads.

| kernel | regs | stack B | spill st/ld B | M=33 | M=128 | M=2047 | M=2048 | M=4096 | M=8192 |
|---|---|---|---|---|---|---|---|---|---|
| v1<24,4> | 255 | 408 | 832/1300 | 105.5 | 115.7 | 361.5 | 363.5 | 605.2 | 1058.8 |
| v1<24,3> | 255 | 176 | 528/524 | 92.2 | 101.3 | 363.5 | 365.6 | 642.0 | 1143.8 |
| v1<12,2> | 128 | 248 | 432/500 | 52.2 | 68.6 | 410.6 | 411.6 | 707.6 | 1287.2 |
| v2<24,4,reg,1,unrolled> | 255 | 672 | 1252/1820 | 105.5 | 115.7 | 391.2 | 398.3 | 666.6 | 1170.5 |
| v2<24,3,reg,1,unrolled> | 255 | 352 | 832/848 | 87.0 | 95.2 | 373.8 | 379.9 | 666.7 | 1199.1 |
| v2<12,4,reg,1,unrolled> | 255 | 328 | 428/756 | 57.3 | 68.6 | 375.8 | 379.9 | 662.5 | 1205.3 |
| v2<8,4,reg,1,unrolled> | 255 | 192 | 316/432 | 44.0 | 56.3 | 388.1 | 386.0 | 707.6 | 1305.6 |
| v2<6,4,reg,1,unrolled> | 255 | 168 | 300/312 | 37.9 | 50.2 | 404.5 | 403.5 | 812.0 | 1508.4 |
| v2<4,4,reg,1,unrolled> | 255 | 56 | 112/108 | 31.7 | 54.3 | 446.5 | 447.5 | 1060.9 | 2019.3 |
| v2<12,2,reg,1,unrolled> | 255 | 168 | 168/168 | 48.1 | 62.4 | 459.8 | 462.8 | 846.8 | 1623.0 |
| v2<12,2,reg,2,unrolled> | 128 | 456 | 612/920 | 58.3 | 76.8 | 543.7 | 543.7 | 969.7 | 1993.8 |
| v2<12,6,reg,1,unrolled> | 255 | 1072 | 1292/2724 | 89.1 | 93.2 | 541.7 | 543.7 | 1000.4 | 1853.4 |
| v2<8,6,reg,1,unrolled> | 255 | 864 | 1052/2116 | 68.6 | 80.9 | 533.5 | 537.6 | 994.3 | 1881.1 |
| v2<24,4,reg,1,JROLLED> | 255 | 472 | 472/472 | 118.2 | 124.9 | 382.0 | 382.0 | 652.3 | 1158.1 |
| v2<24,3,reg,1,JROLLED> | 255 | 144 | 144/144 | 103.4 | 111.6 | 373.8 | 375.8 | 644.1 | 1150.0 |
| v2<12,4,reg,1,JROLLED> | 255 | 144 | 144/144 | 64.5 | 73.7 | 365.6 | 365.6 | 627.7 | 1125.4 |
| v2<8,4,reg,1,JROLLED> | 255 | 8 | 4/4 | 64.5 | 72.7 | 451.6 | 451.6 | 803.8 | 1434.6 |
| v2<12,6,reg,1,JROLLED> | 255 | 712 | 712/712 | 97.3 | 103.4 | 463.9 | 463.9 | 875.5 | 1643.5 |
| v2<8,6,reg,1,JROLLED> | 255 | 504 | 500/500 | 70.7 | 78.8 | 476.2 | 478.2 | 859.1 | 1588.2 |
| v2<4,4,reg,1,JROLLED> | 255 | **0** | 0/0 | 31.7 | 54.3 | 484.4 | 482.3 | 1059.8 | 2017.3 |
| **v2<24,4,SMEM,1,JROLLED>** | 255 | 16 | 16/16 | 78.9 | 91.1 | **291.8** | **289.8** | **499.7** | **853.0** |
| v2<24,4,SMEM,1,unrolled> | 255 | 848 | 1048/2240 | 103.4 | 113.7 | 447.5 | 451.6 | 816.1 | 1489.9 |
| v2<12,4,SMEM,1,unrolled> | 255 | 640 | 676/1596 | 70.7 | 87.0 | 506.9 | 508.9 | 896.0 | 1662.0 |
| v2<24,3,SMEM,1,unrolled> | 255 | 672 | 684/1724 | 103.4 | 113.7 | 529.4 | 529.4 | 990.2 | 1850.4 |
| v2<8,4,SMEM,1,unrolled> | 255 | 456 | 512/1132 | 52.2 | 70.7 | 527.4 | 527.4 | 951.3 | 1787.9 |
| v2<24,2,SMEM,2,unrolled> | 128 | 992 | 1516/2512 | 103.4 | 121.8 | 803.8 | 801.8 | 1451.0 | 2707.5 |
| v2<12,2,SMEM,2,unrolled> | 128 | 616 | 784/1484 | 70.7 | 95.2 | 848.9 | 855.0 | 1596.4 | 3098.6 |

Readings: (1) only the RAW_SMEM + J_ROLLED combination wins: <24,4> is -20% at M=2048 vs v1<24,4>
(289.8 vs 363.5), -17% at 4096, -19% at 8192, still bitwise. Either knob alone makes things worse:
an unrolled j loop with smem inputs spills 848 B, because ptxas hoists all four j-steps' smem loads.
(2) J_ROLLED alone with register-held inputs gives at best -2..-6% (<12,4>). (3) The only build with
STACK=0 is <4,4,reg,JROLLED>, which is slow. NEXT: <24,4,SMEM,JROLLED> keeps a 16 B stack. Derive
tok_x = tok_res >> 2 in the cp.async issue (frees TM regs), then sweep the SMEM+JROLLED family
(TN 24/12/8/6, TM 2/3/4) for a STACK=0 build. Land exactly one tactic id.

# STATUS branch 148 (idea 212), 04:4x PST 11-05 -- WIP, NOT SUBMITTED

DONE (measured, synced to main 31878478):
- Step 0 receipt, job 1385 (base tree): every v1 mtile instantiation spills. REG/STACK:
  <24,2> 128/400, <24,3> 255/176, <24,4> 255/408, <12,2> 128/248, <12,4> 128/1216, <8,4> 128/1064.
  No PTX .local depot (spills come from ptxas allocation, not dynamic indexing). Full op
  (mtile + bigfuse, cold L2) V4.1-lagged: M=128 tuner (12,1,512,2) 73.8 us (best ksplit<3> 61.4);
  M=2048 tuner (24,..,3) 409.8 us, best (24,4) 406.6; M=4096 (24,4) 762; M=8192 (24,4) 1363.
  (12,4)/(8,4) at the 128-reg cap: 1056-1140 us at M=2048.
- Job 1426 (commit cd73712d): mtile_v2<TN,TM,false,1> (minBlocks=1, int32 offsets, NON-volatile
  W_T loads), ids tile_m = 16+TM. Bitwise PASSED vs ksplit (24,1,bs,1) for every old and new
  tactic at M=33,64,127,128,129,255,512,1000,1024,2047,2048,4096,4099,8192, V4 and V4.1-lagged,
  NaN-poisoned outputs, 8 calls each; M=128 CUDA-graph 12 replays x 7 tactics x 2 modes, 0 bad.
  BUT every v2 build still spilled at 255 regs (STACK: <12,2> 16, <4,4> 536, <6,4> 648, <8,4> 720,
  <12,4> 912, <24,3> 664, <24,4> 1200) and none beat v1 at M>=2048 (e.g. v2 <24,4> 633 us vs
  v1 408). Lesson: dropping `volatile` on the W_T ld.global.L1::evict_last asm let ptxas hoist the
  loads across n/j and blow up the live set; the counted live set (PLAN below) is not what ptxas
  allocates. Only M=128 gained: v2 (12,2) 63.7-65.5 us vs v1 tuner pick 73.8-75.8 (the tuner
  picked it), still above ksplit<3> 61.6.

UNTESTED (committed after 1426):
- W_T loads are volatile again in v2 (same body as v1).
- New template knob J_ROLLED (ids tile_m = 32+TM): `#pragma unroll 1` on the j loop. Exact:
  nothing register-resident is indexed by j. Aim: stop ptxas overlapping j+1's new_r with j's GEMM.

NEXT (exact command order):
1. Fast loop without the lib rebuild: run_gpu_job microbench
   ["{python}", "experiments/ascent/mhc_mtile_v2/sweep.py", "2048,128,33,2047", "10"]
   It prints ptxas regs/stack/spills per v1/v2 instantiation (incl. jrolled and the RAW_SMEM
   cp.async path) and cold-L2 kernel-only time + bitwise vs ksplit<24,1>. Keep only builds with
   stack 0 in the C++/Python tables (acceptance item 2).
2. ["{python}", "experiments/ascent/mhc_mtile_v2/s1_check.py"] (one job: in-tree rebuild, bitwise,
   repeated-exec NaN-poisoned, M=128 graph, step0 regs + tuner picks). Must print v2_spill_free true.
3. run_gpu_pair both arms ["{python}", "experiments/ascent/mhc_mtile_v2/pair.py", "128,512,2048,8192"].

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
