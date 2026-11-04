# Branch 59 (dense_gemm idea 21): mHC fused_hc M-tiled FMA tactic, status

Code commit 003f4bc9 (base main 1d696e72): new `fused_fma_kernels::fused_pmap_gemm_fma_mtile<TN,TM>`
(mhc_fused_fma.cuh), launcher `mhcFusedHcFmaMtileLaunch` + `pickFhcFmaMtile` (mhcFusedHcKernel.cu, table
(24,2),(24,3),(24,4),(12,2),(12,4),(8,4)), op dispatch `backend==1 && tile_m>1` (mhcOp.cpp), Python tactics
`("fused_half_fma", tn, 1, bs, tm)` appended only in the `not mma_ok and M > 32` branch (mhc_cuda.py).
M<=32 tactic list and the SM-count pruning constants unchanged.

## Step 0 (acceptance 1)
- Census, profile 373 prefill rank 0 (jobs 542, 622), per request:
  - MhcFusedHcRunner sites (attention-side fused_hc_lagged entry + FFN-side fused_hc_lagged; same op, same
    tactic list): M=2048 37 calls, fused_pmap_gemm_fma_ksplit<24,1,0,true> 421.3 us + mhcBigFuse 43.2 us
    = 17.19 ms/req; M=128 39 calls, ksplit<3,1,0,true> grid 128x8 43.7 us + bigfuse 4.5 us = 1.88 ms/req.
  - MhcPreMappingRunner sites (non-deferred entry, i.e. Engram layers): M=2048 4 calls,
    mhcGemmSqrsumFmaKernel<12> 241 us + bigfuse 43 us = 1.14 ms/req; M=128 1 call 48 us. NOT covered by
    this tactic (different runner).
- ncu (cold, one launch each):
  - M=2048 current tactic ksplit<24,1> (job 606): 438 us, DRAM 20.5%, L2 hit 91.8%, memory 66% (L2-side),
    occ 34% (register-limited, 126 regs), stalls long_scoreboard 33.7 >> barrier 6.3; 49% excessive
    sectors (W_T/x loads half-used). Floor ~126 us (fact 336, ~189 MB) -> 3.5x off floor. Not DRAM-bound.
  - M=2048 new mtile<24,4> (job 607): 319 us, DRAM 38.6%, L2 hit 88%, 255 regs WITH local-memory spill
    (22.6% of L1TEX sectors are local), occ 16.6%, long_scoreboard 11.0.
  - M=128 current ksplit<3,1> grid 128x8 (job 610): 50.8 us, DRAM 10.5%, L2 hit 94.6%, memory 68%.
  - bigfuse M=2048: 60.6 us under ncu, DRAM 87% (at floor).

## Correctness (acceptance 2)
- Job 541: bitwise PASSED, all 14 tactics vs (fused_half_fma,24,1,bs,1), all 5 outputs, V4 and V4.1-lagged,
  M=33,48,64,100,128,129,256,511,512,1000,1024,2048,2049,4096,8192, every call of 3 warmup + iters, inputs
  unchanged, residual vs fp64 reference within tolerance.

## Timing (acceptance 3), fresh lease, same GPU austin:2, pair 587 (base) / 588 (branch), op = GEMM+bigfuse
| M | base us (tactic) | new us (tactic) | gain |
| 128 | 62.2 (3,1,512,1) | 61.6 (same) | 0 |
| 256 | 94.2 (3,1,512,1) | 94.2 (same) | 0 |
| 512 | 155.7 (3,1,256,1) | 148.4 (24,1,256,3) | 4.7% |
| 1024 | 270.6 (4,1,256,1) | 241.6 (24,1,256,3) | 10.7% |
| 2048 | 510.2 (24,1,256,1) | 415.7 (24,1,256,4) | 18.5% |
| 4096 | 958.5 (24,1,256,1) | 776.4 (24,1,256,4) | 19.0% |
| 8192 | 1739.8 (24,1,256,1) | 1386.5 (24,1,256,4) | 20.3% |
Prefill saving at c1: 37 calls x ~94 us = ~3.5 ms/request (~2% of the validated c1 TTFT 178.8 ms).
Earlier pair 545/546 (session 3, austin:7) agrees.

## Next (follow-up idea, not this branch)
- mtile<24,4> spills (255 regs, local memory 22.6% of sectors). A spill-free TM=4 (fewer accumulators per
  thread, or split TN across warps) should get closer to the ~126 us floor (still 319 us now).
- The 49% excessive sectors on the x/W_T loads exist in both kernels; vectorized 16-B loads would fix them.
- MhcPreMappingRunner's mhcGemmSqrsumFmaKernel<12> at M=2048 (241 us x 4/req) could take the same M-tiling.
