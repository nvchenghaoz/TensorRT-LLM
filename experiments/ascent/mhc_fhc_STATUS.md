# Branch 59 (dense_gemm idea 21): mHC fused_hc M-tiled FMA tactic, status

Code commit 003f4bc9 (base main 1d696e72): new `fused_fma_kernels::fused_pmap_gemm_fma_mtile<TN,TM>`
(mhc_fused_fma.cuh), launcher `mhcFusedHcFmaMtileLaunch` + `pickFhcFmaMtile` (mhcFusedHcKernel.cu, table
(24,2),(24,3),(24,4),(12,2),(12,4),(8,4)), op dispatch `backend==1 && tile_m>1` (mhcOp.cpp), Python tactics
`("fused_half_fma", tn, 1, bs, tm)` appended only in the `not mma_ok and M > 32` branch (mhc_cuda.py).

Evidence so far:
- Job 542 (Step 0 census, profile 373 prefill rank 0): fused_pmap_gemm_fma_ksplit M=2048 37 calls/req, <24,1>,
  421 us mean, 15.59 ms/req; M=128 39 calls/req, <3,1>, 43.7 us, 1.71 ms/req; bigfuse 43 us at 2048.
- Job 541: bitwise check PASSED, all 14 tactics vs (fused_half_fma,24,1,bs,1), all 5 outputs, V4 and V4.1-lagged,
  M=33..8192 incl 2049/4096/8192, every iteration.
- Pair 545/546 (same GPU austin:7), op time us base/new: 512 152.6/145.4, 1024 266.2/235.8, 2048 503.8/407.5,
  4096 923.6/762.1, 8192 1676/1352; <=256 same tactic, unchanged. M=33 read +3.6% (same tactic, 11% noise) and
  check_submission refused on it.
- Pair 568/569 (M=128..8192 only) queued at session end. Next action: when it finishes, check_submission with
  baseline 568, candidate 569, checks [541] and submit. ncu job 500 (Step 0 b) still queued.
- Winner is TN=24, TM=4 at M>=2048, TM=3 at 512/1024. TN=12/8 with TM=4 are slow. A larger TM (6/8) with
  fewer registers per token is the next thing to try.
