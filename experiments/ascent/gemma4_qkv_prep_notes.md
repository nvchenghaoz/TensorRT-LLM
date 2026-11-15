# Branch 7 (idea 8): Gemma4 fused QKV prep on SM120 FA2 — notes

Landing-3 (reordered) kernel commits, kept reachable: fc46a63cf424 (on 8070e5d7) and its rebase
fc9eb5f4358f (on a929b157). The exact version is bd15ad02 + 7095109c.

## Root cause of landing 3's failure (no path bug)
Real-activation dump (job 220 / 228, in-process LLM, 50 sliding layers, c=1, packed 16x300, 4x2000,
4x8000, 8x1000, uneven batches, eager decode): routing, dtypes (BF16 Q, FP8 K/V), position_ids and the
cos_sin table were right. The old kernel differed from main's chain from layer 0 on in Q 7.4e-6, K 2.3e-7
and V 1.3e-7 of elements (K/V at most 1 ulp). Two rounding causes (probe job 210):
1. Sum of squares: tl.sum vs FlashInfer CuTe-DSL RMSNormKernel. CuTe (hd 256/512): one warp per row,
   lane t owns cols [8t, 8t+8) (+ [256+8t, ..) at hd 512), per-lane SEQUENTIAL add (bf16 squares are exact
   in fp32, so fma vs mul+add is the same), then shfl.bfly xor 1,2,4,8,16 = adjacent-pair tree.
2. RoPE: at hd 256 Triton/LLVM contracted y1*cos - y2*sin differently from nvcc in
   pos_enc.cuh vec_apply_llama_rope_cos_sin. Matching: fma(x, cos, -(p*sin)) / fma(x, cos, p*sin), launch
   with enable_fp_fusion=False.
rsqrt (tl.math.rsqrt == rsqrt.approx.ftz) and the bf16->fp8 cast (Triton == torch over all finite bf16)
already matched.

## Outcome-0 (job 246): landing-3 kernel vs main, host prefill plan off/on, one process
66 prompts per arm (isl300@1, isl1k@1, isl8k@1 x10, isl1k@16, 3 x isl8k@4, 8 chat). vs main@0:
main@1 66/66 bitwise; main@0 again 66/66; old kernel @0 and @1 IDENTICAL: 66/66 differ, 15 token flips
(7 with margin >1 nat, max 7.24 at isl300@1, c=1), 28/66 over the check's tolerances, max top-20
|dlogprob| 10.6. New exact kernel @0 and @1: 66/66 bitwise. The host plan is not the cause; flips
occur at c=1 too. Sub-ulp prep differences amplify through 60 NVFP4 layers.
