# SM120 port of MiniMax-M3 mxfp8LowMGemm for the CSA2 decode input projections (branch 86, idea 106)

Base: main ee1e1cb9. GPU: RTX PRO 6000 Blackwell Server (SM120, 188 SMs, 128 MB L2, DRAM ~1507 GB/s).
Kernel: `trtllm::csa2_mxfp8_low_m_gemm` (cpp/tensorrt_llm/kernels/minimaxM3/mxfp8LowMGemm.cu,
`csa2_mxfp8_low_m_gemm_kernel<WARPS, CLUSTER_K, RT, STAGES>`). Each CTA bulk-copies (cp.async.bulk) its 16*RT weight
rows x K/CLUSTER_K columns + UE8M0 scale chunks to smem BEFORE griddepcontrol.wait, then quantizes its activation K
slice in-kernel (byte-equal to trtllm::mxfp8_quantize(x, True), exp2f_rcp output scale) or stages a pre-quantized one
(PreqArgs), runs m16n8k32 e4m3 mma.sync with C zeroed per 32-k block and pow2 scales in FP32, then a fixed-order
warp + cluster split-K reduction (peers st.async their M live columns into the root's smem over DSMEM).
Config tuple = (warps, cluster_k, row_tiles[, stages]); cluster_k > 1 = cluster split-K.

Arms: "main" = mxfp8_quantize + flashinfer_mm_mxfp8 (b12x, core Sm120B12xBlockScaledDenseGemmKernel);
"b12x only" = pre-quantized input, b12x alone; "port" = csa2 kernel with in-kernel quant (M3 port);
"preq only" = csa2 kernel reading a pre-quantized input (quantize kernel not in the chain).

## Job 887 / 897 DIAG (slot austin:4 / austin:5), cold-L2 PDL CUDA-graph chain over weight copies > 2x L2, us per call

| shape | M | floor | main | b12x only cold | b12x only hot | quant only | port cfg | port cold | port hot | port cold no-PDL | preq only cold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1792x5120 | 1 | 6.28 | 13.08 (887) / 13.16 | 10.95 / 11.03 | 7.43 / 7.57 | 2.13 | 8,2,2 | 10.26 / 10.31 | 5.51 / 5.61 | 11.16 / 11.22 | 10.31 / 10.34 |
| 1792x5120 | 1 | | | | | | 4,4,2 | 10.65 / 10.68 | 5.15 / 5.22 | 11.46 / 11.50 | 10.75 / 10.80 |
| 1792x5120 | 4 | 6.28 | 13.92 | 11.37 | 7.59 | 2.13 | 4,4,2 | 12.99 | 7.99 | 13.93 | 13.49 |
| 8192x1280 | 1 | 7.18 | 11.08 | 9.27 | 6.24 | 1.50 | 4,1,2 | 11.08 | 5.20 | 11.58 | 11.16 |
| 8192x1280 | 1 | | | | | | 4,1,1 | 11.61 | 6.48 | 12.07 | 11.00 |
| 8192x1280 | 4 | 7.18 | 11.06 | 9.17 | 6.24 | 1.52 | 4,1,2 | 13.02 | 7.45 | 13.41 | 13.77 |
| 8192x1280 | 4 | | | | | | 4,1,1 | 12.36 | 7.32 | 12.68 | 12.66 |
| 12288x1280 | 1 | 10.76 | 14.77 | 12.93 | 6.28 | 1.58 | 4,1,1 | 15.20 | 8.39 | 15.64 | 15.41 |
| 12288x1280 | 4 | 10.76 | 14.71 | 12.91 | 6.28 | 1.60 | 4,1,1 | 15.66 | 9.90 | 16.04 | 16.46 |
| 4096x1280 | 1 | 3.59 | 7.57 | 5.76 | 3.58 | 1.31 | 4,1,2 | 6.67 | 3.72 | 7.00 | 6.64 |
| 4096x1280 | 1 | | | | | | 4,1,1 | 6.59 | 3.73 | 6.85 | 6.67 |
| 4096x1280 | 4 | 3.59 | 7.57 | 5.73 | 3.75 | 1.45 | 4,1,2 | 7.13 | 4.47 | 7.45 | 7.32 |
| 4096x1280 | 4 | | | | | | 4,1,1 | 7.09 | 4.53 | 7.17 | 7.41 |

(887 crashed after its first row on an M=4 config that does not fit 99 KB; 897 is the full rerun.)

## Sweep (job 835 austin:4, job 925 austin:5): best port config per shape x M vs main (us, chain)

| shape | M=1 main / port (cfg) | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| 1792x5120 | 13.16 / 10.54 (4,2,2) [8,2,2: 10.59] | 13.94 / 10.60 (8,2,2,2) | 13.98 / 12.29 (8,4,1) | 13.98 / 13.37 (8,4,1) |
| 8192x1280 | 10.30 / 10.86 (4,1,2) | 10.31 / 10.90 (4,1,2) | 10.28 / 11.07 (8,1,4) | 10.26 / 13.35 (8,1,2) |
| 12288x1280 | 14.01 / 14.69 (4,1,1) (835) | 14.09 / 14.80 | 14.00 / 15.16 | 13.97 / 16.55 |
| 4096x1280 | 6.79 / 6.26 (4,1,2) (835) | 6.80 / 6.26 | 6.78 / 6.36 (8,1,2) | 6.80 / 7.04 |

Pre-quantized input (quantize kernel + port) is always ~2-3 us slower than in-kernel quant, i.e. the in-kernel
quantization is free. Staging the K slice into 2-4 mbarrier stages (STAGES template) changed nothing (8,2,2,1 10.59
vs 8,2,2,2 10.60 us).

## Where the port's time goes (job 905, per-CTA %globaltimer, ~256 ns resolution, hot L2, no PDL)

4096x1280 (4,1,2): span 3.58 us = weights+x arrive at 2.30 us, mainloop 0.51, epilogue 0.26.
8192x1280 (4,1,2): span 5.12 = arrival 3.07 (last 3.84), mainloop 0.77, epilogue 0.26.
1792x5120 (8,2,2): span 5.38 = arrival 3.33, mainloop 1.02, epilogue 0.51.
More CTAs per SM is slower, not faster: (4,2,1) 8192 hot span 8.96 with last CTA start at 6.14 us; (4,8,1) 1792
hot 8.70 with last start 5.89 (7 CTAs/SM of ~14 KB each). Per-SM L2->smem ingest of the bulk copies is ~20 GB/s.
Cold numbers of job 905 are polluted by the dirty-line write-back of its flush (add_ on 256 MB) and are not used.

## Conclusion

The M3 design (stage the whole CTA slice, then compute) wins only on wq_a||wkv [1792,5120], where b12x runs just
28 CTAs; there it saves 2.6-3.3 us per layer at M=1/2 and 1.7 us at M=4. On the K=1280 projections b12x's
pipelined mainloop hides ~3 us of fixed latency the port exposes, so they stay on quantize + b12x.
