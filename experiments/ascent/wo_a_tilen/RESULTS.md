# Branch 56 (idea 16): SM120 FP8 block-scale strided BMM TileN 32/64 — results

Commit 2968503b on base 1d696e72. Closed by top-lead ruling 12:22 PST: branch 48 (idea 73, landing 18) routes
SM120 decode M<=8 wo_a to a grouped b12x GEMM (~9-10 us) and never calls fp8_block_scaling_bmm_out there.

## Change
- `sm120_utils.cuh` SM120BlockScaledBuilder accepts TileN 32/64/128 (PermMmaTileN per TileN; x2 LDSM for B at
  TileN 32; SW64 bf16 D atom at TileN 32; MoE store atom made instantiable). TileN 128 types are unchanged.
- `fp8_blockscale_gemm_kernel.cuh` strided_batch_gemm_dispatch_sm120: TileN 32 if 4 x tiles(TileN 128) <= SMs,
  else 64 if 2 x fits, else 128. wo_a (2 x N=1024, K=4096): M<=128 -> 32, M=256 -> 64, M>=512 -> 128.
- `gemm_dispatch_sm120` (shared expert gate_up/down) is NOT changed.

## Bitwise (exact class)
- Job 479/488 (harness.cu, flashinfer-bundled CUTLASS): TileN 64 and 32 == TileN 128 bitwise at
  M=1,2,4,8,16,32,64,128,256,512,2048, contiguous D and strided D (o_lora[M,2,1024].transpose(0,1), ldd=2048,
  strideD=1024), NaN-poisoned before each of 3 calls; strided == contiguous.
- Job 488 op check (torch.ops.trtllm.fp8_block_scaling_bmm_out of the rebuilt library vs unmodified
  <TileM,128,4> kernel): bitwise on every call (24 calls/replay x 35 replays = 5 warmup + 30) at
  M=1,2,4,8,128,256,512,2048, contiguous and strided D, outputs NaN-poisoned before every replay; CUDA graphs
  captured and replayed at M=1,2,4,8.

## Timing, cold-L2 CUDA graph (24 rotating weight copies), strided D, us/call
| M | TileN128 (old) | TileN64 | TileN32 | op new (job 488) |
|---|---|---|---|---|
| 1 | 18.7-19.0 | 13.4-13.7 | 11.6-12.2 | 10.83 |
| 2 | 18.3-18.5 | 12.9-13.3 | 11.2-11.7 | 11.94 |
| 4 | 18.8-19.3 | 13.4-14.1 | 11.3-12.3 | 10.75 |
| 8 | 18.2-18.3 | 13.5-13.6 | 11.2-11.6 | 10.76 |
| 64 | 17.9-18.0 | 13.3-13.5 | 10.4-10.6 | - |
| 128 (TileM 64) | 21.55 | 16.73 | 13.46 | 12.97 |
| 256 (TileM 64) | 24.23 | 17.11 | 24.91 | 16.72 |
| 512 (TileM 64) | 24.66 | 32.71 | 36.03 | 24.31 (128) |
| 2048 (TileM 64) | 73.0 | 91.9 | 121.3 | 71.72 (128) |
DRAM floor for the 8.4 MB weight: ~5.6 us. No own-base run_gpu_pair was completed (pair 502/503 cancelled by ruling).
