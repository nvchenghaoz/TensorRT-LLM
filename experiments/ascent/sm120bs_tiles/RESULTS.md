# Branch 204 (idea 235): SM120 FP8 block-scale GEMM/BMM tile heuristic — results

Builds on branch 56 (TileN 32/64 builder variants, commit 2968503b). Base 1d696e72 (pre-landing-61, so
`fused_pmap_gemm_fma_mtile_v3` counts 0 in BOTH arms; the branch's own symbols are the build receipt).

## Change
`fp8_blockscale_gemm_kernel.cuh`: `pick_sm120_blockscale_tile` (M > 64 only; M <= 64 keeps <32,128,4>), used by both
`gemm_dispatch_sm120` (shared expert gate_up/down via fp8_block_scaling_gemm) and `strided_batch_gemm_dispatch_sm120`
(wo_a BMM). Candidates <64,128,4> (prod, wins ties), <32,64,4>, <64,64,4>, <96,128,2>, <128,128,2>; cost =
ceil(tiles/SMs) * (k_tiles * ktile_cost + 3.5 us), ktile_cost (us per 128-K tile per CTA) 0.72/0.41/0.53/0.90/1.08
fitted on sweep 1833. Narrow TileN only when N % 64 == 0. Kill switch TRTLLM_SM120_FP8_BLOCKSCALE_TILE_HEURISTIC=0.
Branch 56's strided M<=64 TileN rule is removed (decode M<=64 unchanged vs main; wo_a decode M<=8 runs b12x anyway).

## Sweep (job 1833, harness.cu/sweep.py, standalone, cold-L2 graph), us/call; all 10 configs bitwise == prod at
M = 1,4,16,64,96,128,200,256,512,1024,2048 for all three shapes
| shape | M | prod 64x128x4 | best |
|---|---|---|---|
| gate_up [M,5120]x[2304,5120] | 128 | 26.73 | 32x64x4 16.58 |
| | 1024 | 61.84 | 128x128x2 45.01 |
| down [M,1152]x[5120,1152] | 128 | 11.57 | 64x64x4 8.57 |
| | 1024 | 40.73 | 128x128x2 30.16 |
| wo_a 2x[M,4096]x[1024,4096] | 128 | 21.83 | 32x64x4 13.42 |
| | 2048 | 68.21 | 96x128x2 57.65 |
64x256x2 and 32x32x4 never win; 64x32x4 ties 32x64x4. The model picks the measured best at every swept M >= 128.

## Pair 1863 (base) / 1864 (branch), same slot austin:1, op level (torch ops), cold L2, 2 passes (A/A), us/call
| case | base | branch | TFLOPS base -> branch |
|---|---|---|---|
| gate_up M128 | 26.72 | 16.12 | 113 -> 187 |
| down M128 | 11.50 | 8.63 | 131 -> 175 |
| wo_a M128 | 21.45 | 13.17 | 100 -> 163 |
| gate_up M1024 | 61.35 | 46.09 | 394 -> 524 |
| down M1024 | 42.02 | 31.38 | 287 -> 385 |
| wo_a M2048 | 72.36 | 58.08 | 475 -> 592 |
| L20-39 layer (sum M128) | 59.67 | 37.92 | -36% |
| L0-19 layer (gate_up/down 1024 + wo_a 2048) | 175.73 | 135.55 | -23% |
Unchanged picks (gate_up M512, down M256, wo_a M512) tie. Output hashes identical across arms for all 15 cases.
Per c1 request (20 layers each): L20-39 -0.435 ms (shared gate_up/down -0.269, wo_a -0.166, all on the main
stream / tail graph); L0-19 wo_a M2048 -0.286 ms on path; L0-19 shared gate_up/down -0.518 ms serial, of which only
the part not hidden by the MoeShared aux-stream overlap (landing 60) counts.
Served per-call cross-check (analysis 326, profile 1581): L0-19 170 us/layer, L20-39 56.4 us/layer vs 175.7/59.7 here.

## Correctness
Check (op_bench.py --check; in both pair arms and in standalone job 1877): every call of every replay (5 + 30, NaN
poisoned) bitwise == the unmodified production <TileM,128,4> kernel at M = 1,2,4,8,16,64,96,128,200,256,512,1024,2048
for gate_up, down, and wo_a (strided D); M=128 and M=1024 shared chain (quant, gate_up, silu*up, quant, down)
CUDA-graph replay == eager every replay, chain hashes equal between base and branch.
