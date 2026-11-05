# Idea 215 / 196 Stage 0: results (branch 131, 11-04)

Main 2fa8e034 (sync_main: already current). engine.json on main = `{"enable_tokenization_cache": true}`,
so there is no moe_tp/moe_ep override and **EP4 is primary**; EP2xTP2 (192 x 1152) is context.
Grid cap (dense_gemm idea 246) is **OFF** on main: `launch_sm120_gemm_kernel`
(cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh:917)
launches `gridDim = num_device_sms` (188), and Kineto shows grid [188,1,1] block [384,1,1] for both
shared GEMMs, against 18 (gate_up, N=2304) / 40 (down, N=5120) output tiles of 32x128 at M <= 4.

Jobs: 1109/1112 smoke; 1122 first full run; **1138 final** (commit 058bbe52, 1 GPU austin:3,
GPU-ea4de9e1, SM 2362-2422 MHz by UUID, every case block dispersion <= 3% (ep4 worst A/B/C 1.49%;
ep4 M1_D3 re-run once by the harness's auto-retry), replay noise 0.2-1.3%).

Method: one 40-layer CUDA graph per arm, production fork/join (maybe_execute_in_parallel under
with_multi_stream), production Sm12xW4a8DecodeState.run with forced-D routing on 5 weight sets
(no expert reuse across layers) + 512 MB read flush per replay; shared = real FP8 block-scale GatedMLP
(40 copies), x0.5. Span = CUDA-event time per replay / 40 after a 30-replay warm-up (median of 10 per
block, median of 6 rotated-order blocks). Kineto spans are inflated by ~27 us/layer (CUPTI) and are
used only for per-node durations; per-node durations include PDL waits (method 114).

Checks: D/E (shared alone) replays bitwise equal to their first replay in every case; routed arms
are bitwise at M>=2 for most D but differ by up to 0.19 (bf16 abs) where tokens hit several local
experts (FC2 finalize scatter-add order, main's own spread); B - (A + D) <= 0.19; all finite.

## EP4 table (job 1138), us per layer (CUDA events)

| case | A routed | B +shared1152 | C +shared640 | D shared alone | B-A | B-C | C-A |
|---|---|---|---|---|---|---|---|
| M1_D0 | 7.55 | 33.10 | 29.85 | 37.91 | 25.55 | 3.25 | 22.30 |
| M1_D1 | 28.85 | 40.07 | 35.48 | 39.24 | 11.22 | 4.58 | 6.64 |
| M1_D2 | 38.27 | 50.81 | 44.67 | 39.24 | 12.54 | 6.14 | 6.40 |
| M1_D3 | 50.56 | 62.46 | 57.47 | 37.51 | 11.91 | 4.99 | 6.91 |
| M1_D4 | 62.26 | 74.19 | 69.20 | 37.55 | 11.93 | 4.99 | 6.93 |
| M1_D5 | 79.21 | 91.33 | 86.27 | 37.52 | 12.13 | 5.06 | 7.07 |
| M1_D6 | 89.14 | 101.68 | 96.36 | 37.53 | 12.54 | 5.33 | 7.22 |
| M2_D0 | 7.58 | 32.41 | 29.49 | 37.05 | 24.83 | 2.92 | 21.91 |
| M2_D1 | 28.39 | 39.60 | 34.97 | 37.07 | 11.21 | 4.63 | 6.58 |
| M2_D2 | 38.07 | 50.87 | 44.77 | 37.05 | 12.80 | 6.09 | 6.71 |
| M2_D3 | 50.54 | 62.49 | 57.37 | 37.06 | 11.95 | 5.12 | 6.83 |
| M2_D4 | 62.46 | 74.50 | 69.35 | 37.06 | 12.04 | 5.15 | 6.89 |
| M2_D6 | 89.09 | 101.73 | 96.41 | 37.07 | 12.65 | 5.32 | 7.32 |
| M2_D8 | 112.18 | 125.49 | 120.27 | 37.06 | 13.31 | 5.22 | 8.09 |
| M4_D1 | 28.47 | 40.06 | 35.05 | 36.74 | 11.59 | 5.02 | 6.57 |
| M4_D2 | 38.02 | 51.53 | 45.39 | 36.74 | 13.52 | 6.14 | 7.37 |
| M4_D4 | 62.65 | 74.72 | 69.61 | 36.76 | 12.08 | 5.12 | 6.96 |
| M4_D6 | 89.09 | 101.86 | 96.41 | 36.74 | 12.77 | 5.45 | 7.32 |
| M4_D8 | 112.43 | 125.72 | 120.73 | 36.74 | 13.29 | 4.99 | 8.30 |
| M4_D10 | 140.39 | 153.98 | 149.38 | 36.76 | 13.59 | 4.61 | 8.98 |
| M4_D12 | 171.39 | 180.28 | 176.51 | 36.76 | 8.88 | 3.76 | 5.12 |

Arm E (shared 640 cols alone) = 29.9-31.6 us/layer. EP2xTP2 (context, job 1138): B-A 11.4-14.0 us/layer at
D>=1 (23.8-24.4 at D=0), B-C 4.1-5.2, D 36.8-37.5.

## Per-node split, EP4, job 1138 (us per layer, Kineto, includes PDL waits)

Shared, arm D (alone) / arm B (beside routed):
- M1_D2: quant 3.16/2.91, gate_up GEMM 23.03/30.73, silu_and_mul 0.89/2.53, quant 2.15/2.65,
  down GEMM 8.80/9.24, x0.5 0.89/0.91.
- M4_D4: quant 3.22/2.99, gate_up 20.76/28.22, silu 0.87/1.09, quant 2.13/2.68, down 8.53/11.94,
  x0.5 0.88/1.16.
- Both GEMMs: grid [188,1,1] block [384,1,1]; output tiles 18 (gate_up) / 40 (down) at M <= 4.
Routed, arm A / arm B: M1_D2 fc1 22.25/27.49, fc2 13.46/14.19; M4_D4 fc1 38.81/50.20, fc2 21.21/21.35
(route_assign 1.6, q0_route 1.8-1.9, unchanged). FI fc1/fc2 also launch 188 x 384 persistent CTAs.

## stage0_expmax.py (job 1138), EP4, E[max_r] x 40 ms/step vs window 910 ITL (c1 6.86 / c2 7.88 / c4 9.44)

REAL generator calibrated to fact 822's c4 histogram: (w 0.85, Dirichlet alpha 0.014) gives
1.32/83.02/15.66 vs 1.5/82.8/15.7 (max off 0.22 pp, OK). The brief's original (0.5, 2.0) gives
0.15/45.33/54.52 (40 pp off), so it was replaced.

| gen M | MAIN | SHARD (diff +- SE) | SKEW (diff +- SE) |
|---|---|---|---|
| REAL M1 | 2.410 (35.1% ITL) | 2.194 (-0.216 +- 0.002) | 1.987 (-0.423 +- 0.001) |
| REAL M2 | 3.044 (38.6%) | 2.837 (-0.207 +- 0.000) | 2.595 (-0.449 +- 0.001) |
| REAL M4 | 3.955 (41.9%) | 3.744 (-0.211 +- 0.000) | 3.479 (-0.476 +- 0.001) |
| UNIF M1 | 2.412 | 2.196 (-0.216) | 1.989 (-0.423) |
| UNIF M4 | 5.240 | 5.043 (-0.197) | 4.746 (-0.494) |

EP2xTP2 (SHARD only): -0.20 ms/step at M1/M2/M4.

VERDICT: BUILD SKEW: SHARD: UNIFORM M1 -0.2162+-0.0018 ms, REAL M1 -0.2164+-0.0018 ms, UNIFORM M4
-0.1969+-0.0012 ms, REAL M4 -0.2112+-0.0004 ms; SKEW: UNIFORM M1 -0.4232+-0.0013 ms, REAL M1
-0.4228+-0.0013 ms, UNIFORM M4 -0.4938+-0.0013 ms, REAL M4 -0.4761+-0.0011 ms
