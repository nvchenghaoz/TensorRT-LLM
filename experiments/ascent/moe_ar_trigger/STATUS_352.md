# Idea 352 (branch 220) STAGE 1 = S0 + S1: ceiling for folding the hc_attn mHC K1 into csa2 low-M GEMM A

Base: 9ac8398e (landings 1-77; landing 75 = branch 206 fast K1 PRESENT, TRTLLM_MHC_K1_FAST default 1,
mhc_lagged_split.py:468; TRTLLM_CSA2_A_K1_FOLD absent). Landings 78/79 do not touch this chain (plan 3603 (2)).
Job 2012, node smc521ge-0046, slot austin:1 (GPU-8ef1ab86, NUMA 0), 1 GPU, 36 boundaries (7 with index_wq_b),
P=480. Command:

    {python} experiments/ascent/moe_ar_trigger/chain.py --mode both352 --ckpt {checkpoint} --batches 1 2 4 \
        --arms tree tree_fast0 ideal --ws 2 5 --iters 8 --reps 40 --inner 5

## S1 table (us/boundary median [p10, p90], ABBA, 40 reps x 5 replays)

| M | W (us)       | tree (fast K1)        | tree_fast0            | ideal (no K1)         | tree - ideal | tree - fast0 |
|---|--------------|-----------------------|-----------------------|-----------------------|--------------|--------------|
| 1 | 1.84 (p50)   | 25.33 [25.28, 25.36]  | 25.40 [25.37, 25.43]  | 22.25 [22.23, 22.29]  | 3.08 [3.00, 3.13] | -0.07 [-0.14, -0.01] (in noise) |
| 1 | 1.77 (p10)   | 25.26 [25.21, 25.29]  | 25.32 [25.28, 25.36]  | 22.19 [22.17, 22.22]  | 3.06 [3.00, 3.12] | -0.07 [-0.15, 0.01] (in noise) |
| 2 | 2.34 (p50)   | 27.86 [27.80, 27.88]  | 28.22 [28.18, 28.23]  | 23.47 [23.43, 23.52]  | 4.39 [4.28, 4.46] | -0.36 [-0.44, -0.30] |
| 2 | 1.89 (p10)   | 27.48 [27.44, 27.52]  | 27.72 [27.68, 27.77]  | 23.43 [23.40, 23.48]  | 4.05 [3.96, 4.13] | -0.25 [-0.33, -0.15] |
| 4 | 2.86 (p50)   | 29.70 [29.67, 29.72]  | 30.10 [30.07, 30.14]  | 23.95 [23.92, 23.99]  | 5.75 [5.67, 5.81] | -0.40 [-0.48, -0.34] |
| 4 | 2.03 (p10)   | 29.31 [29.27, 29.34]  | 29.74 [29.70, 29.77]  | 23.92 [23.89, 23.94]  | 5.39 [5.33, 5.45] | -0.43 [-0.50, -0.36] |

Arm noise (p90-p10)/median <= 0.41% in every cell. Bitwise (check352, same job, 8 iterations, new inputs each,
NaN-poisoned x_prev and outputs, W = 2 and 5): tree, tree_fast0 and ideal A/B outputs (qr, q, swa, index_q;
115 tensors per replay) equal the eager no-overlap reference and each other, 0 mismatches; K1 layer_input
(tree, tree_fast0) and ideal's precomputed layer_input equal the reference; no Triton JIT during capture.

VERDICT: GO. The ceiling vs the fast K1 is 3.08 us/boundary at M=1 and 4.39 at M=2 (>= 1.0 and far beyond noise)
= 0.111 / 0.158 ms/step over 36 boundaries (upper bound; a real fold pays K1's reduction inside A).

## S0 facts
- A at M=1: chain config (WARPS, CLUSTER_K, RT) = (4, 4, 2), L2PF off; M=2: (8, 2, 2), L2PF (pos 1, frac 154);
  M=4: (8, 4, 1), L2PF (pos 2, frac 102) (low_m_gemm.py _CHAIN_A_CONFIGS / _L2PF_DEFAULT). C++ instances with
  STAGES/PF: (4,4,2,1), (8,2,2,1), (8,4,1,1) (mxfp8LowMGemm.cu :1530+).
- K1 in the chain: P=480, W=15, W_PAD=16 -> _mhc_lagged_layer_input_fast_kernel at M=1/2/4 (any tactic P of
  96/224/256/480 gives W_PAD 4/8/8/16, so the fast K1 runs at both hc_attn and hc_ffn wherever K1 runs; hc_ffn
  K1 is folded into dsv4FusedRouterKernel at M<=2). hc_attn K1 is wait-first at M=1/2 (K1_TRIGGER_FIRST_MIN_TOKENS=4,
  mhc_lagged_split.py:482), trigger-first at M=4. MoE AR trigger EARLY at M=1/2/4 in the tree arm.
- hc_attn layer_input readers: A = low_m_mm in DeepseekV41Attention._project_inputs
  (csa2/module.py:437, call :766 / chain :455); aux _prepare_auxiliary (module.py:583, called :759 on the aux
  stream and :790) -> _prepare_global (compressor) and small_n_linear(index_weights_proj) (:600). So a fold must
  still materialize layer_input for the aux-stream readers (or give them their own copy).

## NEXT (resumed run, builds on this branch)
Fold: A' = csa2_mxfp8_low_m_gemm with a K1 prologue after griddepcontrol.wait (redundant per CTA, P-thread order
replayed for bitwise), writing a disjoint slice of layer_input for the aux readers, gated M<=2
(TRTLLM_CSA2_A_K1_FOLD). Budget: the fold must cost < 3.1 (M1) / 4.4 (M2) us/boundary vs ideal.
