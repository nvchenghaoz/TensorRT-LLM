# Idea 386 (branch 229): hc_attn mHC K1 folded into csa2 low-M GEMM A at decode M <= 2

Base: main ef96a2e4 (landings 1-83; TRTLLM_CSA2_A_K1_FOLD absent on main). Builds on branch 220 (chain.py, STATUS_352).

## Code
- `cpp/tensorrt_llm/kernels/minimaxM3/mxfp8LowMGemm.cu`: template flag `K1FOLD` on `csa2_mxfp8_low_m_gemm_kernel`
  (new last kernel parameter `K1foldArgs`; K1FOLD=false instances unchanged code). Instances: (4,4,2,1) and (8,2,2,1),
  each with and without the L2-prefetch PF flag. Each CTA (rank r of the K cluster) computes K1's layer_input for its
  own K slice after griddepcontrol.wait, bitwise `_mhc_lagged_layer_input_fast_kernel`:
  K1 warp w walks 32-chunk blocks 32w + iP; one hardware warp replays a K1 warp's lanes on a block; the running sum
  crosses ranks as a DSMEM carry (st.async + mbarrier, aliased into the receiver's xq_s) and ends in the warp
  butterfly, whose value is broadcast to every rank (16 floats x M of new smem). Then layer_input -> MXFP8 in
  registers (quad amax), tile-0 cluster writes layer_input. Plan (per rank/warp block list) built on the host
  (`csa2K1FoldPlan`), passed in the kernel parameters. Residual/coefficient rows are L2-prefetched at kernel start,
  before the CTA's weight bulk copies.
- New op `trtllm::csa2_mxfp8_low_m_gemm_k1fold` (thop/minimaxM3/minimaxM3Mxfp8LowMGemmOp.cpp), audit op
  `trtllm::csa2_low_m_gemm_instance_stats` (occupancy / regs / spills / smem).
- Python: `low_m_gemm.py` (A_K1_FOLD_ENV_ON = TRTLLM_CSA2_A_K1_FOLD, A_K1_FOLD_MAX_M = TRTLLM_CSA2_A_K1_FOLD_MAX_M
  default 2, `k1fold_plan_ok`, `k1fold_applies`, `k1fold_warmup`, `low_m_mm_k1fold`); `csa2/module.py`
  (`stash_k1_fold`, `_take_k1_fold`, `_project_inputs(k1_fold=)`, the aux stream runs `K1FoldPending.run_copy()`);
  `hyper_connection.py` (`defer_k1_max_tokens`, the hc_attn gate's own row limit); `modeling_deepseekv41.py`
  (`_attn_k1_fold_eligible`, stash before self_attn, done-check after).

## Results
- Job 2083 (commit 25ebc847, first kernel version): check.py PASS (bitwise rows/layer_input/xq/xsf for M=1/2,
  P in {96 (M=1 only), 224, 256, 480}, PF on/off, 72 calls each, NaN-poisoned; graphs 8 iters; skip-wait control
  caught 10/10, with-wait exact 10/10; occupancy equal: A M1 94 clusters / 2 CTAs per SM, M2 94 / 1; dyn smem
  +144 B; regs M1 80 -> 167, M2 90 -> 128, 0 spill bytes). chain.py check352 PASS (fold arm bitwise at M=1/2,
  tree runs the fold at M=1/2 and K1 at M=4). layer_check failed on a stand-in attribute (fixed), SASS dump empty.
- Pair 2088 (base) / 2089 (head), first kernel version: SLOWER. Head tree (fold) 29.94 vs tree_k1 25.18 us/boundary
  at M1 (+4.8), 33.76 vs 27.65 at M2 (+6.1); fold - ideal 7.9 / 10.4 us. Diagnosis: A streams 9.2 MB of weights
  from DRAM (~6 us) starting during the producer; the fold's post-wait residual loads then queue behind them; plus
  a per-thread device-side plan (integer divisions) on the prologue path. Second version (commit b5bebc56):
  host-built plan + L2 prefetch of the K1 inputs at kernel start + all loads issued before math. UNTESTED yet.

- Job 2097 (commit 8892f9c6, second kernel version): check.py PASS (now P=96 also at M=2; regs M1 156, M2 209, 0
  spill, occupancy unchanged 94 clusters); head chain us/boundary fold (tree) vs tree_k1: M1 24.86 vs 25.44
  (-0.58 [-0.63,-0.50]), M2 26.11 vs 28.40 (-2.29 [-2.44,-2.21]), M4 29.77 vs 29.77; fold - ideal 2.63 / 2.64.

- SUBMITTED EVIDENCE (code = 8892f9c6, evidence commit 61d0d919):
  - Pair 2106 (base ef96a2e4) / 2107 (head), austin:4, chain.py tree arm at W_crit p50, us/boundary:
    M1 25.185 -> 24.694 (-1.95%), M2 27.940 -> 25.723 (-7.9%), M4 29.478 -> 29.506 (unchanged). Same pair: SASS
    hashes of all 32 K1FOLD=false csa2_mxfp8_low_m_gemm_kernel instances identical base vs head (libth_common.so).
  - Correctness 2110: check.py, layer_check.py (M=1/2/4 FOLD/NOFOLD/SLEEP/NOPDL vs REF bitwise, 717 tensors),
    chain check352 (no JIT in capture) all PASS.

## NEXT (later work)
- fold - ideal is still 2.6 (M1) / 2.3 (M2) us/boundary: the 2 DSMEM hops (carry, warp values) and the post-wait
  x_prev/residual loads. M=3/4 (c4) keep K1; an M<=4 extension needs a (8,4,1)-config K1FOLD instance and a 4-slot
  plan, sized against this branch's fold - ideal.
