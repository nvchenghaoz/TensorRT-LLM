# Branch 61 (idea 122): submission notes

The kernel change is frozen at the evidence commit. This file only records how to read the evidence.

## Evidence (unchanged, accepted in plan 1434)
- Item 1, the measure: chain pair 642/643 (austin:4). Chain us/call at M=1/2/4/8/16:
  - base: 4.556/6.092/6.604/9.164/15.308
  - candidate: 3.532/4.044/5.324/8.140/14.796
- Item 2, correctness: check 644. Bitwise equal to k1_pdl0 on every iteration at M=1..16. 1000 captured replays at M=1 and M=8 are clean.
- Capture: job 675. 80 lamport ARs per decode step, 0 NCCL, block 160 x grid 4M, PDL=1.
- The diff still contains the env-gated TLLM_AR_LOG_LAUNCH printf. It is host-side and off by default (evaluator: may stay).

## Serving guard (base -> candidate; TTFT mean/p50 ms, ITL mean ms)
| pair | case | TTFT mean | TTFT p50 | ITL |
|---|---|---|---|---|
| 636/637 | c1 | 169.1 -> 178.5 | 165.8 -> 170.8 | 7.38 -> 7.32 |
| 636/637 | c2 | 342.3 -> 352.1 | 340.0 -> 351.6 | 8.44 -> 8.31 |
| 636/637 | c4 | 621.3 -> 634.4 | 615.1 -> 631.7 | 9.92 -> 9.83 |
| 672/673 | c1 | 179.0 -> 175.1 | 170.5 -> 166.3 | 7.37 -> 7.31 |
| 672/673 | c2 | 350.0 -> 351.1 | 348.8 -> 349.6 | 8.44 -> 8.28 |
| 672/673 | c4 | 622.7 -> 620.2 | 615.4 -> 616.3 | 9.92 -> 9.79 |

- TPOT/ITL is flat or better in every case of both pairs.
- In pair 636/637 the candidate's TTFT is +2..6%. Pair 672/673 does not reproduce it.
- The c1 max values of 637 (246.9 ms) and 672 (253.7 ms) are single first-request outliers, the cold-frontend pattern of pitfall 1205.

## Neighbor-load statements (top lead ruling, 21:15)
- **636 (base), 1793766889-1793767423: neighbor: none.** austin:4-7 held only microbench-class jobs: 643 (this branch's synthetic chain-AR arm), 645, 648, 649 and 650.
- **637 (cand), 1793767423-1793768281: neighbor: none.** Only microbench-class jobs ran there: 645, 648, 649 and 650, then 654 and 667.
- **672 (base), timed ~1793769220-357: neighbor possible.**
  - Earlier, austin:4-7 held microbench jobs 654, 667 and 670 and the 1-GPU worker 652, which cannot load the 4-GPU model.
  - By the arm's end (1793769367), window full_model 678 was on austin:4.
  - 678's start and phase cannot be read from this branch, so it may have been in model load during 672's timed windows.
- **673 (cand), timed c1 ~1793770021-048, c2 ~073-095, c4 ~120-142: neighbor yes (partly).**
  - At the arm's server start (1793769707), 678 was serving: GPUs 4-7 at 92 GB, 77-100% util.
  - By the arm's end (1793770152), window full_model 679 had replaced 678.
  - 679's model-load phase may overlap 673's timed windows.
- Pair 672/673 TTFT is flat, so no verdict hinges on an overlapped TTFT (rule 2 not triggered).

## Open gaps
1. Item 3's clause "check.py passes incl. repeated_execution_passed" is not met by any job on this branch.
   - Worker jobs cannot reach the frozen adapter (pitfall 812; top lead ruling 1836 / plan 845).
   - run_serving_pair runs the proxy adapter only.
   - The area lead asked the top lead (22:10, plan 1434) either for a runner-side check on this head or for a ruling that the landing check covers the clause.
2. Untested but unreached: TWOSHOT and the fused RMS_NORM patterns of allreduce_fusion_kernel_launcher.
   - AUTO on a PCIe-only group returns ONESHOT at <=512 KiB (allreduceOp.cpp L1466-1505).
   - modeling_deepseekv4.py L2134/L2139 force PRE/POST_MOE_FUSION False.
