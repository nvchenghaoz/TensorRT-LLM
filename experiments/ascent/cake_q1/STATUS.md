# Branch 145 (idea 245 Q1) STATUS, 05:2x PST 11-05 (end of run)

Measured commit: 1c78969e (unchanged; only this STATUS file was added after it).

## DONE
- Kernel pair (time) 1399/1400 and SASS pair 1401/1402: exact, gains c1 16.7%, c2 17.1%, c4 13.0%, c8 6.5% (kernel_time, cake chain).
- Repeated-execution / live-overlap / skip-wait check 1397: passed.
- 4-GPU audit 1441 (experiments/ascent/cake_q1/server_audit.py, slot austin:4, all 4 phases rc=0):
  - ROUTE PASSES, rank 0, c1 and c4, 40 steps: CakeMergeF32Kernel<true> 40/step, <false> 0; prep, decode_body,
    and K2' each 40/step; the kernel right after the merge is csa2_lowm_wo_a 1600/1600; K2' starts before the merge
    ends 1600/1600 (K2'.start - merge.end p10/p50/p90 = -6.24/-5.82/-5.28 us at c1, -4.90/-4.61/-4.32 us at c4);
    allreduce_fusion_kernel_oneshot follows the AR-feeding K3' 1600 times (the other K3' successors are the next
    csa2_mxfp8_low_m_gemm and mlaRoPEInplaceKernel, as expected for 3 K3'-named GEMMs/layer). Ranks 1-3 same.
  - BITWISE ITEM FAILS, AND THE CAUSE IS NOT Q1: tokens and logprobs differ between arms (attn_pdl0, pdl0, cake0 vs
    default). But the FIRST generated token's logprob, which comes from the 768-token PREFILL (M=768, so no
    decode switch runs), already differs in every arm: default -3.416, attn_pdl0 -3.143, pdl0 -3.496,
    cake0 -3.528 (same token id 4258). So each fresh LLM lifetime gives different numbers (autotuner tactic
    choice / non-deterministic prefill), and random-token prompts amplify this into token divergence after 2-3
    tokens. The audit had no default-vs-default control, so it cannot show bitwise equality for any switch.
  - Graph capture lines for 1/2/4/8: not extracted by the script (it greps only *.log under artifacts); look in
    job 1441 stderr.log (12 MB).

## UNTESTED
- Bitwise A1 vs A2/A3 under a deterministic setup.

## NEXT (exact next GPU job, 4 GPUs, one job)
Fix server_audit.py (experiments/ only, so the measured commit stays 1c78969e):
1. Add arm A1b = default again in a second lifetime as a control. Use real-text prompts, at least 64 decode tokens.
2. Make prefill deterministic across lifetimes: same autotuner cache / TLLM autotuner off in every arm.
   Better: compare in ONE LLM lifetime by toggling the switch between captures. That is impossible for
   graph-captured env reads, so use teacher forcing instead: feed default's tokens as prompts and compare
   decode-step logprobs.
3. GATE: A1 == A1b bitwise first. Only then A2/A3 == A1 is meaningful. If A1 != A1b, report the per-arm max
   |dlogprob| against the A1/A1b spread, and ask the lead to amend item 3(d) to "within the lifetime-to-lifetime
   spread" (the reference tolerance in experiments/ascent-benchmark/reference.json is calibrated the same way).
4. Grep stderr.log for the CUDA graph capture lines (batch 1/2/4/8, 0 eager fallbacks).
Then resubmit with acceptance item 3 linked to the new job, plus 1441 for the route checks (a)-(c).
