# Idea 94 (branch 94): LOWPRECISION FP8-payload two-shot AR for prefill-size messages: status

Base ab62f1d6 (landings 30-32 included; TRTLLM_B12X_WO_B_WIDE_ROWS, TRTLLM_MHC_SPLIT_LAGGED and
TRTLLM_GATE_FORWARD_LEGACY_BODY were left unset in every arm). Source commit for all evidence: 52d60259
(cpp/tensorrt_llm/thop/allreduceOp.cpp, tensorrt_llm/_torch/distributed/ops.py, engine.json, bench.py). Later commits
only touch experiments/.

## HANDOFF table (job 929, austin:4-7; full JSON at artifacts lp_ar_table.json)
Per call on 4 ranks, [M,5120] bf16, max-rank median of 3 x 100:
| M | t_LP us | t_PHB us | t_NCCL_default us | LP saving vs PHB |
|---|---|---|---|---|
| 2048 | 482.1 | 783.5 | 1085.3 | 301.3 us |
| 6144 | 1454.8 | 2322.4 | 3233.6 | 867.6 us |
Sequence dispersion <= 0.2% (LP, PHB at 2048); <= 1.05% for every arm at 6144.
Error vs the fp32 sum, 120 iterations, every one checked (e = max|out-ref| / sum_i|x_i|):
- LP realistic: e 0.120, rel_rms 0.0373, max_abs 0.63, mean_abs 0.0556.
- LP heavy (6 channels x50): e 0.120, rel_rms 0.0204, max_abs 18.4, mean_abs 0.0561.
- NCCL (PHB and default bitwise equal): e 0.0113, rel_rms 0.0027, max_abs 0.078 (heavy 2.75), mean_abs 0.0036.
- LP is about 14x NCCL in rel_rms. That is the E4M3 arithmetic floor: 3 mantissa bits give a half-ULP rms of about 3.6%.
- LP is deterministic and finite. The 1000-call robustness run (reset barrier every 250 calls, alternating heavy inputs) had
  0 nonfinite calls and max rel_rms 0.0373, with no hang.
Routing: under strategy LOWPRECISION, M = 1, 2, 4, 8, 16 (CUDA graph) and 128 rows are bitwise equal to AUTO on all ranks
and run the same kernels. 2048 rows run lowPrecisionPreprocessKernel + lowPrecisionTwoShotAllReduceKernel.
STOP RULE did not fire.

## Paired microbench (run_gpu_pair 949 base / 950 head, austin:0-3, served strategy read from each tree's engine.json, PHB)
- 2048: 785.5 -> 478.0 us/call
- 6144: 2327 -> 1434 us/call
- Decode M = 1/2/4: unchanged (same error values; the graph-replay timing is noise-dominated at ~13 us)

## Serving pair (run_serving_pair 947 base / 948 head, austin:4-7, back to back)
Each cell is mean / p50 / min:
| case | base TTFT (ms) | head TTFT (ms) | base TPOT (ms) | head TPOT (ms) |
|---|---|---|---|---|
| c1 | 161.3 / 153.2 / 151.7 | 155.7 / 151.3 / 149.4 | 6.834 | 6.828 |
| c2 | 322.6 / 311.5 / 308.0 | 312.9 / 308.3 / 305.0 | 7.886 | 7.891 |
| c4 | 576.8 / 573.5 / 570.7 | 531.7 / 528.8 / 521.8 | 9.361 | 9.554 |
Realized r at c1 = 5.6 ms (mean) / (40 x 0.301 ms) = 0.46; on p50 it is 0.16.
The gate refused this pair. Base proxy noise is 0.0383 > 0.03, from the c1 TTFT max of 227 ms in base arm 947. Check
server.log for a 'full tokenizations' line at that request (ruling 04:48: one rerun is allowed if it is named).
c4 TPOT +2.1% needs a look; decode ARs are unchanged.

## Branch 98 (idea 193, continuation), main bf2855d6
- Synced onto bf2855d6 (landing 35, host-only prefix_token_cache.py; no allreduce code touched).
- Job 1007 (4 GPU, austin:0-3, commit ca8d2919): trace.py + bench.py sweep in one job. PASSED.
  Rank 0 EXTEND step: 40 lowPrecisionTwoShot + 40 ncclDevKernel_AllReduce (128-row replay ARs), 0 lamport.
  DECODE steps (10): 0 lowPrecision, 0 NCCL, 400 oneshot_lamport in total over the decode scopes.
  Sweep: LP 2048 e 0.120 / rel_rms 0.0373 (heavy 0.0204), NCCL 0.0027; 120 iterations every one checked,
  deterministic; 1000-call run with reset barrier every 250: 0 nonfinite, IPC flags (host counter equal on all
  ranks and advanced by exactly the call count; barrier_in slots == last flag, none at next flag) OK before every
  reset; every 25th in-chain output re-run alone bitwise equal. Routing M=1..16 and 128 bitwise equal to AUTO.
- Serving pair 1 (austin:4-7, back to back): 1028 base bf2855d6 / 1029 candidate ca8d2919. REFUSED by the gate:
  base proxy noise 0.0908 (base c1 TTFT outliers: mean 222.9, p50 183.1, min 160.0, max 337.6 ms).
  | case | base TTFT mean/p50/min | head TTFT mean/p50/min | base TPOT | head TPOT |
  | c1 | 222.9 / 183.1 / 160.0 | 160.3 / 160.1 / 154.4 | 6.805 | 6.823 |
  | c2 | 331.4 / 332.1 / 321.9 | 323.0 / 321.6 / 316.3 | 7.947 | 7.921 |
  | c4 | 480.5 / 479.4 / 474.1 | 440.1 / 438.6 / 428.7 | 9.374 (rse 0.27%) | 9.402 (rse 0.35%) |
  c4 TPOT +0.30%, within noise: the +2.1% of 947/948 did not reproduce.
- Serving pair 2 (the one allowed re-run, austin:0-3, back to back): 1040 base bf2855d6 / 1041 candidate 926562e5.
  Base noise 0.0168, candidate 0.0088 (within the 3% gate).
  | case | base TTFT mean/p50/min | head TTFT mean/p50/min | base TPOT | head TPOT |
  | c1 | 156.3 / 155.7 / 149.7 | 150.4 / 148.8 / 146.6 | 6.778 | 6.789 (+0.16%) |
  | c2 | 311.1 / 309.5 / 305.5 | 305.7 / 305.3 / 297.3 | 7.885 | 7.905 (+0.26%) |
  | c4 | 490.4 / 474.6 / 468.7 | 443.5 / 434.8 / 430.1 | 9.489 (rse 0.45%) | 9.463 (rse 0.15%) (-0.27%) |
  c4 TPOT: +0.30% in pair 1, -0.27% in pair 2; both inside the per-arm noise. The +2.1% of 947/948 is not reproduced.

## Branch 119 (idea 225, landing attempt), main 2fa8e034
- sync_main: already on 2fa8e034 (landing 36, no AR file touched), so the sync was a no-op and no rebuild was needed.
- engine.json (merged): {"enable_tokenization_cache": true, "allreduce_strategy": "LOWPRECISION"}. That is main's only key plus LP.
- The earlier pair 1047/1048 was cancelled by a runner recovery before it measured anything.
- Serving pair 1059 base 2fa8e034 / 1060 candidate. Both arms ran back to back on austin:4-7; the tool picks the half.
  Both valid. Gate noise (max per-case rse): base 0.0094, cand 0.0228.
  | case | metric | base mean/p50/p90/p99/std (n) | cand mean/p50/p90/p99/std (n) | delta | rse b/c |
  | c1 | TTFT | 157.95/156.07/166.98/169.17/5.14 (12) | 158.50/154.23/165.20/190.43/12.49 (12) | +0.35% | 0.94%/2.28% |
  | c1 | TPOT | 6.600/6.577/6.648/6.717/0.048 | 6.620/6.585/6.708/6.731/0.063 | +0.31% | 0.21%/0.27% |
  | c2 | TTFT | 321.60/320.37/328.63/330.53/5.15 (16) | 321.11/319.47/345.06/350.87/16.61 (16) | -0.15% | 0.40%/1.29% |
  | c2 | TPOT | 7.741/7.744/7.802/7.844/0.052 | 7.791/7.752/7.889/7.974/0.079 | +0.64% | 0.17%/0.25% |
  | c4 | TTFT | 477.26/474.30/494.33/495.93/8.76 (24) | 440.19/439.40/448.53/449.59/5.58 (24) | -7.77% | 0.37%/0.26% |
  | c4 | TPOT | 9.414/9.328/9.710/9.717/0.205 | 9.320/9.313/9.406/9.411/0.054 | -1.00% | 0.44%/0.12% |
- Guard verdict. Read per metric, c2 TPOT +0.64% is beyond its own rse (0.25%, about 2.1 sigma combined).
  c1 TPOT +0.31% is marginal against its 0.27% rse. Read with the gate's per-arm noise floor (2.28%), every guard
  passes. Per ruling 6121(a) the worker did NOT submit, and the area lead decides.
  Mechanism: decode routing is unchanged. ops.py sends sub-2 MiB messages to AUTO, and trace 1007 shows the same
  400 lamport calls. c4 TPOT moving -1.0% (4x its rse) with identical decode code shows that within-arm rse
  understates the run-to-run TPOT spread.
- Acceptance items 1 and 3 cite job 1007, which belongs to branch 98. check_submission says it is not citable from
  branch 119. A submission needs a fresh trace.py and bench.py robust run on this branch, or an evaluator waiver.

## Pending / next GPU actions (older, branch 94)
1. Correctness check as kind 'worker' on committed source:
   run_gpu_job kind=worker gpus=4 ["{python}", "experiments/ascent/lp_ar/bench.py", "sweep"] (timeout 900).
   check_submission asked for a worker_check receipt.
2. In-server c1 trace (acceptance 3), FIRST GPU job next session (top lead ruling 05:38/05:40):
   run_gpu_job kind=worker gpus=4 timeout 1500 ["{python}", "experiments/ascent/lp_ar_trace/trace.py", "{checkpoint}"].
   Jobs 951 and 966 were both cancelled at session end before they ran (966 was cancelled while queued, per the
   no-4-GPU-jobs-in-the-last-20-min ruling). Commit 1ce87f14 fixed trace.py per the evaluator note: kernels are now
   assigned to executor steps through the step[EXTEND/DECODE] record_function scopes (kernel correlation -> launch ->
   CPU scope, with gpu_user_annotation as fallback), profiling CPU+GPU. It no longer splits at the first lamport, since
   decode M=1-16 runs NCCL RING (job 929). The output gives per-step counts for every step; the pass condition is
   rank-0 EXTEND step lp_calls == 40 and ncclDevKernel_AllReduce == 40, plus 0 lowPrecision kernels in DECODE steps.
   Then ONE same-half run_serving_pair (c4 TPOT +2.1% in 947/948 to reproduce or clear). Expect 40 lowPrecisionTwoShot + 40 ncclDevKernel_AllReduce on
   rank 0 in the prefill step, and lamport-only decode.
3. Serving: if the 227 ms outlier is a named tokenization miss, run ONE more run_serving_pair on the same base.
   Otherwise re-pair.
4. Submit measure=serving_latency, numerics=mode_switch, labels 1/2/4. Landing accuracy (check.py / accuracy.py) decides.

## Branch 139 (idea 242) state, 03:5x PST 11-05
- Code = branch 119 (allreduceOp.cpp whole-message LP threshold, ops.py <2 MiB -> AUTO, engine.json
  {"enable_tokenization_cache": true, "allreduce_strategy": "LOWPRECISION"}) + experiments/ascent/lp_ar_trace/trace_graph.py.
- sync_main #1 -> 93bc94c6 (landings 37-39: dsv3RouterGemm.cu; csa2/module.py, low_m_gemm.py, mxfp8LowMGemm.cu,
  minimaxM3Mxfp8LowMGemmOp.cpp; csa2LowMWo.cu/Op.cpp, b12x_wo.py). No waiver file. Pair 1314/1315 queued, cancelled (stale base).
- sync_main #2 -> 8898cb6b (landings 40-44; 44 = _dsv41_tail_moe_graph.py + modeling_deepseekv41.py -> waiver 2(b) fires).
- Trace job 1360 (austin:0, commit 5bbc31d5 on 8898cb6b) PASS: rank 0 EXTEND lp_calls 40 + NCCL AR 40; DECODE lowPrecision 0,
  lamport 400; tail graph captured layers 20-39 on all 4 ranks (80 lines), 0 'capture failed'.
- sync_main #3 -> d87d08d9 (landings 45-48; 47 = indexer, 48 = router Stage 2 in modeling_deepseekv4.py fused_forward
  ~1516-1687/~2226-2277, dsv4FusedRouter*, sm12x decode MoE: no AR file / AR call site (AllReduce at :1926/:2518 untouched)).
  No re-trace needed. Head e5acf75a.
- Serving pair 1427 (base d87d08d9) / 1428 (cand e5acf75a), austin:0, back to back. RESULT (mean ms, rse):
  c1 TTFT 142.63 (1.83%) -> 134.31 (1.01%) -5.8% | c1 TPOT 6.0132 (0.14%) -> 6.0126 (0.13%) -0.01%
  c2 TTFT 297.96 (0.97%) -> 285.91 (0.69%) -4.0% | c2 TPOT 7.1040 (0.18%) -> 7.0692 (0.15%) -0.49%
  c4 TTFT 447.61 (0.37%) -> 410.40 (0.41%) -8.3% | c4 TPOT 8.4269 (0.53%) -> 8.5868 (0.54%) +1.90%
  Gate floor (max per-case rse) base 1.83% / cand 1.01% -> 1.83%. c4 TPOT +1.90% is beyond it by 0.07 pt: guard FAILS
  under the brief's binding rule -> NOT submitted; branch closed for the area lead to decide.
  Also: check_submission marks trace 1360 stale (measured pre-sync commit 5bbc31d5); a submit would need a fresh trace/check
  on the head.
- Next: when 1427/1428 report, judge c4 TTFT (target) and guards c1/c2 TTFT+TPOT, c4 TPOT against the gate floor; submit
  serving_latency mode_switch, target label 4.

## Branch 154 (idea 284, landing re-attempt of 139), 04:5x PST 11-05
- Head 1d73a827 (= branch 139's code, on main d87d08d9). No sync_main: top lead ruled landings 49 (Engram decode
  all-gather) and 50 (sampler argmax, main f0043b93) touch no AR call site or file this branch edits; landing merges
  three-way. Step (1): 'main unmoved' for this branch's base (d87d08d9); no merge done.
- Step (2) trace job 1442 (4 GPU, austin:4-7, commit 1d73a827, rc 0, 254 s): PASS.
  passed=true, routing_ok=true, tail_graph_engaged=true, captured_pairs 80 (layers 20-39 on ranks 0-3), capture_failed [].
  rank0_prefill {lp_calls 40, nccl_allreduce 40, lamport_before_decode 0}; rank0_decode {lowPrecision 0,
  ncclDevKernel_AllReduce 0, oneshot_lamport 400}; profiled request prompt 34816 / cached 32768.
  In-trace lp_twoshot_us mostly 473-478 us per [2048,5120] call (outliers 530-1104 us, first call 1045 us).
  Trace durations, not timing evidence.
- Step (3b) serving pair 1447/1448 was queued as an evaluator fallback and CANCELLED by the top lead (waiver stands; it
  could not finish before 05:43). Do not re-queue this session.
- Step (4) check_submission REFUSED citing pair 1427/1428 (verbatim):
    problems: ["GPU job 1427 belongs to another branch"]
    acceptance item 2 gaps: ["job 1427 is not one of this branch's GPU jobs", "job 1428 is not one of this branch's GPU jobs"]
  So the waiver cannot be applied mechanically: the submission tool only accepts this branch's own jobs.
- NEXT (first thing next run): ONE run_serving_pair on this branch (base = its base, one 4-GPU half, no NCCL_* /
  TRTLLM_ENABLE_PDL env), then submit serving_latency mode_switch, target label 4 (c4 TTFT), guards per plan 2588(b)
  (c1/c2 TTFT: gate floor; c1/c2/c4 TPOT: max(gate floor, 2.9%)), checks [1442] (or a re-trace if a sync touches
  ops.py / allreduceOp.cpp / an AR call site / engine.json's AR key), acceptance item 1 -> 1442, item 2 -> the new pair.
