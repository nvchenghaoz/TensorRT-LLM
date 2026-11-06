# Branch 215 / idea 360 / job 354: PUSH (posted-write) FP8 two-shot for the LOWPRECISION prefill AR

## (a) Base and STEP 0
- SYNC 08:3x PST: top lead event 11824 (landing 77 = branch 199 merged first, main 9ac8398e) asked for sync_main
  before the final pair. Queued dev job 1954 (old base, never started) was cancelled; sync_main -> base
  `9ac8398edbf2e452e8715e8edc45ed6373fac1f9`, clean three-way merge; landing 77 touches no file of this branch
  (diff_branch after sync: only this branch's .cu / allreduce_helper.py / experiments files). Timing table adds
  [1024,5120] (77's served chunk shape at T=4096/6144).
- Original branch base: `a9ae1a75e113fcc97ae4c76b89de15664c5469b9` (build_on 201).
- `git diff <base> --stat` before the first edit (diff_branch): only the 3 probe files of branch 201:
  `experiments/ascent/pcie_latency_probe.py` (+958), `experiments/ascent/pcie_probe_driver.sh` (+84),
  `experiments/ascent/pcie_probe_summary.py` (+82).
- Anchors re-grepped on the base: `.h` LP_TWOSHOT_MAX_CTAS 128 / LP_TWOSHOT_DEFAULT_CTAS 16;
  `allreduce_helper.py` LP_TWOSHOT_MAX_CTAS 128 + lowprecision_barrier_bytes; `ops.py:61` 2 MiB threshold;
  dispatch `lowPrecisionAllReduceDispatchRanksPerNode`; preprocess, stage 1/2 and per-block barrier as in the brief.
- NARROWING (11165/11176) applies from the start: 16 CTAs/peer only, dev job + kernel pair only, kernel_time EXACT,
  no serving guard.

## (b) Change (CPU side, committed before any GPU job)
- `customLowPrecisionAllReduceKernels.cu`: new `lowPrecisionTwoShotPushAllReduceKernel` (ranks <= 4), grid
  RANKS x LP_PUSH_CTAS_PER_PEER (16) = 64 CTAs x 512 threads, no preprocess kernel:
  1. entry `multi_gpu_barrier` (unchanged flags [0, RANKS)).
  2. scatter: CTA group g writes chunk (rank + g) % RANKS: quantized with the preprocess kernel's expressions and
     per-warp groups (496 inputs + float scale per 512-byte group), posted-written to that rank's ping buffer at
     slot = my rank (pull keeps it in its own buffer at slot = destination). One release flag per CTA.
  3. reduce: CTA b waits for the 4 scatter flags of lb = b % 16, sums its chunk from LOCAL slots (r + ii) % RANKS
     (the pull order), requantizes with stage 1's expressions and posted-writes the result to every rank's OTHER
     ping/pong buffer at slot = my rank. Release flag per CTA to every rank, wait for all 4.
  4. gather: `lowPrecisionTwoShotSecondStageKernel` unchanged, reading only the LOCAL other buffer.
  The other buffer is free during the call (entry barrier => every rank finished the previous call; nobody passes
  the next call's entry barrier before this rank). No extra workspace.
- `TRTLLM_LP_AR_PUSH` (read per call, same on all ranks): unset/nonzero = push for messages >= 2 MiB (the LP route
  threshold), `0` = main's pull path, bit for bit.
- `LowPrecisionAllReduceParams::deserialize` also fills `peer_comm_buffer_ptrs[tp + i]` with the other parity.
- `allreduce_helper.py` `lowprecision_barrier_bytes`: + tp*16 scatter + tp*tp*16 gather uint64 flags after the pull
  kernel's tp*(1+128); tp 4: 4128 -> 6688 B, still inside one 2 MiB IpcMemory page.

## (c) Invariants (EXACT class)
- Quantization, scales and sum order unchanged: same expressions (division by scale, scale==0 adds raw), same
  per-warp groups, same rank order (r + ii) % RANKS in stage 1, same stage 2 code. Max/abs over a group is order-free.
- Lane 31's unused 12 bytes are zero instead of uninitialized; never read into outputs.

## (d) Sizing (kernel pair only; overwrite with the pair's numbers)
- Predicted per call at [2048,5120]: 2 x 8.1 MB per rank / 43.6 GB/s ~ 372 us + compute vs pull 478.9 us (pair
  1787/1788). Served sizing must use idea 365's Q4(a); if 365 has not finished, size from the kernel pair alone.
- Workspace memory delta per GPU: 0 B (barrier buffer still one 2 MiB page; no new buffers).

## (e) Jobs
- dev job 1983 (head a44f6195, base 9ac8398e): `bash experiments/ascent/lp_push/run.sh {python} dev`, 4 GPUs,
  smc521ge-0043 half austin:4-7 (NUMA 1). Queued 08:28 PST, STARTED 08:42 PST, done 08:49. PASSED:
  - bitwise push vs pull (TRTLLM_LP_AR_PUSH=0), 3 inputs each (one heavy-tailed), all 4 ranks, every element:
    0 mismatches at M = 205/257/1024/1536/2047/2048/3001/4096/6144/8192 (8192 = two workspace parts).
  - 1000 back-to-back push calls, shapes 2048/257/6144 cycling, 4 inputs per shape, reset barrier every 250:
    0 mismatched / 0 non-finite calls; both ping/pong parities at every shape; IPC flags ok before every reset.
  - dispatch: oneshot_lamport at M=1/2/4/8, NCCL at 128 and 204 (< 2 MiB), push kernel only at 205 and 2048 with
    push=1 (no preprocess kernel), preprocess + pull two-shot with push=0.
  - same-process timing (per call, max over ranks, median of 3 x 100): pull vs push us
    1024: 247.4 -> 220.7 (-10.8%); 2048: 478.9 -> 431.6 (-9.9%); 4096: 949.0 -> 867.9 (-8.6%);
    6144: 1448.7 -> 1293.9 (-10.7%); noise 1.2-2.8%.
  - LP IPC allocation per GPU 75,497,472 B (workspace 34,670,592 x 2 + barrier 6,688 B x 2, 2 MiB pages): the
    barrier grows 4128 -> 6688 B inside the same 2 MiB page, so the per-GPU memory delta is 0.
- kernel pair 1995 (base 9ac8398e) / 1996 (head a44f6195): `bash experiments/ascent/lp_push/run.sh {python} pair`,
  gpus=4, queued 08:50 PST.

## DONE / UNTESTED / NEXT
- DONE: kernel, switch, flag sizing, dev/pair script, dev job 1983 (all checks pass).
- NEXT: kernel pair 1995/1996 -> submit kernel_time EXACT. If it cannot run this session, re-queue exactly
  run_gpu_pair(baseline=candidate=["bash","experiments/ascent/lp_push/run.sh","{python}","pair"], gpus=4).
- Measured push is ~432 us at 2048, not the ~380 predicted: each phase is ~2x 8.1 MB per rank, so effective
  ~37.5 GB/s. Possible follow-up (not in this branch): overlap the scatter and reduce phases per warp-group
  chunk flags instead of one flag per CTA.
