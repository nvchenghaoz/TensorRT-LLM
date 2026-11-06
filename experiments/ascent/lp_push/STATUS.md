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
  gpus=4, queued 08:50 PST. Baseline arm ran (smc521ge-0043 austin:0-3); candidate arm cancelled at session end.
- KERNEL PAIR 2005 (base 9ac8398e) / 2006 (head 71b4c280), session 12, NEW LEASE node smc521ge-0046 half austin:0-3
  (NUMA 0), same slot, consecutive. us/call, max over 4 ranks (median of 3 x 100), base pull -> push:
  | M x 5120 | pull | push | delta | noise base/cand | rank dispersion cand |
  | 1024 | 246.6 | 214.6 | -13.0% | 1.8% / 1.9% | 0.01% |
  | 2048 | 478.1 | 433.8 | -9.3% | 1.5% / 4.7% | 0.01% |
  | 4096 | 942.8 | 903.7 | -4.1% | 1.3% / 5.9% | 0.23% |
  | 6144 | 1437.4 | 1362.5 | -5.2% | 1.5% / 5.2% | 0.00% |
  16 CTAs/peer. Workspace memory delta per GPU: 0 B (allocated_per_gpu 75,497,472 in both arms; barrier 4128 -> 6688 B).
  Candidate arm's noise is higher (one GPU sampled at SM 1530 MHz, host load avg rose to 65 from the neighbour
  window job), so 4096/6144 gains are at the noise edge; 1024/2048 clearly beat it. Dev job 1983 (0043) put push at
  -8.6..-10.8% at all four shapes with 1.2-2.8% noise.
- Served sizing from the kernel pair alone (idea 365 Q4(a) not consulted): c-case T=2048 boundaries ~44 us x 40 =
  ~1.8 ms/req kernel time; served gain depends on barrier wait (fact 3520).

- EVALUATOR SEND-BACK (session 12) on pair 2005/2006: candidate arm's noise 4.7-5.9% > 3% and one GPU sampled at
  1530 MHz (neighbour window job 2003 loading on austin:4-7, load avg 57-65 vs ~4 in the base arm), and the
  [lib] grep receipt was uninformative (pitfall 3620). Harness changes (experiments only, no kernel change):
  - run.sh: grep receipt removed; preamble prints hostname, `nvidia-smi topo -p2p r`, per-GPU uuid/clock/memory.used
    and the compute-process list to stdout.
  - dev.py pair: (1) default-path timing as before + each rank's NVML SM clock read while its timed sequence is
    still queued; (2) interleaved in-process A/B TRTLLM_LP_AR_PUSH=0 vs unset at 1024/2048/4096/6144, 3 rounds with
    alternating order (pull-vs-pull control on base); (3) candidate tree only: dev's bitwise (10 shapes) and
    1000-call sequence checks; (4) CUPTI kernel receipt at M=2048 (env unset and =0) on every rank, plus an
    in-process /proc/self/maps scan of every mapped tensorrt_llm .so for the LP kernel / getenv strings.
    The JSON carries tree, kernel_receipt (default_path must be pull on base, push on head), ab_in_process,
    correctness, load_average start/end, per-rank GPU uuid.

- PAIR 2025 (base 9ac8398e) / 2026 (head de1a863d), smc521ge-0046 austin:0-3 (NUMA 0), same slot, consecutive,
  other half running window job 2004 in BOTH arms, loadavg 8-10 (base) / 6-7 (head), clocks under load >= 2220 MHz.
  Receipts OK: base = preprocess + lowPrecisionTwoShotAllReduceKernel (unset and =0); head = push kernel alone
  (unset), pull pair (=0); strings found in libth_common.so (TRTLLM_LP_AR_TWOSHOT_CTAS in both arms).
  | M | pull | push | delta | noise base/head (max p90-p10, seq disp) | head seq disp |
  | 1024 | 246.6 | 214.6 | -13.0% | 1.8% / 1.9% | 0.05% |
  | 2048 | 478.0 | 433.8 | -9.3% | 1.5% / 5.0% | 0.4% |
  | 4096 | 943.1 | 900.7 | -4.5% | 1.4% / 6.4% | 0.8% |
  | 6144 | 1436.4 | 1358.4 | -5.4% | 1.4% / 5.8% | 0.6% |
  Head in-process A/B (=0 vs unset, 3 rounds): -13.0 / -9.3 / -4.8 / -5.1%; base control 0.0 / 0.0 / +0.1 / +0.1%.
  Bitwise 10 shapes and 1000-call sequence pass on 0046 (head arm). The head's p90-p10 at >= 2048 is the push
  kernel's own per-call spread (same in the in-process A/B with pull at 1.1-1.3% in the same process), not the
  neighbour: p10 at 4096 = 864 us ~ 2 x 432, median 900.
- Rulings 13a/13c (3622/3628): claim shapes 1024/2048/8192, 4096/6144 secondary; ABAB 4+4 in both arms with
  per-block neighbour log; receipt with PUSH=1; strings over every .so in tensorrt_llm/libs + mapped. dev.py pair
  updated accordingly (adds 8192, even/odd-call medians, in-job repeat on a trigger).

- PAIR 2032/2033 (head 7d00d213, austin:0-3): INVALID under 13c(3). The neighbour was busy in 50/50 base blocks
  and 0/54 head blocks. Its checks pass. Head ABAB cross-check: 1024 -13.1%, 2048 -9.1%, 8192 -8.1%,
  4096 -4.3%, 6144 -4.9%.
- PAIR 2047 (base 9ac8398e) / 2048 (head 33f14a5c), smc521ge-0046 austin:4-7 (NUMA 1), same slot,
  consecutive. Receipts (CUPTI unset/=1/=0), strings in libth_common.so, bitwise 0/10 shapes and the
  1000-call sequence all PASS. SM clock under load >= 2295 MHz. Neighbour: merge job 2045 on austin:0-3 was
  busy in 50/50 base-arm blocks and 0/53 head-arm blocks.
  Head in-job ABAB, median of 4 block medians (us), PUSH=0 -> PUSH=1:
  | M | PUSH=0 | PUSH=1 | gain | spread0/spread1 | mean gain | cross-arm base -> head |
  | 1024 | 247.5 | 214.6 | -13.3% | 0.40% / 0.09% | -13.5% | 248.6 -> 214.6 |
  | 2048 | 481.95 | 424.45 | -11.9% | 0.04% / 0.49% | -11.7% | 482.2 -> 423.5 (-12.2%) |
  | 4096 | 953.2 | 875.7 | -8.1% | 0.12% / 0.25% | -8.4% | 954.1 -> 874.1 |
  | 6144 | 1451.85 | 1279.05 | -11.9% | 0.10% / 0.48% | -11.6% | 1454.6 -> 1275.5 |
  | 8192 | 1967.9 | 1715.9 | -12.8% | 0.06% / 0.30% | -12.9% | 1969.8 -> 1717.2 (-12.8%) |
  Base-arm control (env ignored): ABAB 0.0% at every shape. Workspace delta per GPU: 0 B (75,497,472 both).
  13g(c) base pull vs head PUSH=0: 2048/4096/6144/8192 hold (0.03-0.19% vs 0.25-0.49%); 1024 FAILS by
  0.04 pp (0.44% vs 0.40%). Reported to the area lead. Third pair 2058/2059 was cancelled by a lead while
  queued. I am waiting for consent for a third pair, or a ruling.
  ms/req at c1 = 40 x 56.5 us (mean delta, 2048) = 2.26 ms/req on austin:4-7.

## DONE / UNTESTED / NEXT
- DONE: kernel, switch, flag sizing, dev/pair script, dev job 1983 (all checks pass).
- DONE: kernel pair 2005/2006 -> submitted kernel_time EXACT.
- (old) NEXT: kernel pair 1995/1996 -> submit kernel_time EXACT. If it cannot run this session, re-queue exactly
  run_gpu_pair(baseline=candidate=["bash","experiments/ascent/lp_push/run.sh","{python}","pair"], gpus=4).
- Measured push is ~432 us at 2048, not the ~380 predicted: each phase is ~2x 8.1 MB per rank, so effective
  ~37.5 GB/s. Possible follow-up (not in this branch): overlap the scatter and reduce phases per warp-group
  chunk flags instead of one flag per CTA.
