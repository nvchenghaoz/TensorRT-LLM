# Idea 374 / branch 226: design D follow-up, MAX_ROWS 6144 -> 8192 via stream-role swap (design B)

Metric: kernel_time (one same-slot 4-GPU run_gpu_pair at served shapes, pair.py) + one serving pair as the
end-to-end guard (plan 3610 (6)). Class REORDERED (chunked LP, same as landing 77).

## DONE
- tensorrt_llm/_torch/modules/mhc/ar_pipeline.py: `TRTLLM_DSV41_PREFILL_AR_PIPELINE_ROLE` (default `swap`).
  swap = chunk ARs on the current (main) stream, chunk mHCs on a priority-0 side stream; all outputs/scratch are
  allocated on main before the loop and kept referenced until main waits on the side's last event. The side stream
  allocates nothing. MAX_ROWS default 8192 under swap (6144 under `comm` = landing 77's roles). Python only, no C++.
- Stage 0, job 2022 (smc521ge-0046 austin:4, GPUs 4-7, NUMA 1), us per boundary, max over ranks:
  - T=4096: serial 1406.5, comm 1127.0, swap_p0 1128.1, swap_pm1 1134.2
  - T=6144: serial 2122.5, comm 1641.1, swap_p0 1644.1, swap_pm1 1649.4
  - T=8192: serial 2827.0, comm 2157.1, swap_p0 2159.6, swap_pm1 2167.4
  - side-stream allocator segments: comm 7/12/16 (70/120/160 MiB), swap 0 at every T on every rank;
    reserved at T=8192: serial 2906 MiB = swap 2906 MiB, comm 3066 MiB.
  - host us per boundary at T=8192: swap ~600, comm 770-900, serial ~101.
  - swap vs comm bitwise on 120/120 calls per T (all 4 ranks); swap vs serial layer_input rel_rms <= 0.038.

- Kernel pair 2053 (base 06efaf15) / 2054 (head 65a035dd), smc521ge-0046 austin:0 (GPUs 0-3, NUMA 0), us per
  boundary max over ranks: T=2048 677.5 -> 676.5 (both unpipelined); T=4096 1127.0 -> 1122.0; T=6144 1638.9 ->
  1638.0; T=8192 2811.6 -> 2156.0 (-23.3%; noise 0.71% / 1.03%). li_rel_vs_ref piped 0.03576 = serial 0.03576.
- Check job 2085 (austin:4): chain_check PASSED (1000 calls x {4096, 4099, 8192}, IPC flags OK every 250 calls,
  0 rerun mismatches, mHC identity bitwise). e2e off,on with E2E_BATCH=4: 40/40 within tolerance, 0 errors,
  8192-token EXTEND steps pipelined (342 LP two-shot all on main, 592 chunk-mHC kernels on the side stream),
  decode 240 lamport / 0 LP per rank, tail graph 80/80 both arms. Its own JSON says passed=false only because its
  KV gate compared per-rank pre-sync GiB strings. The real rank-synced KV v2 device quota: env-0 2.1192 GiB,
  swap 2.2125 GiB (+4.4%; inside-torch profiling peak 2.21 -> 2.16 GiB, outside-torch 8.53 -> 8.41 GiB,
  the env-0 arm ran first with the 251 s JIT warm-up). The swap arm's capacity is not lower.

- Gate (plan 15f): design B only for T > SWAP_ABOVE_ROWS = 6144 (module constant); T <= 6144 runs landing 77's
  run() unchanged. Receipts job 2102 (head 790201cb, austin:0): 4096/6144 chunk ARs on the comm stream (pri -1),
  mHC on the caller stream, 48/48 bitwise vs main's run(); 8192 B (ARs on caller, mHC on side pri 0) 48/48 bitwise
  vs 77's run() forced to 8192 rows; env 0 => plan None for T in 1..16384; plan == main's for T <= 6144; chain
  1000 x {4096, 4099, 8192} PASSED.
- KV capacity (2085, 2090): NOT EQUAL, jitter. Rank-synced KV v2 quota 2085 off/on 2.1192/2.2125 GiB; 2090
  warm(env0)/off(env0)/on 2.4158/2.2793/2.2125 GiB. Two env-0 arms on the same commit differ by 0.137 GiB, more
  than off-vs-on. Inside-torch profiling peak is 2.21 GiB (env0) vs 2.16 GiB (swap) in every arm: the change does
  not add torch memory; the spread is outside-torch (NCCL/graphs/JIT) startup memory.

## METRIC DECLARATION (ruling 12550, before the pairs)
- Kernel pair (pair2.py): PRIMARY T=6145 (served c4 follower step = 3 x 2048 ctx + 1 gen; bounds 5 x 1152 + 385),
  secondary 8192/6144/4096, 2048 no-change. Claim per c4 follower step = 37 eligible boundaries x mean delta at 6145.
- In-job ABAB D (77's run() forced, MAX_ROWS 6400) vs B at 6145, 4+4 blocks. If D is faster beyond B's block spread,
  the final head moves the B gate to T > 6400 (D covers 4096..6400); otherwise B covers 6145.
- Then ONE same-slot run_serving_pair, base 06efaf15 vs final head; submit metric = c4 TTFT (serving_latency), with
  c1/c2/c4 TTFT/TPOT reported; pass = c4 TTFT, c4 TPOT, c2 TTFT not worse beyond the floor.

## UNTESTED / NEXT
- Kernel pair 2053 (base 06efaf15) / 2054 (head), pair.py T in {2048, 4096, 6144, 8192}.
- Untimed checks: chain_check.py (1000 calls x {4096, 4099, 8192}, IPC flags) + e2e_check.py off,on with E2E_BATCH=4
  (8192-token prefill steps; KV capacity lines compared between arms; decode lamport-only / 0 LP; tail graph 80/80).
- Serving pair (guard + acceptance 1: served startup with memory_pool_prepop and the first T=8192 prefill).
