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

## UNTESTED / NEXT
- Kernel pair 2053 (base 06efaf15) / 2054 (head), pair.py T in {2048, 4096, 6144, 8192}.
- Untimed checks: chain_check.py (1000 calls x {4096, 4099, 8192}, IPC flags) + e2e_check.py off,on with E2E_BATCH=4
  (8192-token prefill steps; KV capacity lines compared between arms; decode lamport-only / 0 LP; tail graph 80/80).
- Serving pair (guard + acceptance 1: served startup with memory_pool_prepop and the first T=8192 prefill).
