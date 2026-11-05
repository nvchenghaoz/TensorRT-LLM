# Idea 196 Stage 0 — status

UPDATE (branch 131, idea 215, 11-04 evening): RUN. Final job 1138 (commit 058bbe52); results and
verdict (BUILD SKEW) in RESULTS.md. Harness changes since 77bb689a: spans by CUDA events after a
warm-up (Kineto spans carry ~27 us/layer of CUPTI overhead), per-node split with grid/block,
replay output checks, NVML by UUID, pre-warm + auto re-run on >3% block dispersion, row key `nD`,
expmax REAL calibrated to fact 822.

(Original branch 112 note follows.) Harness written, NOT run. The binding top-lead ruling (~12:58 PST, coach notes 5798/5858)
says a non-focus branch started after 12:35 does CPU-only work this session and queues no GPU job;
plan 1970 (3) defers idea 196 to next session. So no table and no verdict exist yet.

## Files
- `stage0_gpu.py`: 1 GPU, no source edit. 40-layer CUDA graph with the production fork/join
  (`maybe_execute_in_parallel` under `with_multi_stream(True)`). Routed = the production
  `Sm12xW4a8DecodeState.run` with forced-D routing; 5 weight sets, so every layer reads fresh
  experts. Shared = the real FP8 block-scale GatedMLP from `shared_row_split/common.py`
  (with `read_dims` overridden), x0.5, in 40 deepcopies.
  Arms A routed only / B +shared 1152 cols / C +shared 640 cols / D shared 1152 alone /
  E shared 640 alone. They are interleaved over NB=6 blocks of 10 Kineto-traced replays,
  with a 512 MB read-only flush before each replay.
  Per layer the output is span/40, plus fc1, fc2 and shared node sums. Shared nodes are names
  seen in D/E and never in A. Each arm reports block dispersion and replay noise; the SM clock
  comes from NVML by UUID.
- `stage0_expmax.py`: CPU only (numpy).
  - Generators: UNIFORM, and REAL (0.5 uniform + 0.5 Dirichlet(2)). REAL's M=4 histogram is
    checked against fact 822 (1.5/82.8/15.7, within 2 pp).
  - 8192 steps per M.
  - Outputs: MAIN/SHARD/SKEW E[max_r] x40 in ms/step vs window 910 ITL. The SE of
    (design - MAIN) includes a table bootstrap.
  - Ends with a verdict line.

## Commands for next session (each ~10-15 min on 1 GPU, est.)
1. `{python} experiments/ascent/shared_contention_196/stage0_gpu.py {checkpoint}`
   (BENCH_SHAPE=ep4, main's layout; engine.json on main has no moe_tp/moe_ep).
2. `BENCH_SHAPE=ep2tp2 {python} experiments/ascent/shared_contention_196/stage0_gpu.py {checkpoint}`
   for context (or as main's layout if branch 100 landed).
3. In the same job, after step 1/2:
   `{python} experiments/ascent/shared_contention_196/stage0_expmax.py $AUTO_DEV_ARTIFACTS/stage0_ep4.json $AUTO_DEV_ARTIFACTS/stage0_ep2tp2.json`
4. Commit the raw JSONs under experiments/ascent/shared_contention_196/results/.

## Untested risks to watch on first run
- `copy.deepcopy` of a post-load GatedMLP (FP8 block-scale Linear). If it fails, call
  `build_shared_mlp` 40 times instead (slower: CPU randn).
- Kineto must see the aux-stream graph nodes. Check that per-arm node counts
  (`n<count>` in the log) differ by the shared kernels.
- `"fc1"`/`"fc2"` name matches come from fi_l2_prefetch/bench_pair.py (known to work).
