# Branch 128 / idea 233 status (SM120 decode WO projection on the low-M MXFP8 kernels)

## Done (submitted commit fa3db136)
- K2' `csa2_lowm_wo_a_kernel`: cpp/tensorrt_llm/kernels/minimaxM3/csa2LowMWo.cu:185 (inverse RoPE + 1x128 UE8M0
  quant + grouped wo_a, weights staged before griddepcontrol.wait, trigger at start); op
  cpp/tensorrt_llm/thop/minimaxM3/csa2LowMWoOp.cpp.
- K3' = landed `csa2_mxfp8_low_m_gemm` on wo_b [5120,2048].
- Route: tensorrt_llm/_torch/attention/backends/sparse/csa2/b12x_wo.py:396 `LowMWOProjection`, built at :528,
  kill switch `TRTLLM_CSA2_LOWM_WO=0` (falls back to main's b12x chain). Non-gated rows (prefill, M>8) keep main.

## Measured
- Pair 1232/1233 (kernel_time, 40-layer cold-L2 graph via _project_output, austin:7, lease a8bcc448).
- Check 1236 (check.py sections A-E: bytes, numerics, live-overlap PDL + skip-wait control, W-AGG, decode window,
  repeated execution).
- Acceptance 4 (1-GPU form): experiments/ascent/lowm_wo/kineto_audit.py (Kineto route audit + test_module on/off).

## Known behaviour / optional follow-ups (not on this commit, to keep the measured commit unchanged)
- (b) Flags: with `TRTLLM_CSA2_LOWM_WO` on (default), `TRTLLM_B12X_WO_PROJECTION=0` does NOT turn off the SM120
  decode WO fast path (b12x_wo.py:528 runs before the b12x flag check at :539). To run main's generic
  project_sparse_attn_output path, set BOTH `TRTLLM_CSA2_LOWM_WO=0` and `TRTLLM_B12X_WO_PROJECTION=0`.
  `TRTLLM_CSA2_LOWM_WO=0` alone restores main's b12x chain.
- (a) `LowMWOProjection.accepts` (b12x_wo.py:439) does not inspect `positions`; csa2_lowm_wo_a's TORCH_CHECK requires
  a contiguous int32/int64 tensor with M elements. The model always passes that (int32 [M] position ids), but a
  follow-up should make accepts() return False otherwise (needs module.py:651 to pass positions to accepts()).

## Remains (new idea, build_on 128)
- K2' pre-wait wo_a staging deepening, PDL trigger-at-start into K3' tuning, and idea 245's trigger-first window
  (attention AR -> K2').
