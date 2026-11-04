# Idea 156 / branch 105: L20-39 128-row tail MoE-window CUDA graph. STATUS (13:52 PST 11-04)

Staged: Stages 1-2 are done. VERIFY and the serving pair are the next-session follow-up (build_on branch 105).

## Built (commit 1ef5ea79, plus scripts e5ffd3d8 and this note)
- `tensorrt_llm/_torch/models/_dsv41_tail_moe_graph.py` (new):
  - `ENABLED = TRTLLM_DSV4_TAIL_MOE_GRAPH == "1"` and `VERIFY = int(TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY)`, both read once at import.
  - `eligible()`: it requires all of the following.
    - Not `is_cuda_graph`, not capturing, `do_multi_stream()` False.
    - `csa2_replay_query_rows` set (bounded replay engaged with compacted rows), `num_generations == 0`.
    - Exactly 128 rows, no spec_metadata, no image_mask, no ADP, non-hashed gate.
    - Neither TRT-LLM's nor FlashInfer's AutoTuner is in tuning mode.
    - Under TP4 without ADP these inputs are the same on every rank.
  - `moe_window()` is the same code as main's inline window: `_mhc_boundary(hc_ffn, 1, ...)`, then `forward_MoE`, then `post_mapping`, or the deferred 5-tuple.
  - `run()` steps per layer:
    - The first eligible call runs eager. This warms the FlashInfer MXFP8 memo and the CUTLASS tactic lookups at M=128.
    - The second call captures, with `CUDAGraph.capture_begin(pool=<one shared pool>, capture_error_mode="thread_local")` on a side stream.
    - Later calls copy the inputs into static buffers (`empty_strided`, same strides), then call `replay()`.
    - A signature change (shape/stride/dtype/deferred) falls back to eager.
    - If a capture fails, the feature is disabled on that rank and the step runs eager.
- `modeling_deepseekv41.py`: one import, the module bool `_TAIL_MOE_GRAPH`, and one `if _TAIL_MOE_GRAPH and eligible(...)` branch after `attn_join`.
  - When the switch is unset or 0, the forward runs main's code unchanged.
  - Nothing else in the file changed. csa2/metadata.py, decoder_replay.py and _enter_bounded_replay are untouched, so there is no overlap with idea 199.

## Passed: job 1025 (1 GPU, kind=worker, committed source 1ef5ea79), passed=true
- **Import:**
  - Unset or 0: ENABLED=False and `m41._TAIL_MOE_GRAPH=False`.
  - 1: True, VERIFY=0.
  - 1 with VERIFY=8: VERIFY=8.
- **Code receipt:** no `.item()/.tolist()/nonzero/.cpu()/synchronize/torch.tensor(` in any of the 19 Python functions of the segment:
  - moe_window, _mhc_boundary, mHC.fused_hc_lagged/post_mapping, forward_MoE
  - DeepseekV4MoE.forward/compute_routed_output/_use_shared_row_split, Gate.forward, GatedMLP.forward
  - maybe_execute_in_parallel(_deferred_join), AllReduce.forward
  - CUTLASS w4a8 quantize_input/run_moe/_folds_quant/_takes_fi_decode, quantize_input_mxfp8, CutlassFusedMoEBase.run_moe
- **Runtime (synthetic, real code path):**
  - Setup: the module's real run()/moe_window() on fake layers, at hidden 5120, hc 4, M=128.
    - Window: real mHC fused lagged boundary, then an MXFP8 shared-expert chain (trtllm::mxfp8_quantize + trtllm::flashinfer_mm_mxfp8 memo path), then real post_mapping.
    - One deferred and one resolved layer, sharing the pool.
  - Correctness: 9 graphed steps per layer with fresh inputs, every output torch.equal to eager, 0 mismatches.
  - Eligibility: True only at 128 rows. False at 127/129, with a generation row, under is_cuda_graph, without replay rows, and with spec_metadata.
  - Negative control: a `.item()` in the window fails the thread_local capture (cudaErrorStreamCaptureInvalidated), falls back to eager, and the stream stays usable.
  - Host time per window on the synthetic chain: eager 322 us vs graph 47 us (median, including the input copies).

## NOT done / open blockers (follow-up)
1. **4-GPU VERIFY (acceptance item 3), not run.**
   - What must be shown: torch.equal of residual/post_mix/comb_mix/x_ffn(/ffn_pre) graph vs eager, 0 mismatches over >=8 graphed 128-row steps on every rank, and the same AR sizes and strategy as eager.
   - Command: run_gpu_job gpus=4 kind=worker, timeout ~2400 s:
     `["{python}", "experiments/ascent/tail_moe_graph_verify.py", "{checkpoint}", "10", "8"]`
   - Pass: one `[TAIL_MOE_GRAPH VERIFY SUMMARY] rank r: 20 layers (20-39), 160 graphed windows compared, 0 mismatches` line per rank (r = 0..3), and no `capture failed` line.
   - The driver (LLM API, TP4/EP4, max_num_tokens 8192, env_overrides) is untested. If `LLM(env_overrides=...)` is refused, export the env vars in the command instead (`env TRTLLM_DSV4_TAIL_MOE_GRAPH=1 TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY=8 ...`).
   - AR receipts: profile_timeline with the same argv, once with VERIFY=0 and once with the switch 0. Per EXTEND step, compare the nccl AllReduce kernel count and names: expect 40 at [2048,5120] (L0-19) + 40 at [128,5120] (L20-39), same RING/LL protocol in both. Prefill AR must stay NCCL ring under capture (landing 27 is decode-only).
2. **Serving pair, not run.**
   - Set `experiments/ascent-config/engine.json` to `{"enable_tokenization_cache": true, "env_overrides": {"TRTLLM_DSV4_TAIL_MOE_GRAPH": "1"}}`, commit, then run_serving_pair. Output class: exact.
   - Report c1/c2/c4 TTFT (p50, min, mean per arm) and TPOT.
   - Expected: c1 TTFT about -12.7 ms per step (7.9%, fact 1653), c2 about -3.9%, c4 flat. TPOT flat.
   - Report the line `c1 tail GPU-BOUND: yes/no` and the L20 host lead (trigger T2 for ideas 159/199).
3. **Risks for VERIFY to settle:**
   - A capture failure on only one rank would put an NCCL op into a discarded graph on that rank alone. VERIFY must show no capture failures. Hardening option: allreduce a capture-ok flag before replay.
   - Capture happens at the 2nd eligible step per layer, i.e. during the harness warmup. Each capture's host ms is logged.
   - Memory: one shared private pool for the 20 graphs. The CUTLASS MoE scratch under capture is allocated per capture on the capture stream (eagerMoeWorkspace.h `get(capturing=true)`), so the eager stream's workspace is untouched.
