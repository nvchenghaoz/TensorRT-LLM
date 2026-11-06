# Branch 188 (idea 332, Cake decode_body entry trigger) STATUS, 20:3x PST 11-05

## DONE (code, untested on GPU)
- Overlay h16_dual_default.cu / h16_single_default.cu: new last kernel arg `int trigger_entry`; at kernel entry
  `if (trigger_entry) griddepcontrol.launch_dependents`. Existing triggers kept. Launch wrappers pass
  `a.trigger_entry`. kernels.h: `DecodeArgs::trigger_entry` appended (other renders ignore it).
- Binding: enable_pdl bit 2 -> trigger_entry; TVM_FFI_CHECK fails if bit 2 reaches a non-default body.
- cake_decode.py: `_BODY_TRIGGER_ENTRY_COUNTS` (default counts 1-8, hypothesis), env
  TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY=1/0 forces on/off; bit 2 only with PDL on, fp32 partials and the
  `default` variant.
- Harness experiments/ascent/cake_q1/chain_332.py (imports chain.py; chain.py untouched): `time --spans`
  (A0 off / A1 default, interleaved, Kineto spans: merge tail, K2'/K3' offsets) and `check --children`
  (bitwise arms, skip-wait control, 8 advancing steps, child processes for TRTLLM_ENABLE_PDL=0 and
  TRTLLM_CSA2_CAKE_F32_PARTIALS=0).
- WAR audit (from source, printed by check): merge<true>/<false>, K2' and K3' (flags 0) write no global memory
  before griddepcontrol.wait (K3' debug stamp only with a non-null ts pointer). K3' triggers only after its
  wait, so nothing beyond K3' (and no next-layer kernel) can launch before this layer's body ends.

## HELD
- Pair 1702/1703 cancelled while queued (top lead 20:1x: hold Stage 0 until landing 57 = branch 184 fold
  render lands; post-57 main routes count<=16 dual calls to the fold render, so the trigger must go there too).

## 21:1x PST
- Branch base = d4372068 (landing 57 in); fold render (both h16 sa+lq+el / sa+pw+lq+el namespaces, both
  kF32Mid instantiations via fold::launch_one) takes trigger_entry. Pair 1708/1709 cancelled before start.
- chain_332.py: DIRTY SLOT guard (exit 3 if > 4 GiB used at start; children skip it via CHAIN332_CHILD).
- Queued: ONE check job (check --children, incl. F32_PARTIALS=0 child) + ONE same-slot pair
  (time --spans --hash-children, counts 1,2,4,8) vs base d4372068.

## RESULT (node smc521ge-0043, commit c833b20d)
- 1731 failed at JIT build: the fold render's CAKE_FOLD_H16_INSTANTIATE macro kept the old signature; fixed in c833b20d.
- Check 1752 PASSED (austin:1): every arm bitwise, 110 iters x counts 1/2/4/8, 8 advancing steps, children
  TRTLLM_ENABLE_PDL=0 and TRTLLM_CSA2_CAKE_F32_PARTIALS=0 bitwise vs main's code under the same env; skip-wait control differs.
- Pair 1764 (main d4372068) / 1765 (branch), austin:2, 40-layer cold-L2 chain, ms/step (noise ~0.45%):
  c1 0.8893 -> 0.8786 (-1.2%), c2 0.8949 -> 0.8748 (-2.2%), c4 0.9604 -> 0.9349 (-2.7%), c8 1.0816 -> 1.0825 (+0.1%, tie).
  In-tree A1-A0: -15.4 / -21.2 / -24.7 / +2.0 us per 40 layers. Spans c1: K2' starts 12.5 us before merge end (A0 6.5),
  K3' ends 15.7 us after merge end (A0 20.0). Output sha256 identical across trees for default / F32_PARTIALS=0 /
  ENABLE_PDL=0 / ATTN_PDL=0 at every count.

## EVALUATOR ROUND (00:0x PST 11-06, commit dd1faf8e)
- Default narrowed: `_BODY_TRIGGER_ENTRY_COUNTS = {1,2,3,4}`; counts >= 5 launch as main (A0 path, trigger_entry=0).
- Pair 1781 (main d4372068) / 1782 (branch), same slot austin:3, SM 2347 MHz, counts 1-8, ms per 40-layer step
  (median, p10-p90; noise 0.33-0.59%):
  c1 0.8837 (0.8824-0.8858) -> 0.8734 (0.8704-0.8755) -1.17% | c2 0.8900 -> 0.8683 -2.44% |
  c3 0.9216 -> 0.8947 -2.92% | c4 0.9554 -> 0.9297 -2.69% | c5 0.9891 -> 0.9895 +0.04% | c6 1.0117 -> 1.0126 +0.09% |
  c7 1.0567 -> 1.0565 -0.02% | c8 1.0735 -> 1.0756 +0.20% (5-8: unchanged path, within noise).
- In-tree forced-ON (A2) vs A0 at 5-8: c5 962.9 vs 989.5 us (-2.7%), c6 999.8 vs 1010.8 (-1.1%), c7 1056.1 vs 1055.7 (tie),
  c8 1077.2 vs 1075.2 (+0.2%, body 7.01 -> 7.57 us/layer). Crossover between 6 and 7. Counts 5/6 are NOT enabled here:
  their cross-tree ON gain was not measured in a pair (no time before the 01:00 hard stop); follow-up = add 5,6 and re-pair.
- Spans c4: K2' start rel merge end A0 -4.7 -> A1 -9.9 us; merge tail 1.7 -> 3.7 us; body 6.56 -> 7.66 us.
- Output sha256 identical across trees at counts 1-8 for default / F32_PARTIALS=0 / ENABLE_PDL=0 / ATTN_PDL=0.
- Check 1802 (check --children --counts 1,2,3,4,8) sat queued > 70 min (coach 9744 hold) and was cancelled at
  01:2x PST per area lead's final instruction (no new GPU job; hard stop 01:45). Submission refused: check 1752
  is on c833b20d, not dd1faf8e.

## WIP CLOSE (01:2x PST 11-06)
- DONE: interface (trigger_entry kernel arg, binding enable_pdl bit 2, TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY 1/0),
  default ON at {1,2,3,4} (commit dd1faf8e). Pair 1781/1782 measured on dd1faf8e (table above, bitwise hashes
  across trees). Check 1752 bitwise on c833b20d (same code, default {1..8}; every count also run with entry OFF).
- UNTESTED: the untimed check on dd1faf8e's bucket edges (counts 3 ON, 8 OFF via default).
- NEXT: ONE 1-GPU job `chain_332.py check {checkpoint} --children --counts 1,2,3,4,8` on dd1faf8e (or later), then
  submit with pair 1781/1782 + that check. Optional follow-up: add 5,6 to the set (in-tree A2 -2.7% / -1.1%) and
  re-pair counts 1-8.

## NEXT (original)
1. When 57 lands: sync_main, add trigger_entry to the fold render's entry/signature/launch (both kF32Mid
   instantiations), extend the binding check + cake_decode.py gating to the fold render body.
2. ONE pair: chain_332.py time --spans (counts 1,2,4,8), then ONE check job: chain_332.py check --children.
3. If 57 fails its check: run the same jobs on the default bodies as is.
