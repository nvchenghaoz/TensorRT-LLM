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

## NEXT (original)
1. When 57 lands: sync_main, add trigger_entry to the fold render's entry/signature/launch (both kF32Mid
   instantiations), extend the binding check + cake_decode.py gating to the fold render body.
2. ONE pair: chain_332.py time --spans (counts 1,2,4,8), then ONE check job: chain_332.py check --children.
3. If 57 fails its check: run the same jobs on the default bodies as is.
