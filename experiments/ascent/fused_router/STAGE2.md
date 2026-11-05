# Idea 67 Stage 2 snapshot (route_assign_decode + MXFP8 q0 from the router's last CTA)

The commit that adds this file is the last tree of branch 134 with Stage 2 wired into production:
- modeling_deepseekv4.py: DeepseekV4Gate.fused_forward(decode_state=...) -> torch.ops.trtllm.dsv4_fused_router_route
  at M <= TRTLLM_DSV4_FUSED_ROUTER_ROUTE_MAX_M (default 2), DeepseekV4MoE._fi_decode_state_for
- w4a8_mxfp4_mxfp8_sm12x_decode.py: Sm12xW4a8DecodeState.preroute_buffers / set_prerouted, run() skips route()
- checks: experiments/ascent/fused_router/check2.py (workspace bitwise EP4 0/96/192/288, EP2xTP2 192)

The following commit removes the Python hand-off (Stage 1 only lands). The C++ op dsv4_fused_router_route and the
dsv4FusedRouterKernel<M,true> instantiations stay in dsv4FusedRouter.cu (unused by production). Cherry-pick this
commit's Python changes for idea 205 / 'Stage 2 of 67'.

Stage 2 snapshot SHA: 4eed0426b7b3eafa0e32983720d765a27e550564. check2.py and common.fused_route need that tree
(preroute_buffers / set_prerouted); on the Stage-1 tree use check.py and bench.py.

DONE (Stage 1, submitted): dsv4FusedRouterKernel<M,false>, M=1..8, bitwise vs dsv3_router_gemm_op + gate_forward.
UNTESTED on this tree: nothing of Stage 2 runs in production here.
NEXT: Stage 2 (route + q0 from the last CTA, M<=2) as its own landing; Stage 3 (idea 259) folds GEMV + gate into
the persistent MoE kernel (keep GEMV in main's MMA order, gate body bitwise <384,6,false,false>).
