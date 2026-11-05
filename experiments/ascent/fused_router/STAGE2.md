# Idea 67 Stage 2 snapshot (route_assign_decode + MXFP8 q0 from the router's last CTA)

The commit that adds this file is the last tree of branch 134 with Stage 2 wired into production:
- modeling_deepseekv4.py: DeepseekV4Gate.fused_forward(decode_state=...) -> torch.ops.trtllm.dsv4_fused_router_route
  at M <= TRTLLM_DSV4_FUSED_ROUTER_ROUTE_MAX_M (default 2), DeepseekV4MoE._fi_decode_state_for
- w4a8_mxfp4_mxfp8_sm12x_decode.py: Sm12xW4a8DecodeState.preroute_buffers / set_prerouted, run() skips route()
- checks: experiments/ascent/fused_router/check2.py (workspace bitwise EP4 0/96/192/288, EP2xTP2 192)

The following commit removes the Python hand-off (Stage 1 only lands). The C++ op dsv4_fused_router_route and the
dsv4FusedRouterKernel<M,true> instantiations stay in dsv4FusedRouter.cu (unused by production). Cherry-pick this
commit's Python changes for idea 205 / 'Stage 2 of 67'.
