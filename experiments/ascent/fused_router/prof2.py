"""Eager launches for ncu (profile_kernels, kernel_regex dsv4FusedRouterKernel, launch_count 8, no skip): per M in
(1, 2): 2 launches of dsv4FusedRouterKernel<M,false> (Stage 1 = main's kernel), then 2 of dsv4FusedRouterKernel<M,true>
(Stage 2, writing the EP4 rank-0 route workspace), distinct router weights. No other launch of the kernel precedes
them (no warmup/autotune of this kernel exists). Order of matching launches: M1 S1, M1 S1, M1 S2, M1 S2, M2 S1 x2,
M2 S2 x2."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SCALE, TOPK, biases, fused, log, make_decode_state, router_weights  # noqa: E402

import torch  # noqa: E402


def main():
    import tensorrt_llm  # noqa: F401
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(5)
    rws = router_weights(gen, 8)
    bs = biases(gen, 8)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    state, _ = make_decode_state(96, 2304, 0)
    li = 0
    for m in (1, 2):
        xs = [torch.randn(m, 5120, generator=gen, device=dev).to(torch.bfloat16) for _ in range(4)]
        # this shape's route workspace (route kernels only; not matched by the regex)
        state.run(xs[0], (torch.arange(m * TOPK, device=dev, dtype=torch.int32) % 96).view(m, TOPK),
                  torch.full((m, TOPK), 1.0 / TOPK, device=dev))
        torch.cuda.synchronize()
        for i in range(2):
            fused(xs[i], rws[li % 8], bs[li % 8], counter)
            li += 1
        torch.cuda.synchronize()
        log("M", m, "stage1 x2 (grid 48 x 640)")
        for i in range(2, 4):
            ws, buf = state.preroute_buffers(xs[i], TOPK)
            torch.ops.trtllm.dsv4_fused_router_route(
                xs[i], rws[li % 8], bs[li % 8], counter, SCALE, 0, 96, ws.offsets, ws.token_map, ws.token_weights,
                ws.dst_rows, ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)
            li += 1
        torch.cuda.synchronize()
        log("M", m, "stage2 x2 (grid 48 x 640), counter", int(counter.item()))


if __name__ == "__main__":
    with torch.no_grad():
        main()
