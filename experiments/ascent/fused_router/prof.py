"""Eager launches for ncu: per M (1, 4), 40 layers of router_gemm_mma_kernel + gate_forward_kernel, then 40 of
dsv4FusedRouterKernel (Stage 1), then (if built) 40 of the Stage 2 route variant, distinct weights (cold)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LAYERS, biases, fused, gate_main, log, router_main, router_weights  # noqa: E402

import torch  # noqa: E402


def main():
    import tensorrt_llm  # noqa: F401
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(5)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    for m in (1, 4):
        xs = [torch.randn(m, 5120, generator=gen, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]
        for li in range(LAYERS):
            gate_main(router_main(xs[li], rws[li]), bs[li])
        for li in range(LAYERS):
            fused(xs[li], rws[li], bs[li], counter)
        torch.cuda.synchronize()
        log("M", m, "done")


if __name__ == "__main__":
    with torch.no_grad():
        main()
