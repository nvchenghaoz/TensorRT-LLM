"""First-call (graph build / kernel JIT) cost of the CUDNN backend per new sequence length.

H3 requests differ in text length, so each prompt gives a new packed M; the backend
builds one cuDNN graph per (shape, strides). This times the first and steady calls at
several M in one process.
"""

import json
import time

import torch

from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention

H, D = 28, 128


def main():
    be = CuDNNAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)
    out = {}
    for M in (37966, 37966 + 37, 37966 - 90, 256, 37966):
        qkv = torch.randn(1, M, 3 * H * D, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, M, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, M, H, D, device="cuda", dtype=torch.bfloat16)
        v = qkv[..., 2 * H * D:].view(1, M, H, D)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        be.forward(q, k, v)
        torch.cuda.synchronize()
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        be.forward(q, k, v)
        torch.cuda.synchronize()
        second = time.perf_counter() - t0
        print(f"M={M}: first {first * 1e3:.1f} ms, second {second * 1e3:.1f} ms", flush=True)
        out[str(M) + ("_again" if str(M) in out else "")] = dict(first_ms=first * 1e3,
                                                                 second_ms=second * 1e3)
    print(json.dumps({"passed": True, "results": out}))


if __name__ == "__main__":
    main()
