"""Idea 17 / ruling (c): ncu driver for main's served GEMM kernels at the t2i M=16896.

Launch order (each kernel twice; profile the second): QKV3 (Pingpong 128x128x128 CLC, L=3),
SwiGLU-FP4 (Pingpong 128x128x128 CLC EPI_VEC), LPT pair (grouped Coop 128x128x128; preceded by
its 1-CTA args kernel), FP8R gate_up arm 1 (Pingpong 128x128x64). Synthetic inputs, no model.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0  # noqa: E402

M = int(os.environ.get("NCU_M", "16896"))


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

    dev = torch.device("cuda")
    strided.preload(dev)
    lpt.preload(dev)
    fg.preload(dev)
    swm = swc.get_module()
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    alpha = torch.tensor([1.0 / 64], dtype=torch.float32, device=dev)
    a3 = torch.tensor([1.0 / 64] * 3, dtype=torch.float32, device=dev)
    q = step0.fp4_set(M, 9216, 3072, g, dev)
    qo = torch.empty(M * 9216, dtype=torch.bfloat16, device=dev)
    s = step0.fp4_set(M, 18432, 3072, g, dev)
    so, sso = swc.allocate_outputs(M, 18432, dev)
    inv = torch.tensor([0.5], dtype=torch.float32, device=dev)
    d = step0.fp4_set(M, 3072, 9216, g, dev)
    at = step0.fp4_set(M, 3072, 3072, g, dev)
    od = torch.empty((M, 3072), dtype=torch.bfloat16, device=dev)
    oa = torch.empty((M, 3072), dtype=torch.bfloat16, device=dev)
    f8a = (torch.randn((M, 3072), device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    f8w = (torch.randn((18432, 3072), device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    sa = torch.rand(M, device=dev, generator=g) * 0.01 + 1e-3
    sb = torch.rand(18432, device=dev, generator=g) * 0.01 + 1e-3
    f8o = torch.empty((M, 18432), dtype=torch.bfloat16, device=dev)
    torch.cuda.synchronize()
    for _ in range(2):
        strided.gemm_qkv3(*q, a3, qo)
    torch.cuda.synchronize()
    for _ in range(2):
        swc.swiglu_packed(*s, alpha, inv, module=swm, output=so, output_scale=sso)
    torch.cuda.synchronize()
    for _ in range(2):
        lpt.run(*d, alpha, od, *at, alpha, oa)
    torch.cuda.synchronize()
    for _ in range(2):
        fg.gemm(f8a, f8w, sa, sb, arm=1, out=f8o)
    torch.cuda.synchronize()
    print("done", flush=True)


if __name__ == "__main__":
    main()
