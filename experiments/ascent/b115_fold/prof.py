"""Branch 115 (idea 182) STEP 0 / diagnostics: run qknorm_rope_fp8 on synthetic planar BF16 planes.

usage: python prof.py [M] [calls]
Warmup 3 calls (JIT), then `calls` calls; profile with launch_skip past the warmup launches.
"""

import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_qknorm_rope_fp8 as qr8

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4176
CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
H = 3072
g = torch.Generator(device="cuda").manual_seed(1)
planes = torch.randn(3, M, H, device="cuda", generator=g).bfloat16()
qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
pos = torch.arange(M, device="cuda", dtype=torch.int64)
print("warmup", flush=True)
for _ in range(3):
    qr8.qknorm_rope_fp8(planes[0], planes[1], planes[2], qn, kn, cs, pos, 128 ** -0.5)
torch.cuda.synchronize()
print("profiled calls", flush=True)
for _ in range(CALLS):
    qr8.qknorm_rope_fp8(planes[0], planes[1], planes[2], qn, kn, cs, pos, 128 ** -0.5)
torch.cuda.synchronize()
print('{"done": true}', flush=True)
