"""Run one arm a few times (for ncu/nsys): ARM=cand_gemm|vquant|main_gemm, M from env."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from op import SM, make_inputs  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided  # noqa: E402

dev = torch.device("cuda")
m = int(os.environ.get("M", "16896"))
arm = os.environ.get("ARM", "cand_gemm")
strided.preload(dev)
epi.preload(dev)
inp = make_inputs(m, dev)
a, w, sfa, sfb, alpha, qn, kn, cs, pos = inp
for i in range(int(os.environ.get("REPS", "6"))):
    if arm == "main_gemm":
        out = torch.empty((m, 3 * 3072), dtype=torch.bfloat16, device=dev)
        strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)
    else:
        epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False)
torch.cuda.synchronize()
print('{"done": true}')
