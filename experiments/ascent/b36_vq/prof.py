"""Branch 36: run the fused QKV op (GEMM + v_quant) at M (PROF_M) with v_quant forward then
reverse, eager, for ncu (kernel_regex v_quant). keep_v=False as served."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idea14_epi"))
from op import SM, make_inputs  # noqa: E402


def main():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    epi.preload(dev)
    for m in [int(x) for x in os.environ.get("PROF_M", "16464,32848").split(",")]:
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = make_inputs(m, dev)
        for rev in (False, True):
            epi.VQ_REVERSE = rev
            for _ in range(2):
                epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False)
            torch.cuda.synchronize()
            print(f"M{m} reverse={rev} done", flush=True)


if __name__ == "__main__":
    main()
