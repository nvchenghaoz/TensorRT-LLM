"""Branch 54 / idea 84: one launch of each fused-QKV GEMM arm at one M, for ncu (no warmup
launches of the GEMM, so the first 4 cutlass device_kernel launches are, in order:
full (served, landing 22), compact (idea 84), nocs (probe), noop (floor)).
Env: N_M (default 16464)."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idea14_epi"))
from b54_variants import build_all  # noqa: E402
from op import N, make_inputs  # noqa: E402
from rope import make_cos_sin  # noqa: E402


def main():
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    m = int(os.environ.get("N_M", "16464"))
    strided.preload(dev)
    epi.preload(dev)
    mod = epi.module()
    var = build_all(["nocs", "noop"], print)
    a, w, sfa, sfb, alpha, qn, kn, _, _ = make_inputs(m, dev)
    cs = make_cos_sin(m, dev, os.environ.get("B_CKPT"))
    tab, idx = epi.build_rope_compact(cs)
    qkv8 = torch.empty((3, m, N), dtype=torch.uint8, device=dev)
    v = torch.empty((m, N), dtype=torch.bfloat16, device=dev)
    parts = torch.zeros(((m + 127) // 128) * (N // 128) + 8, dtype=torch.float32, device=dev)
    ws = strided.workspace(dev, 0)
    base = (a, w, sfa, sfb, alpha, qn, kn, cs, qkv8, v, parts, ws, float(_MIN_SCALING_FACTOR))
    torch.cuda.synchronize()
    print("arms in launch order: full, compact, nocs, noop; M", m, "entries", tab.shape[0],
          flush=True)
    mod.qkv3_epi_fp8(*base)
    mod.qkv3_epi_fp8_compact(*base, tab, idx)
    var["nocs"].qkv3_epi_fp8(*base)
    var["noop"].qkv3_epi_fp8(*base)
    torch.cuda.synchronize()
    print('{"done": true}')


if __name__ == "__main__":
    main()
