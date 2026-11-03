"""Shared setup for branch 66 (idea 104): SM120 fp8_block_scaling_bmm_out into strided o_lora.

TP4 rank shapes for DeepSeek-V4.1-Flash wo_a: G = n_local_groups = 2, R = o_lora_rank = 1024,
D = heads_per_group * head_dim = 8 * 512 = 4096. Weights [G, R, D] fp8 with UE8M0 128x128
block scales in the layout csa2/weights.py builds (transform_sf_into_required_layout).
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

import tensorrt_llm  # noqa: E402,F401  (registers torch.ops.trtllm)
import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils  # noqa: E402
from tensorrt_llm._torch.attention import mla  # noqa: E402

G, R, D = 2, 1024, 4096
LAYERS = 40
WARMUP = 3
ITERS = 15


def make_layer_weights(gen, n_layers=LAYERS):
    weights = []
    for _ in range(n_layers):
        w = (torch.randn((G, R, D), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        exps = torch.randint(-12, -4, (G, R // 128, D // 128), generator=gen, device="cuda")
        sf = torch.pow(2.0, exps.float()).contiguous()
        scale = fp8_utils.transform_sf_into_required_layout(
            sf, mn=R, k=D, recipe=(1, 128, 128), num_groups=G, is_sfa=False
        )
        weights.append((w, scale, sf))
    return weights


def make_inputs(gen, m, n_layers=LAYERS):
    return [
        torch.randn((m, G, D), generator=gen, device="cuda").to(torch.bfloat16)
        for _ in range(n_layers)
    ]


def chain(mat1, w, scale):
    """The production chain of project_sparse_attn_output (fp8 branch)."""
    m = mat1.shape[0]
    o_lora = torch.empty([m, G, R], device=mat1.device, dtype=mat1.dtype)
    mla.fp8_block_scaling_bmm_out(mat1, w, scale, o_lora.transpose(0, 1), None, False)
    return o_lora.flatten(1)


def copy_path_into(mat1, w, scale, out):
    """The base (copy) path, spelled out: new_empty + bmm + out.copy_."""
    mat1_fp8, mat1_scale = fp8_utils.per_token_quant_and_transform(mat1, need_permute102=True)
    output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
    torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, w, mat1_scale, scale, output)
    out.copy_(output)


def sm_clock_mhz():
    try:
        import subprocess

        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.run(
            ["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None
