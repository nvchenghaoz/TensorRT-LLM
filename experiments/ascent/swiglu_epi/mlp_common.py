"""Shared setup for branch 128 GatedMLP-level checks/benchmarks.

Builds Qwen3.8-27B dense-MLP shaped NVFP4 W4A4 GatedMLPs (H=5120, I=17408)
with deterministic random FP4 weights/E4M3 block scales and runs the module's
own post_load_weights (where the candidate tree permutes gate_up rows). Runs
unchanged in the base tree (main) and the candidate tree.
"""
import hashlib
import os
import subprocess

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4  # noqa: F401
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

H, I = 5120, 17408
EPS = 1e-6
dev = "cuda"
GS_X = 448.0 * 6.0 / 12.0
ALPHA_GU = 0.0005
ALPHA_D = 0.002


def tree_info():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    files = subprocess.run(["git", "diff", "--name-only", "66cff73e70c6452eaa12f3261f8cddf0e976668a", "HEAD"],
                           capture_output=True, text=True).stdout.split()
    has_op = hasattr(torch.ops.trtllm, "nvfp4_gate_up_swiglu_nvfp4")
    return {"head": head, "tree": tree, "dirty_count": len([l for l in dirty.splitlines() if l.strip()]),
            "files_vs_base": files, "has_swiglu_epi_op": has_op}


def build_mlp(seed, gs_h):
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    mlp = GatedMLP(hidden_size=H, intermediate_size=I, bias=False, dtype=torch.bfloat16,
                   config=ModelConfig(quant_config=qc), layer_idx=0).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        for lin, n, k in ((mlp.gate_up_proj, 2 * I, H), (mlp.down_proj, H, I)):
            lin.weight.data.copy_(torch.randint(0, 256, lin.weight.shape, dtype=torch.uint8, device=dev,
                                                generator=g))
            sf2 = torch.randint(0x28, 0x38, (n, k // 16), dtype=torch.uint8, device=dev, generator=g)
            lin.weight_scale.data.copy_(torch.ops.trtllm.block_scale_interleave(sf2).view_as(lin.weight_scale))
            lin.input_scale.data.fill_(gs_h if lin is mlp.down_proj else GS_X)
            lin.alpha.data.fill_(ALPHA_D if lin is mlp.down_proj else ALPHA_GU)
            lin.weight_scale_2.data.fill_(1.0)
            if getattr(lin, "inv_input_scale", None) is not None:
                lin.inv_input_scale.data.fill_(1.0 / lin.input_scale.item())
    mlp.post_load_weights()
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        lin.post_load_weights()
    return mlp


def make_input(m, seed, kind="random", quantized=False, gamma=None):
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((m, H), generator=gen, device=dev) * 0.7
    r = torch.randn((m, H), generator=gen, device=dev) * 3.0
    if kind == "outlier":
        idx = torch.randint(0, m, (max(1, m // 64), ), generator=gen, device=dev)
        r[idx] *= 40.0
        x[:, ::97] *= 25.0
    x, r = x.to(torch.bfloat16), r.to(torch.bfloat16)
    if not quantized:
        return (x + r).to(torch.bfloat16)
    if gamma is None:
        gamma = (torch.randn(H, generator=gen, device=dev) * 0.3).to(torch.bfloat16)
    gs = torch.tensor([GS_X], dtype=torch.float32, device=dev)
    fp4, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x, r, gamma, EPS, gs)
    return Fp4QuantizedTensor(fp4, sf)


class DownInputTap:
    """Captures down_proj's input (FP4 bytes + SF) on every forward."""

    def __init__(self, mlp):
        self.last = None
        mlp.down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        x = args[0]
        if isinstance(x, Fp4QuantizedTensor):
            self.last = (x.fp4_tensor.reshape(-1, x.fp4_tensor.shape[-1]).clone(), x.scaling_factor.clone())
        else:
            self.last = (x.clone(), None)


def sha(t):
    if t is None:
        return None
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:20]


def env_note():
    return {k: v for k, v in os.environ.items() if k.startswith("TRTLLM_")}
