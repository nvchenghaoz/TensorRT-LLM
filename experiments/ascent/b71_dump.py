"""Branch 71 (idea 121): dump main's Inductor LN+adaLN-modulate kernel for the FP8R block heads.

Mirrors the FP8R block heads (per-block torch.compile, dynamic=None, emulate_precision_casts):
  single: where-select of [1,2,D] shift/scale tables by a [1,S,1] mask, LN, modulate, quantize_fp8r
  mode:   [1,1,D] rows (views of a [16,1,1,3D] step table), LN, modulate, quantize_fp8r
Sequence per head (own code object each): M=4176 (static), M=4256 (dynamic recompile), M=4176.
Writes the output code to $AUTO_DEV_ARTIFACTS/b71_output_code.log and prints the LN kernels.
"""

import glob
import json
import os
import sys

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
os.environ["TORCH_LOGS"] = "output_code,recompiles"
os.environ["TORCH_LOGS_OUT"] = os.path.join(art, "b71_output_code.log")

import torch  # noqa: E402

from tensorrt_llm._torch.modules.layer_norm import LayerNorm  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r  # noqa: E402

D = 3072
dev = "cuda"


class Single(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = LayerNorm(
            hidden_size=D, eps=1e-6, has_weights=False, has_bias=False, dtype=torch.bfloat16
        ).to(dev)

    def forward(self, x, shift_t, scale_t, gate_t, select):
        shift, scale, gate = tuple(
            torch.where(select, v[:, 1:2], v[:, 0:1]) for v in (shift_t, scale_t, gate_t)
        )
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        data, scales = quantize_fp8r(x_mod.reshape(-1, D))
        return x_mod, data, scales, gate * x


class Mode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = LayerNorm(
            hidden_size=D, eps=1e-6, has_weights=False, has_bias=False, dtype=torch.bfloat16
        ).to(dev)

    def forward(self, x, shift, scale, gate):
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        data, scales = quantize_fp8r(x_mod.reshape(-1, D))
        return x_mod, data, scales, gate * x


torch.manual_seed(0)
single = torch.compile(Single(), dynamic=None, options={"emulate_precision_casts": True})
mode = torch.compile(Mode(), dynamic=None, options={"emulate_precision_casts": True})
res = {}
with torch.inference_mode():
    table = torch.randn(16, 1, 1, 3 * D, device=dev, dtype=torch.bfloat16) * 0.5
    for M in (4176, 4256, 4176):
        T = M - 4096
        x = torch.randn(1, M, D, device=dev, dtype=torch.bfloat16) * 3
        tabs = [torch.randn(1, 2, D, device=dev, dtype=torch.bfloat16) * 0.5 for _ in range(3)]
        select = torch.zeros(1, M, 1, device=dev, dtype=torch.bool)
        select[:, :T] = True
        out = single(x, *tabs, select)
        print("single", M, [tuple(o.shape) for o in out], flush=True)
    for step in (0, 5):
        x = torch.randn(1, 4096, D, device=dev, dtype=torch.bfloat16) * 3
        shift, scale, gate = table[step].chunk(3, dim=-1)
        out = mode(x, shift, scale, gate)
        print("mode", step, shift.storage_offset(), [tuple(o.shape) for o in out], flush=True)
    torch.cuda.synchronize()

log = open(os.environ["TORCH_LOGS_OUT"]).read()
print("log bytes", len(log), flush=True)
# Print every triton kernel definition that contains rsqrt (the LN kernels), with heuristics.
lines = log.splitlines()
i = 0
n = 0
while i < len(lines):
    if "@triton_heuristics" in lines[i] or "@triton.jit" in lines[i]:
        j = i
        while j < len(lines) and "''', device_str=" not in lines[j]:
            j += 1
        block = "\n".join(lines[i : j + 1])
        if "rsqrt" in block:
            n += 1
            print(f"===== LN kernel {n} =====")
            print(block)
        i = j + 1
    else:
        i += 1
print(json.dumps({"ln_kernels": n}))
