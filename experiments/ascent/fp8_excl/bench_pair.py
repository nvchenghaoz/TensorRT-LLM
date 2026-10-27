"""Per-step linear work of FLUX.3 Klein's 5 BF16 checkpoint-excluded image/joint blocks.

One "step" = content_mode_blocks.image.0 (M = 4096 image tokens) + single_blocks.24-27
(M = 4096 + text tokens), each running QKV, attn_out, MLP-up + SwiGLU and MLP-down on
synthetic BF16 inputs at the real shapes (hidden 3072, mlp 9216).

Base tree: BF16 F.linear x3 (q,k,v) + attn_out + gate_up + swiglu + down (what main runs).
Candidate tree (PreparedFP8Sm120 present): shared fp8r quantize of x_mod, fused FP8 QKV,
FP8 attn_out, FP8 gate_up + SwiGLU, FP8 down, through the branch's adapters.
Prints {"batch": {"1": {"kernel_ms", "noise_floor"}}, "output": {...}}.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

TEXT = int(os.environ.get("FP8X_TEXT", "160"))
ITERS = int(os.environ.get("FP8X_ITERS", "60"))
H, MLP = 3072, 9216
CAND = hasattr(primsts, "PreparedFP8Sm120")
print("candidate path" if CAND else "base path", "text", TEXT, flush=True)
dev = "cuda"
torch.manual_seed(1234)


def make_block(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = {
        "qkv": torch.randn(3 * H, H, device=dev, generator=g) * 0.02,
        "out": torch.randn(H, H, device=dev, generator=g) * 0.02,
        "up": torch.randn(2 * MLP, H, device=dev, generator=g) * 0.02,
        "down": torch.randn(H, MLP, device=dev, generator=g) * 0.015,
    }
    w = {k: v.to(torch.bfloat16) for k, v in w.items()}
    if CAND:
        w["a_qkv"] = primsts.PreparedFP8Sm120(w["qkv"])
        w["a_out"] = primsts.PreparedFP8Sm120(w["out"])
        w["a_up"] = primsts.PreparedFP8Sm120(w["up"], swiglu=True)
        w["a_down"] = primsts.PreparedFP8Sm120(w["down"])
    else:
        w["q"], w["k"], w["v"] = (t.contiguous() for t in w["qkv"].split(H, 0))
    return w


rows = [4096] + [4096 + TEXT] * 4
blocks = [make_block(i) for i in range(5)]
inputs = [(torch.randn(m, H, device=dev).to(torch.bfloat16),
           torch.randn(m, H, device=dev).to(torch.bfloat16)) for m in rows]
pristine = [(a.clone(), b.clone()) for a, b in inputs]


@torch.inference_mode()
def block_base(w, x, attn):
    q = F.linear(x, w["q"])
    k = F.linear(x, w["k"])
    v = F.linear(x, w["v"])
    o = F.linear(attn, w["out"])
    h = swiglu(F.linear(x, w["up"]))
    d = F.linear(h, w["down"])
    return q, k, v, o + d


@torch.inference_mode()
def block_cand(w, x, attn):
    data, scales = primsts.quantize_fp8r(x)
    shared = Fp8RowwiseQuantizedTensor(x, data, scales)
    q, k, v = w["a_qkv"](shared).split(H, dim=-1)
    o = w["a_out"](attn)
    h = w["a_up"](shared)
    d = w["a_down"](h)
    return q, k, v, o + d


fn = block_cand if CAND else block_base


def step():
    return [fn(w, x, a) for w, (x, a) in zip(blocks, inputs)]


for _ in range(10):
    step()
torch.cuda.synchronize()
ts = []
for i in range(ITERS):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    step()
    e.record()
    e.synchronize()
    ts.append(s.elapsed_time(e))
    if i % 20 == 0:
        print("iter", i, ts[-1], flush=True)
ts.sort()
med = ts[len(ts) // 2]
noise = (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med
unchanged = all(torch.equal(a, pa) and torch.equal(b, pb) for (a, b), (pa, pb) in zip(inputs, pristine))
print(json.dumps({
    "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
    "output": {"class": "precision" if CAND else "exact"},
    "path": "fp8r" if CAND else "bf16",
    "rows": rows,
    "inputs_unchanged": unchanged,
}))
