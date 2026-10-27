"""Does the FP8R excluded-block path give the same bits in the static and the dynamic Dynamo graph?

Compiles a block-like function (dynamic=None, emulate_precision_casts, like pipeline_flux3) that
runs the branch's PreparedFP8Sm120 path, calls it at S=4176 (static graph), at S=4256 (recompile,
dynamic graph), then at S=4176 again, and compares with the first call and with eager.
"""
import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

H, MLP = 3072, 9216
dev = "cuda"
torch._dynamo.config.cache_size_limit = 128


class Blk(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator(device=dev).manual_seed(seed)
        w = lambda n, k, s: (torch.randn(n, k, device=dev, generator=g) * s).to(torch.bfloat16)
        self.qkv = primsts.PreparedFP8Sm120(w(3 * H, H, 0.02))
        self.out = primsts.PreparedFP8Sm120(w(H, H, 0.02))
        self.up = primsts.PreparedFP8Sm120(w(2 * MLP, H, 0.02), swiglu=True)
        self.down = primsts.PreparedFP8Sm120(w(H, MLP, 0.015))

    def forward(self, x, a, shift, scale):
        xm = (x * (1 + scale) + shift).to(torch.bfloat16)
        data, sc = primsts.quantize_fp8r(xm)
        s = Fp8RowwiseQuantizedTensor(xm, data, sc)
        q, k, v = self.qkv(s).split(H, dim=-1)
        o = self.out(a)
        h = self.up(s)
        d = self.down(h)
        return x + o + d, q.float().pow(2).mean(-1), k + v


def main():
    blk = Blk(0)
    comp = torch.compile(blk, dynamic=None, options={"emulate_precision_casts": True})
    g = torch.Generator(device=dev).manual_seed(3)

    def inputs(m):
        return (
            torch.randn(1, m, H, device=dev, generator=g).to(torch.bfloat16),
            torch.randn(1, m, H, device=dev, generator=g).to(torch.bfloat16),
            torch.randn(1, 1, H, device=dev, generator=g).to(torch.bfloat16) * 0.1,
            torch.randn(1, 1, H, device=dev, generator=g).to(torch.bfloat16) * 0.1,
        )

    a80 = inputs(4176)
    a160 = inputs(4256)
    res = {}
    with torch.inference_mode():
        eager = blk(*a80)
        s1 = comp(*a80)
        s2 = comp(*a80)
        d1 = comp(*a160)
        s3 = comp(*a80)
        s4 = comp(*a80)
        eq = lambda u, v: all(torch.equal(p, q) for p, q in zip(u, v))
        res["static_repeat_equal"] = eq(s1, s2)
        res["after_dynamic_equal_to_static"] = eq(s1, s3)
        res["dynamic_repeat_equal"] = eq(s3, s4)
        res["static_equal_eager"] = eq(s1, eager)
        res["dynamic_equal_eager"] = eq(s3, eager)
        res["maxdiff_static_vs_dynamic"] = [
            (p.float() - q.float()).abs().max().item() for p, q in zip(s1, s3)
        ]
    print(json.dumps(res), flush=True)
    torch._dynamo.reset()
    print(json.dumps({"passed": res["after_dynamic_equal_to_static"], **res}))


if __name__ == "__main__":
    main()
