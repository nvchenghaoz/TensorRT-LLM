"""Does the FP8R excluded-block path give the same bits in the static and the dynamic Dynamo graph?

Compiles a single-block-like forward (dynamic=None, emulate_precision_casts, like pipeline_flux3):
adaLN-style modulation, QKV, fp32 QK-norm, RoPE, SDPA, attn_out, SwiGLU MLP, gated residual.
Variants:
  bf16      main's path (separate q/k/v F.linear, F.linear + swiglu)
  fp8_view  branch path as committed (fused FP8 QKV, q/k/v are strided views)
  fp8_contig  fused FP8 QKV but q/k/v made contiguous
  fp8_sep   three FP8 GEMMs for q, k, v (contiguous outputs, no copy)
Calls at S=4176 (static graph) twice, S=4256 (recompile -> dynamic graph), then S=4176 twice
again, and checks the S=4176 outputs are bitwise equal before and after the recompile.
"""
import json

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.visual_gen.attention.attention import apply_rotary_emb
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

H, MLP, NH, HD = 3072, 9216, 24, 128
dev = "cuda"
torch._dynamo.config.cache_size_limit = 128


class Blk(torch.nn.Module):
    def __init__(self, seed, variant):
        super().__init__()
        self.variant = variant
        g = torch.Generator(device=dev).manual_seed(seed)
        w = lambda n, k, s: (torch.randn(n, k, device=dev, generator=g) * s).to(torch.bfloat16)
        wqkv, wo, wu, wd = w(3 * H, H, 0.02), w(H, H, 0.02), w(2 * MLP, H, 0.02), w(H, MLP, 0.015)
        self.nq = torch.nn.Parameter(1 + 0.1 * torch.randn(HD, device=dev, generator=g).to(torch.bfloat16), requires_grad=False)
        self.nk = torch.nn.Parameter(1 + 0.1 * torch.randn(HD, device=dev, generator=g).to(torch.bfloat16), requires_grad=False)
        if variant == "bf16":
            self.register_buffer("wq", wqkv[:H].contiguous())
            self.register_buffer("wk", wqkv[H:2 * H].contiguous())
            self.register_buffer("wv", wqkv[2 * H:].contiguous())
            self.register_buffer("wo", wo)
            self.register_buffer("wu", wu)
            self.register_buffer("wd", wd)
        else:
            if variant == "fp8_sep":
                self.q = primsts.PreparedFP8Sm120(wqkv[:H])
                self.k = primsts.PreparedFP8Sm120(wqkv[H:2 * H])
                self.v = primsts.PreparedFP8Sm120(wqkv[2 * H:])
            else:
                self.qkv = primsts.PreparedFP8Sm120(wqkv)
            self.out = primsts.PreparedFP8Sm120(wo)
            self.up = primsts.PreparedFP8Sm120(wu, swiglu=True)
            self.down = primsts.PreparedFP8Sm120(wd)

    def forward(self, x, shift, scale, gate, cos, sin):
        b, s = x.shape[:2]
        xm = (F.layer_norm(x.float(), (H,), eps=1e-6) * (1 + scale.float()) + shift.float()).to(torch.bfloat16)
        if self.variant == "bf16":
            q, k, v = F.linear(xm, self.wq), F.linear(xm, self.wk), F.linear(xm, self.wv)
        else:
            data, sc = primsts.quantize_fp8r(xm)
            sh = Fp8RowwiseQuantizedTensor(xm, data, sc)
            if self.variant == "fp8_sep":
                q, k, v = self.q(sh), self.k(sh), self.v(sh)
            else:
                q, k, v = self.qkv(sh).split(H, dim=-1)
                if self.variant == "fp8_contig":
                    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q = q.view(b, -1, NH, HD)
        k = k.view(b, -1, NH, HD)
        v = v.view(b, -1, NH, HD)
        q, k = q.float(), k.float()
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * self.nq.float()
        k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * self.nk.float()
        q = apply_rotary_emb(q, cos, sin).to(v.dtype)
        k = apply_rotary_emb(k, cos, sin).to(v.dtype)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        attn = attn.transpose(1, 2).flatten(2)
        if self.variant == "bf16":
            o = F.linear(attn, self.wo)
            d = F.linear(swiglu(F.linear(xm, self.wu).reshape(-1, 2 * MLP)), self.wd).reshape(b, s, H)
        else:
            o = self.out(attn)
            d = self.down(self.up(sh)).reshape(b, s, H)
        return x + gate * (o + d)


def run(variant):
    blk = Blk(0, variant)
    comp = torch.compile(blk, dynamic=None, options={"emulate_precision_casts": True})
    g = torch.Generator(device=dev).manual_seed(3)

    def inputs(m):
        return (
            torch.randn(1, m, H, device=dev, generator=g).to(torch.bfloat16),
            torch.randn(1, 1, H, device=dev, generator=g).to(torch.bfloat16) * 0.1,
            torch.randn(1, 1, H, device=dev, generator=g).to(torch.bfloat16) * 0.1,
            torch.randn(1, 1, H, device=dev, generator=g).to(torch.bfloat16),
            torch.randn(1, m, 1, HD, device=dev, generator=g).float(),
            torch.randn(1, m, 1, HD, device=dev, generator=g).float(),
        )

    a80, a160 = inputs(4176), inputs(4256)
    with torch.inference_mode():
        s1 = comp(*a80)
        s2 = comp(*a80)
        comp(*a160)
        s3 = comp(*a80)
        s4 = comp(*a80)
    res = {
        "static_repeat_equal": torch.equal(s1, s2),
        "dynamic_repeat_equal": torch.equal(s3, s4),
        "static_equal_after_recompile": torch.equal(s1, s3),
        "maxdiff": (s1.float() - s3.float()).abs().max().item(),
    }
    torch._dynamo.reset()
    return res


def main():
    out = {}
    for variant in ("bf16", "fp8_view", "fp8_contig", "fp8_sep"):
        out[variant] = run(variant)
        print(variant, json.dumps(out[variant]), flush=True)
    print(json.dumps({"passed": True, "variants": out}))


if __name__ == "__main__":
    main()
