"""Synthetic per-rank MiniMax-H3 attention inputs (TP2): B=1, H=28, D=128, BF16.

Q/K: per-head RMSNorm (non-unit weight) + RoPE, fresh contiguous [B,S,H,D] buffers
(as the fused qk_norm_rope kernel writes them). V: strided view of the fused qkv
projection output (row stride 3*H*D = 10752). Returned in SDPA layout [B,H,S,D].
"""
import torch

H, D = 28, 128


def make_inputs(S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(1, S, 3 * H * D, device="cuda", dtype=torch.bfloat16, generator=g)
    wq = (1.0 + 0.2 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    wk = (1.0 + 0.2 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    ang = pos[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None, :]

    def norm_rope(x, w):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        rot = torch.cat([-x2, x1], -1)
        return (x * cos + rot * sin).to(torch.bfloat16).contiguous()

    q = norm_rope(qkv[..., : H * D].view(1, S, H, D), wq)
    k = norm_rope(qkv[..., H * D: 2 * H * D].view(1, S, H, D), wk)
    v = qkv[..., 2 * H * D:].view(1, S, H, D)
    assert v.stride(1) == 3 * H * D
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), qkv
