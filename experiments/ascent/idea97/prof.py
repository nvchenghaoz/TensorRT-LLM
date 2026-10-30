"""Idea 97 profiling targets. usage: python prof.py b | decode <checkpoint_root>
  b      : kernel A then kernel B once each at T=65536 (ncu target: _pv_prologue_kernel).
  decode : served VAE decoder (bf16 trunk + FP8 convs, compiled) at 2048^2; 2 warm decodes,
           then one decode inside cudaProfilerStart/Stop (nsys capture=region).
"""

import sys

import torch

MODE = sys.argv[1]


@torch.inference_mode()
def run_b():
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

    C, T = 512, 65536
    g = torch.Generator(device="cuda").manual_seed(0)
    q = (torch.randn(T, C, device="cuda", generator=g) * 1.5).to(torch.bfloat16).t()
    k = (torch.randn(T, C, device="cuda", generator=g) * 1.5).to(torch.bfloat16).t()
    v = torch.randn(C, T, device="cuda", generator=g)
    s_v = (v.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)
    vq = (v / s_v).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    s, mp = sa.qk_tilemax(q, k)
    o, _, _ = sa.pv_prologue(s, mp, vq, s_v.reshape(C).contiguous(), C**-0.5)
    torch.cuda.synchronize()
    print("{\"passed\": true}")


@torch.inference_mode()
def run_decode(root):
    sys.argv = [sys.argv[0], root]
    sys.path.insert(0, "experiments/ascent/idea97")
    import vae_check as vc

    vc.ROOT = root
    vae, dt = vc.build()
    z = vc.latent(0, dt)
    for _ in range(2):
        vae.decode(z)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("decode")
    vae.decode(z)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("{\"passed\": true}")


if MODE == "b":
    run_b()
else:
    run_decode(sys.argv[2])
