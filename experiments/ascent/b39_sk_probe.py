"""Branch 39 probe: does FlashInfer stock SM120 StreamK split K (reordered) on the FLUX LINEAR shapes?

Randn inputs give near-exact fp32 accumulation, so a K split can be invisible. Here operands get a
wide per-16-block dynamic range (2**U(-SPREAD, SPREAD) per block) plus outlier channels, so fp32
accumulation rounds and a split-K reduction would show up as differences vs TRT-LLM t0 (sequential).
DP arms are the control (must stay bitwise). Each arm repeated NREP times (determinism).
"""

import json
import os

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

SPREAD = float(os.environ.get("B39_SPREAD", "10"))
ARMS = [int(x) for x in os.environ.get("B39_ARMS", "16,17,18,19,22,23,26,27").split(",")]
NREP = 5
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def wide(rows, k, g, outlier_cols=True):
    x = torch.randn(rows, k, device="cuda", generator=g)
    blk = torch.rand(rows, k // 16, device="cuda", generator=g) * 2 * SPREAD - SPREAD
    x = x * torch.exp2(blk).repeat_interleave(16, dim=1)
    if outlier_cols:
        cols = torch.randint(0, k, (k // 64,), device="cuda", generator=g)
        x[:, cols] *= 64.0
    return x.to(torch.bfloat16)


def ordered(x_bf16):
    s = x_bf16.view(torch.int16).int()
    return torch.where(s >= 0, s, -32768 - s)


def main():
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4

    mod = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    res = {}
    for M in (4096, 4176, 4256):
        for name, N, K in (("attn_out", H, H), ("down", H, FF)):
            g = torch.Generator(device="cuda").manual_seed(M + K)
            x = wide(M, K, g)
            w = wide(N, K, g, outlier_cols=False)
            sa = float(x.float().abs().amax() / 2688.0)
            sw = float(w.float().abs().amax() / 2688.0)
            a, sfa = fp.nvfp4_quantize(x, torch.tensor([1 / sa], device="cuda"))
            b, sfb = fp.nvfp4_quantize(w, torch.tensor([1 / sw], device="cuda"))
            alpha = _scalar(a.device, sa * sw)
            # TRT-LLM FP4GemmRunner tactic 0 (main's route), called directly
            ref = fp._sm120_fp4_gemm_runner().run_gemm(a, b, sfa, sfb, alpha, 0, 0, None, None).clone()
            row = {}
            for idx in ARMS:
                outs = []
                for _ in range(NREP):
                    o = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
                    mod.fp4_gemm(a, b, sfa, sfb, alpha, o, ws, idx)
                    outs.append(o)
                torch.cuda.synchronize()
                o = outs[0]
                rep = all(torch.equal(o, x2) for x2 in outs[1:])
                d = (o.float() - ref.float()).abs()
                ulp = (ordered(o) - ordered(ref)).abs()
                row[idx] = {"bitwise_t0": bool(torch.equal(o, ref)), "repeat_bitwise": rep,
                            "diff_frac": float((d > 0).float().mean()),
                            "max_ulp": int(ulp.max()),
                            "max_abs": float(d.max()),
                            "finite": bool(torch.isfinite(o).all())}
                log(f"M{M} {name} idx{idx}: {row[idx]}")
            res[f"M{M}_{name}"] = row
    print(json.dumps({"passed": True, "spread": SPREAD, "res": res}))


if __name__ == "__main__":
    main()
