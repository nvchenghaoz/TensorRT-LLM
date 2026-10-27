"""Untimed correctness for PreparedFP8Sm120 (branch 3, idea 19), real shapes, synthetic data.

Sequence matches the timed bench (10 warmup + 60 iterations of the 5 excluded blocks, S=4096+T).
For every iteration:
  * outputs bitwise equal to iteration 0 (determinism),
  * inputs unchanged (no mutation),
and once per block/op:
  * error of the FP8 path vs an fp32 emulation of the fp8r recipe (dequantized E4M3 x and W,
    fp32 matmul): checks the kernel computes the intended math (tolerance: rel 1e-2, BF16 output
    rounding + accumulation order),
  * error of the FP8 path and of the BF16 path (what main runs) vs the fp32 BF16-weight
    reference: the precision cost, reported.
Also runs T in {80, 160, 240, 400} to show the path is shape-generic (no per-shape compile).
"""
import json

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

H, MLP = 3072, 9216
dev = "cuda"


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def mk(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = {
        "qkv": torch.randn(3 * H, H, device=dev, generator=g) * 0.02,
        "out": torch.randn(H, H, device=dev, generator=g) * 0.02,
        "up": torch.randn(2 * MLP, H, device=dev, generator=g) * 0.02,
        "down": torch.randn(H, MLP, device=dev, generator=g) * 0.015,
    }
    w = {k: v.to(torch.bfloat16) for k, v in w.items()}
    w["a_qkv"] = primsts.PreparedFP8Sm120(w["qkv"])
    w["a_out"] = primsts.PreparedFP8Sm120(w["out"])
    w["a_up"] = primsts.PreparedFP8Sm120(w["up"], swiglu=True)
    w["a_down"] = primsts.PreparedFP8Sm120(w["down"])
    return w


@torch.inference_mode()
def cand(w, x, a):
    data, sc = primsts.quantize_fp8r(x)
    s = Fp8RowwiseQuantizedTensor(x, data, sc)
    qkv = w["a_qkv"](s)
    o = w["a_out"](a)
    h = w["a_up"](s)
    d = w["a_down"](h)
    return qkv, o, h, d


def emu(x, wq, ws):
    """fp32 emulation of the fp8r recipe: per-row E4M3 activations x per-channel E4M3 weights."""
    data, sc = primsts.quantize_fp8r(x)
    xd = data.float() * sc.float().reshape(-1, 1)
    wd = wq.float() * ws.float().reshape(-1, 1)
    return xd @ wd.t()


@torch.inference_mode()
def errors(w, x, a):
    qkv, o, h, d = cand(w, x, a)
    res = {}
    # weight quantization vs the BF16 checkpoint weight (per-channel, fp8r recipe)
    wq, ws = primsts.quantize_fp8r_weight(w["qkv"])
    res["qkv_vs_emu"] = rel(qkv, emu(x, wq, ws))
    res["qkv_vs_fp32"] = rel(qkv, x.float() @ w["qkv"].float().t())
    res["qkv_bf16_vs_fp32"] = rel(F.linear(x, w["qkv"]), x.float() @ w["qkv"].float().t())
    wq, ws = primsts.quantize_fp8r_weight(w["out"])
    res["out_vs_emu"] = rel(o, emu(a, wq, ws))
    res["out_vs_fp32"] = rel(o, a.float() @ w["out"].float().t())
    res["out_bf16_vs_fp32"] = rel(F.linear(a, w["out"]), a.float() @ w["out"].float().t())
    wq, ws = primsts.quantize_fp8r_weight(w["up"])
    up_emu = emu(x, wq, ws).to(torch.bfloat16)
    res["up_swiglu_vs_emu"] = rel(h, swiglu(up_emu))
    up32 = x.float() @ w["up"].float().t()
    g32, u32 = up32.chunk(2, dim=-1)
    sw32 = F.silu(g32) * u32
    res["up_swiglu_vs_fp32"] = rel(h, sw32)
    res["up_swiglu_bf16_vs_fp32"] = rel(swiglu(F.linear(x, w["up"])), sw32)
    wq, ws = primsts.quantize_fp8r_weight(w["down"])
    res["down_vs_emu"] = rel(d, emu(h, wq, ws))
    res["down_vs_fp32"] = rel(d, h.float() @ w["down"].float().t())
    res["down_bf16_vs_fp32"] = rel(F.linear(h, w["down"]), h.float() @ w["down"].float().t())
    return res


def main():
    torch.manual_seed(0)
    blocks = [mk(i) for i in range(5)]
    passed = True
    report = {}
    for text in (80, 160, 240, 400):
        rows = [4096] + [4096 + text] * 4
        g = torch.Generator(device=dev).manual_seed(7)
        inputs = [
            (
                torch.randn(m, H, device=dev, generator=g).to(torch.bfloat16),
                torch.randn(m, H, device=dev, generator=g).to(torch.bfloat16),
            )
            for m in rows
        ]
        pristine = [(x.clone(), a.clone()) for x, a in inputs]
        # same structure as the timed bench: 10 warmup + 60 iterations
        for _ in range(10):
            [cand(w, x, a) for w, (x, a) in zip(blocks, inputs)]
        ref0 = None
        det = True
        unchanged = True
        for it in range(60):
            outs = [cand(w, x, a) for w, (x, a) in zip(blocks, inputs)]
            unchanged &= all(
                torch.equal(x, px) and torch.equal(a, pa)
                for (x, a), (px, pa) in zip(inputs, pristine)
            )
            if ref0 is None:
                ref0 = [tuple(t.clone() for t in o) for o in outs]
            else:
                det &= all(
                    torch.equal(t, r) for o, ro in zip(outs, ref0) for t, r in zip(o, ro)
                )
        errs = [errors(w, x, a) for w, (x, a) in zip(blocks, inputs)]
        worst = {k: max(e[k] for e in errs) for k in errs[0]}
        emu_ok = all(v < 1e-2 for k, v in worst.items() if k.endswith("_vs_emu"))
        ok = det and unchanged and emu_ok
        passed &= ok
        report[str(text)] = {
            "rows": rows,
            "deterministic_60_iters": det,
            "inputs_unchanged": unchanged,
            "kernel_matches_fp8r_emulation": emu_ok,
            "worst_rel_err": worst,
        }
        print("T", text, json.dumps(report[str(text)]), flush=True)
    print(
        json.dumps(
            {
                "passed": passed,
                "compared": "PreparedFP8Sm120 vs fp32 emulation of the fp8r recipe (tol 1e-2 rel), "
                "vs iteration 0 (bitwise), inputs vs pristine copy; precision cost vs fp32 "
                "BF16-weight reference reported next to main's BF16 path",
                "output_class": "precision",
                "cases": report,
            }
        )
    )


if __name__ == "__main__":
    main()
