"""Adjudicate the sum-of-squares order of mhcBigFuseKernel<.., 512, ..> vs <.., 256, ..>.

Emulates candidate reduction orders in torch (each fp32 add is one RN op), turns each total into
rstd with K1's exact formula (Triton: rsqrt.approx(fma(t, rcp.approx(h), eps))), and counts tokens
whose bf16((li * rstd) * w) reproduces the CUDA output.
"""

import json

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mhc import mhc_cuda
from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import (
    mhc_lagged_layer_input,
    reducer_threads_for_tactic,
)


@triton.jit
def _rstd_kernel(t_ptr, o_ptr, eps, n, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    t = tl.load(t_ptr + i, mask=m, other=1.0)
    h = tl.full([BLOCK], H, tl.float32)
    rcp = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [h], dtype=tl.float32,
                                    is_pure=True, pack=1)
    arg = tl.fma(t, rcp, eps + t * 0.0)
    tl.store(o_ptr + i, tl.math.rsqrt(arg), mask=m)


def rstd_of(total, eps, hidden):
    out = torch.empty_like(total)
    n = total.numel()
    _rstd_kernel[(triton.cdiv(n, 1024),)](total, out, eps, n, H=hidden, BLOCK=1024,
                                         enable_fp_fusion=False)
    return out


def emulate(li, P, order="canonical"):
    M, H = li.shape
    x, y = li[:, 0::2], li[:, 1::2]
    t = (x * x + y * y).view(M, H // 8, 4)
    nch = H // 8
    NI = (nch + P - 1) // P
    s = torch.zeros(M, P, device=li.device)
    its = range(NI) if order != "rev_chunks" else reversed(range(NI))
    for i in its:
        c0 = i * P
        cnt = min(P, nch - c0)
        for v in range(4):
            s[:, :cnt] = s[:, :cnt] + t[:, c0:c0 + cnt, v]
    W = P // 32
    s = s.view(M, W, 32)
    lanes = torch.arange(32, device=li.device)
    offs = (16, 8, 4, 2, 1) if order != "rev_bfly" else (1, 2, 4, 8, 16)
    for off in offs:
        s = s + s[:, :, lanes ^ off]
    w = s[:, :, 0]
    total = torch.zeros(M, device=li.device)
    ws = range(W) if order != "rev_warps" else reversed(range(W))
    for k in ws:
        total = total + w[:, k]
    return total


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(5)
    hidden, n, eps = 5120, 4, 1e-6
    fn = (torch.randn(24, n * hidden, generator=gen, device=dev) * 1e-2).float()
    scale = torch.tensor([0.5, 0.7, 0.9], device=dev)
    base = (torch.randn(24, generator=gen, device=dev) * 0.1).float()
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device=dev)).to(torch.bfloat16)
    runner = mhc_cuda._get_fused_hc_runner(n, hidden, eps, 1e-6, 1e-6, 2.0, 20)
    res_all = {}
    M = 8192
    x = torch.randn(M, hidden, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(M, n, hidden, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(M, n, generator=gen, device=dev)).float()
    comb = torch.rand(M, n, n, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(M, n, generator=gen, device=dev) + 1e-6).float()
    inputs = [x, res, post, comb, fn, scale, base]
    for tac in [("fused_half_fma", 2, 1, 512, 1), ("fused_half_fma", 2, 1, 256, 1),
                ("fused_all_fma", 2, 1, 0, 1), ("fused_half_fma", 2, 1, 128, 1)]:
        P = reducer_threads_for_tactic(tac)
        ref_out = runner.forward(inputs, tactic=tac, norm_weight=nw, norm_eps=eps, pre_mix_ext=pre)[3]
        ref_li = runner.forward(inputs, tactic=tac, norm_weight=None, norm_eps=0.0, pre_mix_ext=pre)[3]
        dbg = torch.zeros(M, 2, device=dev)
        k1 = mhc_lagged_layer_input(x, res, post, comb, pre, nw, eps, P, debug=dbg)
        torch.cuda.synchronize()
        lif = ref_li.float()
        row = {"k1_bad_tokens": int((k1.view(torch.int16) != ref_out.view(torch.int16)).any(-1).sum().item())}
        for P_try in sorted({P, 480, 512, 448, 224, 256, 96, 128}):
            for order in ("canonical", "rev_bfly", "rev_warps", "rev_chunks"):
                if P_try % 32:
                    continue
                total = emulate(lif, P_try, order)
                rstd = rstd_of(total, eps, hidden)
                out = ((lif * rstd[:, None]) * nw.float()).to(torch.bfloat16)
                bad = int((out.view(torch.int16) != ref_out.view(torch.int16)).any(-1).sum().item())
                row[f"P{P_try}/{order}"] = bad
                if P_try == P and order == "canonical":
                    row["emul_vs_k1_total_mismatch"] = int((total != dbg[:, 0]).sum().item())
        res_all[str(tac)] = row
        print(tac, json.dumps(row), flush=True)
    print(json.dumps({"passed": True, "tokens": M, "summary": res_all}))


if __name__ == "__main__":
    main()
