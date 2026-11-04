"""K1 total vs the canonical-order torch emulation and vs CUDA, per (P, num_warps); plus rough K1 timing."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_probe3 import emulate  # noqa: E402

from tensorrt_llm._torch.modules.mhc import mhc_cuda  # noqa: E402
from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import (  # noqa: E402
    mhc_lagged_layer_input,
    reducer_threads_for_tactic,
)


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(11)
    hidden, n, eps = 5120, 4, 1e-6
    fn = (torch.randn(24, n * hidden, generator=gen, device=dev) * 1e-2).float()
    scale = torch.tensor([0.5, 0.7, 0.9], device=dev)
    base = (torch.randn(24, generator=gen, device=dev) * 0.1).float()
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device=dev)).to(torch.bfloat16)
    runner = mhc_cuda._get_fused_hc_runner(n, hidden, eps, 1e-6, 1e-6, 2.0, 20)
    M = 8192
    x = torch.randn(M, hidden, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(M, n, hidden, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(M, n, generator=gen, device=dev)).float()
    comb = torch.rand(M, n, n, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(M, n, generator=gen, device=dev) + 1e-6).float()
    inputs = [x, res, post, comb, fn, scale, base]
    summary = {}
    ok = True
    for tac in [("fused_half_fma", 2, 1, 512, 1), ("fused_half_fma", 2, 1, 256, 1),
                ("fused_all_fma", 2, 1, 0, 1), ("fused_half_fma", 2, 1, 128, 1)]:
        P = reducer_threads_for_tactic(tac)
        ref_out = runner.forward(inputs, tactic=tac, norm_weight=nw, norm_eps=eps, pre_mix_ext=pre)[3]
        ref_li = runner.forward(inputs, tactic=tac, norm_weight=None, norm_eps=0.0, pre_mix_ext=pre)[3]
        emu = emulate(ref_li.float(), P)
        for nwp in (4, 8, 16):
            dbg = torch.zeros(M, 2, device=dev)
            out = mhc_lagged_layer_input(x, res, post, comb, pre, nw, eps, P, debug=dbg, num_warps=nwp)
            torch.cuda.synchronize()
            bad = int((out.view(torch.int16) != ref_out.view(torch.int16)).any(-1).sum().item())
            tot_bad = int((dbg[:, 0] != emu).sum().item())
            # rough timing at decode M (warm L2), production path (no debug)
            times = {}
            for m in (1, 4, 8):
                args = (x[:m], res[:m], post[:m], comb[:m], pre[:m], nw, eps, P)
                for _ in range(20):
                    mhc_lagged_layer_input(*args, num_warps=nwp)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(200):
                    mhc_lagged_layer_input(*args, num_warps=nwp)
                b.record()
                b.synchronize()
                times[m] = round(a.elapsed_time(b) * 1000 / 200, 2)
            key = f"P{P}/w{nwp}"
            summary[key] = {"out_bad_tokens": bad, "total_vs_emulation_bad": tot_bad, "us_eager": times}
            if nwp == 8 and (bad or tot_bad):
                ok = False
            print(key, json.dumps(summary[key]), flush=True)
    print(json.dumps({"passed": ok, "tokens": M, "summary": summary}))


if __name__ == "__main__":
    main()
