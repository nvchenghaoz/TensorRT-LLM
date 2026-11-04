"""Which mean/rsqrt formula reproduces the fast-math CUDA RMSNorm in mhc_fused_hc bitwise?

Many tokens (M=4096) per tactic so rare 1-ulp rstd differences show up. For each DIV_MODE of
K1 counts tokens whose layer_input differs from mhc_fused_hc's.
"""

import json
import os

import torch

from tensorrt_llm._torch.modules.mhc import mhc_cuda
from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import (
    mhc_lagged_layer_input,
    reducer_threads_for_tactic,
)

MODES = {0: "a/h+eps (div.full)", 1: "div_rn+eps", 2: "fma(a,rcp.approx(h),eps)",
         3: "a*rcp.approx(h)+eps", 4: "div.approx.ftz+eps", 5: "div.full.ftz+eps"}


def main():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    hidden, n = 5120, 4
    rms_eps = 1e-6
    fn = (torch.randn(24, n * hidden, generator=gen, device=dev) * 1e-2).float()
    scale = torch.tensor([0.5, 0.7, 0.9], device=dev)
    base = (torch.randn(24, generator=gen, device=dev) * 0.1).float()
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device=dev)).to(torch.bfloat16)
    runner = mhc_cuda._get_fused_hc_runner(n, hidden, rms_eps, 1e-6, 1e-6, 2.0, 20)
    M = int(os.environ.get("PROBE_M", "4096"))
    tactics = [("fused_half_fma", 2, 1, 128, 1), ("fused_half_fma", 2, 1, 256, 1),
               ("fused_half_fma", 2, 1, 512, 1), ("fused_all_fma", 2, 1, 0, 1)]
    totals = {m: 0 for m in MODES}
    report = {}
    for scale_x in (0.05, 1.0, 20.0):
        x = (torch.randn(M, hidden, generator=gen, device=dev) * scale_x).to(torch.bfloat16)
        res = (torch.randn(M, n, hidden, generator=gen, device=dev) * scale_x).to(torch.bfloat16)
        post = (2.0 * torch.rand(M, n, generator=gen, device=dev)).float()
        comb = torch.rand(M, n, n, generator=gen, device=dev)
        comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
        pre = (torch.rand(M, n, generator=gen, device=dev) + 1e-6).float()
        for tac in tactics:
            ref = runner.forward([x, res, post, comb, fn, scale, base], tactic=tac,
                                 norm_weight=nw, norm_eps=rms_eps, pre_mix_ext=pre)[3]
            P = reducer_threads_for_tactic(tac)
            row = {}
            for m in MODES:
                li = mhc_lagged_layer_input(x, res, post, comb, pre, nw, rms_eps, P, div_mode=m)
                torch.cuda.synchronize()
                bad_tok = int((li.view(torch.int16) != ref.view(torch.int16)).any(-1).sum().item())
                row[m] = bad_tok
                totals[m] += bad_tok
            report[f"x{scale_x}:{tac}"] = row
            print(f"scale={scale_x} tactic={tac} P={P}: tokens mismatching per mode {row}", flush=True)
    print("totals:", {MODES[m]: v for m, v in totals.items()}, flush=True)
    best = [MODES[m] for m, v in totals.items() if v == 0]
    print(json.dumps({"passed": bool(best), "exact_modes": best, "tokens_per_case": M,
                      "totals": {MODES[m]: v for m, v in totals.items()}}))


if __name__ == "__main__":
    main()
