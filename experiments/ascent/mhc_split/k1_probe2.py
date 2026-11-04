"""Locate K1's remaining P=480 (bigfuse BS=512) mismatch: collapse (li) or rstd, and by how many ulps."""

import json

import torch

from tensorrt_llm._torch.modules.mhc import mhc_cuda
from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import (
    mhc_lagged_layer_input,
    reducer_threads_for_tactic,
)


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
    summary = {}
    for tac in [("fused_half_fma", 2, 1, 512, 1), ("fused_half_fma", 2, 4, 512, 1),
                ("fused_half_fma", 2, 1, 256, 1)]:
        for M in (4096, 64):
            x = torch.randn(M, hidden, generator=gen, device=dev).to(torch.bfloat16)
            res = torch.randn(M, n, hidden, generator=gen, device=dev).to(torch.bfloat16)
            post = (2.0 * torch.rand(M, n, generator=gen, device=dev)).float()
            comb = torch.rand(M, n, n, generator=gen, device=dev)
            comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
            pre = (torch.rand(M, n, generator=gen, device=dev) + 1e-6).float()
            inputs = [x, res, post, comb, fn, scale, base]
            ref = runner.forward(inputs, tactic=tac, norm_weight=nw, norm_eps=eps, pre_mix_ext=pre)
            ref_out, ref_res = ref[3], ref[0]
            ref_li = runner.forward(inputs, tactic=tac, norm_weight=None, norm_eps=0.0, pre_mix_ext=pre)[3]
            P = reducer_threads_for_tactic(tac)
            dbg = torch.zeros(M, 2, device=dev, dtype=torch.float32)
            li = mhc_lagged_layer_input(x, res, post, comb, pre, nw, eps, P, debug=dbg, debug_li=True)
            out = mhc_lagged_layer_input(x, res, post, comb, pre, nw, eps, P, debug=dbg)
            torch.cuda.synchronize()
            li_bad_tok = int((li.view(torch.int16) != ref_li.view(torch.int16)).any(-1).sum().item())
            bad_rows = (out.view(torch.int16) != ref_out.view(torch.int16)).any(-1).nonzero().flatten()
            ks = []
            for r in bad_rows[:30].tolist():
                rstd = dbg[r, 1].clone()
                lif = ref_li[r].float()
                nwf = nw.float()
                found = None
                for k in range(-6, 7):
                    cand = rstd.clone()
                    step = torch.tensor(float("inf") if k > 0 else 0.0, device=dev)
                    for _ in range(abs(k)):
                        cand = torch.nextafter(cand, step)
                    o = ((lif * cand) * nwf).to(torch.bfloat16)
                    if bool((o.view(torch.int16) == ref_out[r].view(torch.int16)).all()):
                        found = k
                        break
                ks.append(found)
            key = f"{tac}/M{M}"
            summary[key] = {"P": P, "li_mismatch_tokens": li_bad_tok, "out_mismatch_tokens": int(bad_rows.numel()),
                            "rstd_ulp_offsets_that_reproduce": ks}
            print(key, json.dumps(summary[key]), flush=True)
    print(json.dumps({"passed": True, "summary": summary}))


if __name__ == "__main__":
    main()
