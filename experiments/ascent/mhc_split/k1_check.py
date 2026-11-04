"""K1 (Triton lagged layer_input) vs the real mhc_fused_hc op, every valid tactic, M=1/2/4/8.

Prints per (M, tactic, sq_mode) the number of mismatching bf16 elements of layer_input,
the PTX audit (no ld.global / st.global before griddepcontrol.wait), and a final JSON line.
"""

import json
import os
import re
import sys

import torch

from tensorrt_llm._torch.modules.mhc import mhc_cuda
from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import (
    _mhc_lagged_layer_input_kernel,
    mhc_lagged_layer_input,
    reducer_threads_for_tactic,
)

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT", "")


def load_cfg():
    try:
        with open(os.path.join(CKPT, "config.json")) as f:
            c = json.load(f)
        return c
    except Exception as e:  # noqa: BLE001
        print("config read failed", e, flush=True)
        return {}


def make_inputs(M, hidden, gen, dev):
    x = torch.randn(M, hidden, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(M, 4, hidden, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(M, 4, generator=gen, device=dev)).float().contiguous()
    comb = torch.rand(M, 4, 4, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(M, 4, generator=gen, device=dev) + 1e-6).float().contiguous()
    return x, res, post, comb, pre


def main():
    cfg = load_cfg()
    hidden = int(cfg.get("hidden_size", 5120))
    iters = int(cfg.get("hc_sinkhorn_iters", 20))
    hc_eps = float(cfg.get("hc_eps", 1e-6))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
    print(f"hidden={hidden} sinkhorn_iters={iters} hc_eps={hc_eps} rms_eps={rms_eps}", flush=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    n = 4
    K = n * hidden
    fn = (torch.randn(24, K, generator=gen, device=dev) * 1e-2).float().contiguous()
    scale = torch.tensor([0.5, 0.7, 0.9], device=dev, dtype=torch.float32)
    base = (torch.randn(24, generator=gen, device=dev) * 0.1).float().contiguous()
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device=dev)).to(torch.bfloat16)
    runner = mhc_cuda._get_fused_hc_runner(n, hidden, rms_eps, hc_eps, hc_eps, 2.0, iters)

    modes = {"approxdiv": dict(), "w4": dict(num_warps=4), "w16": dict(num_warps=16)}
    all_ok = {k: True for k in modes}
    bad_tot = {k: 0 for k in modes}
    trials = int(os.environ.get("K1_TRIALS", "4"))
    for M in (1, 2, 4, 8, 16):
        for trial in range(trials):
            x, res, post, comb, pre = make_inputs(M, hidden, gen, dev)
            inputs = [x, res, post, comb, fn, scale, base]
            tactics = runner.get_valid_tactics(inputs, None)
            tactics = [t for t in tactics if reducer_threads_for_tactic(t) is not None]
            line = {k: 0 for k in modes}
            for tac in tactics:
                ref = runner.forward(inputs, tactic=tac, norm_weight=nw, norm_eps=rms_eps,
                                     pre_mix_ext=pre)
                li_ref = ref[3]
                P = reducer_threads_for_tactic(tac)
                for name, kw in modes.items():
                    li = mhc_lagged_layer_input(x, res, post, comb, pre, nw, rms_eps, P, **kw)
                    torch.cuda.synchronize()
                    bad = int((li.view(torch.int16) != li_ref.view(torch.int16)).sum().item())
                    if bad:
                        all_ok[name] = False
                        bad_tot[name] += bad
                        line[name] += 1
                        if name == "approxdiv":
                            print(f"  MISMATCH M={M} trial={trial} tactic={tac} P={P} bad={bad}", flush=True)
            print(f"M={M} trial={trial}: {len(tactics)} tactics; tactics with mismatches per mode {line}",
                  flush=True)
    print("total mismatching elements per mode:", bad_tot, flush=True)

    # PTX audit for every compiled K1 variant.
    audits = []
    cache = getattr(_mhc_lagged_layer_input_kernel, "device_caches", None)
    kernels = []
    if cache:
        for dc in cache.values():
            kc = dc[0] if isinstance(dc, tuple) else dc
            try:
                kernels.extend(kc.values())
            except Exception:  # noqa: BLE001
                pass
    for k in kernels:
        ptx = k.asm.get("ptx", "")
        body = ptx
        w = body.find("griddepcontrol.wait")
        first_ld = min([m.start() for m in re.finditer(r"\b(ld|st)\.global", body)] or [len(body)])
        has_pdl = w >= 0
        audits.append({
            "has_wait": has_pdl,
            "wait_before_first_global": (not has_pdl) or w < first_ld,
            "launch_dependents": "griddepcontrol.launch_dependents" in body,
            "name": getattr(k, "name", "?"),
        })
    print("PTX audits:", audits, flush=True)
    passed = all_ok["approxdiv"]
    ptx_ok = all(a["wait_before_first_global"] for a in audits) and any(a["has_wait"] for a in audits)
    print(json.dumps({"passed": bool(passed and ptx_ok), "bitwise_by_mode": all_ok,
                      "compared": "K1 layer_input vs mhc_fused_hc(pre_mix_ext) layer_input, every valid "
                                  "FMA tactic, M=1/2/4/8/16, bitwise",
                      "ptx_ok": ptx_ok, "n_audited": len(audits)}))


if __name__ == "__main__":
    main()
