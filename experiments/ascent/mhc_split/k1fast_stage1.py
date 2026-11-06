"""Fast K1 stage 1: bitwise per (P, num_warps) at 8192 tokens, then cold-L2 graph timing.

Bitwise: fast K1 layer_input vs main's K1 (fast=False) and vs mhc_fused_hc's layer_input
(runner.forward(...)[3]); fast K1 sum-of-squares totals vs k1_probe3.emulate (CUDA order) and
vs main K1's totals. Timing: 40 launches per graph on distinct input sets, L2 flushed before
every replay, ABBA rounds; arms main/w4, fast/w4, fast/w8, fast/w16, floor/w4 (no reduction).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_probe3 import emulate  # noqa: E402
from k1fast_common import EPS, HIDDEN, NHC, ColdGraph, abba, log, make_set, stats  # noqa: E402

from tensorrt_llm._torch.modules.mhc import mhc_cuda  # noqa: E402
from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import (  # noqa: E402
    mhc_lagged_layer_input,
    reducer_threads_for_tactic,
)

TACS = [("fused_half_fma", 2, 1, 512, 1), ("fused_half_fma", 2, 1, 256, 1),
        ("fused_half_fma", 2, 1, 128, 1), ("fused_all_fma", 2, 1, 0, 1)]
NWS = [int(v) for v in os.environ.get("K1_NWS", "4,8,16").split(",")]


def bitwise():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(5)
    fn = (torch.randn(24, NHC * HIDDEN, generator=gen, device=dev) * 1e-2).float()
    scale = torch.tensor([0.5, 0.7, 0.9], device=dev)
    base = (torch.randn(24, generator=gen, device=dev) * 0.1).float()
    runner = mhc_cuda._get_fused_hc_runner(NHC, HIDDEN, EPS, 1e-6, 1e-6, 2.0, 20)
    M = 8192
    ok = True
    rows = {}
    for seed in (5, 6):
        g2 = torch.Generator(device=dev)
        g2.manual_seed(seed)
        x, res, post, comb, pre, nw = make_set(M, g2, dev)
        inputs = [x, res, post, comb, fn, scale, base]
        for tac in TACS:
            P = reducer_threads_for_tactic(tac)
            ref_out = runner.forward(inputs, tactic=tac, norm_weight=nw, norm_eps=EPS, pre_mix_ext=pre)[3]
            ref_li = runner.forward(inputs, tactic=tac, norm_weight=None, norm_eps=0.0, pre_mix_ext=pre)[3]
            emu = emulate(ref_li.float(), P)
            dbg_m = torch.zeros(M, 2, device=dev)
            main = mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, debug=dbg_m, fast=False)
            main_nd = mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, fast=False)
            main_tf = mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, fast=False, enable_pdl=True,
                                             trigger_first=True)
            torch.cuda.synchronize()
            row = {"main_vs_ref_bad": int((main.view(torch.int16) != ref_out.view(torch.int16)).sum().item()),
                   "main_tf_vs_main": int((main_tf.view(torch.int16) != main_nd.view(torch.int16)).sum().item())}
            for nwp in NWS:
                dbg = torch.zeros(M, 2, device=dev)
                out_d = mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, debug=dbg, fast=True,
                                               num_warps=nwp)
                out = torch.full_like(x, float("nan"))
                mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, fast=True, num_warps=nwp, out=out)
                out_tf = torch.full_like(x, float("nan"))
                mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, fast=True, num_warps=nwp, out=out_tf,
                                       enable_pdl=True, trigger_first=True)
                out_np = torch.full_like(x, float("nan"))
                mhc_lagged_layer_input(x, res, post, comb, pre, nw, EPS, P, fast=True, num_warps=nwp, out=out_np,
                                       enable_pdl=False)
                torch.cuda.synchronize()
                r = {
                    "vs_main": int((out.view(torch.int16) != main_nd.view(torch.int16)).sum().item()),
                    "tf_vs_main": int((out_tf.view(torch.int16) != main_nd.view(torch.int16)).sum().item()),
                    "nopdl_vs_main": int((out_np.view(torch.int16) != main_nd.view(torch.int16)).sum().item()),
                    "vs_ref": int((out.view(torch.int16) != ref_out.view(torch.int16)).sum().item()),
                    "dbg_vs_main": int((out_d.view(torch.int16) != main.view(torch.int16)).sum().item()),
                    "tot_vs_emu": int((dbg[:, 0] != emu).sum().item()),
                    "tot_vs_main": int((dbg[:, 0] != dbg_m[:, 0]).sum().item()),
                    "rstd_vs_main": int((dbg[:, 1] != dbg_m[:, 1]).sum().item()),
                }
                if any(r.values()):
                    ok = False
                row[f"w{nwp}"] = r
            if row["main_vs_ref_bad"] or row["main_tf_vs_main"]:
                ok = False
            rows[f"seed{seed}/P{P}"] = row
            log(f"seed {seed} P={P}: {json.dumps(row)}")
    return ok, rows


def timing():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    n = 40
    rounds = int(os.environ.get("K1_ROUNDS", "30"))
    res = {}
    Ps = [int(v) for v in os.environ.get("K1_PS", "480,96,224,256").split(",")]
    for M in (1, 2, 4, 8):
        sets = [make_set(M, gen, dev) for _ in range(n)]
        outs = [torch.empty(M, HIDDEN, device=dev, dtype=torch.bfloat16) for _ in range(n)]
        for P in Ps:
            def mk(**kw):
                def call(st, kw=kw):
                    x, r, po, c, pr, nw = st
                    return mhc_lagged_layer_input(x, r, po, c, pr, nw, EPS, P, **kw)
                return call
            arms = {"main/w4": mk(fast=False, num_warps=4)}
            for nwp in NWS:
                arms[f"fast/w{nwp}"] = mk(fast=True, num_warps=nwp)
            arms["floor/w4"] = mk(fast=True, num_warps=4, test_floor=True)
            graphs = {k: ColdGraph(v, sets) for k, v in arms.items()}
            t = abba(graphs, flush, rounds)
            row = {k: stats(v) for k, v in t.items()}
            res[f"M{M}/P{P}"] = row
            log(f"M={M} P={P}: " + ", ".join(f"{k} {v['median']:.3f} [{v['p10']:.3f},{v['p90']:.3f}]"
                                            for k, v in row.items()))
            del graphs
        del outs
    return res


def ptx_audit():
    """Fast kernel: no st.global before griddepcontrol.wait; the ld.global before it (nw only)."""
    import re

    from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as _split
    rows, ok = [], True
    for dc in (getattr(_split._mhc_lagged_layer_input_fast_kernel, "device_caches", None) or {}).values():
        kc = dc[0] if isinstance(dc, tuple) else dc
        for k in list(kc.values()):
            ptx = k.asm.get("ptx", "")
            w = ptx.find("griddepcontrol.wait")
            if w < 0:
                continue
            pre = ptx[:w]
            n_ld = len(re.findall(r"\bld\.global", pre))
            n_st = len(re.findall(r"\bst\.global", pre))
            rows.append({"ld_before_wait": n_ld, "st_before_wait": n_st,
                         "launch_dependents": "griddepcontrol.launch_dependents" in ptx})
            ok = ok and n_st == 0
    return ok, rows


def main():
    torch.cuda.set_device(0)
    ok, rows = bitwise()
    log("bitwise passed:", ok)
    aok, audit = ptx_audit()
    log("ptx audit:", aok, audit)
    tim = timing() if os.environ.get("K1_SKIP_TIMING", "0") != "1" else {}
    print(json.dumps({"passed": bool(ok and aok), "tokens": 8192, "bitwise": rows, "ptx_audit": audit,
                      "timing_us": tim}))


if __name__ == "__main__":
    main()
