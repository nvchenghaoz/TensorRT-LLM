"""Idea 237 correctness + repeated-execution check (candidate tree), one EP2xTP2 rank.

(A) Gate: with TRTLLM_SM12X_W4A8_FI_PREFILL at its default ("graph") the leaf's _takes_fi_chain
    sends M > 32 calls to FlashInfer only while capturing; eager prefill and every M <= 32 decode call
    decide exactly as main (_takes_fi_decode unchanged). With the gate =0 (subprocess) nothing
    above 32 tokens ever takes the chain, in or out of capture, and prefill_warm_tokens() is empty.
(B) Decode control: M 1..32 run() kernels = the decode set (no prefill route/tiles).
(C) Repeated execution at the bench's counts (3 warmup + 1 capture + 3 + 40 graph replays, each
    after the read-only flush) at prefill M including route-path edges and odd sizes: every eager
    call and every replay compared with the FP32 dequantized reference, with main's CUTLASS on the
    same inputs, and with the chain's own eager replay band (5 eager repeats); no NaN/Inf. The graph
    is captured the way _dsv41_tail_moe_graph captures (side stream, shared pool, thread_local).
"""

import argparse
import json
import os
import subprocess
import sys
import types

import torch

import stage0 as S
from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8 import TrtllmCutlassW4a8Mxfp4Mxfp8Impl as Impl


def fake_backend(st, hidden):
    ns = types.SimpleNamespace(_fi_decode=st, hidden_size=hidden)
    ns._fi_max_tokens = Impl._fi_max_tokens
    ns._takes_fi_decode = lambda x: Impl._takes_fi_decode(ns, x)
    return ns


def gate_probe(hidden):
    """Returns {m: (eager, capturing)} decisions of _takes_fi_chain."""
    ns = fake_backend(object(), hidden)
    out = {}
    s = torch.cuda.Stream()
    for m in (1, 8, 32, 33, 128, 2048, 8192, 8193):
        x = torch.zeros(m, hidden, dtype=torch.bfloat16, device="cuda")
        eager = Impl._takes_fi_chain(ns, x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin(capture_error_mode="thread_local")
            cap = Impl._takes_fi_chain(ns, x)
            g.capture_end()
        out[m] = (bool(eager), bool(cap))
    return out


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({ev.key[:90] for ev in prof.key_averages() if getattr(ev, "device_time_total", 0) > 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gate_only", type=int, default=0)
    ap.add_argument("--cases", default="33:60,37:80,42:90,43:90,48:101,128:66,128:166,129:166,512:192,"
                                       "1000:180,2048:120,2049:192,4096:192,8191:192")
    args = ap.parse_args()
    H, I_full, E_glob, topk, limit = S.read_cfg(args.ckpt)
    if args.gate_only:
        g = gate_probe(H)
        print(json.dumps({"mode": S.dec.FI_PREFILL_MODE, "gate": {str(k): v for k, v in g.items()},
                          "warm": S.dec.prefill_warm_tokens()}))
        return
    passed = True
    report = {}
    # (A) gate, default mode in-process; =0 in a subprocess (env read at import)
    g = gate_probe(H)
    exp = {m: ((m <= 32), (m <= 32) or (32 < m <= 8192)) for m in g}
    gate_ok = all(g[m] == exp[m] for m in g)
    env = dict(os.environ, TRTLLM_SM12X_W4A8_FI_PREFILL="0")
    r = subprocess.run([sys.executable, __file__, "--ckpt", args.ckpt, "--gate_only", "1"], env=env,
                       capture_output=True, text=True, timeout=600)
    try:
        g0 = json.loads(r.stdout.strip().splitlines()[-1])
        gate0_ok = (g0["mode"] == "0" and g0["warm"] == [] and
                    all(tuple(v) == ((int(k) <= 32), (int(k) <= 32)) for k, v in g0["gate"].items()))
    except Exception as e:  # noqa: BLE001
        g0, gate0_ok = {"error": repr(e), "stderr": r.stderr[-800:]}, False
    report["gate_default"] = {str(k): v for k, v in g.items()}
    report["gate_default_ok"] = gate_ok
    report["gate0"] = g0
    report["gate0_ok"] = gate0_ok
    passed &= gate_ok and gate0_ok
    S.log("gate default", g, "ok", gate_ok, "| gate=0", g0, "ok", gate0_ok)

    E, I = E_glob // 2, I_full // 2
    p = S.Problem(H, I, E, E_glob, topk, limit)
    p.st.warmup(topk)
    p.st.warmup_prefill(topk)
    S.log("warm tokens", S.dec.prefill_warm_tokens())
    S.FLUSH = torch.ones(128 * 1024 * 1024, dtype=torch.float32, device="cuda")

    # (B) decode control: kernels at M 1..32 contain no prefill route kernel
    dec_bad = []
    for m in (1, 2, 4, 8, 16, 32):
        x, sel, sc, _, _ = p.inputs(m, min(E, 6 * m))
        ks = kernels_of(lambda: p.st.run(x, sel, sc))
        if any(k.startswith(("count_routes", "prefix_cursor", "route_assign_kernel")) for k in ks):
            dec_bad.append((m, ks))
        if m in (1, 32):
            report[f"decode_kernels_M{m}"] = ks
    report["decode_control_ok"] = not dec_bad
    passed &= not dec_bad

    # (C) repeated execution
    cases = [tuple(int(v) for v in c.split(":")) for c in args.cases.split(",")]
    with S.autotune():
        for m, d in cases:
            x, sel, sc, _, _ = p.inputs(m, d)
            p.cutlass(x, sel, sc)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    cs = torch.cuda.Stream()
    for m, d in cases:
        key = f"M{m}_D{d}"
        x, sel, sc, n_act, n_loc = p.inputs(m, d)
        x0, sel0, sc0 = x.clone(), sel.clone(), sc.clone()
        ref = p.reference(x, sel, sc)
        oc = p.cutlass(x, sel, sc).float()
        cut_err = S.err(oc, ref)["max_abs_over_absmax"]
        eager = [p.st.run(x, sel, sc).float().clone() for _ in range(5)]
        absmax = ref.abs().max().item()
        band = max((e - eager[0]).abs().max().item() for e in eager)
        tol = max(2 * band, absmax * 2 ** -7)
        st = {"iters": 0, "max_vs_ref": 0.0, "max_vs_cutlass": 0.0, "max_vs_eager0": 0.0, "nonfinite": 0,
              "out_of_band": 0}

        def cmp(o):
            o = o.float()
            st["iters"] += 1
            if not bool(torch.isfinite(o).all()):
                st["nonfinite"] += 1
                return
            st["max_vs_ref"] = max(st["max_vs_ref"], S.err(o, ref)["max_abs_over_absmax"])
            st["max_vs_cutlass"] = max(st["max_vs_cutlass"], S.err(o, oc)["max_abs_over_absmax"])
            dv = (o - eager[0]).abs().max().item()
            st["max_vs_eager0"] = max(st["max_vs_eager0"], dv)
            if dv > tol:
                st["out_of_band"] += 1

        for e in eager:
            cmp(e)
        # capture as the tail MoE graph does
        for _ in range(3):
            cmp(p.st.run(x, sel, sc))  # eager below 33 / outside capture: the leaf would run CUTLASS,
            # but run() itself is the chain; this checks repeated eager chain calls
        cs.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(cs):
            g.capture_begin(pool=pool, capture_error_mode="thread_local")
            out = p.st.run(x, sel, sc)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(cs)
        for i in range(43):
            S.flush()
            g.replay()
            torch.cuda.synchronize()
            cmp(out)
        inputs_ok = torch.equal(x, x0) and torch.equal(sel, sel0) and torch.equal(sc, sc0)
        ok = (st["nonfinite"] == 0 and st["out_of_band"] == 0 and inputs_ok
              and st["max_vs_ref"] <= max(1.25 * cut_err, 0.06))
        passed &= ok
        report[key] = {**st, "active": n_act, "local_pairs": n_loc, "tiles": S.dec.prefill_tiles(m),
                       "cutlass_vs_ref": cut_err, "eager_band": band, "band_tol": tol, "inputs_unchanged": inputs_ok,
                       "ok": ok}
        S.log(key, json.dumps(report[key]))
        del g, out
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(passed), "compared": "FI SM12x prefill chain (leaf _run_prefill, tile table) "
                      "every eager call + 43 graph replays (tail-graph style capture) vs FP32 dequantized reference, "
                      "vs main's CUTLASS fused_moe on the same inputs, and vs its own 5-call eager band; gate default "
                      "and =0 decisions; decode M<=32 kernel set", "report": report}))


if __name__ == "__main__":
    main()
