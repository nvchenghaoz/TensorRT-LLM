"""Idea 116 (branch 50): untimed repeated-execution check that mirrors bench_pair.py call for call.

usage: repeat_check.py --checkpoint CK --captures DIR      (arm chosen by the caller's env:
       pins on = default; kill switches = TRTLLM_NVFP4_DOWN_PIN=0 TRTLLM_NVFP4_GATEUP_PIN=0)

Phase T ("timed" inputs) is bench_pair.one() without the CUDA events: the same weights
(make_weight(n, k, 1000 + i)), the same warm_gpu() and in-process tune() (one M=8192 call per
shape under autotune()), the same flush buffer, the same key order (gate_up@M for every M with
gate_up_chain@M right after it for M in CHAIN_M, then down@M), the same activations
(make_act(m, k, 11)) and alpha, the same pre-chain chunker call, and per key the bench.timed
sequence: WARMUP + ITERS calls, flush.zero_() before each, no reset in between (protocol.seq).
Phase R (extra inputs, same process, same tuner state): real captures L0/30/59 sliced to each M,
then synthetic seeds 1, 2 and heavy-tailed rows on L0 weights, per M in the order gate_up, down,
gate_up (A,B,A; A' compared to A), each key run through the same 35-call sequence.
Per key: every call int16-equal (chain: byte-equal) to call 1; every call's FP32-dequant-reference
max-abs error <= 1.10x main's pick (the tuner-cache pick run explicitly) + 1 BF16 ulp; inputs
bitwise unchanged after the sequence; the chain equal to gelu_tanh_mul_fp4_quant(main's pick).
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol  # noqa: E402
import bench_pair  # noqa: E402
import check  # noqa: E402

bench = protocol.bench
census = check.census
log = bench.log


def ref_of(xq, xsf, alpha, m, k, wdq):
    one = torch.ones(1, device="cuda")
    return (bench.dequant(xq, xsf, one, m, k) @ wdq.t()) * alpha.float()


def ulp_of(ref):
    return 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)


class Runner:
    def __init__(self, dims, flush):
        self.dims, self.flush = dims, flush
        self.t = bench.tco()
        from tensorrt_llm._torch.kernels.fused_ops import gate_up_gelu_fp4_chunked as gc
        from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
        self.gc, self.gq = gc, gelu_tanh_mul_fp4_quant
        self.dis = torch.tensor([0.25], dtype=torch.float32, device="cuda")
        self.rows = []

    def key(self, tag, name, m, xq, xsf, wq, wsf, alpha, wdq, first=None, chain=False):
        n, k = self.dims[name]
        inputs = [xq, wq, xsf, wsf, alpha]
        snap = protocol.snapshot(inputs)
        ref = ref_of(xq, xsf, alpha, m, k, wdq)
        out, calls, nd, err = protocol.seq(lambda: bench.gemm(xq, xsf, wq, wsf, alpha), self.flush, ref)
        main_out, main_tac = check.tuned_explicit(inputs, n, k, m)
        main_err = (main_out.float() - ref).abs().max().item()
        lim = 1.10 * main_err + ulp_of(ref)
        ratio = max(err) / lim
        pin = self.t.nvfp4_pin_info(n, k, m)
        row = {"key": f"{name}@{m}", "src": tag, "calls": calls, "differ_from_call1": sum(1 for v in nd if v),
               "mismatch_elems": int(sum(nd)), "ref_ratio_max": ratio, "err_max": max(err), "main_err": main_err,
               "main_pick": main_tac, "pin": pin, "vs_main_pick": check.diff(out, main_out),
               "inputs_unchanged": protocol.unchanged(snap, inputs), "finite": bool(torch.isfinite(out.float()).all())}
        if first is not None:
            row["A_prime_vs_A"] = check.diff(out, first)
        row["ok"] = (calls == protocol.CALLS and row["differ_from_call1"] == 0 and ratio <= 1.0
                     and row["inputs_unchanged"] and row["finite"]
                     and row.get("A_prime_vs_A", {"n_diff": 0})["n_diff"] == 0)
        self._log(row)
        self.rows.append(row)
        crow = None
        if chain:
            crow = self.chain(tag, m, xq, xsf, wq, wsf, alpha, main_out)
        del ref, main_out
        return out, row, crow

    def chain(self, tag, m, xq, xsf, wq, wsf, alpha, main_out):
        gc, gq, dis = self.gc, self.gq, self.dis
        bl = bench.BACKENDS.split(",")
        inputs = [xq, wq, xsf, wsf, alpha]
        snap = protocol.snapshot(inputs)
        # bench_pair calls the chunker once before timing the chain (the 'engaged' probe).
        engaged = gc.gate_up_gelu_fp4_chunked(xq, xsf, wq, wsf, alpha, torch.bfloat16, bl, dis, 1280) is not None

        def fn():
            r = gc.gate_up_gelu_fp4_chunked(xq, xsf, wq, wsf, alpha, torch.bfloat16, bl, dis, 1280)
            if r is None:
                r = gq(bench.gemm(xq, xsf, wq, wsf, alpha), dis)
            return tuple(x.reshape(-1) for x in r)

        out, calls, nd, _ = protocol.seq(fn, self.flush)
        ref = tuple(x.reshape(-1) for x in gq(main_out, dis))
        eq = all(torch.equal(protocol._bits(a), protocol._bits(b)) for a, b in zip(out, ref))
        row = {"key": f"gate_up_chain@{m}", "src": tag, "calls": calls, "differ_from_call1": sum(1 for v in nd if v),
               "mismatch_elems": int(sum(nd)), "chunk_engaged": engaged, "equal_unchunked_main_pick": eq,
               "inputs_unchanged": protocol.unchanged(snap, inputs)}
        row["ok"] = calls == protocol.CALLS and row["differ_from_call1"] == 0 and eq and row["inputs_unchanged"]
        self._log(row)
        self.rows.append(row)
        return row

    @staticmethod
    def _log(r):
        extra = (f"ref_ratio_max={r['ref_ratio_max']:.4f} main={r['main_pick']} vs_main={r['vs_main_pick']}"
                 if "ref_ratio_max" in r else
                 f"chunk={r['chunk_engaged']} eq_unchunked_main={r['equal_unchunked_main_pick']}")
        log(f"[key] {'OK ' if r['ok'] else 'BAD'} {r['src']} {r['key']}: calls {r['calls']}/{protocol.CALLS}, "
            f"differ from call 1: {r['differ_from_call1']}, {extra}, inputs unchanged={r['inputs_unchanged']}"
            + (f" A'={r['A_prime_vs_A']}" if "A_prime_vs_A" in r else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--captures", required=True)
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    t0 = time.time()
    dims = bench.read_dims(args.checkpoint)
    t = bench.tco()
    arm = "pins_on" if (t.NVFP4_GATEUP_PIN and t.NVFP4_DOWN_PIN) else (
        "pins_off" if not (t.NVFP4_GATEUP_PIN or t.NVFP4_DOWN_PIN) else "mixed")
    log(f"[arm] {arm} gate_up_pin={t.NVFP4_GATEUP_PIN} down_pin={t.NVFP4_DOWN_PIN} "
        f"env GATEUP={os.environ.get('TRTLLM_NVFP4_GATEUP_PIN')} DOWN={os.environ.get('TRTLLM_NVFP4_DOWN_PIN')}")
    log(protocol.protocol_line())
    # ---- Phase T: bench_pair.one() order and inputs.
    weights = {}
    for i, name in enumerate(("gate_up", "down")):
        n, k = dims[name]
        weights[name] = bench.make_weight(n, k, 1000 + i)
    bench.warm_gpu()
    ts = bench.tune(weights, dims)
    flush = protocol.make_flush()
    R = Runner(dims, flush)
    for name, ms in (("gate_up", bench_pair.GU_M), ("down", bench_pair.DN_M)):
        wq, wsf, wgs = weights[name]
        n, k = dims[name]
        wdq = bench.dequant(wq, wsf, torch.ones(1, device="cuda"), n, k)
        for m in ms:
            xq, xsf, xgs = bench.make_act(m, k, 11)
            alpha = (1.0 / (xgs * wgs)).float()
            out, _, _ = R.key("timed", name, m, xq, xsf, wq, wsf, alpha, wdq,
                              chain=(name == "gate_up" and m in bench_pair.CHAIN_M))
            del out
        del wdq
        torch.cuda.empty_cache()
    log(f"[phase] T done {time.time() - t0:.0f}s")
    # ---- Phase R: real captures and synthetic rows, A,B,A per M.
    d = args.captures
    man, bad = census.load_manifest(d)
    by = {}
    for e in man["entries"]:
        by.setdefault((e["layer"], e["shape"]), []).append(e)
    alphas0 = {e["shape"]: e["alpha"] for e in man["entries"] if e["layer"] == 0}
    all_m = sorted(set(bench_pair.GU_M) | set(bench_pair.DN_M))
    for layer in check.LAYERS:
        w = census.load_weights(d, man, layer)
        wdq = {nm: bench.dequant(w[nm][0], w[nm][1], torch.ones(1, device="cuda"), *dims[nm]) for nm in w}
        loaded = {}

        def src_for(name, m):
            for e in sorted(by.get((layer, name), []), key=lambda e: e["M"]):
                if e["M"] >= m:
                    if e["file"] not in loaded:
                        tt = torch.load(os.path.join(d, e["file"]))
                        loaded[e["file"]] = (tt["act_fp4"].cuda(), tt["act_sf"].cuda(),
                                             tt["alpha"].cuda().float().reshape(-1)[:1])
                    return e, loaded[e["file"]]
            return None, None

        for m in all_m:
            first = None
            for j, name in enumerate(("gate_up", "down", "gate_up")):
                e, src = src_for(name, m)
                if e is None:
                    log(f"[skip] L{layer} {name}@{m}: no capture with >= {m} rows")
                    continue
                n, k = dims[name]
                xq, xsf = check.slice_rows(src[0], src[1], m, k)
                out, _, _ = R.key(f"real-L{layer}" + ("-A'" if j == 2 else ""), name, m, xq, xsf, w[name][0],
                                  w[name][1], src[2], wdq[name], first=first if j == 2 else None,
                                  chain=(j == 0 and m in bench_pair.CHAIN_M))
                if j == 0:
                    first = out
                del out
            del first
        del loaded, wdq, w
        torch.cuda.empty_cache()
    log(f"[phase] real done {time.time() - t0:.0f}s")
    w0 = census.load_weights(d, man, 0)
    wdq0 = {nm: bench.dequant(w0[nm][0], w0[nm][1], torch.ones(1, device="cuda"), *dims[nm]) for nm in w0}
    for variant, seed, heavy in (("seed1", 1, False), ("seed2", 2, False), ("heavy", 3, True)):
        for m in all_m:
            first = None
            for j, name in enumerate(("gate_up", "down", "gate_up")):
                n, k = dims[name]
                xq, xsf, _ = bench.make_act(m, k, seed * 7919 + m, heavy)
                alpha = torch.tensor([alphas0[name]], dtype=torch.float32, device="cuda")
                out, _, _ = R.key(variant + ("-A'" if j == 2 else ""), name, m, xq, xsf, w0[name][0], w0[name][1],
                                  alpha, wdq0[name], first=first if j == 2 else None,
                                  chain=(j == 0 and m in bench_pair.CHAIN_M))
                if j == 0:
                    first = out
                del out
            del first
    log(f"[phase] synthetic done {time.time() - t0:.0f}s")
    # ---- per key (and arm) summary over every source.
    summary = {}
    for r in R.rows:
        s = summary.setdefault(r["key"], {"sources": 0, "calls_min": protocol.CALLS, "differ_from_call1": 0,
                                           "ref_ratio_max": 0.0, "inputs_unchanged": True, "ok": True})
        s["sources"] += 1
        s["calls_min"] = min(s["calls_min"], r["calls"])
        s["differ_from_call1"] += r["differ_from_call1"]
        if "ref_ratio_max" in r:
            s["ref_ratio_max"] = max(s["ref_ratio_max"], r["ref_ratio_max"])
        s["inputs_unchanged"] &= r["inputs_unchanged"]
        s["ok"] &= r["ok"]
    for kk, s in summary.items():
        log(f"[summary] {arm} {kk}: calls {s['calls_min']}/{protocol.CALLS}, differ from call 1: "
            f"{s['differ_from_call1']}, ref ratio max {s['ref_ratio_max']:.4f}, inputs unchanged "
            f"{s['inputs_unchanged']} ({s['sources']} sources) {'OK' if s['ok'] else 'BAD'}")
    fails = [f"{r['src']}|{r['key']}" for r in R.rows if not r["ok"]]
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, f"repeat_check_{arm}.json"), "w") as f:
        json.dump({"arm": arm, "protocol": protocol.protocol_line(), "rows": R.rows, "summary": summary}, f, indent=1)
    print(json.dumps({"passed": not fails and not bad, "arm": arm, "protocol": protocol.protocol_line(),
                      "tune_seconds": ts, "n_keys_run": len(R.rows), "fails": fails[:30], "sha_mismatch": bad,
                      "summary": summary, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
