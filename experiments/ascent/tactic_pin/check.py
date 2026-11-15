"""Idea 116 (branch 50): numerics of the NVFP4 tactic pins (untimed, no model load).

usage: check.py --checkpoint CK --captures DIR

Real activations: the shared capture set (manifest sha256 verified), layers 0/30/59, sliced to
each served M by whole 128-row SF tiles (rows [0, M) of the smallest capture with >= M rows).
Synthetic: seeds 1, 2 and heavy-tailed randn*2^U(-8,8) rows with layer-0 real weights.
Tuning as the engine warmup does: one M=8192 call per shape under autotune() (real L0 weights).

Per (source, layer, shape, M), in the order gate_up M, down M, gate_up M (A, B, A):
  - pinned output: trtllm::nvfp4_gemm with the pins on, called WARMUP+ITERS times; every call
    int16-equal to call 1; the A' call int16-equal to A;
  - kill switches: pins off (both module flags 0) == the tuned pick run explicitly from the
    tuner cache (the code path main runs), int16;
  - every tactic main can run at that M (TRT CUTLASS c; cuBLASLt i of the actual-M list; for the
    gate_up dense range cuBLASLt i of the dense bucket's list): int16 mismatch count, max abs and
    max BF16-ULP difference against the pinned output; whether main's tuner offers it;
  - FP32 dequant reference: pinned error <= 1.10x the tuned pick's error + 1 BF16 ulp
    (max-abs) and <= 1.10x (relative Frobenius);
  - gate_up M >= 1664: the chunked chain engages with the pins and is byte-equal to the
    unchunked chain gelu_tanh_mul_fp4_quant(nvfp4_gemm(.)) with the pins.
Inputs are checked unchanged after each case.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nvfp4_dense"))
import bench  # noqa: E402
import census  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402  (WARMUP, ITERS, flush from bench_pair.py)

log = bench.log
WARMUP, ITERS = protocol.WARMUP, protocol.ITERS
_FLUSH = []


def flush_buf():
    if not _FLUSH:
        _FLUSH.append(protocol.make_flush())
    return _FLUSH[0]
GU_M = [300, 1000, 1200, 1500, 1800, 2000, 2300, 4000, 4800]
DN_M = [300, 1000, 1200, 1500, 1800, 2000, 2300, 4000, 4800]
LAYERS = (0, 30, 59)


def tco():
    return bench.tco()


def set_pins(on):
    t = tco()
    t.NVFP4_GATEUP_PIN = on
    t.NVFP4_DOWN_PIN = on


def ulp_diff(a, b):
    """max |ordered-int16(a) - ordered-int16(b)| for BF16 tensors."""
    def ordered(x):
        i = x.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)
    return int((ordered(a) - ordered(b)).abs().max().item())


def diff(a, b):
    nd = int((a.view(torch.int16) != b.view(torch.int16)).sum().item())
    if nd == 0:
        return {"n_diff": 0}
    return {"n_diff": nd, "max_abs": float((a.float() - b.float()).abs().max().item()), "max_ulp": ulp_diff(a, b)}


def tuned_explicit(inputs, n, k, m):
    """Run main's code path explicitly: the tuner-cache pick for this M (dense key in range)."""
    t = tco()
    from tensorrt_llm._torch.autotuner import AutoTuner
    tuner = AutoTuner.get()
    bl = bench.BACKENDS.split(",")
    shapes = tuple(x.size() for x in inputs)
    rng = t._nvfp4_dense_range(n, k)
    if rng is not None and rng[0] <= m <= rng[1]:
        dr = t.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl, dense_min_m=rng[0])
        hit, _, tac, _ = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm_dense", [dr], shapes,
                                                            t._nvfp4_dense_tuning_config(*rng))
        if hit and tac != -1:
            return dr.forward(inputs, tactic=tac), f"dense:{tac}"
    r = t.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bl)
    hit, _, tac, _ = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm", [r], shapes,
                                                        t.NVFP4GemmUnifiedRunner.tuning_config)
    return r.forward(inputs, tactic=tac), f"gemm:{tac}"


def all_tactics(xq, xsf, wq, wsf, alpha, m, n, k):
    t = tco()
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
    fp4 = torch.classes.trtllm.FP4GemmRunner(torch.bfloat16, int(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4))
    lt = torch.classes.trtllm.CublasLtFP4GemmRunner(torch.bfloat16)
    rng = t._nvfp4_dense_range(n, k)
    dense = rng is not None and rng[0] <= m <= rng[1]
    out = []
    for c in range(fp4.get_num_configs()):
        out.append((f"cutlass{c}", lambda c=c: fp4.run_gemm(xq, wq, xsf, wsf, alpha, 0, c, None, None),
                    {"main_offered": True}))
    for i in range(lt.get_num_heuristic_algos(xq, wq, xsf, wsf)):
        cfg = list(lt.get_algo_config(xq, wq, xsf, wsf, i, -1))
        out.append((f"lt{i}", lambda i=i: lt.run_gemm(xq, wq, xsf, wsf, alpha, 0, i, None, None),
                    {"cfg": cfg, "main_offered": not dense}))
    if dense:
        b = t.nvfp4_dense_bucket(m, rng[0])
        hx = xq.new_empty((b, xq.shape[1]))
        for i in range(lt.get_num_heuristic_algos(hx, wq, xsf, wsf)):
            cfg = list(lt.get_algo_config(xq, wq, xsf, wsf, i, b))
            if cfg[1]:
                continue
            out.append((f"lt_b{b}_{i}", lambda i=i: lt.run_gemm_bucketed(xq, wq, xsf, wsf, alpha, 0, i, b, None,
                                                                          None),
                        {"cfg": cfg, "main_offered": cfg[5] <= 1 and cfg[6] == 0}))
    return out


def slice_rows(xq, xsf, m, k):
    sf_rows = ((k // 16 + 3) // 4) * 4 * 128
    tiles = (m + 127) // 128
    return xq[:m].contiguous(), xsf.reshape(-1)[: tiles * sf_rows].contiguous()


class Case:
    def __init__(self, tag, name, m, xq, xsf, alpha, wq, wsf, xgs=None, wgs=None):
        self.tag, self.name, self.m = tag, name, m
        self.xq, self.xsf, self.alpha, self.wq, self.wsf = xq, xsf, alpha, wq, wsf
        self.xgs, self.wgs = xgs, wgs


def run_case(cs, dims, wref_cache, first_out=None):
    n, k = dims[cs.name]
    m = cs.m
    inputs = [cs.xq, cs.wq, cs.xsf, cs.wsf, cs.alpha]
    snap = [x.clone() for x in inputs]
    t = tco()
    set_pins(True)
    # The timed sequence of bench_pair.py: WARMUP + ITERS calls, 256 MiB flush before each.
    pin, calls, per_call, _ = protocol.seq(lambda: bench.gemm(cs.xq, cs.xsf, cs.wq, cs.wsf, cs.alpha), flush_buf())
    rep_bad = int(sum(per_call))
    pin_info = t.nvfp4_pin_info(n, k, m)
    set_pins(False)
    off = bench.gemm(cs.xq, cs.xsf, cs.wq, cs.wsf, cs.alpha)
    explicit, tuned_tac = tuned_explicit(inputs, n, k, m)
    set_pins(True)
    torch.cuda.synchronize()
    row = {"case": cs.tag, "shape": cs.name, "M": m, "pin": pin_info, "tuned": tuned_tac,
           "calls": calls, "repeat_mismatch": rep_bad,
           "repeat_mismatch_calls": [i + 1 for i, v in enumerate(per_call) if v], "pin_vs_tuned": diff(pin, off),
           "killswitch_off_vs_main_path": diff(off, explicit)}
    if first_out is not None:
        row["A_prime_vs_A"] = diff(pin, first_out)
    tacs = {}
    for label, fn, meta in all_tactics(cs.xq, cs.xsf, cs.wq, cs.wsf, cs.alpha, m, n, k):
        try:
            o = fn()
            torch.cuda.synchronize()
            tacs[label] = dict(diff(o, pin), **meta)
            del o
        except Exception as e:  # noqa: BLE001
            tacs[label] = {"error": repr(e)[:160], **meta}
    row["vs_tactics"] = tacs
    # FP32 reference (the dequantized operands).
    if cs.name not in wref_cache:
        wgs = cs.wgs if cs.wgs is not None else torch.ones(1, device="cuda")
        wref_cache[cs.name] = bench.dequant(cs.wq, cs.wsf, wgs, n, k)
    xgs = cs.xgs if cs.xgs is not None else torch.ones(1, device="cuda")
    # alpha folds both global scales: ref = alpha * (xdq*xgs) @ (wdq*wgs)^T
    wgs = cs.wgs if cs.wgs is not None else torch.ones(1, device="cuda")
    ref = (bench.dequant(cs.xq, cs.xsf, xgs, m, k) * xgs) @ (wref_cache[cs.name] * wgs).t() * cs.alpha.float()
    ec, em = pin.float() - ref, off.float() - ref
    ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
    tol = {"pin_maxabs": ec.abs().max().item(), "tuned_maxabs": em.abs().max().item(),
           "pin_relfro": (ec.norm() / ref.norm()).item(), "tuned_relfro": (em.norm() / ref.norm()).item()}
    tol["ok"] = (tol["pin_maxabs"] <= 1.10 * tol["tuned_maxabs"] + ulp
                 and tol["pin_relfro"] <= 1.10 * tol["tuned_relfro"] + 1e-6)
    row["ref"] = tol
    del ref, ec, em
    # chunked chain
    if cs.name == "gate_up" and m >= 1664:
        from tensorrt_llm._torch.kernels.fused_ops import gate_up_gelu_fp4_chunked as gc
        from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
        dis = torch.tensor([0.25], dtype=torch.float32, device="cuda")
        ch = gc.gate_up_gelu_fp4_chunked(cs.xq, cs.xsf, cs.wq, cs.wsf, cs.alpha, torch.bfloat16,
                                         bench.BACKENDS.split(","), dis, 1280)
        if ch is None:
            row["chunk"] = {"engaged": False}
        else:
            fq, fsf = gelu_tanh_mul_fp4_quant(pin, dis)
            row["chunk"] = {"engaged": True, "fp4_equal": bool(torch.equal(ch[0].view(torch.uint8),
                                                                           fq.view(torch.uint8))),
                            "sf_equal": bool(torch.equal(ch[1].view(torch.uint8).reshape(-1),
                                                         fsf.view(torch.uint8).reshape(-1)))}
    row["inputs_unchanged"] = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) if a.dtype != torch.float32
                                  else torch.equal(a, b) for a, b in zip(snap, inputs))
    row["finite"] = bool(torch.isfinite(pin.float()).all())
    return row, pin


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile
    import tempfile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    p = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(p)
    tr = json.load(open(p))
    os.remove(p)
    return [(e["name"][:110], e.get("args", {}).get("grid")) for e in tr["traceEvents"] if e.get("cat") == "kernel"]


def pin_report(name, m, n, k, xq, xsf, wq, wsf, alpha):
    """Heuristic list at M (index -> config), the index the pin resolves to, and the kernel that
    trtllm::nvfp4_gemm runs at M with the pins on and off."""
    t = tco()
    lt = torch.classes.trtllm.CublasLtFP4GemmRunner(torch.bfloat16)
    lst = [list(lt.get_algo_config(xq, wq, xsf, wsf, i, -1)) for i in range(lt.get_num_heuristic_algos(xq, wq, xsf,
                                                                                                         wsf))]
    inputs = [xq, wq, xsf, wsf, alpha]
    set_pins(True)
    ident = t._nvfp4_pin_entry("gemm", n, k, m)
    resolved = t.nvfp4_pinned_tactic("gemm", inputs, torch.bfloat16, -1, bench.BACKENDS.split(","))
    k_on = kernels_of(lambda: bench.gemm(xq, xsf, wq, wsf, alpha))
    set_pins(False)
    k_off = kernels_of(lambda: bench.gemm(xq, xsf, wq, wsf, alpha))
    set_pins(True)
    r = {"shape": name, "M": m, "pin": str(ident), "resolved": str(resolved), "list_at_M": lst,
         "kernels_pin_on": k_on, "kernels_pin_off": k_off}
    log(f"[resolve] {name}@{m} pin={ident} -> {resolved}; list={lst}; on={k_on}; off={k_off}")
    return r


def forced_miss(xq, xsf, wq, wsf, alpha, n, k):
    """gate_up M=300 with the gemm_dense lookup forced to miss (-1): the pin must still run nvjet
    256x128x128 bitwise equal to the dense hit; with the gate_up pin off the miss must run main's
    'gemm' bucket-256 pick (main's code path), bitwise equal to that pick run explicitly."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    m = xq.shape[0]
    set_pins(True)
    hit = bench.gemm(xq, xsf, wq, wsf, alpha)
    k_hit = kernels_of(lambda: bench.gemm(xq, xsf, wq, wsf, alpha))
    tuner = AutoTuner.get()
    orig = tuner.choose_one

    def choose(custom_op, runners, tuning_config, inputs, **kw):
        if custom_op == "trtllm::nvfp4_gemm::gemm_dense" and inputs[0].shape[0] == m:
            return runners[0], -1
        return orig(custom_op, runners, tuning_config, inputs, **kw)

    tuner.choose_one = choose
    try:
        miss_on = bench.gemm(xq, xsf, wq, wsf, alpha)
        k_miss_on = kernels_of(lambda: bench.gemm(xq, xsf, wq, wsf, alpha))
        set_pins(False)
        miss_off = bench.gemm(xq, xsf, wq, wsf, alpha)
        k_miss_off = kernels_of(lambda: bench.gemm(xq, xsf, wq, wsf, alpha))
    finally:
        tuner.choose_one = orig
        set_pins(True)
    t = tco()
    r = t.NVFP4GemmUnifiedRunner(0, torch.bfloat16, bench.BACKENDS.split(","))
    shapes = tuple(x.size() for x in (xq, wq, xsf, wsf, alpha))
    _, _, tac, _ = tuner.profiling_cache.search_cache("trtllm::nvfp4_gemm::gemm", [r], shapes,
                                                      t.NVFP4GemmUnifiedRunner.tuning_config)
    main_fallback = r.forward([xq, wq, xsf, wsf, alpha], tactic=tac)
    torch.cuda.synchronize()
    res = {"M": m, "dense_hit_kernels": k_hit, "miss_pin_on_kernels": k_miss_on, "miss_pin_off_kernels": k_miss_off,
           "miss_pin_on_vs_hit": diff(miss_on, hit), "miss_pin_off_vs_main_gemm_pick": diff(miss_off, main_fallback),
           "main_gemm_pick": str(tac), "miss_pin_off_vs_hit": diff(miss_off, hit)}
    res["ok"] = (any("256x128x128" in a for a, _ in k_miss_on) and res["miss_pin_on_vs_hit"]["n_diff"] == 0
                 and res["miss_pin_off_vs_main_gemm_pick"]["n_diff"] == 0)
    log(f"[forced-miss] {json.dumps(res)}")
    return res


def verdict(row):
    ok = (row["repeat_mismatch"] == 0 and row["calls"] == protocol.CALLS
          and row["killswitch_off_vs_main_path"]["n_diff"] == 0
          and row["ref"]["ok"] and row["inputs_unchanged"] and row["finite"]
          and row.get("A_prime_vs_A", {"n_diff": 0})["n_diff"] == 0)
    if "chunk" in row:
        ok = ok and row["chunk"].get("engaged") and row["chunk"]["fp4_equal"] and row["chunk"]["sf_equal"]
    if all(v is None for v in row["pin"].values()):
        # Outside the pinned ranges the pinned build runs main's tuned pick.
        ok = ok and row["pin_vs_tuned"]["n_diff"] == 0
    if row["shape"] == "gate_up":
        # gate_up pins must be int16-equal to every order-preserving tactic main can pick.
        bad = [lab for lab, x in row["vs_tactics"].items() if x.get("main_offered") and x.get("n_diff", 1) != 0
               and not _is_reduction(x)]
        row["gate_up_nonequal_main_tactics"] = bad
        ok = ok and not bad and row["pin_vs_tuned"]["n_diff"] == 0
    return ok


def _is_reduction(x):
    cfg = x.get("cfg")
    return cfg is not None and (cfg[5] > 1 or cfg[5] == -2 or cfg[6] != 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--captures", required=True)
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    dims = bench.read_dims(args.checkpoint)
    t = tco()
    log(f"[pins] table={t._NVFP4_PINS} gate_up_on={t.NVFP4_GATEUP_PIN} down_on={t.NVFP4_DOWN_PIN}")
    log(protocol.protocol_line())
    d = args.captures
    man, bad = census.load_manifest(d)
    by = {}
    for e in man["entries"]:
        by.setdefault((e["layer"], e["shape"]), []).append(e)
    w0 = census.load_weights(d, man, 0)
    alphas0 = {e["shape"]: e["alpha"] for e in man["entries"] if e["layer"] == 0}
    wts = {s: (w0[s][0], w0[s][1], torch.tensor([alphas0[s]], dtype=torch.float32, device="cuda")) for s in w0}
    tune_s = census.tune_real(wts, dims)
    log(f"[tune] {tune_s:.1f}s")
    rows, fails = [], []
    # Pin resolution per pinned served M, and the forced gemm_dense miss at gate_up M=300 (real L0).
    extra = {"resolve": [], "forced_miss": None}
    for name, m in (("gate_up", 300), ("down", 1200), ("down", 1500), ("down", 1800), ("down", 2000)):
        ents = sorted([e for e in by.get((0, name), []) if e["M"] >= m], key=lambda e: e["M"])
        if not ents:
            continue
        tt = torch.load(os.path.join(d, ents[0]["file"]))
        n, k = dims[name]
        xq, xsf = slice_rows(tt["act_fp4"].cuda(), tt["act_sf"].cuda(), m, k)
        alpha = tt["alpha"].cuda().float().reshape(-1)[:1]
        extra["resolve"].append(pin_report(name, m, n, k, xq, xsf, w0[name][0], w0[name][1], alpha))
        if name == "gate_up":
            extra["forced_miss"] = forced_miss(xq, xsf, w0[name][0], w0[name][1], alpha, n, k)
            if not extra["forced_miss"]["ok"]:
                fails.append("forced_miss")
        del tt
    for r in extra["resolve"]:
        if r["pin"] != "None" and not str(r["resolved"]).startswith("('"):
            fails.append(f"resolve {r['shape']}@{r['M']}")
    for layer in LAYERS:
        w = census.load_weights(d, man, layer)
        srcs = {}
        for name in ("gate_up", "down"):
            ents = sorted(by.get((layer, name), []), key=lambda e: e["M"])
            srcs[name] = ents
        loaded = {}
        wref_layer = {}

        def src_for(name, m):
            for e in srcs[name]:
                if e["M"] >= m:
                    if e["file"] not in loaded:
                        tt = torch.load(os.path.join(d, e["file"]))
                        loaded[e["file"]] = (tt["act_fp4"].cuda(), tt["act_sf"].cuda(),
                                             tt["alpha"].cuda().float().reshape(-1)[:1])
                    return e, loaded[e["file"]]
            return None, None

        for m in sorted(set(GU_M) | set(DN_M)):
            first = None
            seq = [("gate_up", m), ("down", m), ("gate_up", m)] if m in GU_M and m in DN_M else \
                [(nm, m) for nm in ("gate_up", "down") if m in (GU_M if nm == "gate_up" else DN_M)]
            for j, (name, mm) in enumerate(seq):
                e, src = src_for(name, mm)
                if e is None:
                    log(f"[skip] L{layer} {name} M={mm}: no capture with >= {mm} rows")
                    continue
                n, k = dims[name]
                xq, xsf = slice_rows(src[0], src[1], mm, k)
                cs = Case(f"real|L{layer}|{name}|M{mm}|src={e['prompt']}:s{e['step']}:M{e['M']}", name, mm, xq, xsf,
                          src[2], w[name][0], w[name][1])
                row, out = run_case(cs, dims, wref_layer, first if (j == 2) else None)
                if j == 0:
                    first = out
                ok = verdict(row)
                row["ok"] = ok
                rows.append(row)
                if not ok:
                    fails.append(row["case"])
                log(f"[case] {'OK ' if ok else 'BAD'} {row['case']} pin={row['pin']} tuned={row['tuned']} "
                    f"pin_vs_tuned={row['pin_vs_tuned']} calls={row['calls']} rep={row['repeat_mismatch']} "
                    f"off_vs_main={row['killswitch_off_vs_main_path']} Aprime={row.get('A_prime_vs_A')} "
                    f"ref={ {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row['ref'].items()} } "
                    f"chunk={row.get('chunk')}")
                log("       vs_tactics: " + "; ".join(f"{lab}:{x.get('n_diff')}/{x.get('max_abs', 0):.3g}/"
                                                      f"{x.get('max_ulp', 0)}{'' if x.get('main_offered') else '(n/o)'}"
                                                      for lab, x in row["vs_tactics"].items()))
                del out
        del loaded, wref_layer
        torch.cuda.empty_cache()
    # synthetic, layer-0 weights
    wref = {}
    for variant, seed, heavy in (("seed1", 1, False), ("seed2", 2, False), ("heavy", 3, True)):
        for m in sorted(set(GU_M) | set(DN_M)):
            first = None
            for j, name in enumerate(("gate_up", "down", "gate_up")):
                n, k = dims[name]
                xq, xsf, xgs = bench.make_act(m, k, seed * 7919 + m, heavy)
                wq, wsf = w0[name]
                cs = Case(f"synthetic-{variant}|L0|{name}|M{m}", name, m, xq, xsf,
                          torch.tensor([alphas0[name]], dtype=torch.float32, device="cuda"), wq, wsf)
                row, out = run_case(cs, dims, wref, first if j == 2 else None)
                if j == 0:
                    first = out
                ok = verdict(row)
                row["ok"] = ok
                rows.append(row)
                if not ok:
                    fails.append(row["case"])
                log(f"[case] {'OK ' if ok else 'BAD'} {row['case']} tuned={row['tuned']} "
                    f"pin_vs_tuned={row['pin_vs_tuned']} calls={row['calls']} rep={row['repeat_mismatch']} "
                    f"off_vs_main={row['killswitch_off_vs_main_path']} chunk={row.get('chunk')} "
                    f"ref_ok={row['ref']['ok']}")
                del out
    # down pins: worst difference against each tactic main can pick, per served M (real rows)
    down_summary = {}
    for r in rows:
        if r["shape"] != "down" or not r["case"].startswith("real") or not (r["pin"].get("gemm")):
            continue
        s = down_summary.setdefault(r["M"], {})
        for lab, x in r["vs_tactics"].items():
            if not x.get("main_offered"):
                continue
            cur = s.setdefault(lab, {"n_diff_max": 0, "max_abs": 0.0, "max_ulp": 0, "cfg": x.get("cfg")})
            cur["n_diff_max"] = max(cur["n_diff_max"], x.get("n_diff", 0))
            cur["max_abs"] = max(cur["max_abs"], x.get("max_abs", 0.0))
            cur["max_ulp"] = max(cur["max_ulp"], x.get("max_ulp", 0))
    for m, s in sorted(down_summary.items()):
        log(f"[down-pin] M={m}: " + "; ".join(f"{lab} nd<={v['n_diff_max']} abs<={v['max_abs']:.3g} "
                                              f"ulp<={v['max_ulp']}" for lab, v in s.items()))
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, "pin_check.json"), "w") as f:
        json.dump({"rows": rows, "down_pin_vs_main_tactics": down_summary, "extra": extra}, f, indent=1)
    passed = not fails and not bad
    print(json.dumps({"passed": passed, "fails": fails[:30], "n_cases": len(rows), "sha_mismatch": bad,
                      "forced_miss": extra["forced_miss"],
                      "resolve": [{k: r[k] for k in ("shape", "M", "pin", "resolved", "kernels_pin_on",
                                                     "kernels_pin_off")} for r in extra["resolve"]],
                      "down_pin_vs_main_tactics": down_summary,
                      "protocol": protocol.protocol_line(),
                      "per_case_repeat": [[r["case"], r["calls"], r["repeat_mismatch"]] for r in rows],
                      "compared": "pinned trtllm::nvfp4_gemm vs every tactic main can pick (int16, max abs, max "
                                  f"ULP), repeated {protocol.CALLS} calls ({WARMUP}+{ITERS}, 256 MiB flush before "
                                  "each, from bench_pair.py) every call int16 vs call 1, A,B,A order, kill switches off vs main's "
                                  "path (tuner pick run explicitly), FP32 dequant reference (<=1.10x + 1 ulp), "
                                  "chunked chain byte-equal; real captures L0/30/59 sliced to served M + "
                                  "synthetic seeds 1/2/heavy"}))


if __name__ == "__main__":
    main()
