"""Idea 55 (denoiser_gemm, branch 48): t256 (256x128x128) instance of landing 7's fused LPT pair +
flag-wait residual (trtllm::flux3_sm120_nvfp4_out_pair_residual, csrc/flux3_fp4_lpt_pair_residual_
sm120.cu built with -DFLUX3_LPTR_T256), routed at 8192 <= M < 24576 only.

Shapes (FLUX.3 klein 2048^2, T_pad=80): down [3072 x 9216] K=9216 + attn_out [3072 x 3072], NVFP4
W4A4 128x4 scales; attn_out activation pre-quantized (COND_ATTN_NFO=1) except the nfo0 row (BF16
attn, quantized inside the op). Gate forms: 16464 select ([1,2,D] + bool), 16384 row ([1,1,D]),
16384 sel ([1,2,D] + bool, t2i block 27), 16384 u8s ([1,3,D] + u8 index sliced from a 32848 one,
i2i block 27), 32848 u8x3 and 4176 select (128 instance on both trees). Synthetic, fixed seeds.

modes:
  bench : run_gpu_pair arm. CUDA graphs: one denoise step of each case's fused tails (post-L18
          counts: t2i 27 x 16464 select + 4 x 16384 row + 1 x 16384 sel; i2i 9 x 16384 row +
          1 x 16384 u8s) and 20-call per-M graphs; seeded random graph order per sample.
  table : branch only. 20-call graphs per M: fused-128 at main's route (AlongN) and its A/A copy,
          t256 at raster 2 / 0, t256 prefetch modes; seeded random order; median paired diffs.
  check : branch only, untimed, bitwise (see check()).
"""

import gc
import json
import os
import random
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

H, FF = 3072, 9216
SETS = 3
WARMUP = 3
ITERS = 20


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make_case(M, seed, form, nfo0=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_attn = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    x_down = torch.randn(M, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    s_attn, s_down_in, s_wout, s_wdown = map(_gs, (x_attn, x_down, w_out, w_down))
    a_down, sf_down_in = fp.nvfp4_quantize(x_down, _inv(s_down_in))
    q_out, sf_out = fp.nvfp4_quantize(w_out, _inv(s_wout))
    q_down, sf_down = fp.nvfp4_quantize(w_down, _inv(s_wdown))
    out_ad = PreparedNVFP4(q_out, sf_out, s_attn, s_wout)
    down_ad = PreparedNVFP4(q_down, sf_down, s_down_in, s_wdown)
    attn_q, attn_sf = fp.nvfp4_quantize_prepared(x_attn, out_ad.weight)
    del x_down
    x = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, generator=g)
    if form in ("select", "sel"):  # [1, 2, D] table + bool [1, S, 1]
        gate = torch.randn(1, 2, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        first = 16384 if M > 16384 else M - 80
        index = (torch.arange(M, device=dev) >= first).view(1, M, 1)
    elif form == "row":  # [1, 1, D]
        gate = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        index = None
    elif form == "u8x3":  # [1, 3, D] table + uint8 row index
        gate = torch.randn(1, 3, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        index = torch.randint(0, 3, (1, M, 1), device=dev, dtype=torch.uint8, generator=g)
    elif form == "u8s":  # i2i block 27: [1, 3, D] + u8 index sliced from the 32848-row one
        gate = torch.randn(1, 3, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        full = torch.randint(0, 3, (1, 32848, 1), device=dev, dtype=torch.uint8, generator=g)
        index = full[:, :M]
    else:
        raise ValueError(form)
    c = {
        "M": M, "form": form, "down_in": a_down, "down_sf": sf_down_in, "attn_q": attn_q,
        "attn_sf": attn_sf, "out_ad": out_ad, "down_ad": down_ad, "x": x, "gate": gate,
        "index": index, "nfo0": nfo0,
    }
    if nfo0:
        c["attn_bf16"] = x_attn
    else:
        del x_attn
    return c


def inputs(c):
    t = [c["down_in"], c["down_sf"], c["attn_q"], c["attn_sf"], c["out_ad"].weight,
         c["out_ad"].scale, c["down_ad"].weight, c["down_ad"].scale, c["x"], c["gate"]]
    if c["index"] is not None:
        t.append(c["index"])
    if c["nfo0"]:
        t.append(c["attn_bf16"])
    return t


def run_pair(c):
    return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        c["attn_q"], c["attn_sf"], c["out_ad"].weight, c["out_ad"].scale)


def _res_select(x, table, select, attn_y, mlp_y):
    gate = torch.where(select, table[:, 1:2], table[:, 0:1])
    output = attn_y.reshape(*x.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
    return x + gate * output


def _res_row(x, gate, attn_y, mlp_y):
    output = attn_y.reshape(*x.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
    return x + gate * output


_OPTS = {"emulate_precision_casts": True}
res_select = torch.compile(_res_select, dynamic=False, options=_OPTS)
res_row = torch.compile(_res_row, dynamic=False, options=_OPTS)


def run_base(c):
    """Pair + Inductor residual (the per-token gate for table + u8 forms)."""
    attn_y, mlp_y = run_pair(c)
    if c["form"] in ("select", "sel"):
        return res_select(c["x"], c["gate"], c["index"], attn_y, mlp_y)
    if c["form"] == "row":
        return res_row(c["x"], c["gate"], attn_y, mlp_y)
    gate_tok = c["gate"][0][c["index"].reshape(-1).long()].unsqueeze(0).contiguous()
    return res_row(c["x"], gate_tok, attn_y, mlp_y)


def run_fused(c):
    if c["nfo0"]:
        attn, sc = c["attn_bf16"], None
    else:
        attn, sc = c["attn_q"], c["attn_sf"]
    return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_residual(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        attn, sc, c["out_ad"].weight, c["out_ad"].scale, c["x"], c["gate"], c["index"])


def lptr_mod():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr
    return lptr


def run_forced(c, inst, raster=None):
    """The op with its instance (and optionally raster) forced: monkeypatch _launcher/raster_for."""
    lptr = lptr_mod()
    saved_l, saved_r = lptr._launcher, lptr.raster_for
    fn = lptr.t256_module().lpt_pair_residual_t256 if inst == "t256" else \
        lptr.module().lpt_pair_residual
    lptr._launcher = lambda rows: fn
    if raster is not None:
        lptr.raster_for = lambda rows: raster
    try:
        return run_fused(c)
    finally:
        lptr._launcher, lptr.raster_for = saved_l, saved_r


def emulate_ref(c, attn_y, mlp_y):
    x = c["x"].reshape(-1, H).float()
    if c["index"] is None:
        g = c["gate"].reshape(1, H).float()
    else:
        g = c["gate"].reshape(-1, H)[c["index"].reshape(-1).long()].float()
    o = (attn_y.float() + mlp_y.float()).bfloat16().float()
    p = (g * o).bfloat16().float()
    return (x + p).bfloat16().reshape(c["x"].shape)


def stat(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2], v[n // 10], v[(n * 9) // 10]


def capture(fn_seq, stream):
    with torch.cuda.stream(stream):
        for f in fn_seq:
            f()
        torch.cuda.synchronize()
        gc.collect()
        gc.disable()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                outs = [f() for f in fn_seq]
        finally:
            gc.enable()
    torch.cuda.synchronize()
    return graph, outs


def time_graphs(graphs, samples, warm, calls, seed=55):
    """Seeded random graph order per sample (balanced predecessors). calls: per-graph divisor."""
    names = list(graphs)
    rng = random.Random(seed)
    for _ in range(warm):
        for n in names:
            graphs[n].replay()
    torch.cuda.synchronize()
    out = {n: [] for n in names}
    for i in range(samples):
        order = names[:]
        rng.shuffle(order)
        evs = []
        for n in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            evs.append((n, a, b))
        torch.cuda.synchronize()
        for n, a, b in evs:
            out[n].append(a.elapsed_time(b) * 1e3 / calls.get(n, 1))
        if i % 20 == 0:
            log(f"  sample {i}: " + " ".join(f"{n}={out[n][-1]:.1f}us" for n in names))
    return out


class _Sentinel:
    lpt_pair_residual_t256 = "t256"
    lpt_pair_residual = "128"


def route_of(lptr, M):
    """Which instance _launcher(M) picks: '128' or 't256' (module builders patched to sentinels)."""
    saved = lptr.module, lptr.t256_module
    lptr.module = lptr.t256_module = lambda: _Sentinel
    try:
        return lptr._launcher(M)
    finally:
        lptr.module, lptr.t256_module = saved


def preload():
    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    lptr = lptr_mod()
    lptr.preload(dev)
    has = hasattr(lptr, "_launcher")
    msg = f"tree: {'branch' if has else 'base'} (enabled={lptr.enabled()}, " \
          f"info128={lptr.module().lpt_pair_residual_info()}"
    if has:
        m = lptr.t256_module()
        msg += (f", info256={m.lpt_pair_residual_t256_info()}, "
                f"attrs128={m.lpt_pair_residual_func_attrs(0)} "
                f"attrs256={m.lpt_pair_residual_func_attrs(1)} (regs*1e6+local B)")
        msg += ", route " + " ".join(
            f"{M}->{route_of(lptr, M)}" for M in (4176, 8191, 8192, 16384, 16464, 24575, 24576,
                                                  32848))
    log(msg + ")")
    return lptr


# Post-L18 per-step calls of this op (plan 771(2)): t2i 432/16 = 27 at 16464, 80/16 = 5 at 16384;
# i2i 160/16 = 10 at 16384 (block 27 in the [1,3,D] + sliced u8 form).
CASES = {
    "t2i-2048": [(16464, "select", 27), (16384, "row", 4), (16384, "sel", 1)],
    "i2i-2048": [(16384, "row", 9), (16384, "u8s", 1)],
}
PER_M = [(16464, "select"), (16384, "row"), (16384, "u8s")]


def bench():
    preload()
    try:
        log(subprocess.run(["git", "log", "-3", "--oneline"], capture_output=True, text=True,
                           timeout=20).stdout)
    except Exception as e:  # noqa: BLE001
        log(f"git log unavailable: {e}")
    pool = {}
    for parts in CASES.values():
        for M, form, _ in parts:
            if (M, form) not in pool:
                pool[(M, form)] = [make_case(M, 1000 * M + 17 * i + len(form), form)
                                   for i in range(SETS)]
    pristine = {k: [[t.clone() for t in inputs(c)] for c in cs] for k, cs in pool.items()}
    graphs, calls = {}, {}
    s = torch.cuda.Stream()
    for case, parts in CASES.items():
        seq = []
        for M, form, n in parts:
            cs = pool[(M, form)]
            seq += [lambda c=cs[i % SETS]: run_fused(c) for i in range(n)]
        graphs[case], _ = capture(seq, s)
        calls[case] = 1
    for M, form in PER_M:
        cs = pool[(M, form)]
        key = f"M{M}_{form}"
        graphs[key], _ = capture([lambda c=cs[i % SETS]: run_fused(c) for i in range(20)], s)
        calls[key] = 20
    res = time_graphs(graphs, samples=100, warm=10, calls=calls)
    detail = {}
    for k, v in res.items():
        med, p10, p90 = stat(v)
        detail[k] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                     "noise_floor": round((p90 - p10) / med, 4)}
        log(f"{k}: median {med:.2f} us p10 {p10:.2f} p90 {p90:.2f}")
    for k, cs in pool.items():
        for p, c in zip(pristine[k], cs):
            assert all(torch.equal(x, y) for x, y in zip(p, inputs(c))), "inputs changed"
    print(json.dumps({
        "batch": {k: {"kernel_ms": detail[k]["us"] / 1e3, "noise_floor": detail[k]["noise_floor"]}
                  for k in CASES},
        "detail_us": detail,
        "unit": "case keys: one CUDA graph replay = one denoise step of the case's fused "
                "pair+residual tails (post-L18 counts); M* keys: us per call (20-call graphs)",
        "output": {"class": "exact"},
    }))


def table():
    lptr = preload()
    calls = 20
    rows = {}
    shapes = PER_M
    if os.environ.get("IDEA55_SHAPES"):
        shapes = [(int(s.split(":")[0]), s.split(":")[1])
                  for s in os.environ["IDEA55_SHAPES"].split(",")]
    for M, form in shapes:
        cs = [make_case(M, 7 * M + i + len(form), form) for i in range(SETS)]
        s = torch.cuda.Stream()

        def seq(f):
            return [lambda c=cs[i % SETS]: f(c) for i in range(calls)]

        graphs = {}
        # f128 at main's route (raster_for: 2 at 8192 <= M < 24576); at M >= 24576 forced to raster
        # 2 (branch 42's fused-128 AlongN route; timing only, not routed here).
        r128 = 2 if M >= 24576 else None
        graphs["f128_main"], _ = capture(seq(lambda c: run_forced(c, "128", r128)), s)
        graphs["f128_main_AA"], _ = capture(seq(lambda c: run_forced(c, "128", r128)), s)
        graphs["t256_r2"], _ = capture(seq(lambda c: run_forced(c, "t256", 2)), s)
        if os.environ.get("IDEA55_R0", "0") == "1":
            graphs["t256_r0"], _ = capture(seq(lambda c: run_forced(c, "t256", 0)), s)
        for pf in os.environ.get("IDEA55_PF", "1,2,4").split(","):
            os.environ["FLUX3_LPTR_T256_PF"] = pf
            graphs[f"t256_r2_pf{pf}"], _ = capture(seq(lambda c: run_forced(c, "t256", 2)), s)
            del os.environ["FLUX3_LPTR_T256_PF"]
        graphs["routed"], _ = capture(seq(run_fused), s)
        res = time_graphs(graphs, samples=100, warm=10, calls={k: calls for k in graphs})
        row = {}
        base = res["f128_main"]
        for n, v in res.items():
            med, p10, p90 = stat(v)
            diffs = sorted(b - a for a, b in zip(base, v))
            row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                      "med_paired_diff_vs_f128_us": round(diffs[len(diffs) // 2], 2),
                      "ratio": round(med / stat(base)[0], 4)}
        rows[f"M{M}_{form}"] = row
        log(f"M={M} {form}: {json.dumps(row)}")
        del graphs, cs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"table_us_per_call": rows, "calls_per_replay": calls, "samples": 60}))


def check():
    lptr = preload()
    dev = torch.device("cuda", torch.cuda.current_device())
    ws = lptr.workspace(dev)
    epoch_word = 128 // 4

    def epoch():
        return int(ws[:256].view(torch.int32)[epoch_word].item())

    results = {}
    passed = True
    # Routing table.
    probe = (1, 4176, 8191, 8192, 16384, 16464, 24575, 24576, 32848)
    want = {M: (8192 <= M < 24576) for M in probe}
    got = {M: route_of(lptr, M) == "t256" for M in probe}
    route_ok = got == want
    passed &= route_ok
    results["route_t256"] = {str(M): got[M] for M in probe}
    log(f"route t256: {got} ok {route_ok}")

    configs = [(16464, "select", False), (16384, "row", False), (16384, "row", True),
               (16384, "sel", False), (16384, "u8s", False), (4176, "select", False),
               (32848, "u8x3", False)]
    pool = {}
    for M, form, nfo0 in configs:
        c = make_case(M, 100 + M + len(form) + int(nfo0), form, nfo0)
        key = f"M{M}_{form}" + ("_nfo0" if nfo0 else "")
        pool[key] = c
        pristine = [t.clone() for t in inputs(c)]
        attn_y, mlp_y = run_pair(c)
        ref = run_base(c)
        emu = emulate_ref(c, attn_y, mlp_y)
        ref_vs_emu = bool(torch.equal(ref, emu))
        r128 = run_forced(c, "128").reshape(c["x"].shape).clone()  # main's route: 128 instance
        ok_all = ref_vs_emu and bool(torch.equal(r128, ref))
        results[key] = {"pair_inductor_vs_f32_emulation": ref_vs_emu,
                        "f128_vs_pair_inductor": bool(torch.equal(r128, ref))}
        arms = [("routed", lambda: run_fused(c)),
                ("t256_r2", lambda: run_forced(c, "t256", 2)),
                ("t256_r0", lambda: run_forced(c, "t256", 0)),
                ("f128", lambda: run_forced(c, "128"))]
        for arm, fn in arms:
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            n_ref = n_128 = 0
            epochs = []
            for i in range(ITERS):
                e0 = epoch()
                out = fn().reshape(c["x"].shape)
                torch.cuda.synchronize()
                epochs.append(epoch() - e0)
                n_ref += bool(torch.equal(out, ref))
                n_128 += bool(torch.equal(out, r128))
                out.fill_(float("nan"))
                del out
            ok = n_ref == ITERS and n_128 == ITERS and set(epochs) == {1}
            ok_all &= ok
            results[key][arm] = {"bitwise_vs_pair_inductor": f"{n_ref}/{ITERS}",
                                 "bitwise_vs_f128": f"{n_128}/{ITERS}",
                                 "epoch_step": sorted(set(epochs)), "ok": ok}
        # run_raw with a NaN-poisoned mlp_y scratch every call (a broken flag wait would read NaN).
        if not c["nfo0"]:
            down_alpha = torch.tensor([c["down_ad"].input_global_scale *
                                       c["down_ad"].weight_global_scale], device=dev,
                                      dtype=torch.float32) if hasattr(
                c["down_ad"], "input_global_scale") else None
            if down_alpha is not None:
                attn_alpha = torch.tensor([c["out_ad"].input_global_scale *
                                           c["out_ad"].weight_global_scale], device=dev,
                                          dtype=torch.float32)
                mlp_y_s = torch.empty((M, H), device=dev, dtype=torch.bfloat16)
                n_ok = 0
                for i in range(WARMUP + ITERS):
                    mlp_y_s.fill_(float("nan"))
                    out = lptr.run_raw(
                        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
                        down_alpha, c["attn_q"], c["out_ad"].weight, c["attn_sf"],
                        c["out_ad"].scale, attn_alpha, c["x"], c["gate"], c["index"],
                        mlp_y=mlp_y_s).reshape(c["x"].shape)
                    torch.cuda.synchronize()
                    n_ok += bool(torch.equal(out, ref))
                    del out
                ok = n_ok == WARMUP + ITERS
                ok_all &= ok
                results[key]["run_raw_nan_mlp_y"] = f"{n_ok}/{WARMUP + ITERS}"
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
        ok_all &= inputs_ok
        results[key]["inputs_unchanged"] = inputs_ok
        results[key]["ok"] = ok_all
        passed &= ok_all
        c["ref"] = ref
        c["r128"] = r128
        log(key, json.dumps(results[key]))
        del attn_y, mlp_y, emu
        torch.cuda.synchronize()

    # Kill switch: T256=0 -> the 128 instance at every M, byte for byte.
    os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256"] = "0"
    lptr.t256_enabled.cache_clear()
    kill_route = {M: route_of(lptr, M) == "128" for M in probe}
    kill_out_ok = True
    for key in ("M16464_select", "M16384_row", "M16384_u8s"):
        c = pool[key]
        out = run_fused(c).reshape(c["x"].shape)
        torch.cuda.synchronize()
        kill_out_ok &= bool(torch.equal(out, c["r128"]))
        del out
    del os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256"]
    lptr.t256_enabled.cache_clear()
    kill_ok = all(kill_route.values()) and kill_out_ok
    results["kill_switch"] = {"route_128_all": all(kill_route.values()), "outputs": kill_out_ok}
    passed &= kill_ok
    log(f"kill switch: {results['kill_switch']}")

    # Repeat sequence (untimed) over the shared counter/epoch/flag workspace (top-lead ruling
    # 01:45): the served graphs (on the synced tree the i2i graph runs t256 at 16384 and fused-128
    # r2 at 32848, landing 23) in t2i -> i2i -> i2i -> t2i ... order, one warmup round + ROUNDS
    # rounds, an eager M=4176 call (128 instance) after every graph. Before every round the
    # pristine inputs are copied back in place; before every replay the graph outputs are
    # NaN-poisoned. Every output is compared bitwise with round 0, with the kill-switch graphs
    # (captured with TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256=0, replayed once before round 0) and
    # with pair + Inductor. After every replay / eager call: epoch advanced by exactly the number
    # of launches, and the tile counter >= the last launch's tile count (every tile claimed).
    rounds = int(os.environ.get("IDEA55_ROUNDS", "50"))
    t2i_seq = ([pool["M16464_select"]] * 27 + [pool["M16384_row"]] * 4 + [pool["M16384_sel"]])
    i2i_seq = ([pool["M16384_row"]] * 4 + [pool["M16384_row_nfo0"]] * 5 + [pool["M16384_u8s"]]
               + [pool["M32848_u8x3"]] * 2)
    s = torch.cuda.Stream()
    g_t2i, o_t2i = capture([lambda c=c: run_fused(c) for c in t2i_seq], s)
    g_i2i, o_i2i = capture([lambda c=c: run_fused(c) for c in i2i_seq], s)
    os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256"] = "0"
    lptr.t256_enabled.cache_clear()
    g_t2i_k, o_t2i_k = capture([lambda c=c: run_fused(c) for c in t2i_seq], s)
    g_i2i_k, o_i2i_k = capture([lambda c=c: run_fused(c) for c in i2i_seq], s)
    del os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256"]
    lptr.t256_enabled.cache_clear()
    small = pool["M4176_select"]
    pristine_all = {k: [t.clone() for t in inputs(c)] for k, c in pool.items()}

    def counter():
        return int(ws[:4].view(torch.int32)[0].item())

    def tiles(M):
        bm = 256 if route_of(lptr, M) == "t256" else 128
        return 2 * ((M + bm - 1) // bm) * (H // 128)

    # Kill-switch outputs (128 instance at every M).
    kill = {}
    for name, g, outs in (("t2i", g_t2i_k, o_t2i_k), ("i2i", g_i2i_k, o_i2i_k)):
        for o in outs:
            o.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        kill[name] = [o.clone() for o in outs]
    kill_vs_ref = sum(not torch.equal(k.reshape(c["x"].shape), c["ref"])
                      for name, cs in (("t2i", t2i_seq), ("i2i", i2i_seq))
                      for k, c in zip(kill[name], cs))
    mism = {"vs_round0": 0, "vs_kill_switch": 0, "vs_pair_inductor": 0}
    state_bad = 0
    total = 0
    seq_log = []
    round0 = {}
    counters = {}
    for rnd in range(-1, rounds):  # -1 = warmup round (compared too, not the round-0 reference)
        for k, c in pool.items():
            for p, t in zip(pristine_all[k], inputs(c)):
                t.copy_(p)
        torch.cuda.synchronize()
        order = [("t2i", g_t2i, o_t2i, t2i_seq), ("i2i", g_i2i, o_i2i, i2i_seq)]
        if rnd % 2:
            order.reverse()
        bad_rnd = 0
        for name, g, outs, cs in order:
            for o in outs:
                o.fill_(float("nan"))
            e0 = epoch()
            g.replay()
            torch.cuda.synchronize()
            d_ep, cnt = epoch() - e0, counter()
            ok_state = d_ep == len(cs) and cnt >= tiles(cs[-1]["M"])
            counters.setdefault(name, set()).add(cnt)
            if rnd == 0:
                round0[name] = [o.clone() for o in outs]
            for i, (o, c) in enumerate(zip(outs, cs)):
                o = o.reshape(c["x"].shape)
                b_ref = not torch.equal(o, c["ref"])
                b_kill = not torch.equal(o, kill[name][i].reshape(c["x"].shape))
                b_r0 = rnd > 0 and not torch.equal(o, round0[name][i].reshape(c["x"].shape))
                mism["vs_pair_inductor"] += b_ref
                mism["vs_kill_switch"] += b_kill
                mism["vs_round0"] += b_r0
                bad_rnd += b_ref or b_kill or b_r0
            total += len(outs)
            e0 = epoch()
            o = run_fused(small).reshape(small["x"].shape)
            torch.cuda.synchronize()
            d_ep_s, cnt_s = epoch() - e0, counter()
            ok_state &= d_ep_s == 1 and cnt_s >= tiles(small["M"])
            counters.setdefault("eager4176", set()).add(cnt_s)
            b_s = not torch.equal(o, small["ref"])
            mism["vs_pair_inductor"] += b_s
            bad_rnd += b_s
            total += 1
            del o
            state_bad += not ok_state
            seq_log.append(f"{name}:{bad_rnd}{'' if ok_state else '!state'}")
        if rnd % 10 == 0:
            log(f"  repeat round {rnd}: mismatches {mism} state_bad {state_bad}")
    inputs_ok = all(all(torch.equal(p, t) for p, t in zip(pristine_all[k], inputs(c)))
                    for k, c in pool.items())
    rep_ok = (sum(mism.values()) == 0 and kill_vs_ref == 0 and state_bad == 0 and inputs_ok)
    results["repeat_sequence"] = {
        "rounds_after_warmup": rounds, "mismatches": mism, "kill_switch_vs_pair_inductor":
        kill_vs_ref, "compared": total, "state_checks_failed": state_bad,
        "tile_counter_values": {k: sorted(v) for k, v in counters.items()},
        "tile_counts_last_launch": {"t2i": tiles(t2i_seq[-1]["M"]),
                                    "i2i": tiles(i2i_seq[-1]["M"]),
                                    "eager4176": tiles(small["M"])},
        "inputs_unchanged": inputs_ok, "order": " ".join(seq_log)}
    passed &= rep_ok
    log(f"repeat sequence: {results['repeat_sequence']}")
    print(json.dumps({"passed": bool(passed), "compared_against": "pair (flux3_sm120_nvfp4_out_"
                      "pair_prequant) + Inductor emulate_precision_casts residual, f32 emulation, "
                      "and the 128 instance (main's route)", "results": results}))


def ncu():
    """For profile_kernels: 2 x fused-128 r2 then 2 x t256 r2 at M=16464 select (after warmup)."""
    preload()
    c = make_case(16464, 4242, "select")
    for _ in range(2):
        run_forced(c, "128")
        run_forced(c, "t256")
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for inst in ("128", "128", "t256", "t256"):
        run_forced(c, inst)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    log("ncu arm done")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    {"bench": bench, "table": table, "check": check, "ncu": ncu}[mode]()
