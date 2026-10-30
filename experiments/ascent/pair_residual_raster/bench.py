"""Idea 59 (denoiser_gemm, branch 21): raster order of landing 7's fused-128 pair + gated residual
(trtllm::flux3_sm120_nvfp4_out_pair_residual, csrc/flux3_fp4_lpt_pair_residual_sm120.cu) by M range.

Shapes (FLUX.3 klein 2048^2, served T_pad=80): t2i single blocks M=16464 ([1,2,D] table + bool
select), mode / image_cond blocks M=16384 ([1,1,D] row), i2i single blocks M=32848 ([1,3,D] table
+ uint8 index: branch 15's gate form, not yet routed on main). down [3072 x 9216] K=9216, attn_out
[3072 x 3072]; NVFP4 W4A4, 128x4 scales, attn_out's activation pre-quantized. Synthetic tensors,
fixed seeds, 3 input sets per M cycled. No model load.

  bench   : run_gpu_pair arm (both trees; uses only the op). One CUDA graph per case = one denoise
            step of the case's fused block tails:
              t2i-2048: 28 x (16464, select) + 4 x (16384, row)
              i2i-2048: 9 x (16384, row)  (4 image mode + 5 image_cond blocks; main's routes)
              i2i-2048-b15: 28 x (32848, u8x3) + 9 x (16384, row)  (branch 15's route)
            20 warm replays, 60 timed (CUDA events), median and (p90-p10)/median.
  table   : branch only. Interleaved 20-call graphs per M: fused raster 0 (main), 2 (AlongN),
            1 (AlongM), 0 again (A/A), routed; at 32848 also main's route (bare pair via
            flux3_sm120_lpt_pair.run + compiled per-token-gate residual) and its A/A.
  check   : branch only, untimed. Bitwise per M/form: fused raster 0/1/2 and the routed op vs
            main's pair + compiled residual and vs an f32 emulation, every call of WARMUP+ITERS
            with NaN-poisoned outputs, inputs unchanged, epoch +1 per call; resolved raster logged;
            kill switch (RASTER=0) routes raster 0 at every M; then the repeat sequence:
            t2i graph / i2i graph, 6 rounds alternating order, over the shared
            counter/epoch/flag workspace, an eager M=4176 call after every graph, outputs
            NaN-poisoned before every replay and compared bitwise with the references.
"""

import gc
import json
import os
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


def make_case(M, seed, form):
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
    del x_attn, x_down
    x = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, generator=g)
    if form == "select":  # single blocks t2i: [1, 2, D] table + bool [1, S, 1] (True on text rows)
        gate = torch.randn(1, 2, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        first = 16384 if M > 16384 else M - 80
        index = (torch.arange(M, device=dev) >= first).view(1, M, 1)
    elif form == "row":  # mode blocks: [1, 1, D]
        gate = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        index = None
    elif form == "u8x3":  # i2i single blocks (branch 15): [1, 3, D] table + uint8 row index
        gate = torch.randn(1, 3, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        index = torch.randint(0, 3, (1, M, 1), device=dev, dtype=torch.uint8, generator=g)
    else:
        raise ValueError(form)
    c = {
        "M": M, "form": form, "down_in": a_down, "down_sf": sf_down_in, "attn_q": attn_q,
        "attn_sf": attn_sf, "out_ad": out_ad, "down_ad": down_ad, "x": x, "gate": gate,
        "index": index,
    }
    if form == "u8x3":  # main's i2i form: the per-token gate [1, M, D] the 3-tuple mod carries
        c["gate_tok"] = gate[0][index.reshape(-1).long()].unsqueeze(0).contiguous()
    return c


def inputs(c):
    t = [c["down_in"], c["down_sf"], c["attn_q"], c["attn_sf"], c["out_ad"].weight,
         c["out_ad"].scale, c["down_ad"].weight, c["down_ad"].scale, c["x"], c["gate"]]
    if c["index"] is not None:
        t.append(c["index"])
    if "gate_tok" in c:
        t.append(c["gate_tok"])
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
    """Main's pair + Inductor residual (at u8x3: main's i2i route with the per-token gate)."""
    attn_y, mlp_y = run_pair(c)
    if c["form"] == "select":
        return res_select(c["x"], c["gate"], c["index"], attn_y, mlp_y)
    if c["form"] == "row":
        return res_row(c["x"], c["gate"], attn_y, mlp_y)
    return res_row(c["x"], c["gate_tok"], attn_y, mlp_y)


def run_fused(c):
    return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_residual(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        c["attn_q"], c["attn_sf"], c["out_ad"].weight, c["out_ad"].scale,
        c["x"], c["gate"], c["index"])


def run_fused_r(c, raster):
    """The op at a forced raster (patches the module's M-range route for this call)."""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr

    saved = lptr.raster_for
    lptr.raster_for = lambda rows: raster
    try:
        return run_fused(c)
    finally:
        lptr.raster_for = saved


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
    # thread_local capture + no GC inside: a capture-unsafe call from another thread or a
    # finalizer must not invalidate the capture (job 332 lost one at its 9th graph).
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


def time_graphs(graphs, samples, warm, calls):
    names = list(graphs)
    for _ in range(warm):
        for n in names:
            graphs[n].replay()
    torch.cuda.synchronize()
    out = {n: [] for n in names}
    for i in range(samples):
        order = names[i % len(names):] + names[:i % len(names)]
        evs = []
        for n in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            evs.append((n, a, b))
        torch.cuda.synchronize()
        for n, a, b in evs:
            out[n].append(a.elapsed_time(b) * 1e3 / calls)
        if i % 20 == 0:
            log(f"  sample {i}: " + " ".join(f"{n}={out[n][-1]:.1f}us" for n in names))
    return out


CASES = {
    "t2i-2048": [(16464, "select", 28), (16384, "row", 4)],
    "i2i-2048": [(16384, "row", 9)],
    "i2i-2048-b15": [(32848, "u8x3", 28), (16384, "row", 9)],
}


def preload():
    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr

    lptr.preload(dev)
    has = hasattr(lptr, "raster_for")
    msg = f"tree: {'branch' if has else 'base'} (residual enabled={lptr.enabled()}, " \
          f"info={lptr.module().lpt_pair_residual_info()}"
    if has:
        msg += (f", raster 16464->{lptr.raster_for(16464)} 16384->{lptr.raster_for(16384)} "
                f"32848->{lptr.raster_for(32848)} 4176->{lptr.raster_for(4176)}")
    log(msg + ")")
    return lptr


def bench():
    preload()
    pool = {}
    for parts in CASES.values():
        for M, form, _ in parts:
            if (M, form) not in pool:
                pool[(M, form)] = [make_case(M, 1000 * M + 17 * i + len(form), form)
                                   for i in range(SETS)]
    pristine = {k: [[t.clone() for t in inputs(c)] for c in cs] for k, cs in pool.items()}
    result = {}
    for case, parts in CASES.items():
        seq = []
        for M, form, calls in parts:
            cs = pool[(M, form)]
            seq += [lambda c=cs[i % SETS]: run_fused(c) for i in range(calls)]
        s = torch.cuda.Stream()
        g, _ = capture(seq, s)
        res = time_graphs({case: g}, samples=60, warm=20, calls=1)
        med, p10, p90 = stat(res[case])
        result[case] = {"kernel_ms": med / 1e3, "noise_floor": (p90 - p10) / med,
                        "p10_ms": p10 / 1e3, "p90_ms": p90 / 1e3, "calls": len(seq)}
        log(f"{case}: one step of {len(seq)} fused tails: median {med/1e3:.4f} ms "
            f"p10 {p10/1e3:.4f} p90 {p90/1e3:.4f}")
        del g
        torch.cuda.synchronize()
    for k, cs in pool.items():
        for p, c in zip(pristine[k], cs):
            assert all(torch.equal(x, y) for x, y in zip(p, inputs(c))), "inputs changed"
    print(json.dumps({
        "batch": {k: {"kernel_ms": v["kernel_ms"], "noise_floor": v["noise_floor"]}
                  for k, v in result.items()},
        "detail": result,
        "unit": "one CUDA graph replay = one denoise step of the case's fused pair+residual "
                "block tails; 3 input sets per M cycled",
        "output": {"class": "exact"},
    }))


def table():
    lptr = preload()
    calls = 20
    rows = {}
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for M, form in ((16464, "select"), (16384, "row"), (32848, "u8x3")):
        if only and str(M) not in only:
            continue
        cs = [make_case(M, 7 * M + i, form) for i in range(SETS)]
        s = torch.cuda.Stream()

        def seq(f):
            return [lambda c=cs[i % SETS]: f(c) for i in range(calls)]

        graphs = {}
        graphs["fused_r0"], _ = capture(seq(lambda c: run_fused_r(c, 0)), s)
        r0_res = int(lptr.module().lpt_pair_residual_last_raster())
        log(f"M={M}: raster 0 resolves to {r0_res} (1 AlongM, 2 AlongN)")
        graphs["fused_r2"], _ = capture(seq(lambda c: run_fused_r(c, 2)), s)
        graphs["fused_r1"], _ = capture(seq(lambda c: run_fused_r(c, 1)), s)
        graphs["fused_r0_AA"], _ = capture(seq(lambda c: run_fused_r(c, 0)), s)
        graphs["routed"], _ = capture(seq(run_fused), s)
        if M >= 24576:
            graphs["main_route"], _ = capture(seq(run_base), s)
            graphs["main_route_AA"], _ = capture(seq(run_base), s)
        res = time_graphs(graphs, samples=60, warm=10, calls=calls)
        row = {"raster0_resolves_to": r0_res, "routed_raster": lptr.raster_for(M)}
        for n, v in res.items():
            med, p10, p90 = stat(v)
            row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2)}
        r0 = row["fused_r0"]["us"]
        row["AA_delta_pct"] = round(100 * (row["fused_r0_AA"]["us"] - r0) / r0, 2)
        row["r2_vs_r0_pct"] = round(100 * (row["fused_r2"]["us"] - r0) / r0, 2)
        row["r1_vs_r0_pct"] = round(100 * (row["fused_r1"]["us"] - r0) / r0, 2)
        if "main_route" in row:
            mr = row["main_route"]["us"]
            row["main_AA_delta_pct"] = round(100 * (row["main_route_AA"]["us"] - mr) / mr, 2)
            row["r2_vs_main_route_pct"] = round(100 * (row["fused_r2"]["us"] - mr) / mr, 2)
            row["r0_vs_main_route_pct"] = round(100 * (r0 - mr) / mr, 2)
        rows[f"M{M}_{form}"] = row
        log(f"M={M} {form}: {json.dumps(row)}")
        del graphs, cs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"table_us_per_call": rows, "calls_per_replay": calls, "samples": 60}))


def check():
    lptr = preload()
    ws = lptr.workspace(torch.device("cuda", torch.cuda.current_device()))
    epoch_word = 128 // 4

    def epoch():
        return int(ws[:256].view(torch.int32)[epoch_word].item())

    results = {}
    passed = True
    configs = [(16464, "select"), (16384, "row"), (32848, "u8x3"), (4176, "select"),
               (16464, "u8x3")]
    pool = {}
    for M, form in configs:
        c = make_case(M, 100 + M + len(form), form)
        pool[(M, form)] = c
        pristine = [t.clone() for t in inputs(c)]
        attn_y, mlp_y = run_pair(c)
        ref = run_base(c)
        emu = emulate_ref(c, attn_y, mlp_y)
        ref_vs_emu = bool(torch.equal(ref, emu))
        key = f"M{M}_{form}"
        results[key] = {"main_vs_f32_emulation_bitwise": ref_vs_emu}
        ok_all = ref_vs_emu
        for arm, rast in (("r0", 0), ("r1", 1), ("r2", 2), ("routed", None)):
            fn = (lambda: run_fused(c)) if rast is None else (lambda r=rast: run_fused_r(c, r))
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            res_r = int(lptr.module().lpt_pair_residual_last_raster())
            n_ref = n_emu = 0
            epochs = []
            for i in range(ITERS):
                e0 = epoch()
                out = fn().reshape(c["x"].shape)
                torch.cuda.synchronize()
                epochs.append(epoch() - e0)
                n_ref += bool(torch.equal(out, ref))
                n_emu += bool(torch.equal(out, emu))
                out.fill_(float("nan"))
                del out
            ok = n_ref == ITERS and n_emu == ITERS and set(epochs) == {1}
            ok_all &= ok
            results[key][arm] = {"bitwise_vs_main": f"{n_ref}/{ITERS}",
                                 "bitwise_vs_f32_emulation": f"{n_emu}/{ITERS}",
                                 "epoch_step": sorted(set(epochs)),
                                 "resolved_raster": res_r, "ok": ok}
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
        ok_all &= inputs_ok
        results[key]["inputs_unchanged"] = inputs_ok
        results[key]["routed_raster"] = lptr.raster_for(M)
        results[key]["ok"] = ok_all
        passed &= ok_all
        c["ref"] = ref
        log(key, json.dumps(results[key]))
        del attn_y, mlp_y, emu
        torch.cuda.synchronize()

    # Kill switch: RASTER=0 routes raster 0 (main's launch: raster_order never assigned) at every M.
    probe = (4176, 8191, 8192, 16384, 16464, 24575, 24576, 32848)
    os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_RASTER"] = "0"
    lptr.raster_enabled.cache_clear()
    kill = {M: lptr.raster_for(M) for M in probe}
    c = pool[(16464, "select")]
    out = run_fused(c).reshape(c["x"].shape)
    torch.cuda.synchronize()
    kill_resolved = int(lptr.module().lpt_pair_residual_last_raster())
    kill_ok = all(v == 0 for v in kill.values()) and bool(torch.equal(out, c["ref"]))
    del out
    del os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_RASTER"]
    lptr.raster_enabled.cache_clear()
    on = {M: lptr.raster_for(M) for M in probe}
    passed &= kill_ok
    log(f"kill switch: raster_for {kill} resolved {kill_resolved} ok {kill_ok}; default: {on}")

    # Repeat sequence over the shared workspace: t2i graph (28 x 16464 select + 4 x 16384 row) and
    # i2i graph (28 x 32848 u8x3 + 9 x 16384 row), routed op; 6 rounds, alternating order.
    t2i_seq = [pool[(16464, "select")]] * 28 + [pool[(16384, "row")]] * 4
    i2i_seq = [pool[(32848, "u8x3")]] * 28 + [pool[(16384, "row")]] * 9
    small = pool[(4176, "select")]
    s = torch.cuda.Stream()
    graphs, outs = {}, {}
    for name, sq in (("t2i", t2i_seq), ("i2i", i2i_seq)):
        graphs[name], outs[name] = capture([lambda c=c: run_fused(c) for c in sq], s)
    seqs = {"t2i": t2i_seq, "i2i": i2i_seq}
    mismatches = checked = 0
    order_log = []
    for r in range(6):
        order = ["t2i", "i2i", "t2i"] if r % 2 == 0 else ["i2i", "t2i", "i2i"]
        for name in order:
            for o in outs[name]:
                o.fill_(float("nan"))
            graphs[name].replay()
            torch.cuda.synchronize()
            for o, c in zip(outs[name], seqs[name]):
                checked += 1
                mismatches += not bool(torch.equal(o.reshape(c["x"].shape), c["ref"]))
            so = run_fused(small).reshape(small["x"].shape)  # eager small-M call (raster 0)
            torch.cuda.synchronize()
            checked += 1
            mismatches += not bool(torch.equal(so, small["ref"]))
            del so
            order_log.append(name)
    inputs_final = True
    passed &= mismatches == 0
    log(f"repeat sequence {order_log}: {mismatches} mismatches of {checked} outputs")
    print(json.dumps({
        "passed": bool(passed),
        "compared_against": "main's LPT pair (flux3_sm120_nvfp4_out_pair_prequant) + compiled "
                            "emulate_precision_casts residual (per-token gate at u8x3), and an "
                            "f32 emulation",
        "per_case": results,
        "kill_switch": {"raster_for": kill, "resolved": kill_resolved, "ok": kill_ok,
                        "default_raster_for": on},
        "repeat_sequence": {"order": order_log, "outputs_checked": checked,
                            "mismatches": mismatches, "inputs_final": inputs_final},
    }))


if __name__ == "__main__":
    {"bench": bench, "table": table, "check": check}[sys.argv[1]]()
