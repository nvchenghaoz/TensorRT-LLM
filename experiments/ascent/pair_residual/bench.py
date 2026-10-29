"""Idea 2 (denoiser_fusion, branch 5): gated residual x + gate*(attn_y + mlp_y) in the SM120 LPT
pair's attn_out-tile epilogue (trtllm::flux3_sm120_nvfp4_out_pair_residual) vs main's LPT pair
(trtllm::flux3_sm120_nvfp4_out_pair_prequant) + the Inductor residual kernel.

Shapes (FLUX.3 klein 2048^2, served T=80): t2i single blocks M=16464 (28/step, [1,2,D] gate table +
bool select, text rows [16384,16464)), image mode blocks M=16384 (4/step, [1,1,D] gate row); i2i:
the 28 single blocks (M=32848) keep the per-token gate (not fused, identical in both arms,
excluded); 4 image mode blocks M=16384. The check also covers M=32848 with a 3-row uint8-indexed
table (idea 24's layout) and a partial-tile M=4176. down [3072 x 9216], attn_out [3072 x 3072]; NVFP4 W4A4, 128x4 scales; attn_out's
activation pre-quantized (the FMHA nvfp4_out path). Synthetic tensors, fixed seeds, 3 distinct
input sets per M cycled (working set >> 128 MB L2). No model load.

  bench base|fused : run_gpu_pair unit. One CUDA graph per case holding one denoise step of the
                     case's NVFP4 block tails; 20 warm replays, 60 timed (CUDA events).
                     base = main's op + torch.compile(emulate_precision_casts) residual exactly as
                     the block writes it; fused = the new op.
  table            : branch only, diagnostics: per M, 20-call graphs of pair only / pair+residual /
                     fused, interleaved samples.
  check            : untimed. Per M and gate form: fused bitwise vs base (main's op + compiled
                     residual) and vs an eager f32 emulation, every call of WARMUP+ITERS with the
                     output poisoned in between, inputs unchanged, epoch advancing by one per call;
                     a CUDA graph of all forms replayed 60x back to back, every replay bitwise; a
                     partial-tile M (4176) and a uint8 3-row table.
"""

import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
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
    if form == "select":  # single blocks: [1, 2, D] table + bool [1, S, 1] (True on text rows)
        gate = torch.randn(1, 2, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        first = 16384 if M > 16384 else M - 80  # served t2i: image [0,16384), text [16384,16464)
        index = (torch.arange(M, device=dev) >= first).view(1, M, 1)
    elif form == "row":  # mode blocks: [1, 1, D]
        gate = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        index = None
    elif form == "u8x3":  # future 3-chunk table + uint8 row index
        gate = torch.randn(1, 3, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.3
        index = torch.randint(0, 3, (1, M, 1), device=dev, dtype=torch.uint8, generator=g)
    else:
        raise ValueError(form)
    return {
        "M": M, "form": form, "down_in": a_down, "down_sf": sf_down_in, "attn_q": attn_q,
        "attn_sf": attn_sf, "out_ad": out_ad, "down_ad": down_ad, "x": x, "gate": gate,
        "index": index,
    }


def inputs(c):
    t = [c["down_in"], c["down_sf"], c["attn_q"], c["attn_sf"], c["out_ad"].weight,
         c["out_ad"].scale, c["down_ad"].weight, c["down_ad"].scale, c["x"], c["gate"]]
    if c["index"] is not None:
        t.append(c["index"])
    return t


def run_pair(c):
    return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        c["attn_q"], c["attn_sf"], c["out_ad"].weight, c["out_ad"].scale)


def _res_select(x, table, select, attn_y, mlp_y):
    # Flux3SingleStreamBlock.forward: _flux3_select_mod + the tail of the block.
    gate = torch.where(select, table[:, 1:2], table[:, 0:1])
    output = attn_y.reshape(*x.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
    return x + gate * output


def _res_row(x, gate, attn_y, mlp_y):
    # Flux3ModeBlock.forward tail.
    output = attn_y.reshape(*x.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
    return x + gate * output


def _res_u8(x, table, index, attn_y, mlp_y):
    gate = table[0][index.reshape(-1).long()].unsqueeze(0)
    output = attn_y.reshape(*x.shape[:-1], -1) + mlp_y.reshape(*x.shape[:-1], -1)
    return x + gate * output


_OPTS = {"emulate_precision_casts": True}
res_select = torch.compile(_res_select, dynamic=False, options=_OPTS)
res_row = torch.compile(_res_row, dynamic=False, options=_OPTS)
res_u8 = torch.compile(_res_u8, dynamic=False, options=_OPTS)


def run_base(c):
    attn_y, mlp_y = run_pair(c)
    if c["form"] == "select":
        return res_select(c["x"], c["gate"], c["index"], attn_y, mlp_y)
    if c["form"] == "row":
        return res_row(c["x"], c["gate"], attn_y, mlp_y)
    return res_u8(c["x"], c["gate"], c["index"], attn_y, mlp_y)


def run_fused(c):
    return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_residual(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        c["attn_q"], c["attn_sf"], c["out_ad"].weight, c["out_ad"].scale,
        c["x"], c["gate"], c["index"])


def emulate_ref(c, attn_y, mlp_y):
    """Eager f32 emulation of the emulate_precision_casts math (BF16 round after every op)."""
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
        for f in fn_seq[:6]:
            f()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = [f() for f in fn_seq]
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


# Per denoise step: (M, form, calls).
CASES = {
    # Served (T=80): 28 single blocks at M=16464 ([1,2,D] table + select), 4 NVFP4 image mode
    # blocks at M=16384 ([1,1,D] row).
    "t2i-2048": [(16464, "select", 28), (16384, "row", 4)],
    # Served i2i: the single blocks (M=32848) build a per-token gate and keep main's Inductor
    # residual in both arms (not timed here); the 4 NVFP4 image mode blocks (M=16384) are fused
    # (job 136 census: 64 fused launches per i2i request = 4 x 16 steps).
    "i2i-2048": [(16384, "row", 4)],
}


def preload():
    fp.sm120_linear_preload(torch.device("cuda", torch.cuda.current_device()))
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import (
            flux3_sm120_lpt_pair_residual as lptr,
        )

        lptr.preload(torch.device("cuda", torch.cuda.current_device()))
        log(f"tree: branch (residual enabled={lptr.enabled()}, "
            f"info={lptr.module().lpt_pair_residual_info()})")
    except ImportError:
        log("tree: base (no flux3_sm120_lpt_pair_residual)")


def bench(arm):
    preload()
    fn = {"base": run_base, "fused": run_fused}[arm]
    pool = {}
    for case, parts in CASES.items():
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
            seq += [lambda c=cs[i % SETS]: fn(c) for i in range(calls)]
        s = torch.cuda.Stream()
        g, _ = capture(seq, s)
        res = time_graphs({case: g}, samples=60, warm=20, calls=1)
        med, p10, p90 = stat(res[case])
        result[case] = {"kernel_ms": med / 1e3, "noise_floor": (p90 - p10) / med,
                        "p10_ms": p10 / 1e3, "p90_ms": p90 / 1e3, "calls": len(seq)}
        log(f"{case} [{arm}]: one step of {len(seq)} block tails: median {med/1e3:.4f} ms "
            f"p10 {p10/1e3:.4f} p90 {p90/1e3:.4f}")
        del g
        torch.cuda.synchronize()
    for k, cs in pool.items():
        for p, c in zip(pristine[k], cs):
            assert all(torch.equal(x, y) for x, y in zip(p, inputs(c))), "inputs changed"
    print(json.dumps({
        "batch": {k: {"kernel_ms": v["kernel_ms"], "noise_floor": v["noise_floor"]}
                  for k, v in result.items()},
        "detail": result, "arm": arm,
        "unit": "one CUDA graph replay = one denoise step of the case's NVFP4 block tails "
                "(LPT pair + gated residual); 3 input sets per M cycled",
        "output": {"class": "exact"},
    }))


def diag():
    """Timing attribution of the fused epilogue (FLUX3_LPTR_DIAG variants have wrong outputs)."""
    preload()
    calls = 20
    rows = {}
    for M, form in ((16464, "select"),):
        cs = [make_case(M, 7 * M + i, form) for i in range(SETS)]
        s = torch.cuda.Stream()

        def seq(f):
            return [lambda c=cs[i % SETS]: f(c) for i in range(calls)]

        graphs = {}
        graphs["pair_only"], _ = capture(seq(run_pair), s)
        graphs["pair_plus_residual"], _ = capture(seq(run_base), s)
        for d, name in ((0, "fused"), (8, "no_prefetch"), (1, "no_loads"), (2, "no_wait"),
                        (9, "no_loads_no_prefetch"), (4, "store_attn_y_only"),
                        (12, "store_attn_y_only_no_prefetch")):
            os.environ["FLUX3_LPTR_DIAG"] = str(d)
            graphs[name], _ = capture(seq(run_fused), s)
        os.environ["FLUX3_LPTR_DIAG"] = "0"
        res = time_graphs(graphs, samples=40, warm=10, calls=calls)
        row = {n: round(stat(v)[0], 2) for n, v in res.items()}
        rows[f"M{M}_{form}"] = row
        log(f"M={M} {form}: {json.dumps(row)}")
    print(json.dumps({"diag_us_per_call": rows}))


def table():
    preload()
    calls = 20
    rows = {}
    for M, form in ((16464, "select"), (32848, "u8x3"), (16384, "row")):
        cs = [make_case(M, 7 * M + i, form) for i in range(SETS)]
        s = torch.cuda.Stream()

        def seq(f):
            return [lambda c=cs[i % SETS]: f(c) for i in range(calls)]

        graphs = {}
        graphs["pair_only"], _ = capture(seq(run_pair), s)
        graphs["pair_plus_residual"], _ = capture(seq(run_base), s)
        graphs["fused"], _ = capture(seq(run_fused), s)
        graphs["pair_plus_residual_AA"], _ = capture(seq(run_base), s)
        res = time_graphs(graphs, samples=60, warm=10, calls=calls)
        row = {}
        for n, v in res.items():
            med, p10, p90 = stat(v)
            row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2)}
        row["residual_us"] = round(row["pair_plus_residual"]["us"] - row["pair_only"]["us"], 2)
        row["fused_minus_pair_us"] = round(row["fused"]["us"] - row["pair_only"]["us"], 2)
        row["saving_us"] = round(row["pair_plus_residual"]["us"] - row["fused"]["us"], 2)
        rows[f"M{M}_{form}"] = row
        log(f"M={M} {form}: {json.dumps(row)}")
        del graphs
        torch.cuda.synchronize()
    print(json.dumps({"table": rows, "calls_per_replay": calls, "samples": 60}))


def check():
    preload()
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr

    ws = lptr.workspace(torch.device("cuda", torch.cuda.current_device()))
    epoch_word = 128 // 4  # carve(): counter at 0, epoch at the next 128-B slot

    def epoch():
        return int(ws[:256].view(torch.int32)[epoch_word].item())

    results = {}
    passed = True
    configs = [(16464, "select"), (16384, "row"), (32848, "u8x3"), (4176, "select"),
               (16464, "u8x3")]
    cases = []
    for M, form in configs:
        c = make_case(M, 100 + M + len(form), form)
        cases.append(c)
        pristine = [t.clone() for t in inputs(c)]
        attn_y, mlp_y = run_pair(c)
        ref = run_base(c)
        emu = emulate_ref(c, attn_y, mlp_y)
        ref_vs_emu = bool(torch.equal(ref, emu))
        for _ in range(WARMUP):
            run_fused(c)
        torch.cuda.synchronize()
        n_ref = n_emu = 0
        epochs = []
        max_abs = 0.0
        for i in range(ITERS):
            e0 = epoch()
            out = run_fused(c).reshape(c["x"].shape)
            torch.cuda.synchronize()
            epochs.append(epoch() - e0)
            n_ref += bool(torch.equal(out, ref))
            n_emu += bool(torch.equal(out, emu))
            max_abs = max(max_abs, float((out.float() - ref.float()).abs().max()))
            out.fill_(float("nan"))
            del out
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
        ok = n_ref == ITERS and n_emu == ITERS and inputs_ok and set(epochs) == {1} and ref_vs_emu
        passed &= ok
        key = f"M{M}_{form}"
        results[key] = {
            "bitwise_vs_main_pair_plus_inductor": f"{n_ref}/{ITERS}",
            "bitwise_vs_f32_emulation": f"{n_emu}/{ITERS}",
            "main_vs_f32_emulation_bitwise": ref_vs_emu,
            "max_abs_vs_main": max_abs, "inputs_unchanged": inputs_ok,
            "epoch_step_per_call": sorted(set(epochs)), "ok": ok,
        }
        log(key, json.dumps(results[key]))
        c["ref"] = ref
        del attn_y, mlp_y, emu
        torch.cuda.synchronize()
    # Graph replay: all forms in one graph, 60 back-to-back replays (no sync in between), each
    # replay's outputs copied out and compared after.
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for c in cases:
            run_fused(c)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            g_outs = [run_fused(c) for c in cases]
    torch.cuda.synchronize()
    replays = 60
    n_good = 0
    e0 = epoch()
    snaps = []
    for r in range(replays):
        for o in g_outs:
            o.fill_(float("nan"))
        graph.replay()
        if r % 6 == 0:
            snaps.append([o.clone() for o in g_outs])
    torch.cuda.synchronize()
    e1 = epoch()
    for snap in snaps:
        n_good += all(torch.equal(o.reshape(c["x"].shape), c["ref"]) for o, c in zip(snap, cases))
    last_ok = all(torch.equal(o.reshape(c["x"].shape), c["ref"]) for o, c in zip(g_outs, cases))
    g_ok = n_good == len(snaps) and last_ok and (e1 - e0) == replays * len(cases)
    passed &= g_ok
    results["graph_replays"] = {
        "replays_back_to_back": replays, "launches_per_replay": len(cases),
        "snapshots_bitwise": f"{n_good}/{len(snaps)}", "last_replay_bitwise": last_ok,
        "epoch_advance": e1 - e0, "expected_epoch_advance": replays * len(cases), "ok": g_ok,
    }
    log("graph", json.dumps(results["graph_replays"]))
    del graph
    # Kill switch: TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=0 disables the route.
    os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL"] = "0"
    lptr.enabled.cache_clear()
    results["kill_switch_disables"] = not lptr.enabled()
    passed &= results["kill_switch_disables"]
    os.environ["TRTLLM_FLUX3_SM120_PAIR_RESIDUAL"] = "1"
    lptr.enabled.cache_clear()
    print(json.dumps({"passed": bool(passed), "results": results,
                      "reference": "main's trtllm::flux3_sm120_nvfp4_out_pair_prequant + "
                                   "torch.compile(emulate_precision_casts) gated residual as the "
                                   "blocks write it; plus an eager f32 emulation"}))


def prof(arm):
    """One M=16896 select call per arm after warmup (for ncu/nsys; not timing evidence)."""
    preload()
    c = make_case(16896, 5, "select")
    fn = {"base": run_base, "fused": run_fused, "pair": run_pair}[arm]
    for _ in range(3):
        fn(c)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(2):
        fn(c)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    log("prof done")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "prof":
        prof(sys.argv[2] if len(sys.argv) > 2 else "fused")
    elif cmd == "bench":
        bench(sys.argv[2] if len(sys.argv) > 2 else "fused")
    elif cmd == "table":
        table()
    elif cmd == "diag":
        diag()
    else:
        check()
