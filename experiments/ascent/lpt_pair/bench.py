"""Idea 162: NVFP4 block-tail pair (down K=9216 || attn_out K=3072, N=3072) as ONE persistent
SM120 CUTLASS launch with a longest-first dynamic tile order (trtllm::flux3_sm120_nvfp4_out_pair).

Shapes (FLUX.3 klein 1024^2, T=80): single blocks M=4176 (24/step), image-mode blocks M=4096
(4/step); down [3072 x 9216], attn_out [3072 x 3072]; NVFP4 W4A4, 128x4 scale layouts. Synthetic
tensors, fixed seeds, 4 distinct weight/activation sets per M (cycled, so the working set exceeds
the 128 MB L2 like the real per-block weights). No model load.

  bench  : run_gpu_pair unit. torch.ops.trtllm.flux3_sm120_nvfp4_out_pair (incl. attn_out's
           activation quantize) in ONE CUDA graph holding 2 denoise steps of the served mix
           (2 x (24 @M4176 + 4 @M4096) = 56 calls); 40 warm replays, 150 timed replays.
           Base tree = main's two-stream pair; branch = the one-launch LPT pair.
  table  : branch only. Per M, graph replay of 50 calls x 150 samples, arms interleaved per
           sample: main two-stream (incl. quant) + its A/A repeat, LPT dynamic/static with
           serial quant (the op) and hoisted quant (pre-quantized attn), main GEMMs only.
  check  : untimed. Per M and scheduler mode: LPT outputs bitwise vs FlashInfer stock tactic 17
           and TRT-LLM FP4GemmRunner tactic 0 per GEMM, every call of WARMUP+ITERS vs reference
           and call 0 with outputs poisoned before each call, inputs unchanged, tile counter value
           after each call, CUDA-graph replay x5 vs eager; main's t18 vs t0 max-abs-diff; env
           TRTLLM_FLUX3_SM120_PAIR_KERNEL=0 path bitwise vs main's two-stream ops and graph capture.
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
CASES_PER_M = 4
WARMUP = 3
ITERS = 20


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make_case(M, seed):
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
    c = {
        "M": M,
        "attn": x_attn,
        "down_in": a_down,
        "down_sf": sf_down_in,
        "out_ad": out_ad,
        "down_ad": down_ad,
        "alpha_down": _scalar(x_attn.device, s_down_in * s_wdown),
        "alpha_attn": _scalar(x_attn.device, s_attn * s_wout),
    }
    # Hoisted quant arm input: attn_out's activation quantized once, outside the timed unit.
    c["attn_q"], c["attn_sf"] = fp.nvfp4_quantize_prepared(x_attn, out_ad.weight)
    return c


def inputs(c):
    return [c["attn"], c["down_in"], c["down_sf"], c["attn_q"], c["attn_sf"],
            c["out_ad"].weight, c["out_ad"].scale, c["down_ad"].weight, c["down_ad"].scale,
            c["alpha_down"], c["alpha_attn"]]


def set_pair_env(kernel: str, sched: str = "dynamic"):
    os.environ["TRTLLM_FLUX3_SM120_PAIR_KERNEL"] = kernel
    os.environ["TRTLLM_FLUX3_SM120_PAIR_SCHED"] = sched
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

        lpt.enabled.cache_clear()
        lpt.sched_mode.cache_clear()
    except ImportError:
        pass


def run_op(c):
    return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        c["attn"], c["out_ad"].weight, c["out_ad"].scale,
    )


def run_lpt_hoisted(c, mode, grid_ctas=0):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    M = c["M"]
    down = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)
    attn = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)
    lpt.run(c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale, c["alpha_down"],
            down, c["attn_q"], c["out_ad"].weight, c["attn_sf"], c["out_ad"].scale,
            c["alpha_attn"], attn, mode=mode, grid_ctas=grid_ctas)
    return attn, down


def fi_linear(a, w, sfa, sfw, alpha, tactic):
    out = torch.empty((a.shape[0], w.shape[0]), device="cuda", dtype=torch.bfloat16)
    fp._sm120_fi_fp4_module().fp4_gemm(a, w, sfa, sfw, alpha, out, fp._sm120_fi_workspace(a.device),
                                       tactic)
    return out


def run_main_hoisted(c):
    """main's two-stream pair (t18 both), attn_out's activation pre-quantized."""
    return fp._sm120_tail_pair(
        c["attn"].device,
        lambda: fi_linear(c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
                          c["alpha_down"], fp._sm120_linear_fi_tactic()),
        lambda: fi_linear(c["attn_q"], c["out_ad"].weight, c["attn_sf"], c["out_ad"].scale,
                          c["alpha_attn"], fp._sm120_linear_fi_tactic()),
    )


def stat(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2]
    return med, v[n // 10], v[(n * 9) // 10]


def capture(fn_seq, stream):
    """CUDA graph of the call sequence fn_seq (list of zero-arg callables) on stream."""
    with torch.cuda.stream(stream):
        for f in fn_seq[:4]:
            f()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for f in fn_seq:
                f()
    torch.cuda.synchronize()
    return graph


def time_graphs(graphs, samples, warm, calls_per_replay):
    """Interleaved replays; returns {name: [us per call]}."""
    names = list(graphs)
    for i in range(warm):
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
            out[n].append(a.elapsed_time(b) * 1e3 / calls_per_replay)
        if i % 50 == 0:
            log(f"  sample {i}: " + " ".join(f"{n}={out[n][-1]:.1f}" for n in names))
    return out


def preload():
    fp.sm120_linear_preload(torch.device("cuda", torch.cuda.current_device()))


def bench():
    preload()
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

        log(f"tree: branch (lpt enabled={lpt.enabled()}, sched={lpt.sched_mode()}, "
            f"info={lpt.module().lpt_pair_info()})")
    except ImportError:
        log("tree: base (no flux3_sm120_lpt_pair)")
    cases = {M: [make_case(M, 1000 * M + i) for i in range(CASES_PER_M)] for M in (4176, 4096)}
    pristine = {M: [[t.clone() for t in inputs(c)] for c in cs] for M, cs in cases.items()}
    seq = []
    # Served mix per step since landing 56 (single_blocks.24-27 on NVFP4): 28 @M4176 + 4 @M4096
    # = 32 pairs/step, 512/request. LPT_BENCH_SINGLE=24 gives the pre-56 mix of jobs 1297/1298.
    n_single = int(os.environ.get("LPT_BENCH_SINGLE", "28"))
    for _ in range(2):
        for i in range(n_single):
            c = cases[4176][i % CASES_PER_M]
            seq.append(lambda c=c: run_op(c))
        for i in range(4):
            c = cases[4096][i % CASES_PER_M]
            seq.append(lambda c=c: run_op(c))
    s = torch.cuda.Stream()
    g = capture(seq, s)
    res = time_graphs({"pair": g}, samples=150, warm=40, calls_per_replay=1)
    for M, cs in cases.items():
        for p, c in zip(pristine[M], cs):
            assert all(torch.equal(x, y) for x, y in zip(p, inputs(c))), "inputs changed"
    med, p10, p90 = stat(res["pair"])
    ms = med / 1e3
    log(f"2-step served pair mix ({len(seq)} calls): median {ms:.4f} ms p10 {p10/1e3:.4f} "
        f"p90 {p90/1e3:.4f}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": ms, "noise_floor": (p90 - p10) / med}},
        "unit": "one CUDA graph replay = 2 denoise steps of NVFP4 tail pairs "
                f"(2 x ({n_single} @M4176 + 4 @M4096) calls of trtllm::flux3_sm120_nvfp4_out_pair, "
                "incl. attn_out's activation quantize); 4 distinct weight sets per M",
        "per_step_ms": ms / 2,
        "samples": len(res["pair"]),
        "output": {"class": "reordered"},
    }))


def table():
    preload()
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    log(f"lpt_pair_info={lpt.module().lpt_pair_info()} (stages*1e6 + smem bytes)")
    calls = 50
    ideal = {4176: 212.0, 4096: 212.0 * 1536 / 1584}
    rows = {}
    for M in (4176, 4096):
        cs = [make_case(M, 7 * M + i) for i in range(CASES_PER_M)]
        tiles = 2 * ((M + 127) // 128) * (H // 128)
        s = torch.cuda.Stream()

        def seq(fn):
            return [lambda c=cs[i % CASES_PER_M]: fn(c) for i in range(calls)]

        graphs = {}
        set_pair_env("0")
        graphs["main_pair_incl_quant"] = capture(seq(run_op), s)
        graphs["main_pair_incl_quant_AA"] = capture(seq(run_op), s)
        graphs["main_gemms_hoisted"] = capture(seq(run_main_hoisted), s)
        set_pair_env("1", "dynamic")
        graphs["lpt_dyn_serial_quant"] = capture(seq(run_op), s)
        set_pair_env("1", "static")
        graphs["lpt_static_serial_quant"] = capture(seq(run_op), s)
        set_pair_env("1", "ondemand")
        graphs["lpt_ondemand_serial_quant"] = capture(seq(run_op), s)
        set_pair_env("1", "dynamic")
        graphs["lpt_dyn_hoisted"] = capture(seq(lambda c: run_lpt_hoisted(c, 1)), s)
        graphs["lpt_ondemand_hoisted"] = capture(seq(lambda c: run_lpt_hoisted(c, 2)), s)
        graphs["lpt_static_hoisted"] = capture(seq(lambda c: run_lpt_hoisted(c, 0)), s)
        if os.environ.get("LPT_TABLE_ONETILE", "0") == "1":
            graphs["onetile_per_cta_hoisted"] = capture(
                seq(lambda c: run_lpt_hoisted(c, 0, grid_ctas=tiles)), s)
        log(f"M={M}: captured {len(graphs)} graphs, timing")
        res = time_graphs(graphs, samples=150, warm=20, calls_per_replay=calls)
        row = {}
        for n, v in res.items():
            med, p10, p90 = stat(v)
            row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                      "gap_to_ideal_us": round(med - ideal[M], 2)}
        base = row["main_pair_incl_quant"]["us"]
        for n in row:
            row[n]["delta_vs_main_us"] = round(row[n]["us"] - base, 2)
        rows[f"M{M}"] = row
        log(f"M={M}: {json.dumps(row, indent=1)}")
        del graphs
        torch.cuda.synchronize()
    set_pair_env("1", "dynamic")
    print(json.dumps({"table": rows, "calls_per_replay": calls, "samples": 150,
                      "ideal_us": ideal, "clocks": "default (not locked)"}))


def check():
    preload()
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    results = {}
    passed = True
    for M in (4176, 4096):
        c = make_case(M, 100 + M)
        tiles = 2 * ((M + 127) // 128) * (H // 128)
        pristine = [t.clone() for t in inputs(c)]
        # Protected references: the unmodified stock kernels on pristine inputs.
        ref = {}
        for name, (a, w, sfa, sfw, al) in {
            "down": (c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
                     c["alpha_down"]),
            "attn": (c["attn_q"], c["out_ad"].weight, c["attn_sf"], c["out_ad"].scale,
                     c["alpha_attn"]),
        }.items():
            t17 = fi_linear(a, w, sfa, sfw, al, 17)
            t18 = fi_linear(a, w, sfa, sfw, al, 18)
            t0 = fp._sm120_fp4_gemm_runner().run_gemm(a, w, sfa, sfw, al, 0, 0, None, None)
            ref[name] = {"t17": t17, "t18": t18, "t0": t0}
            info = {
                "t17_vs_t0_bitwise": bool(torch.equal(t17, t0)),
                "t18_vs_t0_bitwise": bool(torch.equal(t18, t0)),
                "t18_vs_t0_max_abs": float((t18.float() - t0.float()).abs().max()),
                "t0_abs_max": float(t0.float().abs().max()),
            }
            results[f"M{M}_{name}_stock"] = info
            log(f"M={M} {name} stock refs: {info}")
        # The pair op's attn quantize must equal the hoisted one (same kernel, same scale).
        q2, sf2 = fp.nvfp4_quantize_prepared(c["attn"], c["out_ad"].weight)
        quant_ok = torch.equal(q2, c["attn_q"]) and torch.equal(sf2, c["attn_sf"])
        results[f"M{M}_attn_quant_repeat_bitwise"] = bool(quant_ok)
        passed &= quant_ok
        ws = lpt.workspace(c["attn"].device)
        for mode_name, mode in (("dynamic", 1), ("ondemand", 2), ("static", 0)):
            set_pair_env("1", mode_name)
            for t, p in zip(inputs(c), pristine):
                t.copy_(p)
            for _ in range(WARMUP):
                run_op(c)
            torch.cuda.synchronize()
            n_ref = n_first = n_t0 = 0
            counters = set()
            first = None
            for i in range(ITERS):
                attn, down = run_op(c)
                torch.cuda.synchronize()
                counters.add(int(ws[:4].view(torch.int32).item()))
                if first is None:
                    first = (attn.clone(), down.clone())
                n_ref += bool(torch.equal(attn, ref["attn"]["t17"])
                              and torch.equal(down, ref["down"]["t17"]))
                n_t0 += bool(torch.equal(attn, ref["attn"]["t0"])
                             and torch.equal(down, ref["down"]["t0"]))
                n_first += bool(torch.equal(attn, first[0]) and torch.equal(down, first[1]))
                # poison the freed blocks so a skipped tile in the next call cannot pass
                attn.fill_(float("nan"))
                down.fill_(float("nan"))
                del attn, down
            inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
            max_abs_t18 = max(
                float((first[0].float() - ref["attn"]["t18"].float()).abs().max()),
                float((first[1].float() - ref["down"]["t18"].float()).abs().max()))
            # hoisted entry, same kernel
            ha, hd = run_lpt_hoisted(c, mode)
            hoisted_ok = torch.equal(ha, ref["attn"]["t17"]) and torch.equal(hd, ref["down"]["t17"])
            # CUDA graph capture + replays vs eager (outputs poisoned before each replay)
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                run_op(c)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    g_attn, g_down = run_op(c)
            torch.cuda.synchronize()
            g_ok = True
            for _ in range(5):
                g_attn.fill_(float("nan"))
                g_down.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                g_ok &= torch.equal(g_attn, first[0]) and torch.equal(g_down, first[1])
                counters.add(int(ws[:4].view(torch.int32).item()))
            del graph
            ok = (n_ref == ITERS and n_first == ITERS and inputs_ok and hoisted_ok and g_ok)
            if mode >= 1:
                ok &= counters == {tiles}
            passed &= ok
            key = f"M{M}_lpt_{mode_name}"
            results[key] = {
                "bitwise_vs_fi_t17": f"{n_ref}/{ITERS}", "bitwise_vs_trtllm_t0": f"{n_t0}/{ITERS}",
                "bitwise_vs_call0": f"{n_first}/{ITERS}", "inputs_unchanged": inputs_ok,
                "hoisted_entry_bitwise_t17": hoisted_ok, "graph_replay_x5_bitwise_vs_eager": g_ok,
                "tile_counter_after_calls": sorted(counters), "expected_counter": tiles,
                "max_abs_vs_main_t18": max_abs_t18, "ok": ok,
            }
            log(key, json.dumps(results[key]))
        # TRTLLM_FLUX3_SM120_PAIR_KERNEL=0 restores main's two-stream path (t18 both), eager and
        # under CUDA-graph capture.
        set_pair_env("0")
        attn, down = run_op(c)
        off_ok = torch.equal(attn, ref["attn"]["t18"]) and torch.equal(down, ref["down"]["t18"])
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            run_op(c)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                g_attn, g_down = run_op(c)
        torch.cuda.synchronize()
        g_off = True
        for _ in range(3):
            g_attn.fill_(float("nan"))
            g_down.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            g_off &= torch.equal(g_attn, ref["attn"]["t18"]) and torch.equal(g_down, ref["down"]["t18"])
        del graph
        results[f"M{M}_env0_main_path"] = {"bitwise_vs_main_t18_ops": off_ok,
                                           "graph_capture_replay_x3": g_off}
        passed &= off_ok and g_off
        log(f"M={M} env=0: eager {off_ok} graph {g_off}")
        set_pair_env("1", "dynamic")
    print(json.dumps({
        "passed": bool(passed),
        "reference": "FlashInfer stock SM120 fp4_gemm tactic 17 (DP 128x128x128) and TRT-LLM "
                     "FP4GemmRunner tactic 0 per GEMM on pristine inputs; main's tactic 18 "
                     "outputs for the env=0 path",
        "output_class_vs_main": "reordered (main runs StreamK tactic 18; LPT is bitwise to t0/t17)",
        "results": results,
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    {"bench": bench, "table": table, "check": check}[sys.argv[1]]()
