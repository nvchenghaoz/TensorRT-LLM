"""Idea 75: ff.down_proj || attn_out on two streams in one op (single blocks + image mode blocks).

Shapes (FLUX.3 klein 1024^2, T=80 text tokens): single blocks M=4096+T=4176, image mode blocks
M=4096; attn_out [3072 x 3072], down [3072 x 9216]. NVFP4 blocks run the TRT-LLM CUTLASS SM120
LINEAR GEMM (down input = the real packed-SwiGLU kernel output); FP8R blocks run quantize_fp8r +
torch._scaled_mm rowwise. Synthetic tensors, fixed seeds, no model load.

  bench main : main's sequential path (down via flux3_nvfp4_linear_prepared / PreparedFP8Sm120,
               then attn_out via PreparedNVFP4.forward / PreparedFP8Sm120), as the block runs it.
  bench pair : the branch's one-op pair (trtllm::flux3_sm120_{nvfp4,fp8r}_out_pair).
     Timed unit: one denoise step's worth of tails = 24 NVFP4@4176 + 4 FP8R@4176 +
     4 NVFP4@4096 + 1 FP8R@4096 calls, CUDA events, median of ITERS.
  table      : interleaved per iteration main / pair down-first / pair attn_out-first, per path
               and M in (4096, 4176, 4256), median of ITERS (branch tree only).
  check      : untimed, bitwise vs main's sequential outputs (protected reference = main's
               unmodified ops on pristine inputs), WARMUP + ITERS calls vs reference and call 0,
               no-sync stress with allocator reuse, inputs unchanged, both launch orders, CUDA
               graph capture + 3 replays, torch.compile of main's path vs the op.
"""

import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedFP8Sm120, PreparedNVFP4

WARMUP = 3
ITERS = 60
STEP_ITERS = 100
SPIN_CYCLES = 6_000_000  # ~2.5 ms at 2.4 GHz
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def nvfp4_case(M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_attn = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    x_mlp = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    s_attn, s_mlp, s_wout, s_wup, s_wdown = map(_gs, (x_attn, x_mlp, w_out, w_up, w_down))
    s_down_in = 0.05
    a_mlp, sf_mlp = fp.nvfp4_quantize(x_mlp, _inv(s_mlp))
    q_out, sf_out = fp.nvfp4_quantize(w_out, _inv(s_wout))
    q_up, sf_up = fp.nvfp4_quantize(w_up, _inv(s_wup))
    q_down, sf_down = fp.nvfp4_quantize(w_down, _inv(s_wdown))
    packed, packed_sf = flux3_sm120_nvfp4.run(
        "packed", a_mlp, q_up, sf_mlp, sf_up, s_mlp, s_wup, output_inverse_scale=_inv(s_down_in)
    )
    out_ad = PreparedNVFP4(q_out, sf_out, s_attn, s_wout)
    down_ad = PreparedNVFP4(q_down, sf_down, s_down_in, s_wdown)
    return {
        "kind": "nvfp4",
        "M": M,
        "attn": x_attn,
        "down_in": packed.contiguous(),
        "down_sf": packed_sf.contiguous(),
        "out_ad": out_ad,
        "down_ad": down_ad,
    }


def fp8r_case(M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_attn = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    hidden = torch.randn(M, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.5
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    return {
        "kind": "fp8r",
        "M": M,
        "attn": x_attn,
        "down_in": hidden,
        "out_ad": PreparedFP8Sm120(w_out),
        "down_ad": PreparedFP8Sm120(w_down),
    }


def inputs(c):
    ts = [c["attn"], c["down_in"]] + ([c["down_sf"]] if c["kind"] == "nvfp4" else [])
    for ad in (c["out_ad"], c["down_ad"]):
        ts += [ad.weight, ad.scale]
    return ts


def run_main(c):
    """Main's sequential order in the block: MLP (down) first, then attn_out."""
    if c["kind"] == "nvfp4":
        down = torch.ops.trtllm.flux3_nvfp4_linear_prepared(
            c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale
        )
    else:
        down = c["down_ad"](c["down_in"])
    attn = c["out_ad"](c["attn"])
    return attn, down


def run_pair(c):
    if c["kind"] == "nvfp4":
        return torch.ops.trtllm.flux3_sm120_nvfp4_out_pair(
            c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
            c["attn"], c["out_ad"].weight, c["out_ad"].scale,
        )
    return torch.ops.trtllm.flux3_sm120_fp8r_out_pair(
        c["down_in"], c["down_ad"].weight, c["down_ad"].scale,
        c["attn"], c["out_ad"].weight, c["out_ad"].scale,
    )


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def step_cases():
    cases = {
        "nvfp4_4176": nvfp4_case(4176, 1),
        "fp8r_4176": fp8r_case(4176, 2),
        "nvfp4_4096": nvfp4_case(4096, 3),
        "fp8r_4096": fp8r_case(4096, 4),
    }
    plan = [("nvfp4_4176", 24), ("fp8r_4176", 4), ("nvfp4_4096", 4), ("fp8r_4096", 1)]
    return cases, plan


def bench(arm):
    fn = run_main if arm == "main" else run_pair
    cases, plan = step_cases()
    pristine = {k: [t.clone() for t in inputs(c)] for k, c in cases.items()}
    log(f"bench arm={arm} linear_backend={fp._sm120_linear_backend()}")
    for _ in range(WARMUP):
        for k, n in plan:
            for _ in range(n):
                fn(cases[k])
    torch.cuda.synchronize()
    for k, c in cases.items():
        assert all(torch.equal(p, t) for p, t in zip(pristine[k], inputs(c))), k
    # Each iteration starts from an idle GPU behind a ~3 ms spin, so the host has enqueued the
    # step ahead of the GPU (the served denoise is GPU-bound, ~98% busy) and the events measure
    # GPU time only, not launch gaps or the previous iteration's tail.
    step_times = []
    for i in range(STEP_ITERS):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SPIN_CYCLES)
        a.record()
        for k, n in plan:
            for _ in range(n):
                fn(cases[k])
        b.record()
        b.synchronize()
        step_times.append(a.elapsed_time(b))
        if i % 25 == 0:
            log(f"  step iter {i}: {step_times[-1]:.3f} ms")
    step_med, step_noise = stat(step_times)
    log(f"step tails: {step_med:.3f} ms (noise {step_noise:.4f})")
    per = {}
    for k, c in cases.items():
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(ITERS)]
        for i in range(ITERS):
            ev[i][0].record()
            fn(c)
            ev[i][1].record()
        torch.cuda.synchronize()
        med, noise = stat([a.elapsed_time(b) for a, b in ev])
        per[k] = {"us": round(med * 1e3, 2), "noise": round(noise, 4)}
        log(f"  {k}: {med*1e3:.1f} us (noise {noise:.4f})")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": step_med, "noise_floor": step_noise}},
        "unit": "one denoise step of block tails: 24 NVFP4@4176 + 4 FP8R@4176 + 4 NVFP4@4096 "
                "+ 1 FP8R@4096 (down + attn_out incl. attn_out's activation quantize)",
        "per_case": per,
        "arm": arm,
        "output": {"class": "exact"},
    }))


def table():
    rows = {}
    for kind, make in (("nvfp4", nvfp4_case), ("fp8r", fp8r_case)):
        for M in (4096, 4176, 4256):
            c = make(M, M)
            variants = {
                "main_sequential": lambda: run_main(c),
                "pair_down_first": lambda: (setattr(fp, "_SM120_TAIL_PAIR_DOWN_FIRST", True),
                                            run_pair(c)),
                "pair_attn_first": lambda: (setattr(fp, "_SM120_TAIL_PAIR_DOWN_FIRST", False),
                                            run_pair(c)),
            }
            for _ in range(WARMUP):
                for f in variants.values():
                    f()
            ev = {k: [] for k in variants}
            for _ in range(ITERS):
                for k, f in variants.items():
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    f()
                    b.record()
                    ev[k].append((a, b))
            torch.cuda.synchronize()
            fp._SM120_TAIL_PAIR_DOWN_FIRST = True
            row = {}
            for k, pairs in ev.items():
                med, noise = stat([a.elapsed_time(b) for a, b in pairs])
                row[k] = {"us": round(med * 1e3, 2), "noise": round(noise, 4)}
            base = row["main_sequential"]["us"]
            for k in ("pair_down_first", "pair_attn_first"):
                row[k]["saved_us"] = round(base - row[k]["us"], 2)
            rows[f"{kind}_M{M}"] = row
            log(f"{kind} M={M}: {json.dumps(row)}")
    print(json.dumps({"table": rows, "iters": ITERS, "clocks": "default (not locked)"}))


def check():
    results = {}
    passed = True
    for kind, make in (("nvfp4", nvfp4_case), ("fp8r", fp8r_case)):
        for M in (4096, 4176, 4256):
            c = make(M, 100 + M)
            pristine = [t.clone() for t in inputs(c)]
            ref_attn, ref_down = (t.clone() for t in run_main(c))
            for down_first in (True, False):
                fp._SM120_TAIL_PAIR_DOWN_FIRST = down_first
                for p, t in zip(pristine, inputs(c)):
                    t.copy_(p)
                for _ in range(WARMUP):
                    run_pair(c)
                inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
                n_ref = n_first = 0
                first = None
                max_abs = 0.0
                for i in range(ITERS):
                    a, d = run_pair(c)
                    if first is None:
                        first = (a.clone(), d.clone())
                    n_ref += bool(torch.equal(a, ref_attn) and torch.equal(d, ref_down))
                    n_first += bool(torch.equal(a, first[0]) and torch.equal(d, first[1]))
                    max_abs = max(max_abs, float((a.float() - ref_attn.float()).abs().max()),
                                  float((d.float() - ref_down.float()).abs().max()))
                # no-sync stress: outputs freed each call (allocator reuse across streams)
                bad = torch.zeros((), device="cuda", dtype=torch.int64)
                for i in range(ITERS):
                    a, d = run_pair(c)
                    bad += (a != ref_attn).sum() + (d != ref_down).sum()
                    del a, d
                    if i % 3 == 0:
                        junk = torch.empty_like(ref_attn).fill_(7)  # reuse pressure
                        del junk
                stress_bad = int(bad.item())
                inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
                ok = n_ref == ITERS and n_first == ITERS and stress_bad == 0 and inputs_ok
                passed &= ok
                key = f"{kind}_M{M}_{'down' if down_first else 'attn'}_first"
                results[key] = {"bitwise_vs_main": f"{n_ref}/{ITERS}",
                                "bitwise_vs_call0": f"{n_first}/{ITERS}",
                                "max_abs": max_abs, "nosync_stress_mismatches": stress_bad,
                                "inputs_unchanged": inputs_ok, "ok": ok}
                log(key, json.dumps(results[key]))
            fp._SM120_TAIL_PAIR_DOWN_FIRST = True
            # CUDA graph capture of the op (with its side stream) and 3 replays
            run_pair(c)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    g_attn, g_down = run_pair(c)
            torch.cuda.current_stream().wait_stream(s)
            g_ok = True
            for _ in range(3):
                g_attn.zero_(), g_down.zero_()
                graph.replay()
                torch.cuda.synchronize()
                g_ok &= torch.equal(g_attn, ref_attn) and torch.equal(g_down, ref_down)
            del graph
            results[f"{kind}_M{M}_cuda_graph_replay_x3"] = g_ok
            passed &= g_ok
            log(f"{kind} M={M} cuda graph x3 bitwise: {g_ok}")
        # torch.compile of main's path (Inductor lowering of _scaled_mm etc.) vs the op
        c = make(4176, 7)
        torch._dynamo.reset()
        comp_main = torch.compile(lambda: run_main(c), options={"emulate_precision_casts": True})
        comp_pair = torch.compile(lambda: run_pair(c), options={"emulate_precision_casts": True})
        cm = comp_main()
        cp = comp_pair()
        ep = run_pair(c)
        comp_ok = all(torch.equal(x, y) for x, y in zip(cm, cp)) and all(
            torch.equal(x, y) for x, y in zip(cm, ep))
        results[f"{kind}_M4176_compiled_main_vs_op"] = comp_ok
        passed &= comp_ok
        log(f"{kind} compiled main vs op bitwise: {comp_ok}")
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's unmodified sequential ops on pristine inputs (down then attn_out), "
                     "bitwise; plus torch.compile(main path) vs op",
        "results": results,
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1]
    if mode == "bench":
        bench(sys.argv[2])
    elif mode == "table":
        table()
    else:
        check()
