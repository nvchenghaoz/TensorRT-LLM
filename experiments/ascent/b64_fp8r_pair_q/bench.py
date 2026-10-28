"""Idea 119 / branch 64: FP8R tail pair through the fused swiglu_fp8r_quant + new op
trtllm::flux3_sm120_fp8r_out_pair_q.

Shapes (FLUX.3 klein 1024^2, T=80): single blocks 24-27 M=4176, content_mode_blocks.image.0
M=4096; H=3072, FF=9216 (gate_up [18432 x 3072], down [3072 x 9216], attn_out [3072 x 3072]).
Synthetic tensors, fixed seeds, no model load. mlp_input is an Fp8RowwiseQuantizedTensor built
with quantize_fp8r, as _flux3_shared_fp8_input does for FP8R blocks.

Timed unit: primsts.sm120_tail_pair_forward(block, attn, mlp_input) whole (gate_up GEMM, SwiGLU,
down quantize, down || attn_out pair) -- one denoise step's FP8R tails = 4 calls @4176 + 1 @4096.
  bench : module defaults (flag on). On the base tree that is main's path (the flag does not
          reach the pair there); on the branch it is the fused path.
  ab    : branch only, in-process interleaved flag off (main's path) vs on, per M.
  check : branch only, untimed, bitwise.
L2: a 256 MB buffer is overwritten before every timed iteration (cold L2 for weights/inputs),
then a ~2.5 ms GPU spin so the host enqueues ahead; clocks default (not locked).
"""

import json
import sys
from types import SimpleNamespace

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedFP8Sm120

H, FF = 3072, 9216
WARMUP = 10
ITERS = 60
STEP_ITERS = 150
STEP_WARMUP = 200  # ~1.3 s of back-to-back work: clocks settle before timing
SPIN_CYCLES = 6_000_000
PLAN = [(4176, 4), (4096, 1)]


def log(*a):
    print(*a, flush=True)


def make_case(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    attn = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    data, scales = quantize_fp8r(x)
    block = SimpleNamespace(
        attn_out=SimpleNamespace(_flux3_primsts_fp8=PreparedFP8Sm120(w_out)),
        ff=SimpleNamespace(
            gate_up_proj=SimpleNamespace(_flux3_primsts_fp8=PreparedFP8Sm120(w_up, swiglu=True)),
            down_proj=SimpleNamespace(_flux3_primsts_fp8=PreparedFP8Sm120(w_down)),
        ),
    )
    return {"M": M, "x": x, "attn": attn, "data": data, "scales": scales, "block": block}


def tensors(c):
    b = c["block"]
    ts = [c["x"], c["attn"], c["data"], c["scales"]]
    for ad in (b.attn_out._flux3_primsts_fp8, b.ff.gate_up_proj._flux3_primsts_fp8,
               b.ff.down_proj._flux3_primsts_fp8):
        ts += [ad.weight, ad.scale]
    return ts


def restore(c, pristine):
    for p, t in zip(pristine, tensors(c)):
        t.copy_(p)


def call(c):
    mlp = Fp8RowwiseQuantizedTensor(c["x"], c["data"], c["scales"])
    return primsts.sm120_tail_pair_forward(c["block"], c["attn"], mlp)


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def bench():
    cases = {M: make_case(M, M) for M, _ in PLAN}
    pristine = {M: [t.clone() for t in tensors(c)] for M, c in cases.items()}
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    has_q = hasattr(torch.ops.trtllm, "flux3_sm120_fp8r_out_pair_q")
    log(f"bench flag={primsts._FP8R_FUSED_SWIGLU_QUANT} tree_has_pair_q={has_q}")
    for M, c in cases.items():
        restore(c, pristine[M])
    for i in range(STEP_WARMUP):
        for M, n in PLAN:
            for _ in range(n):
                assert call(cases[M]) is not None
        if i % 50 == 0:
            torch.cuda.synchronize()
            log(f"  warmup {i}")
    torch.cuda.synchronize()
    for M, c in cases.items():
        assert all(torch.equal(p, t) for p, t in zip(pristine[M], tensors(c))), M
        restore(c, pristine[M])
    step, per = [], {M: [] for M, _ in PLAN}
    for i in range(STEP_ITERS):
        flush.fill_(i & 0xFF)
        torch.cuda._sleep(SPIN_CYCLES)
        evs = []
        a = torch.cuda.Event(enable_timing=True)
        a.record()
        prev = a
        for M, n in PLAN:
            for _ in range(n):
                call(cases[M])
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                evs.append((M, prev, e))
                prev = e
        prev.synchronize()
        step.append(a.elapsed_time(prev))
        for M, s, e in evs:
            per[M].append(s.elapsed_time(e))
        if i % 20 == 0:
            log(f"  iter {i}: step {step[-1]:.3f} ms")
    med, noise = stat(step)
    per_out = {}
    for M, v in per.items():
        m, n = stat(v)
        per_out[f"M{M}"] = {"us": round(m * 1e3, 2), "noise": round(n, 4), "n": len(v)}
    log(f"step FP8R tails {med:.4f} ms noise {noise:.4f}; per call {per_out}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "unit": "one denoise step of FP8R block tails: 4 x sm120_tail_pair_forward@M4176 + "
                "1 x @M4096 (gate_up GEMM + SwiGLU/quant + down||attn_out pair), CUDA events, "
                f"median of {STEP_ITERS}, L2 flushed (256 MB fill) per iteration, default clocks",
        "per_call": per_out,
        "fused_pair_path": bool(has_q and primsts._FP8R_FUSED_SWIGLU_QUANT),
        "output": {"class": "exact"},
    }))


def ab():
    rows = {}
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for M in (4176, 4096):
        c = make_case(M, M)
        pristine = [t.clone() for t in tensors(c)]
        restore(c, pristine)
        for _ in range(WARMUP):
            for flag in (False, True):
                primsts._FP8R_FUSED_SWIGLU_QUANT = flag
                call(c)
        torch.cuda.synchronize()
        restore(c, pristine)
        ts = {False: [], True: []}
        for i in range(ITERS):
            for flag in ((False, True) if i % 2 == 0 else (True, False)):
                primsts._FP8R_FUSED_SWIGLU_QUANT = flag
                flush.fill_(i & 0xFF)
                torch.cuda._sleep(SPIN_CYCLES)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                call(c)
                b.record()
                b.synchronize()
                ts[flag].append(a.elapsed_time(b))
        primsts._FP8R_FUSED_SWIGLU_QUANT = True
        r = {}
        for flag, name in ((False, "main_path"), (True, "fused_path")):
            m, n = stat(ts[flag])
            r[name] = {"us": round(m * 1e3, 2), "noise": round(n, 4)}
        r["saved_us"] = round(r["main_path"]["us"] - r["fused_path"]["us"], 2)
        rows[f"M{M}"] = r
        log(f"M={M}: {json.dumps(r)}")
    print(json.dumps({"ab": rows, "iters": ITERS, "l2": "flushed per iteration",
                      "clocks": "default"}))


def compile_check(c, flag):
    """fullgraph compile (no graph breaks) of the forward; record which custom ops are traced."""
    primsts._FP8R_FUSED_SWIGLU_QUANT = flag
    torch._dynamo.reset()
    seen = []

    def backend(gm, example_inputs):
        seen.append(str(gm.graph))
        return gm.forward

    def fn(x, data, scales, attn):
        mlp = Fp8RowwiseQuantizedTensor(x, data, scales)
        return primsts.sm120_tail_pair_forward(c["block"], attn, mlp)

    out_fx = torch.compile(fn, backend=backend, fullgraph=True)(
        c["x"], c["data"], c["scales"], c["attn"])
    g = "\n".join(seen)
    torch._dynamo.reset()
    out_ind = torch.compile(fn, fullgraph=True)(c["x"], c["data"], c["scales"], c["attn"])
    torch._dynamo.reset()
    primsts._FP8R_FUSED_SWIGLU_QUANT = True
    main_pair = g.replace("flux3_sm120_fp8r_out_pair_q", "")
    return {
        "graphs": len(seen),
        "has_swiglu_fp8r_quant": "flux3_swiglu_fp8r_quant" in g,
        "has_out_pair_q": "flux3_sm120_fp8r_out_pair_q" in g,
        "has_out_pair_main": "flux3_sm120_fp8r_out_pair" in main_pair,
    }, out_fx, out_ind


def check():
    results = {}
    passed = True
    for M in (4176, 4096):
        c = make_case(M, 1000 + M)
        pristine = [t.clone() for t in tensors(c)]
        # Protected reference: main's path (flag off: up() -> silu_and_mul -> fp8r_out_pair).
        restore(c, pristine)
        primsts._FP8R_FUSED_SWIGLU_QUANT = False
        ref = tuple(t.clone() for t in call(c))
        # Explicit main-op reference too (independent of the flag dispatch).
        b = c["block"]
        up, down, out = (b.ff.gate_up_proj._flux3_primsts_fp8, b.ff.down_proj._flux3_primsts_fp8,
                         b.attn_out._flux3_primsts_fp8)
        hidden = up(Fp8RowwiseQuantizedTensor(c["x"], c["data"], c["scales"]))
        ref2 = torch.ops.trtllm.flux3_sm120_fp8r_out_pair(
            hidden, down.weight, down.scale, c["attn"].reshape(-1, H), out.weight, out.scale)
        flag_off_is_main = all(torch.equal(x, y) for x, y in zip(ref, ref2))
        primsts._FP8R_FUSED_SWIGLU_QUANT = True
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, tensors(c)))
        restore(c, pristine)
        # Same sequence as timing: WARMUP calls, then ITERS compared calls.
        for _ in range(WARMUP):
            call(c)
        torch.cuda.synchronize()
        inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, tensors(c)))
        restore(c, pristine)
        n_ref = n_first = 0
        first = None
        max_abs = 0.0
        for i in range(ITERS):
            a, d = call(c)
            if first is None:
                first = (a.clone(), d.clone())
            n_ref += bool(torch.equal(a, ref[0]) and torch.equal(d, ref[1]))
            n_first += bool(torch.equal(a, first[0]) and torch.equal(d, first[1]))
            max_abs = max(max_abs, float((a.float() - ref[0].float()).abs().max()),
                          float((d.float() - ref[1].float()).abs().max()))
        inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, tensors(c)))
        # no-sync stress: outputs freed each call, allocator reuse across streams
        bad = torch.zeros((), device="cuda", dtype=torch.int64)
        for i in range(ITERS):
            a, d = call(c)
            bad += (a != ref[0]).sum() + (d != ref[1]).sum()
            del a, d
        stress_bad = int(bad.item())
        # CUDA graph capture + 3 replays
        restore(c, pristine)
        call(c)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                g_a, g_d = call(c)
        torch.cuda.current_stream().wait_stream(s)
        g_ok = True
        for _ in range(3):
            g_a.zero_(), g_d.zero_()
            graph.replay()
            torch.cuda.synchronize()
            g_ok &= torch.equal(g_a, ref[0]) and torch.equal(g_d, ref[1])
        del graph
        inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, tensors(c)))
        ok = (flag_off_is_main and n_ref == ITERS and n_first == ITERS and stress_bad == 0
              and g_ok and inputs_ok)
        r = {"flag_off_equals_main_ops": flag_off_is_main,
             "bitwise_vs_main": f"{n_ref}/{ITERS}", "bitwise_vs_call0": f"{n_first}/{ITERS}",
             "max_abs": max_abs, "nosync_stress_mismatches": stress_bad,
             "cuda_graph_replay_x3_bitwise": g_ok, "inputs_unchanged": inputs_ok, "ok": ok}
        log(f"M={M} eager: {json.dumps(r)}")
        # torch.compile: fullgraph (no breaks) for both flag settings; ops traced
        try:
            info_on, fx_on, ind_on = compile_check(c, True)
            info_off, fx_off, ind_off = compile_check(c, False)
            comp_ok = (info_on["has_swiglu_fp8r_quant"] and info_on["has_out_pair_q"]
                       and info_off["has_out_pair_main"] and not info_off["has_out_pair_q"]
                       and info_on["graphs"] == 1 and info_off["graphs"] == 1
                       and all(torch.equal(x, y) for x, y in zip(fx_on, ref))
                       and all(torch.equal(x, y) for x, y in zip(ind_on, ind_off)))
            r["compile"] = {"flag_on": info_on, "flag_off": info_off,
                            "fx_on_bitwise_vs_main": all(torch.equal(x, y)
                                                         for x, y in zip(fx_on, ref)),
                            "inductor_on_vs_inductor_off_bitwise": all(
                                torch.equal(x, y) for x, y in zip(ind_on, ind_off)),
                            "inductor_on_vs_eager_main_bitwise": all(
                                torch.equal(x, y) for x, y in zip(ind_on, ref)),
                            "ok": comp_ok}
        except Exception as e:  # noqa: BLE001
            comp_ok = False
            r["compile"] = {"error": repr(e)[:800], "ok": False}
        primsts._FP8R_FUSED_SWIGLU_QUANT = True
        r["ok"] = ok and comp_ok
        passed &= r["ok"]
        results[f"M{M}"] = r
        log(f"M={M}: {json.dumps(r)}")
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's FP8R tail-pair path (up() -> silu_and_mul -> fp8r_out_pair, i.e. flag "
                     "off, itself checked bitwise vs the explicit main ops) on pristine inputs; "
                     f"{WARMUP} warmup + {ITERS} compared calls, bitwise",
        "results": results,
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_grad_enabled(False)  # PreparedFP8.can_run requires grad off, as in serving
    mode = sys.argv[1]
    {"bench": bench, "ab": ab, "check": check}[mode]()
