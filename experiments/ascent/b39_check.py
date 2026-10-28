"""Branch 39 correctness check (untimed): LINEAR op on FlashInfer stock SM120 StreamK vs TRT-LLM t0.

Protected reference: main's route, TRT-LLM FP4GemmRunner tactic 0 (run directly), plus an fp32
reference of the same math (dequantized NVFP4 operands, fp32 matmul, * alpha) for scale.
Pre-declared tolerance (reordered: StreamK adds fp32 K-partials in another order):
  benchmark inputs (randn activations, real packed-SwiGLU producer for down) -> every element
  within 1 BF16 ulp of t0, at M = 4096/4176/4256 for attn_out and down;
  every iteration bitwise equal to iteration 0 (deterministic reduction); inputs unchanged.
Also reported (informational): wide-dynamic-range stress inputs, where cancellation makes ulp
distance meaningless; error normalised by the row's fp32 |ref| max, and fp32-ref error of both.
Sequence matches the timed one: WARMUP=5 calls, then ITERS=60 compared calls per case.
Plus: two consecutive 16-step x (attn_out, down) x (M=4176, 4096) sequences bitwise equal; two
streams running attn_out || down concurrently bitwise equal to serial (per-stream workspace);
no per-call workspace allocation (positive control: a 1-byte workspace does allocate).
"""

import json
import sys

import torch

sys.path.insert(0, "experiments/ascent")
import tensorrt_llm  # noqa: E402,F401
from b39_linear_sweep import dequant, make_case, ordered  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import (  # noqa: E402
    _scalar,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402

WARMUP = 5
ITERS = 60


def log(*a):
    print(*a, flush=True)


def op(args):
    return torch.ops.trtllm.flux3_primsts_nvfp4_linear(*args)


def t0(args):
    a, w, sfa, sfw, gsa, gsw = args
    alpha = _scalar(a.device, gsa * gsw)
    return fp._sm120_fp4_gemm_runner().run_gemm(a, w, sfa, sfw, alpha, 0, 0, None, None)


def tensors(args):
    return [x for x in args if isinstance(x, torch.Tensor)]


def errors(out, ref, args):
    a, w, sfa, sfw, gsa, gsw = args
    M, N, K = a.shape[0], w.shape[0], a.shape[1] * 2
    ref32 = (dequant(a, sfa, M, K) @ dequant(w, sfw, N, K).t()) * (gsa * gsw)
    d = (out.float() - ref.float()).abs()
    rowmax = ref32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    e_out = (out.float() - ref32).abs()
    e_ref = (ref.float() - ref32).abs()
    r = {
        "max_ulp_vs_t0": int((ordered(out) - ordered(ref)).abs().max()),
        "diff_frac_vs_t0": float((d > 0).float().mean()),
        "max_abs_vs_t0": float(d.max()),
        "max_rel_vs_t0": float((d / ref.float().abs().clamp_min(1e-30)).max()),
        "max_diff_over_rowmax": float((d / rowmax).max()),
        "cand_relL2_vs_fp32": float(e_out.norm() / ref32.norm()),
        "t0_relL2_vs_fp32": float(e_ref.norm() / ref32.norm()),
        "cand_max_err_over_rowmax_fp32": float((e_out / rowmax).max()),
        "t0_max_err_over_rowmax_fp32": float((e_ref / rowmax).max()),
    }
    return r


def wide_case(M, N, K, seed, spread=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def wide(rows, outl):
        x = torch.randn(rows, K, device="cuda", generator=g)
        blk = torch.rand(rows, K // 16, device="cuda", generator=g) * 2 * spread - spread
        x = x * torch.exp2(blk).repeat_interleave(16, dim=1)
        if outl:
            cols = torch.randint(0, K, (K // 64,), device="cuda", generator=g)
            x[:, cols] *= 32.0
        return x.to(torch.bfloat16)

    x, w = wide(M, True), wide(N, False)
    sa, sw = float(x.float().abs().amax() / 2688.0), float(w.float().abs().amax() / 2688.0)
    a, sfa = fp.nvfp4_quantize(x, torch.tensor([1 / sa], device="cuda"))
    b, sfb = fp.nvfp4_quantize(w, torch.tensor([1 / sw], device="cuda"))
    return (a, b, sfa, sfb, sa, sw)


def main():
    passed = True
    res = {"backend": fp._sm120_linear_backend(), "tactic": fp._sm120_linear_fi_tactic()}
    log(res)
    fp.sm120_linear_preload(torch.device("cuda"))
    # ---- 1. protocol sequence on benchmark inputs ----
    for M in (4096, 4176, 4256):
        case = make_case(M, seed=M)
        for name, args in case.items():
            pristine = [x.clone() for x in tensors(args)]
            ref = t0(args).clone()
            for _ in range(WARMUP):
                op(args)
            ok_in = all(torch.equal(p, x) for p, x in zip(pristine, tensors(args)))
            first = None
            n_first = n_tol = 0
            max_ulp = 0
            for i in range(ITERS):
                out = op(args)
                if first is None:
                    first = out.clone()
                n_first += bool(torch.equal(out, first))
                u = int((ordered(out) - ordered(ref)).abs().max())
                max_ulp = max(max_ulp, u)
                n_tol += u <= 1
            ok_in &= all(torch.equal(p, x) for p, x in zip(pristine, tensors(args)))
            r = errors(first, ref, args)
            r.update({"iters_within_1ulp": f"{n_tol}/{ITERS}", "iters_bitwise_iter0": f"{n_first}/{ITERS}",
                      "bitwise_t0": bool(torch.equal(first, ref)), "inputs_unchanged": ok_in,
                      "max_ulp_all_iters": max_ulp})
            ok = n_tol == ITERS and n_first == ITERS and ok_in
            r["ok"] = ok
            passed &= ok
            res[f"bench_M{M}_{name}"] = r
            log(f"bench M{M} {name}: {r}")
    # ---- 2. stress inputs (informational) ----
    for M in (4096, 4176):
        for name, K in (("attn_out", 3072), ("down", 9216)):
            args = wide_case(M, 3072, K, seed=M + K)
            ref = t0(args).clone()
            outs = [op(args).clone() for _ in range(3)]
            r = errors(outs[0], ref, args)
            r["repeat_bitwise"] = all(torch.equal(outs[0], o) for o in outs[1:])
            passed &= r["repeat_bitwise"]
            res[f"stress_M{M}_{name}"] = r
            log(f"stress M{M} {name}: {r}")
    # ---- 3. two consecutive 16-step sequences ----
    cases = {M: make_case(M, seed=M + 1) for M in (4176, 4096)}

    def sequence():
        outs = []
        for _ in range(16):
            for M in (4176, 4096):
                outs.append(op(cases[M]["attn_out"]).clone())
                outs.append(op(cases[M]["down"]).clone())
        return outs

    s1, s2 = sequence(), sequence()
    seq_ok = all(torch.equal(x, y) for x, y in zip(s1, s2)) and all(
        torch.equal(s1[i], s1[i % 4]) for i in range(len(s1)))
    res["two_sequences_bitwise"] = seq_ok
    passed &= seq_ok
    log(f"two 16-step sequences bitwise: {seq_ok}")
    # ---- 4. concurrent streams ----
    c = cases[4176]
    r_attn, r_down = op(c["attn_out"]).clone(), op(c["down"]).clone()
    s_a, s_b = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    conc_ok = True
    for _ in range(5):
        with torch.cuda.stream(s_a):
            oa = op(c["attn_out"])
        with torch.cuda.stream(s_b):
            od = op(c["down"])
        torch.cuda.synchronize()
        conc_ok &= torch.equal(oa, r_attn) and torch.equal(od, r_down)
    res["two_stream_concurrent_bitwise"] = conc_ok
    res["workspaces"] = len(fp._SM120_FI_WORKSPACES)
    passed &= conc_ok
    log(f"two-stream concurrent bitwise: {conc_ok}, workspaces={len(fp._SM120_FI_WORKSPACES)}")
    # ---- 5. no per-call allocation ----
    torch.cuda.synchronize()

    def n_alloc():
        return torch.cuda.memory_stats()["allocation.all.allocated"]

    a, w, sfa, sfw, gsa, gsw = cases[4176]["down"]
    alpha = _scalar(a.device, gsa * gsw)
    out = torch.empty(a.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
    mod = fp._sm120_fi_fp4_module()
    tiny = torch.empty(1, dtype=torch.uint8, device="cuda")
    allocs = {}
    for label, ws in (("tiny_ws_control", tiny), ("preloaded_ws", fp._sm120_fi_workspace(a.device))):
        for (M, s) in ((4176, "down"), (4176, "attn_out"), (4096, "down"), (4096, "attn_out")):
            a, w, sfa, sfw, gsa, gsw = cases[M][s]
            out = torch.empty(a.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
            alpha = _scalar(a.device, gsa * gsw)
            mod.fp4_gemm(a, w, sfa, sfw, alpha, out, ws, fp._sm120_linear_fi_tactic())
            torch.cuda.synchronize()
            n0 = n_alloc()
            for _ in range(10):
                mod.fp4_gemm(a, w, sfa, sfw, alpha, out, ws, fp._sm120_linear_fi_tactic())
            torch.cuda.synchronize()
            allocs[f"{label}_M{M}_{s}"] = n_alloc() - n0
    res["allocations_per_10_calls"] = allocs
    control_detects = all(v > 0 for k, v in allocs.items() if k.startswith("tiny"))
    no_alloc = all(v == 0 for k, v in allocs.items() if k.startswith("preloaded"))
    res["alloc_detection_works"] = control_detects
    res["no_per_call_alloc"] = no_alloc
    passed &= no_alloc
    log(f"allocs: {allocs}")
    print(json.dumps({"passed": bool(passed),
                      "reference": "TRT-LLM FP4GemmRunner tactic 0 (main's route) + fp32 reference; "
                                   "tolerance <=1 BF16 ulp per element on benchmark inputs, "
                                   "bitwise iteration-to-iteration",
                      "results": res}))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
