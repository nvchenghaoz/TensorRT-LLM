"""Branch 63: CUTLASS SM120 NVFP4 gate_up (t2 256x128x128) + register SwiGLU->NVFP4 epilogue.

Shapes (FLUX.3 Klein single blocks): gate_up N=18432, K=3072, M=4176 (single 0-23) / 4096.

  build : build/load the JIT module, print stages/smem, one bitwise comparison with diagnostics.
  bench : CUDA events, WARMUP then ITERS back-to-back calls, median and (p90-p10)/median.
          The routed op (trtllm::flux3_primsts_nvfp4_swiglu_packed: base -> CuTe run('packed'),
          candidate -> the CUTLASS module) is the arm's metric; run('packed') called directly and
          plain FP4GemmRunner t2 (BF16 out) are yardsticks in the same process.
  check : untimed sequence with the same WARMUP/ITERS; every iteration's PACKED data bytes
          (rows < M), the full scale buffer (incl. padded-row scales) and the down GEMM fed by them
          compared bitwise with the protected reference (main's unmodified CuTe run('packed')) and
          with iteration 0; inputs checked against pristine copies; CUDA graph capture + replay x3.
"""

import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4 as m
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

WARMUP = 10
ITERS = 100
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def make_case(M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_mlp = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02

    def gs(t):
        return float(t.abs().amax().float() / (448.0 * 6.0))

    def inv(s):
        return torch.tensor([1.0 / s], device=dev, dtype=torch.float32)

    s_mlp, s_wup, s_wdown = map(gs, (x_mlp, w_up, w_down))
    s_down_in = 0.05
    a_mlp, sf_mlp = fp.nvfp4_quantize(x_mlp, inv(s_mlp))
    q_up, sf_up = fp.nvfp4_quantize(w_up, inv(s_wup))
    q_down, sf_down = fp.nvfp4_quantize(w_down, inv(s_wdown))
    packed_args = (a_mlp, q_up, sf_mlp, sf_up, s_mlp, s_wup)
    packed_kw = {"output_inverse_scale": inv(s_down_in)}
    down_rest = (q_down, sf_down, s_down_in, s_wdown)
    return packed_args, packed_kw, down_rest


def tensors(args, kw):
    return [a for a in list(args) + list(kw.values()) if isinstance(a, torch.Tensor)]


def ref_packed(args, kw):
    """Protected reference: main's unmodified CuTe kernel."""
    return m.run("packed", *args, **kw)


def routed(args, kw):
    """The production entry (custom op) as this tree routes it."""
    return fp.nvfp4_swiglu_packed(*args, kw["output_inverse_scale"])


def cand(args, kw):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as c

    a, w, sfa, sfw, s_in, s_w = args
    alpha = m._scalar(a.device, s_in * s_w)
    return c.swiglu_packed(a, w, sfa, sfw, alpha, kw["output_inverse_scale"])


def down(data, scale, rest):
    q_down, sf_down, s_in, s_w = rest
    return torch.ops.trtllm.flux3_primsts_nvfp4_linear(data, q_down, scale, sf_down, s_in, s_w)


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(ITERS)
    ]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return stat([s.elapsed_time(e) for s, e in ev])


def has_cand():
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass  # noqa

        return True
    except ImportError:
        return False


def compare(M, seed, fn):
    args, kw, rest = make_case(M, seed=seed)
    rd, rs = ref_packed(args, kw)
    rd, rs = rd.clone(), rs.clone()
    cd, cs = fn(args, kw)
    torch.cuda.synchronize()
    out = {
        "data_shape": [list(cd.shape), list(rd.shape)],
        "scale_numel": [int(cs.numel()), int(rs.numel())],
    }
    if cd.shape == rd.shape and cs.shape == rs.shape:
        dm = cd != rd
        sm = cs != rs
        out["data_mismatch"] = int(dm.sum())
        out["scale_mismatch"] = int(sm.sum())
        if dm.any():
            idx = dm.nonzero()[:8].tolist()
            out["data_first"] = [[i, j, int(cd[i, j]), int(rd[i, j])] for i, j in idx]
            rows_bad = dm.any(dim=1).nonzero().flatten()
            cols_bad = dm.any(dim=0).nonzero().flatten()
            out["data_rows_bad"] = [int(rows_bad.numel()), rows_bad[:8].tolist()]
            out["data_cols_bad"] = [int(cols_bad.numel()), cols_bad[:16].tolist()]
        if sm.any():
            idx = sm.nonzero().flatten()[:8].tolist()
            out["scale_first"] = [[i, int(cs[i]), int(rs[i])] for i in idx]
        out["down_equal"] = bool(
            torch.equal(down(cd.contiguous(), cs, rest), down(rd.contiguous(), rs, rest))
        )
    return out


def div_test(mod, n=1 << 26):
    """Branch-free division vs div.rn.f32 on random operands (bitwise where div_safe)."""
    g = torch.Generator(device="cuda").manual_seed(123)
    out = {}
    cases = {
        # realistic: a = value*gate, b = 1 + ex2(.)
        "swiglu": (
            torch.sign(torch.rand(n, device="cuda", generator=g) - 0.5)
            * torch.exp2(torch.rand(n, device="cuda", generator=g) * 70 - 50),
            1.0 + torch.exp2(torch.rand(n, device="cuda", generator=g) * 80 - 40),
        ),
        "wide": (
            torch.sign(torch.rand(n, device="cuda", generator=g) - 0.5)
            * torch.exp2(torch.rand(n, device="cuda", generator=g) * 240 - 120),
            torch.exp2(torch.rand(n, device="cuda", generator=g) * 240 - 120),
        ),
        "bits": (
            torch.randint(0, 2**31 - 1, (n,), device="cuda", generator=g, dtype=torch.int32)
            .view(torch.float32),
            torch.randint(0, 2**31 - 1, (n,), device="cuda", generator=g, dtype=torch.int32)
            .view(torch.float32),
        ),
    }
    ok = True
    for name, (a, b) in cases.items():
        a = a.float().contiguous()
        b = b.float().contiguous()
        fast = torch.empty_like(a)
        rn = torch.empty_like(a)
        safe = torch.empty(n, device="cuda", dtype=torch.uint8)
        mod.flux3_div_test(a, b, fast, rn, safe)
        torch.cuda.synchronize()
        s = safe.bool()
        mism = int((fast.view(torch.int32)[s] != rn.view(torch.int32)[s]).sum())
        torch_eq = int((rn.view(torch.int32) != (a / b).view(torch.int32)).sum())
        out[name] = {"safe": int(s.sum()), "n": n, "fast_vs_rn_mismatch": mism,
                     "rn_vs_torch_mismatch": torch_eq}
        ok &= mism == 0
        log("div_test", name, out[name])
    return ok, out


def build():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as c

    mod = c.get_module()
    log("stages", mod.flux3_swiglu_fp4_stages(), "smem", mod.flux3_swiglu_fp4_smem_bytes())
    div_ok, div_res = div_test(mod)
    res = {}
    for M in (4176, 4096):
        res[M] = compare(M, M, cand)
        log(f"M={M}", res[M])
    # quick timing
    args, kw, _ = make_case(4176)
    a, w, sfa, sfw, s_in, s_w = args
    alpha = m._scalar(a.device, s_in * s_w)
    runner = fp._sm120_fp4_gemm_runner()
    rows = {}
    for name, fn in (
        ("ref_packed", lambda: ref_packed(args, kw)),
        ("cand", lambda: cand(args, kw)),
        ("cutlass_t2", lambda: runner.run_gemm(a, w, sfa, sfw, alpha, 0, 2, None, None)),
    ):
        med, noise = time_fn(fn)
        rows[name] = round(med * 1e3, 1)
        log(f"{name}: {med * 1e3:.1f} us noise {noise:.3f}")
    ok = div_ok and all(
        r.get("data_mismatch") == 0 and r.get("scale_mismatch") == 0 for r in res.values()
    )
    print(json.dumps({"passed": ok, "div_test": div_res,
                      "results": {str(k): v for k, v in res.items()}, "us": rows}))


def gpu_warmup(seconds=4.0):
    """Bring clocks to steady state before timing (round 0 of job 1013 drifted ~10%)."""
    import time

    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()


def bench():
    out = {}
    runner = fp._sm120_fp4_gemm_runner()
    gpu_warmup()
    cand_avail = has_cand()
    if cand_avail:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as c

        c.get_module()
    for M in (4176, 4096):
        args, kw, _ = make_case(M)
        pristine = [t.clone() for t in tensors(args, kw)]
        a, w, sfa, sfw, s_in, s_w = args
        alpha = m._scalar(a.device, s_in * s_w)
        calls = [
            ("routed_op", lambda: routed(args, kw)),
            ("run_packed_cute", lambda: ref_packed(args, kw)),
            ("cutlass_t2_bf16", lambda: runner.run_gemm(a, w, sfa, sfw, alpha, 0, 2, None, None)),
        ]
        if cand_avail:
            calls.append(("cand_cutlass_swiglu_fp4", lambda: cand(args, kw)))
        rows = {}
        for rnd in range(2):
            for name, fn in calls:
                med, noise = time_fn(fn)
                rows.setdefault(name, []).append((med, noise))
                log(f"M={M} round {rnd} {name}: {med * 1e3:.1f} us (noise {noise:.3f})")
        for p, t in zip(pristine, tensors(args, kw)):
            assert torch.equal(p, t), "input mutated"
        out[M] = rows
    med, noise = out[4176]["routed_op"][-1]
    print(
        json.dumps(
            {
                "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
                "table_us": {
                    str(M): {k: [round(x[0] * 1e3, 1) for x in v] for k, v in r.items()}
                    for M, r in out.items()
                },
                "noise": {
                    str(M): {k: [round(x[1], 4) for x in v] for k, v in r.items()}
                    for M, r in out.items()
                },
                "metric": "trtllm::flux3_primsts_nvfp4_swiglu_packed N=18432 K=3072 M=4176, "
                "unpadded activation, CUDA events median of 100",
                "output": {"class": "exact"},
            }
        )
    )


def check():
    results = {}
    passed = True
    for M in (4096, 4176, 4256):
        args, kw, rest = make_case(M, seed=M)
        pristine = [t.clone() for t in tensors(args, kw)]
        ref_data, ref_scale = ref_packed(args, kw)
        ref_data, ref_scale = ref_data.clone(), ref_scale.clone()
        ref_down = down(ref_data.contiguous(), ref_scale, rest).clone()
        for _ in range(WARMUP):
            routed(args, kw)
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, tensors(args, kw)))
        first = None
        n_ref = n_first = n_down = 0
        mism = 0
        for i in range(ITERS):
            data, scale = routed(args, kw)
            if first is None:
                first = (data.clone(), scale.clone())
            same = torch.equal(data, ref_data) and torch.equal(scale, ref_scale)
            n_ref += same
            if not same and data.shape == ref_data.shape and scale.shape == ref_scale.shape:
                mism = max(mism, int((data != ref_data).sum()) + int((scale != ref_scale).sum()))
            n_first += torch.equal(data, first[0]) and torch.equal(scale, first[1])
            if i < 3 or i == ITERS - 1:
                n_down += torch.equal(down(data.contiguous(), scale, rest), ref_down)
        inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, tensors(args, kw)))
        ref2 = ref_packed(args, kw)
        ref_stable = torch.equal(ref2[0], ref_data) and torch.equal(ref2[1], ref_scale)
        ok = (
            n_ref == ITERS
            and n_first == ITERS
            and n_down == 4
            and inputs_ok
            and ref_stable
            and tuple(first[0].shape) == tuple(ref_data.shape)
        )
        passed &= ok
        results[f"M{M}"] = {
            "packed_bitwise_vs_main_ref": f"{n_ref}/{ITERS}",
            "packed_bitwise_vs_iter0": f"{n_first}/{ITERS}",
            "down_bitwise_vs_main_ref": f"{n_down}/4",
            "max_mismatched_bytes": mism,
            "inputs_unchanged": inputs_ok,
            "ref_stable_ABA": ref_stable,
            "data_shape": list(first[0].shape),
            "scale_numel": int(first[1].numel()),
            "ok": ok,
        }
        log(f"M={M}: {results[f'M{M}']}")
    # CUDA graph capture / replay x3 at M=4176 and M=4096
    for M in (4176, 4096):
        args, kw, rest = make_case(M, seed=7 + M)
        rd, rs = ref_packed(args, kw)
        rd, rs = rd.clone(), rs.clone()
        routed(args, kw)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                gd, gs_ = routed(args, kw)
        graph_ok = True
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            graph_ok &= torch.equal(gd, rd) and torch.equal(gs_, rs)
        results[f"cuda_graph_replay_x3_bitwise_M{M}"] = graph_ok
        passed &= graph_ok
        log(f"cuda graph replay bitwise M={M}: {graph_ok}")
    import os

    results["route"] = (
        "cutlass" if has_cand() and os.environ.get("TRTLLM_FLUX3_SM120_SWIGLU_CUTLASS", "1") != "0"
        else "cute"
    )
    print(
        json.dumps(
            {
                "passed": bool(passed),
                "reference": "main's unmodified CuTe run('packed') (flux3_sm120_nvfp4.py) in-process, "
                "bitwise on PACKED data rows<M, full scale buffer incl. padded-row scales, and the "
                "down GEMM output",
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"build": build, "bench": bench, "check": check}[mode]()
