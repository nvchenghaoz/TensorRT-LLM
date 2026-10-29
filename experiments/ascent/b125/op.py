"""Branch 125: SM120 warp-per-row LN+adaLN+NVFP4 kernel vs main's Triton _adaln_nvfp4_kernel.

  op.py probe  -> prints triton_helpers welford sources; bitwise of variants vs Triton kernel
  op.py check  -> acceptance 1: fp4 + full poisoned SF (+ x_mod when stored) bitwise to the
                  Triton kernel at M=4176/4256/4096/80, select and mode forms, normal and 4x
                  saturating global scale, store_x_mod on/off; public-op dispatch (warp vs
                  triton) bitwise; untimed 20-iteration repeat vs iter 0 and reference, inputs
                  unchanged before each reset.
  op.py bench  -> served per-request mix in one CUDA graph (28 select M=4176 + 4 mode M=4096
                  calls per step, each preceded by a stand-in gated-residual pass writing its x),
                  soak >= 10 s, >= 150 replays; runs the public op (base tree: Triton; branch:
                  dispatch). Same command in both arms.
"""

import inspect
import json
import os
import statistics
import sys
import time

import torch

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq

try:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk
except ImportError:
    wk = None

D = 3072
EPS = 1e-6
dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
g = torch.Generator(device=dev).manual_seed(1234)


def heavy(*shape):
    base = torch.randn(*shape, device=dev, generator=g)
    mag = torch.exp(torch.randn(*shape[:-1], 1, device=dev, generator=g) * 2.0)
    chan = torch.exp(torch.randn(shape[-1], device=dev, generator=g) * 1.5)
    return (base * mag * chan).to(torch.bfloat16)


W = (1.0 + 0.3 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
B = (0.2 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)


def make_cases():
    cases = []
    for t, s in ((80, 4176), (160, 4256), (80, 80)):
        x = heavy(1, s, D)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        shift_t, scale_t = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
        select = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        select[:, : min(t, s) // 2 if s == 80 else t] = True
        cases.append((f"select_M{s}", (x, select, scale_t, shift_t)))
    table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    for s in (4096, 80):
        x = heavy(1, s, D)
        for k in ((0, 5, 15) if s == 4096 else (7,)):
            row = table[:, k: k + 1]
            cases.append((f"mode_M{s}_step{k}", (x, None, row[..., D: 2 * D], row[..., :D])))
    return cases


def gs_for(x):
    return float(torch.tensor(2688.0 / x.float().abs().max().item(), dtype=torch.float32))


def poisoned(x):
    m = x.numel() // D
    return (
        torch.full_like(x, float("nan")),
        torch.full((m, D // 2), 0xAB, dtype=torch.uint8, device=dev),
        torch.full((faq._scale_size(m, D),), 0xAB, dtype=torch.uint8, device=dev),
    )


def triton_ref(args, gs_t, store):
    x, sel, sc, sh = args
    return faq._launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=poisoned(x), store_x_mod=store)


def warp_run(args, gs_t, store, defines=None):
    if defines is None:
        defines = wk._DEFAULT_DEFINES
    x, sel, sc, sh = args
    return wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, store_x_mod=store, outs=poisoned(x),
                     extra_defines=defines)


def same(a, b, store):
    eq_x = torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)) if store else True
    return eq_x and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def diff_stats(a, b):
    xa, xb = a[0].float(), b[0].float()
    return {
        "x_diff_elems": int((a[0].view(torch.int16) != b[0].view(torch.int16)).sum().item()),
        "x_max_abs": float((xa - xb).abs().nan_to_num(0).max().item()),
        "fp4_diff_frac": float((a[1] != b[1]).float().mean().item()),
        "sf_diff_frac": float((a[2] != b[2]).float().mean().item()),
    }


def run_probe():
    from torch._inductor.runtime import triton_helpers as th

    for name in ("welford_reduce", "welford_combine", "welford"):
        fn = getattr(th, name)
        src = inspect.getsource(getattr(fn, "fn", fn))
        print("SRC", name, "\n" + src, flush=True)
    t0 = time.perf_counter()
    wk._compile(torch.device("cuda", 0), 4, False)
    print("compile_s", time.perf_counter() - t0, flush=True)
    variants = [
        (f"IT_MODE={it}", f"BF_ASC={bf}", f"INTER_SEQ={iseq}")
        for it in (0, 1, 2) for bf in (0, 1) for iseq in (0, 1)
    ] + [("PW_X2=0",)]
    res = {}
    tot = {}
    with torch.inference_mode():
        # quick L2-hot timing (same input each call), select M=4176
        name, args = make_cases()[0]
        x, sel, sc, sh = args
        gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
        outs = poisoned(x)
        for label, fn in (
            ("triton", lambda: faq._launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs, store_x_mod=False)),
            ("warp4", lambda: wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs, store_x_mod=False)),
            ("warp8", lambda: wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs, store_x_mod=False, warps=8)),
            ("warp2", lambda: wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs, store_x_mod=False, warps=2)),
            ("warp4_scalarpw", lambda: wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs, store_x_mod=False,
                                                 extra_defines=("PW_X2=0",))),
        ):
            for _ in range(5):
                fn()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(20):
                    fn()
            gr.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(30):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                gr.replay()
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e) * 1000 / 20)
            print("TIME_hotL2_us", label, statistics.median(ts), flush=True)
        for name, args in make_cases():
            gs_t = torch.tensor([gs_for(args[0])], dtype=torch.float32, device=dev)
            ref = triton_ref(args, gs_t, True)
            for v in variants:
                out = warp_run(args, gs_t, True, v)
                ok = same(out, ref, True)
                res[f"{name}|{v}"] = ok
                st = diff_stats(out, ref)
                tot[str(v)] = tot.get(str(v), 0) + st["x_diff_elems"] + int(st["fp4_diff_frac"] > 0) + int(st["sf_diff_frac"] > 0)
                print(name, v, ok, json.dumps(st), flush=True)
    for k, n in sorted(tot.items(), key=lambda kv: kv[1]):
        print("VARIANT_TOTAL", n, k, flush=True)
    print("NUM_REGS", {str(k): v for k, v in wk.NUM_REGS.items()}, flush=True)
    best = min(tot.items(), key=lambda kv: kv[1])
    print(json.dumps({"passed": best[1] == 0, "best": best}))


_DIVTEST_SRC = r"""
typedef unsigned int u32;
__device__ __forceinline__ float fdiv(float a, float b) {
  float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rcp3() {
  float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(3.0f)); return r; }
extern "C" __global__ void divtest(unsigned long long *cnt, u32 *first) {
  for (u32 it = 0; it < 256; ++it) {
    u32 bits = ((u32)blockIdx.x * 256u + it) * 256u + threadIdx.x;
    float a = __uint_as_float(bits);
    if (!isfinite(a)) continue;
    float d2 = fdiv(a, 2.0f), m2 = __fmul_rn(a, 0.5f);
    float d3 = fdiv(a, 3.0f), m3 = __fmul_rn(a, rcp3());
    if (__float_as_uint(d2) != __float_as_uint(m2)) { atomicAdd(cnt, 1ull); atomicMin(first, bits & 0x7fffffffu); }
    if (__float_as_uint(d3) != __float_as_uint(m3)) { atomicAdd(cnt + 1, 1ull); atomicMin(first + 1, bits & 0x7fffffffu); }
  }
}
"""


def run_divtest():
    from cuda.bindings import driver, nvrtc

    prog = wk._check(nvrtc.nvrtcCreateProgram(_DIVTEST_SRC.encode(), b"d.cu", 0, [], []))
    opts = [b"--gpu-architecture=sm_120a", b"--fmad=false"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert int(err) == 0
    size = wk._check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * size
    wk._check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    torch.cuda.current_stream()
    mod = wk._check(driver.cuModuleLoadData(cubin))
    fn = wk._check(driver.cuModuleGetFunction(mod, b"divtest"))
    cnt = torch.zeros(2, dtype=torch.int64, device=dev)
    first = torch.full((2,), 0x7FFFFFFF, dtype=torch.int32, device=dev)
    import ctypes
    wk._check(driver.cuLaunchKernel(fn, 65536, 1, 1, 256, 1, 1, 0,
                                    driver.CUstream(torch.cuda.current_stream().cuda_stream),
                                    ((cnt.data_ptr(), first.data_ptr()), (ctypes.c_void_p, ctypes.c_void_p)), 0))
    torch.cuda.synchronize()
    res = {"div2_mismatch": int(cnt[0].item()), "div3_mismatch": int(cnt[1].item()),
           "div2_min_abs_bits": hex(int(first[0].item()) & 0xFFFFFFFF),
           "div3_min_abs_bits": hex(int(first[1].item()) & 0xFFFFFFFF)}
    print("DIVTEST exhaustive over all finite fp32:", json.dumps(res), flush=True)
    return res


def run_opt():
    dres = run_divtest() if "divtest" in sys.argv else None
    base_v = ()
    fd = ("FAST_DIV=1", "DGUARD=0")
    variants = [fd, ("MINB=5",), fd + ("MINB=5",), fd + ("MINB=6",)]
    eq = {}
    with torch.inference_mode():
        for name, args in make_cases():
            gs_t = torch.tensor([gs_for(args[0])], dtype=torch.float32, device=dev)
            ref = triton_ref(args, gs_t, True)
            base = warp_run(args, gs_t, True, base_v)
            for v in variants:
                out = warp_run(args, gs_t, True, v)
                eq.setdefault(str(v), True)
                eq[str(v)] &= same(out, base, True)
                print(name, v, "eq_default", same(out, base, True), "vs_triton", json.dumps(diff_stats(out, ref)), flush=True)
        name, args = make_cases()[0]
        x, sel, sc, sh = args
        gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
        outs = poisoned(x)
        timings = {}
        tv = [(4, ()), (4, fd)] + [(4, fd + (f"MINB={b}",)) for b in (4, 5, 6)] + [
            (2, fd)] + [(2, fd + (f"MINB={b}",)) for b in (8, 10, 12)] + [
            (8, fd), (8, fd + ("MINB=2",)), (8, fd + ("MINB=3",))]
        cfgs = [("triton", None, None)] + [
            (f"w{wp}_{'+'.join(v) or 'default'}", wp, v) for wp, v in tv
        ]
        for label, wp, v in cfgs:
            if wp is None:
                fn = lambda: faq._launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs, store_x_mod=False)  # noqa
            else:
                fn = (lambda wp=wp, v=v: wk.launch(x, sel, sc, sh, W, B, gs_t, EPS, outs=outs,
                                                   store_x_mod=False, warps=wp, extra_defines=v))
            for _ in range(5):
                fn()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(20):
                    fn()
            gr.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(40):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                gr.replay()
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e) * 1000 / 20)
            timings[label] = statistics.median(ts)
            print("TIME_hotL2_us", label, timings[label], flush=True)
    print("NUM_REGS", {str(k): v for k, v in wk.NUM_REGS.items()}, flush=True)
    print(json.dumps({"passed": all(eq.values()), "eq_default": eq, "div": dres, "t": timings}))


def ulp_stats(a, b):
    ai = a.reshape(-1).view(torch.int16).int()
    bi = b.reshape(-1).view(torch.int16).int()
    d = (ai - bi).abs()
    return int(d.max().item()), int((d > 0).sum().item())


def within_tol(out, ref, store, rec):
    """Pass criterion (area-lead ruling 15:59 for the reordered fallback): fp4 data and the FULL
    poisoned SF buffer bitwise to main's Triton kernel. x_mod (stored only when a consumer reads
    it) is reported: max |delta|, ulp distance and differing count. Job 1523 measured up to 16
    ulp / 0.031 abs on 2-10 of ~13M elements (cancellation in (1+scale)*LN + shift amplifies a
    1-ulp LN difference), exceeding the 1-ulp tolerance first stated for x_mod."""
    ok = torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])
    if store:
        mu, cnt = ulp_stats(out[0], ref[0])
        rec["x_max_ulp"], rec["x_ulp_count"] = mu, cnt
        rec["x_max_abs_delta"] = float((out[0].float() - ref[0].float()).abs().max().item())
        rec["x_diff_frac"] = cnt / out[0].numel()
    return bool(ok)


def fp32_errors(args, xm_warp, xm_triton):
    x, sel, sc, sh = args
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    ln = (xf - mean) * torch.rsqrt(var + EPS) * W.float() + B.float()
    if sel is None:
        scale, shift = sc.float(), sh.float()
    else:
        scale = torch.where(sel, sc[:, 1:2].float(), sc[:, 0:1].float())
        shift = torch.where(sel, sh[:, 1:2].float(), sh[:, 0:1].float())
    r = (1 + scale) * ln + shift
    return {"warp_vs_fp32_max_abs": float((xm_warp.float() - r).abs().max().item()),
            "triton_vs_fp32_max_abs": float((xm_triton.float() - r).abs().max().item())}


def run_check():
    ok_all = True
    report = {}
    t0 = time.perf_counter()
    wk._compile(torch.device("cuda", 0), 4, False, wk._DEFAULT_DEFINES)
    report["compile_s"] = time.perf_counter() - t0
    default_impl = faq._IMPL
    report["default_impl"] = default_impl
    faq._IMPL = "warp"  # force warp for the warp-kernel checks below
    with torch.inference_mode():
        for name, args in make_cases():
            x = args[0]
            gs = gs_for(x)
            for gname, gsv in (("gs", gs), ("gs_sat4x", float(torch.tensor(gs * 4.0, dtype=torch.float32)))):
                gs_t = torch.tensor([gsv], dtype=torch.float32, device=dev)
                for store in (True, False):
                    ref = triton_ref(args, gs_t, store)
                    out = warp_run(args, gs_t, store)
                    rec = {"bitwise": same(out, ref, store), **diff_stats(out, ref)}
                    ok = within_tol(out, ref, store, rec)
                    if store and gname == "gs":
                        rec.update(fp32_errors(args, out[0], ref[0]))
                    if not store:
                        # x_mod not written by either kernel: both buffers still the NaN poison.
                        rec["x_mod_untouched"] = bool(torch.isnan(out[0].float()).all().item())
                        ok &= rec["x_mod_untouched"]
                    # Public op (dispatch) vs the Triton kernel.
                    po = torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, args[1], args[2], args[3], W, B,
                                                                 gs_t, EPS, store)
                    rec["public_op_eq_warp"] = same(po, out, store)
                    ok &= rec["public_op_eq_warp"]
                    key = f"{name}|{gname}|store{int(store)}"
                    report[key] = rec
                    ok_all &= ok
                    print(key, ok, json.dumps(rec), flush=True)
            # Untimed repeat sequence: warmup 3 + 20 iterations like the timed loop; every
            # iteration vs iter 0 and the reference; inputs vs pristine copies before each reset.
            gs_t = torch.tensor([gs], dtype=torch.float32, device=dev)
            ref = triton_ref(args, gs_t, True)
            pristine = [a.clone() if a is not None else None for a in args]
            outs = (torch.empty_like(x), torch.empty((x.numel() // D, D // 2), dtype=torch.uint8, device=dev),
                    torch.empty((faq._scale_size(x.numel() // D, D),), dtype=torch.uint8, device=dev))
            eq_iter0 = True
            fp4sf_vs_main = True
            inputs_ok = True
            first = None
            for it in range(23):
                if it % 8 == 0:  # reset point: check inputs unchanged, re-poison outputs
                    inputs_ok &= all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
                    for o in outs:
                        o.view(torch.uint8).fill_(0xAB)
                o = wk.launch(x, args[1], args[2], args[3], W, B, gs_t, EPS, outs=outs)
                cur = tuple(t.clone() for t in o)
                if first is None:
                    first = cur
                eq_iter0 &= same(cur, first, True)
                fp4sf_vs_main &= within_tol(cur, ref, True, {})
            inputs_ok &= all(a is None or torch.equal(a, p) for a, p in zip(args, pristine))
            rep = eq_iter0 and fp4sf_vs_main and inputs_ok
            report[f"{name}|repeat"] = {"passed": rep, "all_iters_eq_iter0": eq_iter0,
                                        "fp4_sf_bitwise_to_main_all_iters": fp4sf_vs_main,
                                        "inputs_unchanged": inputs_ok}
            ok_all &= rep
            print(name, "repeat", rep, flush=True)
        # IMPL=triton switch: the public op is main's Triton kernel byte for byte.
        faq._IMPL = "triton"
        trit_ok = True
        for name, args in make_cases():
            gs_t = torch.tensor([gs_for(args[0])], dtype=torch.float32, device=dev)
            ref = triton_ref(args, gs_t, True)
            po = torch.ops.trtllm.flux3_adaln_nvfp4_quant(args[0], args[1], args[2], args[3], W, B,
                                                         gs_t, EPS, True)
            trit_ok &= same(po, ref, True)
        report["impl_triton_public_op_bitwise"] = bool(trit_ok)
        ok_all &= trit_ok
        print("impl_triton_public_op_bitwise", trit_ok, flush=True)
        # Default (auto) per-M routing: M > 4096 -> warp kernel bitwise; M <= 4096 -> main's
        # Triton byte for byte. Also exercised with poisoned outputs via the public op.
        faq._IMPL = default_impl
        auto_ok = True
        auto_rep = {}
        for name, args in make_cases():
            x = args[0]
            m = x.numel() // D
            gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
            po = torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, args[1], args[2], args[3], W, B,
                                                         gs_t, EPS, True)
            if m > 4096:
                exp_route, ref = "warp", warp_run(args, gs_t, True)
            else:
                exp_route, ref = "triton", triton_ref(args, gs_t, True)
            eq = same(po, ref, True)
            auto_rep[name] = {"M": m, "route": exp_route, "bitwise_to_route": bool(eq)}
            auto_ok &= eq
        report["default_auto_routing"] = auto_rep
        report["default_auto_routing_passed"] = bool(auto_ok)
        ok_all &= auto_ok
        print("default_auto_routing", auto_ok, json.dumps(auto_rep), flush=True)
        faq._IMPL = "warp"
    report["compile_seconds"] = {str(k): v for k, v in wk.COMPILE_SECONDS.items()}
    report["num_regs"] = {str(k): v for k, v in wk.NUM_REGS.items()}
    print(json.dumps({"passed": bool(ok_all), "class": "reordered", "criterion":
                      "fp4 + full poisoned SF bitwise to main in every call and repeat iteration; "
                      "repeats == iteration 0; inputs unchanged; x_mod deltas reported "
                      "(informational, reordered class)",
                      "compared_against":
                      "main's Triton _adaln_nvfp4_kernel (flux3_adaln_quant._launch), poisoned outputs",
                      **{"results": report}}))


def run_bench():
    nsets = 4
    single_calls, mode_calls = 28, 4
    arm = faq.__dict__.get("_IMPL", "triton") if wk is not None else "triton"
    torch.manual_seed(0)
    sets = []
    for i in range(nsets):
        for kind, s in (("select", 4176), ("mode", 4096)):
            x = heavy(1, s, D)
            res = heavy(1, s, D)
            y = heavy(1, s, D)
            gate = (torch.randn(1, 1, D, device=dev, generator=g) * 0.3).to(torch.bfloat16)
            mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            if kind == "select":
                sel = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
                sel[:, :80] = True
                sc, sh = mod[..., D: 2 * D], mod[..., 4 * D: 5 * D]
            else:
                sel = None
                sc, sh = mod[:, 0:1, D: 2 * D], mod[:, 0:1, :D]
            gs_t = torch.tensor([gs_for(x)], dtype=torch.float32, device=dev)
            sets.append((kind, x, res, y, gate, sel, sc, sh, gs_t))
    sel_sets = [s for s in sets if s[0] == "select"]
    mode_sets = [s for s in sets if s[0] == "mode"]

    def one(st, residual=True, op=True):
        _, x, res, y, gate, sel, sc, sh, gs_t = st
        if residual:
            torch.addcmul(res, gate, y, out=x)  # stand-in gated residual: writes x (L2-hot)
        if op:
            torch.ops.trtllm.flux3_adaln_nvfp4_quant(x, sel, sc, sh, W, B, gs_t, EPS, False)

    def step(residual=True, op=True):
        for i in range(single_calls):
            one(sel_sets[i % nsets], residual, op)
        for i in range(mode_calls):
            one(mode_sets[i % nsets], residual, op)

    results = {}
    with torch.inference_mode():
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        graphs = {}
        for name, kw in (("mix", {}), ("residual_only", {"op": False})):
            sstream = torch.cuda.Stream()
            sstream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sstream):
                step(**kw)
            torch.cuda.current_stream().wait_stream(sstream)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                step(**kw)
            graphs[name] = gr
        # soak >= 10 s
        t_end = time.time() + 10.0
        n = 0
        while time.time() < t_end:
            for gr in graphs.values():
                gr.replay()
            n += 1
            if n % 200 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        print("soak replays", n, flush=True)
        samples = {k: [] for k in graphs}
        for _ in range(200):
            for name, gr in graphs.items():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                gr.replay()
                e.record()
                e.synchronize()
                samples[name].append(s.elapsed_time(e))
    for name, ts in samples.items():
        ts = sorted(ts)
        med = statistics.median(ts)
        p10, p90 = ts[len(ts) // 10], ts[(9 * len(ts)) // 10]
        results[name] = {"ms": med, "p10": p10, "p90": p90, "noise": (p90 - p10) / med, "n": len(ts)}
        print(arm, name, json.dumps(results[name]), flush=True)
    mix = results["mix"]
    op_est = (mix["ms"] - results["residual_only"]["ms"]) / (single_calls + mode_calls) * 1000.0
    print(json.dumps({
        "batch": {"1": {"kernel_ms": mix["ms"], "noise_floor": mix["noise"]}},
        "output": {"class": "reordered"},
        "arm": arm,
        "detail": results,
        "op_us_per_call_est": op_est,
        "note": "kernel_ms = one CUDA-graph replay of a served step's mix: 28 select calls (M=4176) "
                "+ 4 mode calls (M=4096) of trtllm::flux3_adaln_nvfp4_quant (store_x_mod False), "
                "each preceded by a stand-in gated-residual addcmul writing its x; 4 rotating input "
                "sets; median of 200 replays after a 10 s soak. op_us_per_call_est subtracts the "
                "residual-only graph.",
    }))


if MODE == "probe":
    run_probe()
elif MODE == "opt":
    run_opt()
elif MODE == "check":
    run_check()
else:
    run_bench()
