"""Untimed correctness + repeated-execution check for the SM120 BF16 cublas_mm range table (candidate tree).

Per shape (sliding/global qkv, sliding/global o_proj) and protocol M:
  * the exact call sequence of bench_pair.py (5 warmup + 20 timed calls, weight copy i % 4), untimed;
    every call's output is compared bitwise with the first call that used the same weight copy;
  * reference (b): FP32 torch.matmul of the same BF16 inputs; reference (a): main's op F.linear.
    Acceptance: cand error vs (b) (max-abs and rel-Frobenius) <= 1.10 x main's error + 1 BF16 ulp at the
    output magnitude; cand-vs-main max-abs difference reported;
  * the table config applied by C++ is checked by recomputing with ctypes cuBLASLt using the config parsed
    from cpp/tensorrt_llm/thop/cublasScaledMMLut.h (bitwise equal expected);
  * inputs unchanged after the sequence.
Workspace reuse: shape order A, B, A (and M order permuted) gives bitwise identical A outputs.
CUDA graphs: cublas_mm captured at decode M (1, 8, 64, 128) and prefill M (300, 8000) replays bitwise equal to eager.
Kernel names for selected cases from torch.profiler.

usage: python check.py <checkpoint>
"""
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt import Lt, read_shapes  # noqa: E402

PROTO_M = [300, 512, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
EXTRA_M = [27, 129, 600, 900, 999, 1100, 1152, 1500, 1800, 2100, 2400, 2700, 3000, 3600, 4097, 4500, 5100,
           5700, 6000, 6300, 6900, 7000, 7040, 7200, 7500, 7999, 8050, 8150]


def bits_equal(a, b):
    """Raw BF16 bit equality (torch.equal would treat -0.0 == +0.0)."""
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))
NCOPY = 4
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def parse_table(path):
    txt = open(path).read()
    body = txt[txt.index("inline const RangeAlgoListType sm120_bf16_range_algo_list"):]
    body = body[:body.index("\n};")]
    tables = {}
    for blk in re.finditer(r"\{\{(\d+), (\d+)\},\s*\n\s*\{\s*\n(.*?)\n\s*\}\},", body, re.S):
        k, n = int(blk.group(1)), int(blk.group(2))
        ents = [(int(m), tuple(int(v) for v in cfg.split(","))) for m, cfg in
                re.findall(r"\{(\d+), \{([-\d, ]+)\}\}", blk.group(3))]
        tables[(k, n)] = ents
    return tables


def lookup(ents, m):
    for m_max, cfg in ents:
        if m <= m_max:
            return cfg if cfg[0] >= 0 else None
    return None


def bf16_ulp(x):
    # 1 BF16 ulp at magnitude x: 2^(floor(log2 x) - 7)
    return 2.0 ** (torch.floor(torch.log2(torch.tensor(float(x)))).item() - 7)


def main():
    ckpt = sys.argv[1]
    shapes, info = read_shapes(ckpt)
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.models.modeling_gemma4 as mg
    import inspect
    from tensorrt_llm._utils import get_sm_version
    gemma_flag = "use_custom_cublas_mm = True" in inspect.getsource(mg.Gemma4Attention.__init__)
    routes = {k: mg._CublasMmTokenGate(v) for k, v in mg._SM120_BF16_CUBLAS_MM_ROUTES.items()}
    log("sm", get_sm_version(), "gemma4 sets cublas_mm:", gemma_flag,
        {str(k): list(v) for k, v in mg._SM120_BF16_CUBLAS_MM_ROUTES.items()}, json.dumps(info))
    wt = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    tables = parse_table(os.path.join(wt, "cpp/tensorrt_llm/thop/cublasScaledMMLut.h"))
    log("parsed table keys", {str(k): len(v) for k, v in tables.items()})
    # Every token count Gemma4 routes to cublas_mm must hit a pinned (non-heuristic) table entry.
    consistent = set(routes) == set(tables)
    for key, rs in mg._SM120_BF16_CUBLAS_MM_ROUTES.items():
        for lo, hi in rs:
            for m in range(lo, hi + 1):
                if lookup(tables.get(key, []), m) is None:
                    consistent = False
                    log(f"!! route {key} m={m} has no pinned table entry")
                    break
    log("routes/table consistent:", consistent)
    lt = Lt()

    def cand(x, W):
        return torch.ops.trtllm.cublas_mm(x, W.t(), None, None)

    ok = consistent
    report = {}
    n_routed_cases = 0
    n_routed_bitwise = 0
    worst = {"err_ratio_maxabs": 0.0, "err_ratio_frob": 0.0, "cand_vs_main_maxabs": 0.0}
    a_outputs = {}
    for sname, (N, K) in shapes.items():
        ents = tables.get((K, N))
        if ents is None:
            log(f"!! no table for {sname} (K={K}, N={N})")
            ok = False
        torch.manual_seed(0)
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NCOPY)]
        xmax = torch.randn(max(PROTO_M), K, device="cuda", dtype=torch.bfloat16)
        W_snap = [w.clone() for w in Ws]
        x_snap = xmax.clone()
        for M in PROTO_M + EXTRA_M:
            x = xmax[:M]
            first = {}
            nmis = 0
            for i in range(25):
                o = cand(x, Ws[i % NCOPY])
                j = i % NCOPY
                if j not in first:
                    first[j] = o.clone()
                elif not bits_equal(o, first[j]):
                    nmis += 1
            o = first[0]
            ref = torch.matmul(x.float(), Ws[0].float().t())
            main = F.linear(x, Ws[0])
            e_c = (o.float() - ref).abs().max().item()
            e_m = (main.float() - ref).abs().max().item()
            f_c = ((o.float() - ref).norm() / ref.norm()).item()
            f_m = ((main.float() - ref).norm() / ref.norm()).item()
            ulp = bf16_ulp(ref.abs().max().item())
            d_cm = (o.float() - main.float()).abs().max().item()
            pass_max = e_c <= 1.10 * e_m + ulp
            pass_frob = f_c <= 1.10 * f_m + ulp / max(ref.abs().max().item(), 1e-30)
            finite = bool(torch.isfinite(o).all())
            # the C++ table config, recomputed with ctypes cuBLASLt
            cfg = lookup(ents, M) if ents else None
            if cfg is not None:
                out = torch.empty_like(o)
                lt.matmul(x, Ws[0], out, lt.make_algo(cfg))
                table_bitwise = bits_equal(out, o)
            else:
                out = torch.empty_like(o)
                lt.matmul(x, Ws[0], out, None)
                table_bitwise = bits_equal(out, o)
            gate = routes.get((K, N))
            routed = gate is not None and M in gate
            # what the model runs at this M: cublas_mm if routed, else main's F.linear
            main_bits = bits_equal(o, main) and all(
                bits_equal(cand(x, Ws[j]), F.linear(x, Ws[j])) for j in (1, 2))
            n_bits_diff = int((o.view(torch.int16) != main.view(torch.int16)).sum().item())
            case_ok = (nmis == 0 and pass_max and pass_frob and finite and table_bitwise
                       and o.dtype == torch.bfloat16 and (main_bits or not routed))
            ok &= case_ok
            worst["err_ratio_maxabs"] = max(worst["err_ratio_maxabs"], e_c / max(e_m, 1e-30))
            worst["err_ratio_frob"] = max(worst["err_ratio_frob"], f_c / max(f_m, 1e-30))
            worst["cand_vs_main_maxabs"] = max(worst["cand_vs_main_maxabs"], d_cm)
            report[f"{sname}_M{M}"] = dict(repeat_mismatches=nmis, cand_maxabs=e_c, main_maxabs=e_m, cand_frob=f_c,
                                           main_frob=f_m, cand_vs_main_maxabs=d_cm, table_cfg=cfg,
                                           table_bitwise=table_bitwise, routed=routed, bitwise_vs_main=main_bits,
                                           bits_diff=n_bits_diff, ok=case_ok)
            if routed:
                n_routed_cases += 1
                n_routed_bitwise += int(main_bits)
            log(f"{sname} M={M}: routed={routed} bitwise_vs_main={main_bits} ({n_bits_diff} elems differ) "
                f"repeat_mismatch={nmis} maxabs cand {e_c:.3e} main {e_m:.3e} | frob cand {f_c:.3e} main {f_m:.3e} | "
                f"cfg {cfg} table_bitwise={table_bitwise} -> {'OK' if case_ok else 'FAIL'}")
        inputs_ok = all(torch.equal(a, b) for a, b in zip(Ws, W_snap)) and torch.equal(xmax, x_snap)
        ok &= inputs_ok
        log(f"{sname} inputs unchanged: {inputs_ok}")
        # first-pass A outputs for the A,B,A permutation check
        a_outputs[sname] = {M: cand(xmax[:M], Ws[0]).clone() for M in (300, 1200, 8000)}
        del W_snap, x_snap
        torch.cuda.empty_cache()
        a_outputs[sname]["_inputs"] = (Ws[0], xmax)
    # A, B, A with permuted M order: workspace carries no state
    perm_ok = True
    names = list(shapes)
    for a, b in [(names[0], names[2]), (names[1], names[3]), (names[2], names[0])]:
        Wa, xa = a_outputs[a]["_inputs"]
        Wb, xb = a_outputs[b]["_inputs"]
        _ = cand(xb[:4000], Wb)
        for M in (8000, 300, 1200):
            o = cand(xa[:M], Wa)
            perm_ok &= bits_equal(o, a_outputs[a][M])
            _ = cand(xb[:M], Wb)
    ok &= perm_ok
    log("A,B,A permutation bitwise:", perm_ok)
    # CUDA-graph capture/replay at decode and prefill M
    graph_ok = True
    Wq, xq = a_outputs[names[0]]["_inputs"]
    Wo, xo = a_outputs[names[2]]["_inputs"]
    for M in (1, 8, 64, 128, 300, 8000):
        for W, xs in ((Wq, xq), (Wo, xo)):
            xin = xs[:M].clone()
            eager = cand(xin, W).clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    cand(xin, W)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = cand(xin, W)
            outs = []
            for _ in range(3):
                g.replay()
                torch.cuda.synchronize()
                outs.append(gout.clone())
            good = all(bits_equal(o, eager) for o in outs)
            graph_ok &= good
            log(f"cuda graph M={M} N={W.shape[0]} K={W.shape[1]}: replay==eager {good}")
    ok &= graph_ok
    # Routing through the real UnquantizedLinearMethod.apply with Gemma4's SM120 flags:
    # in-table shapes and M in range -> cublas_mm; other M -> F.linear (main's dispatch, bitwise).
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod
    meth = UnquantizedLinearMethod()
    route_ok = True
    for sname in names:
        W, xs = a_outputs[sname]["_inputs"]
        N, K = W.shape
        gate = routes.get((K, N))
        mod = SimpleNamespace(weight=W, use_custom_cublas_mm=gate is not None, custom_cublas_mm_tokens=gate,
                              use_cute_dsl_bf16_gemm=False, all_reduce=None, mapping=None)
        for M in (1, 8, 64, 127, 128, 299, 300, 512, 1000, 1279, 1280, 4000, 8000, 8192):
            o = meth.apply(mod, xs[:M], None)
            inside = gate is not None and M in gate
            exp = cand(xs[:M], W) if inside else F.linear(xs[:M], W)
            good = bits_equal(o, exp) and bits_equal(o, F.linear(xs[:M], W))
            route_ok &= good
            log(f"route {sname} M={M}: {'cublas_mm' if inside else 'F.linear'} bitwise {good}")
    # A shape absent from the table (K=4096, N=6144): Gemma4 would not flag it, so apply() runs F.linear exactly as
    # main; and trtllm::cublas_mm itself falls through to main's dispatch (no table hit -> cuBLASLt heuristic).
    torch.manual_seed(1)
    Wz = torch.randn(6144, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    xz = torch.randn(1000, 4096, device="cuda", dtype=torch.bfloat16)
    gz = routes.get((4096, 6144))
    modz = SimpleNamespace(weight=Wz, use_custom_cublas_mm=gz is not None, custom_cublas_mm_tokens=gz,
                           use_cute_dsl_bf16_gemm=False, all_reduce=None, mapping=None)
    oz = meth.apply(modz, xz, None)
    absent_flinear = bits_equal(oz, F.linear(xz, Wz)) and not modz.use_custom_cublas_mm
    outz = torch.empty(1000, 6144, device="cuda", dtype=torch.bfloat16)
    lt.matmul(xz, Wz, outz, None)
    absent_cublas_fallthrough = bits_equal(cand(xz, Wz), outz)
    kz = {}
    try:
        from torch.profiler import ProfilerActivity, profile
        for tag, fn in (("apply", lambda: meth.apply(modz, xz, None)), ("F.linear", lambda: F.linear(xz, Wz))):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            kz[tag] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    except Exception as e:  # noqa: BLE001
        log("profiler failed", e)
    same_kernel = kz.get("apply") == kz.get("F.linear")
    log(f"absent shape K=4096 N=6144: apply()==F.linear bitwise {absent_flinear}, same kernel {same_kernel} {kz}; "
        f"cublas_mm == cuBLASLt heuristic (main's cublas_mm dispatch) {absent_cublas_fallthrough}")
    route_ok &= absent_flinear and absent_cublas_fallthrough and same_kernel
    ok &= route_ok
    # kernel names (which nvjet tile runs) for a few cases
    kn = {}
    try:
        from torch.profiler import ProfilerActivity, profile
        for sname in ("sl_o", "gl_o", "sl_qkv", "gl_qkv"):
            W, xs = a_outputs[sname]["_inputs"]
            for M in (300, 1200, 8000):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    cand(xs[:M], W)
                    torch.cuda.synchronize()
                names_k = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
                kn[f"{sname}_M{M}"] = names_k
                log(f"kernels {sname} M={M}: {names_k}")
    except Exception as e:  # noqa: BLE001
        log("profiler failed", e)
    log("worst", json.dumps(worst))
    print(json.dumps({"passed": bool(ok), "compared": "trtllm::cublas_mm (SM120 range table) vs FP32 torch.matmul "
                      "reference and main F.linear; 25-call repeated sequence bitwise; A,B,A permutation; CUDA graph "
                      "replay; table config recomputed via cuBLASLt; Linear routing and absent-shape fall-through",
                      "graph_ok": graph_ok, "perm_ok": perm_ok, "route_ok": route_ok,
                      "routes_table_consistent": consistent,
                      "worst": worst, "n_cases": len(report), "routed_cases": n_routed_cases,
                      "routed_bitwise_vs_main": n_routed_bitwise,
                      "failed": [k for k, v in report.items() if not v["ok"]]}))


if __name__ == "__main__":
    main()
