"""CSA2 small-N BF16 Linears at decode: cuBLAS (F.linear) vs the Triton small-N GEMV.

Per-rank shapes (DeepSeek-V4.1-Flash TP4; every rank holds full copies):
  index_weights_proj [32, 5120] (FULL + REINDEX), compressor wkv [512, 5120]
  (ratio 1) / wkv_gate [1024, 5120] (ratio 2) (FULL), index_wk [128, 512] (FULL).

Modes (argv[1]; argv[2] = checkpoint dir for the layer census):
  bench  - (pair) two-stream window per FULL / REINDEX layer: landing 8's fused
           main-stream projections (DeepseekV41Attention._project_inputs, b12x MXFP8)
           on main, the aux chain (compressor Linear -> RMSNorm -> index_wk ->
           index_weights_proj -> scale mul) on an aux stream, joined; cold-L2 CUDA
           graph. kernel_ms = per decode step (n_full * full + n_reindex * reindex).
           Also isolated per-shape cold-L2 graph timings. The tree's
           ``small_n_linear`` is used if it exists, else the plain Linear (cuBLAS).
  sweep  - per-shape cold-L2 timings of cuBLAS vs several GEMV configs.
  check  - numerics vs FP32 reference and vs cuBLAS, determinism, CUDA-graph
           capture/replay (single- and two-stream) at M=1,2,4,8, routing (M>8 = cuBLAS).
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HIDDEN, HEAD_DIM, IDX_HEADS, IDX_DIM = 5120, 512, 32, 128
EPS = 1e-6
WARMUP, ITERS = 5, 40
SEED = 4321
MS = (1, 2, 4, 8)


def log(*a):
    print(*a, flush=True)


try:
    from tensorrt_llm._torch.modules.small_n_gemv import small_n_linear
    HAVE_GEMV = True
except ImportError:
    HAVE_GEMV = False

    def small_n_linear(lin, x):
        return lin(x)


def census(path):
    try:
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        cfg = cfg.get("text_config", cfg)
        from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Mode
        layout = CSA2Layout(tuple(cfg["compress_ratios"]), tuple(cfg["kv_source_layer_ids"]),
                            tuple(cfg["index_source_layer_ids"]), cfg["candidate_source_layer_id"],
                            cfg["candidate_topk_blocks"], cfg["candidate_block_size"],
                            cfg["index_topk"], cfg["sliding_window"])
        full, reindex = [], []
        for i in range(cfg["num_hidden_layers"]):
            lay = layout.layer(i)
            if lay.mode == CSA2Mode.FULL:
                full.append((i, lay.compress_ratio))
            elif lay.mode == CSA2Mode.REINDEX:
                reindex.append(i)
        return full, reindex, {k: cfg.get(k) for k in ("head_dim", "index_n_heads",
                                                         "index_head_dim", "hidden_size")}
    except Exception as e:
        log("census fallback:", type(e).__name__, e)
        return [(2, 2), (8, 2), (14, 2), (20, 2)], [24, 28, 32, 36], {}


def nn_linear(k, n, gen):
    lin = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        lin.weight.copy_(torch.randn((n, k), generator=gen, device="cuda") * k ** -0.5)
    return lin


def trt_linear(k, n, gen):
    from tensorrt_llm._torch.modules.linear import Linear
    with torch.device("cuda"):
        lin = Linear(k, n, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(torch.randn((n, k), generator=gen, device="cuda") * k ** -0.5)
    return lin


class Aux:
    def __init__(self, ratio, gen, full=True):
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm
        self.full = full
        self.ratio = ratio
        if full:
            self.comp = trt_linear(HIDDEN, HEAD_DIM * ratio, gen)
            with torch.device("cuda"):
                self.norm = RMSNorm(hidden_size=HEAD_DIM, eps=EPS, dtype=torch.bfloat16)
            self.index_wk = nn_linear(HEAD_DIM, IDX_DIM, gen)
        self.iwp = nn_linear(HIDDEN, IDX_HEADS, gen)
        self.scale = IDX_DIM ** -0.5 * IDX_HEADS ** -0.5

    def __call__(self, h):
        out = []
        if self.full:
            kv = small_n_linear(self.comp, h)
            latent = self.norm(kv[:, :HEAD_DIM].contiguous())
            out.append(small_n_linear(self.index_wk, latent))
        p = small_n_linear(self.iwp, h)
        out.append(torch.mul(p, self.scale, out=torch.empty(p.shape, dtype=torch.float32,
                                                            device=p.device)))
        return out


def make_main(gen):
    import csa2_shared_proj_bench as spb
    probe = spb.Probe(True, gen)
    assert spb.attach_branch(probe), "landing-8 _project_inputs missing"
    return lambda h: spb.branch(probe, h)


def b12x_active(main_fn, h):
    from torch.profiler import ProfilerActivity, profile
    main_fn(h)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        main_fn(h)
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    return names


def time_graph(fn_capture, flush, warm=WARMUP, iters=ITERS):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        fn_capture()
        fn_capture()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            fn_capture()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    ts = []
    for i in range(warm + iters):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        e.synchronize()
        if i >= warm:
            ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
    del graph
    return med, (p90 - p10) / med


def two_stream(main_fn, aux, h):
    aux_stream = torch.cuda.Stream()
    start, done = torch.cuda.Event(), torch.cuda.Event()

    def run():
        start.record()
        with torch.cuda.stream(aux_stream):
            start.wait()
            aux(h)
            done.record()
        main_fn(h)
        done.wait()
    return run


def bench():
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    full, reindex, dims = census(sys.argv[2] if len(sys.argv) > 2 else "")
    log("gemv in tree:", HAVE_GEMV, "FULL (layer, ratio):", full, "REINDEX:", reindex, dims)
    ratios = sorted({r for _, r in full}) or [2]
    main_fn = make_main(gen)
    names = b12x_active(main_fn, torch.randn((2, HIDDEN), device="cuda").to(torch.bfloat16))
    b12x = any("b12x" in n.lower() or "blockscaled" in n.lower() for n in names)
    log("main-stream kernels (M=2):", names)
    auxes = {r: Aux(r, gen, True) for r in ratios}
    aux_re = Aux(2, gen, False)
    batch, extra = {}, {}
    for m in MS:
        h = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
        rec = {}
        for r in ratios:
            rec[f"full_r{r}"] = time_graph(two_stream(main_fn, auxes[r], h), flush)
            rec[f"aux_full_r{r}"] = time_graph(lambda: auxes[r](h), flush)
        rec["reindex"] = time_graph(two_stream(main_fn, aux_re, h), flush)
        rec["aux_reindex"] = time_graph(lambda: aux_re(h), flush)
        rec["main_only"] = time_graph(lambda: main_fn(h), flush)
        # isolated per-shape
        a = auxes[ratios[-1]]
        lat = torch.randn((m, HEAD_DIM), generator=gen, device="cuda").to(torch.bfloat16)
        for name, lin, x in ((f"comp_{a.comp.weight.shape[0]}x5120", a.comp, h),
                             ("iwp_32x5120", a.iwp, h), ("index_wk_128x512", a.index_wk, lat)):
            rec[name] = time_graph(lambda: small_n_linear(lin, x), flush)
        for k, (med, nf) in rec.items():
            log(f"M={m} {k}: {med:.2f} us (noise {nf:.3f})")
        step = sum(rec[f"full_r{r}"][0] for _, r in full) + len(reindex) * rec["reindex"][0]
        nf = max(v[1] for k, v in rec.items() if k.startswith(("full", "reindex")))
        exposed = (sum(rec[f"full_r{r}"][0] for _, r in full) - len(full) * rec["main_only"][0]
                   + len(reindex) * (rec["reindex"][0] - rec["main_only"][0]))
        batch[str(m)] = {"kernel_ms": step / 1000.0, "noise_floor": nf,
                         "aux_exposed_us_per_step": exposed,
                         "per_layer_us": {k: round(v[0], 3) for k, v in rec.items()}}
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}, "gemv": HAVE_GEMV,
                      "b12x_active": b12x, "full": full, "reindex": reindex}))


def sweep():
    from tensorrt_llm._torch.modules import small_n_gemv as sg
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    shapes = ((32, 5120), (512, 5120), (1024, 5120), (128, 512))
    cfgs = [(1, 256, 4), (1, 512, 4), (1, 1024, 4), (1, 256, 8), (2, 256, 4), (2, 512, 4),
            (2, 1024, 8), (4, 256, 4), (4, 512, 8), (8, 256, 8)]
    table = {}
    for n, k in shapes:
        w = (torch.randn((n, k), generator=gen, device="cuda") * k ** -0.5).to(torch.bfloat16)
        for m in MS:
            x = torch.randn((m, k), generator=gen, device="cuda").to(torch.bfloat16)
            row = {"cublas": time_graph(lambda: F.linear(x, w), flush, 3, 20)[0],
                   "default": time_graph(lambda: sg.small_n_gemv(x, w), flush, 3, 20)[0]}
            for c in cfgs:
                if k % c[1]:
                    continue
                row[str(c)] = time_graph(lambda: sg.small_n_gemv(x, w, c), flush, 3, 20)[0]
            best = min((v, kk) for kk, v in row.items() if kk != "cublas")
            log(f"N={n} K={k} M={m}: cublas {row['cublas']:.2f} default {row['default']:.2f} "
                f"best {best[1]} {best[0]:.2f}  all " +
                " ".join(f"{kk}={v:.2f}" for kk, v in row.items()))
            table[f"{n}x{k}_M{m}"] = row
    print(json.dumps({"sweep": table}))


def check():
    from tensorrt_llm._torch.modules import small_n_gemv as sg
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    failures, report = [], {}
    shapes = ((32, 5120), (512, 5120), (1024, 5120), (128, 512))
    bf16_ulp = 2 ** -8
    for n, k in shapes:
        w = (torch.randn((n, k), generator=gen, device="cuda") * k ** -0.5).to(torch.bfloat16)
        for m in (1, 2, 3, 4, 5, 8, 9, 16, 2048):
            x = torch.randn((m, k), generator=gen, device="cuda").to(torch.bfloat16)
            ref = x.double() @ w.double().t()
            cub = F.linear(x, w)
            routed = sg.maybe_small_n_gemv(x, w)
            key = f"{n}x{k}_M{m}"
            if m > sg.MAX_M or n > sg.MAX_N:
                ok = routed is None and torch.equal(sg.small_n_linear(
                    type("L", (), {"weight": w, "bias": None, "__call__": lambda s, t: F.linear(t, w)})(), x), cub)
                report[key] = {"routed": routed is not None, "cublas_bitwise": ok}
                if not ok:
                    failures.append(key + " not cuBLAS")
                continue
            if routed is None:
                failures.append(key + " not routed")
                continue
            outs = [sg.maybe_small_n_gemv(x, w) for _ in range(5)]
            det = all(torch.equal(o, routed) for o in outs)
            scale = ref.abs().max().item()
            e_g = (routed.double() - ref).abs().max().item()
            e_c = (cub.double() - ref).abs().max().item()
            rel_g = e_g / scale
            tol_ok = torch.allclose(routed.double(), ref, rtol=1.6e-2, atol=1.6e-2)
            cmp_ok = e_g <= 1.25 * e_c + bf16_ulp * scale
            vs_cub = (routed.double() - cub.double()).abs().max().item()
            report[key] = {"max_abs_vs_fp64": e_g, "cublas_max_abs_vs_fp64": e_c,
                           "max_rel_vs_fp64": rel_g, "max_abs_vs_cublas": vs_cub,
                           "bitwise_eq_cublas": bool(torch.equal(routed, cub)),
                           "deterministic": det, "finite": bool(torch.isfinite(routed).all()),
                           "shape_ok": routed.shape == cub.shape and routed.stride() == cub.stride()}
            log(key, report[key])
            if not (det and tol_ok and cmp_ok and report[key]["finite"] and report[key]["shape_ok"]):
                failures.append(key)
    # Module hook: small_n_linear on an nn.Linear equals the GEMV at decode M.
    lin = nn_linear(HIDDEN, IDX_HEADS, gen)
    for m in MS:
        x = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
        if not torch.equal(sg.small_n_linear(lin, x), sg.small_n_gemv(x, lin.weight)):
            failures.append(f"hook M={m}")
    report["hook"] = "checked M=1,2,4,8"
    # CUDA-graph capture/replay, single stream and fork/join, repeated with fresh inputs.
    gq = {}
    for two in (False, True):
        aux = Aux(2, gen, True)
        main_fn = make_main(gen) if two else (lambda h: None)
        for m in MS:
            static = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
            aux_stream = torch.cuda.Stream()
            start, done = torch.cuda.Event(), torch.cuda.Event()
            holder = {}

            def run():
                if two:
                    start.record()
                    with torch.cuda.stream(aux_stream):
                        start.wait()
                        holder["o"] = aux(static)
                        done.record()
                    main_fn(static)
                    done.wait()
                else:
                    holder["o"] = aux(static)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                run()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=s):
                    run()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            outs = holder["o"]
            errs = []
            for i in range(WARMUP + ITERS):
                g = torch.Generator(device="cuda").manual_seed(SEED + 31 * i + m)
                x = torch.randn((m, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
                static.copy_(x)  # reset point: fresh input each replay
                graph.replay()
                torch.cuda.synchronize()
                eager = aux(x)
                bit = all(torch.equal(a, b) for a, b in zip(outs, eager))
                p_ref = (x.double() @ aux.iwp.weight.double().t()) * aux.scale
                errs.append((outs[-1].double() - p_ref).abs().max().item())
                if not bit:
                    failures.append(f"graph two={two} M={m} iter {i}")
            gq[f"{'two_stream' if two else 'single'}_M{m}"] = {
                "iters": WARMUP + ITERS, "max_abs_err_vs_fp64": max(errs)}
            del graph
    report["graph"] = gq
    log("graph", gq)
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "Triton small-N GEMV vs FP64 reference and cuBLAS F.linear; "
                                  "graph replays vs eager; M>8 stays cuBLAS bitwise",
                      "report": report}, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    torch.manual_seed(SEED)
    {"bench": bench, "sweep": sweep, "check": check}[mode]()
